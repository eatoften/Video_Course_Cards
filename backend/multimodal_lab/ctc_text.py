from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .schemas import CharacterVocabularySpec


class CtcTextError(ValueError):
    pass


@dataclass(frozen=True)
class CtcTargetBatch:
    values: Tensor
    lengths: Tensor
    sequences: tuple[tuple[int, ...], ...]


def normalize_line_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split())


class CharacterTokenizer:
    def __init__(self, spec: CharacterVocabularySpec) -> None:
        expected_hash = vocabulary_sha256(spec.characters)
        if spec.sha256 != expected_hash:
            raise CtcTextError(
                "Vocabulary hash mismatch: "
                f"expected {expected_hash}, got {spec.sha256}."
            )
        self.spec = spec
        self._character_to_id = {
            character: index + 2
            for index, character in enumerate(spec.characters)
        }
        self._id_to_character = {
            index: character
            for character, index in self._character_to_id.items()
        }

    @classmethod
    def fit(cls, texts: Sequence[str]) -> CharacterTokenizer:
        normalized_texts = [normalize_line_text(text) for text in texts]
        if not normalized_texts or any(not text for text in normalized_texts):
            raise CtcTextError("Vocabulary training text cannot be empty.")
        characters = sorted(set("".join(normalized_texts)))
        return cls(
            CharacterVocabularySpec(
                characters=characters,
                sha256=vocabulary_sha256(characters),
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> CharacterTokenizer:
        spec = CharacterVocabularySpec.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
        return cls(spec)

    @property
    def blank_id(self) -> int:
        return self.spec.blank_id

    @property
    def unknown_id(self) -> int:
        return self.spec.unknown_id

    @property
    def vocabulary_size(self) -> int:
        return len(self.spec.characters) + 2

    def encode(
        self,
        text: str,
        *,
        allow_unknown: bool = True,
    ) -> list[int]:
        normalized = normalize_line_text(text)
        if not normalized:
            raise CtcTextError("Cannot encode an empty line.")

        encoded: list[int] = []
        for character in normalized:
            token_id = self._character_to_id.get(character)
            if token_id is None:
                if not allow_unknown:
                    raise CtcTextError(
                        f"Character {character!r} is outside the frozen vocabulary."
                    )
                token_id = self.unknown_id
            encoded.append(token_id)
        return encoded

    def decode(self, token_ids: Sequence[int]) -> str:
        pieces: list[str] = []
        for token_id in token_ids:
            if token_id == self.blank_id:
                raise CtcTextError(
                    "Raw tokenizer decoding cannot contain the CTC blank token."
                )
            if token_id == self.unknown_id:
                pieces.append(self.spec.unknown_token)
                continue
            character = self._id_to_character.get(token_id)
            if character is None:
                raise CtcTextError(f"Unknown token ID: {token_id}.")
            pieces.append(character)
        return "".join(pieces)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            self.spec.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )


def vocabulary_sha256(characters: Sequence[str]) -> str:
    payload = json.dumps(
        {
            "blank_id": 0,
            "unknown_id": 1,
            "normalization": "NFKC-whitespace-v1",
            "case_sensitive": True,
            "characters": list(characters),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pack_ctc_targets(
    texts: Sequence[str],
    tokenizer: CharacterTokenizer,
    *,
    device: torch.device | str | None = None,
    allow_unknown: bool = False,
) -> CtcTargetBatch:
    if not texts:
        raise CtcTextError("A CTC target batch cannot be empty.")
    sequences = tuple(
        tuple(tokenizer.encode(text, allow_unknown=allow_unknown))
        for text in texts
    )
    values = torch.tensor(
        [token_id for sequence in sequences for token_id in sequence],
        dtype=torch.long,
        device=device,
    )
    lengths = torch.tensor(
        [len(sequence) for sequence in sequences],
        dtype=torch.long,
    )
    return CtcTargetBatch(
        values=values,
        lengths=lengths,
        sequences=sequences,
    )


def required_ctc_timesteps(token_ids: Sequence[int]) -> int:
    if not token_ids:
        raise CtcTextError("A CTC target sequence cannot be empty.")
    adjacent_repeats = sum(
        current == previous
        for previous, current in zip(token_ids, token_ids[1:])
    )
    return len(token_ids) + adjacent_repeats


def ctc_loss(
    logits: Tensor,
    input_lengths: Tensor,
    targets: CtcTargetBatch,
    *,
    blank_id: int = 0,
) -> Tensor:
    if logits.ndim != 3:
        raise CtcTextError("CTC logits must have shape [batch, time, vocab].")
    batch_size, max_time, vocabulary_size = logits.shape
    if vocabulary_size <= blank_id:
        raise CtcTextError("blank_id is outside the model vocabulary.")
    if input_lengths.shape != (batch_size,):
        raise CtcTextError("input_lengths must contain one value per sample.")
    if targets.lengths.shape != (batch_size,):
        raise CtcTextError("target lengths must contain one value per sample.")

    cpu_input_lengths = input_lengths.detach().to(dtype=torch.long, device="cpu")
    if torch.any(cpu_input_lengths <= 0) or torch.any(
        cpu_input_lengths > max_time
    ):
        raise CtcTextError("CTC input lengths must be within the logit time axis.")

    for sample_index, sequence in enumerate(targets.sequences):
        required = required_ctc_timesteps(sequence)
        available = int(cpu_input_lengths[sample_index].item())
        if available < required:
            raise CtcTextError(
                f"Sample {sample_index} needs at least {required} CTC steps "
                f"but only {available} are available."
            )

    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
    return F.ctc_loss(
        log_probs,
        targets.values.to(logits.device),
        cpu_input_lengths,
        targets.lengths.to(dtype=torch.long, device="cpu"),
        blank=blank_id,
        reduction="mean",
        zero_infinity=False,
    )


def collapse_ctc_path(
    path: Sequence[int],
    *,
    blank_id: int = 0,
) -> list[int]:
    collapsed: list[int] = []
    previous: int | None = None
    for token_id in path:
        if token_id != previous and token_id != blank_id:
            collapsed.append(token_id)
        previous = token_id
    return collapsed


def greedy_ctc_decode(
    logits: Tensor,
    input_lengths: Tensor,
    tokenizer: CharacterTokenizer,
) -> list[str]:
    token_sequences = greedy_ctc_decode_token_ids(
        logits,
        input_lengths,
        blank_id=tokenizer.blank_id,
        vocabulary_size=tokenizer.vocabulary_size,
    )
    return [tokenizer.decode(token_ids) for token_ids in token_sequences]


def greedy_ctc_decode_token_ids(
    logits: Tensor,
    input_lengths: Tensor,
    *,
    blank_id: int,
    vocabulary_size: int,
) -> list[list[int]]:
    if logits.ndim != 3:
        raise CtcTextError("CTC logits must have shape [batch, time, vocab].")
    batch_size, max_time, output_vocabulary_size = logits.shape
    if output_vocabulary_size != vocabulary_size:
        raise CtcTextError(
            "Model vocabulary size does not match the frozen tokenizer."
        )
    if input_lengths.shape != (batch_size,):
        raise CtcTextError("input_lengths must contain one value per sample.")

    best_paths = logits.argmax(dim=-1).detach().to(device="cpu")
    decoded: list[list[int]] = []
    for sample_index, length_value in enumerate(input_lengths.tolist()):
        length = int(length_value)
        if length <= 0 or length > max_time:
            raise CtcTextError("A decode length is outside the logit time axis.")
        token_ids = collapse_ctc_path(
            best_paths[sample_index, :length].tolist(),
            blank_id=blank_id,
        )
        decoded.append(token_ids)
    return decoded
