import torch
import pytest

from multimodal_lab.ctc_text import (
    CharacterTokenizer,
    CtcTextError,
    ctc_loss,
    greedy_ctc_decode,
    pack_ctc_targets,
    required_ctc_timesteps,
)
from multimodal_lab.schemas import CharacterVocabularySpec


def test_character_tokenizer_round_trips_and_freezes_hash(tmp_path):
    tokenizer = CharacterTokenizer.fit(["cat", "book"])
    output_path = tmp_path / "vocabulary.json"

    tokenizer.save(output_path)
    loaded = CharacterTokenizer.load(output_path)

    assert loaded.decode(loaded.encode("book")) == "book"
    assert loaded.spec.sha256 == tokenizer.spec.sha256
    assert loaded.blank_id == 0
    assert loaded.unknown_id == 1


def test_character_tokenizer_detects_tampered_vocabulary():
    with pytest.raises(CtcTextError, match="hash mismatch"):
        CharacterTokenizer(
            CharacterVocabularySpec(
                characters=["a", "b"],
                sha256="0" * 64,
            )
        )


def test_greedy_decoder_collapses_repeats_before_removing_blank():
    tokenizer = CharacterTokenizer.fit(["cat"])
    c, a, t = tokenizer.encode("cat")
    path = [c, c, tokenizer.blank_id, a, tokenizer.blank_id, t, t]
    logits = torch.full((1, len(path), tokenizer.vocabulary_size), -10.0)
    for time_index, token_id in enumerate(path):
        logits[0, time_index, token_id] = 10.0

    decoded = greedy_ctc_decode(
        logits,
        torch.tensor([len(path)]),
        tokenizer,
    )

    assert decoded == ["cat"]


def test_adjacent_repeated_characters_need_an_extra_ctc_timestep():
    tokenizer = CharacterTokenizer.fit(["book"])
    encoded = tokenizer.encode("book")

    assert len(encoded) == 4
    assert required_ctc_timesteps(encoded) == 5


def test_target_packing_and_ctc_loss_follow_pytorch_shapes():
    tokenizer = CharacterTokenizer.fit(["cat", "book"])
    targets = pack_ctc_targets(["cat", "book"], tokenizer)
    logits = torch.randn(
        2,
        12,
        tokenizer.vocabulary_size,
        requires_grad=True,
    )

    loss = ctc_loss(
        logits,
        torch.tensor([12, 12]),
        targets,
        blank_id=tokenizer.blank_id,
    )
    loss.backward()

    assert targets.lengths.tolist() == [3, 4]
    assert targets.values.ndim == 1
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_ctc_loss_rejects_time_axes_too_short_for_repeated_labels():
    tokenizer = CharacterTokenizer.fit(["book"])
    targets = pack_ctc_targets(["book"], tokenizer)
    logits = torch.randn(1, 4, tokenizer.vocabulary_size)

    with pytest.raises(CtcTextError, match="at least 5 CTC steps"):
        ctc_loss(logits, torch.tensor([4]), targets)
