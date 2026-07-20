from __future__ import annotations

from pathlib import Path

import torch

from ..ctc_text import CharacterTokenizer
from ..models import CnnCtcReader
from ..reader_config import ReaderExperimentConfig
from ..schemas import CharacterVocabularySpec


class ReaderCheckpointError(ValueError):
    pass


def reader_checkpoint_metadata(
    config: ReaderExperimentConfig,
    *,
    experiment_config_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "experiment_config_sha256": experiment_config_sha256,
        "dataset_sha256": config.data.dataset_sha256,
        "split_sha256": config.data.split_sha256,
        "model_config": config.model.model_dump(mode="json"),
    }


def load_frozen_reader_checkpoint(
    path: str | Path,
    *,
    model: CnnCtcReader,
    config: ReaderExperimentConfig,
    tokenizer: CharacterTokenizer,
    experiment_config_sha256: str,
    device: torch.device | str,
) -> dict[str, object]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Reader checkpoint does not exist: {checkpoint_path}"
        )
    payload = torch.load(
        checkpoint_path,
        map_location=torch.device(device),
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise ReaderCheckpointError("Reader checkpoint payload must be a mapping.")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ReaderCheckpointError("Reader checkpoint metadata is missing.")
    expected_metadata = reader_checkpoint_metadata(
        config,
        experiment_config_sha256=experiment_config_sha256,
    )
    if metadata != expected_metadata:
        raise ReaderCheckpointError(
            "Reader checkpoint metadata does not match the frozen experiment."
        )

    vocabulary_payload = payload.get("vocabulary")
    try:
        checkpoint_vocabulary = CharacterVocabularySpec.model_validate(
            vocabulary_payload
        )
    except ValueError as exc:
        raise ReaderCheckpointError(
            "Reader checkpoint vocabulary is invalid."
        ) from exc
    if checkpoint_vocabulary != tokenizer.spec:
        raise ReaderCheckpointError(
            "Reader checkpoint vocabulary does not match train-only vocabulary."
        )
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ReaderCheckpointError("Reader checkpoint model state is missing.")
    model.load_state_dict(state_dict, strict=True)
    return payload
