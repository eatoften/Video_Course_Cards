from pathlib import Path

import pytest
import torch

from multimodal_lab.ctc_text import CharacterTokenizer
from multimodal_lab.models import CnnCtcReader
from multimodal_lab.reader_config import (
    CnnCtcEncoderConfig,
    ReaderDataConfig,
    ReaderExperimentConfig,
    ReaderOptimizationConfig,
)
from multimodal_lab.training.reader_checkpoint import (
    ReaderCheckpointError,
    load_frozen_reader_checkpoint,
    reader_checkpoint_metadata,
)


def make_experiment() -> ReaderExperimentConfig:
    return ReaderExperimentConfig(
        experiment_id="checkpoint-test",
        seed=1,
        model=CnnCtcEncoderConfig(
            channels=[8, 16],
            temporal_downsample=2,
            output_features=16,
            dropout=0,
        ),
        data=ReaderDataConfig(
            dataset_path="dataset.jsonl",
            dataset_sha256="a" * 64,
            split_path="split.json",
            split_sha256="b" * 64,
            audit_path="audit.json",
            audit_sha256="c" * 64,
            expected_vocabulary_sha256="d" * 64,
            image_root=".",
            target_height=8,
            max_image_width=32,
            batch_size=2,
            num_workers=0,
        ),
        optimization=ReaderOptimizationConfig(
            epochs=1,
            learning_rate=0.001,
            weight_decay=0,
            gradient_clip_norm=1,
            early_stopping_patience=1,
            mixed_precision=False,
        ),
    )


def save_checkpoint(
    path: Path,
    *,
    model: CnnCtcReader,
    tokenizer: CharacterTokenizer,
    config: ReaderExperimentConfig,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocabulary": tokenizer.spec.model_dump(mode="json"),
            "metadata": reader_checkpoint_metadata(
                config,
                experiment_config_sha256="e" * 64,
            ),
        },
        path,
    )


def test_frozen_checkpoint_round_trip(tmp_path: Path):
    config = make_experiment()
    tokenizer = CharacterTokenizer.fit(["ab"])
    model = CnnCtcReader(config.model, tokenizer.vocabulary_size)
    checkpoint = tmp_path / "reader.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        tokenizer=tokenizer,
        config=config,
    )
    restored = CnnCtcReader(config.model, tokenizer.vocabulary_size)

    load_frozen_reader_checkpoint(
        checkpoint,
        model=restored,
        config=config,
        tokenizer=tokenizer,
        experiment_config_sha256="e" * 64,
        device="cpu",
    )

    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)


def test_frozen_checkpoint_rejects_another_config_hash(tmp_path: Path):
    config = make_experiment()
    tokenizer = CharacterTokenizer.fit(["ab"])
    model = CnnCtcReader(config.model, tokenizer.vocabulary_size)
    checkpoint = tmp_path / "reader.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        tokenizer=tokenizer,
        config=config,
    )

    with pytest.raises(ReaderCheckpointError, match="metadata"):
        load_frozen_reader_checkpoint(
            checkpoint,
            model=model,
            config=config,
            tokenizer=tokenizer,
            experiment_config_sha256="f" * 64,
            device="cpu",
        )
