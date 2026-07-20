from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..annotation_io import load_jsonl
from ..ctc_text import CharacterTokenizer
from ..line_crop_dataset import (
    LineCropBatch,
    LineCropDataset,
    LineImageTransform,
    collate_line_crops,
    partition_by_lecture_split,
)
from ..reader_config import (
    ReaderExperimentConfig,
    VerifiedReaderDataContract,
    verify_reader_data_contract,
)
from ..schemas import DatasetSplit, LineCropSample


@dataclass(frozen=True)
class ReaderTrainingDataBundle:
    contract: VerifiedReaderDataContract
    tokenizer: CharacterTokenizer
    train_loader: DataLoader
    validation_loader: DataLoader
    sample_counts: dict[DatasetSplit, int]


@dataclass(frozen=True)
class ReaderTestDataBundle:
    contract: VerifiedReaderDataContract
    tokenizer: CharacterTokenizer
    test_loader: DataLoader
    sample_count: int


def build_reader_training_data_bundle(
    config: ReaderExperimentConfig,
    *,
    project_root: str | Path,
) -> ReaderTrainingDataBundle:
    contract, tokenizer, partitions, transform = _load_verified_data(
        config,
        project_root=project_root,
    )
    generator = torch.Generator().manual_seed(config.seed)
    return ReaderTrainingDataBundle(
        contract=contract,
        tokenizer=tokenizer,
        train_loader=_make_loader(
            partitions[DatasetSplit.train],
            contract=contract,
            transform=transform,
            config=config,
            shuffle=True,
            generator=generator,
        ),
        validation_loader=_make_loader(
            partitions[DatasetSplit.validation],
            contract=contract,
            transform=transform,
            config=config,
            shuffle=False,
        ),
        sample_counts={
            DatasetSplit.train: len(partitions[DatasetSplit.train]),
            DatasetSplit.validation: len(partitions[DatasetSplit.validation]),
        },
    )


def build_reader_test_data_bundle(
    config: ReaderExperimentConfig,
    *,
    project_root: str | Path,
) -> ReaderTestDataBundle:
    contract, tokenizer, partitions, transform = _load_verified_data(
        config,
        project_root=project_root,
    )
    test_samples = partitions[DatasetSplit.test]
    return ReaderTestDataBundle(
        contract=contract,
        tokenizer=tokenizer,
        test_loader=_make_loader(
            test_samples,
            contract=contract,
            transform=transform,
            config=config,
            shuffle=False,
        ),
        sample_count=len(test_samples),
    )


def _load_verified_data(
    config: ReaderExperimentConfig,
    *,
    project_root: str | Path,
):
    contract = verify_reader_data_contract(config, project_root=project_root)
    samples = load_jsonl(contract.dataset_path, LineCropSample)
    partitions = partition_by_lecture_split(samples, contract.split)
    tokenizer = CharacterTokenizer.fit(
        [sample.normalized_text for sample in partitions[DatasetSplit.train]]
    )
    if tokenizer.spec.sha256 != config.data.expected_vocabulary_sha256:
        raise ValueError("Training-only tokenizer does not match the frozen hash.")
    transform = LineImageTransform(
        target_height=config.data.target_height,
        max_width=config.data.max_image_width,
        verify_hashes=True,
    )
    return contract, tokenizer, partitions, transform


def _make_loader(
    samples: list[LineCropSample],
    *,
    contract: VerifiedReaderDataContract,
    transform: LineImageTransform,
    config: ReaderExperimentConfig,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    dataset = LineCropDataset(
        samples,
        image_root=contract.image_root,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and torch.cuda.is_available(),
        drop_last=False,
        collate_fn=partial(
            collate_line_crops,
            width_multiple=config.model.temporal_downsample,
        ),
        generator=generator if shuffle else None,
    )
