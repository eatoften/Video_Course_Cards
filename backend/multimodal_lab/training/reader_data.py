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
class ReaderDataBundle:
    contract: VerifiedReaderDataContract
    tokenizer: CharacterTokenizer
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader
    sample_counts: dict[DatasetSplit, int]


def build_reader_data_bundle(
    config: ReaderExperimentConfig,
    *,
    project_root: str | Path,
) -> ReaderDataBundle:
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
    datasets = {
        split: LineCropDataset(
            split_samples,
            image_root=contract.image_root,
            transform=transform,
        )
        for split, split_samples in partitions.items()
    }
    collate = partial(collate_line_crops, width_multiple=4)
    generator = torch.Generator().manual_seed(config.seed)

    def make_loader(split: DatasetSplit, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            datasets[split],
            batch_size=config.data.batch_size,
            shuffle=shuffle,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            drop_last=False,
            collate_fn=collate,
            generator=generator if shuffle else None,
        )

    return ReaderDataBundle(
        contract=contract,
        tokenizer=tokenizer,
        train_loader=make_loader(DatasetSplit.train, shuffle=True),
        validation_loader=make_loader(DatasetSplit.validation, shuffle=False),
        test_loader=make_loader(DatasetSplit.test, shuffle=False),
        sample_counts={
            split: len(split_samples)
            for split, split_samples in partitions.items()
        },
    )
