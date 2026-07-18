from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .annotation_io import load_jsonl
from .line_crop_dataset import (
    make_lecture_level_split,
    partition_by_lecture_split,
)
from .page_reading import sha256_file
from .schemas import DatasetSplit, LineCropSample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a deterministic leakage-free lecture split."
    )
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_sha256 = sha256_file(args.samples)
    expected_hash = (args.expected_dataset_sha256 or "").strip().lower()
    if expected_hash and expected_hash != dataset_sha256:
        raise ValueError(
            "Frozen line-crop dataset hash mismatch: "
            f"expected {expected_hash}, got {dataset_sha256}."
        )
    samples = load_jsonl(args.samples, LineCropSample)
    split = make_lecture_level_split(
        samples,
        dataset_sha256=dataset_sha256,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    partitions = partition_by_lecture_split(samples, split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        split.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    print(split.model_dump_json(indent=2))
    print(
        "sample_counts "
        + " ".join(
            f"{dataset_split.value}={len(partitions[dataset_split])}"
            for dataset_split in (
                DatasetSplit.train,
                DatasetSplit.validation,
                DatasetSplit.test,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
