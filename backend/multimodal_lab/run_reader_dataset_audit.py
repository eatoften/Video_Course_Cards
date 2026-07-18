from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .annotation_io import load_jsonl
from .experiment_protocol import ExperimentPhase, audit_reader_dataset
from .page_reading import sha256_file
from .schemas import LectureSplitManifest, LineCropSample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a frozen line-reader dataset before model training."
    )
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--expected-split-sha256")
    parser.add_argument(
        "--phase",
        choices=[phase.value for phase in ExperimentPhase],
        default=ExperimentPhase.formal.value,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_sha256 = sha256_file(args.samples)
    split_sha256 = sha256_file(args.split)
    _require_expected_hash(
        "dataset",
        actual=dataset_sha256,
        expected=args.expected_dataset_sha256,
    )
    _require_expected_hash(
        "split",
        actual=split_sha256,
        expected=args.expected_split_sha256,
    )

    samples = load_jsonl(args.samples, LineCropSample)
    split = LectureSplitManifest.model_validate_json(
        args.split.read_text(encoding="utf-8")
    )
    report = audit_reader_dataset(
        samples,
        split,
        dataset_sha256=dataset_sha256,
        phase=ExperimentPhase(args.phase),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    print(report.model_dump_json(indent=2))
    return 0 if report.passed else 2


def _require_expected_hash(
    label: str,
    *,
    actual: str,
    expected: str | None,
) -> None:
    normalized = (expected or "").strip().lower()
    if normalized and normalized != actual:
        raise ValueError(
            f"Frozen {label} hash mismatch: expected {normalized}, got {actual}."
        )


if __name__ == "__main__":
    raise SystemExit(main())
