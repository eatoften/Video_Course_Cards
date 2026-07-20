from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .annotation_io import load_jsonl
from .page_reading import sha256_file
from .reader_coverage_audit import audit_reader_content_coverage
from .schemas import LectureSplitManifest, LineCropSample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit reader labels by content and length slices."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--short-text-maximum", type=int, default=4)
    parser.add_argument("--long-text-minimum", type=int, default=48)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    samples = load_jsonl(args.dataset, LineCropSample)
    split = LectureSplitManifest.model_validate_json(
        args.split.read_text(encoding="utf-8")
    )
    report = audit_reader_content_coverage(
        samples,
        split,
        dataset_sha256=sha256_file(args.dataset),
        split_sha256=sha256_file(args.split),
        short_text_maximum=args.short_text_maximum,
        long_text_minimum=args.long_text_minimum,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    print(report.model_dump_json(indent=2))
    print(f"coverage_audit_sha256={sha256_file(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
