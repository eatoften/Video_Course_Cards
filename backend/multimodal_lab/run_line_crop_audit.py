from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .annotation_io import load_jsonl
from .line_crop_audit import render_line_crop_audit_sheets
from .page_reading import sha256_file
from .schemas import LineCropSample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render hash-verified line crops for visual label auditing."
    )
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--image-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-samples-sha256")
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--rows", type=int, default=15)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_sha256 = sha256_file(args.samples)
    expected = (args.expected_samples_sha256 or "").strip().lower()
    if expected and expected != dataset_sha256:
        raise ValueError(
            f"Frozen sample hash mismatch: expected {expected}, "
            f"got {dataset_sha256}."
        )
    report = render_line_crop_audit_sheets(
        load_jsonl(args.samples, LineCropSample),
        image_root=args.image_root,
        output_dir=args.output_dir,
        dataset_sha256=dataset_sha256,
        columns=args.columns,
        rows=args.rows,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
