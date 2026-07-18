from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .annotation_io import load_jsonl, write_jsonl
from .line_crop_dataset import (
    build_exact_match_line_crops,
    line_cropper_version,
)
from .page_reading import sha256_file
from .schemas import (
    LineCropDatasetManifest,
    LineLabelSource,
    PageContent,
    StablePageReference,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Crop OCR text polygons whose labels exactly match manual page gold."
        )
    )
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--page-contents", required=True, type=Path)
    parser.add_argument("--image-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--padding-pixels", type=int, default=2)
    parser.add_argument("--expected-references-sha256")
    parser.add_argument("--expected-page-contents-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    references_sha256 = _verify_hash(
        args.references,
        args.expected_references_sha256,
        label="references",
    )
    page_contents_sha256 = _verify_hash(
        args.page_contents,
        args.expected_page_contents_sha256,
        label="page contents",
    )
    references = load_jsonl(args.references, StablePageReference)
    contents = load_jsonl(args.page_contents, PageContent)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = build_exact_match_line_crops(
        references,
        contents,
        image_root=args.image_root,
        output_dir=args.output_dir / "crops",
        padding_pixels=args.padding_pixels,
    )
    samples_path = args.output_dir / "line_crop_samples.jsonl"
    write_jsonl(samples_path, samples)
    dataset_sha256 = sha256_file(samples_path)
    manifest = LineCropDatasetManifest(
        dataset_id=dataset_sha256,
        samples_path=str(samples_path),
        sample_count=len(samples),
        references_path=str(args.references),
        references_sha256=references_sha256,
        page_contents_path=str(args.page_contents),
        page_contents_sha256=page_contents_sha256,
        cropper_version=line_cropper_version(args.padding_pixels),
        label_source=LineLabelSource.manual_exact_match,
    )
    rendered = manifest.model_dump_json(indent=2)
    (args.output_dir / "line_crop_manifest.json").write_text(
        rendered + "\n",
        encoding="utf-8",
    )
    print(rendered)
    return 0


def _verify_hash(path: Path, expected: str | None, *, label: str) -> str:
    actual = sha256_file(path)
    normalized_expected = (expected or "").strip().lower()
    if normalized_expected and normalized_expected != actual:
        raise ValueError(
            f"Frozen {label} hash mismatch: expected "
            f"{normalized_expected}, got {actual}."
        )
    return actual


if __name__ == "__main__":
    raise SystemExit(main())
