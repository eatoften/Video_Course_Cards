from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from .annotation_io import load_jsonl, write_jsonl
from .formal_reader_dataset import (
    build_source_aligned_line_crops,
    build_synthetic_line_crops,
    combine_line_crop_components,
    make_line_review_template,
    source_aligned_cropper_version,
)
from .experiment_protocol import audit_reader_dataset
from .line_crop_dataset import make_lecture_level_split
from .page_reading import sha256_file
from .schemas import (
    LineCropReviewRecord,
    LineCropSample,
    PageContent,
    StablePageReference,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build leakage-aware formal line-reader datasets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    template = subparsers.add_parser(
        "review-template",
        help="Enumerate every OCR detector polygon for explicit review.",
    )
    template.add_argument("--page-contents", required=True, type=Path)
    template.add_argument("--output", required=True, type=Path)
    template.add_argument("--expected-page-contents-sha256")

    reviewed = subparsers.add_parser(
        "reviewed",
        help="Build real video crops from a completed source-alignment review.",
    )
    reviewed.add_argument("--references", required=True, type=Path)
    reviewed.add_argument("--page-contents", required=True, type=Path)
    reviewed.add_argument("--reviews", required=True, type=Path)
    reviewed.add_argument("--image-root", type=Path, default=Path.cwd())
    reviewed.add_argument("--output-dir", required=True, type=Path)
    reviewed.add_argument("--padding-pixels", type=int, default=2)
    reviewed.add_argument("--expected-references-sha256")
    reviewed.add_argument("--expected-page-contents-sha256")
    reviewed.add_argument("--expected-reviews-sha256")

    synthetic = subparsers.add_parser(
        "synthetic",
        help="Render deterministic train-only line images from gold text.",
    )
    synthetic.add_argument("--references", required=True, type=Path)
    synthetic.add_argument("--image-root", type=Path, default=Path.cwd())
    synthetic.add_argument("--output-dir", required=True, type=Path)
    synthetic.add_argument("--font", required=True, type=Path)
    synthetic.add_argument("--seed", type=int, default=17)
    synthetic.add_argument("--variants-per-line", type=int, default=4)
    synthetic.add_argument("--min-characters", type=int, default=2)
    synthetic.add_argument("--max-characters", type=int, default=48)
    synthetic.add_argument("--expected-references-sha256")
    synthetic.add_argument("--expected-font-sha256")

    combine = subparsers.add_parser(
        "combine",
        help="Combine frozen components without rewriting their samples.",
    )
    combine.add_argument("--component", action="append", required=True, type=Path)
    combine.add_argument("--output", required=True, type=Path)
    combine.add_argument("--manifest", required=True, type=Path)

    audit = subparsers.add_parser(
        "audit",
        help="Freeze a lecture-level split and audit it before model training.",
    )
    audit.add_argument("--dataset", required=True, type=Path)
    audit.add_argument("--split-output", required=True, type=Path)
    audit.add_argument("--audit-output", required=True, type=Path)
    audit.add_argument("--seed", type=int, default=1)
    audit.add_argument("--validation-fraction", type=float, default=0.2)
    audit.add_argument("--test-fraction", type=float, default=0.2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "review-template":
        return _review_template(args)
    if args.command == "reviewed":
        return _reviewed(args)
    if args.command == "synthetic":
        return _synthetic(args)
    if args.command == "combine":
        return _combine(args)
    if args.command == "audit":
        return _audit(args)
    raise AssertionError(f"Unhandled command: {args.command}")


def _review_template(args: argparse.Namespace) -> int:
    contents_hash = _verify_hash(
        args.page_contents,
        args.expected_page_contents_sha256,
        label="page contents",
    )
    contents = load_jsonl(args.page_contents, PageContent)
    records = make_line_review_template(contents)
    write_jsonl(args.output, records)
    _print_json(
        {
            "command": "review-template",
            "page_contents_sha256": contents_hash,
            "candidate_count": len(records),
            "output": str(args.output),
            "output_sha256": sha256_file(args.output),
        }
    )
    return 0


def _reviewed(args: argparse.Namespace) -> int:
    source_hashes = {
        "references": _verify_hash(
            args.references,
            args.expected_references_sha256,
            label="references",
        ),
        "page_contents": _verify_hash(
            args.page_contents,
            args.expected_page_contents_sha256,
            label="page contents",
        ),
        "reviews": _verify_hash(
            args.reviews,
            args.expected_reviews_sha256,
            label="reviews",
        ),
    }
    references = load_jsonl(args.references, StablePageReference)
    contents = load_jsonl(args.page_contents, PageContent)
    reviews = load_jsonl(args.reviews, LineCropReviewRecord)
    samples = build_source_aligned_line_crops(
        references,
        contents,
        reviews,
        image_root=args.image_root,
        output_dir=args.output_dir / "crops",
        padding_pixels=args.padding_pixels,
    )
    samples_path = args.output_dir / "line_crop_samples.jsonl"
    write_jsonl(samples_path, samples)
    manifest = _component_manifest(
        mode="source_aligned",
        samples_path=samples_path,
        samples=samples,
        source_hashes=source_hashes,
        parameters={
            "cropper_version": source_aligned_cropper_version(
                args.padding_pixels
            ),
            "padding_pixels": args.padding_pixels,
        },
    )
    _write_json(args.output_dir / "component_manifest.json", manifest)
    _print_json(manifest)
    return 0


def _synthetic(args: argparse.Namespace) -> int:
    source_hashes = {
        "references": _verify_hash(
            args.references,
            args.expected_references_sha256,
            label="references",
        ),
        "font": _verify_hash(
            args.font,
            args.expected_font_sha256,
            label="font",
        ),
    }
    references = load_jsonl(args.references, StablePageReference)
    samples = build_synthetic_line_crops(
        references,
        image_root=args.image_root,
        output_dir=args.output_dir / "crops",
        font_path=args.font,
        seed=args.seed,
        variants_per_line=args.variants_per_line,
        min_characters=args.min_characters,
        max_characters=args.max_characters,
    )
    samples_path = args.output_dir / "line_crop_samples.jsonl"
    write_jsonl(samples_path, samples)
    manifest = _component_manifest(
        mode="synthetic_render",
        samples_path=samples_path,
        samples=samples,
        source_hashes=source_hashes,
        parameters={
            "seed": args.seed,
            "variants_per_line": args.variants_per_line,
            "min_characters": args.min_characters,
            "max_characters": args.max_characters,
        },
    )
    _write_json(args.output_dir / "component_manifest.json", manifest)
    _print_json(manifest)
    return 0


def _combine(args: argparse.Namespace) -> int:
    components = [
        load_jsonl(component_path, LineCropSample)
        for component_path in args.component
    ]
    component_hashes = {
        str(path): sha256_file(path)
        for path in args.component
    }
    combined = combine_line_crop_components(components)
    write_jsonl(args.output, combined)
    manifest = {
        "schema_version": "1.0",
        "dataset_path": str(args.output),
        "dataset_sha256": sha256_file(args.output),
        "sample_count": len(combined),
        "lecture_ids": sorted({sample.lecture_id for sample in combined}),
        "label_sources": dict(
            sorted(Counter(sample.label_source.value for sample in combined).items())
        ),
        "components": component_hashes,
    }
    _write_json(args.manifest, manifest)
    _print_json(manifest)
    return 0


def _audit(args: argparse.Namespace) -> int:
    samples = load_jsonl(args.dataset, LineCropSample)
    dataset_hash = sha256_file(args.dataset)
    split = make_lecture_level_split(
        samples,
        dataset_sha256=dataset_hash,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    report = audit_reader_dataset(
        samples,
        split,
        dataset_sha256=dataset_hash,
    )
    _write_json(args.split_output, split.model_dump(mode="json"))
    _write_json(args.audit_output, report.model_dump(mode="json"))
    _print_json(
        {
            "command": "audit",
            "passed": report.passed,
            "dataset_sha256": dataset_hash,
            "split_sha256": sha256_file(args.split_output),
            "audit_sha256": sha256_file(args.audit_output),
            "split_sample_counts": report.split_sample_counts,
            "problems": report.problems,
            "warnings": report.warnings,
        }
    )
    return 0 if report.passed else 1


def _component_manifest(
    *,
    mode: str,
    samples_path: Path,
    samples: Sequence[LineCropSample],
    source_hashes: dict[str, str],
    parameters: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "mode": mode,
        "samples_path": str(samples_path),
        "samples_sha256": sha256_file(samples_path),
        "sample_count": len(samples),
        "lecture_ids": sorted({sample.lecture_id for sample in samples}),
        "label_sources": dict(
            sorted(Counter(sample.label_source.value for sample in samples).items())
        ),
        "source_hashes": source_hashes,
        "parameters": parameters,
    }


def _verify_hash(path: Path, expected: str | None, *, label: str) -> str:
    actual = sha256_file(path)
    normalized = (expected or "").strip().lower()
    if normalized and actual != normalized:
        raise ValueError(
            f"Frozen {label} hash mismatch: expected {normalized}, got {actual}."
        )
    return actual


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
