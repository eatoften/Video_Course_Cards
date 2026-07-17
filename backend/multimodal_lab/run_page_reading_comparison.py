from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path

from app.source_asset_store import list_source_units_for_asset

from .annotation_io import load_jsonl, write_jsonl
from .page_reading import (
    GoldReferencePageReader,
    NativeSourcePageReader,
    PageReadContext,
    PageReader,
    RapidOcrPageReader,
    prepare_page_context,
)
from .page_reading_evaluation import evaluate_page_contents
from .schemas import (
    PageContent,
    PageReaderKind,
    PageReaderRunResult,
    PageReadingComparisonReport,
    StablePageReference,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare gold, native-source, and training-free OCR page readers "
            "on one stable-frame reference bundle."
        )
    )
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--image-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-asset-id")
    parser.add_argument(
        "--readers",
        nargs="+",
        choices=[reader.value for reader in PageReaderKind],
        default=[reader.value for reader in PageReaderKind],
    )
    parser.add_argument(
        "--expected-references-sha256",
        help="Fail before inference if the gold reference file changed.",
    )
    parser.add_argument(
        "--held-out",
        action="store_true",
        help="Mark the report as held-out rather than calibration-only.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference_bytes = args.references.read_bytes()
    references_sha256 = hashlib.sha256(reference_bytes).hexdigest()
    expected_hash = (args.expected_references_sha256 or "").strip().lower()
    if expected_hash and references_sha256 != expected_hash:
        raise ValueError(
            "Reference hash does not match the frozen benchmark: "
            f"expected {expected_hash}, got {references_sha256}."
        )

    references = load_jsonl(args.references, StablePageReference)
    if not references:
        raise ValueError("The reference bundle is empty.")
    lecture_ids = {reference.lecture_id for reference in references}
    if len(lecture_ids) != 1:
        raise ValueError("One comparison run must contain exactly one lecture.")

    requested_kinds = list(
        dict.fromkeys(PageReaderKind(item) for item in args.readers)
    )
    if (
        PageReaderKind.native_source in requested_kinds
        and not args.source_asset_id
    ):
        raise ValueError("--source-asset-id is required for native_source.")

    contexts = [
        prepare_page_context(reference, image_root=args.image_root)
        for reference in references
    ]
    readers = _build_readers(
        requested_kinds,
        source_asset_id=args.source_asset_id,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_results: dict[str, PageReaderRunResult] = {}
    for reader in readers:
        contents = _read_all(reader, contexts)
        aggregate, page_results = evaluate_page_contents(references, contents)
        output_path = args.output_dir / f"{reader.kind.value}_page_contents.jsonl"
        evaluation_path = args.output_dir / f"{reader.kind.value}_evaluation.json"
        write_jsonl(output_path, contents)
        run_result = PageReaderRunResult(
            reader=reader.kind,
            output_path=str(output_path),
            evaluation_path=str(evaluation_path),
            aggregate=aggregate,
            pages=page_results,
        )
        evaluation_path.write_text(
            run_result.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        run_results[reader.kind.value] = run_result

    report = PageReadingComparisonReport(
        calibration_only=not args.held_out,
        lecture_id=next(iter(lecture_ids)),
        references_path=str(args.references),
        references_sha256=references_sha256,
        source_asset_id=args.source_asset_id,
        readers=run_results,
    )
    rendered = report.model_dump_json(indent=2)
    (args.output_dir / "comparison_report.json").write_text(
        rendered + "\n",
        encoding="utf-8",
    )
    print(rendered)
    return 0


def _build_readers(
    kinds: Sequence[PageReaderKind],
    *,
    source_asset_id: str | None,
) -> list[PageReader]:
    readers: list[PageReader] = []
    for kind in kinds:
        if kind is PageReaderKind.gold_reference:
            readers.append(GoldReferencePageReader())
        elif kind is PageReaderKind.native_source:
            assert source_asset_id is not None
            units = list_source_units_for_asset(source_asset_id)
            readers.append(NativeSourcePageReader(units))
        elif kind is PageReaderKind.rapidocr:
            readers.append(RapidOcrPageReader())
    return readers


def _read_all(
    reader: PageReader,
    contexts: Sequence[PageReadContext],
) -> list[PageContent]:
    return [reader.read(context) for context in contexts]


if __name__ == "__main__":
    raise SystemExit(main())
