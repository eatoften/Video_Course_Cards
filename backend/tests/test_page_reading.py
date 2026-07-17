from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from app.source_asset import SourceUnit
from multimodal_lab.page_reading import (
    GoldReferencePageReader,
    NativeSourcePageReader,
    PageReadingError,
    PageReader,
    RapidOcrPageReader,
    assemble_page_text,
    compute_page_reader_cache_key,
    prepare_page_context,
)
from multimodal_lab.page_reading_evaluation import evaluate_page_contents
from multimodal_lab.run_page_reading_comparison import main as comparison_main
from multimodal_lab.schemas import (
    GoldTextScope,
    PageContent,
    PageContentBlock,
    PageReaderKind,
    StablePageReference,
)


def make_reference(image_path: str, **updates) -> StablePageReference:
    values = {
        "page_event_id": "event-1",
        "lecture_id": "lecture-1",
        "stable_frame_timestamp": 12.5,
        "image_path": image_path,
        "page_number": 3,
        "gold_text": "Singular Value Decomposition",
        "gold_text_scope": GoldTextScope.verbatim_content,
        "technical_terms": ["SVD"],
        "gold_concepts": ["matrix factorization"],
    }
    values.update(updates)
    return StablePageReference(**values)


def make_context(tmp_path, **reference_updates):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"stable-frame")
    reference = make_reference(str(image), **reference_updates)
    return reference, prepare_page_context(reference, image_root=tmp_path)


def test_page_content_requires_keyframe_provenance():
    with pytest.raises(ValidationError, match="image_sha256"):
        PageContent(
            page_event_id="event-1",
            lecture_id="lecture-1",
            page_number=1,
            stable_frame_timestamp=1.0,
            image_path="frame.jpg",
            reader=PageReaderKind.gold_reference,
            reader_version="v1",
            preprocessing_version="v1",
            cache_key="a" * 64,
            raw_text="text",
            normalized_text="text",
            ordered_blocks=[PageContentBlock(order=0, text="text")],
            latency_seconds=0,
        )


def test_assemble_page_text_requires_deterministic_consecutive_order():
    assert assemble_page_text(
        [
            PageContentBlock(order=1, text="second"),
            PageContentBlock(order=0, text="first"),
        ]
    ) == "first\nsecond"

    with pytest.raises(PageReadingError, match="consecutive"):
        assemble_page_text([PageContentBlock(order=2, text="third")])


def test_cache_key_changes_with_image_model_and_preprocessing():
    base = {
        "image_sha256": "a" * 64,
        "reader": PageReaderKind.rapidocr,
        "reader_version": "reader-v1",
        "preprocessing_version": "preprocess-v1",
    }
    original = compute_page_reader_cache_key(**base)

    assert compute_page_reader_cache_key(
        **{**base, "image_sha256": "b" * 64}
    ) != original
    assert compute_page_reader_cache_key(
        **{**base, "reader_version": "reader-v2"}
    ) != original
    assert compute_page_reader_cache_key(
        **{**base, "preprocessing_version": "preprocess-v2"}
    ) != original


def test_rapidocr_adapter_preserves_raw_symbols_and_maps_blocks(tmp_path):
    reference, context = make_context(
        tmp_path,
        gold_text="SVD uses lambda",
    )
    result = SimpleNamespace(
        txts=("SVD", "lambda: lambda"),
        scores=(0.9, 0.8),
        boxes=np.asarray(
            [
                [[0, 0], [10, 0], [10, 5], [0, 5]],
                [[0, 6], [20, 6], [20, 11], [0, 11]],
            ],
            dtype=np.float32,
        ),
    )
    reader = RapidOcrPageReader(engine=lambda path: result)

    content = reader.read(context)

    assert isinstance(reader, PageReader)
    assert content.raw_text == "SVD\nlambda: lambda"
    assert content.normalized_text == "SVD lambda: lambda"
    assert content.image_sha256 == context.image_sha256
    assert content.ordered_blocks[1].polygon == [
        (0.0, 6.0),
        (20.0, 6.0),
        (20.0, 11.0),
        (0.0, 11.0),
    ]
    assert content.confidence == pytest.approx(0.85)
    assert content.reader is PageReaderKind.rapidocr
    assert reference.gold_text == "SVD uses lambda"


def test_rapidocr_adapter_filters_course_chrome_and_attribution(tmp_path):
    _, context = make_context(tmp_path)
    result = SimpleNamespace(
        txts=(
            "Image Classification",
            "This image is CC0 1.0 public domain",
            "Stanford",
            "Stanford CS231n",
        ),
        scores=(0.99, 0.98, 0.97, 0.96),
        boxes=np.asarray(
            [
                [[10, 10], [180, 10], [180, 25], [10, 25]],
                [[10, 50], [120, 50], [120, 55], [10, 55]],
                [[150, 82], [195, 82], [195, 90], [150, 90]],
                [[0, 93], [80, 93], [80, 99], [0, 99]],
            ],
            dtype=np.float32,
        ),
        img=np.zeros((100, 200, 3), dtype=np.uint8),
    )

    content = RapidOcrPageReader(engine=lambda path: result).read(context)

    assert content.raw_text == "Image Classification"
    assert len(content.ordered_blocks) == 1


def test_native_reader_removes_header_page_number_and_attribution(tmp_path):
    _, context = make_context(tmp_path)
    reader = NativeSourcePageReader(
        [
            SourceUnit(
                id="unit-3",
                asset_id="asset-1",
                unit_type="page",
                ordinal=2,
                text=(
                    "Stanford CS231n Lecture 3 - April 7, 20263\n"
                    "Image Classification\n"
                    "3\n"
                    "This image by Example is\n"
                    "licensed under CC-BY 2.0\n"
                    "A Core Task in Computer Vision"
                ),
                locator={"page_number": 3},
            )
        ]
    )

    content = reader.read(context)

    assert content.raw_text == (
        "Image Classification\nA Core Task in Computer Vision"
    )


def test_native_reader_uses_source_unit_and_abstains_for_missing_page(tmp_path):
    reference, context = make_context(tmp_path)
    unit = SourceUnit(
        id="unit-3",
        asset_id="asset-1",
        unit_type="page",
        ordinal=2,
        text="Singular Value Decomposition",
        locator={"page_number": 3},
    )
    reader = NativeSourcePageReader([unit])

    content = reader.read(context)

    assert content.source_asset_id == "asset-1"
    assert content.source_unit_id == "unit-3"
    assert content.raw_text == reference.gold_text

    missing_reference, missing_context = make_context(
        tmp_path,
        page_event_id="event-2",
        page_number=4,
    )
    missing = reader.read(missing_context)
    assert missing.abstained is True
    assert "page 4" in missing.abstention_reason
    assert missing_reference.page_number == 4


def test_gold_native_and_ocr_share_one_evaluation_interface(tmp_path):
    reference, context = make_context(tmp_path)
    gold_content = GoldReferencePageReader().read(context)
    native_content = NativeSourcePageReader(
        [
            SourceUnit(
                id="unit-3",
                asset_id="asset-1",
                unit_type="page",
                ordinal=2,
                text=reference.gold_text,
                locator={"page_number": 3},
            )
        ]
    ).read(context)
    ocr_content = RapidOcrPageReader(
        engine=lambda path: SimpleNamespace(
            txts=(reference.gold_text,),
            scores=(1.0,),
            boxes=np.asarray([[[0, 0], [10, 0], [10, 5], [0, 5]]]),
        )
    ).read(context)

    for content in (gold_content, native_content, ocr_content):
        aggregate, pages = evaluate_page_contents([reference], [content])
        assert aggregate.character_error_rate == 0
        assert aggregate.word_error_rate == 0
        assert aggregate.exact_match_rate == 1
        assert pages[0].metrics.exact_match is True


def test_comparison_cli_writes_gold_outputs_and_marks_held_out(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"stable-frame")
    references = tmp_path / "references.jsonl"
    references.write_text(
        make_reference(str(image)).model_dump_json() + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "comparison"

    exit_code = comparison_main(
        [
            "--references",
            str(references),
            "--image-root",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--readers",
            "gold_reference",
            "--held-out",
        ]
    )

    report = PageContent.model_validate_json(
        (output_dir / "gold_reference_page_contents.jsonl").read_text()
    )
    assert exit_code == 0
    assert report.reader is PageReaderKind.gold_reference
    assert (output_dir / "comparison_report.json").is_file()


def test_comparison_cli_checks_reference_hash_before_reading_images(tmp_path):
    references = tmp_path / "references.jsonl"
    references.write_text(
        make_reference("missing.jpg").model_dump_json() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frozen benchmark"):
        comparison_main(
            [
                "--references",
                str(references),
                "--output-dir",
                str(tmp_path / "comparison"),
                "--readers",
                "gold_reference",
                "--expected-references-sha256",
                "0" * 64,
            ]
        )


def test_evaluator_rejects_semantic_summary_as_ocr_gold(tmp_path):
    reference, context = make_context(
        tmp_path,
        gold_text_scope=GoldTextScope.semantic_summary,
    )
    content = GoldReferencePageReader().read(context)

    with pytest.raises(ValueError, match="require verbatim_content"):
        evaluate_page_contents([reference], [content])
