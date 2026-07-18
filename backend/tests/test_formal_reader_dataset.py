from pathlib import Path

import pytest
from PIL import Image

from multimodal_lab.formal_reader_dataset import (
    SOURCE_ALIGNED_REVIEW_METHOD,
    build_source_aligned_line_crops,
    build_synthetic_line_crops,
    make_line_review_template,
)
from multimodal_lab.line_crop_dataset import LineCropDatasetError
from multimodal_lab.metrics import normalize_ocr_text
from multimodal_lab.page_reading import sha256_file
from multimodal_lab.schemas import (
    GoldTextScope,
    LineLabelSource,
    LineReviewDecision,
    LineReviewExclusionReason,
    PageContent,
    PageContentBlock,
    PageReaderKind,
    StablePageReference,
)


def make_page(tmp_path: Path) -> tuple[StablePageReference, PageContent]:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    image_hash = sha256_file(image_path)
    reference = StablePageReference(
        page_event_id="event-1",
        lecture_id="lecture-validation",
        stable_frame_timestamp=10,
        image_path=str(image_path),
        page_number=2,
        gold_text="Correct label",
        gold_text_scope=GoldTextScope.verbatim_content,
    )
    blocks = [
        PageContentBlock(
            order=0,
            text="OCR label",
            polygon=[(10, 10), (110, 10), (110, 35), (10, 35)],
        ),
        PageContentBlock(
            order=1,
            text="Stanford",
            polygon=[(120, 70), (190, 70), (190, 90), (120, 90)],
        ),
    ]
    raw_text = "\n".join(block.text for block in blocks)
    content = PageContent(
        page_event_id=reference.page_event_id,
        lecture_id=reference.lecture_id,
        page_number=reference.page_number,
        stable_frame_timestamp=reference.stable_frame_timestamp,
        image_path=str(image_path),
        image_sha256=image_hash,
        reader=PageReaderKind.rapidocr,
        reader_version="rapidocr-test",
        preprocessing_version="test",
        cache_key="a" * 64,
        raw_text=raw_text,
        normalized_text=normalize_ocr_text(raw_text, case_sensitive=True),
        ordered_blocks=blocks,
        latency_seconds=0,
    )
    return reference, content


def test_reviewed_builder_requires_a_decision_for_every_polygon(tmp_path):
    reference, content = make_page(tmp_path)
    reviews = make_line_review_template([content])

    with pytest.raises(LineCropDatasetError, match="still pending"):
        build_source_aligned_line_crops(
            [reference],
            [content],
            reviews,
            image_root=tmp_path,
            output_dir=tmp_path / "crops",
        )


def test_reviewed_builder_uses_corrected_source_label_not_ocr_text(tmp_path):
    reference, content = make_page(tmp_path)
    template = make_line_review_template([content])
    reviews = [
        template[0].model_copy(
            update={
                "decision": LineReviewDecision.include,
                "corrected_text": "Correct label",
                "review_method": SOURCE_ALIGNED_REVIEW_METHOD,
            }
        ),
        template[1].model_copy(
            update={
                "decision": LineReviewDecision.exclude,
                "exclusion_reason": LineReviewExclusionReason.non_content,
                "review_method": SOURCE_ALIGNED_REVIEW_METHOD,
            }
        ),
    ]

    samples = build_source_aligned_line_crops(
        [reference],
        [content],
        reviews,
        image_root=tmp_path,
        output_dir=tmp_path / "crops",
    )

    assert len(samples) == 1
    assert samples[0].text == "Correct label"
    assert samples[0].text != content.ordered_blocks[0].text
    assert samples[0].label_source is LineLabelSource.source_aligned


def test_synthetic_builder_is_deterministic_and_train_only(tmp_path):
    reference = StablePageReference(
        page_event_id="train-page",
        lecture_id="lecture-train",
        stable_frame_timestamp=1,
        image_path="unused.png",
        page_number=1,
        gold_text="Alpha line\nBeta line",
        gold_text_scope=GoldTextScope.verbatim_content,
    )

    first = build_synthetic_line_crops(
        [reference],
        image_root=tmp_path,
        output_dir=tmp_path / "first",
        font_path=None,
        seed=5,
        variants_per_line=2,
    )
    second = build_synthetic_line_crops(
        [reference],
        image_root=tmp_path,
        output_dir=tmp_path / "second",
        font_path=None,
        seed=5,
        variants_per_line=2,
    )

    assert len(first) == 4
    assert [sample.sample_id for sample in first] == [
        sample.sample_id for sample in second
    ]
    assert [sample.crop_sha256 for sample in first] == [
        sample.crop_sha256 for sample in second
    ]
    assert {sample.label_source for sample in first} == {
        LineLabelSource.synthetic_render
    }
