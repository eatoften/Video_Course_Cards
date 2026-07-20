from __future__ import annotations

from multimodal_lab.source_line_alignment import (
    SOURCE_ALIGNMENT_METHOD_V2,
    SourceLineAlignmentOverride,
    SourceLineAlignmentPolicy,
    align_source_line_reviews,
)
from multimodal_lab.schemas import (
    GoldTextScope,
    LineReviewDecision,
    LineReviewExclusionReason,
    PageContent,
    PageContentBlock,
    PageReaderKind,
    StablePageReference,
)


def test_source_alignment_covers_every_polygon_and_uses_source_labels() -> None:
    reference, content = _page(
        gold_text="Numerical gradient: slow, approximate, easy to write\n3072",
        blocks=[
            "Numerical gradient: slow , approximate , easy to write",
            "3072",
            "Stanford CS231n",
            "+",
        ],
    )

    reviews, records, summary = align_source_line_reviews([reference], [content])

    assert len(reviews) == len(records) == 4
    assert reviews[0].decision is LineReviewDecision.include
    assert reviews[0].corrected_text == (
        "Numerical gradient: slow, approximate, easy to write"
    )
    assert reviews[1].decision is LineReviewDecision.include
    assert reviews[1].corrected_text == "3072"
    assert reviews[2].exclusion_reason is LineReviewExclusionReason.outside_scope
    assert reviews[3].exclusion_reason is LineReviewExclusionReason.math_or_symbol_only
    assert all(review.review_method == SOURCE_ALIGNMENT_METHOD_V2 for review in reviews)
    assert summary.included_count == 2
    assert summary.excluded_count == 2


def test_source_alignment_can_label_a_contiguous_source_span() -> None:
    reference, content = _page(
        gold_text="Computer Vision is everywhere!",
        blocks=["Computer Vision"],
    )

    reviews, _, _ = align_source_line_reviews([reference], [content])

    assert reviews[0].corrected_text == "Computer Vision"


def test_source_alignment_tie_break_prefers_exact_surface_text() -> None:
    reference, content = _page(
        gold_text="dog, cat, truck, plane\ncat\ntruck",
        blocks=["cat", "truck"],
    )

    reviews, _, _ = align_source_line_reviews([reference], [content])

    assert [review.corrected_text for review in reviews] == ["cat", "truck"]


def test_source_alignment_tie_break_preserves_visible_case() -> None:
    reference, content = _page(
        gold_text="Backprop with Matrices\nJacobian matrices",
        blocks=["matrices"],
    )

    reviews, _, _ = align_source_line_reviews([reference], [content])

    assert reviews[0].corrected_text == "matrices"


def test_source_alignment_marks_near_threshold_records_for_visual_review() -> None:
    reference, content = _page(
        gold_text="Backpropagation",
        blocks=["Back propogtion"],
    )
    policy = SourceLineAlignmentPolicy(
        minimum_include_score=0.95,
        review_band_score=0.5,
    )

    reviews, records, summary = align_source_line_reviews(
        [reference],
        [content],
        policy=policy,
    )

    assert reviews[0].decision is LineReviewDecision.exclude
    assert records[0].needs_visual_review is True
    assert summary.visual_review_count == 1


def test_source_alignment_override_is_explicit_and_auditable() -> None:
    reference, content = _page(
        gold_text="fully-connected networks",
        blocks=["(fu  -, u r -l,,"],
    )
    override = SourceLineAlignmentOverride(
        page_event_id=reference.page_event_id,
        source_block_order=0,
        decision=LineReviewDecision.include,
        corrected_text="fully-connected networks",
        rationale="Stable frame and official slide visually identify this line.",
    )

    reviews, records, summary = align_source_line_reviews(
        [reference],
        [content],
        overrides=[override],
    )

    assert reviews[0].decision is LineReviewDecision.include
    assert reviews[0].corrected_text == "fully-connected networks"
    assert records[0].overridden is True
    assert "Manual protocol override" in records[0].rationale
    assert summary.override_count == 1


def _page(
    *,
    gold_text: str,
    blocks: list[str],
) -> tuple[StablePageReference, PageContent]:
    reference = StablePageReference(
        page_event_id="lecture-page-1",
        lecture_id="lecture",
        stable_frame_timestamp=10,
        image_path="frame.png",
        page_number=1,
        gold_text=gold_text,
        gold_text_scope=GoldTextScope.verbatim_content,
    )
    content = PageContent(
        page_event_id=reference.page_event_id,
        lecture_id=reference.lecture_id,
        page_number=reference.page_number,
        stable_frame_timestamp=reference.stable_frame_timestamp,
        image_path=reference.image_path,
        image_sha256="a" * 64,
        reader=PageReaderKind.rapidocr,
        reader_version="rapidocr-test",
        preprocessing_version="test",
        cache_key="b" * 64,
        raw_text="\n".join(blocks),
        normalized_text=" ".join(blocks),
        ordered_blocks=[
            PageContentBlock(
                order=index,
                text=text,
                polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
            )
            for index, text in enumerate(blocks)
        ],
        latency_seconds=0,
    )
    return reference, content
