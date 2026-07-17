import pytest
from pydantic import ValidationError

from multimodal_lab.schemas import (
    CardEvaluationCounts,
    SlideTransitionAnnotation,
    StablePageReference,
    TransitionEventType,
)


def test_transition_annotation_requires_stability_after_change_start():
    with pytest.raises(ValidationError, match="stable_at_seconds"):
        SlideTransitionAnnotation(
            event_id="event-1",
            lecture_id="lecture-2",
            change_start_seconds=12.0,
            stable_at_seconds=11.5,
            from_page=1,
            to_page=2,
            event_type=TransitionEventType.page_change,
        )


@pytest.mark.parametrize(
    ("event_type", "from_page", "to_page", "message"),
    [
        (TransitionEventType.page_change, 4, 4, "different page"),
        (TransitionEventType.content_build, 4, 5, "same from_page"),
        (TransitionEventType.enter_slide, 4, 5, "from_page=None"),
        (TransitionEventType.leave_slide, 4, 5, "to_page=None"),
        (TransitionEventType.non_semantic_motion, 4, 5, "cannot change"),
    ],
)
def test_transition_annotation_enforces_event_page_shape(
    event_type,
    from_page,
    to_page,
    message,
):
    with pytest.raises(ValidationError, match=message):
        SlideTransitionAnnotation(
            event_id="event-1",
            lecture_id="lecture-2",
            change_start_seconds=12.0,
            stable_at_seconds=12.5,
            from_page=from_page,
            to_page=to_page,
            event_type=event_type,
        )


def test_stable_page_reference_cleans_and_deduplicates_labels():
    reference = StablePageReference(
        page_event_id="event-1",
        lecture_id="lecture-2",
        stable_frame_timestamp=15.0,
        image_path="frames/lecture-2/event-1.png",
        page_number=7,
        gold_text="Convolutional Neural Networks",
        technical_terms=[" CNN ", "ResNet", "CNN"],
        gold_concepts=["convolution", "convolution"],
    )

    assert reference.technical_terms == ["CNN", "ResNet"]
    assert reference.gold_concepts == ["convolution"]


def test_card_evaluation_counts_reject_invalid_subtotals():
    with pytest.raises(ValidationError, match="supported_claims"):
        CardEvaluationCounts(
            supported_claims=4,
            total_claims=3,
            recovered_concepts=1,
            gold_concepts=2,
            correct_citations=1,
            total_citations=1,
            claims_with_valid_citation=1,
            claims_requiring_citation=1,
            accepted_without_edit=1,
            generated_cards=1,
            usable_cards=1,
            eligible_concepts=2,
        )
