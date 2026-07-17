import pytest

from multimodal_lab.metrics import (
    character_error_rate,
    evaluate_card_conversion,
    evaluate_page_reading,
    evaluate_transition_detection,
    levenshtein_distance,
    precision_recall_f1,
    technical_term_recall,
    word_error_rate,
)
from multimodal_lab.schemas import (
    CardEvaluationCounts,
    SlideTransitionAnnotation,
    SlideTransitionPrediction,
    TransitionEventType,
)


def make_annotation(
    event_id: str,
    start: float,
    stable_at: float,
    event_type: TransitionEventType,
) -> SlideTransitionAnnotation:
    page_states = {
        TransitionEventType.page_change: (1, 2),
        TransitionEventType.content_build: (2, 2),
        TransitionEventType.enter_slide: (None, 2),
        TransitionEventType.leave_slide: (2, None),
        TransitionEventType.non_semantic_motion: (None, None),
    }
    from_page, to_page = page_states[event_type]
    return SlideTransitionAnnotation(
        event_id=event_id,
        lecture_id="lecture-2",
        change_start_seconds=start,
        stable_at_seconds=stable_at,
        from_page=from_page,
        to_page=to_page,
        event_type=event_type,
    )


def test_precision_recall_f1_treats_empty_event_set_as_perfect():
    result = precision_recall_f1(
        true_positives=0,
        false_positives=0,
        false_negatives=0,
    )

    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_transition_metrics_use_one_to_one_matching_and_count_duplicates():
    annotations = [
        make_annotation(
            "page-change",
            10.0,
            10.5,
            TransitionEventType.page_change,
        ),
        make_annotation(
            "cursor-motion",
            20.0,
            20.0,
            TransitionEventType.non_semantic_motion,
        ),
        make_annotation(
            "content-build",
            30.0,
            31.0,
            TransitionEventType.content_build,
        ),
    ]
    predictions = [
        SlideTransitionPrediction(timestamp_seconds=10.2),
        SlideTransitionPrediction(timestamp_seconds=10.4),
        SlideTransitionPrediction(timestamp_seconds=20.0),
        SlideTransitionPrediction(timestamp_seconds=31.5),
        SlideTransitionPrediction(timestamp_seconds=45.0),
    ]

    result = evaluate_transition_detection(
        annotations,
        predictions,
        video_duration_seconds=3600,
        tolerance_seconds=1.0,
    )

    assert result.true_positives == 2
    assert result.false_positives == 3
    assert result.false_negatives == 0
    assert result.precision == pytest.approx(0.4)
    assert result.recall == 1.0
    assert result.f1 == pytest.approx(4 / 7)
    assert result.duplicate_predictions == 1
    assert result.mean_absolute_timing_error_seconds == pytest.approx(0.85)
    assert result.mean_detection_delay_seconds == pytest.approx(0.85)
    assert result.false_positives_per_hour == 3.0
    assert result.matched_event_pairs == [(0, 0), (2, 3)]


def test_transition_metrics_can_require_the_correct_event_type():
    annotation = make_annotation(
        "page-change",
        10.0,
        10.2,
        TransitionEventType.page_change,
    )
    prediction = SlideTransitionPrediction(
        timestamp_seconds=10.1,
        event_type=TransitionEventType.content_build,
    )

    relaxed = evaluate_transition_detection(
        [annotation],
        [prediction],
        video_duration_seconds=60,
    )
    strict = evaluate_transition_detection(
        [annotation],
        [prediction],
        video_duration_seconds=60,
        require_event_type=True,
    )

    assert relaxed.f1 == 1.0
    assert strict.true_positives == 0
    assert strict.false_positives == 1
    assert strict.false_negatives == 1
    assert strict.f1 == 0.0


def test_non_semantic_predictions_are_treated_as_detector_abstentions():
    result = evaluate_transition_detection(
        [],
        [
            SlideTransitionPrediction(
                timestamp_seconds=5.0,
                event_type=TransitionEventType.non_semantic_motion,
            )
        ],
        video_duration_seconds=60,
    )

    assert result.f1 == 1.0
    assert result.false_positives == 0


def test_transition_metrics_reject_timestamps_outside_the_video():
    with pytest.raises(ValueError, match="Prediction time"):
        evaluate_transition_detection(
            [],
            [SlideTransitionPrediction(timestamp_seconds=61.0)],
            video_duration_seconds=60,
        )


def test_levenshtein_distance_handles_strings_and_token_sequences():
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance(
        ["deep", "neural", "networks"],
        ["deep", "neural", "network"],
    ) == 1


def test_page_reading_metrics_measure_text_and_technical_terms():
    result = evaluate_page_reading(
        "CNN uses a ResNet-50 backbone",
        "cnn uses a ResNet-50 backbon",
        technical_terms=["CNN", "ResNet-50", "SVD"],
    )

    assert result.character_error_rate == pytest.approx(1 / 29)
    assert result.word_error_rate == pytest.approx(1 / 5)
    assert result.exact_match is False
    assert result.technical_term_recall == pytest.approx(2 / 3)


def test_text_normalization_and_term_boundaries_are_explicit():
    assert character_error_rate("Deep   Learning", "deep learning") == 0.0
    assert word_error_rate("Deep   Learning", "deep learning") == 0.0
    assert technical_term_recall(["CNN"], "A CNNs model") == 0.0


def test_text_metrics_reject_empty_references():
    with pytest.raises(ValueError, match="Reference text"):
        character_error_rate("", "prediction")
    with pytest.raises(ValueError, match="Reference text"):
        word_error_rate("", "prediction")
    with pytest.raises(ValueError, match="technical term"):
        technical_term_recall([], "prediction")


def test_card_conversion_metrics_keep_precision_and_coverage_separate():
    result = evaluate_card_conversion(
        CardEvaluationCounts(
            supported_claims=8,
            total_claims=10,
            recovered_concepts=6,
            gold_concepts=8,
            correct_citations=7,
            total_citations=8,
            claims_with_valid_citation=7,
            claims_requiring_citation=10,
            accepted_without_edit=4,
            generated_cards=5,
            usable_cards=5,
            eligible_concepts=8,
        )
    )

    assert result.grounded_claim_precision == pytest.approx(0.8)
    assert result.concept_recall == pytest.approx(0.75)
    assert result.citation_correctness == pytest.approx(0.875)
    assert result.citation_completeness == pytest.approx(0.7)
    assert result.no_edit_acceptance_rate == pytest.approx(0.8)
    assert result.usable_card_conversion == pytest.approx(0.625)


def test_zero_generated_cards_do_not_receive_free_acceptance_credit():
    result = evaluate_card_conversion(
        CardEvaluationCounts(
            supported_claims=0,
            total_claims=0,
            recovered_concepts=0,
            gold_concepts=2,
            correct_citations=0,
            total_citations=0,
            claims_with_valid_citation=0,
            claims_requiring_citation=0,
            accepted_without_edit=0,
            generated_cards=0,
            usable_cards=0,
            eligible_concepts=2,
        )
    )

    assert result.grounded_claim_precision == 0.0
    assert result.no_edit_acceptance_rate == 0.0
    assert result.usable_card_conversion == 0.0
