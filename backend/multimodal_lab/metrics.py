from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

from .schemas import (
    CardConversionMetrics,
    CardEvaluationCounts,
    PageReadingMetrics,
    PrecisionRecallF1,
    SlideTransitionAnnotation,
    SlideTransitionPrediction,
    TransitionDetectionMetrics,
    TransitionEventType,
)


SequenceItem = TypeVar("SequenceItem")
POSITIVE_TRANSITION_TYPES = frozenset(
    {
        TransitionEventType.page_change,
        TransitionEventType.content_build,
        TransitionEventType.enter_slide,
        TransitionEventType.leave_slide,
    }
)


@dataclass(frozen=True)
class _MatchState:
    matched_count: int
    total_absolute_error: float
    pairs: tuple[tuple[int, int], ...]


def precision_recall_f1(
    *,
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> PrecisionRecallF1:
    counts = (true_positives, false_positives, false_negatives)
    if any(count < 0 for count in counts):
        raise ValueError("Metric counts cannot be negative.")

    if true_positives == false_positives == false_negatives == 0:
        precision = recall = f1 = 1.0
    else:
        precision_denominator = true_positives + false_positives
        recall_denominator = true_positives + false_negatives
        precision = (
            true_positives / precision_denominator
            if precision_denominator
            else 1.0
        )
        recall = (
            true_positives / recall_denominator
            if recall_denominator
            else 1.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

    return PrecisionRecallF1(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def evaluate_transition_detection(
    annotations: Sequence[SlideTransitionAnnotation],
    predictions: Sequence[SlideTransitionPrediction],
    *,
    video_duration_seconds: float,
    tolerance_seconds: float = 1.0,
    require_event_type: bool = False,
) -> TransitionDetectionMetrics:
    if video_duration_seconds <= 0:
        raise ValueError("video_duration_seconds must be positive.")
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds cannot be negative.")

    _validate_event_times(
        annotations,
        predictions,
        video_duration_seconds=video_duration_seconds,
    )

    gold_events = sorted(
        (
            (index, event)
            for index, event in enumerate(annotations)
            if event.event_type in POSITIVE_TRANSITION_TYPES
        ),
        key=lambda item: item[1].change_start_seconds,
    )
    predicted_events = sorted(
        (
            (index, event)
            for index, event in enumerate(predictions)
            if event.event_type is not TransitionEventType.non_semantic_motion
        ),
        key=lambda item: item[1].timestamp_seconds,
    )

    sorted_pairs = _align_transition_events(
        [event for _, event in gold_events],
        [event for _, event in predicted_events],
        tolerance_seconds=tolerance_seconds,
        require_event_type=require_event_type,
    )
    original_pairs = [
        (gold_events[gold_index][0], predicted_events[prediction_index][0])
        for gold_index, prediction_index in sorted_pairs
    ]

    matched_prediction_indexes = {
        prediction_index
        for _, prediction_index in sorted_pairs
    }
    duplicate_predictions = sum(
        1
        for prediction_index, (_, prediction) in enumerate(predicted_events)
        if prediction_index not in matched_prediction_indexes
        and any(
            _can_match_transition(
                annotation,
                prediction,
                tolerance_seconds=tolerance_seconds,
                require_event_type=require_event_type,
            )
            for _, annotation in gold_events
        )
    )

    true_positives = len(sorted_pairs)
    false_positives = len(predicted_events) - true_positives
    false_negatives = len(gold_events) - true_positives
    base_metrics = precision_recall_f1(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )

    delays = [
        predicted_events[prediction_index][1].timestamp_seconds
        - gold_events[gold_index][1].change_start_seconds
        for gold_index, prediction_index in sorted_pairs
    ]

    duration_hours = video_duration_seconds / 3600

    return TransitionDetectionMetrics(
        **base_metrics.model_dump(),
        matched_event_pairs=original_pairs,
        duplicate_predictions=duplicate_predictions,
        mean_absolute_timing_error_seconds=(
            sum(abs(delay) for delay in delays) / len(delays)
            if delays
            else None
        ),
        mean_detection_delay_seconds=(
            sum(delays) / len(delays)
            if delays
            else None
        ),
        false_positives_per_hour=false_positives / duration_hours,
    )


def _validate_event_times(
    annotations: Sequence[SlideTransitionAnnotation],
    predictions: Sequence[SlideTransitionPrediction],
    *,
    video_duration_seconds: float,
) -> None:
    if any(
        annotation.stable_at_seconds > video_duration_seconds
        for annotation in annotations
    ):
        raise ValueError("Annotation time exceeds the video duration.")
    if any(
        prediction.timestamp_seconds > video_duration_seconds
        for prediction in predictions
    ):
        raise ValueError("Prediction time exceeds the video duration.")


def _align_transition_events(
    annotations: Sequence[SlideTransitionAnnotation],
    predictions: Sequence[SlideTransitionPrediction],
    *,
    tolerance_seconds: float,
    require_event_type: bool,
) -> tuple[tuple[int, int], ...]:
    row_count = len(annotations) + 1
    column_count = len(predictions) + 1
    states: list[list[_MatchState | None]] = [
        [None] * column_count
        for _ in range(row_count)
    ]
    states[0][0] = _MatchState(0, 0.0, ())

    for annotation_index in range(row_count):
        for prediction_index in range(column_count):
            state = states[annotation_index][prediction_index]
            if state is None:
                continue

            if annotation_index < len(annotations):
                _keep_better_state(
                    states,
                    annotation_index + 1,
                    prediction_index,
                    state,
                )

            if prediction_index < len(predictions):
                _keep_better_state(
                    states,
                    annotation_index,
                    prediction_index + 1,
                    state,
                )

            if (
                annotation_index < len(annotations)
                and prediction_index < len(predictions)
                and _can_match_transition(
                    annotations[annotation_index],
                    predictions[prediction_index],
                    tolerance_seconds=tolerance_seconds,
                    require_event_type=require_event_type,
                )
            ):
                timing_error = abs(
                    predictions[prediction_index].timestamp_seconds
                    - annotations[annotation_index].change_start_seconds
                )
                matched_state = _MatchState(
                    matched_count=state.matched_count + 1,
                    total_absolute_error=state.total_absolute_error + timing_error,
                    pairs=state.pairs + ((annotation_index, prediction_index),),
                )
                _keep_better_state(
                    states,
                    annotation_index + 1,
                    prediction_index + 1,
                    matched_state,
                )

    final_state = states[-1][-1]
    if final_state is None:
        return ()
    return final_state.pairs


def _can_match_transition(
    annotation: SlideTransitionAnnotation,
    prediction: SlideTransitionPrediction,
    *,
    tolerance_seconds: float,
    require_event_type: bool,
) -> bool:
    if require_event_type and prediction.event_type != annotation.event_type:
        return False

    return (
        annotation.change_start_seconds - tolerance_seconds
        <= prediction.timestamp_seconds
        <= annotation.stable_at_seconds + tolerance_seconds
    )


def _keep_better_state(
    states: list[list[_MatchState | None]],
    row: int,
    column: int,
    candidate: _MatchState,
) -> None:
    current = states[row][column]
    if current is None or _state_is_better(candidate, current):
        states[row][column] = candidate


def _state_is_better(candidate: _MatchState, current: _MatchState) -> bool:
    if candidate.matched_count != current.matched_count:
        return candidate.matched_count > current.matched_count
    if candidate.total_absolute_error != current.total_absolute_error:
        return candidate.total_absolute_error < current.total_absolute_error
    return candidate.pairs < current.pairs


def normalize_ocr_text(text: str, *, case_sensitive: bool = False) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = " ".join(normalized.split())
    return normalized if case_sensitive else normalized.casefold()


def levenshtein_distance(
    reference: Sequence[SequenceItem],
    hypothesis: Sequence[SequenceItem],
) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference

    previous_row = list(range(len(hypothesis) + 1))

    for reference_index, reference_item in enumerate(reference, start=1):
        current_row = [reference_index]
        for hypothesis_index, hypothesis_item in enumerate(hypothesis, start=1):
            insertion = current_row[hypothesis_index - 1] + 1
            deletion = previous_row[hypothesis_index] + 1
            substitution = previous_row[hypothesis_index - 1] + (
                reference_item != hypothesis_item
            )
            current_row.append(min(insertion, deletion, substitution))
        previous_row = current_row

    return previous_row[-1]


def character_error_rate(
    reference_text: str,
    predicted_text: str,
    *,
    case_sensitive: bool = False,
) -> float:
    reference = normalize_ocr_text(
        reference_text,
        case_sensitive=case_sensitive,
    )
    prediction = normalize_ocr_text(
        predicted_text,
        case_sensitive=case_sensitive,
    )
    if not reference:
        raise ValueError("Reference text cannot be empty when calculating CER.")
    return levenshtein_distance(reference, prediction) / len(reference)


def word_error_rate(
    reference_text: str,
    predicted_text: str,
    *,
    case_sensitive: bool = False,
) -> float:
    reference_words = normalize_ocr_text(
        reference_text,
        case_sensitive=case_sensitive,
    ).split()
    predicted_words = normalize_ocr_text(
        predicted_text,
        case_sensitive=case_sensitive,
    ).split()
    if not reference_words:
        raise ValueError("Reference text cannot be empty when calculating WER.")
    return levenshtein_distance(reference_words, predicted_words) / len(
        reference_words
    )


def technical_term_recall(
    technical_terms: Sequence[str],
    predicted_text: str,
) -> float:
    normalized_prediction = normalize_ocr_text(predicted_text)
    normalized_terms = {
        normalize_ocr_text(term)
        for term in technical_terms
        if term.strip()
    }
    if not normalized_terms:
        raise ValueError("At least one technical term is required.")

    matched_terms = sum(
        _contains_exact_phrase(normalized_prediction, term)
        for term in normalized_terms
    )
    return matched_terms / len(normalized_terms)


def _contains_exact_phrase(text: str, phrase: str) -> bool:
    escaped_phrase = re.escape(phrase).replace(r"\ ", r"\s+")
    pattern = rf"(?<!\w){escaped_phrase}(?!\w)"
    return re.search(pattern, text) is not None


def evaluate_page_reading(
    reference_text: str,
    predicted_text: str,
    *,
    technical_terms: Sequence[str] = (),
    case_sensitive: bool = False,
) -> PageReadingMetrics:
    normalized_reference = normalize_ocr_text(
        reference_text,
        case_sensitive=case_sensitive,
    )
    normalized_prediction = normalize_ocr_text(
        predicted_text,
        case_sensitive=case_sensitive,
    )

    return PageReadingMetrics(
        character_error_rate=character_error_rate(
            reference_text,
            predicted_text,
            case_sensitive=case_sensitive,
        ),
        word_error_rate=word_error_rate(
            reference_text,
            predicted_text,
            case_sensitive=case_sensitive,
        ),
        exact_match=normalized_reference == normalized_prediction,
        technical_term_recall=(
            technical_term_recall(technical_terms, predicted_text)
            if technical_terms
            else None
        ),
    )


def evaluate_card_conversion(
    counts: CardEvaluationCounts,
) -> CardConversionMetrics:
    return CardConversionMetrics(
        grounded_claim_precision=_ratio_or_zero(
            counts.supported_claims,
            counts.total_claims,
        ),
        concept_recall=counts.recovered_concepts / counts.gold_concepts,
        citation_correctness=_ratio_or_zero(
            counts.correct_citations,
            counts.total_citations,
        ),
        citation_completeness=(
            counts.claims_with_valid_citation / counts.claims_requiring_citation
            if counts.claims_requiring_citation
            else 1.0
        ),
        no_edit_acceptance_rate=_ratio_or_zero(
            counts.accepted_without_edit,
            counts.generated_cards,
        ),
        usable_card_conversion=(
            counts.usable_cards / counts.eligible_concepts
        ),
    )


def _ratio_or_zero(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
