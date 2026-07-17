from __future__ import annotations

import math
from collections.abc import Sequence

from .metrics import (
    evaluate_page_reading,
    levenshtein_distance,
    normalize_ocr_text,
    technical_term_recall,
)
from .schemas import (
    GoldTextScope,
    PageContent,
    PageReadingAggregateMetrics,
    PageReadingSampleResult,
    StablePageReference,
)


class PageReadingEvaluationError(ValueError):
    pass


def evaluate_page_contents(
    references: Sequence[StablePageReference],
    contents: Sequence[PageContent],
) -> tuple[PageReadingAggregateMetrics, list[PageReadingSampleResult]]:
    if not references:
        raise PageReadingEvaluationError("At least one page reference is required.")
    if not contents:
        raise PageReadingEvaluationError("At least one page output is required.")
    non_verbatim = [
        reference.page_event_id
        for reference in references
        if reference.gold_text_scope is not GoldTextScope.verbatim_content
    ]
    if non_verbatim:
        raise PageReadingEvaluationError(
            "CER/WER require verbatim_content references; non-verbatim pages="
            f"{non_verbatim}."
        )

    references_by_id = _index_unique(
        references,
        key=lambda item: item.page_event_id,
        label="reference page_event_id",
    )
    contents_by_id = _index_unique(
        contents,
        key=lambda item: item.page_event_id,
        label="page output page_event_id",
    )
    missing = references_by_id.keys() - contents_by_id.keys()
    extra = contents_by_id.keys() - references_by_id.keys()
    if missing or extra:
        raise PageReadingEvaluationError(
            f"Reader outputs do not match references; missing={sorted(missing)}, "
            f"extra={sorted(extra)}."
        )

    readers = {content.reader for content in contents}
    if len(readers) != 1:
        raise PageReadingEvaluationError(
            "One evaluation run must contain exactly one reader."
        )

    page_results: list[PageReadingSampleResult] = []
    total_character_edits = 0
    total_reference_characters = 0
    total_word_edits = 0
    total_reference_words = 0
    matched_technical_terms = 0
    total_technical_terms = 0
    exact_match_count = 0
    latencies: list[float] = []

    for reference in references:
        content = contents_by_id[reference.page_event_id]
        _validate_provenance(reference, content)
        predicted_text = "" if content.abstained else content.raw_text
        metrics = evaluate_page_reading(
            reference.gold_text,
            predicted_text,
            technical_terms=reference.technical_terms,
        )
        page_results.append(
            PageReadingSampleResult(
                page_event_id=reference.page_event_id,
                page_number=reference.page_number,
                reader=content.reader,
                abstained=content.abstained,
                latency_seconds=content.latency_seconds,
                metrics=metrics,
            )
        )

        normalized_reference = normalize_ocr_text(reference.gold_text)
        normalized_prediction = normalize_ocr_text(predicted_text)
        reference_words = normalized_reference.split()
        prediction_words = normalized_prediction.split()
        total_character_edits += levenshtein_distance(
            normalized_reference,
            normalized_prediction,
        )
        total_reference_characters += len(normalized_reference)
        total_word_edits += levenshtein_distance(
            reference_words,
            prediction_words,
        )
        total_reference_words += len(reference_words)
        exact_match_count += metrics.exact_match
        latencies.append(content.latency_seconds)

        unique_terms = {
            normalize_ocr_text(term)
            for term in reference.technical_terms
            if term.strip()
        }
        total_technical_terms += len(unique_terms)
        matched_technical_terms += sum(
            technical_term_recall([term], predicted_text) == 1.0
            for term in unique_terms
        )

    page_count = len(references)
    aggregate = PageReadingAggregateMetrics(
        page_count=page_count,
        abstention_count=sum(content.abstained for content in contents),
        character_error_rate=(
            total_character_edits / total_reference_characters
        ),
        word_error_rate=total_word_edits / total_reference_words,
        exact_match_count=exact_match_count,
        exact_match_rate=exact_match_count / page_count,
        technical_term_recall=(
            matched_technical_terms / total_technical_terms
            if total_technical_terms
            else None
        ),
        mean_latency_seconds=sum(latencies) / len(latencies),
        p95_latency_seconds=_nearest_rank_percentile(latencies, 0.95),
    )
    return aggregate, page_results


def _validate_provenance(
    reference: StablePageReference,
    content: PageContent,
) -> None:
    if content.lecture_id != reference.lecture_id:
        raise PageReadingEvaluationError(
            f"Output {content.page_event_id} belongs to the wrong lecture."
        )
    if content.page_number != reference.page_number:
        raise PageReadingEvaluationError(
            f"Output {content.page_event_id} has the wrong page number."
        )
    if content.stable_frame_timestamp != reference.stable_frame_timestamp:
        raise PageReadingEvaluationError(
            f"Output {content.page_event_id} has the wrong frame timestamp."
        )


def _nearest_rank_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise PageReadingEvaluationError("Cannot calculate an empty percentile.")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _index_unique(items, *, key, label: str):
    indexed = {}
    for item in items:
        value = key(item)
        if value in indexed:
            raise PageReadingEvaluationError(f"Duplicate {label}: {value}")
        indexed[value] = item
    return indexed
