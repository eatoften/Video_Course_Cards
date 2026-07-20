from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import Self

from pydantic import BaseModel, Field, model_validator

from .schemas import (
    LineCropReviewRecord,
    LineReviewDecision,
    LineReviewExclusionReason,
    PageContent,
    StablePageReference,
)


SOURCE_ALIGNMENT_METHOD_V2 = "official-deck-line-alignment-v2"
_TOKEN_PATTERN = re.compile(r"\S+")
_COMPARISON_PUNCTUATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)


class SourceLineAlignmentError(ValueError):
    pass


class SourceLineAlignmentPolicy(BaseModel):
    schema_version: str = "1.0"
    minimum_include_score: float = Field(default=0.82, ge=0, le=1)
    review_band_score: float = Field(default=0.65, ge=0, le=1)
    minimum_label_characters: int = Field(default=2, ge=1)
    maximum_token_delta: int = Field(default=2, ge=0)
    boilerplate_texts: list[str] = Field(
        default_factory=lambda: [
            "stanford",
            "stanford cs231n",
            "youtube",
            "search",
            "create",
        ]
    )

    @model_validator(mode="after")
    def validate_threshold_order(self) -> Self:
        if self.review_band_score > self.minimum_include_score:
            raise ValueError(
                "review_band_score cannot exceed minimum_include_score."
            )
        return self


class SourceLineAlignmentOverride(BaseModel):
    page_event_id: str = Field(min_length=1)
    source_block_order: int = Field(ge=0)
    decision: LineReviewDecision
    corrected_text: str | None = None
    exclusion_reason: LineReviewExclusionReason | None = None
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        corrected = (self.corrected_text or "").strip()
        if self.decision is LineReviewDecision.pending:
            raise ValueError("Alignment overrides cannot remain pending.")
        if self.decision is LineReviewDecision.include:
            if not corrected or self.exclusion_reason is not None:
                raise ValueError(
                    "An include override requires corrected_text only."
                )
        elif corrected or self.exclusion_reason is None:
            raise ValueError(
                "An exclude override requires exclusion_reason only."
            )
        return self


class SourceLineAlignmentRecord(BaseModel):
    page_event_id: str
    page_number: int | None
    source_block_order: int
    detected_text: str
    matched_source_text: str | None
    match_score: float = Field(ge=0, le=1)
    decision: LineReviewDecision
    exclusion_reason: LineReviewExclusionReason | None = None
    needs_visual_review: bool
    overridden: bool = False
    rationale: str


class SourceLineAlignmentSummary(BaseModel):
    schema_version: str = "1.0"
    review_method: str = SOURCE_ALIGNMENT_METHOD_V2
    candidate_count: int = Field(ge=1)
    included_count: int = Field(ge=0)
    excluded_count: int = Field(ge=0)
    visual_review_count: int = Field(ge=0)
    override_count: int = Field(ge=0)
    exclusion_reasons: dict[str, int]
    minimum_include_score: float = Field(ge=0, le=1)
    review_band_score: float = Field(ge=0, le=1)


def align_source_line_reviews(
    references: Sequence[StablePageReference],
    contents: Sequence[PageContent],
    *,
    policy: SourceLineAlignmentPolicy | None = None,
    overrides: Sequence[SourceLineAlignmentOverride] = (),
) -> tuple[
    list[LineCropReviewRecord],
    list[SourceLineAlignmentRecord],
    SourceLineAlignmentSummary,
]:
    """Align detector polygons to official source lines without OCR labels."""

    active_policy = policy or SourceLineAlignmentPolicy()
    references_by_id = _index_unique(
        references,
        key=lambda item: item.page_event_id,
        label="reference page_event_id",
    )
    contents_by_id = _index_unique(
        contents,
        key=lambda item: item.page_event_id,
        label="page content page_event_id",
    )
    if references_by_id.keys() != contents_by_id.keys():
        raise SourceLineAlignmentError(
            "Page contents must match source references exactly."
        )
    overrides_by_key = _index_unique(
        overrides,
        key=lambda item: (item.page_event_id, item.source_block_order),
        label="alignment override page/block key",
    )
    candidate_keys = {
        (content.page_event_id, block.order)
        for content in contents
        for block in content.ordered_blocks
        if block.polygon is not None
    }
    unknown_override_keys = set(overrides_by_key) - candidate_keys
    if unknown_override_keys:
        raise SourceLineAlignmentError(
            f"Overrides target unknown detector blocks: {sorted(unknown_override_keys)}"
        )

    boilerplate = {
        _compact_for_comparison(text)
        for text in active_policy.boilerplate_texts
    }
    reviews: list[LineCropReviewRecord] = []
    records: list[SourceLineAlignmentRecord] = []

    for content in contents:
        reference = references_by_id[content.page_event_id]
        source_candidates = _source_candidates(reference.gold_text.splitlines())
        for block in content.ordered_blocks:
            if block.polygon is None:
                continue
            key = (content.page_event_id, block.order)
            matched_text, score = _best_source_match(
                block.text,
                source_candidates,
                maximum_token_delta=active_policy.maximum_token_delta,
            )
            decision, reason, rationale = _automatic_decision(
                detected_text=block.text,
                matched_text=matched_text,
                score=score,
                policy=active_policy,
                boilerplate=boilerplate,
            )
            override = overrides_by_key.get(key)
            if override is not None:
                decision = override.decision
                matched_text = (
                    override.corrected_text.strip()
                    if override.corrected_text is not None
                    else None
                )
                reason = override.exclusion_reason
                rationale = f"Manual protocol override: {override.rationale.strip()}"

            needs_visual_review = (
                override is None
                and active_policy.review_band_score
                <= score
                < active_policy.minimum_include_score
            )
            notes = (
                f"source_match_score={score:.6f}; {rationale}"
            )
            reviews.append(
                LineCropReviewRecord(
                    page_event_id=content.page_event_id,
                    page_content_cache_key=content.cache_key,
                    source_image_sha256=content.image_sha256,
                    source_block_order=block.order,
                    detected_text=block.text,
                    decision=decision,
                    corrected_text=(
                        matched_text
                        if decision is LineReviewDecision.include
                        else None
                    ),
                    exclusion_reason=(
                        reason
                        if decision is LineReviewDecision.exclude
                        else None
                    ),
                    review_method=SOURCE_ALIGNMENT_METHOD_V2,
                    notes=notes,
                )
            )
            records.append(
                SourceLineAlignmentRecord(
                    page_event_id=content.page_event_id,
                    page_number=content.page_number,
                    source_block_order=block.order,
                    detected_text=block.text,
                    matched_source_text=matched_text,
                    match_score=score,
                    decision=decision,
                    exclusion_reason=reason,
                    needs_visual_review=needs_visual_review,
                    overridden=override is not None,
                    rationale=rationale,
                )
            )

    if not reviews:
        raise SourceLineAlignmentError(
            "No detector polygons are available for source alignment."
        )
    decision_counts = Counter(review.decision.value for review in reviews)
    exclusion_counts = Counter(
        review.exclusion_reason.value
        for review in reviews
        if review.exclusion_reason is not None
    )
    summary = SourceLineAlignmentSummary(
        candidate_count=len(reviews),
        included_count=decision_counts[LineReviewDecision.include.value],
        excluded_count=decision_counts[LineReviewDecision.exclude.value],
        visual_review_count=sum(record.needs_visual_review for record in records),
        override_count=sum(record.overridden for record in records),
        exclusion_reasons=dict(sorted(exclusion_counts.items())),
        minimum_include_score=active_policy.minimum_include_score,
        review_band_score=active_policy.review_band_score,
    )
    return reviews, records, summary


def _automatic_decision(
    *,
    detected_text: str,
    matched_text: str | None,
    score: float,
    policy: SourceLineAlignmentPolicy,
    boilerplate: set[str],
) -> tuple[
    LineReviewDecision,
    LineReviewExclusionReason | None,
    str,
]:
    compact = _compact_for_comparison(detected_text)
    if compact in boilerplate:
        return (
            LineReviewDecision.exclude,
            LineReviewExclusionReason.outside_scope,
            "Known player/footer boilerplate is outside the frozen slide text.",
        )
    if len(compact) < policy.minimum_label_characters:
        return (
            LineReviewDecision.exclude,
            LineReviewExclusionReason.math_or_symbol_only,
            "The detector block is too short for an independently auditable label.",
        )
    if not any(character.isascii() and character.isalnum() for character in compact):
        return (
            LineReviewDecision.exclude,
            LineReviewExclusionReason.unreadable,
            "The block contains no recoverable ASCII alphanumeric content.",
        )
    if matched_text is not None and score >= policy.minimum_include_score:
        return (
            LineReviewDecision.include,
            None,
            "The polygon aligns to an official-deck line or contiguous token span.",
        )
    return (
        LineReviewDecision.exclude,
        LineReviewExclusionReason.outside_scope,
        "No official-deck line span reached the frozen include threshold.",
    )


def _source_candidates(
    source_lines: Sequence[str],
) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for raw_line in source_lines:
        line = raw_line.strip()
        tokens = _TOKEN_PATTERN.findall(line)
        for start in range(len(tokens)):
            for end in range(start + 1, len(tokens) + 1):
                candidate = " ".join(tokens[start:end])
                comparison = _compact_for_comparison(candidate)
                candidate_key = _exact_surface_key(candidate)
                if comparison and candidate_key not in seen:
                    candidates.append(candidate)
                    seen.add(candidate_key)
    return candidates


def _best_source_match(
    detected_text: str,
    source_candidates: Sequence[str],
    *,
    maximum_token_delta: int,
) -> tuple[str | None, float]:
    detected = _compact_for_comparison(detected_text)
    if not detected:
        return None, 0.0
    detected_tokens = max(1, len(_TOKEN_PATTERN.findall(detected_text)))
    detected_surface = _surface_for_comparison(detected_text)
    best_text: str | None = None
    best_score = 0.0
    best_rank: tuple[float, int, int, float, int, int, int] | None = None
    for candidate_index, candidate in enumerate(source_candidates):
        candidate_tokens = max(1, len(_TOKEN_PATTERN.findall(candidate)))
        if abs(candidate_tokens - detected_tokens) > maximum_token_delta:
            continue
        comparison = _compact_for_comparison(candidate)
        score = SequenceMatcher(None, detected, comparison).ratio()
        candidate_surface = _surface_for_comparison(candidate)
        rank = (
            score,
            int(candidate.strip() == detected_text.strip()),
            int(candidate_surface == detected_surface),
            SequenceMatcher(
                None,
                detected_surface,
                candidate_surface,
            ).ratio(),
            -abs(len(comparison) - len(detected)),
            -abs(candidate_tokens - detected_tokens),
            -candidate_index,
        )
        if best_rank is None or rank > best_rank:
            best_text = candidate
            best_score = score
            best_rank = rank
    return best_text, best_score


def _compact_for_comparison(text: str) -> str:
    normalized = _surface_for_comparison(text)
    return "".join(
        character
        for character in normalized
        if character.isalnum()
    )


def _surface_for_comparison(text: str) -> str:
    return _exact_surface_key(text).casefold()


def _exact_surface_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).translate(
        _COMPARISON_PUNCTUATION
    )
    return " ".join(normalized.split())


def _index_unique(items, *, key, label: str):
    indexed = {}
    for item in items:
        value = key(item)
        if value in indexed:
            raise SourceLineAlignmentError(f"Duplicate {label}: {value}")
        indexed[value] = item
    return indexed
