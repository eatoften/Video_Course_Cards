from __future__ import annotations

from enum import Enum
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator


class TransitionEventType(str, Enum):
    page_change = "page_change"
    content_build = "content_build"
    enter_slide = "enter_slide"
    leave_slide = "leave_slide"
    non_semantic_motion = "non_semantic_motion"


class DatasetSplit(str, Enum):
    smoke = "smoke"
    train = "train"
    validation = "validation"
    test = "test"


class LectureDatasetManifest(BaseModel):
    schema_version: str = "1.0"
    lecture_id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)
    course_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    video_path: str = Field(min_length=1)
    transcript_path: str = Field(min_length=1)
    source_deck_urls: list[str] = Field(min_length=1)
    source_asset_id: str | None = None
    duration_seconds: float = Field(gt=0)
    pilot_start_seconds: float = Field(ge=0)
    pilot_end_seconds: float = Field(gt=0)
    split: DatasetSplit = DatasetSplit.smoke

    @model_validator(mode="after")
    def validate_pilot_interval(self) -> Self:
        if self.pilot_end_seconds <= self.pilot_start_seconds:
            raise ValueError(
                "pilot_end_seconds must be greater than pilot_start_seconds."
            )
        if self.pilot_end_seconds > self.duration_seconds:
            raise ValueError("Pilot interval cannot exceed the lecture duration.")
        return self


class SlideTransitionAnnotation(BaseModel):
    event_id: str = Field(min_length=1)
    lecture_id: str = Field(min_length=1)
    change_start_seconds: float = Field(ge=0)
    stable_at_seconds: float = Field(ge=0)
    from_page: int | None = Field(default=None, ge=1)
    to_page: int | None = Field(default=None, ge=1)
    event_type: TransitionEventType
    notes: str | None = None

    @model_validator(mode="after")
    def validate_time_order(self) -> Self:
        if self.stable_at_seconds < self.change_start_seconds:
            raise ValueError(
                "stable_at_seconds must be greater than or equal to "
                "change_start_seconds."
            )
        return self

    @model_validator(mode="after")
    def validate_page_state(self) -> Self:
        if self.event_type is TransitionEventType.page_change:
            if self.from_page is None or self.to_page is None:
                raise ValueError("page_change requires from_page and to_page.")
            if self.from_page == self.to_page:
                raise ValueError("page_change must move to a different page.")
        elif self.event_type is TransitionEventType.content_build:
            if self.from_page is None or self.from_page != self.to_page:
                raise ValueError(
                    "content_build requires the same from_page and to_page."
                )
        elif self.event_type is TransitionEventType.enter_slide:
            if self.from_page is not None or self.to_page is None:
                raise ValueError(
                    "enter_slide requires from_page=None and a to_page."
                )
        elif self.event_type is TransitionEventType.leave_slide:
            if self.from_page is None or self.to_page is not None:
                raise ValueError(
                    "leave_slide requires a from_page and to_page=None."
                )
        elif (
            self.from_page is not None
            and self.to_page is not None
            and self.from_page != self.to_page
        ):
            raise ValueError(
                "non_semantic_motion cannot change the current page."
            )
        return self


class SlideTransitionPrediction(BaseModel):
    timestamp_seconds: float = Field(ge=0)
    event_type: TransitionEventType | None = None
    score: float | None = Field(default=None, ge=0, le=1)


class StablePageReference(BaseModel):
    page_event_id: str = Field(min_length=1)
    lecture_id: str = Field(min_length=1)
    stable_frame_timestamp: float = Field(ge=0)
    image_path: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    gold_text: str = Field(min_length=1)
    technical_terms: list[str] = Field(default_factory=list)
    gold_concepts: list[str] = Field(default_factory=list)

    @field_validator("technical_terms", "gold_concepts")
    @classmethod
    def validate_nonempty_unique_values(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()

        for value in values:
            normalized = value.strip()
            if not normalized:
                raise ValueError("Reference terms and concepts cannot be blank.")
            if normalized not in seen:
                cleaned.append(normalized)
                seen.add(normalized)

        return cleaned


class PrecisionRecallF1(BaseModel):
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)


class TransitionDetectionMetrics(PrecisionRecallF1):
    matched_event_pairs: list[tuple[int, int]] = Field(default_factory=list)
    duplicate_predictions: int = Field(ge=0)
    mean_absolute_timing_error_seconds: float | None = Field(default=None, ge=0)
    mean_detection_delay_seconds: float | None = None
    false_positives_per_hour: float = Field(ge=0)


class PageReadingMetrics(BaseModel):
    character_error_rate: float = Field(ge=0)
    word_error_rate: float = Field(ge=0)
    exact_match: bool
    technical_term_recall: float | None = Field(default=None, ge=0, le=1)


class CardEvaluationCounts(BaseModel):
    supported_claims: int = Field(ge=0)
    total_claims: int = Field(ge=0)
    recovered_concepts: int = Field(ge=0)
    gold_concepts: int = Field(ge=1)
    correct_citations: int = Field(ge=0)
    total_citations: int = Field(ge=0)
    claims_with_valid_citation: int = Field(ge=0)
    claims_requiring_citation: int = Field(ge=0)
    accepted_without_edit: int = Field(ge=0)
    generated_cards: int = Field(ge=0)
    usable_cards: int = Field(ge=0)
    eligible_concepts: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_subtotals(self) -> Self:
        pairs = [
            ("supported_claims", self.supported_claims, "total_claims", self.total_claims),
            ("recovered_concepts", self.recovered_concepts, "gold_concepts", self.gold_concepts),
            ("correct_citations", self.correct_citations, "total_citations", self.total_citations),
            (
                "claims_with_valid_citation",
                self.claims_with_valid_citation,
                "claims_requiring_citation",
                self.claims_requiring_citation,
            ),
            (
                "accepted_without_edit",
                self.accepted_without_edit,
                "generated_cards",
                self.generated_cards,
            ),
            ("usable_cards", self.usable_cards, "eligible_concepts", self.eligible_concepts),
        ]

        for numerator_name, numerator, denominator_name, denominator in pairs:
            if numerator > denominator:
                raise ValueError(
                    f"{numerator_name} cannot exceed {denominator_name}."
                )

        return self


class CardConversionMetrics(BaseModel):
    grounded_claim_precision: float = Field(ge=0, le=1)
    concept_recall: float = Field(ge=0, le=1)
    citation_correctness: float = Field(ge=0, le=1)
    citation_completeness: float = Field(ge=0, le=1)
    no_edit_acceptance_rate: float = Field(ge=0, le=1)
    usable_card_conversion: float = Field(ge=0, le=1)


class AnnotationBundleSummary(BaseModel):
    lecture_count: int = Field(ge=0)
    transition_event_count: int = Field(ge=0)
    positive_transition_count: int = Field(ge=0)
    stable_page_reference_count: int = Field(ge=0)
