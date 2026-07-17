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


class GoldTextScope(str, Enum):
    semantic_summary = "semantic_summary"
    verbatim_content = "verbatim_content"


class TransitionDetectorVariant(str, Enum):
    scene_only = "scene_only"
    scene_state = "scene_state"
    spatial_state = "spatial_state"


class PixelCrop(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class TransitionBaselineConfig(BaseModel):
    schema_version: str = "1.0"
    profile_name: str = Field(min_length=1)
    slide_crop: PixelCrop
    sample_fps: float = Field(default=2.0, gt=0)
    scene_scale_width: int = Field(default=320, gt=0)
    raw_frame_width: int = Field(default=160, gt=0)
    raw_frame_height: int = Field(default=90, gt=0)
    scene_score_threshold: float = Field(default=0.002, ge=0, le=1)
    stable_lookahead_seconds: float = Field(default=1.0, gt=0)
    changed_pixel_luma_threshold: float = Field(default=10.0, gt=0, le=255)
    header_fraction: float = Field(default=1 / 6, gt=0, lt=1)
    footer_fraction: float = Field(default=1 / 9, gt=0, lt=1)
    marker_footer_fraction: float = Field(default=0.08, gt=0, lt=1)
    marker_red_minimum: int = Field(default=45, ge=0, le=255)
    marker_red_to_green_ratio: float = Field(default=1.35, gt=0)
    marker_red_to_blue_ratio: float = Field(default=1.2, gt=0)
    marker_enter_ratio: float = Field(default=0.65, ge=0, le=1)
    marker_exit_ratio: float = Field(default=0.35, ge=0, le=1)
    slide_state_min_run_samples: int = Field(default=2, ge=1)
    minimum_body_changed_ratio: float = Field(default=0.004, ge=0, le=1)
    page_change_header_ratio: float = Field(default=0.025, ge=0, le=1)
    minimum_nonsemantic_changed_ratio: float = Field(
        default=0.01,
        ge=0,
        le=1,
    )
    footer_overlay_changed_ratio: float = Field(default=0.1, ge=0, le=1)
    grouping_gap_seconds: float = Field(default=0.75, ge=0)
    nms_window_seconds: float = Field(default=1.5, ge=0)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> Self:
        if self.marker_exit_ratio > self.marker_enter_ratio:
            raise ValueError(
                "marker_exit_ratio cannot exceed marker_enter_ratio."
            )
        if self.header_fraction + self.footer_fraction >= 1:
            raise ValueError(
                "header_fraction and footer_fraction must leave a body region."
            )
        return self


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
    gold_text_scope: GoldTextScope = GoldTextScope.semantic_summary
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


class PageReaderKind(str, Enum):
    gold_reference = "gold_reference"
    native_source = "native_source"
    rapidocr = "rapidocr"


class PageContentBlock(BaseModel):
    order: int = Field(ge=0)
    text: str = Field(min_length=1)
    polygon: list[tuple[float, float]] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
    )
    confidence: float | None = Field(default=None, ge=0, le=1)


class PageContent(BaseModel):
    schema_version: str = "1.0"
    page_event_id: str = Field(min_length=1)
    lecture_id: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    stable_frame_timestamp: float = Field(ge=0)
    image_path: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reader: PageReaderKind
    reader_version: str = Field(min_length=1)
    preprocessing_version: str = Field(min_length=1)
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_text: str = ""
    normalized_text: str = ""
    ordered_blocks: list[PageContentBlock] = Field(default_factory=list)
    technical_terms: list[str] = Field(default_factory=list)
    source_asset_id: str | None = None
    source_unit_id: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    latency_seconds: float = Field(ge=0)
    abstained: bool = False
    abstention_reason: str | None = None

    @model_validator(mode="after")
    def validate_content_state(self) -> Self:
        orders = [block.order for block in self.ordered_blocks]
        if orders != list(range(len(self.ordered_blocks))):
            raise ValueError(
                "Page content block orders must be consecutive and sorted."
            )

        assembled_text = "\n".join(
            block.text for block in self.ordered_blocks
        )
        if self.abstained:
            if self.raw_text or self.normalized_text or self.ordered_blocks:
                raise ValueError("An abstained page cannot contain reader text.")
            if not (self.abstention_reason or "").strip():
                raise ValueError("An abstained page requires a reason.")
        else:
            if not self.raw_text.strip():
                raise ValueError("A non-abstained page requires text.")
            if self.raw_text != assembled_text:
                raise ValueError(
                    "raw_text must be assembled from ordered_blocks."
                )
            if not self.normalized_text.strip():
                raise ValueError("A non-abstained page requires normalized text.")
            if self.abstention_reason is not None:
                raise ValueError(
                    "A non-abstained page cannot have an abstention reason."
                )
        return self


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


class TransitionBaselineEvaluationReport(BaseModel):
    calibration_only: bool
    interval_duration_seconds: float = Field(gt=0)
    tolerance_seconds: float = Field(ge=0)
    prediction_count: int = Field(ge=0)
    predictions_by_type: dict[str, int] = Field(default_factory=dict)
    relaxed: TransitionDetectionMetrics
    typed: TransitionDetectionMetrics


class TransitionVariantRunResult(BaseModel):
    detector_variant: TransitionDetectorVariant
    detection_seconds: float = Field(ge=0)
    prediction_path: str = Field(min_length=1)
    evaluation_path: str = Field(min_length=1)
    evaluation: TransitionBaselineEvaluationReport


class TransitionComparisonReport(BaseModel):
    schema_version: str = "1.0"
    calibration_only: bool
    video_path: str = Field(min_length=1)
    config_path: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_name: str = Field(min_length=1)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    tolerance_seconds: float = Field(ge=0)
    feature_extraction_seconds: float = Field(ge=0)
    scene_score_count: int = Field(ge=0)
    sampled_frame_count: int = Field(ge=0)
    variants: dict[str, TransitionVariantRunResult] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds.")
        return self


class PageReadingMetrics(BaseModel):
    character_error_rate: float = Field(ge=0)
    word_error_rate: float = Field(ge=0)
    exact_match: bool
    technical_term_recall: float | None = Field(default=None, ge=0, le=1)


class PageReadingSampleResult(BaseModel):
    page_event_id: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    reader: PageReaderKind
    abstained: bool
    latency_seconds: float = Field(ge=0)
    metrics: PageReadingMetrics


class PageReadingAggregateMetrics(BaseModel):
    page_count: int = Field(ge=1)
    abstention_count: int = Field(ge=0)
    character_error_rate: float = Field(ge=0)
    word_error_rate: float = Field(ge=0)
    exact_match_count: int = Field(ge=0)
    exact_match_rate: float = Field(ge=0, le=1)
    technical_term_recall: float | None = Field(default=None, ge=0, le=1)
    mean_latency_seconds: float = Field(ge=0)
    p95_latency_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.abstention_count > self.page_count:
            raise ValueError("abstention_count cannot exceed page_count.")
        if self.exact_match_count > self.page_count:
            raise ValueError("exact_match_count cannot exceed page_count.")
        return self


class PageReaderRunResult(BaseModel):
    reader: PageReaderKind
    output_path: str = Field(min_length=1)
    evaluation_path: str = Field(min_length=1)
    aggregate: PageReadingAggregateMetrics
    pages: list[PageReadingSampleResult] = Field(min_length=1)


class PageReadingComparisonReport(BaseModel):
    schema_version: str = "1.0"
    calibration_only: bool
    lecture_id: str = Field(min_length=1)
    references_path: str = Field(min_length=1)
    references_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_asset_id: str | None = None
    readers: dict[str, PageReaderRunResult] = Field(min_length=1)


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
