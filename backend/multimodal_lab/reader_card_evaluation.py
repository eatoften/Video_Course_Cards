from __future__ import annotations

import hashlib
import random
import statistics
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from .annotation_io import load_jsonl, write_jsonl
from .ctc_text import normalize_line_text
from .metrics import evaluate_card_conversion, levenshtein_distance
from .reader_card_cascade import (
    ReaderCardCascadeProtocol,
    ReaderCardGenerationRecord,
    ReaderCardGenerationStatus,
    ReaderCardSystemName,
)
from .schemas import (
    CardConversionMetrics,
    CardEvaluationCounts,
    StablePageReference,
)


class ReaderCardCitationReview(BaseModel):
    card_index: int = Field(ge=0)
    claim_index: int = Field(ge=0)
    evidence_index: int = Field(ge=0)
    quote: str = Field(min_length=1)
    best_gold_line: str = Field(min_length=1)
    source_similarity: float = Field(ge=0, le=1)
    automatic_match: bool
    correct_against_source: bool | None = None
    notes: str | None = None


class ReaderCardClaimReview(BaseModel):
    card_index: int = Field(ge=0)
    claim_index: int = Field(ge=0)
    claim_text: str = Field(min_length=1)
    supported_by_source: bool | None = None
    citations: list[ReaderCardCitationReview] = Field(min_length=1)
    notes: str | None = None


class ReaderCardReviewRecord(BaseModel):
    schema_version: str = "1.0"
    protocol_id: str = Field(min_length=1)
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    system_name: ReaderCardSystemName
    page_event_id: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    gold_concepts: list[str] = Field(min_length=1)
    gold_text: str = Field(min_length=1)
    gold_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_status: ReaderCardGenerationStatus
    generated_card_count: int = Field(ge=0, le=1)
    concept_recovered: bool | None = None
    claim_reviews: list[ReaderCardClaimReview] = Field(default_factory=list)
    accepted_without_edit: bool | None = None
    usable_card: bool | None = None
    completed: bool = False
    reviewer_id: str | None = None
    review_method: str | None = None
    reviewed_at: datetime | None = None
    notes: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.system_name, self.page_event_id

    @model_validator(mode="after")
    def validate_review(self) -> Self:
        decisions = (
            self.concept_recovered,
            self.accepted_without_edit,
            self.usable_card,
        )
        if self.completed:
            if any(value is None for value in decisions):
                raise ValueError("A completed review requires all card decisions.")
            if any(
                claim.supported_by_source is None
                or any(
                    citation.correct_against_source is None
                    for citation in claim.citations
                )
                for claim in self.claim_reviews
            ):
                raise ValueError("A completed review requires all claim decisions.")
            if not self.reviewer_id or not self.review_method or self.reviewed_at is None:
                raise ValueError("A completed review requires reviewer metadata.")
        if self.generated_card_count == 0:
            if self.claim_reviews:
                raise ValueError("A missing card cannot have claim reviews.")
            if any(value is True for value in decisions):
                raise ValueError("A missing card cannot pass downstream review.")
        if self.usable_card is True:
            citations = [
                citation
                for claim in self.claim_reviews
                for citation in claim.citations
            ]
            if (
                self.generated_card_count != 1
                or self.concept_recovered is not True
                or not self.claim_reviews
                or any(
                    claim.supported_by_source is not True
                    for claim in self.claim_reviews
                )
                or any(
                    citation.correct_against_source is not True
                    for citation in citations
                )
            ):
                raise ValueError(
                    "A usable card must recover the concept and be fully grounded."
                )
        return self


class ReaderCardManualDecision(BaseModel):
    system_name: ReaderCardSystemName
    page_event_id: str = Field(min_length=1)
    concept_recovered: bool
    supported_claims: list[bool]
    correct_citations: list[list[bool]]
    accepted_without_edit: bool
    usable_card: bool
    notes: str = Field(min_length=1)

    @property
    def key(self) -> tuple[str, str]:
        return self.system_name, self.page_event_id


class ReaderCardDecisionBundle(BaseModel):
    schema_version: str = "1.0"
    protocol_id: str = Field(min_length=1)
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_id: str = Field(min_length=1)
    review_method: str = Field(min_length=1)
    decisions: list[ReaderCardManualDecision] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_decisions(self) -> Self:
        keys = [decision.key for decision in self.decisions]
        if len(set(keys)) != len(keys):
            raise ValueError("Manual card decisions contain duplicate keys.")
        return self


class ReaderCardSystemEvaluation(BaseModel):
    system_name: ReaderCardSystemName
    page_count: int = Field(ge=1)
    successful_generation_count: int = Field(ge=0)
    failed_generation_count: int = Field(ge=0)
    mean_generation_seconds: float = Field(ge=0)
    counts: CardEvaluationCounts
    metrics: CardConversionMetrics


class PairedBinaryBootstrapDifference(BaseModel):
    system_a: ReaderCardSystemName
    system_b: ReaderCardSystemName
    metric: Literal["concept_recovery", "usable_card_conversion"]
    point_difference_a_minus_b: float
    confidence_level: float = Field(gt=0, lt=1)
    lower_bound: float
    upper_bound: float
    iterations: int = Field(ge=1000)
    seed: int = Field(ge=0)
    probability_a_higher_than_b: float = Field(ge=0, le=1)


class ReaderCardCascadeEvaluationReport(BaseModel):
    schema_version: str = "1.0"
    protocol_id: str = Field(min_length=1)
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str = Field(min_length=1)
    model_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_count: int = Field(ge=1)
    systems: dict[str, ReaderCardSystemEvaluation]
    paired_bootstrap: list[PairedBinaryBootstrapDifference]
    reviewer_id: str = Field(min_length=1)
    review_method: str = Field(min_length=1)
    limitations: list[str] = Field(min_length=1)


def build_reader_card_review_template(
    records: Sequence[ReaderCardGenerationRecord],
    references: Sequence[StablePageReference],
    *,
    protocol: ReaderCardCascadeProtocol,
    protocol_sha256: str,
) -> list[ReaderCardReviewRecord]:
    references_by_id = {
        reference.page_event_id: reference for reference in references
    }
    reviews: list[ReaderCardReviewRecord] = []
    for record in records:
        reference = references_by_id.get(record.page_event_id)
        if reference is None:
            raise ValueError(f"Missing gold reference for {record.page_event_id}.")
        response_cards = record.response.cards if record.response is not None else []
        if len(response_cards) > 1:
            raise ValueError("The frozen cascade allows at most one card per page.")
        claim_reviews: list[ReaderCardClaimReview] = []
        for card_index, card in enumerate(response_cards):
            for claim_index, claim in enumerate(card.claims):
                citations = []
                for evidence_index, evidence in enumerate(claim.evidence):
                    best_line, similarity = best_source_line_match(
                        evidence.quote,
                        reference.gold_text.splitlines(),
                    )
                    citations.append(
                        ReaderCardCitationReview(
                            card_index=card_index,
                            claim_index=claim_index,
                            evidence_index=evidence_index,
                            quote=evidence.quote,
                            best_gold_line=best_line,
                            source_similarity=similarity,
                            automatic_match=(
                                similarity >= protocol.citation_similarity_threshold
                            ),
                        )
                    )
                claim_reviews.append(
                    ReaderCardClaimReview(
                        card_index=card_index,
                        claim_index=claim_index,
                        claim_text=claim.text,
                        citations=citations,
                    )
                )
        reviews.append(
            ReaderCardReviewRecord(
                protocol_id=protocol.protocol_id,
                protocol_sha256=protocol_sha256,
                system_name=record.system_name,
                page_event_id=record.page_event_id,
                page_number=record.page_number,
                gold_concepts=reference.gold_concepts,
                gold_text=reference.gold_text,
                gold_text_sha256=_text_sha256(reference.gold_text),
                generation_status=record.status,
                generated_card_count=len(response_cards),
                claim_reviews=claim_reviews,
            )
        )
    return reviews


def best_source_line_match(
    quote: str,
    gold_lines: Sequence[str],
) -> tuple[str, float]:
    normalized_quote = normalize_line_text(quote)
    candidates = [line for line in gold_lines if normalize_line_text(line)]
    if not normalized_quote or not candidates:
        return "<no source line>", 0.0
    scored = [
        (
            line,
            _edit_similarity(normalized_quote, normalize_line_text(line)),
        )
        for line in candidates
    ]
    return max(scored, key=lambda item: item[1])


def load_reader_card_reviews(
    path: str | Path,
) -> list[ReaderCardReviewRecord]:
    reviews = load_jsonl(path, ReaderCardReviewRecord)
    keys = [review.key for review in reviews]
    if len(set(keys)) != len(keys):
        raise ValueError("Card review records contain duplicate system/page keys.")
    return reviews


def write_reader_card_reviews(
    path: str | Path,
    reviews: Sequence[ReaderCardReviewRecord],
) -> None:
    write_jsonl(path, reviews)


def load_reader_card_decision_bundle(
    path: str | Path,
) -> ReaderCardDecisionBundle:
    return ReaderCardDecisionBundle.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def apply_reader_card_decisions(
    reviews: Sequence[ReaderCardReviewRecord],
    bundle: ReaderCardDecisionBundle,
) -> list[ReaderCardReviewRecord]:
    if not reviews:
        raise ValueError("A decision bundle requires a review template.")
    protocol_id = reviews[0].protocol_id
    protocol_sha256 = reviews[0].protocol_sha256
    if bundle.protocol_id != protocol_id:
        raise ValueError("Decision bundle protocol ID does not match the template.")
    if bundle.protocol_sha256 != protocol_sha256:
        raise ValueError("Decision bundle protocol hash does not match the template.")
    decisions_by_key = {decision.key: decision for decision in bundle.decisions}
    review_keys = {review.key for review in reviews}
    if set(decisions_by_key) != review_keys:
        raise ValueError("Decision bundle does not cover every review exactly once.")

    reviewed_at = datetime.now(UTC)
    completed: list[ReaderCardReviewRecord] = []
    for review in reviews:
        decision = decisions_by_key[review.key]
        if len(decision.supported_claims) != len(review.claim_reviews):
            raise ValueError(f"Claim decisions do not align for {review.key}.")
        if len(decision.correct_citations) != len(review.claim_reviews):
            raise ValueError(f"Citation decisions do not align for {review.key}.")
        claim_reviews = []
        for claim, supported, citation_decisions in zip(
            review.claim_reviews,
            decision.supported_claims,
            decision.correct_citations,
            strict=True,
        ):
            if len(citation_decisions) != len(claim.citations):
                raise ValueError(
                    f"Citation decisions do not align for {review.key}."
                )
            claim_reviews.append(
                claim.model_copy(
                    update={
                        "supported_by_source": supported,
                        "citations": [
                            citation.model_copy(
                                update={"correct_against_source": is_correct}
                            )
                            for citation, is_correct in zip(
                                claim.citations,
                                citation_decisions,
                                strict=True,
                            )
                        ],
                    }
                )
            )
        completed.append(
            ReaderCardReviewRecord.model_validate(
                review.model_copy(
                    update={
                        "concept_recovered": decision.concept_recovered,
                        "claim_reviews": claim_reviews,
                        "accepted_without_edit": (
                            decision.accepted_without_edit
                        ),
                        "usable_card": decision.usable_card,
                        "completed": True,
                        "reviewer_id": bundle.reviewer_id,
                        "review_method": bundle.review_method,
                        "reviewed_at": reviewed_at,
                        "notes": decision.notes,
                    }
                ).model_dump()
            )
        )
    return completed


def evaluate_reader_card_cascade(
    records: Sequence[ReaderCardGenerationRecord],
    reviews: Sequence[ReaderCardReviewRecord],
    references: Sequence[StablePageReference],
    *,
    protocol: ReaderCardCascadeProtocol,
    protocol_sha256: str,
    generation_records_sha256: str,
    review_records_sha256: str,
) -> ReaderCardCascadeEvaluationReport:
    validate_reviews_against_generation(
        records,
        reviews,
        references,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )
    reference_ids = [reference.page_event_id for reference in references]
    reviewer_ids = {review.reviewer_id for review in reviews}
    review_methods = {review.review_method for review in reviews}
    if len(reviewer_ids) != 1 or None in reviewer_ids:
        raise ValueError("The frozen study requires one identified source auditor.")
    if len(review_methods) != 1 or None in review_methods:
        raise ValueError("The frozen study requires one review method.")
    records_by_key = {record.key: record for record in records}
    reviews_by_key = {review.key: review for review in reviews}
    systems: tuple[ReaderCardSystemName, ...] = (
        "cnn_v2",
        "vit_v1",
        "rapidocr_stored",
    )
    system_evaluations: dict[str, ReaderCardSystemEvaluation] = {}
    outcomes: dict[str, dict[str, list[int]]] = {}
    for system_name in systems:
        system_records = [
            records_by_key[(system_name, page_event_id)]
            for page_event_id in reference_ids
        ]
        system_reviews = [
            reviews_by_key[(system_name, page_event_id)]
            for page_event_id in reference_ids
        ]
        counts = _aggregate_card_counts(system_reviews)
        system_evaluations[system_name] = ReaderCardSystemEvaluation(
            system_name=system_name,
            page_count=len(reference_ids),
            successful_generation_count=sum(
                record.status is ReaderCardGenerationStatus.succeeded
                for record in system_records
            ),
            failed_generation_count=sum(
                record.status is ReaderCardGenerationStatus.failed
                for record in system_records
            ),
            mean_generation_seconds=statistics.fmean(
                record.elapsed_seconds for record in system_records
            ),
            counts=counts,
            metrics=evaluate_card_conversion(counts),
        )
        outcomes[system_name] = {
            "concept_recovery": [
                int(review.concept_recovered is True) for review in system_reviews
            ],
            "usable_card_conversion": [
                int(review.usable_card is True) for review in system_reviews
            ],
        }

    pairs: tuple[tuple[ReaderCardSystemName, ReaderCardSystemName], ...] = (
        ("cnn_v2", "vit_v1"),
        ("cnn_v2", "rapidocr_stored"),
        ("vit_v1", "rapidocr_stored"),
    )
    bootstraps = []
    seed_offset = 0
    for metric in ("concept_recovery", "usable_card_conversion"):
        for system_a, system_b in pairs:
            bootstraps.append(
                paired_binary_bootstrap_difference(
                    system_a,
                    outcomes[system_a][metric],
                    system_b,
                    outcomes[system_b][metric],
                    metric=metric,
                    seed=protocol.bootstrap_seed + seed_offset,
                    iterations=protocol.bootstrap_iterations,
                    confidence_level=protocol.confidence_level,
                )
            )
            seed_offset += 1

    return ReaderCardCascadeEvaluationReport(
        protocol_id=protocol.protocol_id,
        protocol_sha256=protocol_sha256,
        generation_records_sha256=generation_records_sha256,
        review_records_sha256=review_records_sha256,
        model=protocol.model,
        model_digest=protocol.model_digest,
        page_count=len(reference_ids),
        systems=system_evaluations,
        paired_bootstrap=bootstraps,
        reviewer_id=next(iter(reviewer_ids)),  # type: ignore[arg-type]
        review_method=next(iter(review_methods)),  # type: ignore[arg-type]
        limitations=[
            "One model-assisted source auditor reviewed all cards; a human second rater and inter-rater reliability are unavailable.",
            "The study isolates slide OCR text and does not include lecture audio.",
            "RapidOCR uses stored detector polygons, so missed page detections are excluded.",
            "Sixteen pages provide a small test sample and therefore wide uncertainty.",
            "Some official text references omit equations visible in the slide image; image inspection takes precedence in the source audit.",
        ],
    )


def validate_reviews_against_generation(
    records: Sequence[ReaderCardGenerationRecord],
    reviews: Sequence[ReaderCardReviewRecord],
    references: Sequence[StablePageReference],
    *,
    protocol: ReaderCardCascadeProtocol,
    protocol_sha256: str,
) -> None:
    records_by_key = {record.key: record for record in records}
    reviews_by_key = {review.key: review for review in reviews}
    expected_keys = {
        (system_name, reference.page_event_id)
        for system_name in ("cnn_v2", "vit_v1", "rapidocr_stored")
        for reference in references
    }
    if set(records_by_key) != expected_keys or set(reviews_by_key) != expected_keys:
        raise ValueError("Generation and review keys do not cover the frozen study.")
    references_by_id = {
        reference.page_event_id: reference for reference in references
    }
    for key in sorted(expected_keys):
        record = records_by_key[key]
        review = reviews_by_key[key]
        reference = references_by_id[record.page_event_id]
        if not review.completed:
            raise ValueError(f"Review is still pending for {key}.")
        if review.protocol_id != protocol.protocol_id:
            raise ValueError(f"Review protocol ID changed for {key}.")
        if review.protocol_sha256 != protocol_sha256:
            raise ValueError(f"Review protocol hash changed for {key}.")
        if review.gold_concepts != reference.gold_concepts:
            raise ValueError(f"Gold concepts changed for {key}.")
        if review.gold_text_sha256 != _text_sha256(reference.gold_text):
            raise ValueError(f"Gold source text changed for {key}.")
        cards = record.response.cards if record.response is not None else []
        if review.generated_card_count != len(cards):
            raise ValueError(f"Generated card count changed for {key}.")
        expected_claims = [
            (card_index, claim_index, claim)
            for card_index, card in enumerate(cards)
            for claim_index, claim in enumerate(card.claims)
        ]
        if len(review.claim_reviews) != len(expected_claims):
            raise ValueError(f"Claim review count changed for {key}.")
        for claim_review, (card_index, claim_index, claim) in zip(
            review.claim_reviews,
            expected_claims,
            strict=True,
        ):
            if (
                claim_review.card_index != card_index
                or claim_review.claim_index != claim_index
                or claim_review.claim_text != claim.text
            ):
                raise ValueError(f"Claim review no longer matches generation for {key}.")
            if len(claim_review.citations) != len(claim.evidence):
                raise ValueError(f"Citation review count changed for {key}.")
            for evidence_index, (citation, evidence) in enumerate(
                zip(claim_review.citations, claim.evidence, strict=True)
            ):
                if (
                    citation.card_index != card_index
                    or citation.claim_index != claim_index
                    or citation.evidence_index != evidence_index
                    or citation.quote != evidence.quote
                ):
                    raise ValueError(
                        f"Citation review no longer matches generation for {key}."
                    )


def paired_binary_bootstrap_difference(
    system_a: ReaderCardSystemName,
    outcomes_a: Sequence[int],
    system_b: ReaderCardSystemName,
    outcomes_b: Sequence[int],
    *,
    metric: Literal["concept_recovery", "usable_card_conversion"],
    seed: int,
    iterations: int,
    confidence_level: float,
) -> PairedBinaryBootstrapDifference:
    if len(outcomes_a) != len(outcomes_b) or not outcomes_a:
        raise ValueError("Paired outcomes must have equal, non-zero length.")
    if any(value not in (0, 1) for value in (*outcomes_a, *outcomes_b)):
        raise ValueError("Paired binary outcomes must contain only zero and one.")
    point_difference = statistics.fmean(outcomes_a) - statistics.fmean(outcomes_b)
    random_generator = random.Random(seed)
    differences = []
    for _ in range(iterations):
        indices = [
            random_generator.randrange(len(outcomes_a))
            for _ in range(len(outcomes_a))
        ]
        differences.append(
            statistics.fmean(outcomes_a[index] for index in indices)
            - statistics.fmean(outcomes_b[index] for index in indices)
        )
    differences.sort()
    alpha = 1 - confidence_level
    lower_index = max(0, int((alpha / 2) * iterations))
    upper_index = min(
        iterations - 1,
        int((1 - alpha / 2) * iterations) - 1,
    )
    return PairedBinaryBootstrapDifference(
        system_a=system_a,
        system_b=system_b,
        metric=metric,
        point_difference_a_minus_b=point_difference,
        confidence_level=confidence_level,
        lower_bound=differences[lower_index],
        upper_bound=differences[upper_index],
        iterations=iterations,
        seed=seed,
        probability_a_higher_than_b=(
            sum(difference > 0 for difference in differences) / iterations
        ),
    )


def _aggregate_card_counts(
    reviews: Sequence[ReaderCardReviewRecord],
) -> CardEvaluationCounts:
    claim_reviews = [claim for review in reviews for claim in review.claim_reviews]
    citations = [
        citation for claim in claim_reviews for citation in claim.citations
    ]
    return CardEvaluationCounts(
        supported_claims=sum(
            claim.supported_by_source is True for claim in claim_reviews
        ),
        total_claims=len(claim_reviews),
        recovered_concepts=sum(
            review.concept_recovered is True for review in reviews
        ),
        gold_concepts=sum(len(review.gold_concepts) for review in reviews),
        correct_citations=sum(
            citation.correct_against_source is True for citation in citations
        ),
        total_citations=len(citations),
        claims_with_valid_citation=sum(
            any(
                citation.correct_against_source is True
                for citation in claim.citations
            )
            for claim in claim_reviews
        ),
        claims_requiring_citation=len(claim_reviews),
        accepted_without_edit=sum(
            review.accepted_without_edit is True for review in reviews
        ),
        generated_cards=sum(review.generated_card_count for review in reviews),
        usable_cards=sum(review.usable_card is True for review in reviews),
        eligible_concepts=sum(len(review.gold_concepts) for review in reviews),
    )


def _edit_similarity(left: str, right: str) -> float:
    denominator = max(len(left), len(right))
    if denominator == 0:
        return 1.0
    return 1 - (levenshtein_distance(left, right) / denominator)


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
