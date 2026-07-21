from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


RagQuestionCategory = Literal[
    "factual",
    "concept",
    "comparison",
    "multi_hop",
    "unanswerable",
]
RagBenchmarkSplit = Literal["development", "test"]
RagReviewStatus = Literal["pending", "accepted", "rejected"]
RagClaimSupport = Literal["supported", "partial", "unsupported"]
RagAuthoringMethod = Literal[
    "manual",
    "model_assisted",
    "deterministic_template",
]
RetrievalSystemName = Literal[
    "bm25",
    "dense",
    "hybrid_rrf",
    "dense_graph_noisy",
    "dense_graph_trusted",
]


def utc_now() -> datetime:
    return datetime.now(UTC)


class RagCorpusEvidence(BaseModel):
    evidence_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> "RagCorpusEvidence":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Evidence end must be greater than start.")
        return self


class RagCorpusClaim(BaseModel):
    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    evidence: list[RagCorpusEvidence] = Field(min_length=1)


class RagCorpusCard(BaseModel):
    card_id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)
    lecture_name: str = Field(min_length=1)
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    document_text: str = Field(min_length=1)
    content_status: str = Field(min_length=1)
    source_start_seconds: float = Field(ge=0)
    source_end_seconds: float = Field(gt=0)
    claims: list[RagCorpusClaim] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_source_range(self) -> "RagCorpusCard":
        if self.source_end_seconds <= self.source_start_seconds:
            raise ValueError("Card source end must be greater than start.")
        return self


class RagCorpusRelation(BaseModel):
    relation_id: str = Field(min_length=1)
    source_card_id: str = Field(min_length=1)
    target_card_id: str = Field(min_length=1)
    relation_type: str = Field(min_length=1)
    score: float = Field(ge=-1.0, le=1.0)
    method: str = Field(min_length=1)
    status: str = Field(min_length=1)
    explanation: str | None = None

    @model_validator(mode="after")
    def validate_distinct_cards(self) -> "RagCorpusRelation":
        if self.source_card_id == self.target_card_id:
            raise ValueError("A relation cannot be a self edge.")
        return self


class RagCorpusSnapshot(BaseModel):
    schema_version: str = "1.0"
    snapshot_id: str = Field(min_length=1)
    course_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    source_database_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cards: list[RagCorpusCard] = Field(min_length=1)
    relations: list[RagCorpusRelation] = Field(default_factory=list)


class RagEvidenceReference(BaseModel):
    card_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> "RagEvidenceReference":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Evidence end must be greater than start.")
        return self


class RagBenchmarkItem(BaseModel):
    question_id: str = Field(min_length=1)
    category: RagQuestionCategory
    split: RagBenchmarkSplit
    question: str = Field(min_length=1)
    answerable: bool
    reference_answer: str | None = None
    gold_card_ids: list[str] = Field(default_factory=list)
    gold_claim_ids: list[str] = Field(default_factory=list)
    evidence: list[RagEvidenceReference] = Field(default_factory=list)
    graph_path_card_ids: list[str] = Field(default_factory=list)
    authoring_method: RagAuthoringMethod
    review_status: RagReviewStatus = "pending"
    review_notes: str | None = None

    @model_validator(mode="after")
    def validate_ground_truth_shape(self) -> "RagBenchmarkItem":
        self.question = " ".join(self.question.strip().split())
        if not self.question:
            raise ValueError("Question cannot be blank.")
        if len(self.gold_card_ids) != len(set(self.gold_card_ids)):
            raise ValueError("Gold card ids must be unique.")
        if len(self.gold_claim_ids) != len(set(self.gold_claim_ids)):
            raise ValueError("Gold claim ids must be unique.")

        if self.answerable:
            if not self.reference_answer or not self.reference_answer.strip():
                raise ValueError("Answerable questions need a reference answer.")
            if not self.gold_card_ids or not self.gold_claim_ids or not self.evidence:
                raise ValueError("Answerable questions need card, claim, and evidence labels.")
        elif any(
            [
                self.reference_answer,
                self.gold_card_ids,
                self.gold_claim_ids,
                self.evidence,
                self.graph_path_card_ids,
            ]
        ):
            raise ValueError("Unanswerable questions cannot carry positive labels.")

        if self.category in {"factual", "concept"} and self.answerable:
            if len(self.gold_card_ids) != 1:
                raise ValueError("Single-card categories require exactly one gold card.")
        if self.category in {"comparison", "multi_hop"}:
            if not self.answerable or len(self.gold_card_ids) < 2:
                raise ValueError("Comparison and multi-hop questions require at least two cards.")
        if self.category == "multi_hop":
            if len(self.graph_path_card_ids) < 2:
                raise ValueError("Multi-hop questions require an explicit graph path.")
            if not set(self.graph_path_card_ids).issubset(self.gold_card_ids):
                raise ValueError("Every graph-path card must be a gold card.")
        if self.category == "unanswerable" and self.answerable:
            raise ValueError("The unanswerable category must be labeled unanswerable.")
        return self


class RagBenchmarkDataset(BaseModel):
    schema_version: str = "1.0"
    benchmark_id: str = Field(min_length=1)
    course_id: str = Field(min_length=1)
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime = Field(default_factory=utc_now)
    annotation_method: str = Field(min_length=1)
    confirmatory_status: Literal[
        "development",
        "pending_human_review",
        "sealed",
        "opened",
    ] = "development"
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    items: list[RagBenchmarkItem] = Field(min_length=1)


class RagGraphDecision(BaseModel):
    source_card_id: str = Field(min_length=1)
    target_card_id: str = Field(min_length=1)
    accepted: bool
    relation_type: Literal[
        "prerequisite",
        "related",
        "example_of",
        "contrast_with",
        "part_of",
    ] | None = None
    reviewer_id: str = Field(min_length=1)
    review_notes: str = Field(min_length=1)
    review_method: RagAuthoringMethod = "manual"

    @model_validator(mode="after")
    def validate_decision(self) -> "RagGraphDecision":
        if self.source_card_id == self.target_card_id:
            raise ValueError("A graph decision cannot describe a self edge.")
        if self.accepted and self.relation_type is None:
            raise ValueError("Accepted graph edges need a semantic type.")
        if not self.accepted and self.relation_type is not None:
            raise ValueError("Rejected graph edges cannot carry a semantic type.")
        return self


class RagClaimDecision(BaseModel):
    card_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    support: RagClaimSupport
    reviewer_id: str = Field(min_length=1)
    review_method: RagAuthoringMethod = "manual"
    review_notes: str = Field(min_length=1)


class RagAnnotationReview(BaseModel):
    schema_version: str = "1.0"
    review_id: str = Field(min_length=1)
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime = Field(default_factory=utc_now)
    review_status: Literal["candidate", "human_verified"] = "candidate"
    claim_decisions: list[RagClaimDecision] = Field(default_factory=list)
    graph_decisions: list[RagGraphDecision] = Field(default_factory=list)
    review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RagSingleQuestionSeed(BaseModel):
    claim_id: str = Field(min_length=1)
    split: RagBenchmarkSplit
    review_notes: str = Field(min_length=1)


class RagPairQuestionSeed(BaseModel):
    source_claim_id: str = Field(min_length=1)
    target_claim_id: str = Field(min_length=1)
    relation_type: Literal[
        "prerequisite",
        "related",
        "example_of",
        "contrast_with",
        "part_of",
    ]
    split: RagBenchmarkSplit
    review_notes: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_distinct_claims(self) -> "RagPairQuestionSeed":
        if self.source_claim_id == self.target_claim_id:
            raise ValueError("A pair seed requires two distinct claims.")
        return self


class RagUnanswerableQuestionSeed(BaseModel):
    question: str = Field(min_length=1)
    split: RagBenchmarkSplit
    review_notes: str = Field(min_length=1)


class RagBenchmarkSeed(BaseModel):
    schema_version: str = "1.0"
    seed_id: str = Field(min_length=1)
    course_id: str = Field(min_length=1)
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    single_questions: list[RagSingleQuestionSeed] = Field(min_length=1)
    paired_questions: list[RagPairQuestionSeed] = Field(min_length=1)
    unanswerable_questions: list[RagUnanswerableQuestionSeed] = Field(min_length=1)


class RagExperimentProtocol(BaseModel):
    schema_version: str = "1.0"
    protocol_id: str = Field(min_length=1)
    corpus_path: str = Field(min_length=1)
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_path: str = Field(min_length=1)
    benchmark_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trusted_graph_path: str | None = None
    trusted_graph_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    embedding_model: str = Field(min_length=1)
    top_k_values: list[int] = Field(default_factory=lambda: [1, 3, 5])
    bm25_k1: float = Field(default=1.2, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    rrf_k: int = Field(default=60, ge=1)
    graph_seed_k: int = Field(default=2, ge=1)
    graph_weight: float = Field(default=0.35, ge=0)
    graph_hops: int = Field(default=1, ge=1, le=1)
    confidence_score_policy: Literal[
        "system_top_score",
        "dense_anchor_for_graph",
    ] = "dense_anchor_for_graph"
    bootstrap_iterations: int = Field(default=5000, ge=100)
    bootstrap_seed: int = 20260721

    @model_validator(mode="after")
    def validate_protocol(self) -> "RagExperimentProtocol":
        if sorted(set(self.top_k_values)) != self.top_k_values:
            raise ValueError("top_k_values must be sorted and unique.")
        if not self.top_k_values or self.top_k_values[0] < 1:
            raise ValueError("top_k_values must contain positive values.")
        if bool(self.trusted_graph_path) != bool(self.trusted_graph_sha256):
            raise ValueError("Trusted graph path and hash must be provided together.")
        return self


class RankedCard(BaseModel):
    card_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    score: float
    retrieval_source: str = Field(min_length=1)


class RetrievalRecord(BaseModel):
    question_id: str = Field(min_length=1)
    category: RagQuestionCategory
    split: RagBenchmarkSplit
    system: RetrievalSystemName
    elapsed_milliseconds: float = Field(ge=0)
    query_encoding_milliseconds: float = Field(default=0, ge=0)
    ranking_milliseconds: float = Field(default=0, ge=0)
    confidence_score: float | None = None
    ranked_cards: list[RankedCard] = Field(default_factory=list)


class RagEmbeddingRecord(BaseModel):
    card_id: str = Field(min_length=1)
    vector: list[float] = Field(min_length=1)


class RagEmbeddingSnapshot(BaseModel):
    schema_version: str = "1.0"
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str = Field(min_length=1)
    dimension: int = Field(ge=1)
    normalized: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    indexing_milliseconds: float = Field(ge=0)
    records: list[RagEmbeddingRecord] = Field(min_length=1)
    embeddings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RetrievalMetricSlice(BaseModel):
    question_count: int = Field(ge=0)
    answerable_count: int = Field(ge=0)
    unanswerable_count: int = Field(ge=0)
    hit_rate_at_k: dict[int, float]
    set_recall_at_k: dict[int, float]
    joint_recall_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    mean_reciprocal_rank: float = Field(ge=0, le=1)
    unanswerable_false_retrieval_rate: float = Field(ge=0, le=1)
    answerability_f1: float = Field(ge=0, le=1)
    confidence_threshold: float
    mean_latency_milliseconds: float = Field(ge=0)
    median_latency_milliseconds: float = Field(ge=0)
    p95_latency_milliseconds: float = Field(ge=0)


class RetrievalSystemReport(BaseModel):
    system: RetrievalSystemName
    split: RagBenchmarkSplit
    overall: RetrievalMetricSlice
    by_category: dict[RagQuestionCategory, RetrievalMetricSlice]


class RagGroundedCitation(BaseModel):
    card_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)


class RagGeneratedClaim(BaseModel):
    text: str = Field(min_length=1)
    citations: list[RagGroundedCitation] = Field(min_length=1)


class RagGroundedAnswerPayload(BaseModel):
    answerable: bool
    answer: str = Field(min_length=1)
    claims: list[RagGeneratedClaim] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_answer_shape(self) -> "RagGroundedAnswerPayload":
        if self.answerable and not self.claims:
            raise ValueError("An answerable response needs at least one cited claim.")
        if not self.answerable and self.claims:
            raise ValueError("An abstention cannot include generated claims.")
        return self


class RagGroundedAnswerRecord(BaseModel):
    question_id: str = Field(min_length=1)
    category: RagQuestionCategory
    split: RagBenchmarkSplit
    system: RetrievalSystemName
    context_card_ids: list[str] = Field(default_factory=list)
    context_characters: int = Field(ge=0)
    answerable_prediction: bool | None = None
    answer: str | None = None
    claims: list[RagGeneratedClaim] = Field(default_factory=list)
    latency_milliseconds: float = Field(ge=0)
    generation_error: str | None = None
    raw_output: str | None = None

    @model_validator(mode="after")
    def validate_generation_result(self) -> "RagGroundedAnswerRecord":
        if self.generation_error is None:
            if self.answerable_prediction is None or self.answer is None:
                raise ValueError("Successful generation needs a prediction and answer.")
        elif self.answerable_prediction is not None or self.answer is not None:
            raise ValueError("Failed generation cannot include a parsed answer.")
        return self


class RagAnswerExperimentProtocol(BaseModel):
    schema_version: str = "1.0"
    protocol_id: str = Field(min_length=1)
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieval_report_path: str = Field(min_length=1)
    retrieval_report_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: RagBenchmarkSplit = "development"
    systems: list[RetrievalSystemName] = Field(min_length=1)
    model: str = Field(min_length=1)
    model_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=500, ge=1)
    top_k: int = Field(default=5, ge=1)
    context_budget_characters: int = Field(default=6000, ge=500)
    prompt_version: str = Field(min_length=1)
    semantic_evaluation_model: str = Field(min_length=1)
    confidence_thresholds: dict[RetrievalSystemName, float] = Field(
        default_factory=dict
    )
