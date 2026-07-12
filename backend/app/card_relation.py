from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .job import utc_now
from .knowledge_card import KnowledgeCardReviewState


CardRelationType = Literal[
    "semantic_similarity",
    "prerequisite",
    "related",
    "example_of",
    "contrast_with",
    "part_of",
]
CardRelationMethod = Literal[
    "cosine_similarity",
    "local_llm",
    "manual",
]
CardRelationStatus = Literal[
    "suggested",
    "accepted",
    "rejected",
    "hidden",
]
CardRelationSemanticType = Literal[
    "prerequisite",
    "related",
    "example_of",
    "contrast_with",
    "part_of",
]
CardRelationClassification = Literal[
    "prerequisite",
    "related",
    "example_of",
    "contrast_with",
    "part_of",
    "unclear",
]

DEFAULT_CARD_RELATION_TYPE: CardRelationType = "semantic_similarity"
DEFAULT_CARD_RELATION_METHOD: CardRelationMethod = "cosine_similarity"
DEFAULT_CARD_RELATION_STATUS: CardRelationStatus = "suggested"
DEFAULT_CARD_RELATION_THRESHOLD = 0.72
DEFAULT_CARD_RELATION_TOP_K = 5


class CardRelation(BaseModel):
    id: str
    course_id: str
    source_card_id: str
    target_card_id: str
    relation_type: CardRelationType = DEFAULT_CARD_RELATION_TYPE
    score: float = Field(ge=-1.0, le=1.0)
    method: CardRelationMethod = DEFAULT_CARD_RELATION_METHOD
    model: str | None = None
    explanation: str | None = None
    status: CardRelationStatus = DEFAULT_CARD_RELATION_STATUS
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_distinct_cards(self) -> "CardRelation":
        if self.source_card_id == self.target_card_id:
            raise ValueError("Card relation cannot point to itself.")

        return self


class CardRelationUpdate(BaseModel):
    relation_type: CardRelationType | None = None
    explanation: str | None = None
    status: CardRelationStatus | None = None


class CardRelationCreate(BaseModel):
    source_card_id: str = Field(min_length=1)
    target_card_id: str = Field(min_length=1)
    relation_type: CardRelationSemanticType
    explanation: str | None = Field(default=None, max_length=2000)
    status: CardRelationStatus = "accepted"

    @model_validator(mode="after")
    def validate_distinct_cards(self) -> "CardRelationCreate":
        if self.source_card_id == self.target_card_id:
            raise ValueError("Card relation cannot point to itself.")

        return self


class CardRelationClassifyRequest(BaseModel):
    model: str | None = Field(default=None, max_length=200)


class CardRelationClassificationResult(BaseModel):
    source_relation_id: str
    classification: CardRelationClassification
    explanation: str
    model: str
    relation: CardRelation | None = None


class CardRelationRecomputeRequest(BaseModel):
    threshold: float = Field(
        default=DEFAULT_CARD_RELATION_THRESHOLD,
        ge=-1.0,
        le=1.0,
    )
    top_k: int = Field(default=DEFAULT_CARD_RELATION_TOP_K, ge=1, le=50)


class CardRelationRecomputeResult(BaseModel):
    course_id: str
    total_cards: int
    embedded_cards: int
    skipped_cards: int
    relations_written: int
    threshold: float
    top_k: int


class RelatedCard(BaseModel):
    relation_id: str
    card_id: str
    job_id: str
    title: str
    summary: str
    tags: list[str] = Field(default_factory=list)
    review_state: KnowledgeCardReviewState
    relation_type: CardRelationType
    score: float = Field(ge=-1.0, le=1.0)
    method: CardRelationMethod
    status: CardRelationStatus
    explanation: str | None = None
    source_start_seconds: float = Field(ge=0)
    source_end_seconds: float = Field(ge=0)


class CardRelatedCardsResponse(BaseModel):
    card_id: str
    related: list[RelatedCard] = Field(default_factory=list)


class CardRelationGraphNode(BaseModel):
    id: str
    job_id: str
    title: str
    summary: str
    tags: list[str] = Field(default_factory=list)
    review_state: KnowledgeCardReviewState
    source_start_seconds: float = Field(ge=0)
    source_end_seconds: float = Field(ge=0)


class CardRelationGraphEdge(BaseModel):
    id: str
    source: str
    target: str
    relation_type: CardRelationType
    score: float = Field(ge=-1.0, le=1.0)
    method: CardRelationMethod
    status: CardRelationStatus
    explanation: str | None = None


class CourseCardRelationsGraph(BaseModel):
    course_id: str
    nodes: list[CardRelationGraphNode] = Field(default_factory=list)
    edges: list[CardRelationGraphEdge] = Field(default_factory=list)
