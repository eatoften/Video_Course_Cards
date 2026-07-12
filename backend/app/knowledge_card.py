from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from .job import utc_now


from .review_item import ReviewItem, ReviewItemCreate


KnowledgeCardKind = Literal[
    "concept",
    "definition",
    "process",
    "comparison",
    "example",
    "formula",
]
KnowledgeCardContentStatus = Literal["draft", "reviewed", "needs_fix"]


class KnowledgeCardEvidence(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    quote: str = Field(min_length=1)
    segment_start_seconds: float = Field(ge=0)
    segment_end_seconds: float = Field(ge=0)


class KnowledgeCardClaim(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    text: str = Field(min_length=1)
    evidence: list[KnowledgeCardEvidence] = Field(min_length=1)


class KnowledgeCardBase(BaseModel):
    card_kind: KnowledgeCardKind = "concept"
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    key_points: list[str] = Field(default_factory=list)
    claims: list[KnowledgeCardClaim] = Field(min_length=1)
    unsupported_terms: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    content_status: KnowledgeCardContentStatus = "draft"
    source_start_seconds: float = Field(ge=0)
    source_end_seconds: float = Field(ge=0)
    provider: str | None = None
    model: str | None = None


class KnowledgeCardCreate(KnowledgeCardBase):
    review_items: list[ReviewItemCreate] = Field(default_factory=list)


class KnowledgeCardUpdate(BaseModel):
    card_kind: KnowledgeCardKind | None = None
    title: str | None = Field(default=None, min_length=1)
    summary: str | None = Field(default=None, min_length=1)
    key_points: list[str] | None = None
    claims: list[KnowledgeCardClaim] | None = Field(default=None, min_length=1)
    unsupported_terms: list[str] | None = None
    tags: list[str] | None = None
    content_status: KnowledgeCardContentStatus | None = None
    source_start_seconds: float | None = Field(default=None, ge=0)
    source_end_seconds: float | None = Field(default=None, ge=0)
    provider: str | None = None
    model: str | None = None


class KnowledgeCard(KnowledgeCardBase):
    id: str
    job_id: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class KnowledgeCardDetail(KnowledgeCard):
    review_items: list[ReviewItem] = Field(default_factory=list)


class KnowledgeCardIndexItem(BaseModel):
    id: str
    job_id: str
    title: str
    summary: str
    card_kind: KnowledgeCardKind
    tags: list[str] = Field(default_factory=list)
    content_status: KnowledgeCardContentStatus
    review_item_count: int = 0
    source_video: str | None = None
    source_start_seconds: float
    source_end_seconds: float
    note_count: int = 0
    learning_document_count: int = 0
    created_at: datetime
    updated_at: datetime
