from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .job import utc_now


KnowledgeCardDifficulty = Literal["easy", "medium", "hard"]


class KnowledgeCardEvidence(BaseModel):
    quote: str = Field(min_length=1)
    segment_start_seconds: float = Field(ge=0)
    segment_end_seconds: float = Field(ge=0)


class KnowledgeCardClaim(BaseModel):
    text: str = Field(min_length=1)
    evidence: list[KnowledgeCardEvidence] = Field(min_length=1)


class KnowledgeCardBase(BaseModel):
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    key_points: list[str] = Field(default_factory=list)
    claims: list[KnowledgeCardClaim] = Field(min_length=1)
    unsupported_terms: list[str] = Field(default_factory=list)
    question: str | None = None
    answer: str | None = None
    difficulty: KnowledgeCardDifficulty = "medium"
    source_start_seconds: float = Field(ge=0)
    source_end_seconds: float = Field(ge=0)
    provider: str | None = None
    model: str | None = None


class KnowledgeCardCreate(KnowledgeCardBase):
    pass


class KnowledgeCardUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1)
    summary: str | None = Field(default=None, min_length=1)
    key_points: list[str] | None = None
    claims: list[KnowledgeCardClaim] | None = Field(default=None, min_length=1)
    unsupported_terms: list[str] | None = None
    question: str | None = None
    answer: str | None = None
    difficulty: KnowledgeCardDifficulty | None = None
    source_start_seconds: float | None = Field(default=None, ge=0)
    source_end_seconds: float | None = Field(default=None, ge=0)
    provider: str | None = None
    model: str | None = None


class KnowledgeCard(KnowledgeCardBase):
    id: str
    job_id: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
