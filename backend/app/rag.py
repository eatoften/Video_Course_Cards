from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from .knowledge_card import KnowledgeCardClaim


class RagRetrieveRequest(BaseModel):
    question: str = Field(min_length=1)
    course_id: str | None = None
    job_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        question = " ".join(value.strip().split())

        if not question:
            raise ValueError("Question is required.")

        return question

    @field_validator("course_id", "job_id")
    @classmethod
    def clean_optional_id(cls, value: str | None) -> str | None:
        if value is None:
            return None

        stripped = value.strip()

        return stripped or None


class RetrievedCard(BaseModel):
    card_id: str
    job_id: str
    title: str
    summary: str
    score: float = Field(ge=-1.0, le=1.0)
    source_start_seconds: float = Field(ge=0)
    source_end_seconds: float = Field(ge=0)
    key_points: list[str] = Field(default_factory=list)
    claims: list[KnowledgeCardClaim] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_source_range(self) -> "RetrievedCard":
        if self.source_end_seconds <= self.source_start_seconds:
            raise ValueError(
                "Card source end must be greater than start."
            )

        return self


class RagRetrieveResponse(BaseModel):
    question: str
    results: list[RetrievedCard] = Field(default_factory=list)
