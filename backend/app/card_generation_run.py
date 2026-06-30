from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .job import utc_now
from .transcript_chunk import TranscriptChunkGenerationRequest


CardGenerationRunStatus = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "canceled",
]
CardGenerationRunMode = Literal["auto"]


class AutoCardGenerationRequest(BaseModel):
    model: str | None = Field(default=None, max_length=200)
    focus: str | None = Field(default=None, max_length=500)
    card_count_per_chunk: int = Field(default=2, ge=1, le=6)
    regenerate_chunks: bool = False
    max_chunks: int | None = Field(default=None, ge=1, le=300)
    chunking: TranscriptChunkGenerationRequest = Field(
        default_factory=TranscriptChunkGenerationRequest
    )


class CardGenerationRunError(BaseModel):
    chunk_id: str | None = None
    chunk_index: int | None = None
    message: str


class CardGenerationRun(BaseModel):
    id: str
    job_id: str
    mode: CardGenerationRunMode = "auto"
    status: CardGenerationRunStatus = "pending"
    model: str | None = None
    card_count_per_chunk: int = 2
    total_chunks: int = 0
    completed_chunks: int = 0
    succeeded_chunks: int = 0
    failed_chunks: int = 0
    cards_created: int = 0
    error_message: str | None = None
    errors: list[CardGenerationRunError] = Field(default_factory=list)
    request: AutoCardGenerationRequest = Field(
        default_factory=AutoCardGenerationRequest
    )
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
