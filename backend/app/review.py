from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .job import utc_now
from .knowledge_card import KnowledgeCardClaim
from .review_item import ReviewItem


ReviewRating = Literal["again", "hard", "good", "easy"]
ReviewPhase = Literal["new", "learning", "review", "relearning"]


class ReviewProgress(BaseModel):
    review_item_id: str
    fsrs_card_id: int
    fsrs_state: int = Field(ge=1, le=3)
    step: int | None = None
    due_at: datetime = Field(default_factory=utc_now)
    stability: float | None = None
    fsrs_difficulty: float | None = None
    last_reviewed_at: datetime | None = None
    review_count: int = Field(default=0, ge=0)
    lapse_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def phase(self) -> ReviewPhase:
        if self.review_count == 0:
            return "new"
        return {1: "learning", 2: "review", 3: "relearning"}[self.fsrs_state]


class ReviewEvent(BaseModel):
    id: str
    review_item_id: str
    rating: ReviewRating
    reviewed_at: datetime
    response_time_ms: int | None = Field(default=None, ge=0)
    previous_phase: ReviewPhase
    next_phase: ReviewPhase
    due_before: datetime
    due_after: datetime
    scheduled_days: float = Field(ge=0)


class ReviewQueueItem(BaseModel):
    review_item: ReviewItem
    progress: ReviewProgress
    phase: ReviewPhase
    card_id: str
    card_title: str
    card_summary: str
    card_kind: str
    claims: list[KnowledgeCardClaim] = Field(default_factory=list)
    topic_id: str | None = None
    topic_title: str | None = None
    source_start_seconds: float = Field(ge=0)
    source_end_seconds: float = Field(ge=0)


class ReviewQueue(BaseModel):
    course_id: str
    topic_id: str | None = None
    due_count: int
    new_count: int
    learning_count: int
    review_count: int
    relearning_count: int
    items: list[ReviewQueueItem] = Field(default_factory=list)


class ReviewRatingRequest(BaseModel):
    rating: ReviewRating
    response_time_ms: int | None = Field(default=None, ge=0, le=3_600_000)


class ReviewRatingResult(BaseModel):
    progress: ReviewProgress
    event: ReviewEvent
