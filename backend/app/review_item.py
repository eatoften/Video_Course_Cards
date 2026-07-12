from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .job import utc_now


ReviewItemType = Literal["basic", "cloze", "explain", "compare", "apply"]
ReviewItemSource = Literal["generated", "manual", "local_llm"]
ReviewItemStatus = Literal["active", "disabled"]


class ReviewItemBase(BaseModel):
    item_type: ReviewItemType = "basic"
    prompt: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)
    source_claim_ids: list[str] = Field(default_factory=list)
    source: ReviewItemSource = "manual"
    status: ReviewItemStatus = "active"


class ReviewItemCreate(ReviewItemBase):
    pass


class ReviewItemUpdate(BaseModel):
    item_type: ReviewItemType | None = None
    prompt: str | None = Field(default=None, min_length=1)
    expected_answer: str | None = Field(default=None, min_length=1)
    source_claim_ids: list[str] | None = None
    status: ReviewItemStatus | None = None


class ReviewItem(ReviewItemBase):
    id: str
    card_id: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
