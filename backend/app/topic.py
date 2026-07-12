from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .job import utc_now
from .knowledge_card import KnowledgeCardIndexItem
from .learning_coverage import CourseLearningCoverage


TopicMethod = Literal["manual", "embedding_cluster", "local_llm", "system"]
TopicStatus = Literal["suggested", "accepted", "hidden"]
TopicCardRole = Literal["primary", "supporting", "example"]
TopicRelationType = Literal["prerequisite", "related", "contrast_with"]


class Topic(BaseModel):
    id: str
    course_id: str
    parent_topic_id: str | None = None
    title: str = Field(min_length=1)
    summary: str | None = None
    position: int = Field(default=0, ge=0)
    depth: int = Field(default=0, ge=0, le=3)
    method: TopicMethod = "manual"
    status: TopicStatus = "accepted"
    is_system: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class TopicCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    summary: str | None = Field(default=None, max_length=2000)
    parent_topic_id: str | None = None
    position: int | None = Field(default=None, ge=0)


class TopicUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    summary: str | None = Field(default=None, max_length=2000)
    parent_topic_id: str | None = None
    position: int | None = Field(default=None, ge=0)
    status: TopicStatus | None = None


class TopicCardMembership(BaseModel):
    id: str
    topic_id: str
    card_id: str
    role: TopicCardRole = "primary"
    position: int = Field(default=0, ge=0)
    method: TopicMethod = "manual"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    status: TopicStatus = "accepted"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SetPrimaryTopicRequest(BaseModel):
    topic_id: str = Field(min_length=1)
    position: int | None = Field(default=None, ge=0)


class TopicRelation(BaseModel):
    id: str
    course_id: str
    source_topic_id: str
    target_topic_id: str
    relation_type: TopicRelationType
    explanation: str | None = None
    method: TopicMethod = "manual"
    status: TopicStatus = "accepted"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_distinct_topics(self) -> "TopicRelation":
        if self.source_topic_id == self.target_topic_id:
            raise ValueError("Topic relation cannot point to itself.")
        return self


class TopicRelationCreate(BaseModel):
    source_topic_id: str = Field(min_length=1)
    target_topic_id: str = Field(min_length=1)
    relation_type: TopicRelationType
    explanation: str | None = Field(default=None, max_length=2000)


class CourseMap(BaseModel):
    course_id: str
    topics: list[Topic] = Field(default_factory=list)
    memberships: list[TopicCardMembership] = Field(default_factory=list)
    topic_relations: list[TopicRelation] = Field(default_factory=list)
    cards: list[KnowledgeCardIndexItem] = Field(default_factory=list)
    coverage: CourseLearningCoverage = Field(default_factory=CourseLearningCoverage)


class TopicSuggestionRequest(BaseModel):
    target_topic_count: int | None = Field(default=None, ge=2, le=20)
    use_local_llm: bool = True
    model: str | None = Field(default=None, max_length=200)


class TopicSuggestionResult(BaseModel):
    course_id: str
    eligible_cards: int
    suggested_topics: list[Topic] = Field(default_factory=list)
    suggested_memberships: int
    embedding_model: str | None = None
    naming_method: TopicMethod
    warning: str | None = None
    mean_coherence: float | None = None
    singleton_topic_count: int = 0
    largest_topic_size: int = 0
    cluster_sizes: list[int] = Field(default_factory=list)


class TopicMergeRequest(BaseModel):
    source_topic_ids: list[str] = Field(min_length=1)


class TopicSplitRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    summary: str | None = Field(default=None, max_length=2000)
    card_ids: list[str] = Field(min_length=1)
