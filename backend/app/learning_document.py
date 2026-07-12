from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .job import utc_now


LearningDocumentStatus = Literal["draft", "reviewed", "needs_fix"]
LearningDocumentGenerationMode = Literal["manual", "local_llm", "imported"]
LearningDocumentCardRole = Literal[
    "primary_anchor",
    "supporting",
    "example",
    "contrast",
    "prerequisite",
]
LearningDocumentSourceType = Literal["card_claim", "source_unit"]
LearningDocumentChangeSource = Literal["manual", "local_llm", "imported"]


class LearningDocument(BaseModel):
    id: str
    course_id: str
    title: str = Field(min_length=1, max_length=300)
    summary: str = Field(default="", max_length=4000)
    body_markdown: str = ""
    status: LearningDocumentStatus = "draft"
    generation_mode: LearningDocumentGenerationMode = "manual"
    provider: str | None = None
    model: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class LearningDocumentCreate(BaseModel):
    title: str | None = Field(default=None, max_length=300)
    summary: str = Field(default="", max_length=4000)
    body_markdown: str = ""


class LearningDocumentUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=300)
    summary: str | None = Field(default=None, max_length=4000)
    body_markdown: str | None = None
    status: LearningDocumentStatus | None = None


class LearningDocumentCardLink(BaseModel):
    id: str
    document_id: str
    card_id: str
    role: LearningDocumentCardRole
    position: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)


class LearningDocumentCardLinkCreate(BaseModel):
    card_id: str = Field(min_length=1)
    role: LearningDocumentCardRole = "supporting"
    position: int = Field(default=0, ge=0)


class LearningDocumentSource(BaseModel):
    id: str
    document_id: str
    source_type: LearningDocumentSourceType
    source_id: str
    card_id: str | None = None
    label: str
    quote: str
    locator: dict[str, object] = Field(default_factory=dict)
    position: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)


class LearningDocumentVersion(BaseModel):
    id: str
    document_id: str
    version_number: int = Field(ge=1)
    title: str
    summary: str
    body_markdown: str
    change_source: LearningDocumentChangeSource
    provider: str | None = None
    model: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class LearningDocumentDetail(LearningDocument):
    card_links: list[LearningDocumentCardLink] = Field(default_factory=list)
    sources: list[LearningDocumentSource] = Field(default_factory=list)
    versions: list[LearningDocumentVersion] = Field(default_factory=list)


class LearningDocumentGenerateRequest(BaseModel):
    source_asset_ids: list[str] = Field(default_factory=list)
    supporting_card_ids: list[str] = Field(default_factory=list)
    focus: str | None = Field(default=None, max_length=2000)
    model: str | None = Field(default=None, max_length=200)


class LearningDocumentGenerationResult(BaseModel):
    document: LearningDocumentDetail
    selected_source_units: int
    selected_cards: int
    warning: str | None = None


class LearningDocumentRestoreRequest(BaseModel):
    version_number: int = Field(ge=1)
