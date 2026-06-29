from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .job import utc_now


KnowledgeCardNoteType = Literal[
    "user_note",
    "llm_explanation",
    "web_tutorial",
    "practice_question",
]
KnowledgeCardNoteSource = Literal["user", "local_llm", "web_llm"]


class KnowledgeCardNoteReference(BaseModel):
    title: str | None = None
    url: str | None = None
    accessed_at: datetime | None = None


class KnowledgeCardNoteBase(BaseModel):
    note_type: KnowledgeCardNoteType = "user_note"
    title: str | None = Field(default=None, min_length=1)
    body: str = Field(min_length=1)
    source: KnowledgeCardNoteSource = "user"
    sources: list[KnowledgeCardNoteReference] = Field(default_factory=list)


class KnowledgeCardNoteCreate(KnowledgeCardNoteBase):
    pass


class KnowledgeCardNoteUpdate(BaseModel):
    note_type: KnowledgeCardNoteType | None = None
    title: str | None = Field(default=None, min_length=1)
    body: str | None = Field(default=None, min_length=1)
    source: KnowledgeCardNoteSource | None = None
    sources: list[KnowledgeCardNoteReference] | None = None


class KnowledgeCardNote(KnowledgeCardNoteBase):
    id: str
    card_id: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
