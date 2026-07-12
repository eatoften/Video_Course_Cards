from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .job import utc_now


SourceAssetType = Literal["video", "audio", "pptx", "pdf", "docx", "text"]
SourceAssetStatus = Literal["pending", "ready", "failed"]
SourceUnitType = Literal[
    "transcript_segment",
    "slide",
    "page",
    "paragraph",
    "text",
    "video_frame",
]


class SourceAsset(BaseModel):
    id: str
    course_id: str
    job_id: str | None = None
    asset_type: SourceAssetType
    original_filename: str
    stored_path: str
    mime_type: str | None = None
    size_bytes: int = Field(ge=0)
    sha256: str
    extraction_status: SourceAssetStatus = "pending"
    metadata: dict[str, object] = Field(default_factory=dict)
    error_message: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SourceUnit(BaseModel):
    id: str
    asset_id: str
    unit_type: SourceUnitType
    ordinal: int = Field(ge=0)
    text: str = Field(min_length=1)
    locator: dict[str, object] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class SourceAssetDetail(SourceAsset):
    unit_count: int = 0


class SourceAssetImportResult(BaseModel):
    asset: SourceAssetDetail
    units: list[SourceUnit] = Field(default_factory=list)
