from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from .job import utc_now


DEFAULT_CHUNKER_VERSION = "semantic-v1"


class TranscriptChunkGenerationRequest(BaseModel):
    context_radius: int = Field(default=1, ge=0, le=5)
    min_chunk_seconds: float = Field(default=120.0, ge=0)
    max_chunk_seconds: float = Field(default=360.0, gt=0)
    boundary_percentile: float = Field(default=90.0, ge=0, le=100)
    replace_existing: bool = True

    @model_validator(mode="after")
    def validate_duration_window(self):
        if self.max_chunk_seconds < self.min_chunk_seconds:
            raise ValueError(
                "max_chunk_seconds must be greater than or equal to "
                "min_chunk_seconds."
            )

        return self


class TranscriptChunk(BaseModel):
    id: str
    course_id: str
    job_id: str
    chunk_index: int = Field(ge=0)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    text: str = Field(min_length=1)
    segment_ids: list[int] = Field(min_length=1)
    chunker_version: str = DEFAULT_CHUNKER_VERSION
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_time_range(self):
        if self.end_seconds < self.start_seconds:
            raise ValueError(
                "end_seconds must be greater than or equal to start_seconds."
            )

        return self
