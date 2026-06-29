from datetime import datetime, timezone

from pydantic import BaseModel, Field


DEFAULT_COURSE_ID = "uncategorized"
DEFAULT_COURSE_TITLE = "Uncategorized"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CourseBase(BaseModel):
    title: str = Field(min_length=1)
    description: str | None = None


class CourseCreate(CourseBase):
    pass


class CourseUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1)
    description: str | None = None


class Course(CourseBase):
    id: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    job_count: int = 0
    card_count: int = 0
