from enum import Enum

from pydantic import BaseModel
from pathlib import Path

from .media_metadata import VideoMetadata


class VideoJobStatus(str, Enum):
    uploaded = "uploaded"
    probing = "probing"
    extracting_audio = "extracting_audio"
    transcribing = "transcribing"
    completed = "completed"
    failed = "failed"

class VideoJob(BaseModel):
    id: str
    video_path: Path
    status: VideoJobStatus
    metadata: VideoMetadata | None = None


JOB_STORE: dict[str,VideoJob] = {}