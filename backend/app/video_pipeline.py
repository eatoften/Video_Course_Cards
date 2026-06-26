from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from .audio_extractor import extract_audio
from .media_metadata import VideoMetadata, extract_video_metadata
from .media_probe import probe_video
from .transcript_store import save_transcription
from .transcription import (
    FasterWhisperTranscriber,
    TranscriptionResult,
)

from .job import VideoJob, VideoJobStatus

JobUpdateCallback = Callable[[VideoJob], None]


class VideoProcessingResult(BaseModel):
    metadata: VideoMetadata
    audio_path: Path
    transcript_path: Path
    transcription: TranscriptionResult


class VideoPipeline:
    def __init__(self, transcriber: FasterWhisperTranscriber) -> None:
        self._transcriber = transcriber

    def process(
        self,
        video_path: Path,
        artifact_root: Path,
        job: VideoJob | None = None,
        on_job_update: JobUpdateCallback | None = None,
    ) -> VideoProcessingResult:

        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # 1. probe
        if job:
            job.status = VideoJobStatus.probing
            if on_job_update:
                on_job_update(job)

        raw_media_data = probe_video(video_path)

        metadata = extract_video_metadata(raw_media_data)

        if job:
            job.metadata = metadata
            if on_job_update:
                on_job_update(job)

        # 2. audio path
        video_id = video_path.stem

        audio_path = artifact_root / "audio" / f"{video_id}.wav"
        transcript_path = artifact_root / "transcripts" / f"{video_id}.json"

        # 3. extract audio
        if job:
            job.status = VideoJobStatus.extracting_audio
            if on_job_update:
                on_job_update(job)

        saved_audio_path = extract_audio(video_path, audio_path)

        # 4. transcription
        if job:
            job.status = VideoJobStatus.transcribing
            if on_job_update:
                on_job_update(job)

        transcription = self._transcriber.transcribe(saved_audio_path)

        # 5. save transcript
        saved_transcript_path = save_transcription(
            transcription,
            transcript_path,
        )

        if job:
            job.transcript_path = saved_transcript_path
            job.status = VideoJobStatus.completed
            if on_job_update:
                on_job_update(job)

        return VideoProcessingResult(
            metadata=metadata,
            audio_path=saved_audio_path,
            transcript_path=saved_transcript_path,
            transcription=transcription,
        )
