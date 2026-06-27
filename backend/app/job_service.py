import shutil
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from .job import VideoJob, VideoJobStatus, utc_now
from .job_store import (
    create_job,
    get_job,
    list_jobs,
    update_job,
)
from .media_probe import MediaProbeError, probe_video
from .transcript_store import load_transcription
from .transcription import TranscriptionResult
from .video_pipeline import VideoPipeline


ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
}

TERMINAL_STATUSES = {
    VideoJobStatus.completed,
    VideoJobStatus.failed,
}


class JobServiceError(Exception):
    pass


class MissingFilenameError(JobServiceError):
    pass


class UnsupportedVideoExtensionError(JobServiceError):
    pass


class UnsupportedContentTypeError(JobServiceError):
    pass


class InvalidVideoError(JobServiceError):
    pass


class JobNotFoundError(JobServiceError):
    pass


class InvalidJobStatusError(JobServiceError):
    pass


class TranscriptNotReadyError(JobServiceError):
    pass


def create_video_job(
    *,
    video_file: BinaryIO,
    original_filename: str | None,
    content_type: str | None,
    upload_dir: Path,
) -> VideoJob:
    if not original_filename:
        raise MissingFilenameError(
            "Uploaded file must have a filename."
        )

    suffix = Path(original_filename).suffix.lower()

    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise UnsupportedVideoExtensionError(
            f"Unsupported video extension: {suffix or 'none'}"
        )

    if not content_type or not content_type.startswith("video/"):
        raise UnsupportedContentTypeError(
            f"Unsupported content type: {content_type}"
        )

    video_id = uuid4().hex
    destination = upload_dir / f"{video_id}{suffix}"

    try:
        upload_dir.mkdir(parents=True, exist_ok=True)

        with destination.open("wb") as output_file:
            shutil.copyfileobj(video_file, output_file)

        probe_video(destination)

    except MediaProbeError as exc:
        destination.unlink(missing_ok=True)

        raise InvalidVideoError(
            "Uploaded file is not a valid video."
        ) from exc

    except Exception:
        destination.unlink(missing_ok=True)
        raise

    now = utc_now()
    job = VideoJob(
        id=video_id,
        video_path=destination,
        status=VideoJobStatus.uploaded,
        original_filename=original_filename,
        stored_name=destination.name,
        size_bytes=destination.stat().st_size,
        created_at=now,
        updated_at=now,
    )

    create_job(job)

    return job


def list_video_jobs() -> list[VideoJob]:
    return list_jobs()


def get_video_job(job_id: str) -> VideoJob:
    job = get_job(job_id)

    if job is None:
        raise JobNotFoundError("Job not found.")

    return job


def start_job(job_id: str) -> VideoJob:
    job = get_video_job(job_id)

    if job.status != VideoJobStatus.uploaded:
        raise InvalidJobStatusError(
            f"Job cannot run from status: {job.status.value}"
        )

    _mark_started(job)
    update_job(job)

    return job


def retry_job(job_id: str) -> VideoJob:
    job = get_video_job(job_id)

    if job.status != VideoJobStatus.failed:
        raise InvalidJobStatusError(
            f"Job cannot retry from status: {job.status.value}"
        )

    job.metadata = None
    job.transcript_path = None

    _mark_started(job)
    update_job(job)

    return job


def run_job_pipeline(
    job_id: str,
    get_pipeline: Callable[[], VideoPipeline],
    artifact_root: Path,
) -> None:
    try:
        job = get_video_job(job_id)
    except JobNotFoundError:
        return

    try:
        pipeline = get_pipeline()

        pipeline.process(
            video_path=job.video_path,
            artifact_root=artifact_root,
            job=job,
            on_job_update=save_job_progress,
        )

        save_job_progress(job)

    except Exception as exc:
        job.status = VideoJobStatus.failed
        job.error_message = str(exc)

        save_job_progress(job)


def save_job_progress(job: VideoJob) -> None:
    now = utc_now()

    job.updated_at = now

    if job.status == VideoJobStatus.probing and job.started_at is None:
        job.started_at = now

    if job.status in TERMINAL_STATUSES and job.completed_at is None:
        job.completed_at = now

    update_job(job)


def get_job_transcript(job_id: str) -> TranscriptionResult:
    job = get_video_job(job_id)

    if job.transcript_path is None:
        raise TranscriptNotReadyError(
            "Transcript is not available for this job."
        )

    try:
        return load_transcription(job.transcript_path)
    except FileNotFoundError as exc:
        raise TranscriptNotReadyError(
            "Transcript file is missing."
        ) from exc


def _mark_started(job: VideoJob) -> None:
    now = utc_now()

    job.status = VideoJobStatus.probing
    job.error_message = None
    job.started_at = now
    job.completed_at = None
    job.updated_at = now
