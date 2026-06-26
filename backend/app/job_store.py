from pathlib import Path
from sqlite3 import Row

from .db import connect, ensure_db
from .job import VideoJob, VideoJobStatus
from .media_metadata import VideoMetadata


def _metadata_to_json(job: VideoJob) -> str | None:
    if job.metadata is None:
        return None

    return job.metadata.model_dump_json()


def _path_to_text(path: Path | None) -> str | None:
    if path is None:
        return None

    return str(path)


def _row_to_job(row: Row) -> VideoJob:
    metadata = None
    if row["metadata"] is not None:
        metadata = VideoMetadata.model_validate_json(row["metadata"])

    transcript_path = None
    if row["transcript_path"] is not None:
        transcript_path = Path(row["transcript_path"])

    return VideoJob(
        id=row["id"],
        video_path=Path(row["video_path"]),
        status=VideoJobStatus(row["status"]),
        metadata=metadata,
        transcript_path=transcript_path,
        error_message=row["error_message"],
    )


def create_job(job: VideoJob) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                id,
                video_path,
                status,
                metadata,
                transcript_path,
                error_message
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                job.id,
                str(job.video_path),
                job.status.value,
                _metadata_to_json(job),
                _path_to_text(job.transcript_path),
                job.error_message,
            ),
        )


def get_job(job_id: str) -> VideoJob | None:
    ensure_db()

    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()

    if row is None:
        return None

    return _row_to_job(row)


def update_job(job: VideoJob) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = ?,
                metadata = ?,
                transcript_path = ?,
                error_message = ?
            WHERE id = ?
            """,
            (
                job.status.value,
                _metadata_to_json(job),
                _path_to_text(job.transcript_path),
                job.error_message,
                job.id,
            ),
        )


def clear_jobs() -> None:
    ensure_db()

    with connect() as conn:
        conn.execute("DELETE FROM jobs")
