from app.job import VideoJob, VideoJobStatus
from app.job_store import create_job, get_job, update_job
from app.media_metadata import VideoMetadata


def test_job_store_round_trips_job(tmp_path):
    job = VideoJob(
        id="job-123",
        video_path=tmp_path / "lecture.mp4",
        status=VideoJobStatus.uploaded,
    )

    create_job(job)

    loaded_job = get_job(job.id)

    assert loaded_job == job


def test_job_store_updates_job_result(tmp_path):
    job = VideoJob(
        id="job-123",
        video_path=tmp_path / "lecture.mp4",
        status=VideoJobStatus.uploaded,
    )

    create_job(job)

    job.status = VideoJobStatus.completed
    job.metadata = VideoMetadata(
        duration_seconds=10.0,
        width=1920,
        height=1080,
        video_codec="h264",
        has_audio=True,
    )
    job.transcript_path = tmp_path / "transcripts" / "lecture.json"

    update_job(job)

    loaded_job = get_job(job.id)

    assert loaded_job == job
