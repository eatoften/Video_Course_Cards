from app.job import VideoJob, VideoJobStatus
from app.job_store import create_job, get_job, list_jobs, update_job
from app.media_metadata import VideoMetadata


def test_job_store_round_trips_job(tmp_path):
    job = VideoJob(
        id="job-123",
        video_path=tmp_path / "lecture.mp4",
        status=VideoJobStatus.uploaded,
        original_filename="lecture.mp4",
        stored_name="job-123.mp4",
        size_bytes=123,
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


def test_job_store_lists_jobs(tmp_path):
    first_job = VideoJob(
        id="job-1",
        video_path=tmp_path / "first.mp4",
        status=VideoJobStatus.uploaded,
    )
    second_job = VideoJob(
        id="job-2",
        video_path=tmp_path / "second.mp4",
        status=VideoJobStatus.failed,
    )

    create_job(first_job)
    create_job(second_job)

    jobs = list_jobs()

    assert {job.id for job in jobs} == {"job-1", "job-2"}
