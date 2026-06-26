from pathlib import Path

from fastapi.testclient import TestClient

import app.main as main
from app.job import (
    VideoJob,
    VideoJobStatus,
)
from app.job_store import create_job as save_job
from app.job_store import get_job
from app.media_metadata import VideoMetadata


client = TestClient(main.app)


def create_job(
    tmp_path: Path,
    status: VideoJobStatus = VideoJobStatus.uploaded,
) -> VideoJob:
    video_path = tmp_path / "lecture.mp4"
    video_path.write_bytes(b"fake video")

    job = VideoJob(
        id="job-123",
        video_path=video_path,
        status=status,
    )

    save_job(job)

    return job


def test_run_job_completes_uploaded_job(
    monkeypatch,
    tmp_path,
):
    job = create_job(tmp_path)

    metadata = VideoMetadata(
        duration_seconds=10.0,
        width=1920,
        height=1080,
        video_codec="h264",
        has_audio=True,
    )

    transcript_path = (
        tmp_path
        / "transcripts"
        / "lecture.json"
    )

    calls = []

    class FakePipeline:
        def process(
            self,
            video_path,
            artifact_root,
            job,
            on_job_update=None,
        ):
            calls.append("process")

            assert video_path == job.video_path
            assert artifact_root == main.DATA_DIR

            job.status = VideoJobStatus.probing
            job.metadata = metadata
            job.transcript_path = transcript_path
            job.status = VideoJobStatus.completed

    fake_pipeline = FakePipeline()

    monkeypatch.setattr(
        main,
        "get_video_pipeline",
        lambda: fake_pipeline,
    )

    response = client.post(
        f"/jobs/{job.id}/run"
    )

    assert response.status_code == 202

    data = response.json()

    assert data["id"] == job.id
    assert data["status"] == "probing"
    assert data["metadata"] is None
    assert data["transcript_path"] is None
    assert data["error_message"] is None

    assert calls == ["process"]

    stored_job = get_job(job.id)

    assert stored_job is not None
    assert stored_job.status == VideoJobStatus.completed
    assert stored_job.metadata == metadata
    assert stored_job.transcript_path == transcript_path


def test_run_job_returns_404_for_missing_job():
    response = client.post(
        "/jobs/missing-job/run"
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Job not found."
    }


def test_run_job_returns_409_when_job_is_not_uploaded(
    monkeypatch,
    tmp_path,
):
    job = create_job(
        tmp_path,
        status=VideoJobStatus.completed,
    )

    def fail_if_pipeline_is_requested():
        raise AssertionError(
            "Pipeline must not load for a completed job"
        )

    monkeypatch.setattr(
        main,
        "get_video_pipeline",
        fail_if_pipeline_is_requested,
    )

    response = client.post(
        f"/jobs/{job.id}/run"
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": (
            "Job cannot run from status: completed"
        )
    }

    assert job.status == VideoJobStatus.completed


def test_run_job_marks_job_failed_when_pipeline_fails(
    monkeypatch,
    tmp_path,
):
    job = create_job(tmp_path)

    class FailingPipeline:
        def process(
            self,
            video_path,
            artifact_root,
            job,
            on_job_update=None,
        ):
            job.status = VideoJobStatus.transcribing

            raise RuntimeError(
                "Whisper inference failed"
            )

    monkeypatch.setattr(
        main,
        "get_video_pipeline",
        lambda: FailingPipeline(),
    )

    response = client.post(
        f"/jobs/{job.id}/run"
    )

    assert response.status_code == 202

    stored_job = get_job(job.id)

    assert stored_job is not None
    assert stored_job.status == VideoJobStatus.failed
    assert (
        stored_job.error_message
        == "Whisper inference failed"
    )
