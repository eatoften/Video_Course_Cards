from fastapi.testclient import TestClient

import app.main as main
from app.job import (
    JOB_STORE,
    VideoJob,
    VideoJobStatus,
)


client = TestClient(main.app)


def test_upload_video_creates_job(monkeypatch, tmp_path):
    monkeypatch.setattr(
        main,
        "UPLOAD_DIR",
        tmp_path,
    )

    def fake_probe_video(file_path):
        assert file_path.is_file()

        return {
            "streams": [
                {
                    "codec_type": "video",
                }
            ],
            "format": {},
        }

    monkeypatch.setattr(
        main,
        "probe_video",
        fake_probe_video,
    )

    video_content = b"fake video content"

    files = {
        "video": (
            "lecture.mp4",
            video_content,
            "video/mp4",
        )
    }

    response = client.post(
        "/videos",
        files=files,
    )

    assert response.status_code == 201

    data = response.json()
    job_id = data["id"]

    assert data["status"] == "uploaded"
    assert job_id in JOB_STORE

    job = JOB_STORE[job_id]

    assert job.id == job_id
    assert job.status == VideoJobStatus.uploaded
    assert job.video_path == (
        tmp_path / data["stored_name"]
    )
    assert job.metadata is None




def test_get_job_returns_stored_job(tmp_path):
    video_path = tmp_path / "lecture.mp4"

    job = VideoJob(
        id="job-123",
        video_path=video_path,
        status=VideoJobStatus.uploaded,
    )

    JOB_STORE[job.id] = job

    response = client.get(
        "/jobs/job-123"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == "job-123"
    assert data["video_path"] == str(video_path)
    assert data["status"] == "uploaded"
    assert data["metadata"] is None



def test_get_job_returns_404_when_job_does_not_exist():
    response = client.get(
        "/jobs/missing-job"
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Job not found."
    }