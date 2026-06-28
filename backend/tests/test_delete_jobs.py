from fastapi.testclient import TestClient

import app.main as main
from app.job import VideoJob, VideoJobStatus
from app.job_store import create_job, get_job
from app.knowledge_card import KnowledgeCard
from app.knowledge_card_store import create_card, list_cards_for_job


client = TestClient(main.app)


def test_delete_job_removes_job_cards_and_artifacts(
    monkeypatch,
    tmp_path,
):
    data_dir = tmp_path / "data"
    upload_dir = data_dir / "uploads"
    audio_dir = data_dir / "audio"
    transcript_dir = data_dir / "transcripts"
    upload_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    transcript_dir.mkdir(parents=True)

    video_path = upload_dir / "job-123.mp4"
    transcript_path = transcript_dir / "job-123.json"
    audio_path = audio_dir / "job-123.wav"
    video_path.write_bytes(b"video")
    transcript_path.write_text("{}", encoding="utf-8")
    audio_path.write_bytes(b"audio")

    job = VideoJob(
        id="job-123",
        video_path=video_path,
        status=VideoJobStatus.completed,
        transcript_path=transcript_path,
    )
    create_job(job)
    create_card(
        KnowledgeCard(
            id="card-123",
            job_id=job.id,
            title="Linear Algebra",
            summary="A saved card.",
            claims=[
                {
                    "text": "Linear algebra is important.",
                    "evidence": [
                        {
                            "quote": "Linear algebra is important.",
                            "segment_start_seconds": 0.0,
                            "segment_end_seconds": 5.0,
                        }
                    ],
                }
            ],
            source_start_seconds=0.0,
            source_end_seconds=5.0,
        )
    )

    monkeypatch.setattr(main, "DATA_DIR", data_dir)

    response = client.delete(f"/jobs/{job.id}")

    assert response.status_code == 204
    assert get_job(job.id) is None
    assert list_cards_for_job(job.id) == []
    assert not video_path.exists()
    assert not transcript_path.exists()
    assert not audio_path.exists()


def test_delete_job_returns_404_for_missing_job():
    response = client.delete("/jobs/missing-job")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Job not found."
    }
