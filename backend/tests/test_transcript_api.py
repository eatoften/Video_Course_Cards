from fastapi.testclient import TestClient

import app.main as main
from app.job import VideoJob, VideoJobStatus
from app.job_store import create_job
from app.transcript_store import save_transcription
from app.transcription import (
    TranscriptSegment,
    TranscriptionResult,
)


client = TestClient(main.app)


def test_get_job_transcript_returns_transcript(tmp_path):
    transcript = TranscriptionResult(
        language="en",
        language_probability=0.99,
        duration_seconds=2.0,
        segments=[
            TranscriptSegment(
                start_seconds=0.0,
                end_seconds=2.0,
                text="Hello world",
            )
        ],
    )
    transcript_path = (
        tmp_path
        / "transcripts"
        / "lecture.json"
    )
    save_transcription(transcript, transcript_path)

    job = VideoJob(
        id="job-123",
        video_path=tmp_path / "lecture.mp4",
        status=VideoJobStatus.completed,
        transcript_path=transcript_path,
    )
    create_job(job)

    response = client.get(
        "/jobs/job-123/transcript"
    )

    assert response.status_code == 200
    assert response.json() == transcript.model_dump()


def test_get_job_transcript_returns_409_when_not_ready(tmp_path):
    job = VideoJob(
        id="job-123",
        video_path=tmp_path / "lecture.mp4",
        status=VideoJobStatus.uploaded,
    )
    create_job(job)

    response = client.get(
        "/jobs/job-123/transcript"
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Transcript is not available for this job."
    }
