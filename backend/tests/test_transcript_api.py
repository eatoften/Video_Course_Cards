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


def test_get_job_context_returns_overlapping_segments(tmp_path):
    transcript = TranscriptionResult(
        language="en",
        language_probability=0.99,
        duration_seconds=8.0,
        segments=[
            TranscriptSegment(
                start_seconds=0.0,
                end_seconds=2.0,
                text="Intro",
            ),
            TranscriptSegment(
                start_seconds=2.0,
                end_seconds=4.0,
                text="Gradient descent",
            ),
            TranscriptSegment(
                start_seconds=4.0,
                end_seconds=6.0,
                text="Learning rate",
            ),
            TranscriptSegment(
                start_seconds=6.0,
                end_seconds=8.0,
                text="Summary",
            ),
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
        original_filename="lecture.mp4",
        transcript_path=transcript_path,
    )
    create_job(job)

    response = client.get(
        "/jobs/job-123/context",
        params={
            "start_seconds": 1.5,
            "end_seconds": 5.0,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["job_id"] == "job-123"
    assert data["source_video"] == "lecture.mp4"
    assert data["start_seconds"] == 1.5
    assert data["end_seconds"] == 5.0
    assert [
        segment["text"]
        for segment in data["segments"]
    ] == [
        "Intro",
        "Gradient descent",
        "Learning rate",
    ]
    assert data["text"] == (
        "Intro\n"
        "Gradient descent\n"
        "Learning rate"
    )


def test_get_job_context_returns_400_for_invalid_window(tmp_path):
    job = VideoJob(
        id="job-123",
        video_path=tmp_path / "lecture.mp4",
        status=VideoJobStatus.completed,
    )
    create_job(job)

    response = client.get(
        "/jobs/job-123/context",
        params={
            "start_seconds": 5.0,
            "end_seconds": 5.0,
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Context end must be greater than start."
    }


def test_get_job_context_returns_409_when_transcript_not_ready(tmp_path):
    job = VideoJob(
        id="job-123",
        video_path=tmp_path / "lecture.mp4",
        status=VideoJobStatus.uploaded,
    )
    create_job(job)

    response = client.get(
        "/jobs/job-123/context",
        params={
            "start_seconds": 0.0,
            "end_seconds": 10.0,
        },
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Transcript is not available for this job."
    }
