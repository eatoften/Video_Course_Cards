from fastapi.testclient import TestClient

import app.main as main
import app.transcript_chunk_service as transcript_chunk_service
from app.job import VideoJob, VideoJobStatus
from app.job_store import create_job
from app.transcript_chunk import TranscriptChunk
from app.transcript_chunk_store import (
    list_chunks_for_job,
    replace_chunks_for_job,
)
from app.transcript_chunker import (
    SemanticChunkerConfig,
    chunk_transcript_segments,
)
from app.transcript_store import save_transcription
from app.transcription import TranscriptSegment, TranscriptionResult


client = TestClient(main.app)


class FakeEmbedder:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.texts: list[str] = []

    def embed_texts(
        self,
        texts,
        *,
        batch_size=None,
    ) -> list[list[float]]:
        self.texts = list(texts)

        return self.vectors[:len(self.texts)]


def test_chunker_splits_on_high_semantic_distance():
    segments = [
        TranscriptSegment(
            start_seconds=0,
            end_seconds=30,
            text="Image classification assigns labels to images.",
        ),
        TranscriptSegment(
            start_seconds=30,
            end_seconds=60,
            text="Nearest neighbor compares a test image to examples.",
        ),
        TranscriptSegment(
            start_seconds=60,
            end_seconds=90,
            text="Loss functions measure prediction mistakes.",
        ),
        TranscriptSegment(
            start_seconds=90,
            end_seconds=120,
            text="Optimization updates parameters using gradients.",
        ),
    ]
    embedder = FakeEmbedder(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
        ]
    )

    chunks = chunk_transcript_segments(
        segments,
        embedder=embedder,
        config=SemanticChunkerConfig(
            context_radius=0,
            min_chunk_seconds=30,
            max_chunk_seconds=300,
            boundary_percentile=80,
        ),
        course_id="course-1",
        job_id="job-1",
    )

    assert len(chunks) == 2
    assert chunks[0].segment_ids == [0, 1]
    assert chunks[0].start_seconds == 0
    assert chunks[0].end_seconds == 60
    assert chunks[1].segment_ids == [2, 3]
    assert chunks[1].start_seconds == 60
    assert chunks[1].end_seconds == 120
    assert embedder.texts == [
        "Image classification assigns labels to images.",
        "Nearest neighbor compares a test image to examples.",
        "Loss functions measure prediction mistakes.",
        "Optimization updates parameters using gradients.",
    ]


def test_chunk_store_replaces_job_chunks():
    first_chunk = TranscriptChunk(
        id="chunk-1",
        course_id="course-1",
        job_id="job-1",
        chunk_index=0,
        start_seconds=0,
        end_seconds=60,
        text="First chunk",
        segment_ids=[0, 1],
    )
    replacement_chunk = TranscriptChunk(
        id="chunk-2",
        course_id="course-1",
        job_id="job-1",
        chunk_index=0,
        start_seconds=60,
        end_seconds=120,
        text="Replacement chunk",
        segment_ids=[2, 3],
    )

    replace_chunks_for_job("job-1", [first_chunk])
    replace_chunks_for_job("job-1", [replacement_chunk])

    chunks = list_chunks_for_job("job-1")

    assert [
        chunk.id
        for chunk in chunks
    ] == ["chunk-2"]
    assert chunks[0].segment_ids == [2, 3]


def test_generate_and_list_job_chunks_api(monkeypatch, tmp_path):
    transcript = TranscriptionResult(
        language="en",
        language_probability=0.99,
        duration_seconds=120,
        segments=[
            TranscriptSegment(
                start_seconds=0,
                end_seconds=30,
                text="Image classification assigns labels to images.",
            ),
            TranscriptSegment(
                start_seconds=30,
                end_seconds=60,
                text="Nearest neighbor compares a test image to examples.",
            ),
            TranscriptSegment(
                start_seconds=60,
                end_seconds=90,
                text="Loss functions measure prediction mistakes.",
            ),
            TranscriptSegment(
                start_seconds=90,
                end_seconds=120,
                text="Optimization updates parameters using gradients.",
            ),
        ],
    )
    transcript_path = tmp_path / "transcripts" / "lecture.json"
    save_transcription(transcript, transcript_path)
    job = VideoJob(
        id="job-1",
        video_path=tmp_path / "lecture.mp4",
        status=VideoJobStatus.completed,
        transcript_path=transcript_path,
    )
    create_job(job)
    fake_embedder = FakeEmbedder(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
        ]
    )
    monkeypatch.setattr(
        transcript_chunk_service,
        "_create_default_embedder",
        lambda: fake_embedder,
    )

    response = client.post(
        "/jobs/job-1/chunks",
        json={
            "context_radius": 0,
            "min_chunk_seconds": 30,
            "max_chunk_seconds": 300,
            "boundary_percentile": 80,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 2
    assert data[0]["job_id"] == "job-1"
    assert data[0]["course_id"] == "uncategorized"
    assert data[0]["segment_ids"] == [0, 1]
    assert data[0]["start_seconds"] == 0
    assert data[0]["end_seconds"] == 60

    list_response = client.get("/jobs/job-1/chunks")

    assert list_response.status_code == 200
    assert [
        chunk["segment_ids"]
        for chunk in list_response.json()
    ] == [
        [0, 1],
        [2, 3],
    ]


def test_generate_job_chunks_returns_409_when_transcript_not_ready(tmp_path):
    job = VideoJob(
        id="job-1",
        video_path=tmp_path / "lecture.mp4",
        status=VideoJobStatus.uploaded,
    )
    create_job(job)

    response = client.post("/jobs/job-1/chunks")

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Transcript is not available for this job."
    }
