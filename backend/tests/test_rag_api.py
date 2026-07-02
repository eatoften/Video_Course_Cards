from fastapi.testclient import TestClient

import app.main as main
import app.rag_service as rag_service
from app.course import CourseCreate, DEFAULT_COURSE_ID
from app.course_service import create_video_course
from app.job import VideoJob, VideoJobStatus
from app.job_store import create_job
from app.knowledge_card import (
    KnowledgeCard,
    KnowledgeCardClaim,
    KnowledgeCardEvidence,
)
from app.knowledge_card_store import create_card


client = TestClient(main.app)


class FakeEmbedder:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def embed_texts(
        self,
        texts,
        *,
        batch_size=None,
    ) -> list[list[float]]:
        return [
            self._vector_for_text(text)
            for text in texts
        ]

    def _vector_for_text(self, text: str) -> list[float]:
        if text == "What is linear algebra?":
            return [1.0, 0.0]

        if text == "What is deep learning?":
            return [0.0, 1.0]

        if "Title:\nLinear Algebra" in text:
            return [1.0, 0.0]

        if "Title:\nDeep Learning" in text:
            return [0.0, 1.0]

        return [0.5, 0.5]


def create_uploaded_job(
    tmp_path,
    *,
    job_id: str,
    course_id: str = DEFAULT_COURSE_ID,
) -> VideoJob:
    job = VideoJob(
        id=job_id,
        course_id=course_id,
        video_path=tmp_path / f"{job_id}.mp4",
        status=VideoJobStatus.completed,
        original_filename=f"{job_id}.mp4",
    )
    create_job(job)

    return job


def create_test_card(
    *,
    card_id: str,
    job_id: str,
    title: str,
    start_seconds: float,
) -> KnowledgeCard:
    card = KnowledgeCard(
        id=card_id,
        job_id=job_id,
        title=title,
        summary=f"{title} summary.",
        key_points=[f"{title} key point"],
        claims=[
            KnowledgeCardClaim(
                text=f"{title} claim.",
                evidence=[
                    KnowledgeCardEvidence(
                        quote=f"{title} evidence.",
                        segment_start_seconds=start_seconds,
                        segment_end_seconds=start_seconds + 5.0,
                    )
                ],
            )
        ],
        source_start_seconds=start_seconds,
        source_end_seconds=start_seconds + 10.0,
        tags=[title.lower()],
    )
    create_card(card)

    return card


def test_rag_retrieve_returns_ranked_cards(monkeypatch, tmp_path):
    job = create_uploaded_job(tmp_path, job_id="job-1")
    linear_card = create_test_card(
        card_id="card-linear",
        job_id=job.id,
        title="Linear Algebra",
        start_seconds=0.0,
    )
    create_test_card(
        card_id="card-deep",
        job_id=job.id,
        title="Deep Learning",
        start_seconds=20.0,
    )
    monkeypatch.setattr(
        rag_service,
        "_create_default_embedder",
        lambda: FakeEmbedder(),
    )

    response = client.post(
        "/rag/retrieve",
        json={
            "question": "What is linear algebra?",
            "course_id": DEFAULT_COURSE_ID,
            "top_k": 2,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == "What is linear algebra?"
    assert [
        result["card_id"]
        for result in data["results"]
    ] == [
        linear_card.id,
        "card-deep",
    ]
    assert data["results"][0]["score"] == 1.0
    assert data["results"][0]["summary"] == "Linear Algebra summary."
    assert data["results"][0]["claims"] == [
        claim.model_dump(mode="json")
        for claim in linear_card.claims
    ]


def test_rag_retrieve_can_scope_to_job(monkeypatch, tmp_path):
    first_job = create_uploaded_job(tmp_path, job_id="job-1")
    second_job = create_uploaded_job(tmp_path, job_id="job-2")
    create_test_card(
        card_id="card-linear",
        job_id=first_job.id,
        title="Linear Algebra",
        start_seconds=0.0,
    )
    create_test_card(
        card_id="card-deep",
        job_id=second_job.id,
        title="Deep Learning",
        start_seconds=20.0,
    )
    monkeypatch.setattr(
        rag_service,
        "_create_default_embedder",
        lambda: FakeEmbedder(),
    )

    response = client.post(
        "/rag/retrieve",
        json={
            "question": "What is deep learning?",
            "job_id": first_job.id,
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    assert [
        result["card_id"]
        for result in response.json()["results"]
    ] == ["card-linear"]


def test_rag_retrieve_returns_400_for_course_job_mismatch(
    monkeypatch,
    tmp_path,
):
    course = create_video_course(CourseCreate(title="Other course"))
    job = create_uploaded_job(tmp_path, job_id="job-1")
    monkeypatch.setattr(
        rag_service,
        "_create_default_embedder",
        lambda: FakeEmbedder(),
    )

    response = client.post(
        "/rag/retrieve",
        json={
            "question": "What is linear algebra?",
            "course_id": course.id,
            "job_id": job.id,
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Job does not belong to the requested course."
    }


def test_rag_retrieve_returns_404_for_missing_scope():
    response = client.post(
        "/rag/retrieve",
        json={
            "question": "What is linear algebra?",
            "job_id": "missing-job",
        },
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Job not found."
    }
