import pytest

import app.rag_service as rag_service
from app.course import CourseCreate, DEFAULT_COURSE_ID
from app.course_service import create_video_course
from app.card_embedding_store import get_card_embedding
from app.job import VideoJob, VideoJobStatus
from app.job_store import create_job
from app.knowledge_card import (
    KnowledgeCard,
    KnowledgeCardClaim,
    KnowledgeCardEvidence,
)
from app.knowledge_card_store import create_card
from app.rag import RagRetrieveRequest


class FakeEmbedder:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(
        self,
        texts,
        *,
        batch_size=None,
    ) -> list[list[float]]:
        self.calls.append(list(texts))

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


def test_retrieve_cards_embeds_course_cards_and_ranks_results(tmp_path):
    job = create_uploaded_job(tmp_path, job_id="job-1")
    linear_card = create_test_card(
        card_id="card-linear",
        job_id=job.id,
        title="Linear Algebra",
        start_seconds=0.0,
    )
    deep_card = create_test_card(
        card_id="card-deep",
        job_id=job.id,
        title="Deep Learning",
        start_seconds=20.0,
    )
    fake_embedder = FakeEmbedder()

    response = rag_service.retrieve_cards(
        RagRetrieveRequest(
            question="What is linear algebra?",
            course_id=DEFAULT_COURSE_ID,
            top_k=2,
        ),
        embedder=fake_embedder,
    )

    assert [
        result.card_id
        for result in response.results
    ] == [
        linear_card.id,
        deep_card.id,
    ]
    assert response.results[0].score == pytest.approx(1.0)
    assert response.results[0].summary == "Linear Algebra summary."
    assert response.results[0].claims == linear_card.claims
    assert get_card_embedding(linear_card.id) is not None
    assert get_card_embedding(deep_card.id) is not None
    assert len(fake_embedder.calls) == 2


def test_retrieve_cards_can_scope_to_one_job(tmp_path):
    first_job = create_uploaded_job(tmp_path, job_id="job-1")
    second_job = create_uploaded_job(tmp_path, job_id="job-2")
    first_card = create_test_card(
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
    fake_embedder = FakeEmbedder()

    response = rag_service.retrieve_cards(
        RagRetrieveRequest(
            question="What is deep learning?",
            job_id=first_job.id,
            top_k=5,
        ),
        embedder=fake_embedder,
    )

    assert [
        result.card_id
        for result in response.results
    ] == [first_card.id]


def test_retrieve_cards_rejects_course_job_mismatch(tmp_path):
    course = create_video_course(
        CourseCreate(title="Other course")
    )
    job = create_uploaded_job(tmp_path, job_id="job-1")
    fake_embedder = FakeEmbedder()

    with pytest.raises(
        rag_service.RagScopeMismatchError,
        match="does not belong",
    ):
        rag_service.retrieve_cards(
            RagRetrieveRequest(
                question="What is linear algebra?",
                course_id=course.id,
                job_id=job.id,
            ),
            embedder=fake_embedder,
        )

    assert fake_embedder.calls == []


def test_retrieve_cards_returns_empty_results_without_cards(tmp_path):
    job = create_uploaded_job(tmp_path, job_id="job-1")
    fake_embedder = FakeEmbedder()

    response = rag_service.retrieve_cards(
        RagRetrieveRequest(
            question="What is linear algebra?",
            job_id=job.id,
        ),
        embedder=fake_embedder,
    )

    assert response.results == []
    assert fake_embedder.calls == []
