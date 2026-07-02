import pytest
from pydantic import ValidationError

from app.rag import RagRetrieveRequest, RetrievedCard


def test_rag_retrieve_request_cleans_question_and_optional_ids():
    request = RagRetrieveRequest(
        question="  What   is   SVD?  ",
        course_id="  course-1  ",
        job_id="   ",
    )

    assert request.question == "What is SVD?"
    assert request.course_id == "course-1"
    assert request.job_id is None
    assert request.top_k == 5
    assert request.min_score is None


def test_rag_retrieve_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        RagRetrieveRequest(question="   ")


def test_rag_retrieve_request_validates_top_k():
    with pytest.raises(ValidationError):
        RagRetrieveRequest(
            question="What is SVD?",
            top_k=0,
        )

    with pytest.raises(ValidationError):
        RagRetrieveRequest(
            question="What is SVD?",
            top_k=51,
        )


def test_rag_retrieve_request_validates_min_score_range():
    with pytest.raises(ValidationError):
        RagRetrieveRequest(
            question="What is SVD?",
            min_score=-1.01,
        )

    with pytest.raises(ValidationError):
        RagRetrieveRequest(
            question="What is SVD?",
            min_score=1.01,
        )


def test_retrieved_card_validates_score_and_source_range():
    card = RetrievedCard(
        card_id="card-1",
        job_id="job-1",
        title="SVD",
        summary="A matrix factorization.",
        score=0.82,
        source_start_seconds=12.0,
        source_end_seconds=18.0,
    )

    assert card.card_id == "card-1"
    assert card.score == 0.82

    with pytest.raises(ValidationError):
        RetrievedCard(
            card_id="card-1",
            job_id="job-1",
            title="SVD",
            summary="A matrix factorization.",
            score=1.01,
            source_start_seconds=12.0,
            source_end_seconds=18.0,
        )

    with pytest.raises(ValidationError):
        RetrievedCard(
            card_id="card-1",
            job_id="job-1",
            title="SVD",
            summary="A matrix factorization.",
            score=0.82,
            source_start_seconds=12.0,
            source_end_seconds=12.0,
        )
