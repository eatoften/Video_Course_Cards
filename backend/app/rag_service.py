from __future__ import annotations

from dataclasses import dataclass

from . import card_embedding_service
from . import course_service
from . import job_service
from .card_embedding import CardEmbedding
from .card_embedding_store import (
    list_card_embeddings_for_course,
    list_card_embeddings_for_job,
)
from .course import DEFAULT_COURSE_ID
from .embedding import EmbeddingError, SentenceTransformerEmbedder, TextEmbedder
from .knowledge_card import KnowledgeCard
from .knowledge_card_store import (
    list_cards_for_course,
    list_cards_for_job,
)
from .rag import RagRetrieveRequest, RagRetrieveResponse
from .rag_retriever import CardEmbeddingCandidate, rank_cards_by_similarity


class RagServiceError(Exception):
    pass


class RagScopeMismatchError(RagServiceError):
    pass


class RagRetrievalError(RagServiceError):
    pass


@dataclass(frozen=True)
class _RetrievalScope:
    course_id: str
    job_id: str | None = None


def retrieve_cards(
    request: RagRetrieveRequest,
    *,
    embedder: TextEmbedder | None = None,
) -> RagRetrieveResponse:
    active_embedder = embedder or _create_default_embedder()
    scope = _resolve_scope(request)

    if scope.job_id is not None:
        card_embedding_service.embed_job_cards(
            scope.job_id,
            embedder=active_embedder,
        )
        cards = list_cards_for_job(scope.job_id)
        embeddings = list_card_embeddings_for_job(scope.job_id)
    else:
        card_embedding_service.embed_course_cards(
            scope.course_id,
            embedder=active_embedder,
        )
        cards = list_cards_for_course(scope.course_id)
        embeddings = list_card_embeddings_for_course(scope.course_id)

    candidates = _pair_cards_with_embeddings(cards, embeddings)

    if not candidates:
        return RagRetrieveResponse(
            question=request.question,
            results=[],
        )

    query_vector = _embed_query(
        request.question,
        embedder=active_embedder,
    )

    try:
        results = rank_cards_by_similarity(
            query_vector,
            candidates,
            top_k=request.top_k,
            min_score=request.min_score,
        )
    except ValueError as exc:
        raise RagRetrievalError(str(exc)) from exc

    return RagRetrieveResponse(
        question=request.question,
        results=results,
    )


def _resolve_scope(request: RagRetrieveRequest) -> _RetrievalScope:
    if request.job_id is not None:
        job = job_service.get_video_job(request.job_id)

        if (
            request.course_id is not None
            and request.course_id != job.course_id
        ):
            raise RagScopeMismatchError(
                "Job does not belong to the requested course."
            )

        return _RetrievalScope(
            course_id=job.course_id,
            job_id=job.id,
        )

    course_id = request.course_id or DEFAULT_COURSE_ID
    course = course_service.get_video_course(course_id)

    return _RetrievalScope(course_id=course.id)


def _pair_cards_with_embeddings(
    cards: list[KnowledgeCard],
    embeddings: list[CardEmbedding],
) -> list[CardEmbeddingCandidate]:
    cards_by_id = {
        card.id: card
        for card in cards
    }
    candidates: list[CardEmbeddingCandidate] = []

    for embedding in embeddings:
        card = cards_by_id.get(embedding.card_id)

        if card is None:
            continue

        candidates.append((card, embedding))

    return candidates


def _embed_query(
    question: str,
    *,
    embedder: TextEmbedder,
) -> list[float]:
    try:
        vectors = embedder.embed_texts([question])
    except EmbeddingError as exc:
        raise RagRetrievalError(str(exc)) from exc

    if len(vectors) != 1:
        raise RagRetrievalError(
            "Embedding model returned the wrong number of query vectors."
        )

    return vectors[0]


def _create_default_embedder() -> TextEmbedder:
    return SentenceTransformerEmbedder()
