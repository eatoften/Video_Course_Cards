from __future__ import annotations

from collections.abc import Sequence

from . import course_service
from . import job_service
from .card_embedding import (
    CardEmbedding,
    CardEmbeddingBatchResult,
    CardEmbeddingStatus,
)
from .card_embedding_store import (
    get_card_embedding_info,
    upsert_card_embeddings,
)
from .card_embedding_text import (
    build_card_embedding_text,
    hash_card_embedding_text,
)
from .embedding import EmbeddingError, SentenceTransformerEmbedder, TextEmbedder
from .job import utc_now
from .knowledge_card import KnowledgeCard
from .knowledge_card_store import (
    get_card,
    list_cards_for_course,
    list_cards_for_job,
)
from .settings import get_embedding_settings


class CardEmbeddingServiceError(Exception):
    pass


class CardEmbeddingCardNotFoundError(CardEmbeddingServiceError):
    pass


class CardEmbeddingGenerationError(CardEmbeddingServiceError):
    pass


def embed_card(
    card_id: str,
    *,
    embedder: TextEmbedder | None = None,
) -> CardEmbeddingBatchResult:
    card = get_card(card_id)

    if card is None:
        raise CardEmbeddingCardNotFoundError("Knowledge card not found.")

    return _embed_cards([card], embedder=embedder)


def embed_job_cards(
    job_id: str,
    *,
    embedder: TextEmbedder | None = None,
) -> CardEmbeddingBatchResult:
    job = job_service.get_video_job(job_id)

    return _embed_cards(
        list_cards_for_job(job.id),
        embedder=embedder,
    )


def embed_course_cards(
    course_id: str,
    *,
    embedder: TextEmbedder | None = None,
) -> CardEmbeddingBatchResult:
    course = course_service.get_video_course(course_id)

    return _embed_cards(
        list_cards_for_course(course.id),
        embedder=embedder,
    )


def get_card_embedding_status(card_id: str) -> CardEmbeddingStatus:
    card = get_card(card_id)

    if card is None:
        raise CardEmbeddingCardNotFoundError("Knowledge card not found.")

    return _embedding_status_for_cards([card])


def get_job_card_embedding_status(job_id: str) -> CardEmbeddingStatus:
    job = job_service.get_video_job(job_id)

    return _embedding_status_for_cards(list_cards_for_job(job.id))


def get_course_card_embedding_status(course_id: str) -> CardEmbeddingStatus:
    course = course_service.get_video_course(course_id)

    return _embedding_status_for_cards(list_cards_for_course(course.id))


def _embed_cards(
    cards: Sequence[KnowledgeCard],
    *,
    embedder: TextEmbedder | None,
) -> CardEmbeddingBatchResult:
    active_embedder = embedder or _create_default_embedder()
    model_name = _embedder_model_name(active_embedder)
    pending_cards: list[KnowledgeCard] = []
    pending_texts: list[str] = []
    pending_hashes: list[str] = []

    for card in cards:
        text = build_card_embedding_text(card)
        text_hash = hash_card_embedding_text(text)
        existing_embedding = get_card_embedding_info(card.id)

        if (
            existing_embedding is not None
            and existing_embedding.model == model_name
            and existing_embedding.text_hash == text_hash
        ):
            continue

        pending_cards.append(card)
        pending_texts.append(text)
        pending_hashes.append(text_hash)

    if not pending_cards:
        return CardEmbeddingBatchResult(
            total_cards=len(cards),
            embedded_cards=0,
            skipped_cards=len(cards),
            model=model_name,
            dimension=_status_dimension_for_cards(cards),
        )

    try:
        vectors = active_embedder.embed_texts(pending_texts)
    except EmbeddingError as exc:
        raise CardEmbeddingGenerationError(str(exc)) from exc

    if len(vectors) != len(pending_cards):
        raise CardEmbeddingGenerationError(
            "Embedding model returned the wrong number of vectors."
        )

    dimension = len(vectors[0]) if vectors else None
    now = utc_now()
    embeddings = [
        CardEmbedding(
            card_id=card.id,
            model=model_name,
            dimension=len(vector),
            text_hash=text_hash,
            vector=vector,
            created_at=now,
            updated_at=now,
        )
        for card, text_hash, vector in zip(
            pending_cards,
            pending_hashes,
            vectors,
        )
    ]

    upsert_card_embeddings(embeddings)

    return CardEmbeddingBatchResult(
        total_cards=len(cards),
        embedded_cards=len(embeddings),
        skipped_cards=len(cards) - len(embeddings),
        model=model_name,
        dimension=dimension,
    )


def _embedding_status_for_cards(
    cards: Sequence[KnowledgeCard],
) -> CardEmbeddingStatus:
    model_name = get_embedding_settings().model
    embedded_cards = 0
    missing_cards = 0
    stale_cards = 0
    dimension: int | None = None

    for card in cards:
        text_hash = hash_card_embedding_text(
            build_card_embedding_text(card)
        )
        embedding = get_card_embedding_info(card.id)

        if embedding is None:
            missing_cards += 1
            continue

        if dimension is None:
            dimension = embedding.dimension

        if embedding.model != model_name or embedding.text_hash != text_hash:
            stale_cards += 1
        else:
            embedded_cards += 1

    return CardEmbeddingStatus(
        total_cards=len(cards),
        embedded_cards=embedded_cards,
        missing_cards=missing_cards,
        stale_cards=stale_cards,
        model=model_name,
        dimension=dimension,
    )


def _status_dimension_for_cards(
    cards: Sequence[KnowledgeCard],
) -> int | None:
    for card in cards:
        embedding = get_card_embedding_info(card.id)

        if embedding is not None:
            return embedding.dimension

    return None


def _create_default_embedder() -> TextEmbedder:
    return SentenceTransformerEmbedder()


def _embedder_model_name(embedder: TextEmbedder) -> str:
    model_name = getattr(embedder, "model_name", None)

    if isinstance(model_name, str) and model_name.strip():
        return model_name.strip()

    return get_embedding_settings().model
