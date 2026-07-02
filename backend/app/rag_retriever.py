from __future__ import annotations

from collections.abc import Sequence

from .card_embedding import CardEmbedding
from .embedding import EmbeddingVector, cosine_similarity
from .knowledge_card import KnowledgeCard
from .rag import RetrievedCard


CardEmbeddingCandidate = tuple[KnowledgeCard, CardEmbedding]


def rank_cards_by_similarity(
    query_vector: EmbeddingVector,
    candidates: Sequence[CardEmbeddingCandidate],
    *,
    top_k: int,
    min_score: float | None = None,
) -> list[RetrievedCard]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if min_score is not None and not -1.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between -1.0 and 1.0.")

    if not query_vector:
        raise ValueError("Query vector cannot be empty.")

    ranked_cards: list[RetrievedCard] = []

    for card, embedding in candidates:
        if card.id != embedding.card_id:
            raise ValueError(
                "Card and embedding ids must match."
            )

        score = cosine_similarity(query_vector, embedding.vector)

        if min_score is not None and score < min_score:
            continue

        ranked_cards.append(
            RetrievedCard(
                card_id=card.id,
                job_id=card.job_id,
                title=card.title,
                summary=card.summary,
                score=score,
                source_start_seconds=card.source_start_seconds,
                source_end_seconds=card.source_end_seconds,
                key_points=card.key_points,
                claims=card.claims,
                tags=card.tags,
            )
        )

    return sorted(
        ranked_cards,
        key=lambda card: card.score,
        reverse=True,
    )[:top_k]
