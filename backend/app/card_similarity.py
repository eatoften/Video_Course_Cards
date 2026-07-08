from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from .card_embedding import CardEmbedding
from .embedding import cosine_similarity


@dataclass(frozen=True)
class CardSimilarityCandidate:
    source_card_id: str
    target_card_id: str
    score: float
    model: str
    dimension: int


def rank_related_embeddings(
    source_embedding: CardEmbedding,
    candidate_embeddings: Sequence[CardEmbedding],
    *,
    top_k: int,
    threshold: float,
) -> list[CardSimilarityCandidate]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between -1.0 and 1.0.")

    candidates: list[CardSimilarityCandidate] = []

    for candidate_embedding in candidate_embeddings:
        if candidate_embedding.card_id == source_embedding.card_id:
            continue

        if candidate_embedding.model != source_embedding.model:
            continue

        if candidate_embedding.dimension != source_embedding.dimension:
            continue

        try:
            score = cosine_similarity(
                source_embedding.vector,
                candidate_embedding.vector,
            )
        except ValueError:
            continue

        if score < threshold:
            continue

        candidates.append(
            CardSimilarityCandidate(
                source_card_id=source_embedding.card_id,
                target_card_id=candidate_embedding.card_id,
                score=score,
                model=source_embedding.model,
                dimension=source_embedding.dimension,
            )
        )

    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate.score,
            candidate.target_card_id,
        ),
    )[:top_k]


def build_similarity_candidates(
    embeddings: Sequence[CardEmbedding],
    *,
    top_k: int,
    threshold: float,
) -> list[CardSimilarityCandidate]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between -1.0 and 1.0.")

    relation_candidates: list[CardSimilarityCandidate] = []

    for source_embedding in embeddings:
        relation_candidates.extend(
            rank_related_embeddings(
                source_embedding,
                embeddings,
                top_k=top_k,
                threshold=threshold,
            )
        )

    return relation_candidates
