from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from app.embedding import cosine_similarity

from .schemas import RagCorpusCard, RagCorpusRelation, RankedCard


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*", re.IGNORECASE)


def tokenize_for_retrieval(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


@dataclass(frozen=True)
class Bm25Index:
    card_ids: tuple[str, ...]
    term_frequencies: tuple[Counter[str], ...]
    document_lengths: tuple[int, ...]
    inverse_document_frequencies: dict[str, float]
    average_document_length: float
    k1: float
    b: float

    @classmethod
    def build(
        cls,
        cards: Sequence[RagCorpusCard],
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> "Bm25Index":
        if not cards:
            raise ValueError("BM25 requires at least one card.")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 parameters.")
        frequencies = tuple(Counter(tokenize_for_retrieval(card.document_text)) for card in cards)
        lengths = tuple(sum(items.values()) for items in frequencies)
        document_frequency: Counter[str] = Counter()
        for items in frequencies:
            document_frequency.update(items.keys())
        document_count = len(cards)
        idf = {
            term: math.log(1.0 + (document_count - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }
        return cls(
            card_ids=tuple(card.card_id for card in cards),
            term_frequencies=frequencies,
            document_lengths=lengths,
            inverse_document_frequencies=idf,
            average_document_length=sum(lengths) / document_count,
            k1=k1,
            b=b,
        )

    def rank(self, query: str, *, top_k: int | None = None) -> list[RankedCard]:
        query_terms = tokenize_for_retrieval(query)
        scores = []
        for card_id, frequencies, length in zip(
            self.card_ids,
            self.term_frequencies,
            self.document_lengths,
        ):
            score = 0.0
            normalization = self.k1 * (
                1.0 - self.b
                + self.b * length / max(self.average_document_length, 1e-12)
            )
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                score += self.inverse_document_frequencies.get(term, 0.0) * (
                    frequency * (self.k1 + 1.0) / (frequency + normalization)
                )
            scores.append((card_id, score))
        return _to_ranked_cards(scores, source="bm25", top_k=top_k)


def rank_dense(
    query_vector: Sequence[float],
    card_vectors: Mapping[str, Sequence[float]],
    *,
    top_k: int | None = None,
) -> list[RankedCard]:
    if not query_vector:
        raise ValueError("Dense query vector cannot be empty.")
    scores = [
        (card_id, cosine_similarity(query_vector, vector))
        for card_id, vector in card_vectors.items()
    ]
    return _to_ranked_cards(scores, source="dense", top_k=top_k)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RankedCard]],
    *,
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[RankedCard]:
    if not rankings:
        raise ValueError("RRF requires at least one ranking.")
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive.")
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[str] = set()
        for item in ranking:
            if item.card_id in seen:
                raise ValueError("One ranking cannot contain duplicate cards.")
            seen.add(item.card_id)
            scores[item.card_id] += 1.0 / (rrf_k + item.rank)
    return _to_ranked_cards(scores.items(), source="hybrid_rrf", top_k=top_k)


def graph_rerank_one_hop(
    dense_ranking: Sequence[RankedCard],
    relations: Sequence[RagCorpusRelation],
    *,
    seed_k: int,
    graph_weight: float,
    top_k: int | None = None,
    source: str = "dense_graph",
) -> list[RankedCard]:
    if seed_k < 1:
        raise ValueError("seed_k must be positive.")
    if graph_weight < 0:
        raise ValueError("graph_weight cannot be negative.")
    if not dense_ranking:
        return []

    dense_by_id = {item.card_id: item for item in dense_ranking}
    scores = {
        item.card_id: 1.0 / (60 + item.rank)
        for item in dense_ranking
    }
    seed_ids = {item.card_id for item in dense_ranking[:seed_k]}
    adjacency: defaultdict[str, list[tuple[str, float]]] = defaultdict(list)
    for relation in relations:
        edge_score = max(0.0, relation.score)
        adjacency[relation.source_card_id].append((relation.target_card_id, edge_score))
        adjacency[relation.target_card_id].append((relation.source_card_id, edge_score))

    for seed_id in seed_ids:
        seed = dense_by_id[seed_id]
        seed_score = 1.0 / (60 + seed.rank)
        for neighbor_id, edge_score in adjacency.get(seed_id, []):
            if neighbor_id not in scores:
                continue
            scores[neighbor_id] += graph_weight * seed_score * edge_score
    return _to_ranked_cards(scores.items(), source=source, top_k=top_k)


def _to_ranked_cards(
    scores: Iterable[tuple[str, float]],
    *,
    source: str,
    top_k: int | None,
) -> list[RankedCard]:
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive when provided.")
    ordered = sorted(scores, key=lambda item: (-float(item[1]), str(item[0])))
    if top_k is not None:
        ordered = ordered[:top_k]
    return [
        RankedCard(
            card_id=str(card_id),
            rank=index,
            score=float(score),
            retrieval_source=source,
        )
        for index, (card_id, score) in enumerate(ordered, start=1)
    ]
