from __future__ import annotations

import random
import statistics
from collections import Counter, defaultdict
from itertools import combinations

from app.embedding import cosine_similarity

from .io import sha256_value
from .reviews import audit_annotation_review
from .schemas import (
    RagAnnotationReview,
    RagCorpusSnapshot,
    RagEmbeddingSnapshot,
    RagGraphDecision,
)


def analyze_graph_organization(
    corpus: RagCorpusSnapshot,
    review: RagAnnotationReview,
    embeddings: RagEmbeddingSnapshot,
    *,
    neighborhood_k: int = 5,
    random_samples: int = 5000,
    seed: int = 20260721,
) -> dict[str, object]:
    """Measure graph coverage and associations not captured by dense neighbors."""
    if neighborhood_k < 1:
        raise ValueError("neighborhood_k must be positive.")
    if random_samples < 100:
        raise ValueError("random_samples must be at least 100.")
    audit_annotation_review(review, corpus)
    if embeddings.corpus_sha256 != corpus.snapshot_sha256:
        raise ValueError("Embedding snapshot does not match the corpus.")

    cards = {card.card_id: card for card in corpus.cards}
    if len(cards) != len(corpus.cards):
        raise ValueError("Corpus contains duplicate card ids.")
    vectors = {record.card_id: record.vector for record in embeddings.records}
    if len(vectors) != len(embeddings.records):
        raise ValueError("Embedding snapshot contains duplicate card ids.")
    if set(vectors) != set(cards):
        raise ValueError("Embedding snapshot must contain every corpus card exactly once.")
    if any(len(vector) != embeddings.dimension for vector in vectors.values()):
        raise ValueError("Embedding vector dimension does not match the snapshot.")
    expected_embeddings_sha256 = sha256_value(
        [record.model_dump(mode="json") for record in embeddings.records]
    )
    if embeddings.embeddings_sha256 != expected_embeddings_sha256:
        raise ValueError("Embedding snapshot hash is not canonical.")

    accepted = [decision for decision in review.graph_decisions if decision.accepted]
    if not accepted:
        raise ValueError("Graph organization audit requires accepted edges.")
    edge_pairs = {_pair(decision.source_card_id, decision.target_card_id) for decision in accepted}
    if len(edge_pairs) != len(accepted):
        raise ValueError("Accepted graph contains duplicate undirected edges.")

    pair_similarities = {
        (source_id, target_id): cosine_similarity(
            vectors[source_id],
            vectors[target_id],
        )
        for source_id, target_id in combinations(sorted(cards), 2)
    }
    adjacency: defaultdict[str, set[str]] = defaultdict(set)
    for source_id, target_id in edge_pairs:
        adjacency[source_id].add(target_id)
        adjacency[target_id].add(source_id)

    covered_ids = set(adjacency)
    components = _connected_components(covered_ids, adjacency)
    dense_ranks = _dense_neighbor_ranks(covered_ids, cards, pair_similarities)
    edge_rows = [
        _edge_row(decision, cards, pair_similarities, dense_ranks)
        for decision in accepted
    ]
    edge_rows.sort(
        key=lambda row: (
            -max(
                int(row["source_to_target_dense_rank"]),
                int(row["target_to_source_dense_rank"]),
            ),
            str(row["source_card_id"]),
            str(row["target_card_id"]),
        )
    )
    directed_ranks = [
        rank
        for row in edge_rows
        for rank in (
            int(row["source_to_target_dense_rank"]),
            int(row["target_to_source_dense_rank"]),
        )
    ]
    edge_cosines = [float(row["cosine_similarity"]) for row in edge_rows]
    matched_null = _matched_random_nonedge_baseline(
        cards,
        pair_similarities,
        edge_pairs,
        same_lecture_count=sum(bool(row["same_lecture"]) for row in edge_rows),
        cross_lecture_count=sum(not bool(row["same_lecture"]) for row in edge_rows),
        observed_mean=statistics.fmean(edge_cosines),
        random_samples=random_samples,
        seed=seed,
    )
    isolated_count = len(cards) - len(covered_ids)
    endpoints_outside = sum(rank > neighborhood_k for rank in directed_ranks)
    edges_with_nonlocal_endpoint = sum(
        max(
            int(row["source_to_target_dense_rank"]),
            int(row["target_to_source_dense_rank"]),
        )
        > neighborhood_k
        for row in edge_rows
    )
    relation_counts = Counter(decision.relation_type for decision in accepted)

    return {
        "schema_version": "1.0",
        "study": "Card graph as an associative knowledge structure",
        "status": "exploratory_candidate_graph",
        "corpus_sha256": corpus.snapshot_sha256,
        "review_sha256": review.review_sha256,
        "embeddings_sha256": embeddings.embeddings_sha256,
        "configuration": {
            "neighborhood_k": neighborhood_k,
            "matched_random_samples": random_samples,
            "random_seed": seed,
            "lecture_matching": "same-versus-cross-lecture edge counts",
        },
        "structure": {
            "card_count": len(cards),
            "accepted_edge_count": len(accepted),
            "covered_node_count": len(covered_ids),
            "node_coverage": len(covered_ids) / len(cards),
            "isolated_node_count": isolated_count,
            "edge_component_count": len(components),
            "component_count_including_isolates": len(components) + isolated_count,
            "edge_component_sizes": [len(component) for component in components],
            "largest_edge_component_size": len(components[0]),
            "relation_type_counts": dict(sorted(relation_counts.items())),
            "same_lecture_edge_count": sum(
                bool(row["same_lecture"]) for row in edge_rows
            ),
            "cross_lecture_edge_count": sum(
                not bool(row["same_lecture"]) for row in edge_rows
            ),
        },
        "association": {
            "accepted_edge_cosine_mean": statistics.fmean(edge_cosines),
            "matched_random_nonedge": matched_null,
            "directed_endpoint_count": len(directed_ranks),
            "dense_neighbor_rank_mean": statistics.fmean(directed_ranks),
            "dense_neighbor_rank_median": statistics.median(directed_ranks),
            "endpoints_outside_dense_top_k": endpoints_outside,
            "endpoints_outside_dense_top_k_rate": endpoints_outside / len(directed_ranks),
            "edges_with_nonlocal_endpoint": edges_with_nonlocal_endpoint,
            "edges_with_nonlocal_endpoint_rate": edges_with_nonlocal_endpoint / len(edge_rows),
        },
        "largest_component_examples": [
            {
                "size": len(component),
                "cards": [
                    {
                        "card_id": card_id,
                        "title": cards[card_id].title,
                        "lecture_name": cards[card_id].lecture_name,
                    }
                    for card_id in component
                ],
            }
            for component in components[:5]
        ],
        "nonlocal_association_examples": [
            row
            for row in edge_rows
            if max(
                int(row["source_to_target_dense_rank"]),
                int(row["target_to_source_dense_rank"]),
            )
            > neighborhood_k
        ][:10],
        "limitations": [
            "The graph decisions are model-assisted candidates, not independent human labels.",
            "Graph coverage is sparse, so this audit cannot establish large-scale behavior.",
            "Dense-neighbor novelty measures structural difference, not educational usefulness.",
            "The matched random baseline controls lecture locality but not concept frequency or degree.",
        ],
    }


def _matched_random_nonedge_baseline(
    cards,
    pair_similarities: dict[tuple[str, str], float],
    edge_pairs: set[tuple[str, str]],
    *,
    same_lecture_count: int,
    cross_lecture_count: int,
    observed_mean: float,
    random_samples: int,
    seed: int,
) -> dict[str, object]:
    same_lecture_pool = []
    cross_lecture_pool = []
    for pair, similarity in pair_similarities.items():
        if pair in edge_pairs:
            continue
        source_id, target_id = pair
        pool = (
            same_lecture_pool
            if cards[source_id].job_id == cards[target_id].job_id
            else cross_lecture_pool
        )
        pool.append(similarity)
    if len(same_lecture_pool) < same_lecture_count:
        raise ValueError("Not enough same-lecture non-edges for the matched baseline.")
    if len(cross_lecture_pool) < cross_lecture_count:
        raise ValueError("Not enough cross-lecture non-edges for the matched baseline.")

    generator = random.Random(seed)
    sample_means = []
    for _ in range(random_samples):
        sampled = generator.sample(same_lecture_pool, same_lecture_count)
        sampled.extend(generator.sample(cross_lecture_pool, cross_lecture_count))
        sample_means.append(statistics.fmean(sampled))
    ordered = sorted(sample_means)
    return {
        "mean": statistics.fmean(sample_means),
        "confidence_interval_95": [
            ordered[round((random_samples - 1) * 0.025)],
            ordered[round((random_samples - 1) * 0.975)],
        ],
        "accepted_minus_random_mean": observed_mean - statistics.fmean(sample_means),
        "monte_carlo_probability_random_at_least_observed": (
            1 + sum(value >= observed_mean for value in sample_means)
        )
        / (random_samples + 1),
    }


def _connected_components(
    covered_ids: set[str],
    adjacency: dict[str, set[str]],
) -> list[list[str]]:
    seen: set[str] = set()
    components = []
    for root in sorted(covered_ids):
        if root in seen:
            continue
        stack = [root]
        seen.add(root)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda component: (-len(component), component))


def _dense_neighbor_ranks(
    source_ids: set[str],
    cards,
    pair_similarities: dict[tuple[str, str], float],
) -> dict[str, dict[str, int]]:
    ranks = {}
    for source_id in sorted(source_ids):
        ranked_targets = sorted(
            (target_id for target_id in cards if target_id != source_id),
            key=lambda target_id: (
                -pair_similarities[_pair(source_id, target_id)],
                target_id,
            ),
        )
        ranks[source_id] = {
            target_id: rank
            for rank, target_id in enumerate(ranked_targets, start=1)
        }
    return ranks


def _edge_row(
    decision: RagGraphDecision,
    cards,
    pair_similarities: dict[tuple[str, str], float],
    dense_ranks: dict[str, dict[str, int]],
) -> dict[str, object]:
    source = cards[decision.source_card_id]
    target = cards[decision.target_card_id]
    return {
        "source_card_id": source.card_id,
        "source_title": source.title,
        "source_lecture": source.lecture_name,
        "target_card_id": target.card_id,
        "target_title": target.title,
        "target_lecture": target.lecture_name,
        "relation_type": decision.relation_type,
        "same_lecture": source.job_id == target.job_id,
        "cosine_similarity": pair_similarities[_pair(source.card_id, target.card_id)],
        "source_to_target_dense_rank": dense_ranks[source.card_id][target.card_id],
        "target_to_source_dense_rank": dense_ranks[target.card_id][source.card_id],
    }


def _pair(source_id: str, target_id: str) -> tuple[str, str]:
    return tuple(sorted((source_id, target_id)))
