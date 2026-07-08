import pytest

from app.card_embedding import CardEmbedding
from app.card_similarity import (
    build_similarity_candidates,
    rank_related_embeddings,
)


def make_embedding(
    card_id: str,
    vector: list[float],
    *,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> CardEmbedding:
    return CardEmbedding(
        card_id=card_id,
        model=model,
        dimension=len(vector),
        text_hash=f"hash-{card_id}",
        vector=vector,
    )


def test_rank_related_embeddings_skips_self_and_ranks_by_score():
    results = rank_related_embeddings(
        make_embedding("card-a", [1.0, 0.0]),
        [
            make_embedding("card-a", [1.0, 0.0]),
            make_embedding("card-b", [0.9, 0.1]),
            make_embedding("card-c", [0.5, 0.5]),
            make_embedding("card-d", [0.0, 1.0]),
        ],
        top_k=2,
        threshold=0.5,
    )

    assert [
        result.target_card_id
        for result in results
    ] == [
        "card-b",
        "card-c",
    ]
    assert results[0].score == pytest.approx(0.993883, abs=1e-6)


def test_rank_related_embeddings_requires_same_model_and_dimension():
    source = make_embedding("card-a", [1.0, 0.0])

    results = rank_related_embeddings(
        source,
        [
            make_embedding("card-b", [0.9, 0.1]),
            make_embedding("card-c", [0.9, 0.1], model="other-model"),
            make_embedding("card-d", [0.9, 0.1, 0.0]),
        ],
        top_k=5,
        threshold=0.0,
    )

    assert [
        result.target_card_id
        for result in results
    ] == ["card-b"]


def test_build_similarity_candidates_builds_directional_edges():
    candidates = build_similarity_candidates(
        [
            make_embedding("card-a", [1.0, 0.0]),
            make_embedding("card-b", [0.9, 0.1]),
            make_embedding("card-c", [0.0, 1.0]),
        ],
        top_k=1,
        threshold=0.7,
    )

    assert {
        (candidate.source_card_id, candidate.target_card_id)
        for candidate in candidates
    } == {
        ("card-a", "card-b"),
        ("card-b", "card-a"),
    }


def test_similarity_input_validation():
    with pytest.raises(ValueError, match="top_k"):
        build_similarity_candidates([], top_k=0, threshold=0.7)

    with pytest.raises(ValueError, match="threshold"):
        build_similarity_candidates([], top_k=5, threshold=1.1)
