import pytest

from app.card_embedding import CardEmbedding
from app.knowledge_card import (
    KnowledgeCard,
    KnowledgeCardClaim,
    KnowledgeCardEvidence,
)
from app.rag_retriever import rank_cards_by_similarity


def make_card(
    card_id: str,
    *,
    title: str,
    start_seconds: float = 0.0,
) -> KnowledgeCard:
    return KnowledgeCard(
        id=card_id,
        job_id="job-1",
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


def make_embedding(
    card_id: str,
    vector: list[float],
) -> CardEmbedding:
    return CardEmbedding(
        card_id=card_id,
        model="sentence-transformers/all-MiniLM-L6-v2",
        dimension=len(vector),
        text_hash=f"hash-{card_id}",
        vector=vector,
    )


def test_rank_cards_by_similarity_returns_closest_cards_first():
    first_card = make_card("card-a", title="Linear Algebra")
    second_card = make_card("card-b", title="Deep Learning")
    third_card = make_card("card-c", title="Statistics")

    results = rank_cards_by_similarity(
        [1.0, 0.0],
        [
            (second_card, make_embedding(second_card.id, [0.0, 1.0])),
            (first_card, make_embedding(first_card.id, [0.9, 0.1])),
            (third_card, make_embedding(third_card.id, [0.5, 0.5])),
        ],
        top_k=3,
    )

    assert [
        result.card_id
        for result in results
    ] == [
        "card-a",
        "card-c",
        "card-b",
    ]
    assert results[0].title == "Linear Algebra"
    assert results[0].summary == "Linear Algebra summary."
    assert results[0].key_points == ["Linear Algebra key point"]
    assert results[0].claims == first_card.claims
    assert results[0].tags == ["linear algebra"]
    assert results[0].score == pytest.approx(0.993883, abs=1e-6)


def test_rank_cards_by_similarity_applies_top_k_and_min_score():
    first_card = make_card("card-a", title="Linear Algebra")
    second_card = make_card("card-b", title="Deep Learning")
    third_card = make_card("card-c", title="Statistics")

    results = rank_cards_by_similarity(
        [1.0, 0.0],
        [
            (first_card, make_embedding(first_card.id, [1.0, 0.0])),
            (second_card, make_embedding(second_card.id, [0.8, 0.2])),
            (third_card, make_embedding(third_card.id, [0.0, 1.0])),
        ],
        top_k=1,
        min_score=0.9,
    )

    assert [
        result.card_id
        for result in results
    ] == ["card-a"]


def test_rank_cards_by_similarity_returns_empty_list_for_empty_candidates():
    assert rank_cards_by_similarity(
        [1.0, 0.0],
        [],
        top_k=5,
    ) == []


def test_rank_cards_by_similarity_validates_inputs():
    card = make_card("card-a", title="Linear Algebra")

    with pytest.raises(ValueError, match="top_k"):
        rank_cards_by_similarity(
            [1.0, 0.0],
            [(card, make_embedding(card.id, [1.0, 0.0]))],
            top_k=0,
        )

    with pytest.raises(ValueError, match="min_score"):
        rank_cards_by_similarity(
            [1.0, 0.0],
            [(card, make_embedding(card.id, [1.0, 0.0]))],
            top_k=5,
            min_score=1.01,
        )

    with pytest.raises(ValueError, match="Query vector"):
        rank_cards_by_similarity(
            [],
            [(card, make_embedding(card.id, [1.0, 0.0]))],
            top_k=5,
        )

    with pytest.raises(ValueError, match="ids must match"):
        rank_cards_by_similarity(
            [1.0, 0.0],
            [(card, make_embedding("different-card", [1.0, 0.0]))],
            top_k=5,
        )

    with pytest.raises(ValueError, match="same dimension"):
        rank_cards_by_similarity(
            [1.0, 0.0],
            [(card, make_embedding(card.id, [1.0, 0.0, 0.0]))],
            top_k=5,
        )
