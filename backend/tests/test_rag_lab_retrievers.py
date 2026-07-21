from __future__ import annotations

from rag_lab.retrievers import (
    Bm25Index,
    graph_rerank_one_hop,
    rank_dense,
    reciprocal_rank_fusion,
)
from rag_lab.schemas import RagCorpusCard, RagCorpusClaim, RagCorpusEvidence, RagCorpusRelation


def _card(card_id: str, text: str) -> RagCorpusCard:
    evidence = RagCorpusEvidence(
        evidence_id=f"e-{card_id}",
        quote=text,
        start_seconds=1,
        end_seconds=2,
    )
    return RagCorpusCard(
        card_id=card_id,
        job_id="job",
        lecture_name="lecture.mp4",
        title=text,
        summary=text,
        document_text=text,
        content_status="reviewed",
        source_start_seconds=1,
        source_end_seconds=2,
        claims=[
            RagCorpusClaim(
                claim_id=f"c-{card_id}",
                text=text,
                evidence=[evidence],
            )
        ],
    )


def test_bm25_ranks_lexical_match_first() -> None:
    index = Bm25Index.build(
        [_card("svd", "orthogonal matrix factorization"), _card("cnn", "image convolution")]
    )
    ranked = index.rank("orthogonal factorization")
    assert ranked[0].card_id == "svd"
    assert ranked[0].score > ranked[1].score


def test_dense_and_rrf_return_deterministic_rankings() -> None:
    dense = rank_dense([1.0, 0.0], {"a": [1.0, 0.0], "b": [0.0, 1.0]})
    lexical = Bm25Index.build([_card("a", "matrix"), _card("b", "vision")]).rank("vision")
    fused = reciprocal_rank_fusion([dense, lexical], rrf_k=60)
    assert [item.card_id for item in dense] == ["a", "b"]
    assert {item.card_id for item in fused} == {"a", "b"}
    assert all(item.retrieval_source == "hybrid_rrf" for item in fused)


def test_graph_rerank_boosts_one_hop_neighbor_under_same_budget() -> None:
    dense = rank_dense(
        [1.0, 0.0],
        {
            "anchor": [1.0, 0.0],
            "distractor": [0.8, 0.2],
            "neighbor": [0.2, 0.8],
        },
    )
    relation = RagCorpusRelation(
        relation_id="r1",
        source_card_id="anchor",
        target_card_id="neighbor",
        relation_type="related",
        score=1.0,
        method="manual",
        status="accepted",
    )
    reranked = graph_rerank_one_hop(
        dense,
        [relation],
        seed_k=1,
        graph_weight=1.0,
        top_k=2,
        source="dense_graph_trusted",
    )
    assert [item.card_id for item in reranked] == ["neighbor", "anchor"]
    assert len(reranked) == 2

