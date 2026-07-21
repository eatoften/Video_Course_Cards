from __future__ import annotations

import pytest

from rag_lab.graph_organization import analyze_graph_organization
from rag_lab.io import sha256_value
from rag_lab.reviews import review_payload_sha256
from rag_lab.schemas import (
    RagAnnotationReview,
    RagCorpusCard,
    RagCorpusClaim,
    RagCorpusEvidence,
    RagCorpusSnapshot,
    RagEmbeddingRecord,
    RagEmbeddingSnapshot,
    RagGraphDecision,
)


def _fixtures():
    corpus_hash = "a" * 64
    cards = [
        _card("a", "lecture-1", "A"),
        _card("b", "lecture-1", "B"),
        _card("c", "lecture-2", "C"),
        _card("d", "lecture-2", "D"),
    ]
    corpus = RagCorpusSnapshot(
        snapshot_id="snapshot",
        course_id="course",
        source_database_sha256="b" * 64,
        snapshot_sha256=corpus_hash,
        cards=cards,
    )
    review = RagAnnotationReview(
        review_id="review",
        corpus_sha256=corpus_hash,
        graph_decisions=[
            RagGraphDecision(
                source_card_id="a",
                target_card_id="b",
                accepted=True,
                relation_type="related",
                reviewer_id="candidate",
                review_notes="Same-lecture relation.",
            ),
            RagGraphDecision(
                source_card_id="b",
                target_card_id="c",
                accepted=True,
                relation_type="prerequisite",
                reviewer_id="candidate",
                review_notes="Cross-lecture relation.",
            ),
        ],
        review_sha256="0" * 64,
    )
    review.review_sha256 = review_payload_sha256(review)
    records = [
        RagEmbeddingRecord(card_id="a", vector=[1.0, 0.0]),
        RagEmbeddingRecord(card_id="b", vector=[0.9, 0.1]),
        RagEmbeddingRecord(card_id="c", vector=[0.1, 0.9]),
        RagEmbeddingRecord(card_id="d", vector=[0.0, 1.0]),
    ]
    embeddings = RagEmbeddingSnapshot(
        corpus_sha256=corpus_hash,
        model="test-model",
        dimension=2,
        indexing_milliseconds=1,
        records=records,
        embeddings_sha256=sha256_value(
            [record.model_dump(mode="json") for record in records]
        ),
    )
    return corpus, review, embeddings


def _card(card_id: str, job_id: str, title: str) -> RagCorpusCard:
    return RagCorpusCard(
        card_id=card_id,
        job_id=job_id,
        lecture_name=f"{job_id}.mp4",
        title=title,
        summary=title,
        document_text=title,
        content_status="draft",
        source_start_seconds=1,
        source_end_seconds=2,
        claims=[
            RagCorpusClaim(
                claim_id=f"claim-{card_id}",
                text=f"Claim {title}",
                evidence=[
                    RagCorpusEvidence(
                        evidence_id=f"evidence-{card_id}",
                        quote=f"Claim {title}",
                        start_seconds=1,
                        end_seconds=2,
                    )
                ],
            )
        ],
    )


def test_graph_organization_audit_measures_coverage_and_matched_null() -> None:
    corpus, review, embeddings = _fixtures()

    report = analyze_graph_organization(
        corpus,
        review,
        embeddings,
        neighborhood_k=1,
        random_samples=100,
        seed=7,
    )

    assert report["structure"]["accepted_edge_count"] == 2
    assert report["structure"]["covered_node_count"] == 3
    assert report["structure"]["isolated_node_count"] == 1
    assert report["structure"]["edge_component_sizes"] == [3]
    assert report["structure"]["same_lecture_edge_count"] == 1
    assert report["structure"]["cross_lecture_edge_count"] == 1
    assert report["association"]["directed_endpoint_count"] == 4
    assert report["association"]["matched_random_nonedge"]["mean"] >= 0


def test_graph_organization_audit_rejects_incomplete_embeddings() -> None:
    corpus, review, embeddings = _fixtures()
    embeddings.records.pop()

    with pytest.raises(ValueError, match="every corpus card"):
        analyze_graph_organization(
            corpus,
            review,
            embeddings,
            random_samples=100,
        )


def test_graph_organization_audit_rejects_modified_embedding_snapshot() -> None:
    corpus, review, embeddings = _fixtures()
    embeddings.records[0].vector = [0.0, 1.0]

    with pytest.raises(ValueError, match="hash is not canonical"):
        analyze_graph_organization(
            corpus,
            review,
            embeddings,
            random_samples=100,
        )
