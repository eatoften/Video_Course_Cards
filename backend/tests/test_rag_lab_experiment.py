from pathlib import Path

import pytest

from rag_lab.io import write_json_atomic
from rag_lab.reviews import require_formal_human_review
from rag_lab.run_grounded_answer_experiment import (
    _validate_resume_manifest,
    _validate_reuse_manifest,
)
from rag_lab.run_retrieval_experiment import (
    _deduplicate_relations,
    _preflight,
    _trusted_relations,
)
from rag_lab.schemas import (
    RagAnnotationReview,
    RagAnswerExperimentProtocol,
    RagBenchmarkDataset,
    RagBenchmarkItem,
    RagCorpusCard,
    RagCorpusClaim,
    RagCorpusEvidence,
    RagCorpusRelation,
    RagCorpusSnapshot,
    RagExperimentProtocol,
    RagGraphDecision,
)
from rag_lab.benchmark import benchmark_payload_sha256
from rag_lab.reviews import review_payload_sha256


def _fixtures():
    corpus_hash = "a" * 64
    card = RagCorpusCard(
        card_id="card-a",
        job_id="job",
        lecture_name="lecture.mp4",
        title="A",
        summary="A",
        document_text="A",
        content_status="draft",
        source_start_seconds=1,
        source_end_seconds=2,
        claims=[
            RagCorpusClaim(
                claim_id="claim-a",
                text="A fact.",
                evidence=[
                    RagCorpusEvidence(
                        evidence_id="evidence-a",
                        quote="A fact.",
                        start_seconds=1,
                        end_seconds=2,
                    )
                ],
            )
        ],
    )
    corpus = RagCorpusSnapshot(
        snapshot_id="corpus",
        course_id="course",
        source_database_sha256="b" * 64,
        snapshot_sha256=corpus_hash,
        cards=[card],
    )
    benchmark = RagBenchmarkDataset(
        benchmark_id="benchmark",
        course_id="course",
        corpus_sha256=corpus_hash,
        annotation_method="candidate",
        confirmatory_status="pending_human_review",
        dataset_sha256="0" * 64,
        items=[
            RagBenchmarkItem(
                question_id="q",
                category="factual",
                split="development",
                question="What is the fact?",
                answerable=True,
                reference_answer="A fact.",
                gold_card_ids=["card-a"],
                gold_claim_ids=["claim-a"],
                evidence=[
                    {
                        "card_id": "card-a",
                        "claim_id": "claim-a",
                        "evidence_id": "evidence-a",
                        "quote": "A fact.",
                        "start_seconds": 1,
                        "end_seconds": 2,
                    }
                ],
                authoring_method="manual",
                review_status="pending",
            )
        ],
    )
    benchmark.dataset_sha256 = benchmark_payload_sha256(benchmark)
    review = RagAnnotationReview(
        review_id="review",
        corpus_sha256=corpus_hash,
        review_status="candidate",
        review_sha256="0" * 64,
    )
    review.review_sha256 = review_payload_sha256(review)
    protocol = RagExperimentProtocol(
        protocol_id="protocol",
        corpus_path="corpus.json",
        corpus_sha256=corpus_hash,
        benchmark_path="benchmark.json",
        benchmark_sha256=benchmark.dataset_sha256,
        trusted_graph_path="review.json",
        trusted_graph_sha256=review.review_sha256,
        embedding_model="model",
    )
    return corpus, benchmark, review, protocol


def test_test_split_is_blocked_before_human_verification() -> None:
    corpus, benchmark, review, protocol = _fixtures()

    with pytest.raises(ValueError, match="Test access is blocked"):
        _preflight(corpus, benchmark, review, protocol, split="test")

    _preflight(corpus, benchmark, review, protocol, split="development")


def test_graph_builders_remove_reciprocal_duplicates_and_keep_reviewed_edges() -> None:
    forward = RagCorpusRelation(
        relation_id="forward",
        source_card_id="a",
        target_card_id="b",
        relation_type="related",
        score=0.7,
        method="cosine",
        status="suggested",
    )
    backward = forward.model_copy(
        update={
            "relation_id": "backward",
            "source_card_id": "b",
            "target_card_id": "a",
            "score": 0.8,
        }
    )
    deduplicated = _deduplicate_relations([forward, backward])
    assert len(deduplicated) == 1
    assert deduplicated[0].score == 0.8

    review = RagAnnotationReview(
        review_id="review",
        corpus_sha256="a" * 64,
        graph_decisions=[
            RagGraphDecision(
                source_card_id="a",
                target_card_id="b",
                accepted=True,
                relation_type="related",
                reviewer_id="human",
                review_notes="Valid edge.",
            )
        ],
        review_sha256="b" * 64,
    )
    trusted = _trusted_relations(review)
    assert len(trusted) == 1
    assert trusted[0].status == "accepted"
    assert trusted[0].score == 1.0


def test_formal_review_rejects_sealed_benchmark_with_unverified_gold_claims() -> None:
    _, benchmark, review, _ = _fixtures()
    benchmark.confirmatory_status = "sealed"
    benchmark.items[0].review_status = "accepted"
    review.review_status = "human_verified"

    with pytest.raises(ValueError, match="unverified gold claims"):
        require_formal_human_review(benchmark, review)


def test_answer_run_protocol_guards_reject_changed_generation_settings(
    tmp_path: Path,
) -> None:
    protocol = RagAnswerExperimentProtocol(
        protocol_id="answer-protocol",
        corpus_sha256="a" * 64,
        benchmark_sha256="b" * 64,
        review_sha256="c" * 64,
        retrieval_report_path="retrieval_report.json",
        retrieval_report_file_sha256="d" * 64,
        systems=["dense", "dense_graph_trusted"],
        model="qwen3:4b",
        model_digest="e" * 64,
        prompt_version="claim-only-v1",
        semantic_evaluation_model="all-MiniLM-L6-v2",
    )
    write_json_atomic(
        tmp_path / "manifest.json",
        {
            "status": "completed",
            "protocol": protocol.model_dump(mode="json"),
        },
    )

    _validate_resume_manifest(tmp_path, protocol)
    _validate_reuse_manifest(tmp_path, protocol)

    changed = protocol.model_copy(update={"prompt_version": "claim-only-v2"})
    with pytest.raises(ValueError, match="Resume protocol"):
        _validate_resume_manifest(tmp_path, changed)
    with pytest.raises(ValueError, match="prompt_version"):
        _validate_reuse_manifest(tmp_path, changed)
