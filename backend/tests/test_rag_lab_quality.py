from rag_lab.quality import audit_benchmark_quality
from rag_lab.schemas import RagBenchmarkDataset, RagBenchmarkItem, RagCorpusSnapshot


def test_quality_audit_flags_answer_copy_and_unsupported_why() -> None:
    corpus = RagCorpusSnapshot(
        snapshot_id="snapshot",
        course_id="course",
        source_database_sha256="a" * 64,
        snapshot_sha256="b" * 64,
        cards=[
            {
                "card_id": "card-a",
                "job_id": "job",
                "lecture_name": "lecture.mp4",
                "title": "A",
                "summary": "A",
                "document_text": "A",
                "content_status": "draft",
                "source_start_seconds": 1,
                "source_end_seconds": 2,
                "claims": [
                    {
                        "claim_id": "claim-a",
                        "text": "Max pooling acts as a nonlinearity.",
                        "evidence": [
                            {
                                "evidence_id": "evidence-a",
                                "quote": "Max pooling acts as a nonlinearity.",
                                "start_seconds": 1,
                                "end_seconds": 2,
                            }
                        ],
                    }
                ],
            }
        ],
    )
    dataset = RagBenchmarkDataset(
        benchmark_id="benchmark",
        course_id="course",
        corpus_sha256="b" * 64,
        annotation_method="candidate",
        dataset_sha256="c" * 64,
        items=[
            RagBenchmarkItem(
                question_id="concept",
                category="concept",
                split="development",
                question="Why is max pooling nonlinear?",
                answerable=True,
                reference_answer="Max pooling acts as a nonlinearity.",
                gold_card_ids=["card-a"],
                gold_claim_ids=["claim-a"],
                evidence=[
                    {
                        "card_id": "card-a",
                        "claim_id": "claim-a",
                        "evidence_id": "evidence-a",
                        "quote": "Max pooling acts as a nonlinearity.",
                        "start_seconds": 1,
                        "end_seconds": 2,
                    }
                ],
                authoring_method="model_assisted",
            )
        ],
    )

    report = audit_benchmark_quality(dataset, corpus)

    assert report["flag_counts"]["unsupported_why_shape"] == 1
