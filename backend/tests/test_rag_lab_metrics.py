from __future__ import annotations

from rag_lab.metrics import (
    evaluate_retrieval_system,
    paired_bootstrap_metric_difference,
    select_confidence_threshold,
)
from rag_lab.schemas import RagBenchmarkItem, RankedCard, RetrievalRecord


def _item(question_id: str, *, answerable: bool, category: str = "factual") -> RagBenchmarkItem:
    if not answerable:
        return RagBenchmarkItem(
            question_id=question_id,
            category="unanswerable",
            split="development",
            question="An unsupported question?",
            answerable=False,
            authoring_method="manual",
            review_status="accepted",
        )
    return RagBenchmarkItem(
        question_id=question_id,
        category=category,
        split="development",
        question="A supported question?",
        answerable=True,
        reference_answer="answer",
        gold_card_ids=["gold"],
        gold_claim_ids=["claim"],
        evidence=[
            {
                "card_id": "gold",
                "claim_id": "claim",
                "evidence_id": "evidence",
                "quote": "answer",
                "start_seconds": 1,
                "end_seconds": 2,
            }
        ],
        authoring_method="manual",
        review_status="accepted",
    )


def _record(question_id: str, ranked: list[tuple[str, float]], *, category: str) -> RetrievalRecord:
    return RetrievalRecord(
        question_id=question_id,
        category=category,
        split="development",
        system="dense",
        elapsed_milliseconds=2,
        ranked_cards=[
            RankedCard(
                card_id=card_id,
                rank=index,
                score=score,
                retrieval_source="dense",
            )
            for index, (card_id, score) in enumerate(ranked, start=1)
        ],
    )


def test_metrics_score_answerable_and_unanswerable_queries() -> None:
    items = [_item("q1", answerable=True), _item("q2", answerable=False)]
    records = [
        _record("q1", [("gold", 0.9), ("other", 0.2)], category="factual"),
        _record("q2", [("other", 0.1)], category="unanswerable"),
    ]
    threshold = select_confidence_threshold(items, records)
    report = evaluate_retrieval_system(
        items,
        records,
        top_k_values=[1, 2],
        confidence_threshold=threshold,
    )
    assert report.overall.hit_rate_at_k[1] == 1.0
    assert report.overall.joint_recall_at_k[1] == 1.0
    assert report.overall.mean_reciprocal_rank == 1.0
    assert report.overall.unanswerable_false_retrieval_rate == 0.0
    assert report.overall.answerability_f1 == 1.0


def test_paired_bootstrap_reports_direction_of_improvement() -> None:
    items = [_item("q1", answerable=True), _item("q2", answerable=True)]
    better = [
        _record("q1", [("gold", 0.9)], category="factual"),
        _record("q2", [("gold", 0.9)], category="factual"),
    ]
    worse = [
        _record("q1", [("other", 0.9)], category="factual"),
        _record("q2", [("other", 0.9)], category="factual"),
    ]

    result = paired_bootstrap_metric_difference(
        items,
        better,
        worse,
        metric="joint_recall",
        k=1,
        iterations=100,
        seed=7,
    )

    assert result["observed_difference"] == 1.0
    assert result["confidence_interval_95"] == [1.0, 1.0]


def test_answerability_uses_explicit_confidence_not_rerank_score() -> None:
    items = [_item("q1", answerable=True), _item("q2", answerable=False)]
    records = [
        _record("q1", [("gold", 0.02)], category="factual").model_copy(
            update={"confidence_score": 0.8}
        ),
        _record("q2", [("other", 0.02)], category="unanswerable").model_copy(
            update={"confidence_score": 0.1}
        ),
    ]

    threshold = select_confidence_threshold(items, records)
    report = evaluate_retrieval_system(
        items,
        records,
        top_k_values=[1],
        confidence_threshold=threshold,
    )

    assert report.overall.answerability_f1 == 1.0
    assert report.overall.unanswerable_false_retrieval_rate == 0.0
