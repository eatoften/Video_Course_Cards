from __future__ import annotations

import argparse
import json
import random
import statistics
from collections.abc import Sequence
from pathlib import Path

from pydantic import TypeAdapter

from .io import load_model, sha256_file, write_json_atomic
from .schemas import (
    RagBenchmarkDataset,
    RagBenchmarkItem,
    RagGroundedAnswerRecord,
    RetrievalRecord,
)


RETRIEVAL_RECORDS = TypeAdapter(list[RetrievalRecord])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare dense and one-hop graph RAG question by question."
    )
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--retrieval-run-dir", required=True, type=Path)
    parser.add_argument("--answer-run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    benchmark = load_model(args.benchmark.resolve(), RagBenchmarkDataset)
    items = [item for item in benchmark.items if item.split == "development"]
    retrieval_dir = args.retrieval_run_dir.resolve()
    answer_dir = args.answer_run_dir.resolve()
    dense_retrieval_path = retrieval_dir / "dense_records.json"
    graph_retrieval_path = retrieval_dir / "dense_graph_trusted_records.json"
    dense_answers_path = answer_dir / "dense_answers.jsonl"
    graph_answers_path = answer_dir / "dense_graph_trusted_answers.jsonl"
    dense_retrieval = _load_retrieval(dense_retrieval_path)
    graph_retrieval = _load_retrieval(graph_retrieval_path)
    dense_answers = _load_answers(dense_answers_path)
    graph_answers = _load_answers(graph_answers_path)
    _check_alignment(items, dense_retrieval, graph_retrieval, dense_answers, graph_answers)

    dense_retrieval_by_id = {record.question_id: record for record in dense_retrieval}
    graph_retrieval_by_id = {record.question_id: record for record in graph_retrieval}
    dense_answers_by_id = {record.question_id: record for record in dense_answers}
    graph_answers_by_id = {record.question_id: record for record in graph_answers}
    per_question = []
    for item in items:
        dense_answer = dense_answers_by_id[item.question_id]
        graph_answer = graph_answers_by_id[item.question_id]
        dense_claim_recall = _gold_claim_recall(item, dense_answer)
        graph_claim_recall = _gold_claim_recall(item, graph_answer)
        dense_citation_precision = _gold_citation_precision(item, dense_answer)
        graph_citation_precision = _gold_citation_precision(item, graph_answer)
        dense_context = dense_retrieval_by_id[item.question_id].ranked_cards
        graph_context = graph_retrieval_by_id[item.question_id].ranked_cards
        per_question.append(
            {
                "question_id": item.question_id,
                "category": item.category,
                "answerable": item.answerable,
                "dense_gold_claim_citation_recall": dense_claim_recall,
                "graph_gold_claim_citation_recall": graph_claim_recall,
                "claim_recall_difference": graph_claim_recall - dense_claim_recall,
                "dense_citation_precision": dense_citation_precision,
                "graph_citation_precision": graph_citation_precision,
                "citation_precision_difference": (
                    graph_citation_precision - dense_citation_precision
                ),
                "dense_answerable_prediction": dense_answer.answerable_prediction,
                "graph_answerable_prediction": graph_answer.answerable_prediction,
                "top_5_order_changed": [card.card_id for card in dense_context]
                != [card.card_id for card in graph_context],
                "top_5_set_changed": {card.card_id for card in dense_context}
                != {card.card_id for card in graph_context},
                "answer_changed": dense_answer.answer != graph_answer.answer,
            }
        )

    slices = {
        "all_answerable": [row for row in per_question if row["answerable"]],
        "single_card": [
            row
            for row in per_question
            if row["category"] in {"factual", "concept"}
        ],
        "comparison": [row for row in per_question if row["category"] == "comparison"],
        "multi_hop": [row for row in per_question if row["category"] == "multi_hop"],
        "unanswerable": [
            row for row in per_question if row["category"] == "unanswerable"
        ],
    }
    slice_reports = {
        name: _slice_report(
            rows,
            iterations=args.iterations,
            seed=args.seed + index * 10,
        )
        for index, (name, rows) in enumerate(slices.items())
    }
    report = {
        "schema_version": "1.0",
        "study": "R4 dense versus accepted-one-hop graph expansion",
        "benchmark_sha256": benchmark.dataset_sha256,
        "input_file_sha256": {
            "benchmark": sha256_file(args.benchmark.resolve()),
            "dense_retrieval": sha256_file(dense_retrieval_path),
            "graph_retrieval": sha256_file(graph_retrieval_path),
            "dense_answers": sha256_file(dense_answers_path),
            "graph_answers": sha256_file(graph_answers_path),
        },
        "dense_system": "dense",
        "graph_system": "dense_graph_trusted",
        "same_top_k": 5,
        "same_generation_prompt": True,
        "same_context_character_budget": True,
        "slice_reports": slice_reports,
        "top_5_order_changed_count": sum(
            bool(row["top_5_order_changed"]) for row in per_question
        ),
        "top_5_set_changed_count": sum(
            bool(row["top_5_set_changed"]) for row in per_question
        ),
        "answer_changed_count": sum(bool(row["answer_changed"]) for row in per_question),
        "claim_recall_wins": sum(
            row["claim_recall_difference"] > 0 for row in per_question
        ),
        "claim_recall_ties": sum(
            row["claim_recall_difference"] == 0 for row in per_question
        ),
        "claim_recall_losses": sum(
            row["claim_recall_difference"] < 0 for row in per_question
        ),
        "improved_questions": [
            row for row in per_question if row["claim_recall_difference"] > 0
        ],
        "harmed_questions": [
            row for row in per_question if row["claim_recall_difference"] < 0
        ],
        "per_question": per_question,
        "limitations": [
            "Development questions and candidate graph edges await independent human review.",
            "Thresholds were selected and evaluated on the same development split.",
            "Only eight development multi-hop questions are available.",
            "Citation recall is not equivalent to semantic answer correctness.",
        ],
    }
    write_json_atomic(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "claim_recall_wins": report["claim_recall_wins"],
                "claim_recall_ties": report["claim_recall_ties"],
                "claim_recall_losses": report["claim_recall_losses"],
                "slice_reports": slice_reports,
            },
            indent=2,
        )
    )
    return 0


def _slice_report(
    rows: list[dict[str, object]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    claim_differences = [float(row["claim_recall_difference"]) for row in rows]
    citation_differences = [
        float(row["citation_precision_difference"]) for row in rows
    ]
    return {
        "question_count": len(rows),
        "gold_claim_citation_recall": _bootstrap_difference(
            claim_differences,
            iterations=iterations,
            seed=seed,
        ),
        "citation_precision": _bootstrap_difference(
            citation_differences,
            iterations=iterations,
            seed=seed + 1,
        ),
    }


def _bootstrap_difference(
    differences: list[float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    if not differences:
        return {
            "observed_difference": None,
            "confidence_interval_95": None,
            "bootstrap_probability_graph_better": None,
        }
    generator = random.Random(seed)
    samples = sorted(
        statistics.fmean(
            differences[generator.randrange(len(differences))]
            for _ in differences
        )
        for _ in range(iterations)
    )
    lower = samples[round((iterations - 1) * 0.025)]
    upper = samples[round((iterations - 1) * 0.975)]
    return {
        "observed_difference": statistics.fmean(differences),
        "confidence_interval_95": [lower, upper],
        "bootstrap_probability_graph_better": (
            sum(value > 0 for value in samples) / iterations
        ),
    }


def _gold_claim_recall(
    item: RagBenchmarkItem,
    answer: RagGroundedAnswerRecord,
) -> float:
    if not item.answerable:
        return 0.0
    cited = {
        citation.claim_id
        for claim in answer.claims
        for citation in claim.citations
    }
    return len(cited.intersection(item.gold_claim_ids)) / len(item.gold_claim_ids)


def _gold_citation_precision(
    item: RagBenchmarkItem,
    answer: RagGroundedAnswerRecord,
) -> float:
    cited = {
        (citation.card_id, citation.claim_id, citation.evidence_id)
        for claim in answer.claims
        for citation in claim.citations
    }
    if not cited:
        return 1.0 if not item.answerable else 0.0
    gold = {
        (evidence.card_id, evidence.claim_id, evidence.evidence_id)
        for evidence in item.evidence
    }
    return len(cited.intersection(gold)) / len(cited)


def _load_retrieval(path: Path) -> list[RetrievalRecord]:
    return RETRIEVAL_RECORDS.validate_json(path.read_text(encoding="utf-8"))


def _load_answers(path: Path) -> list[RagGroundedAnswerRecord]:
    return [
        RagGroundedAnswerRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _check_alignment(
    items: Sequence[RagBenchmarkItem],
    *record_groups: Sequence[object],
) -> None:
    expected = {item.question_id for item in items}
    for records in record_groups:
        observed = {record.question_id for record in records}
        if observed != expected:
            raise ValueError("R4 inputs do not align to the development benchmark.")


if __name__ == "__main__":
    raise SystemExit(main())
