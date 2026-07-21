from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import TypeAdapter

from app.embedding import SentenceTransformerEmbedder
from app.llm_client import LLMClientError, LocalLLMClient

from .answering import (
    GroundedAnswerOutputError,
    build_grounded_context,
    evaluate_grounded_answers,
    generate_grounded_answer,
)
from .benchmark import audit_benchmark
from .io import load_model, sha256_file, write_json_atomic
from .reviews import audit_annotation_review, require_formal_human_review
from .schemas import (
    RagAnnotationReview,
    RagAnswerExperimentProtocol,
    RagBenchmarkDataset,
    RagCorpusSnapshot,
    RagGroundedAnswerRecord,
    RetrievalRecord,
)


RETRIEVAL_RECORDS = TypeAdapter(list[RetrievalRecord])
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run offline fixed-budget grounded-answer generation."
    )
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--retrieval-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument("--reuse-answer-run-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    corpus = load_model(args.corpus.resolve(), RagCorpusSnapshot)
    benchmark = load_model(args.benchmark.resolve(), RagBenchmarkDataset)
    review = load_model(args.review.resolve(), RagAnnotationReview)
    protocol = load_model(args.protocol.resolve(), RagAnswerExperimentProtocol)
    retrieval_run_dir = args.retrieval_run_dir.resolve()
    _preflight(corpus, benchmark, review, protocol, retrieval_run_dir)

    if args.resume_run_dir:
        run_dir = args.resume_run_dir.resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {run_dir}")
        _validate_resume_manifest(run_dir, protocol)
    else:
        run_id = datetime.now(UTC).strftime("rag-r3-%Y%m%dT%H%M%SZ")
        run_dir = args.output_dir.resolve() / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json_atomic(
            run_dir / "manifest.json",
            {
                "run_id": run_id,
                "started_at": datetime.now(UTC).isoformat(),
                "protocol": protocol.model_dump(mode="json"),
                "status": "running",
            },
        )

    items = [item for item in benchmark.items if item.split == protocol.split]
    items_by_id = {item.question_id: item for item in items}
    if args.reuse_answer_run_dir:
        _validate_reuse_manifest(args.reuse_answer_run_dir.resolve(), protocol)
    llm_client = LocalLLMClient()
    all_answers: dict[str, list[RagGroundedAnswerRecord]] = {}
    for system in protocol.systems:
        retrieval_records = _load_retrieval_records(
            retrieval_run_dir / f"{system}_records.json"
        )
        retrieval_records = [
            record for record in retrieval_records if record.question_id in items_by_id
        ]
        if {record.question_id for record in retrieval_records} != set(items_by_id):
            raise ValueError(f"Retrieval records are incomplete for {system}.")
        answer_path = run_dir / f"{system}_answers.jsonl"
        existing = _load_answer_jsonl(answer_path)
        answers_by_id = {record.question_id: record for record in existing}
        reusable_by_id = {}
        if args.reuse_answer_run_dir:
            reusable_path = (
                args.reuse_answer_run_dir.resolve() / f"{system}_answers.jsonl"
            )
            reusable_by_id = {
                record.question_id: record
                for record in _load_answer_jsonl(reusable_path)
            }
        with answer_path.open("a", encoding="utf-8") as handle:
            for index, retrieval in enumerate(retrieval_records, start=1):
                if retrieval.question_id in answers_by_id:
                    continue
                item = items_by_id[retrieval.question_id]
                context, context_characters = build_grounded_context(
                    corpus,
                    retrieval,
                    top_k=protocol.top_k,
                    character_budget=protocol.context_budget_characters,
                )
                started = time.perf_counter()
                raw_output = None
                threshold = protocol.confidence_thresholds.get(system)
                if (
                    threshold is not None
                    and retrieval.confidence_score is not None
                    and retrieval.confidence_score < threshold
                ):
                    record = RagGroundedAnswerRecord(
                        question_id=item.question_id,
                        category=item.category,
                        split=item.split,
                        system=system,
                        context_card_ids=[str(card["card_id"]) for card in context],
                        context_characters=context_characters,
                        answerable_prediction=False,
                        answer="Not enough evidence in the course cards.",
                        claims=[],
                        latency_milliseconds=0.0,
                    )
                    disposition = "gated"
                elif item.question_id in reusable_by_id:
                    reusable = reusable_by_id[item.question_id]
                    context_ids = [str(card["card_id"]) for card in context]
                    if reusable.context_card_ids != context_ids:
                        raise ValueError(
                            f"Reusable context changed for {system}/{item.question_id}."
                        )
                    record = reusable.model_copy(deep=True)
                    disposition = "reused"
                else:
                    disposition = "generated"
                    record = None
                try:
                    if record is None:
                        payload, raw_output = generate_grounded_answer(
                            item.question,
                            context,
                            llm_client=llm_client,
                            model=protocol.model,
                            temperature=protocol.temperature,
                            max_tokens=protocol.max_tokens,
                        )
                        record = RagGroundedAnswerRecord(
                            question_id=item.question_id,
                            category=item.category,
                            split=item.split,
                            system=system,
                            context_card_ids=[str(card["card_id"]) for card in context],
                            context_characters=context_characters,
                            answerable_prediction=payload.answerable,
                            answer=payload.answer,
                            claims=payload.claims,
                            latency_milliseconds=(time.perf_counter() - started) * 1000,
                            raw_output=raw_output,
                        )
                except (GroundedAnswerOutputError, LLMClientError) as exc:
                    if isinstance(exc, GroundedAnswerOutputError):
                        raw_output = exc.raw_output
                    record = RagGroundedAnswerRecord(
                        question_id=item.question_id,
                        category=item.category,
                        split=item.split,
                        system=system,
                        context_card_ids=[str(card["card_id"]) for card in context],
                        context_characters=context_characters,
                        latency_milliseconds=(time.perf_counter() - started) * 1000,
                        generation_error=str(exc),
                        raw_output=raw_output,
                    )
                handle.write(record.model_dump_json() + "\n")
                handle.flush()
                answers_by_id[item.question_id] = record
                print(
                    f"[{system}] {index}/{len(retrieval_records)} "
                    f"{item.question_id} "
                    f"{disposition if record.generation_error is None else 'failed'}",
                    flush=True,
                )
        all_answers[system] = [
            answers_by_id[item.question_id]
            for item in items
        ]

    semantic_embedder = SentenceTransformerEmbedder(
        model_name=protocol.semantic_evaluation_model,
        normalize_embeddings=True,
    )
    evaluations = {}
    for system in protocol.systems:
        retrieval_records = _load_retrieval_records(
            retrieval_run_dir / f"{system}_records.json"
        )
        retrieval_records = [
            record for record in retrieval_records if record.question_id in items_by_id
        ]
        evaluations[system] = evaluate_grounded_answers(
            items,
            retrieval_records,
            all_answers[system],
            semantic_embedder=semantic_embedder,
        )

    report = {
        "schema_version": "1.0",
        "run_id": run_dir.name,
        "phase": "exploratory_development",
        "warning": (
            "The benchmark is a model-assisted candidate pending independent human "
            "review. Semantic correctness and claim entailment remain human-evaluation "
            "fields, not established automatic metrics."
        ),
        "protocol": protocol.model_dump(mode="json"),
        "question_count": len(items),
        "evaluations": evaluations,
    }
    write_json_atomic(run_dir / "grounded_answer_report.json", report)
    write_json_atomic(
        run_dir / "manifest.json",
        {
            "run_id": run_dir.name,
            "completed_at": datetime.now(UTC).isoformat(),
            "protocol": protocol.model_dump(mode="json"),
            "status": "completed",
        },
    )
    review_key = _write_human_review_sheet(
        run_dir / "answer_human_review.md",
        items,
        all_answers,
    )
    write_json_atomic(run_dir / "answer_human_review_key.json", review_key)
    print(json.dumps({"run_dir": str(run_dir), "evaluations": evaluations}, indent=2))
    return 0


def _preflight(
    corpus: RagCorpusSnapshot,
    benchmark: RagBenchmarkDataset,
    review: RagAnnotationReview,
    protocol: RagAnswerExperimentProtocol,
    retrieval_run_dir: Path,
) -> None:
    audit_benchmark(benchmark, corpus, require_accepted=False)
    audit_annotation_review(review, corpus)
    if protocol.corpus_sha256 != corpus.snapshot_sha256:
        raise ValueError("Answer protocol corpus hash mismatch.")
    if protocol.benchmark_sha256 != benchmark.dataset_sha256:
        raise ValueError("Answer protocol benchmark hash mismatch.")
    if protocol.review_sha256 != review.review_sha256:
        raise ValueError("Answer protocol review hash mismatch.")
    report_path = retrieval_run_dir / "retrieval_report.json"
    if sha256_file(report_path) != protocol.retrieval_report_file_sha256:
        raise ValueError("Retrieval report file hash mismatch.")
    if protocol.split == "test":
        try:
            audit_benchmark(benchmark, corpus, require_accepted=True)
            require_formal_human_review(benchmark, review)
        except ValueError as exc:
            raise ValueError(
                "Test answer generation is blocked before complete human verification."
            ) from exc


def _validate_resume_manifest(
    run_dir: Path,
    protocol: RagAnswerExperimentProtocol,
) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Resume directory has no experiment manifest.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != protocol.model_dump(mode="json"):
        raise ValueError("Resume protocol does not match the original run.")


def _validate_reuse_manifest(
    run_dir: Path,
    protocol: RagAnswerExperimentProtocol,
) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Reusable answer directory has no manifest.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = RagAnswerExperimentProtocol.model_validate(manifest.get("protocol"))
    if manifest.get("status") != "completed":
        raise ValueError("Reusable answer run is not complete.")
    invariant_fields = (
        "corpus_sha256",
        "benchmark_sha256",
        "review_sha256",
        "split",
        "model",
        "model_digest",
        "temperature",
        "max_tokens",
        "top_k",
        "context_budget_characters",
        "prompt_version",
    )
    mismatched = [
        field
        for field in invariant_fields
        if getattr(source, field) != getattr(protocol, field)
    ]
    if mismatched:
        raise ValueError(f"Reusable answer protocol changed fields: {mismatched}")
    if not set(protocol.systems).issubset(source.systems):
        raise ValueError("Reusable answer run does not contain every requested system.")


def _load_retrieval_records(path: Path) -> list[RetrievalRecord]:
    return RETRIEVAL_RECORDS.validate_json(path.read_text(encoding="utf-8"))


def _load_answer_jsonl(path: Path) -> list[RagGroundedAnswerRecord]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(RagGroundedAnswerRecord.model_validate_json(line))
    return records


def _write_human_review_sheet(
    path: Path,
    items,
    all_answers: dict[str, list[RagGroundedAnswerRecord]],
) -> dict[str, dict[str, str]]:
    answers = {
        (system, record.question_id): record
        for system, records in all_answers.items()
        for record in records
    }
    lines = [
        "# R3 Grounded Answer Blind Review",
        "",
        "Score correctness and claim support independently of automatic metrics.",
        "Suggested labels: correctness 0/1/2; support 0/1/2; citation correct yes/no.",
        "",
    ]
    review_key: dict[str, dict[str, str]] = {}
    for item in items:
        lines.extend(
            [
                f"## {item.question_id}",
                "",
                f"Question: {item.question}",
                f"Reference: {item.reference_answer or 'UNANSWERABLE'}",
                "",
            ]
        )
        systems = sorted(all_answers)
        if int(hashlib.sha256(item.question_id.encode()).hexdigest(), 16) % 2:
            systems.reverse()
        review_key[item.question_id] = {}
        for response_index, system in enumerate(systems):
            response_label = chr(ord("A") + response_index)
            review_key[item.question_id][response_label] = system
            record = answers[(system, item.question_id)]
            lines.extend(
                [
                    f"### Response {response_label}",
                    "",
                    f"Answer: {record.answer or record.generation_error}",
                    "- Correctness (0/1/2):",
                    "- Claim support (0/1/2):",
                    "- Citation correct (yes/no):",
                    "- Notes:",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")
    return review_key


if __name__ == "__main__":
    raise SystemExit(main())
