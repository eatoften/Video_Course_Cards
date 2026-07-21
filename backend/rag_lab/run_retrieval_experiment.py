from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from app.embedding import SentenceTransformerEmbedder

from .benchmark import audit_benchmark
from .io import load_model, sha256_value, write_json_atomic, write_model_atomic
from .metrics import (
    evaluate_retrieval_system,
    paired_bootstrap_metric_difference,
    select_confidence_threshold,
)
from .retrievers import (
    Bm25Index,
    graph_rerank_one_hop,
    rank_dense,
    reciprocal_rank_fusion,
)
from .reviews import audit_annotation_review, require_formal_human_review
from .schemas import (
    RagAnnotationReview,
    RagBenchmarkDataset,
    RagBenchmarkSplit,
    RagCorpusRelation,
    RagCorpusSnapshot,
    RagEmbeddingRecord,
    RagEmbeddingSnapshot,
    RagExperimentProtocol,
    RetrievalRecord,
    RetrievalSystemName,
)


SYSTEMS: tuple[RetrievalSystemName, ...] = (
    "bm25",
    "dense",
    "hybrid_rrf",
    "dense_graph_noisy",
    "dense_graph_trusted",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run controlled card-retrieval baselines on a frozen split."
    )
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--split",
        choices=["development", "test"],
        default="development",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    corpus = load_model(args.corpus.resolve(), RagCorpusSnapshot)
    benchmark = load_model(args.benchmark.resolve(), RagBenchmarkDataset)
    review = load_model(args.review.resolve(), RagAnnotationReview)
    protocol = load_model(args.protocol.resolve(), RagExperimentProtocol)
    _preflight(corpus, benchmark, review, protocol, split=args.split)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("rag-r2-%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    embedder = SentenceTransformerEmbedder(
        model_name=protocol.embedding_model,
        normalize_embeddings=True,
    )
    indexing_started = time.perf_counter()
    vectors = embedder.embed_texts([card.document_text for card in corpus.cards])
    indexing_ms = (time.perf_counter() - indexing_started) * 1000
    if not vectors or len(vectors) != len(corpus.cards):
        raise ValueError("Card embedding count does not match the frozen corpus.")
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("Card embeddings do not share one dimension.")
    records = [
        RagEmbeddingRecord(card_id=card.card_id, vector=vector)
        for card, vector in zip(corpus.cards, vectors)
    ]
    embeddings_hash = sha256_value(
        [record.model_dump(mode="json") for record in records]
    )
    embedding_snapshot = RagEmbeddingSnapshot(
        corpus_sha256=corpus.snapshot_sha256,
        model=protocol.embedding_model,
        dimension=dimension,
        normalized=True,
        indexing_milliseconds=indexing_ms,
        records=records,
        embeddings_sha256=embeddings_hash,
    )
    write_model_atomic(run_dir / "card_embeddings.json", embedding_snapshot)

    bm25 = Bm25Index.build(
        corpus.cards,
        k1=protocol.bm25_k1,
        b=protocol.bm25_b,
    )
    card_vectors = {record.card_id: record.vector for record in records}
    noisy_relations = _deduplicate_relations(corpus.relations)
    trusted_relations = _trusted_relations(review)
    items = [item for item in benchmark.items if item.split == args.split]
    if not items:
        raise ValueError(f"Benchmark has no {args.split} items.")

    embedder.embed_texts(["warm up the local query encoder"])
    records_by_system: dict[RetrievalSystemName, list[RetrievalRecord]] = {
        system: [] for system in SYSTEMS
    }
    maximum_k = max(protocol.top_k_values)
    for item in items:
        bm25_started = time.perf_counter()
        bm25_ranking = bm25.rank(item.question)
        bm25_ms = (time.perf_counter() - bm25_started) * 1000
        records_by_system["bm25"].append(
            _record(
                item,
                "bm25",
                bm25_ms,
                0.0,
                bm25_ms,
                bm25_ranking[:maximum_k],
                confidence_score=bm25_ranking[0].score,
            )
        )

        encoding_started = time.perf_counter()
        query_vector = embedder.embed_texts([item.question])[0]
        encoding_ms = (time.perf_counter() - encoding_started) * 1000

        dense_started = time.perf_counter()
        dense_ranking = rank_dense(query_vector, card_vectors)
        dense_ms = (time.perf_counter() - dense_started) * 1000
        records_by_system["dense"].append(
            _record(
                item,
                "dense",
                encoding_ms + dense_ms,
                encoding_ms,
                dense_ms,
                dense_ranking[:maximum_k],
                confidence_score=dense_ranking[0].score,
            )
        )

        hybrid_started = time.perf_counter()
        hybrid_dense = rank_dense(query_vector, card_vectors)
        hybrid_bm25 = bm25.rank(item.question)
        hybrid = reciprocal_rank_fusion(
            [hybrid_bm25, hybrid_dense],
            rrf_k=protocol.rrf_k,
        )
        hybrid_ms = (time.perf_counter() - hybrid_started) * 1000
        records_by_system["hybrid_rrf"].append(
            _record(
                item,
                "hybrid_rrf",
                encoding_ms + hybrid_ms,
                encoding_ms,
                hybrid_ms,
                hybrid[:maximum_k],
                confidence_score=hybrid[0].score,
            )
        )

        for system, relations in (
            ("dense_graph_noisy", noisy_relations),
            ("dense_graph_trusted", trusted_relations),
        ):
            graph_started = time.perf_counter()
            graph_dense = rank_dense(query_vector, card_vectors)
            graph_ranking = graph_rerank_one_hop(
                graph_dense,
                relations,
                seed_k=protocol.graph_seed_k,
                graph_weight=protocol.graph_weight,
                source=system,
            )
            graph_ms = (time.perf_counter() - graph_started) * 1000
            records_by_system[system].append(
                _record(
                    item,
                    system,
                    encoding_ms + graph_ms,
                    encoding_ms,
                    graph_ms,
                    graph_ranking[:maximum_k],
                    confidence_score=(
                        graph_dense[0].score
                        if protocol.confidence_score_policy
                        == "dense_anchor_for_graph"
                        else graph_ranking[0].score
                    ),
                )
            )

    reports = {}
    thresholds = {}
    for system, system_records in records_by_system.items():
        threshold = select_confidence_threshold(items, system_records)
        thresholds[system] = threshold
        reports[system] = evaluate_retrieval_system(
            items,
            system_records,
            top_k_values=protocol.top_k_values,
            confidence_threshold=threshold,
        ).model_dump(mode="json")
        write_json_atomic(
            run_dir / f"{system}_records.json",
            [record.model_dump(mode="json") for record in system_records],
        )

    answerable_items = [item for item in items if item.answerable]
    multi_hop_items = [item for item in items if item.category == "multi_hop"]
    single_card_items = [
        item for item in items if item.category in {"factual", "concept"}
    ]
    bootstrap_comparisons = {}
    for system_index, system in enumerate(SYSTEMS):
        if system == "dense":
            continue
        bootstrap_comparisons[f"{system}_minus_dense"] = {
            "overall_ndcg_at_5": paired_bootstrap_metric_difference(
                answerable_items,
                _records_for_items(records_by_system[system], answerable_items),
                _records_for_items(records_by_system["dense"], answerable_items),
                metric="ndcg",
                k=5,
                iterations=protocol.bootstrap_iterations,
                seed=protocol.bootstrap_seed + system_index * 10,
            ),
            "multi_hop_joint_recall_at_3": paired_bootstrap_metric_difference(
                multi_hop_items,
                _records_for_items(records_by_system[system], multi_hop_items),
                _records_for_items(records_by_system["dense"], multi_hop_items),
                metric="joint_recall",
                k=3,
                iterations=protocol.bootstrap_iterations,
                seed=protocol.bootstrap_seed + system_index * 10 + 1,
            ),
            "single_card_ndcg_at_5": paired_bootstrap_metric_difference(
                single_card_items,
                _records_for_items(records_by_system[system], single_card_items),
                _records_for_items(records_by_system["dense"], single_card_items),
                metric="ndcg",
                k=5,
                iterations=protocol.bootstrap_iterations,
                seed=protocol.bootstrap_seed + system_index * 10 + 2,
            ),
        }

    report = {
        "schema_version": "1.0",
        "run_id": run_id,
        "phase": (
            "exploratory_development"
            if args.split == "development"
            else "confirmatory_test"
        ),
        "split": args.split,
        "warning": (
            "Candidate benchmark has not received independent human review. "
            "Development metrics are exploratory and must not be reported as final."
            if benchmark.confirmatory_status == "pending_human_review"
            else None
        ),
        "protocol": protocol.model_dump(mode="json"),
        "corpus_sha256": corpus.snapshot_sha256,
        "benchmark_sha256": benchmark.dataset_sha256,
        "review_sha256": review.review_sha256,
        "embedding_snapshot_sha256": embeddings_hash,
        "embedding_dimension": dimension,
        "card_count": len(corpus.cards),
        "question_count": len(items),
        "noisy_undirected_edge_count": len(noisy_relations),
        "trusted_directed_edge_count": len(trusted_relations),
        "thresholds_selected_on_this_split": thresholds,
        "reports": reports,
        "paired_bootstrap_comparisons": bootstrap_comparisons,
        "latency_note": (
            "Card indexing is offline. Dense-system elapsed time includes one shared "
            "query encoding plus system-specific ranking. Latencies are indicative "
            "single-run wall-clock measurements, not a dedicated throughput benchmark."
        ),
        "card_indexing_milliseconds": indexing_ms,
        "environment": _environment_provenance(),
    }
    write_json_atomic(run_dir / "retrieval_report.json", report)
    print(json.dumps({"run_dir": str(run_dir), **_compact_summary(report)}, indent=2))
    return 0


def _preflight(
    corpus: RagCorpusSnapshot,
    benchmark: RagBenchmarkDataset,
    review: RagAnnotationReview,
    protocol: RagExperimentProtocol,
    *,
    split: RagBenchmarkSplit,
) -> None:
    audit_benchmark(benchmark, corpus, require_accepted=False)
    audit_annotation_review(review, corpus)
    if protocol.corpus_sha256 != corpus.snapshot_sha256:
        raise ValueError("Protocol corpus hash mismatch.")
    if protocol.benchmark_sha256 != benchmark.dataset_sha256:
        raise ValueError("Protocol benchmark hash mismatch.")
    if protocol.trusted_graph_sha256 != review.review_sha256:
        raise ValueError("Protocol trusted-graph review hash mismatch.")
    if split == "test":
        try:
            audit_benchmark(benchmark, corpus, require_accepted=True)
            require_formal_human_review(benchmark, review)
        except ValueError as exc:
            raise ValueError(
                "Test access is blocked until every question and gold claim is "
                "independently human verified."
            ) from exc


def _record(
    item,
    system: RetrievalSystemName,
    elapsed_ms: float,
    encoding_ms: float,
    ranking_ms: float,
    ranking,
    *,
    confidence_score: float,
) -> RetrievalRecord:
    return RetrievalRecord(
        question_id=item.question_id,
        category=item.category,
        split=item.split,
        system=system,
        elapsed_milliseconds=elapsed_ms,
        query_encoding_milliseconds=encoding_ms,
        ranking_milliseconds=ranking_ms,
        confidence_score=confidence_score,
        ranked_cards=ranking,
    )


def _records_for_items(
    records: Sequence[RetrievalRecord],
    items,
) -> list[RetrievalRecord]:
    allowed = {item.question_id for item in items}
    return [record for record in records if record.question_id in allowed]


def _deduplicate_relations(
    relations: Sequence[RagCorpusRelation],
) -> list[RagCorpusRelation]:
    by_pair: dict[tuple[str, str], RagCorpusRelation] = {}
    for relation in relations:
        pair = tuple(sorted((relation.source_card_id, relation.target_card_id)))
        current = by_pair.get(pair)
        if current is None or relation.score > current.score:
            by_pair[pair] = relation.model_copy(
                update={"source_card_id": pair[0], "target_card_id": pair[1]}
            )
    return [by_pair[pair] for pair in sorted(by_pair)]


def _trusted_relations(review: RagAnnotationReview) -> list[RagCorpusRelation]:
    return [
        RagCorpusRelation(
            relation_id=f"reviewed-{index:03d}",
            source_card_id=decision.source_card_id,
            target_card_id=decision.target_card_id,
            relation_type=decision.relation_type or "related",
            score=1.0,
            method="candidate_review",
            status="accepted",
            explanation=decision.review_notes,
        )
        for index, decision in enumerate(review.graph_decisions, start=1)
        if decision.accepted
    ]


def _environment_provenance() -> dict[str, object]:
    commit = _git_output("rev-parse", "HEAD")
    dirty = bool(_git_output("status", "--porcelain"))
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": commit,
        "git_dirty": dirty,
    }


def _git_output(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _compact_summary(report: dict[str, object]) -> dict[str, object]:
    reports = report["reports"]
    assert isinstance(reports, dict)
    return {
        "phase": report["phase"],
        "question_count": report["question_count"],
        "recall_at_5": {
            system: _metric_at_k(values["overall"]["set_recall_at_k"], 5)
            for system, values in reports.items()
        },
        "multi_hop_joint_recall_at_5": {
            system: _metric_at_k(
                values["by_category"]["multi_hop"]["joint_recall_at_k"],
                5,
            )
            for system, values in reports.items()
        },
        "median_latency_ms": {
            system: values["overall"]["median_latency_milliseconds"]
            for system, values in reports.items()
        },
    }


def _metric_at_k(values: dict[object, float], k: int) -> float:
    if k in values:
        return values[k]
    return values[str(k)]


if __name__ == "__main__":
    raise SystemExit(main())
