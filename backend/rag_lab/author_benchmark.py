from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from app.llm_client import LocalLLMClient

from .authoring import author_benchmark
from .benchmark import audit_benchmark
from .io import load_model, write_model_atomic
from .reviews import audit_annotation_review
from .schemas import RagBenchmarkSeed, RagCorpusSnapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Author a provenance-locked RAG benchmark candidate."
    )
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review-output", required=True, type=Path)
    parser.add_argument("--model")
    parser.add_argument("--batch-size", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    corpus = load_model(args.corpus.resolve(), RagCorpusSnapshot)
    seed = load_model(args.seed.resolve(), RagBenchmarkSeed)
    dataset, review = author_benchmark(
        seed,
        corpus,
        llm_client=LocalLLMClient(),
        model=args.model,
        batch_size=args.batch_size,
    )
    benchmark_audit = audit_benchmark(dataset, corpus, require_accepted=False)
    review_audit = audit_annotation_review(review, corpus)
    write_model_atomic(args.output.resolve(), dataset)
    write_model_atomic(args.review_output.resolve(), review)
    print(
        json.dumps(
            {
                "benchmark_output": str(args.output.resolve()),
                "review_output": str(args.review_output.resolve()),
                "benchmark_audit": benchmark_audit,
                "review_audit": review_audit,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
