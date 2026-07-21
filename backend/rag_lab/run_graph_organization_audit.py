from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .graph_organization import analyze_graph_organization
from .io import load_model, sha256_file, write_json_atomic
from .schemas import RagAnnotationReview, RagCorpusSnapshot, RagEmbeddingSnapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a card graph as an associative knowledge structure."
    )
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--neighborhood-k", type=int, default=5)
    parser.add_argument("--random-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    corpus_path = args.corpus.resolve()
    review_path = args.review.resolve()
    embeddings_path = args.embeddings.resolve()
    report = analyze_graph_organization(
        load_model(corpus_path, RagCorpusSnapshot),
        load_model(review_path, RagAnnotationReview),
        load_model(embeddings_path, RagEmbeddingSnapshot),
        neighborhood_k=args.neighborhood_k,
        random_samples=args.random_samples,
        seed=args.seed,
    )
    report["input_file_sha256"] = {
        "corpus": sha256_file(corpus_path),
        "review": sha256_file(review_path),
        "embeddings": sha256_file(embeddings_path),
    }
    output_path = args.output.resolve()
    write_json_atomic(output_path, report)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "structure": report["structure"],
                "association": report["association"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
