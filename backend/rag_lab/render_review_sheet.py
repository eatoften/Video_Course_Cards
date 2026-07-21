from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .io import load_model
from .quality import audit_benchmark_quality
from .schemas import RagBenchmarkDataset, RagCorpusSnapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a human-review sheet for a RAG benchmark candidate."
    )
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quality-output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    corpus = load_model(args.corpus.resolve(), RagCorpusSnapshot)
    benchmark = load_model(args.benchmark.resolve(), RagBenchmarkDataset)
    quality = audit_benchmark_quality(benchmark, corpus)
    flags_by_question: dict[str, list[dict[str, object]]] = {}
    for flag in quality["flags"]:
        flags_by_question.setdefault(str(flag["question_id"]), []).append(flag)

    cards = {card.card_id: card for card in corpus.cards}
    lines = [
        "# R1 Benchmark Human Review Sheet",
        "",
        f"Benchmark: `{benchmark.benchmark_id}`",
        f"Corpus SHA-256: `{benchmark.corpus_sha256}`",
        f"Dataset SHA-256: `{benchmark.dataset_sha256}`",
        "",
        "> This is a candidate generated with model assistance. Check each item against",
        "> the exact evidence before changing its `review_status` to `accepted`.",
        "",
        "Review labels: `[ ] accept` / `[ ] edit` / `[ ] reject`.",
        "",
    ]
    for item in benchmark.items:
        lines.extend(
            [
                f"## {item.question_id} ({item.category}, {item.split})",
                "",
                "- [ ] accept",
                "- [ ] edit",
                "- [ ] reject",
                f"- Question: {item.question}",
                f"- Answerable: `{str(item.answerable).lower()}`",
            ]
        )
        if item.reference_answer:
            lines.append(f"- Reference answer: {item.reference_answer}")
        if item.gold_card_ids:
            lines.append(
                "- Gold cards: "
                + ", ".join(
                    f"`{card_id}` ({cards[card_id].title})"
                    for card_id in item.gold_card_ids
                )
            )
        if item.gold_claim_ids:
            lines.append(
                "- Gold claims: "
                + ", ".join(f"`{claim_id}`" for claim_id in item.gold_claim_ids)
            )
        for evidence in item.evidence:
            lines.append(
                f'- Evidence `{evidence.evidence_id}` [{evidence.start_seconds:.2f}s-'
                f'{evidence.end_seconds:.2f}s]: "{evidence.quote}"'
            )
        for flag in flags_by_question.get(item.question_id, []):
            lines.append(
                f"- Quality flag `{flag['severity']}:{flag['code']}`: {flag['message']}"
            )
        lines.extend(["- Reviewer notes:", "", ""])

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    quality_path = args.quality_output.resolve()
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "review_sheet": str(output_path),
                "quality_report": str(quality_path),
                "flag_count": quality["flag_count"],
                "question_count_with_flags": quality["question_count_with_flags"],
                "flag_counts": quality["flag_counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
