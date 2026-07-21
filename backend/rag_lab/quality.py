from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .retrievers import tokenize_for_retrieval
from .schemas import RagBenchmarkDataset, RagBenchmarkItem, RagCorpusSnapshot


CAUSAL_TERMS = {"because", "causes", "causing", "enables", "prevents", "so", "therefore", "to"}
YES_NO_PREFIXES = (
    "can ",
    "could ",
    "did ",
    "do ",
    "does ",
    "is ",
    "are ",
    "was ",
    "were ",
)


@dataclass(frozen=True)
class BenchmarkQualityFlag:
    question_id: str
    code: str
    severity: str
    message: str


def audit_benchmark_quality(
    dataset: RagBenchmarkDataset,
    corpus: RagCorpusSnapshot,
) -> dict[str, object]:
    claims = {
        claim.claim_id: claim.text
        for card in corpus.cards
        for claim in card.claims
    }
    flags: list[BenchmarkQualityFlag] = []
    for item in dataset.items:
        flags.extend(_item_flags(item, claims))
    counts = Counter(flag.code for flag in flags)
    return {
        "question_count": len(dataset.items),
        "flag_count": len(flags),
        "question_count_with_flags": len({flag.question_id for flag in flags}),
        "flag_counts": dict(sorted(counts.items())),
        "flags": [flag.__dict__ for flag in flags],
    }


def _item_flags(
    item: RagBenchmarkItem,
    claims: dict[str, str],
) -> list[BenchmarkQualityFlag]:
    flags: list[BenchmarkQualityFlag] = []
    question_tokens = set(tokenize_for_retrieval(item.question))
    reference_tokens = set(tokenize_for_retrieval(item.reference_answer or ""))
    if reference_tokens:
        coverage = len(question_tokens & reference_tokens) / len(reference_tokens)
        if coverage >= 0.7:
            flags.append(
                BenchmarkQualityFlag(
                    item.question_id,
                    "high_reference_overlap",
                    "warning",
                    f"Question contains {coverage:.0%} of unique reference-answer tokens.",
                )
            )

    lowered = item.question.lower()
    if item.category == "concept" and lowered.startswith("why "):
        support_tokens = set(tokenize_for_retrieval(item.reference_answer or ""))
        if not support_tokens.intersection(CAUSAL_TERMS):
            flags.append(
                BenchmarkQualityFlag(
                    item.question_id,
                    "unsupported_why_shape",
                    "warning",
                    "The question asks why, but the reference claim is not causal.",
                )
            )

    if item.category == "multi_hop":
        if lowered.startswith(YES_NO_PREFIXES):
            flags.append(
                BenchmarkQualityFlag(
                    item.question_id,
                    "yes_no_multi_hop",
                    "warning",
                    "The multi-hop item can invite an uninformative yes/no response.",
                )
            )
        claim_coverages = []
        for claim_id in item.gold_claim_ids:
            claim_tokens = set(tokenize_for_retrieval(claims[claim_id]))
            claim_coverages.append(
                len(question_tokens & claim_tokens) / max(1, len(claim_tokens))
            )
        if claim_coverages and min(claim_coverages) >= 0.55:
            flags.append(
                BenchmarkQualityFlag(
                    item.question_id,
                    "multi_hop_answer_leakage",
                    "error",
                    "Question substantially repeats every gold claim.",
                )
            )
    return flags
