from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from .benchmark import audit_benchmark, benchmark_payload_sha256
from .io import load_model, write_model_atomic
from .reviews import audit_annotation_review, review_payload_sha256
from .schemas import (
    RagAnnotationReview,
    RagBenchmarkDataset,
    RagClaimDecision,
    RagCorpusSnapshot,
    RagEvidenceReference,
    RagBenchmarkItem,
)


class QuestionRevision(BaseModel):
    question: str = Field(min_length=5)
    gold_claim_ids: list[str] | None = Field(default=None, min_length=1)
    notes: str = Field(min_length=1)


class BenchmarkRevisionSet(BaseModel):
    schema_version: str = "1.0"
    revision_id: str = Field(min_length=1)
    source_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    revisions: dict[str, QuestionRevision]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply tracked question revisions to a benchmark candidate."
    )
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--revisions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review-output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    corpus = load_model(args.corpus.resolve(), RagCorpusSnapshot)
    source = load_model(args.benchmark.resolve(), RagBenchmarkDataset)
    source_review = load_model(args.review.resolve(), RagAnnotationReview)
    revisions = load_model(args.revisions.resolve(), BenchmarkRevisionSet)
    if revisions.source_dataset_sha256 != source.dataset_sha256:
        raise ValueError("Revision source hash does not match the benchmark.")

    cards_by_claim = {
        claim.claim_id: (card, claim)
        for card in corpus.cards
        for claim in card.claims
    }
    items_by_id = {item.question_id: item for item in source.items}
    unknown = set(revisions.revisions) - set(items_by_id)
    if unknown:
        raise ValueError(f"Revision contains unknown question ids: {sorted(unknown)}")

    revised_items = []
    revision_notes_by_claim: dict[str, str] = {}
    for item in source.items:
        revision = revisions.revisions.get(item.question_id)
        if revision is None:
            revised_items.append(item.model_copy(deep=True))
            continue
        updates: dict[str, object] = {
            "question": revision.question,
            "review_status": "pending",
            "review_notes": revision.notes,
        }
        if revision.gold_claim_ids is not None:
            contexts = []
            for claim_id in revision.gold_claim_ids:
                context = cards_by_claim.get(claim_id)
                if context is None:
                    raise ValueError(f"Unknown replacement claim: {claim_id}")
                contexts.append(context)
                revision_notes_by_claim[claim_id] = revision.notes
            gold_cards = list(dict.fromkeys(card.card_id for card, _ in contexts))
            evidence = [
                RagEvidenceReference(
                    card_id=card.card_id,
                    claim_id=claim.claim_id,
                    evidence_id=source_evidence.evidence_id,
                    quote=source_evidence.quote,
                    start_seconds=source_evidence.start_seconds,
                    end_seconds=source_evidence.end_seconds,
                )
                for card, claim in contexts
                for source_evidence in claim.evidence
            ]
            updates.update(
                {
                    "gold_card_ids": gold_cards,
                    "gold_claim_ids": revision.gold_claim_ids,
                    "evidence": evidence,
                    "reference_answer": " ".join(
                        claim.text for _, claim in contexts
                    ),
                }
            )
            if item.category == "multi_hop":
                updates["graph_path_card_ids"] = gold_cards
        revised_items.append(
            RagBenchmarkItem.model_validate(
                {**item.model_dump(mode="python"), **updates}
            )
        )

    revised = source.model_copy(
        update={
            "benchmark_id": f"{source.benchmark_id}-{revisions.revision_id}",
            "annotation_method": (
                source.annotation_method
                + f" Candidate wording revised under {revisions.revision_id}; "
                "independent human review is still pending."
            ),
            "dataset_sha256": "0" * 64,
            "items": revised_items,
        }
    )
    revised.dataset_sha256 = benchmark_payload_sha256(revised)

    used_claim_ids = {
        claim_id
        for item in revised.items
        for claim_id in item.gold_claim_ids
    }
    existing_decisions = {
        decision.claim_id: decision
        for decision in source_review.claim_decisions
    }
    claim_decisions = []
    for claim_id in sorted(used_claim_ids):
        existing = existing_decisions.get(claim_id)
        if existing is not None:
            claim_decisions.append(existing.model_copy(deep=True))
            continue
        card, _ = cards_by_claim[claim_id]
        claim_decisions.append(
            RagClaimDecision(
                card_id=card.card_id,
                claim_id=claim_id,
                support="supported",
                reviewer_id="codex-model-assisted-r1",
                review_method="model_assisted",
                review_notes=revision_notes_by_claim[claim_id],
            )
        )
    revised_review = source_review.model_copy(
        update={
            "review_id": f"{source_review.review_id}-{revisions.revision_id}",
            "claim_decisions": claim_decisions,
            "review_sha256": "0" * 64,
        }
    )
    revised_review.review_sha256 = review_payload_sha256(revised_review)

    benchmark_audit = audit_benchmark(revised, corpus, require_accepted=False)
    review_audit = audit_annotation_review(revised_review, corpus)
    write_model_atomic(args.output.resolve(), revised)
    write_model_atomic(args.review_output.resolve(), revised_review)
    print(
        json.dumps(
            {
                "benchmark_output": str(args.output.resolve()),
                "review_output": str(args.review_output.resolve()),
                "revision_count": len(revisions.revisions),
                "benchmark_audit": benchmark_audit,
                "review_audit": review_audit,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
