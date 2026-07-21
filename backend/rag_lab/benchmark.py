from __future__ import annotations

from collections import Counter

from .io import sha256_value
from .schemas import RagBenchmarkDataset, RagCorpusSnapshot


class RagBenchmarkAuditError(ValueError):
    pass


def benchmark_payload_sha256(dataset: RagBenchmarkDataset) -> str:
    payload = dataset.model_dump(mode="json", exclude={"dataset_sha256", "created_at"})
    return sha256_value(payload)


def audit_benchmark(
    dataset: RagBenchmarkDataset,
    corpus: RagCorpusSnapshot,
    *,
    require_accepted: bool = True,
) -> dict[str, object]:
    errors: list[str] = []
    if dataset.course_id != corpus.course_id:
        errors.append("Benchmark course does not match the corpus.")
    if dataset.corpus_sha256 != corpus.snapshot_sha256:
        errors.append("Benchmark corpus hash does not match the snapshot.")

    cards = {card.card_id: card for card in corpus.cards}
    claims = {
        claim.claim_id: (card.card_id, claim)
        for card in corpus.cards
        for claim in card.claims
    }
    evidence = {
        item.evidence_id: (card.card_id, claim.claim_id, item)
        for card in corpus.cards
        for claim in card.claims
        for item in claim.evidence
    }
    normalized_questions: set[str] = set()
    question_ids: set[str] = set()

    for item in dataset.items:
        if item.question_id in question_ids:
            errors.append(f"Duplicate question id: {item.question_id}")
        question_ids.add(item.question_id)
        normalized = " ".join(item.question.lower().split())
        if normalized in normalized_questions:
            errors.append(f"Duplicate normalized question: {item.question_id}")
        normalized_questions.add(normalized)
        if require_accepted and item.review_status != "accepted":
            errors.append(f"Question is not accepted: {item.question_id}")

        for card_id in item.gold_card_ids:
            if card_id not in cards:
                errors.append(f"Unknown gold card {card_id} in {item.question_id}")
        for claim_id in item.gold_claim_ids:
            owner = claims.get(claim_id)
            if owner is None:
                errors.append(f"Unknown gold claim {claim_id} in {item.question_id}")
            elif owner[0] not in item.gold_card_ids:
                errors.append(f"Gold claim {claim_id} belongs to a non-gold card.")
        for reference in item.evidence:
            source = evidence.get(reference.evidence_id)
            if source is None:
                errors.append(
                    f"Unknown evidence {reference.evidence_id} in {item.question_id}"
                )
                continue
            source_card_id, source_claim_id, source_evidence = source
            if source_card_id != reference.card_id or source_claim_id != reference.claim_id:
                errors.append(f"Evidence ownership mismatch in {item.question_id}")
            if reference.card_id not in item.gold_card_ids:
                errors.append(f"Evidence card is not gold in {item.question_id}")
            if reference.claim_id not in item.gold_claim_ids:
                errors.append(f"Evidence claim is not gold in {item.question_id}")
            if (
                reference.quote != source_evidence.quote
                or reference.start_seconds != source_evidence.start_seconds
                or reference.end_seconds != source_evidence.end_seconds
            ):
                errors.append(f"Evidence content mismatch in {item.question_id}")

    expected_sha256 = benchmark_payload_sha256(dataset)
    if dataset.dataset_sha256 != expected_sha256:
        errors.append("Benchmark dataset hash is not canonical.")

    category_counts = Counter(item.category for item in dataset.items)
    split_counts = Counter(item.split for item in dataset.items)
    split_category_counts = Counter((item.split, item.category) for item in dataset.items)
    result = {
        "passed": not errors,
        "errors": errors,
        "question_count": len(dataset.items),
        "category_counts": dict(sorted(category_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "split_category_counts": {
            f"{split}:{category}": count
            for (split, category), count in sorted(split_category_counts.items())
        },
        "corpus_card_count": len(corpus.cards),
        "corpus_claim_count": sum(len(card.claims) for card in corpus.cards),
        "corpus_relation_count": len(corpus.relations),
        "dataset_sha256": expected_sha256,
    }
    if errors:
        raise RagBenchmarkAuditError("\n".join(errors))
    return result

