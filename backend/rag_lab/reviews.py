from __future__ import annotations

from .io import sha256_value
from .schemas import RagAnnotationReview, RagBenchmarkDataset, RagCorpusSnapshot


class RagReviewAuditError(ValueError):
    pass


def review_payload_sha256(review: RagAnnotationReview) -> str:
    payload = review.model_dump(mode="json", exclude={"review_sha256", "created_at"})
    return sha256_value(payload)


def audit_annotation_review(
    review: RagAnnotationReview,
    corpus: RagCorpusSnapshot,
) -> dict[str, object]:
    errors: list[str] = []
    if review.corpus_sha256 != corpus.snapshot_sha256:
        errors.append("Review corpus hash does not match the snapshot.")

    claims = {
        claim.claim_id: card.card_id
        for card in corpus.cards
        for claim in card.claims
    }
    card_ids = {card.card_id for card in corpus.cards}
    seen_claims: set[str] = set()
    for decision in review.claim_decisions:
        owner = claims.get(decision.claim_id)
        if owner is None:
            errors.append(f"Unknown reviewed claim: {decision.claim_id}")
        elif owner != decision.card_id:
            errors.append(f"Claim ownership mismatch: {decision.claim_id}")
        if decision.claim_id in seen_claims:
            errors.append(f"Duplicate claim review: {decision.claim_id}")
        seen_claims.add(decision.claim_id)

    seen_edges: set[tuple[str, str]] = set()
    for decision in review.graph_decisions:
        if decision.source_card_id not in card_ids:
            errors.append(f"Unknown graph source: {decision.source_card_id}")
        if decision.target_card_id not in card_ids:
            errors.append(f"Unknown graph target: {decision.target_card_id}")
        edge = (decision.source_card_id, decision.target_card_id)
        if edge in seen_edges:
            errors.append(f"Duplicate graph review: {edge}")
        seen_edges.add(edge)

    expected_sha256 = review_payload_sha256(review)
    if review.review_sha256 != expected_sha256:
        errors.append("Review hash is not canonical.")
    if errors:
        raise RagReviewAuditError("\n".join(errors))
    return {
        "passed": True,
        "review_status": review.review_status,
        "claim_decision_count": len(review.claim_decisions),
        "supported_claim_count": sum(
            decision.support == "supported"
            for decision in review.claim_decisions
        ),
        "graph_decision_count": len(review.graph_decisions),
        "accepted_graph_edge_count": sum(
            decision.accepted
            for decision in review.graph_decisions
        ),
        "review_sha256": expected_sha256,
    }


def require_formal_human_review(
    dataset: RagBenchmarkDataset,
    review: RagAnnotationReview,
) -> None:
    if dataset.confirmatory_status not in {"sealed", "opened"}:
        raise RagReviewAuditError("The benchmark is not sealed for formal evaluation.")
    pending = [
        item.question_id
        for item in dataset.items
        if item.review_status != "accepted"
    ]
    if pending:
        raise RagReviewAuditError(
            f"Formal benchmark contains non-accepted questions: {pending[:5]}"
        )
    if review.review_status != "human_verified":
        raise RagReviewAuditError("The annotation review is not human verified.")
    decisions = {
        decision.claim_id: decision
        for decision in review.claim_decisions
    }
    required_claims = {
        claim_id
        for item in dataset.items
        for claim_id in item.gold_claim_ids
    }
    invalid = [
        claim_id
        for claim_id in sorted(required_claims)
        if claim_id not in decisions or decisions[claim_id].support != "supported"
    ]
    if invalid:
        raise RagReviewAuditError(
            f"Formal benchmark has unverified gold claims: {invalid[:5]}"
        )
