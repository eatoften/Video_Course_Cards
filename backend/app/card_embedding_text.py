from __future__ import annotations

import hashlib

from .knowledge_card import KnowledgeCard


def build_card_embedding_text(card: KnowledgeCard) -> str:
    lines: list[str] = [
        "Title:",
        _clean_text(card.title),
        "",
        "Summary:",
        _clean_text(card.summary),
    ]

    if card.key_points:
        lines.extend(["", "Key points:"])
        lines.extend(
            f"- {_clean_text(point)}"
            for point in card.key_points
            if _clean_text(point)
        )

    if card.claims:
        lines.extend(["", "Claims:"])

        for claim in card.claims:
            claim_text = _clean_text(claim.text)

            if not claim_text:
                continue

            lines.append(f"- Claim: {claim_text}")

            for evidence in claim.evidence:
                quote = _clean_text(evidence.quote)

                if quote:
                    lines.append(f"  Evidence: {quote}")

    if card.question or card.answer:
        lines.extend(["", "Active recall:"])

        if card.question:
            lines.append(f"Q: {_clean_text(card.question)}")

        if card.answer:
            lines.append(f"A: {_clean_text(card.answer)}")

    if card.tags:
        tags = [
            _clean_text(tag)
            for tag in card.tags
            if _clean_text(tag)
        ]

        if tags:
            lines.extend(["", "Tags:", ", ".join(tags)])

    return "\n".join(lines).strip()


def hash_card_embedding_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean_text(value: str) -> str:
    return " ".join(value.strip().split())
