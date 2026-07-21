from __future__ import annotations

import json
import re
import statistics
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.embedding import TextEmbedder, cosine_similarity
from app.llm_client import LLMMessage

from .schemas import (
    RagBenchmarkItem,
    RagCorpusCard,
    RagCorpusSnapshot,
    RagGroundedAnswerPayload,
    RagGroundedAnswerRecord,
    RetrievalRecord,
)


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
FENCED_JSON_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


class GroundedAnswerClient(Protocol):
    def create_chat_completion(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, object] | None = None,
    ) -> str:
        pass


class GroundedAnswerOutputError(ValueError):
    def __init__(self, message: str, *, raw_output: str | None = None) -> None:
        super().__init__(message)
        self.raw_output = raw_output


class _LLMGroundedAnswerPayload(BaseModel):
    answerable: bool
    claims: list = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def validate_shape(self) -> "_LLMGroundedAnswerPayload":
        if self.answerable and not self.claims:
            raise ValueError("An answerable response needs cited claims.")
        if not self.answerable and self.claims:
            raise ValueError("An abstention cannot include claims.")
        return self


def build_grounded_context(
    corpus: RagCorpusSnapshot,
    retrieval: RetrievalRecord,
    *,
    top_k: int,
    character_budget: int,
) -> tuple[list[dict[str, object]], int]:
    cards = {card.card_id: card for card in corpus.cards}
    packed: list[dict[str, object]] = []
    used = 2
    for ranked in retrieval.ranked_cards[:top_k]:
        card = cards.get(ranked.card_id)
        if card is None:
            raise ValueError(f"Retrieval references an unknown card: {ranked.card_id}")
        payload = _card_payload(card)
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        added = len(serialized) + (1 if packed else 0)
        if used + added > character_budget:
            break
        packed.append(payload)
        used += added
    if not packed:
        raise ValueError("Context budget is too small for the highest-ranked card.")
    return packed, used


def generate_grounded_answer(
    question: str,
    context: list[dict[str, object]],
    *,
    llm_client: GroundedAnswerClient,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[RagGroundedAnswerPayload, str]:
    allowed = {
        (claim["card_id"], claim["claim_id"], evidence["evidence_id"])
        for card in context
        for claim in card["claims"]
        for evidence in claim["evidence"]
    }
    raw = llm_client.create_chat_completion(
        _answer_messages(question, context),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    try:
        payload = _parse_payload(raw)
        _validate_citations(payload, allowed, raw_output=raw)
        return payload, raw
    except GroundedAnswerOutputError:
        repaired = llm_client.create_chat_completion(
            _repair_messages(question, context, raw),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        payload = _parse_payload(repaired)
        _validate_citations(payload, allowed, raw_output=repaired)
        return payload, repaired


def evaluate_grounded_answers(
    items: Sequence[RagBenchmarkItem],
    retrieval_records: Sequence[RetrievalRecord],
    answer_records: Sequence[RagGroundedAnswerRecord],
    *,
    semantic_embedder: TextEmbedder,
) -> dict[str, object]:
    items_by_id = {item.question_id: item for item in items}
    retrieval_by_id = {record.question_id: record for record in retrieval_records}
    answers_by_id = {record.question_id: record for record in answer_records}
    if set(items_by_id) != set(retrieval_by_id) or set(items_by_id) != set(answers_by_id):
        raise ValueError("Items, retrieval records, and answers must align exactly.")

    successful = [record for record in answer_records if record.generation_error is None]
    answerable_successes = [
        answers_by_id[item.question_id]
        for item in items
        if item.answerable
        and answers_by_id[item.question_id].generation_error is None
        and answers_by_id[item.question_id].answerable_prediction
    ]
    true_positive = false_positive = false_negative = true_negative = 0
    citation_gold = citation_total = gold_evidence_total = gold_evidence_cited = 0
    gold_claim_total = gold_claim_cited = 0
    context_gold_recalls = []
    for item in items:
        answer = answers_by_id[item.question_id]
        predicted = bool(answer.answerable_prediction) if answer.generation_error is None else False
        if item.answerable and predicted:
            true_positive += 1
        elif item.answerable:
            false_negative += 1
        elif predicted:
            false_positive += 1
        else:
            true_negative += 1

        cited = {
            (citation.card_id, citation.claim_id, citation.evidence_id)
            for claim in answer.claims
            for citation in claim.citations
        }
        citation_total += len(cited)

        if item.answerable:
            context_gold_recalls.append(
                len(set(answer.context_card_ids).intersection(item.gold_card_ids))
                / len(item.gold_card_ids)
            )
            gold_evidence = {
                (reference.card_id, reference.claim_id, reference.evidence_id)
                for reference in item.evidence
            }
            cited_claims = {claim_id for _, claim_id, _ in cited}
            citation_gold += len(cited.intersection(gold_evidence))
            gold_evidence_cited += len(cited.intersection(gold_evidence))
            gold_evidence_total += len(gold_evidence)
            gold_claim_cited += len(cited_claims.intersection(item.gold_claim_ids))
            gold_claim_total += len(item.gold_claim_ids)

    similarities = []
    if answerable_successes:
        references = [items_by_id[record.question_id].reference_answer or "" for record in answerable_successes]
        answers = [record.answer or "" for record in answerable_successes]
        vectors = semantic_embedder.embed_texts([*references, *answers])
        midpoint = len(references)
        similarities = [
            cosine_similarity(vectors[index], vectors[midpoint + index])
            for index in range(midpoint)
        ]

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    answerability_f1 = (
        0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    )
    latencies = [record.latency_milliseconds for record in successful]
    generated_claims = [claim for record in successful for claim in record.claims]
    return {
        "question_count": len(items),
        "generation_success_rate": len(successful) / len(items),
        "answerability_confusion": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        },
        "abstention_f1": answerability_f1,
        "context_gold_card_recall": statistics.fmean(context_gold_recalls),
        "gold_claim_citation_recall": gold_claim_cited / max(1, gold_claim_total),
        "gold_evidence_citation_recall": gold_evidence_cited / max(1, gold_evidence_total),
        "citation_precision_against_gold": citation_gold / max(1, citation_total),
        "structural_citation_validity": 1.0 if generated_claims else 0.0,
        "reference_answer_cosine_mean": (
            statistics.fmean(similarities) if similarities else None
        ),
        "mean_latency_milliseconds": statistics.fmean(latencies) if latencies else 0.0,
        "median_latency_milliseconds": statistics.median(latencies) if latencies else 0.0,
        "evaluation_warning": (
            "Reference-answer cosine is a semantic proxy, not answer correctness. "
            "Claim entailment and correctness require independent human review."
        ),
    }


def _card_payload(card: RagCorpusCard) -> dict[str, object]:
    return {
        "card_id": card.card_id,
        "title": card.title,
        "claims": [
            {
                "card_id": card.card_id,
                "claim_id": claim.claim_id,
                "text": claim.text,
                "evidence": [
                    {
                        "evidence_id": evidence.evidence_id,
                        "quote": evidence.quote,
                        "start_seconds": evidence.start_seconds,
                        "end_seconds": evidence.end_seconds,
                    }
                    for evidence in claim.evidence
                ],
            }
            for claim in card.claims
        ],
    }


def _answer_messages(
    question: str,
    context: list[dict[str, object]],
) -> list[LLMMessage]:
    return [
        LLMMessage(
            role="system",
            content=(
                "Answer using only the supplied card claims and exact evidence. If the "
                "evidence does not answer the question, set answerable=false and answer "
                "nothing. Do not use outside knowledge. Return at most two short claims, "
                "each at most 35 words, and cite one or more exact card_id, claim_id, "
                "evidence_id triples from context. Do not add a separate answer field. "
                "Return JSON only: {\"answerable\":true,\"claims\":["
                "{\"text\":\"...\",\"citations\":[{\"card_id\":\"...\","
                "\"claim_id\":\"...\",\"evidence_id\":\"...\"}]}]}."
            ),
        ),
        LLMMessage(
            role="user",
            content=(
                "/no_think\nQUESTION:\n"
                + question
                + "\n\nCONTEXT:\n"
                + json.dumps(context, ensure_ascii=False)
            ),
        ),
    ]


def _repair_messages(
    question: str,
    context: list[dict[str, object]],
    raw: str,
) -> list[LLMMessage]:
    return [
        _answer_messages(question, context)[0],
        LLMMessage(
            role="user",
            content=(
                "/no_think\nRepair the following response into the required JSON. "
                "Use only IDs present in CONTEXT.\n\nCONTEXT:\n"
                + json.dumps(context, ensure_ascii=False)
                + "\n\nRESPONSE:\n"
                + raw
            ),
        ),
    ]


def _parse_payload(raw: str) -> RagGroundedAnswerPayload:
    cleaned = THINK_BLOCK_RE.sub("", raw).strip()
    fenced = FENCED_JSON_RE.search(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    try:
        parsed = _LLMGroundedAnswerPayload.model_validate(json.loads(cleaned))
        claims = parsed.claims
        final_claims = RagGroundedAnswerPayload.model_validate(
            {
                "answerable": parsed.answerable,
                "answer": (
                    " ".join(str(claim["text"]) for claim in claims)
                    if parsed.answerable
                    else "Not enough evidence in the course cards."
                ),
                "claims": claims,
            }
        )
        return final_claims
    except (json.JSONDecodeError, ValidationError) as exc:
        raise GroundedAnswerOutputError(
            "LLM returned invalid grounded-answer JSON.",
            raw_output=raw,
        ) from exc


def _validate_citations(
    payload: RagGroundedAnswerPayload,
    allowed: set[tuple[str, str, str]],
    *,
    raw_output: str,
) -> None:
    for claim in payload.claims:
        for citation in claim.citations:
            key = (citation.card_id, citation.claim_id, citation.evidence_id)
            if key not in allowed:
                raise GroundedAnswerOutputError(
                    f"Citation is outside context: {key}",
                    raw_output=raw_output,
                )
