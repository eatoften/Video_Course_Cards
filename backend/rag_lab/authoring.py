from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from app.llm_client import LLMMessage

from .benchmark import benchmark_payload_sha256
from .reviews import review_payload_sha256
from .schemas import (
    RagAnnotationReview,
    RagBenchmarkDataset,
    RagBenchmarkItem,
    RagBenchmarkSeed,
    RagClaimDecision,
    RagCorpusCard,
    RagCorpusClaim,
    RagCorpusSnapshot,
    RagEvidenceReference,
    RagGraphDecision,
)


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
FENCED_JSON_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


class QuestionAuthoringClient(Protocol):
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


class QuestionAuthoringError(ValueError):
    pass


class SingleQuestionProposal(BaseModel):
    index: int = Field(ge=0)
    factual_question: str = Field(min_length=5)
    concept_question: str = Field(min_length=5)


class PairQuestionProposal(BaseModel):
    index: int = Field(ge=0)
    comparison_question: str = Field(min_length=5)
    multi_hop_question: str = Field(min_length=5)


class _SingleQuestionBatch(BaseModel):
    items: list[SingleQuestionProposal]


class _PairQuestionBatch(BaseModel):
    items: list[PairQuestionProposal]


def author_benchmark(
    seed: RagBenchmarkSeed,
    corpus: RagCorpusSnapshot,
    *,
    llm_client: QuestionAuthoringClient,
    model: str | None = None,
    batch_size: int = 5,
    reviewer_id: str = "codex-model-assisted-r1",
) -> tuple[RagBenchmarkDataset, RagAnnotationReview]:
    if seed.course_id != corpus.course_id:
        raise QuestionAuthoringError("Seed course does not match the corpus.")
    if seed.corpus_sha256 != corpus.snapshot_sha256:
        raise QuestionAuthoringError("Seed corpus hash does not match the snapshot.")
    if batch_size < 1:
        raise QuestionAuthoringError("batch_size must be positive.")

    cards_by_id = {card.card_id: card for card in corpus.cards}
    claims = {
        claim.claim_id: (card, claim)
        for card in corpus.cards
        for claim in card.claims
    }
    single_contexts: list[tuple[RagCorpusCard, RagCorpusClaim]] = []
    for item in seed.single_questions:
        context = claims.get(item.claim_id)
        if context is None:
            raise QuestionAuthoringError(f"Unknown single claim: {item.claim_id}")
        single_contexts.append(context)

    pair_contexts: list[
        tuple[RagCorpusCard, RagCorpusClaim, RagCorpusCard, RagCorpusClaim]
    ] = []
    for item in seed.paired_questions:
        source = claims.get(item.source_claim_id)
        target = claims.get(item.target_claim_id)
        if source is None or target is None:
            raise QuestionAuthoringError(
                "A paired seed references an unknown claim: "
                f"{item.source_claim_id}, {item.target_claim_id}"
            )
        if source[0].card_id == target[0].card_id:
            raise QuestionAuthoringError("A paired question needs two distinct cards.")
        pair_contexts.append((*source, *target))

    single_proposals = _author_single_questions(
        single_contexts,
        llm_client=llm_client,
        model=model,
        batch_size=batch_size,
    )
    pair_proposals = _author_pair_questions(
        pair_contexts,
        llm_client=llm_client,
        model=model,
        batch_size=batch_size,
    )

    items: list[RagBenchmarkItem] = []
    for index, (seed_item, context, proposal) in enumerate(
        zip(seed.single_questions, single_contexts, single_proposals),
        start=1,
    ):
        card, claim = context
        common = {
            "split": seed_item.split,
            "answerable": True,
            "reference_answer": claim.text,
            "gold_card_ids": [card.card_id],
            "gold_claim_ids": [claim.claim_id],
            "evidence": _evidence_references(card, claim),
            "authoring_method": "model_assisted",
            "review_status": "pending",
            "review_notes": seed_item.review_notes,
        }
        items.extend(
            [
                RagBenchmarkItem(
                    question_id=f"factual-{index:03d}",
                    category="factual",
                    question=_normalize_question(proposal.factual_question),
                    **common,
                ),
                RagBenchmarkItem(
                    question_id=f"concept-{index:03d}",
                    category="concept",
                    question=_normalize_question(proposal.concept_question),
                    **common,
                ),
            ]
        )

    for index, (seed_item, context, proposal) in enumerate(
        zip(seed.paired_questions, pair_contexts, pair_proposals),
        start=1,
    ):
        source_card, source_claim, target_card, target_claim = context
        gold_cards = [source_card.card_id, target_card.card_id]
        common = {
            "split": seed_item.split,
            "answerable": True,
            "reference_answer": f"{source_claim.text} {target_claim.text}",
            "gold_card_ids": gold_cards,
            "gold_claim_ids": [source_claim.claim_id, target_claim.claim_id],
            "evidence": [
                *_evidence_references(source_card, source_claim),
                *_evidence_references(target_card, target_claim),
            ],
            "authoring_method": "model_assisted",
            "review_status": "pending",
            "review_notes": seed_item.review_notes,
        }
        items.extend(
            [
                RagBenchmarkItem(
                    question_id=f"comparison-{index:03d}",
                    category="comparison",
                    question=_normalize_question(proposal.comparison_question),
                    **common,
                ),
                RagBenchmarkItem(
                    question_id=f"multi-hop-{index:03d}",
                    category="multi_hop",
                    question=_normalize_question(proposal.multi_hop_question),
                    graph_path_card_ids=gold_cards,
                    **common,
                ),
            ]
        )

    for index, seed_item in enumerate(seed.unanswerable_questions, start=1):
        items.append(
            RagBenchmarkItem(
                question_id=f"unanswerable-{index:03d}",
                category="unanswerable",
                split=seed_item.split,
                question=_normalize_question(seed_item.question),
                answerable=False,
                authoring_method="model_assisted",
                review_status="pending",
                review_notes=seed_item.review_notes,
            )
        )

    dataset = RagBenchmarkDataset(
        benchmark_id=f"{seed.seed_id}-benchmark",
        course_id=seed.course_id,
        corpus_sha256=corpus.snapshot_sha256,
        annotation_method=(
            "Model-assisted question paraphrasing over curator-selected claims; "
            "IDs, evidence quotes, and timestamps were deterministically copied "
            "from the frozen corpus. Pending independent human review."
        ),
        confirmatory_status="pending_human_review",
        dataset_sha256="0" * 64,
        items=items,
    )
    dataset.dataset_sha256 = benchmark_payload_sha256(dataset)

    claim_notes: dict[str, str] = {
        item.claim_id: item.review_notes
        for item in seed.single_questions
    }
    for item in seed.paired_questions:
        claim_notes.setdefault(item.source_claim_id, item.review_notes)
        claim_notes.setdefault(item.target_claim_id, item.review_notes)
    claim_decisions = [
        RagClaimDecision(
            card_id=claims[claim_id][0].card_id,
            claim_id=claim_id,
            support="supported",
            reviewer_id=reviewer_id,
            review_method="model_assisted",
            review_notes=notes,
        )
        for claim_id, notes in sorted(claim_notes.items())
    ]
    graph_decisions = []
    for seed_item, context in zip(seed.paired_questions, pair_contexts):
        source_card, _, target_card, _ = context
        graph_decisions.append(
            RagGraphDecision(
                source_card_id=source_card.card_id,
                target_card_id=target_card.card_id,
                accepted=True,
                relation_type=seed_item.relation_type,
                reviewer_id=reviewer_id,
                review_method="model_assisted",
                review_notes=seed_item.review_notes,
            )
        )
    review = RagAnnotationReview(
        review_id=f"{seed.seed_id}-candidate-review",
        corpus_sha256=corpus.snapshot_sha256,
        review_status="candidate",
        claim_decisions=claim_decisions,
        graph_decisions=graph_decisions,
        review_sha256="0" * 64,
    )
    review.review_sha256 = review_payload_sha256(review)

    if set(cards_by_id) != {card.card_id for card in corpus.cards}:
        raise AssertionError("Corpus card ids unexpectedly changed during authoring.")
    return dataset, review


def _author_single_questions(
    contexts: Sequence[tuple[RagCorpusCard, RagCorpusClaim]],
    *,
    llm_client: QuestionAuthoringClient,
    model: str | None,
    batch_size: int,
) -> list[SingleQuestionProposal]:
    proposals: list[SingleQuestionProposal] = []
    for start in range(0, len(contexts), batch_size):
        batch = contexts[start : start + batch_size]
        payload = [
            {
                "index": offset,
                "lecture": card.lecture_name,
                "title": card.title,
                "claim": claim.text,
                "evidence": [item.quote for item in claim.evidence],
            }
            for offset, (card, claim) in enumerate(batch)
        ]
        raw = llm_client.create_chat_completion(
            _single_messages(payload),
            model=model,
            temperature=0.0,
            max_tokens=1800,
            response_format={"type": "json_object"},
        )
        parsed = _parse_json_model(raw, _SingleQuestionBatch)
        expected = list(range(len(batch)))
        returned = [item.index for item in parsed.items]
        if sorted(returned) != expected:
            raise QuestionAuthoringError(
                f"Single-question indexes changed: expected {expected}, got {returned}."
            )
        proposals.extend(
            item.model_copy(update={"index": start + item.index})
            for item in sorted(parsed.items, key=lambda value: value.index)
        )
    return proposals


def _author_pair_questions(
    contexts: Sequence[
        tuple[RagCorpusCard, RagCorpusClaim, RagCorpusCard, RagCorpusClaim]
    ],
    *,
    llm_client: QuestionAuthoringClient,
    model: str | None,
    batch_size: int,
) -> list[PairQuestionProposal]:
    proposals: list[PairQuestionProposal] = []
    for start in range(0, len(contexts), batch_size):
        batch = contexts[start : start + batch_size]
        payload = [
            {
                "index": offset,
                "source": {"title": source_card.title, "claim": source_claim.text},
                "target": {"title": target_card.title, "claim": target_claim.text},
            }
            for offset, (
                source_card,
                source_claim,
                target_card,
                target_claim,
            ) in enumerate(batch)
        ]
        raw = llm_client.create_chat_completion(
            _pair_messages(payload),
            model=model,
            temperature=0.0,
            max_tokens=1800,
            response_format={"type": "json_object"},
        )
        parsed = _parse_json_model(raw, _PairQuestionBatch)
        expected = list(range(len(batch)))
        returned = [item.index for item in parsed.items]
        if sorted(returned) != expected:
            raise QuestionAuthoringError(
                f"Pair-question indexes changed: expected {expected}, got {returned}."
            )
        proposals.extend(
            item.model_copy(update={"index": start + item.index})
            for item in sorted(parsed.items, key=lambda value: value.index)
        )
    return proposals


def _single_messages(payload: list[dict[str, object]]) -> list[LLMMessage]:
    return [
        LLMMessage(
            role="system",
            content=(
                "You write English retrieval-benchmark questions for lecture notes. "
                "For every indexed item, write one concrete factual question and one "
                "conceptual how/why/what question. Each must be fully answerable by the "
                "supplied claim and evidence alone. Paraphrase instead of copying the "
                "claim, avoid 'according to the card', do not answer, do not invent facts, "
                "and preserve every index. Return JSON only as "
                '{"items":[{"index":0,"factual_question":"...",'
                '"concept_question":"..."}]}.'
            ),
        ),
        LLMMessage(
            role="user",
            content="/no_think\n" + json.dumps(payload, ensure_ascii=False),
        ),
    ]


def _pair_messages(payload: list[dict[str, object]]) -> list[LLMMessage]:
    return [
        LLMMessage(
            role="system",
            content=(
                "You write English multi-card retrieval-benchmark questions. For every "
                "indexed source/target pair, write (1) a comparison or relationship "
                "question and (2) a synthesis question that genuinely requires both "
                "claims. Both questions must be fully answerable from those two claims, "
                "must not mention cards, and must not add outside facts. Preserve every "
                "index. Return JSON only as "
                '{"items":[{"index":0,"comparison_question":"...",'
                '"multi_hop_question":"..."}]}.'
            ),
        ),
        LLMMessage(
            role="user",
            content="/no_think\n" + json.dumps(payload, ensure_ascii=False),
        ),
    ]


def _parse_json_model(raw: str, model_type: type[BaseModel]):
    cleaned = THINK_BLOCK_RE.sub("", raw).strip()
    fenced = FENCED_JSON_RE.search(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    try:
        return model_type.model_validate(json.loads(cleaned))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise QuestionAuthoringError("LLM returned invalid question JSON.") from exc


def _normalize_question(question: str) -> str:
    normalized = " ".join(question.strip().split())
    if not normalized:
        raise QuestionAuthoringError("Generated question is blank.")
    if normalized[-1] not in "?!":
        normalized += "?"
    return normalized


def _evidence_references(
    card: RagCorpusCard,
    claim: RagCorpusClaim,
) -> list[RagEvidenceReference]:
    return [
        RagEvidenceReference(
            card_id=card.card_id,
            claim_id=claim.claim_id,
            evidence_id=evidence.evidence_id,
            quote=evidence.quote,
            start_seconds=evidence.start_seconds,
            end_seconds=evidence.end_seconds,
        )
        for evidence in claim.evidence
    ]
