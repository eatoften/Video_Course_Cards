from __future__ import annotations

import json
import re
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from .card_relation import CardRelationClassification
from .knowledge_card import KnowledgeCard
from .llm_client import LLMClientError, LLMMessage, LLMTimeoutError
from .settings import LLMSettings


THINK_BLOCK_RE = re.compile(
    r"<think>.*?</think>",
    flags=re.DOTALL | re.IGNORECASE,
)
FENCED_JSON_RE = re.compile(
    r"```(?:json)?\s*([\s\S]*?)```",
    flags=re.IGNORECASE,
)


class RelationClassifierClient(Protocol):
    settings: LLMSettings

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


class RelationClassificationError(Exception):
    pass


class RelationClassificationTimeoutError(RelationClassificationError):
    pass


class RelationClassificationOutputError(RelationClassificationError):
    pass


class RelationClassificationPayload(BaseModel):
    relation_type: CardRelationClassification
    explanation: str = Field(min_length=1, max_length=2000)


def classify_card_relation(
    source_card: KnowledgeCard,
    target_card: KnowledgeCard,
    *,
    llm_client: RelationClassifierClient,
    model: str | None = None,
) -> RelationClassificationPayload:
    selected_model = model.strip() if model and model.strip() else None

    try:
        raw_output = llm_client.create_chat_completion(
            _build_messages(source_card, target_card),
            model=selected_model,
            temperature=0.0,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
    except LLMTimeoutError as exc:
        raise RelationClassificationTimeoutError(
            "Local LLM relation classification timed out."
        ) from exc
    except LLMClientError as exc:
        raise RelationClassificationError(
            f"Local LLM unavailable: {exc}"
        ) from exc

    return _parse_payload(raw_output)


def _build_messages(
    source_card: KnowledgeCard,
    target_card: KnowledgeCard,
) -> list[LLMMessage]:
    allowed_types = (
        "prerequisite, related, example_of, contrast_with, part_of, unclear"
    )
    return [
        LLMMessage(
            role="system",
            content=(
                "Classify the directed conceptual relationship from source "
                "card to target card. Use only the supplied card content. "
                f"Allowed relation_type values: {allowed_types}. "
                "Use prerequisite only when understanding the source is "
                "needed before the target; example_of when the source is an "
                "example of the target; part_of when the source is a component "
                "of the target; contrast_with for an explicit conceptual "
                "contrast; related for a meaningful but less specific link; "
                "and unclear when evidence is insufficient. Return one JSON "
                "object with relation_type and a short explanation."
            ),
        ),
        LLMMessage(
            role="user",
            content=json.dumps(
                {
                    "source_card": _card_context(source_card),
                    "target_card": _card_context(target_card),
                },
                ensure_ascii=False,
            ),
        ),
    ]


def _card_context(card: KnowledgeCard) -> dict[str, object]:
    return {
        "title": card.title,
        "summary": card.summary,
        "key_points": card.key_points,
        "claims": [claim.text for claim in card.claims],
        "tags": card.tags,
    }


def _parse_payload(raw_output: str) -> RelationClassificationPayload:
    cleaned = THINK_BLOCK_RE.sub("", raw_output).strip()
    fenced_match = FENCED_JSON_RE.search(cleaned)

    if fenced_match:
        cleaned = fenced_match.group(1).strip()

    try:
        data = json.loads(cleaned)
        return RelationClassificationPayload.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise RelationClassificationOutputError(
            "Local LLM returned an invalid relation classification."
        ) from exc
