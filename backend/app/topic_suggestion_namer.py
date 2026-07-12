from __future__ import annotations

import json
import re
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from .llm_client import LLMClientError, LLMMessage, LLMTimeoutError
from .settings import LLMSettings


FENCED_JSON_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class TopicNamingClient(Protocol):
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


class TopicNamingError(Exception):
    pass


class TopicName(BaseModel):
    cluster_id: int = Field(ge=0)
    title: str = Field(min_length=1, max_length=100)
    summary: str = Field(min_length=1, max_length=500)


class TopicNamesPayload(BaseModel):
    topics: list[TopicName]


def name_topic_clusters(
    clusters: list[dict[str, object]],
    *,
    llm_client: TopicNamingClient,
    model: str | None,
) -> dict[int, TopicName]:
    messages = [
        LLMMessage(
            role="system",
            content=(
                "Name clusters of grounded course knowledge cards. Return a "
                "concise course-topic title and one-sentence summary for every "
                "cluster. Do not invent concepts outside the cards. Return JSON "
                "only: {\"topics\":[{\"cluster_id\":0,\"title\":\"...\","
                "\"summary\":\"...\"}]}"
            ),
        ),
        LLMMessage(
            role="user",
            content=json.dumps({"clusters": clusters}, ensure_ascii=False),
        ),
    ]
    try:
        output = llm_client.create_chat_completion(
            messages,
            model=model,
            temperature=0.0,
            max_tokens=1600,
            response_format={"type": "json_object"},
        )
    except (LLMClientError, LLMTimeoutError) as exc:
        raise TopicNamingError(str(exc)) from exc

    cleaned = THINK_RE.sub("", output).strip()
    match = FENCED_JSON_RE.search(cleaned)
    if match:
        cleaned = match.group(1).strip()
    try:
        payload = TopicNamesPayload.model_validate(json.loads(cleaned))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise TopicNamingError("Local LLM returned invalid topic names.") from exc
    return {topic.cluster_id: topic for topic in payload.topics}
