import json
import re
from typing import Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from . import job_service
from .llm_client import LLMClientError, LLMMessage
from .settings import LLMSettings


Difficulty = Literal["easy", "medium", "hard"]

THINK_BLOCK_RE = re.compile(
    r"<think>.*?</think>",
    flags=re.DOTALL | re.IGNORECASE,
)
FENCED_JSON_RE = re.compile(
    r"```(?:json)?\s*([\s\S]*?)```",
    flags=re.IGNORECASE,
)


class CardLLMClient(Protocol):
    settings: LLMSettings

    def create_chat_completion(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        pass


class CardServiceError(Exception):
    pass


class InvalidCardDraftRequestError(CardServiceError):
    pass


class CardGenerationError(CardServiceError):
    pass


class CardOutputParseError(CardServiceError):
    pass


class CardDraftRequest(BaseModel):
    job_id: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    card_count: int = Field(default=3, ge=1, le=6)
    focus: str | None = Field(default=None, max_length=500)


class KnowledgeCardDraft(BaseModel):
    title: str
    summary: str
    key_points: list[str] = Field(default_factory=list)
    question: str
    answer: str
    difficulty: Difficulty = "medium"
    source_start_seconds: float
    source_end_seconds: float


class CardDraftResponse(BaseModel):
    job_id: str
    source_video: str
    start_seconds: float
    end_seconds: float
    provider: str
    model: str
    cards: list[KnowledgeCardDraft]


class _LLMKnowledgeCard(BaseModel):
    title: str
    summary: str
    key_points: list[str] = Field(default_factory=list)
    question: str
    answer: str
    difficulty: Difficulty = "medium"


class _LLMCardPayload(BaseModel):
    cards: list[_LLMKnowledgeCard]


def draft_knowledge_cards(
    request: CardDraftRequest,
    *,
    llm_client: CardLLMClient,
) -> CardDraftResponse:
    if request.end_seconds <= request.start_seconds:
        raise InvalidCardDraftRequestError(
            "Card draft end must be greater than start."
        )

    context = job_service.get_transcript_context(
        request.job_id,
        request.start_seconds,
        request.end_seconds,
    )

    if not context.text.strip():
        raise InvalidCardDraftRequestError(
            "Selected transcript context is empty."
        )

    messages = _build_generation_messages(
        request=request,
        context=context,
    )

    raw_output = _call_llm(llm_client, messages)

    try:
        payload = _parse_card_payload(raw_output)
    except CardOutputParseError:
        repair_messages = _build_repair_messages(raw_output)
        repaired_output = _call_llm(llm_client, repair_messages)
        payload = _parse_card_payload(repaired_output)

    cards = [
        KnowledgeCardDraft(
            title=card.title.strip(),
            summary=card.summary.strip(),
            key_points=[
                point.strip()
                for point in card.key_points
                if point.strip()
            ],
            question=card.question.strip(),
            answer=card.answer.strip(),
            difficulty=card.difficulty,
            source_start_seconds=context.start_seconds,
            source_end_seconds=context.end_seconds,
        )
        for card in payload.cards[: request.card_count]
    ]

    if not cards:
        raise CardOutputParseError(
            "Local LLM did not return any cards."
        )

    return CardDraftResponse(
        job_id=context.job_id,
        source_video=context.source_video,
        start_seconds=context.start_seconds,
        end_seconds=context.end_seconds,
        provider=llm_client.settings.provider,
        model=llm_client.settings.model,
        cards=cards,
    )


def _call_llm(
    llm_client: CardLLMClient,
    messages: list[LLMMessage],
) -> str:
    try:
        return llm_client.create_chat_completion(messages)
    except LLMClientError as exc:
        raise CardGenerationError(str(exc)) from exc


def _build_generation_messages(
    *,
    request: CardDraftRequest,
    context: job_service.TranscriptContext,
) -> list[LLMMessage]:
    focus = request.focus.strip() if request.focus else "general review"

    system_prompt = """
You are a course knowledge-card generator.
Convert selected transcript evidence into concise, reviewable study cards.

Rules:
1. Return valid JSON only.
2. Do not return Markdown.
3. Do not explain the JSON.
4. Do not output <think> content.
5. Use the same language as the transcript unless the user focus asks otherwise.
6. Every card must be grounded only in the provided transcript.
7. Do not invent facts outside the transcript.
""".strip()

    user_prompt = f"""
/no_think

Source video:
{context.source_video}

Selected time range:
{context.start_seconds:.2f} - {context.end_seconds:.2f} seconds

Requested number of cards:
{request.card_count}

User focus:
{focus}

Transcript evidence:
{context.text}

Return exactly this JSON shape:
{{
  "cards": [
    {{
      "title": "short concept title",
      "summary": "clear explanation grounded in the transcript",
      "key_points": ["important point 1", "important point 2"],
      "question": "active recall question",
      "answer": "short reference answer",
      "difficulty": "easy"
    }}
  ]
}}
""".strip()

    return [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=user_prompt),
    ]


def _build_repair_messages(raw_output: str) -> list[LLMMessage]:
    system_prompt = """
You repair malformed model output into valid JSON.
Return JSON only. Do not add Markdown, explanation, or <think> content.
""".strip()

    user_prompt = f"""
/no_think

The following output should have been valid JSON with this shape:
{{
  "cards": [
    {{
      "title": "short concept title",
      "summary": "clear explanation",
      "key_points": ["point"],
      "question": "active recall question",
      "answer": "reference answer",
      "difficulty": "easy"
    }}
  ]
}}

Repair it into valid JSON only:
{raw_output}
""".strip()

    return [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=user_prompt),
    ]


def _parse_card_payload(raw_output: str) -> _LLMCardPayload:
    json_text = _extract_json(raw_output)

    try:
        data = json.loads(json_text)
        payload = _LLMCardPayload.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise CardOutputParseError(
            "Local LLM output was not valid card JSON."
        ) from exc

    return payload


def _extract_json(raw_output: str) -> str:
    without_thinking = THINK_BLOCK_RE.sub("", raw_output).strip()
    fenced_match = FENCED_JSON_RE.search(without_thinking)

    if fenced_match:
        return fenced_match.group(1).strip()

    start = without_thinking.find("{")
    end = without_thinking.rfind("}")

    if start != -1 and end != -1 and end > start:
        return without_thinking[start : end + 1]

    return without_thinking
