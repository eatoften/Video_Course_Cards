import json
import re
from collections.abc import Iterable
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
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{3,}")
STOPWORDS = {
    "about",
    "also",
    "answer",
    "because",
    "being",
    "card",
    "clear",
    "concept",
    "course",
    "data",
    "does",
    "from",
    "have",
    "into",
    "only",
    "point",
    "question",
    "should",
    "that",
    "their",
    "there",
    "these",
    "this",
    "those",
    "with",
    "what",
    "when",
    "where",
    "which",
    "will",
    "would",
}


class CardLLMClient(Protocol):
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
    model: str | None = Field(default=None, max_length=200)


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
    selected_model = _normalize_model_name(request.model)

    raw_output = _call_llm(
        llm_client,
        messages,
        model=selected_model,
        max_tokens=_card_generation_max_tokens(llm_client),
    )

    try:
        payload = _parse_card_payload(raw_output)
    except CardOutputParseError:
        repair_messages = _build_repair_messages(raw_output)
        repaired_output = _call_llm(
            llm_client,
            repair_messages,
            model=selected_model,
            max_tokens=_card_generation_max_tokens(llm_client),
        )
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
    cards = _filter_grounded_cards(cards, context.text)

    if not cards:
        raise CardGenerationError(
            "Local LLM generated cards that were not grounded in the "
            "selected transcript. Try a shorter window or a larger model."
        )

    return CardDraftResponse(
        job_id=context.job_id,
        source_video=context.source_video,
        start_seconds=context.start_seconds,
        end_seconds=context.end_seconds,
        provider=llm_client.settings.provider,
        model=selected_model or llm_client.settings.model,
        cards=cards,
    )


def _call_llm(
    llm_client: CardLLMClient,
    messages: list[LLMMessage],
    *,
    model: str | None,
    max_tokens: int,
) -> str:
    try:
        output = llm_client.create_chat_completion(
            messages,
            model=model,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
    except LLMClientError as exc:
        raise CardGenerationError(str(exc)) from exc

    if not output.strip():
        retry_max_tokens = max(
            llm_client.settings.max_tokens * 2,
            8192,
        )

        try:
            output = llm_client.create_chat_completion(
                messages,
                model=model,
                max_tokens=max(retry_max_tokens, max_tokens * 2),
                response_format={"type": "json_object"},
            )
        except LLMClientError as exc:
            raise CardGenerationError(str(exc)) from exc

    if not output.strip():
        raise CardGenerationError(
            "Local LLM returned empty content. Try again, select a "
            "shorter transcript window, or increase VCC_LLM_MAX_TOKENS."
        )

    return output


def _card_generation_max_tokens(
    llm_client: CardLLMClient,
) -> int:
    return max(llm_client.settings.max_tokens, 8192)


def _normalize_model_name(model: str | None) -> str | None:
    if model is None:
        return None

    stripped = model.strip()

    return stripped or None


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
8. Each title, summary, question, and answer must use concepts that appear in
   the transcript evidence.
9. If the transcript does not contain enough evidence, return {"cards": []}.
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
<<<TRANSCRIPT
{context.text}
TRANSCRIPT>>>

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


def _filter_grounded_cards(
    cards: list[KnowledgeCardDraft],
    context_text: str,
) -> list[KnowledgeCardDraft]:
    context_terms = _terms([context_text])

    if not context_terms:
        return cards

    grounded_cards: list[KnowledgeCardDraft] = []

    for card in cards:
        card_terms = _terms(
            [
                card.title,
                card.summary,
                *card.key_points,
                card.question,
                card.answer,
            ]
        )

        if len(card_terms & context_terms) >= 2:
            grounded_cards.append(card)

    return grounded_cards


def _terms(values: Iterable[str]) -> set[str]:
    terms: set[str] = set()

    for value in values:
        for match in TOKEN_RE.finditer(value.lower()):
            term = match.group(0)

            if term in STOPWORDS:
                continue

            terms.add(term)

    return terms
