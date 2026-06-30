import json
import re
from dataclasses import dataclass
from time import perf_counter
from collections.abc import Iterable
from typing import Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from . import job_service
from .knowledge_card import KnowledgeCardClaim, KnowledgeCardEvidence
from .llm_client import LLMClientError, LLMMessage, LLMTimeoutError
from .settings import LLMSettings


Difficulty = Literal["easy", "medium", "hard"]
MAX_CONTEXT_CHARACTERS = 4000
MAX_SELECTED_SEGMENTS = 80

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


class CardGenerationTimeoutError(CardGenerationError):
    pass


class NoGroundedClaimsError(CardGenerationError):
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
    claims: list[KnowledgeCardClaim] = Field(min_length=1)
    unsupported_terms: list[str] = Field(default_factory=list)
    question: str
    answer: str
    difficulty: Difficulty = "medium"
    source_start_seconds: float
    source_end_seconds: float


class CardGenerationMetadata(BaseModel):
    provider: str
    model: str
    elapsed_seconds: float
    input_characters: int
    selected_context_characters: int
    selected_segments_count: int
    requested_card_count: int
    raw_card_count: int
    returned_card_count: int
    raw_claim_count: int
    grounded_claim_count: int
    dropped_claim_count: int
    unsupported_terms_count: int
    max_context_characters: int
    max_selected_segments: int


class CardDraftResponse(BaseModel):
    job_id: str
    source_video: str
    start_seconds: float
    end_seconds: float
    provider: str
    model: str
    generation_metadata: CardGenerationMetadata
    cards: list[KnowledgeCardDraft]


class _LLMCardClaim(BaseModel):
    text: str
    evidence_quotes: list[str] = Field(default_factory=list)


class _LLMKnowledgeCard(BaseModel):
    title: str
    summary: str
    key_points: list[str] = Field(default_factory=list)
    claims: list[_LLMCardClaim] = Field(default_factory=list)
    question: str
    answer: str
    difficulty: Difficulty = "medium"


class _LLMCardPayload(BaseModel):
    cards: list[_LLMKnowledgeCard]


@dataclass(frozen=True)
class _GroundingResult:
    cards: list[KnowledgeCardDraft]
    raw_card_count: int
    raw_claim_count: int
    grounded_claim_count: int
    dropped_claim_count: int
    unsupported_terms_count: int


def draft_knowledge_cards(
    request: CardDraftRequest,
    *,
    llm_client: CardLLMClient,
) -> CardDraftResponse:
    started_at = perf_counter()

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

    _validate_context_window(context)

    messages = _build_generation_messages(
        request=request,
        context=context,
    )
    selected_model = _normalize_model_name(request.model)
    response_model = selected_model or llm_client.settings.model
    input_characters = sum(
        len(message.content)
        for message in messages
    )

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

    grounding_result = _build_grounded_cards(
        payload.cards,
        context,
        limit=request.card_count,
    )

    if not grounding_result.cards:
        raise NoGroundedClaimsError(
            "No grounded claims were found in the model output. Try "
            "selecting fewer transcript segments or using a larger model."
        )

    return CardDraftResponse(
        job_id=context.job_id,
        source_video=context.source_video,
        start_seconds=context.start_seconds,
        end_seconds=context.end_seconds,
        provider=llm_client.settings.provider,
        model=response_model,
        generation_metadata=CardGenerationMetadata(
            provider=llm_client.settings.provider,
            model=response_model,
            elapsed_seconds=round(perf_counter() - started_at, 3),
            input_characters=input_characters,
            selected_context_characters=len(context.text),
            selected_segments_count=len(context.segments),
            requested_card_count=request.card_count,
            raw_card_count=grounding_result.raw_card_count,
            returned_card_count=len(grounding_result.cards),
            raw_claim_count=grounding_result.raw_claim_count,
            grounded_claim_count=grounding_result.grounded_claim_count,
            dropped_claim_count=grounding_result.dropped_claim_count,
            unsupported_terms_count=(
                grounding_result.unsupported_terms_count
            ),
            max_context_characters=MAX_CONTEXT_CHARACTERS,
            max_selected_segments=MAX_SELECTED_SEGMENTS,
        ),
        cards=grounding_result.cards,
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
    except LLMTimeoutError as exc:
        raise CardGenerationTimeoutError(
            "Local LLM request timed out. Try selecting fewer transcript "
            "segments or using a smaller/faster model."
        ) from exc
    except LLMClientError as exc:
        raise CardGenerationError(
            f"Local LLM unavailable: {exc}"
        ) from exc

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
        except LLMTimeoutError as exc:
            raise CardGenerationTimeoutError(
                "Local LLM request timed out while retrying empty output. "
                "Try selecting fewer transcript segments."
            ) from exc
        except LLMClientError as exc:
            raise CardGenerationError(
                f"Local LLM unavailable: {exc}"
            ) from exc

    if not output.strip():
        raise CardGenerationError(
            "Local LLM returned empty content. Try again, select a "
            "shorter transcript window, or increase VCC_LLM_MAX_TOKENS."
        )

    return output


def _validate_context_window(
    context: job_service.TranscriptContext,
) -> None:
    context_characters = len(context.text)
    segment_count = len(context.segments)

    if context_characters > MAX_CONTEXT_CHARACTERS:
        raise InvalidCardDraftRequestError(
            "Selected context is too long: "
            f"{context_characters} characters. Please select fewer "
            f"transcript segments. Current limit is "
            f"{MAX_CONTEXT_CHARACTERS} characters."
        )

    if segment_count > MAX_SELECTED_SEGMENTS:
        raise InvalidCardDraftRequestError(
            "Selected context has too many segments: "
            f"{segment_count}. Please select fewer transcript segments. "
            f"Current limit is {MAX_SELECTED_SEGMENTS} segments."
        )


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
Create compact study cards from transcript evidence.

Rules:
1. JSON only. No Markdown, explanations, or <think>.
2. Use the transcript language unless the user focus asks otherwise.
3. Use only facts in the transcript.
4. Return no more than the requested number of cards.
5. Each card needs 1-2 atomic claims.
6. Each claim needs 1-2 evidence_quotes copied exactly from one transcript line.
7. Evidence quotes must not include timestamps.
8. If evidence is weak, return {"cards": []}.
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
{_format_transcript_evidence(context.segments)}
TRANSCRIPT>>>

Return exactly this JSON shape:
{{
  "cards": [
    {{
      "title": "short concept title",
      "summary": "clear explanation grounded in the transcript",
      "key_points": ["important point 1", "important point 2"],
      "claims": [
        {{
          "text": "one atomic factual claim",
          "evidence_quotes": ["exact phrase copied from one transcript line"]
        }}
      ],
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


def _format_transcript_evidence(
    segments: list[job_service.TranscriptSegment],
) -> str:
    return "\n".join(
        (
            f"[{segment.start_seconds:.2f}-{segment.end_seconds:.2f}] "
            f"{segment.text}"
        )
        for segment in segments
    )


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
      "claims": [
        {{
          "text": "one atomic factual claim",
          "evidence_quotes": ["exact phrase from transcript"]
        }}
      ],
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


def _build_grounded_cards(
    llm_cards: list[_LLMKnowledgeCard],
    context: job_service.TranscriptContext,
    *,
    limit: int,
) -> _GroundingResult:
    grounded_cards: list[KnowledgeCardDraft] = []
    raw_claim_count = sum(
        len(card.claims)
        for card in llm_cards
    )
    grounded_claim_count = 0
    unsupported_terms_count = 0

    for llm_card in llm_cards:
        claims = _ground_claims(llm_card.claims, context.segments)

        if not claims:
            continue

        grounded_claim_count += len(claims)

        evidence_items = [
            evidence
            for claim in claims
            for evidence in claim.evidence
        ]
        source_start_seconds = min(
            evidence.segment_start_seconds
            for evidence in evidence_items
        )
        source_end_seconds = max(
            evidence.segment_end_seconds
            for evidence in evidence_items
        )

        unsupported_terms = _find_unsupported_terms(
            llm_card,
            claims,
            context.segments,
        )
        unsupported_terms_count += len(unsupported_terms)

        grounded_cards.append(
            KnowledgeCardDraft(
                title=llm_card.title.strip(),
                summary=llm_card.summary.strip(),
                key_points=[
                    point.strip()
                    for point in llm_card.key_points
                    if point.strip()
                ],
                claims=claims,
                unsupported_terms=unsupported_terms,
                question=llm_card.question.strip(),
                answer=llm_card.answer.strip(),
                difficulty=llm_card.difficulty,
                source_start_seconds=source_start_seconds,
                source_end_seconds=source_end_seconds,
            )
        )

        if len(grounded_cards) >= limit:
            break

    return _GroundingResult(
        cards=grounded_cards,
        raw_card_count=len(llm_cards),
        raw_claim_count=raw_claim_count,
        grounded_claim_count=grounded_claim_count,
        dropped_claim_count=max(raw_claim_count - grounded_claim_count, 0),
        unsupported_terms_count=unsupported_terms_count,
    )


def _ground_claims(
    llm_claims: list[_LLMCardClaim],
    segments: list[job_service.TranscriptSegment],
) -> list[KnowledgeCardClaim]:
    grounded_claims: list[KnowledgeCardClaim] = []

    for llm_claim in llm_claims:
        claim_text = llm_claim.text.strip()

        if not claim_text:
            continue

        evidence_items = [
            evidence
            for quote in llm_claim.evidence_quotes
            if (evidence := _match_evidence_quote(quote, segments))
            is not None
        ]
        evidence_items = _dedupe_evidence(evidence_items)

        if evidence_items:
            grounded_claims.append(
                KnowledgeCardClaim(
                    text=claim_text,
                    evidence=evidence_items,
                )
            )

    return grounded_claims


def _match_evidence_quote(
    quote: str,
    segments: list[job_service.TranscriptSegment],
) -> KnowledgeCardEvidence | None:
    clean_quote = _clean_evidence_quote(quote)
    normalized_quote = _normalize_for_match(clean_quote)

    if len(normalized_quote) < 6:
        return None

    for segment in segments:
        if normalized_quote in _normalize_for_match(segment.text):
            return KnowledgeCardEvidence(
                quote=clean_quote,
                segment_start_seconds=segment.start_seconds,
                segment_end_seconds=segment.end_seconds,
            )

    return None


def _clean_evidence_quote(quote: str) -> str:
    clean_quote = quote.strip().strip('"').strip("'").strip()

    return re.sub(
        r"^\[\s*\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?\s*\]\s*",
        "",
        clean_quote,
    ).strip()


def _normalize_for_match(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _dedupe_evidence(
    evidence_items: list[KnowledgeCardEvidence],
) -> list[KnowledgeCardEvidence]:
    seen: set[tuple[str, float, float]] = set()
    deduped: list[KnowledgeCardEvidence] = []

    for evidence in evidence_items:
        key = (
            _normalize_for_match(evidence.quote),
            evidence.segment_start_seconds,
            evidence.segment_end_seconds,
        )

        if key in seen:
            continue

        seen.add(key)
        deduped.append(evidence)

    return deduped


def _find_unsupported_terms(
    llm_card: _LLMKnowledgeCard,
    claims: list[KnowledgeCardClaim],
    segments: list[job_service.TranscriptSegment],
) -> list[str]:
    context_terms = _terms(
        segment.text
        for segment in segments
    )

    if not context_terms:
        return []

    card_terms = _terms(
        [
            llm_card.title,
            *[
                claim.text
                for claim in claims
            ],
        ]
    )

    return sorted(card_terms - context_terms)[:12]


def _terms(values: Iterable[str]) -> set[str]:
    terms: set[str] = set()

    for value in values:
        for match in TOKEN_RE.finditer(value.lower()):
            term = match.group(0)

            if term in STOPWORDS:
                continue

            terms.add(term)

    return terms
