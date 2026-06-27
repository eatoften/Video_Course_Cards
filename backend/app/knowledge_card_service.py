from uuid import uuid4

from .job import utc_now
from .job_service import get_video_job
from .knowledge_card import (
    KnowledgeCard,
    KnowledgeCardCreate,
    KnowledgeCardUpdate,
)
from .knowledge_card_store import (
    create_card,
    delete_card,
    get_card,
    list_cards_for_job,
    update_card,
)


class KnowledgeCardServiceError(Exception):
    pass


class KnowledgeCardNotFoundError(KnowledgeCardServiceError):
    pass


class InvalidKnowledgeCardError(KnowledgeCardServiceError):
    pass


def list_job_cards(job_id: str) -> list[KnowledgeCard]:
    get_video_job(job_id)

    return list_cards_for_job(job_id)


def save_job_card(
    job_id: str,
    request: KnowledgeCardCreate,
) -> KnowledgeCard:
    get_video_job(job_id)
    _validate_time_range(
        request.source_start_seconds,
        request.source_end_seconds,
    )

    now = utc_now()
    card = KnowledgeCard(
        id=uuid4().hex,
        job_id=job_id,
        title=request.title.strip(),
        summary=request.summary.strip(),
        key_points=_clean_key_points(request.key_points),
        question=_clean_optional_text(request.question),
        answer=_clean_optional_text(request.answer),
        difficulty=request.difficulty,
        source_start_seconds=request.source_start_seconds,
        source_end_seconds=request.source_end_seconds,
        provider=_clean_optional_text(request.provider),
        model=_clean_optional_text(request.model),
        created_at=now,
        updated_at=now,
    )

    create_card(card)

    return card


def update_saved_card(
    card_id: str,
    request: KnowledgeCardUpdate,
) -> KnowledgeCard:
    card = get_card(card_id)

    if card is None:
        raise KnowledgeCardNotFoundError("Knowledge card not found.")

    update_data = request.model_dump(exclude_unset=True)

    if "title" in update_data and request.title is not None:
        card.title = request.title.strip()

    if "summary" in update_data and request.summary is not None:
        card.summary = request.summary.strip()

    if "key_points" in update_data and request.key_points is not None:
        card.key_points = _clean_key_points(request.key_points)

    if "question" in update_data:
        card.question = _clean_optional_text(request.question)

    if "answer" in update_data:
        card.answer = _clean_optional_text(request.answer)

    if "difficulty" in update_data and request.difficulty is not None:
        card.difficulty = request.difficulty

    if (
        "source_start_seconds" in update_data
        and request.source_start_seconds is not None
    ):
        card.source_start_seconds = request.source_start_seconds

    if (
        "source_end_seconds" in update_data
        and request.source_end_seconds is not None
    ):
        card.source_end_seconds = request.source_end_seconds

    if "provider" in update_data:
        card.provider = _clean_optional_text(request.provider)

    if "model" in update_data:
        card.model = _clean_optional_text(request.model)

    _validate_time_range(
        card.source_start_seconds,
        card.source_end_seconds,
    )

    card.updated_at = utc_now()
    update_card(card)

    return card


def delete_saved_card(card_id: str) -> None:
    card = get_card(card_id)

    if card is None:
        raise KnowledgeCardNotFoundError("Knowledge card not found.")

    delete_card(card_id)


def _validate_time_range(
    start_seconds: float,
    end_seconds: float,
) -> None:
    if end_seconds <= start_seconds:
        raise InvalidKnowledgeCardError(
            "Card source end must be greater than start."
        )


def _clean_key_points(key_points: list[str]) -> list[str]:
    return [
        point.strip()
        for point in key_points
        if point.strip()
    ]


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None

    stripped = value.strip()

    return stripped or None
