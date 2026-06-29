from uuid import uuid4

from .job import utc_now
from .knowledge_card_note import (
    KnowledgeCardNote,
    KnowledgeCardNoteCreate,
    KnowledgeCardNoteReference,
    KnowledgeCardNoteUpdate,
)
from .knowledge_card_note_store import (
    create_note,
    delete_note,
    get_note,
    list_notes_for_card,
    update_note,
)
from .knowledge_card_store import get_card


class KnowledgeCardNoteServiceError(Exception):
    pass


class KnowledgeCardNoteNotFoundError(KnowledgeCardNoteServiceError):
    pass


class InvalidKnowledgeCardNoteError(KnowledgeCardNoteServiceError):
    pass


class KnowledgeCardForNoteNotFoundError(KnowledgeCardNoteServiceError):
    pass


def list_card_notes(card_id: str) -> list[KnowledgeCardNote]:
    _ensure_card_exists(card_id)

    return list_notes_for_card(card_id)


def save_card_note(
    card_id: str,
    request: KnowledgeCardNoteCreate,
) -> KnowledgeCardNote:
    _ensure_card_exists(card_id)

    now = utc_now()
    note = KnowledgeCardNote(
        id=uuid4().hex,
        card_id=card_id,
        note_type=request.note_type,
        title=_clean_optional_text(request.title),
        body=_clean_required_text(
            request.body,
            "Note body is required.",
        ),
        source=request.source,
        sources=_clean_sources(request.sources),
        created_at=now,
        updated_at=now,
    )

    create_note(note)

    return note


def update_card_note(
    note_id: str,
    request: KnowledgeCardNoteUpdate,
) -> KnowledgeCardNote:
    note = get_note(note_id)

    if note is None:
        raise KnowledgeCardNoteNotFoundError("Knowledge card note not found.")

    update_data = request.model_dump(exclude_unset=True)

    if "note_type" in update_data and request.note_type is not None:
        note.note_type = request.note_type

    if "title" in update_data:
        note.title = _clean_optional_text(request.title)

    if "body" in update_data and request.body is not None:
        note.body = _clean_required_text(
            request.body,
            "Note body is required.",
        )

    if "source" in update_data and request.source is not None:
        note.source = request.source

    if "sources" in update_data and request.sources is not None:
        note.sources = _clean_sources(request.sources)

    note.updated_at = utc_now()
    update_note(note)

    return note


def delete_card_note(note_id: str) -> None:
    note = get_note(note_id)

    if note is None:
        raise KnowledgeCardNoteNotFoundError("Knowledge card note not found.")

    delete_note(note_id)


def _ensure_card_exists(card_id: str) -> None:
    if get_card(card_id) is None:
        raise KnowledgeCardForNoteNotFoundError("Knowledge card not found.")


def _clean_required_text(value: str, error_message: str) -> str:
    stripped = value.strip()

    if not stripped:
        raise InvalidKnowledgeCardNoteError(error_message)

    return stripped


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None

    stripped = value.strip()

    return stripped or None


def _clean_sources(
    sources: list[KnowledgeCardNoteReference],
) -> list[KnowledgeCardNoteReference]:
    cleaned_sources: list[KnowledgeCardNoteReference] = []

    for source in sources:
        title = _clean_optional_text(source.title)
        url = _clean_optional_text(source.url)

        if title is None and url is None:
            continue

        cleaned_sources.append(
            KnowledgeCardNoteReference(
                title=title,
                url=url,
                accessed_at=source.accessed_at,
            )
        )

    return cleaned_sources
