import json
from datetime import datetime
from sqlite3 import Row

from .db import connect, ensure_db
from .knowledge_card_note import (
    KnowledgeCardNote,
    KnowledgeCardNoteReference,
)


def _datetime_to_text(value: datetime) -> str:
    return value.isoformat()


def _datetime_from_text(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _sources_to_json(note: KnowledgeCardNote) -> str:
    return json.dumps(
        [
            source.model_dump(mode="json")
            for source in note.sources
        ],
        ensure_ascii=False,
    )


def _sources_from_json(value: str) -> list[KnowledgeCardNoteReference]:
    raw_sources = json.loads(value)

    if not isinstance(raw_sources, list):
        return []

    return [
        KnowledgeCardNoteReference.model_validate(source)
        for source in raw_sources
        if isinstance(source, dict)
    ]


def _row_to_note(row: Row) -> KnowledgeCardNote:
    return KnowledgeCardNote(
        id=row["id"],
        card_id=row["card_id"],
        note_type=row["note_type"],
        title=row["title"],
        body=row["body"],
        source=row["source"],
        sources=_sources_from_json(row["sources_json"]),
        created_at=_datetime_from_text(row["created_at"]),
        updated_at=_datetime_from_text(row["updated_at"]),
    )


def create_note(note: KnowledgeCardNote) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO knowledge_card_notes (
                id,
                card_id,
                note_type,
                title,
                body,
                source,
                sources_json,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                note.id,
                note.card_id,
                note.note_type,
                note.title,
                note.body,
                note.source,
                _sources_to_json(note),
                _datetime_to_text(note.created_at),
                _datetime_to_text(note.updated_at),
            ),
        )


def get_note(note_id: str) -> KnowledgeCardNote | None:
    ensure_db()

    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM knowledge_card_notes WHERE id = ?",
            (note_id,),
        ).fetchone()

    if row is None:
        return None

    return _row_to_note(row)


def list_notes_for_card(card_id: str) -> list[KnowledgeCardNote]:
    ensure_db()

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM knowledge_card_notes
            WHERE card_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (card_id,),
        ).fetchall()

    return [_row_to_note(row) for row in rows]


def update_note(note: KnowledgeCardNote) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            """
            UPDATE knowledge_card_notes
            SET note_type = ?,
                title = ?,
                body = ?,
                source = ?,
                sources_json = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                note.note_type,
                note.title,
                note.body,
                note.source,
                _sources_to_json(note),
                _datetime_to_text(note.updated_at),
                note.id,
            ),
        )


def delete_note(note_id: str) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            "DELETE FROM knowledge_card_notes WHERE id = ?",
            (note_id,),
        )


def delete_notes_for_card(card_id: str) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            "DELETE FROM knowledge_card_notes WHERE card_id = ?",
            (card_id,),
        )


def clear_notes() -> None:
    ensure_db()

    with connect() as conn:
        conn.execute("DELETE FROM knowledge_card_notes")
