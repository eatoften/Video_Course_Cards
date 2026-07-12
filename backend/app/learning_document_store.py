from __future__ import annotations

import json
from datetime import datetime, timezone
from sqlite3 import Row

from .db import connect, ensure_db
from .learning_document import (
    LearningDocument,
    LearningDocumentCardLink,
    LearningDocumentDetail,
    LearningDocumentSource,
    LearningDocumentVersion,
)


def _to_text(value: datetime) -> str:
    return value.isoformat()


def _from_text(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _row_to_document(row: Row) -> LearningDocument:
    return LearningDocument(
        id=row["id"], course_id=row["course_id"], title=row["title"],
        summary=row["summary"], body_markdown=row["body_markdown"],
        status=row["status"], generation_mode=row["generation_mode"],
        provider=row["provider"], model=row["model"],
        created_at=_from_text(row["created_at"]),
        updated_at=_from_text(row["updated_at"]),
    )


def _row_to_link(row: Row) -> LearningDocumentCardLink:
    return LearningDocumentCardLink(
        id=row["id"], document_id=row["document_id"], card_id=row["card_id"],
        role=row["role"], position=row["position"],
        created_at=_from_text(row["created_at"]),
    )


def _row_to_source(row: Row) -> LearningDocumentSource:
    return LearningDocumentSource(
        id=row["id"], document_id=row["document_id"],
        source_type=row["source_type"], source_id=row["source_id"],
        card_id=row["card_id"], label=row["label"], quote=row["quote"],
        locator=json.loads(row["locator_json"]), position=row["position"],
        created_at=_from_text(row["created_at"]),
    )


def _row_to_version(row: Row) -> LearningDocumentVersion:
    return LearningDocumentVersion(
        id=row["id"], document_id=row["document_id"],
        version_number=row["version_number"], title=row["title"],
        summary=row["summary"], body_markdown=row["body_markdown"],
        change_source=row["change_source"], provider=row["provider"],
        model=row["model"], created_at=_from_text(row["created_at"]),
    )


def create_learning_document(document: LearningDocument) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO learning_documents (
                id, course_id, title, summary, body_markdown, status,
                generation_mode, provider, model, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document.id, document.course_id, document.title,
                document.summary, document.body_markdown, document.status,
                document.generation_mode, document.provider, document.model,
                _to_text(document.created_at), _to_text(document.updated_at),
            ),
        )


def update_learning_document(document: LearningDocument) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute(
            """
            UPDATE learning_documents
            SET title = ?, summary = ?, body_markdown = ?, status = ?,
                generation_mode = ?, provider = ?, model = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                document.title, document.summary, document.body_markdown,
                document.status, document.generation_mode, document.provider,
                document.model, _to_text(document.updated_at), document.id,
            ),
        )


def get_learning_document(document_id: str) -> LearningDocument | None:
    ensure_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM learning_documents WHERE id = ?",
            (document_id,),
        ).fetchone()
    return _row_to_document(row) if row is not None else None


def list_learning_documents_for_course(course_id: str) -> list[LearningDocument]:
    ensure_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM learning_documents
            WHERE course_id = ? ORDER BY updated_at DESC
            """,
            (course_id,),
        ).fetchall()
    return [_row_to_document(row) for row in rows]


def list_learning_documents_for_card(card_id: str) -> list[LearningDocument]:
    ensure_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT learning_documents.*
            FROM learning_documents
            INNER JOIN learning_document_cards
                ON learning_document_cards.document_id = learning_documents.id
            WHERE learning_document_cards.card_id = ?
            ORDER BY learning_documents.updated_at DESC
            """,
            (card_id,),
        ).fetchall()
    return [_row_to_document(row) for row in rows]


def upsert_document_card_link(link: LearningDocumentCardLink) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO learning_document_cards (
                id, document_id, card_id, role, position, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id, card_id) DO UPDATE SET
                role = excluded.role, position = excluded.position
            """,
            (
                link.id, link.document_id, link.card_id, link.role,
                link.position, _to_text(link.created_at),
            ),
        )


def delete_document_card_link(document_id: str, card_id: str) -> bool:
    ensure_db()
    with connect() as conn:
        cursor = conn.execute(
            """
            DELETE FROM learning_document_cards
            WHERE document_id = ? AND card_id = ?
            """,
            (document_id, card_id),
        )
    return cursor.rowcount > 0


def list_document_card_links(document_id: str) -> list[LearningDocumentCardLink]:
    ensure_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM learning_document_cards
            WHERE document_id = ? ORDER BY position, created_at
            """,
            (document_id,),
        ).fetchall()
    return [_row_to_link(row) for row in rows]


def replace_document_sources(
    document_id: str,
    sources: list[LearningDocumentSource],
) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute(
            "DELETE FROM learning_document_sources WHERE document_id = ?",
            (document_id,),
        )
        conn.executemany(
            """
            INSERT INTO learning_document_sources (
                id, document_id, source_type, source_id, card_id, label,
                quote, locator_json, position, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    source.id, source.document_id, source.source_type,
                    source.source_id, source.card_id, source.label,
                    source.quote, json.dumps(source.locator, ensure_ascii=False),
                    source.position, _to_text(source.created_at),
                )
                for source in sources
            ],
        )


def list_document_sources(document_id: str) -> list[LearningDocumentSource]:
    ensure_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM learning_document_sources
            WHERE document_id = ? ORDER BY position, created_at
            """,
            (document_id,),
        ).fetchall()
    return [_row_to_source(row) for row in rows]


def create_document_version(version: LearningDocumentVersion) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO learning_document_versions (
                id, document_id, version_number, title, summary,
                body_markdown, change_source, provider, model, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version.id, version.document_id, version.version_number,
                version.title, version.summary, version.body_markdown,
                version.change_source, version.provider, version.model,
                _to_text(version.created_at),
            ),
        )


def next_document_version_number(document_id: str) -> int:
    ensure_db()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT COALESCE(MAX(version_number), 0) + 1 AS next_number
            FROM learning_document_versions WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()
    return int(row["next_number"])


def list_document_versions(document_id: str) -> list[LearningDocumentVersion]:
    ensure_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM learning_document_versions
            WHERE document_id = ? ORDER BY version_number DESC
            """,
            (document_id,),
        ).fetchall()
    return [_row_to_version(row) for row in rows]


def get_learning_document_detail(document_id: str) -> LearningDocumentDetail | None:
    document = get_learning_document(document_id)
    if document is None:
        return None
    return LearningDocumentDetail(
        **document.model_dump(),
        card_links=list_document_card_links(document_id),
        sources=list_document_sources(document_id),
        versions=list_document_versions(document_id),
    )


def delete_learning_document(document_id: str) -> bool:
    ensure_db()
    with connect() as conn:
        conn.execute(
            "DELETE FROM learning_document_sources WHERE document_id = ?",
            (document_id,),
        )
        conn.execute(
            "DELETE FROM learning_document_versions WHERE document_id = ?",
            (document_id,),
        )
        conn.execute(
            "DELETE FROM learning_document_cards WHERE document_id = ?",
            (document_id,),
        )
        cursor = conn.execute(
            "DELETE FROM learning_documents WHERE id = ?",
            (document_id,),
        )
    return cursor.rowcount > 0


def document_counts_by_card(course_id: str) -> dict[str, int]:
    ensure_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT learning_document_cards.card_id, COUNT(DISTINCT document_id) count
            FROM learning_document_cards
            INNER JOIN learning_documents
                ON learning_documents.id = learning_document_cards.document_id
            WHERE learning_documents.course_id = ?
            GROUP BY learning_document_cards.card_id
            """,
            (course_id,),
        ).fetchall()
    return {row["card_id"]: int(row["count"]) for row in rows}


def document_ids_by_card(course_id: str) -> dict[str, set[str]]:
    ensure_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT learning_document_cards.card_id,
                   learning_document_cards.document_id
            FROM learning_document_cards
            INNER JOIN learning_documents
                ON learning_documents.id = learning_document_cards.document_id
            WHERE learning_documents.course_id = ?
            """,
            (course_id,),
        ).fetchall()
    result: dict[str, set[str]] = {}
    for row in rows:
        result.setdefault(row["card_id"], set()).add(row["document_id"])
    return result


def move_learning_documents_to_course(
    source_course_id: str,
    target_course_id: str,
) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute(
            "UPDATE learning_documents SET course_id = ? WHERE course_id = ?",
            (target_course_id, source_course_id),
        )


def due_review_counts_by_card(course_id: str) -> dict[str, int]:
    ensure_db()
    now = datetime.now(timezone.utc).isoformat()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT review_items.card_id, COUNT(review_items.id) count
            FROM review_items
            INNER JOIN knowledge_cards ON knowledge_cards.id = review_items.card_id
            INNER JOIN jobs ON jobs.id = knowledge_cards.job_id
            LEFT JOIN review_progress
                ON review_progress.review_item_id = review_items.id
            WHERE jobs.course_id = ? AND review_items.status = 'active'
              AND (review_progress.due_at IS NULL OR review_progress.due_at <= ?)
            GROUP BY review_items.card_id
            """,
            (course_id, now),
        ).fetchall()
    return {row["card_id"]: int(row["count"]) for row in rows}


def clear_learning_documents() -> None:
    ensure_db()
    with connect() as conn:
        conn.execute("DELETE FROM learning_document_sources")
        conn.execute("DELETE FROM learning_document_versions")
        conn.execute("DELETE FROM learning_document_cards")
        conn.execute("DELETE FROM learning_documents")
