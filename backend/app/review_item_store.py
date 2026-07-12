from __future__ import annotations

import json
from datetime import datetime
from sqlite3 import Row

from .db import connect, ensure_db
from .review_item import ReviewItem


def _datetime_to_text(value: datetime) -> str:
    return value.isoformat()


def _datetime_from_text(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _row_to_review_item(row: Row) -> ReviewItem:
    raw_claim_ids = json.loads(row["source_claim_ids"])
    return ReviewItem(
        id=row["id"],
        card_id=row["card_id"],
        item_type=row["item_type"],
        prompt=row["prompt"],
        expected_answer=row["expected_answer"],
        source_claim_ids=[str(value) for value in raw_claim_ids],
        source=row["source"],
        status=row["status"],
        created_at=_datetime_from_text(row["created_at"]),
        updated_at=_datetime_from_text(row["updated_at"]),
    )


def create_review_item(item: ReviewItem) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO review_items (
                id, card_id, item_type, prompt, expected_answer,
                source_claim_ids, source, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.id,
                item.card_id,
                item.item_type,
                item.prompt,
                item.expected_answer,
                json.dumps(item.source_claim_ids, ensure_ascii=False),
                item.source,
                item.status,
                _datetime_to_text(item.created_at),
                _datetime_to_text(item.updated_at),
            ),
        )


def get_review_item(item_id: str) -> ReviewItem | None:
    ensure_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM review_items WHERE id = ?",
            (item_id,),
        ).fetchone()
    return _row_to_review_item(row) if row is not None else None


def list_review_items_for_card(card_id: str) -> list[ReviewItem]:
    ensure_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM review_items
            WHERE card_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (card_id,),
        ).fetchall()
    return [_row_to_review_item(row) for row in rows]


def update_review_item(item: ReviewItem) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute(
            """
            UPDATE review_items
            SET item_type = ?, prompt = ?, expected_answer = ?,
                source_claim_ids = ?, status = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                item.item_type,
                item.prompt,
                item.expected_answer,
                json.dumps(item.source_claim_ids, ensure_ascii=False),
                item.status,
                _datetime_to_text(item.updated_at),
                item.id,
            ),
        )


def delete_review_item(item_id: str) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute("DELETE FROM review_events WHERE review_item_id = ?", (item_id,))
        conn.execute("DELETE FROM review_progress WHERE review_item_id = ?", (item_id,))
        conn.execute("DELETE FROM review_items WHERE id = ?", (item_id,))


def delete_review_items_for_card(card_id: str) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute(
            """
            DELETE FROM review_events WHERE review_item_id IN (
                SELECT id FROM review_items WHERE card_id = ?
            )
            """,
            (card_id,),
        )
        conn.execute(
            """
            DELETE FROM review_progress WHERE review_item_id IN (
                SELECT id FROM review_items WHERE card_id = ?
            )
            """,
            (card_id,),
        )
        conn.execute("DELETE FROM review_items WHERE card_id = ?", (card_id,))


def clear_review_items() -> None:
    ensure_db()
    with connect() as conn:
        conn.execute("DELETE FROM review_events")
        conn.execute("DELETE FROM review_progress")
        conn.execute("DELETE FROM review_items")
