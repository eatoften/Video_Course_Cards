from __future__ import annotations

from datetime import datetime
from sqlite3 import Row

from .db import connect, ensure_db
from .review import ReviewEvent, ReviewProgress


def _to_text(value: datetime) -> str:
    return value.isoformat()


def _from_text(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _row_to_progress(row: Row) -> ReviewProgress:
    return ReviewProgress(
        review_item_id=row["review_item_id"],
        fsrs_card_id=row["fsrs_card_id"],
        fsrs_state=row["fsrs_state"],
        step=row["step"],
        due_at=_from_text(row["due_at"]),
        stability=row["stability"],
        fsrs_difficulty=row["fsrs_difficulty"],
        last_reviewed_at=(
            _from_text(row["last_reviewed_at"])
            if row["last_reviewed_at"]
            else None
        ),
        review_count=row["review_count"],
        lapse_count=row["lapse_count"],
        created_at=_from_text(row["created_at"]),
        updated_at=_from_text(row["updated_at"]),
    )


def get_review_progress(review_item_id: str) -> ReviewProgress | None:
    ensure_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM review_progress WHERE review_item_id = ?",
            (review_item_id,),
        ).fetchone()
    return _row_to_progress(row) if row is not None else None


def upsert_review_progress(progress: ReviewProgress) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO review_progress (
                review_item_id, fsrs_card_id, fsrs_state, step, due_at,
                stability, fsrs_difficulty, last_reviewed_at, review_count,
                lapse_count, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(review_item_id) DO UPDATE SET
                fsrs_state = excluded.fsrs_state,
                step = excluded.step,
                due_at = excluded.due_at,
                stability = excluded.stability,
                fsrs_difficulty = excluded.fsrs_difficulty,
                last_reviewed_at = excluded.last_reviewed_at,
                review_count = excluded.review_count,
                lapse_count = excluded.lapse_count,
                updated_at = excluded.updated_at
            """,
            (
                progress.review_item_id,
                progress.fsrs_card_id,
                progress.fsrs_state,
                progress.step,
                _to_text(progress.due_at),
                progress.stability,
                progress.fsrs_difficulty,
                _to_text(progress.last_reviewed_at)
                if progress.last_reviewed_at
                else None,
                progress.review_count,
                progress.lapse_count,
                _to_text(progress.created_at),
                _to_text(progress.updated_at),
            ),
        )


def create_review_event(event: ReviewEvent) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO review_events (
                id, review_item_id, rating, reviewed_at, response_time_ms,
                previous_phase, next_phase, due_before, due_after,
                scheduled_days
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.review_item_id,
                event.rating,
                _to_text(event.reviewed_at),
                event.response_time_ms,
                event.previous_phase,
                event.next_phase,
                _to_text(event.due_before),
                _to_text(event.due_after),
                event.scheduled_days,
            ),
        )


def list_review_events(review_item_id: str) -> list[ReviewEvent]:
    ensure_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM review_events
            WHERE review_item_id = ?
            ORDER BY reviewed_at DESC
            """,
            (review_item_id,),
        ).fetchall()
    return [
        ReviewEvent(
            id=row["id"],
            review_item_id=row["review_item_id"],
            rating=row["rating"],
            reviewed_at=_from_text(row["reviewed_at"]),
            response_time_ms=row["response_time_ms"],
            previous_phase=row["previous_phase"],
            next_phase=row["next_phase"],
            due_before=_from_text(row["due_before"]),
            due_after=_from_text(row["due_after"]),
            scheduled_days=row["scheduled_days"],
        )
        for row in rows
    ]


def clear_review_progress() -> None:
    ensure_db()
    with connect() as conn:
        conn.execute("DELETE FROM review_events")
        conn.execute("DELETE FROM review_progress")
