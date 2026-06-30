import json
from datetime import datetime
from sqlite3 import Row

from .db import connect, ensure_db
from .knowledge_card import (
    KnowledgeCard,
    KnowledgeCardClaim,
    KnowledgeCardIndexItem,
)


def _datetime_to_text(value: datetime) -> str:
    return value.isoformat()


def _datetime_from_text(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _key_points_to_json(card: KnowledgeCard) -> str:
    return json.dumps(card.key_points, ensure_ascii=False)


def _claims_to_json(card: KnowledgeCard) -> str:
    return json.dumps(
        [
            claim.model_dump(mode="json")
            for claim in card.claims
        ],
        ensure_ascii=False,
    )


def _unsupported_terms_to_json(card: KnowledgeCard) -> str:
    return json.dumps(card.unsupported_terms, ensure_ascii=False)


def _tags_to_json(card: KnowledgeCard) -> str:
    return json.dumps(card.tags, ensure_ascii=False)


def _key_points_from_json(value: str) -> list[str]:
    raw_points = json.loads(value)

    if not isinstance(raw_points, list):
        return []

    return [
        str(point)
        for point in raw_points
        if str(point).strip()
    ]


def _claims_from_json(value: str) -> list[KnowledgeCardClaim]:
    raw_claims = json.loads(value)

    if not isinstance(raw_claims, list):
        return []

    return [
        KnowledgeCardClaim.model_validate(claim)
        for claim in raw_claims
        if isinstance(claim, dict)
    ]


def _unsupported_terms_from_json(value: str) -> list[str]:
    raw_terms = json.loads(value)

    if not isinstance(raw_terms, list):
        return []

    return [
        str(term).strip()
        for term in raw_terms
        if str(term).strip()
    ]


def _tags_from_json(value: str) -> list[str]:
    raw_tags = json.loads(value)

    if not isinstance(raw_tags, list):
        return []

    return [
        str(tag).strip()
        for tag in raw_tags
        if str(tag).strip()
    ]


def _row_to_card(row: Row) -> KnowledgeCard:
    return KnowledgeCard(
        id=row["id"],
        job_id=row["job_id"],
        title=row["title"],
        summary=row["summary"],
        key_points=_key_points_from_json(row["key_points"]),
        claims=_claims_from_json(row["claims"]),
        unsupported_terms=_unsupported_terms_from_json(
            row["unsupported_terms"]
        ),
        question=row["question"],
        answer=row["answer"],
        difficulty=row["difficulty"],
        tags=_tags_from_json(row["tags"]),
        review_state=row["review_state"],
        source_start_seconds=row["source_start_seconds"],
        source_end_seconds=row["source_end_seconds"],
        provider=row["provider"],
        model=row["model"],
        created_at=_datetime_from_text(row["created_at"]),
        updated_at=_datetime_from_text(row["updated_at"]),
    )


def _row_to_card_index_item(row: Row) -> KnowledgeCardIndexItem:
    return KnowledgeCardIndexItem(
        id=row["id"],
        job_id=row["job_id"],
        title=row["title"],
        summary=row["summary"],
        difficulty=row["difficulty"],
        tags=_tags_from_json(row["tags"]),
        review_state=row["review_state"],
        source_video=row["source_video"],
        source_start_seconds=row["source_start_seconds"],
        source_end_seconds=row["source_end_seconds"],
        note_count=row["note_count"],
        created_at=_datetime_from_text(row["created_at"]),
        updated_at=_datetime_from_text(row["updated_at"]),
    )


def create_card(card: KnowledgeCard) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO knowledge_cards (
                id,
                job_id,
                title,
                summary,
                key_points,
                claims,
                unsupported_terms,
                question,
                answer,
                difficulty,
                tags,
                review_state,
                source_start_seconds,
                source_end_seconds,
                provider,
                model,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                card.id,
                card.job_id,
                card.title,
                card.summary,
                _key_points_to_json(card),
                _claims_to_json(card),
                _unsupported_terms_to_json(card),
                card.question,
                card.answer,
                card.difficulty,
                _tags_to_json(card),
                card.review_state,
                card.source_start_seconds,
                card.source_end_seconds,
                card.provider,
                card.model,
                _datetime_to_text(card.created_at),
                _datetime_to_text(card.updated_at),
            ),
        )


def get_card(card_id: str) -> KnowledgeCard | None:
    ensure_db()

    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM knowledge_cards WHERE id = ?",
            (card_id,),
        ).fetchone()

    if row is None:
        return None

    return _row_to_card(row)


def list_cards_for_job(job_id: str) -> list[KnowledgeCard]:
    ensure_db()

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM knowledge_cards
            WHERE job_id = ?
            ORDER BY source_start_seconds ASC, created_at ASC
            """,
            (job_id,),
        ).fetchall()

    return [_row_to_card(row) for row in rows]


def list_cards_for_course(course_id: str) -> list[KnowledgeCard]:
    ensure_db()

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT knowledge_cards.*
            FROM knowledge_cards
            INNER JOIN jobs ON jobs.id = knowledge_cards.job_id
            WHERE jobs.course_id = ?
            ORDER BY
                knowledge_cards.source_start_seconds ASC,
                knowledge_cards.created_at ASC
            """,
            (course_id,),
        ).fetchall()

    return [_row_to_card(row) for row in rows]


def list_cards() -> list[KnowledgeCard]:
    ensure_db()

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT knowledge_cards.*
            FROM knowledge_cards
            INNER JOIN jobs ON jobs.id = knowledge_cards.job_id
            ORDER BY
                jobs.course_id ASC,
                jobs.created_at ASC,
                knowledge_cards.source_start_seconds ASC,
                knowledge_cards.created_at ASC
            """
        ).fetchall()

    return [_row_to_card(row) for row in rows]


def list_card_index_for_course(
    course_id: str,
) -> list[KnowledgeCardIndexItem]:
    ensure_db()

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
                knowledge_cards.id,
                knowledge_cards.job_id,
                knowledge_cards.title,
                knowledge_cards.summary,
                knowledge_cards.difficulty,
                knowledge_cards.tags,
                knowledge_cards.review_state,
                COALESCE(
                    jobs.original_filename,
                    jobs.stored_name,
                    jobs.id
                ) AS source_video,
                knowledge_cards.source_start_seconds,
                knowledge_cards.source_end_seconds,
                knowledge_cards.created_at,
                knowledge_cards.updated_at,
                COUNT(knowledge_card_notes.id) AS note_count
            FROM knowledge_cards
            INNER JOIN jobs ON jobs.id = knowledge_cards.job_id
            LEFT JOIN knowledge_card_notes
                ON knowledge_card_notes.card_id = knowledge_cards.id
            WHERE jobs.course_id = ?
            GROUP BY knowledge_cards.id
            ORDER BY
                knowledge_cards.updated_at DESC,
                knowledge_cards.source_start_seconds ASC
            """,
            (course_id,),
        ).fetchall()

    return [_row_to_card_index_item(row) for row in rows]


def update_card(card: KnowledgeCard) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            """
            UPDATE knowledge_cards
            SET title = ?,
                summary = ?,
                key_points = ?,
                claims = ?,
                unsupported_terms = ?,
                question = ?,
                answer = ?,
                difficulty = ?,
                tags = ?,
                review_state = ?,
                source_start_seconds = ?,
                source_end_seconds = ?,
                provider = ?,
                model = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                card.title,
                card.summary,
                _key_points_to_json(card),
                _claims_to_json(card),
                _unsupported_terms_to_json(card),
                card.question,
                card.answer,
                card.difficulty,
                _tags_to_json(card),
                card.review_state,
                card.source_start_seconds,
                card.source_end_seconds,
                card.provider,
                card.model,
                _datetime_to_text(card.updated_at),
                card.id,
            ),
        )


def delete_card(card_id: str) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            "DELETE FROM knowledge_card_notes WHERE card_id = ?",
            (card_id,),
        )
        conn.execute(
            "DELETE FROM knowledge_cards WHERE id = ?",
            (card_id,),
        )


def delete_cards_for_job(job_id: str) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            """
            DELETE FROM knowledge_card_notes
            WHERE card_id IN (
                SELECT id FROM knowledge_cards WHERE job_id = ?
            )
            """,
            (job_id,),
        )
        conn.execute(
            "DELETE FROM knowledge_cards WHERE job_id = ?",
            (job_id,),
        )


def delete_cards_for_course(course_id: str) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            """
            DELETE FROM knowledge_card_notes
            WHERE card_id IN (
                SELECT knowledge_cards.id
                FROM knowledge_cards
                INNER JOIN jobs ON jobs.id = knowledge_cards.job_id
                WHERE jobs.course_id = ?
            )
            """,
            (course_id,),
        )
        conn.execute(
            """
            DELETE FROM knowledge_cards
            WHERE job_id IN (
                SELECT id FROM jobs WHERE course_id = ?
            )
            """,
            (course_id,),
        )


def clear_cards() -> None:
    ensure_db()

    with connect() as conn:
        conn.execute("DELETE FROM knowledge_card_notes")
        conn.execute("DELETE FROM knowledge_cards")
