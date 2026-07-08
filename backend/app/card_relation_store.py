from __future__ import annotations

from datetime import datetime
from sqlite3 import Row

from .card_relation import CardRelation
from .db import connect, ensure_db


def _datetime_to_text(value: datetime) -> str:
    return value.isoformat()


def _datetime_from_text(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _row_to_relation(row: Row) -> CardRelation:
    return CardRelation(
        id=row["id"],
        course_id=row["course_id"],
        source_card_id=row["source_card_id"],
        target_card_id=row["target_card_id"],
        relation_type=row["relation_type"],
        score=row["score"],
        method=row["method"],
        model=row["model"],
        explanation=row["explanation"],
        status=row["status"],
        created_at=_datetime_from_text(row["created_at"]),
        updated_at=_datetime_from_text(row["updated_at"]),
    )


def _relation_to_params(relation: CardRelation) -> tuple:
    return (
        relation.id,
        relation.course_id,
        relation.source_card_id,
        relation.target_card_id,
        relation.relation_type,
        relation.score,
        relation.method,
        relation.model,
        relation.explanation,
        relation.status,
        _datetime_to_text(relation.created_at),
        _datetime_to_text(relation.updated_at),
    )


def upsert_card_relations(relations: list[CardRelation]) -> None:
    ensure_db()

    if not relations:
        return

    with connect() as conn:
        conn.executemany(
            """
            INSERT INTO card_relations (
                id,
                course_id,
                source_card_id,
                target_card_id,
                relation_type,
                score,
                method,
                model,
                explanation,
                status,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(
                source_card_id,
                target_card_id,
                relation_type,
                method
            ) DO UPDATE SET
                course_id = excluded.course_id,
                score = excluded.score,
                model = excluded.model,
                explanation = excluded.explanation,
                updated_at = excluded.updated_at
            WHERE card_relations.status = 'suggested'
            """,
            [_relation_to_params(relation) for relation in relations],
        )


def replace_suggested_relations_for_course(
    course_id: str,
    relations: list[CardRelation],
    *,
    relation_type: str,
    method: str,
) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            """
            DELETE FROM card_relations
            WHERE course_id = ?
              AND relation_type = ?
              AND method = ?
              AND status = 'suggested'
            """,
            (course_id, relation_type, method),
        )

        if not relations:
            return

        conn.executemany(
            """
            INSERT INTO card_relations (
                id,
                course_id,
                source_card_id,
                target_card_id,
                relation_type,
                score,
                method,
                model,
                explanation,
                status,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(
                source_card_id,
                target_card_id,
                relation_type,
                method
            ) DO UPDATE SET
                course_id = excluded.course_id,
                score = excluded.score,
                model = excluded.model,
                explanation = excluded.explanation,
                updated_at = excluded.updated_at
            WHERE card_relations.status = 'suggested'
            """,
            [_relation_to_params(relation) for relation in relations],
        )


def get_card_relation(relation_id: str) -> CardRelation | None:
    ensure_db()

    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM card_relations WHERE id = ?",
            (relation_id,),
        ).fetchone()

    if row is None:
        return None

    return _row_to_relation(row)


def list_card_relations_for_course(course_id: str) -> list[CardRelation]:
    ensure_db()

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM card_relations
            WHERE course_id = ?
            ORDER BY score DESC, source_card_id ASC, target_card_id ASC
            """,
            (course_id,),
        ).fetchall()

    return [_row_to_relation(row) for row in rows]


def list_related_card_relations(card_id: str) -> list[CardRelation]:
    ensure_db()

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM card_relations
            WHERE source_card_id = ?
            ORDER BY score DESC, target_card_id ASC
            """,
            (card_id,),
        ).fetchall()

    return [_row_to_relation(row) for row in rows]


def update_card_relation(relation: CardRelation) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            """
            UPDATE card_relations
            SET relation_type = ?,
                score = ?,
                method = ?,
                model = ?,
                explanation = ?,
                status = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                relation.relation_type,
                relation.score,
                relation.method,
                relation.model,
                relation.explanation,
                relation.status,
                _datetime_to_text(relation.updated_at),
                relation.id,
            ),
        )


def delete_card_relation(relation_id: str) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            "DELETE FROM card_relations WHERE id = ?",
            (relation_id,),
        )


def delete_card_relations_for_card(card_id: str) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            """
            DELETE FROM card_relations
            WHERE source_card_id = ? OR target_card_id = ?
            """,
            (card_id, card_id),
        )


def delete_card_relations_for_job(job_id: str) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            """
            DELETE FROM card_relations
            WHERE source_card_id IN (
                SELECT id FROM knowledge_cards WHERE job_id = ?
            )
            OR target_card_id IN (
                SELECT id FROM knowledge_cards WHERE job_id = ?
            )
            """,
            (job_id, job_id),
        )


def delete_card_relations_for_course(course_id: str) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            """
            DELETE FROM card_relations
            WHERE course_id = ?
               OR source_card_id IN (
                    SELECT knowledge_cards.id
                    FROM knowledge_cards
                    INNER JOIN jobs ON jobs.id = knowledge_cards.job_id
                    WHERE jobs.course_id = ?
               )
               OR target_card_id IN (
                    SELECT knowledge_cards.id
                    FROM knowledge_cards
                    INNER JOIN jobs ON jobs.id = knowledge_cards.job_id
                    WHERE jobs.course_id = ?
               )
            """,
            (course_id, course_id, course_id),
        )


def clear_card_relations() -> None:
    ensure_db()

    with connect() as conn:
        conn.execute("DELETE FROM card_relations")
