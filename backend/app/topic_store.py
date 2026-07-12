from __future__ import annotations

from datetime import datetime
from sqlite3 import Row

from .db import connect, ensure_db
from .topic import Topic, TopicCardMembership, TopicRelation


def _to_text(value: datetime) -> str:
    return value.isoformat()


def _from_text(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _row_to_topic(row: Row) -> Topic:
    return Topic(
        id=row["id"],
        course_id=row["course_id"],
        parent_topic_id=row["parent_topic_id"],
        title=row["title"],
        summary=row["summary"],
        position=row["position"],
        depth=row["depth"],
        method=row["method"],
        status=row["status"],
        is_system=bool(row["is_system"]),
        created_at=_from_text(row["created_at"]),
        updated_at=_from_text(row["updated_at"]),
    )


def _row_to_membership(row: Row) -> TopicCardMembership:
    return TopicCardMembership(
        id=row["id"],
        topic_id=row["topic_id"],
        card_id=row["card_id"],
        role=row["role"],
        position=row["position"],
        method=row["method"],
        confidence=row["confidence"],
        status=row["status"],
        created_at=_from_text(row["created_at"]),
        updated_at=_from_text(row["updated_at"]),
    )


def _row_to_relation(row: Row) -> TopicRelation:
    return TopicRelation(
        id=row["id"],
        course_id=row["course_id"],
        source_topic_id=row["source_topic_id"],
        target_topic_id=row["target_topic_id"],
        relation_type=row["relation_type"],
        explanation=row["explanation"],
        method=row["method"],
        status=row["status"],
        created_at=_from_text(row["created_at"]),
        updated_at=_from_text(row["updated_at"]),
    )


def create_topic(topic: Topic) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO topics (
                id, course_id, parent_topic_id, title, summary, position,
                depth, method, status, is_system, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic.id, topic.course_id, topic.parent_topic_id, topic.title,
                topic.summary, topic.position, topic.depth, topic.method,
                topic.status, int(topic.is_system), _to_text(topic.created_at),
                _to_text(topic.updated_at),
            ),
        )


def get_topic(topic_id: str) -> Topic | None:
    ensure_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM topics WHERE id = ?",
            (topic_id,),
        ).fetchone()
    return _row_to_topic(row) if row is not None else None


def list_topics_for_course(course_id: str) -> list[Topic]:
    ensure_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM topics WHERE course_id = ?
            ORDER BY depth ASC, position ASC, title ASC
            """,
            (course_id,),
        ).fetchall()
    return [_row_to_topic(row) for row in rows]


def update_topic(topic: Topic) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute(
            """
            UPDATE topics
            SET parent_topic_id = ?, title = ?, summary = ?, position = ?,
                depth = ?, method = ?, status = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                topic.parent_topic_id, topic.title, topic.summary,
                topic.position, topic.depth, topic.method, topic.status,
                _to_text(topic.updated_at), topic.id,
            ),
        )


def delete_topic_and_rehome(
    topic_id: str,
    *,
    fallback_topic_id: str,
    parent_topic_id: str | None,
) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute(
            "UPDATE topics SET parent_topic_id = ? WHERE parent_topic_id = ?",
            (parent_topic_id, topic_id),
        )
        conn.execute(
            "UPDATE topic_card_memberships SET topic_id = ? WHERE topic_id = ?",
            (fallback_topic_id, topic_id),
        )
        conn.execute(
            """
            DELETE FROM topic_relations
            WHERE source_topic_id = ? OR target_topic_id = ?
            """,
            (topic_id, topic_id),
        )
        conn.execute("DELETE FROM topics WHERE id = ?", (topic_id,))


def next_topic_position(course_id: str, parent_topic_id: str | None) -> int:
    ensure_db()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT COALESCE(MAX(position), -1) + 1 AS next_position
            FROM topics
            WHERE course_id = ? AND parent_topic_id IS ?
            """,
            (course_id, parent_topic_id),
        ).fetchone()
    return int(row["next_position"])


def upsert_primary_membership(membership: TopicCardMembership) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute(
            """
            DELETE FROM topic_card_memberships
            WHERE card_id = ? AND role = 'primary' AND status = 'accepted'
            """,
            (membership.card_id,),
        )
        conn.execute(
            """
            INSERT INTO topic_card_memberships (
                id, topic_id, card_id, role, position, method, confidence,
                status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(topic_id, card_id) DO UPDATE SET
                role = excluded.role,
                position = excluded.position,
                method = excluded.method,
                confidence = excluded.confidence,
                status = excluded.status,
                updated_at = excluded.updated_at
            """,
            (
                membership.id, membership.topic_id, membership.card_id,
                membership.role, membership.position, membership.method,
                membership.confidence, membership.status,
                _to_text(membership.created_at), _to_text(membership.updated_at),
            ),
        )


def upsert_topic_membership(membership: TopicCardMembership) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO topic_card_memberships (
                id, topic_id, card_id, role, position, method, confidence,
                status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(topic_id, card_id) DO UPDATE SET
                role = excluded.role,
                position = excluded.position,
                method = excluded.method,
                confidence = excluded.confidence,
                status = excluded.status,
                updated_at = excluded.updated_at
            """,
            (
                membership.id, membership.topic_id, membership.card_id,
                membership.role, membership.position, membership.method,
                membership.confidence, membership.status,
                _to_text(membership.created_at), _to_text(membership.updated_at),
            ),
        )


def list_memberships_for_topic(topic_id: str) -> list[TopicCardMembership]:
    ensure_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM topic_card_memberships
            WHERE topic_id = ? ORDER BY position ASC
            """,
            (topic_id,),
        ).fetchall()
    return [_row_to_membership(row) for row in rows]


def clear_suggested_topics_for_course(course_id: str) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute(
            """
            DELETE FROM topic_card_memberships
            WHERE topic_id IN (
                SELECT id FROM topics
                WHERE course_id = ? AND status = 'suggested'
            )
            """,
            (course_id,),
        )
        conn.execute(
            "DELETE FROM topics WHERE course_id = ? AND status = 'suggested'",
            (course_id,),
        )


def delete_topics_for_course(course_id: str) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute(
            "DELETE FROM topic_relations WHERE course_id = ?",
            (course_id,),
        )
        conn.execute(
            """
            DELETE FROM topic_card_memberships
            WHERE topic_id IN (SELECT id FROM topics WHERE course_id = ?)
            """,
            (course_id,),
        )
        conn.execute(
            "DELETE FROM topics WHERE course_id = ?",
            (course_id,),
        )


def delete_suggested_topic(topic_id: str) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute(
            "DELETE FROM topic_card_memberships WHERE topic_id = ?",
            (topic_id,),
        )
        conn.execute("DELETE FROM topics WHERE id = ?", (topic_id,))


def list_memberships_for_course(course_id: str) -> list[TopicCardMembership]:
    ensure_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT topic_card_memberships.*
            FROM topic_card_memberships
            INNER JOIN topics ON topics.id = topic_card_memberships.topic_id
            WHERE topics.course_id = ?
            ORDER BY topic_card_memberships.position ASC
            """,
            (course_id,),
        ).fetchall()
    return [_row_to_membership(row) for row in rows]


def create_topic_relation(relation: TopicRelation) -> None:
    ensure_db()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO topic_relations (
                id, course_id, source_topic_id, target_topic_id,
                relation_type, explanation, method, status,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                relation.id, relation.course_id, relation.source_topic_id,
                relation.target_topic_id, relation.relation_type,
                relation.explanation, relation.method, relation.status,
                _to_text(relation.created_at), _to_text(relation.updated_at),
            ),
        )


def list_topic_relations_for_course(course_id: str) -> list[TopicRelation]:
    ensure_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM topic_relations WHERE course_id = ?
            ORDER BY relation_type ASC, created_at ASC
            """,
            (course_id,),
        ).fetchall()
    return [_row_to_relation(row) for row in rows]


def delete_topic_relation(relation_id: str) -> bool:
    ensure_db()
    with connect() as conn:
        cursor = conn.execute(
            "DELETE FROM topic_relations WHERE id = ?",
            (relation_id,),
        )
    return cursor.rowcount > 0


def clear_topics() -> None:
    ensure_db()
    with connect() as conn:
        conn.execute("DELETE FROM topic_relations")
        conn.execute("DELETE FROM topic_card_memberships")
        conn.execute("DELETE FROM topics")
