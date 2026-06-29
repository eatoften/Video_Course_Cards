from datetime import datetime
from sqlite3 import Row

from .course import Course
from .db import connect, ensure_db


def _datetime_to_text(value: datetime) -> str:
    return value.isoformat()


def _datetime_from_text(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _row_to_course(row: Row) -> Course:
    keys = set(row.keys())

    return Course(
        id=row["id"],
        title=row["title"],
        description=row["description"],
        created_at=_datetime_from_text(row["created_at"]),
        updated_at=_datetime_from_text(row["updated_at"]),
        job_count=row["job_count"] if "job_count" in keys else 0,
        card_count=row["card_count"] if "card_count" in keys else 0,
    )


def create_course(course: Course) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO courses (
                id,
                title,
                description,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                course.id,
                course.title,
                course.description,
                _datetime_to_text(course.created_at),
                _datetime_to_text(course.updated_at),
            ),
        )


def get_course(course_id: str) -> Course | None:
    ensure_db()

    with connect() as conn:
        row = conn.execute(
            """
            SELECT
                c.*,
                COUNT(DISTINCT j.id) AS job_count,
                COUNT(k.id) AS card_count
            FROM courses c
            LEFT JOIN jobs j ON j.course_id = c.id
            LEFT JOIN knowledge_cards k ON k.job_id = j.id
            WHERE c.id = ?
            GROUP BY c.id
            """,
            (course_id,),
        ).fetchone()

    if row is None:
        return None

    return _row_to_course(row)


def list_courses() -> list[Course]:
    ensure_db()

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
                c.*,
                COUNT(DISTINCT j.id) AS job_count,
                COUNT(k.id) AS card_count
            FROM courses c
            LEFT JOIN jobs j ON j.course_id = c.id
            LEFT JOIN knowledge_cards k ON k.job_id = j.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC, c.title ASC
            """
        ).fetchall()

    return [_row_to_course(row) for row in rows]


def update_course(course: Course) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            """
            UPDATE courses
            SET title = ?,
                description = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                course.title,
                course.description,
                _datetime_to_text(course.updated_at),
                course.id,
            ),
        )


def delete_course(course_id: str) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            "DELETE FROM courses WHERE id = ?",
            (course_id,),
        )
