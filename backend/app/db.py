import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .course import DEFAULT_COURSE_ID, DEFAULT_COURSE_TITLE

DEFAULT_DB_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "jobs.db"
)

_db_path = DEFAULT_DB_PATH
_initialized_paths: set[Path] = set()

JOB_COLUMNS: dict[str, str] = {
    "course_id": "TEXT",
    "original_filename": "TEXT",
    "stored_name": "TEXT",
    "size_bytes": "INTEGER",
    "created_at": "TEXT",
    "updated_at": "TEXT",
    "started_at": "TEXT",
    "completed_at": "TEXT",
}

KNOWLEDGE_CARD_COLUMNS = {
    "id",
    "job_id",
    "title",
    "summary",
    "key_points",
    "claims",
    "unsupported_terms",
    "question",
    "answer",
    "difficulty",
    "source_start_seconds",
    "source_end_seconds",
    "provider",
    "model",
    "created_at",
    "updated_at",
}


def configure_db(db_path: Path) -> None:
    global _db_path

    _db_path = db_path


def get_db_path() -> Path:
    return _db_path


def get_conn() -> sqlite3.Connection:
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = get_conn()

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    db_path = get_db_path()

    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS courses (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            INSERT INTO courses (
                id,
                title,
                description,
                created_at,
                updated_at
            ) VALUES (
                ?,
                ?,
                NULL,
                datetime('now'),
                datetime('now')
            )
            ON CONFLICT(id) DO NOTHING
            """,
            (
                DEFAULT_COURSE_ID,
                DEFAULT_COURSE_TITLE,
            ),
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                course_id TEXT,
                video_path TEXT NOT NULL,
                status TEXT NOT NULL,
                original_filename TEXT,
                stored_name TEXT,
                size_bytes INTEGER,
                metadata TEXT,
                transcript_path TEXT,
                error_message TEXT,
                created_at TEXT,
                updated_at TEXT,
                started_at TEXT,
                completed_at TEXT
            )
            """
        )

        existing_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(jobs)")
        }

        for column_name, column_type in JOB_COLUMNS.items():
            if column_name not in existing_columns:
                conn.execute(
                    f"ALTER TABLE jobs ADD COLUMN {column_name} {column_type}"
                )

        conn.execute(
            """
            UPDATE jobs
            SET course_id = ?
            WHERE course_id IS NULL OR trim(course_id) = ''
            """,
            (DEFAULT_COURSE_ID,),
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_jobs_course_id
            ON jobs (course_id)
            """
        )

        existing_card_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(knowledge_cards)")
        }

        if (
            existing_card_columns
            and not KNOWLEDGE_CARD_COLUMNS.issubset(existing_card_columns)
        ):
            conn.execute("DROP TABLE knowledge_cards")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_cards (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                key_points TEXT NOT NULL,
                claims TEXT NOT NULL,
                unsupported_terms TEXT NOT NULL,
                question TEXT,
                answer TEXT,
                difficulty TEXT NOT NULL,
                source_start_seconds REAL NOT NULL,
                source_end_seconds REAL NOT NULL,
                provider TEXT,
                model TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_knowledge_cards_job_id
            ON knowledge_cards (job_id)
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_card_notes (
                id TEXT PRIMARY KEY,
                card_id TEXT NOT NULL,
                note_type TEXT NOT NULL,
                title TEXT,
                body TEXT NOT NULL,
                source TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_knowledge_card_notes_card_id
            ON knowledge_card_notes (card_id)
            """
        )

    _initialized_paths.add(db_path)


def ensure_db() -> None:
    if get_db_path() not in _initialized_paths:
        init_db()
