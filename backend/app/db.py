import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DEFAULT_DB_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "jobs.db"
)

_db_path = DEFAULT_DB_PATH
_initialized_paths: set[Path] = set()

JOB_COLUMNS: dict[str, str] = {
    "original_filename": "TEXT",
    "stored_name": "TEXT",
    "size_bytes": "INTEGER",
    "created_at": "TEXT",
    "updated_at": "TEXT",
    "started_at": "TEXT",
    "completed_at": "TEXT",
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
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
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

    _initialized_paths.add(db_path)


def ensure_db() -> None:
    if get_db_path() not in _initialized_paths:
        init_db()
