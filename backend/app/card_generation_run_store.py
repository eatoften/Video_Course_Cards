import json
from datetime import datetime
from sqlite3 import Row

from .card_generation_run import (
    AutoCardGenerationRequest,
    CardGenerationRun,
    CardGenerationRunError,
)
from .db import connect, ensure_db


def _datetime_to_text(value: datetime | None) -> str | None:
    if value is None:
        return None

    return value.isoformat()


def _datetime_from_text(value: str | None) -> datetime | None:
    if value is None:
        return None

    return datetime.fromisoformat(value)


def _errors_to_json(run: CardGenerationRun) -> str:
    return json.dumps(
        [
            error.model_dump(mode="json")
            for error in run.errors
        ],
        ensure_ascii=False,
    )


def _errors_from_json(value: str) -> list[CardGenerationRunError]:
    raw_errors = json.loads(value)

    if not isinstance(raw_errors, list):
        return []

    return [
        CardGenerationRunError.model_validate(error)
        for error in raw_errors
        if isinstance(error, dict)
    ]


def _request_to_json(run: CardGenerationRun) -> str:
    return run.request.model_dump_json()


def _request_from_json(value: str) -> AutoCardGenerationRequest:
    return AutoCardGenerationRequest.model_validate_json(value)


def _row_to_run(row: Row) -> CardGenerationRun:
    return CardGenerationRun(
        id=row["id"],
        job_id=row["job_id"],
        mode=row["mode"],
        status=row["status"],
        model=row["model"],
        card_count_per_chunk=row["card_count_per_chunk"],
        total_chunks=row["total_chunks"],
        completed_chunks=row["completed_chunks"],
        succeeded_chunks=row["succeeded_chunks"],
        failed_chunks=row["failed_chunks"],
        cards_created=row["cards_created"],
        error_message=row["error_message"],
        errors=_errors_from_json(row["errors_json"]),
        request=_request_from_json(row["request_json"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        started_at=_datetime_from_text(row["started_at"]),
        completed_at=_datetime_from_text(row["completed_at"]),
    )


def create_run(run: CardGenerationRun) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO card_generation_runs (
                id,
                job_id,
                mode,
                status,
                model,
                card_count_per_chunk,
                total_chunks,
                completed_chunks,
                succeeded_chunks,
                failed_chunks,
                cards_created,
                error_message,
                errors_json,
                request_json,
                created_at,
                updated_at,
                started_at,
                completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.id,
                run.job_id,
                run.mode,
                run.status,
                run.model,
                run.card_count_per_chunk,
                run.total_chunks,
                run.completed_chunks,
                run.succeeded_chunks,
                run.failed_chunks,
                run.cards_created,
                run.error_message,
                _errors_to_json(run),
                _request_to_json(run),
                _datetime_to_text(run.created_at),
                _datetime_to_text(run.updated_at),
                _datetime_to_text(run.started_at),
                _datetime_to_text(run.completed_at),
            ),
        )


def get_run(run_id: str) -> CardGenerationRun | None:
    ensure_db()

    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM card_generation_runs WHERE id = ?",
            (run_id,),
        ).fetchone()

    if row is None:
        return None

    return _row_to_run(row)


def list_runs_for_job(job_id: str) -> list[CardGenerationRun]:
    ensure_db()

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM card_generation_runs
            WHERE job_id = ?
            ORDER BY created_at DESC
            """,
            (job_id,),
        ).fetchall()

    return [
        _row_to_run(row)
        for row in rows
    ]


def update_run(run: CardGenerationRun) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            """
            UPDATE card_generation_runs
            SET status = ?,
                model = ?,
                card_count_per_chunk = ?,
                total_chunks = ?,
                completed_chunks = ?,
                succeeded_chunks = ?,
                failed_chunks = ?,
                cards_created = ?,
                error_message = ?,
                errors_json = ?,
                request_json = ?,
                updated_at = ?,
                started_at = ?,
                completed_at = ?
            WHERE id = ?
            """,
            (
                run.status,
                run.model,
                run.card_count_per_chunk,
                run.total_chunks,
                run.completed_chunks,
                run.succeeded_chunks,
                run.failed_chunks,
                run.cards_created,
                run.error_message,
                _errors_to_json(run),
                _request_to_json(run),
                _datetime_to_text(run.updated_at),
                _datetime_to_text(run.started_at),
                _datetime_to_text(run.completed_at),
                run.id,
            ),
        )


def delete_runs_for_job(job_id: str) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            "DELETE FROM card_generation_runs WHERE job_id = ?",
            (job_id,),
        )


def clear_runs() -> None:
    ensure_db()

    with connect() as conn:
        conn.execute("DELETE FROM card_generation_runs")
