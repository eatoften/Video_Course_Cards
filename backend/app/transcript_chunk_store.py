import json
from datetime import datetime
from sqlite3 import Row

from .db import connect, ensure_db
from .transcript_chunk import TranscriptChunk


def _datetime_to_text(value: datetime) -> str:
    return value.isoformat()


def _datetime_from_text(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _segment_ids_to_json(chunk: TranscriptChunk) -> str:
    return json.dumps(chunk.segment_ids, ensure_ascii=False)


def _segment_ids_from_json(value: str) -> list[int]:
    raw_segment_ids = json.loads(value)

    if not isinstance(raw_segment_ids, list):
        return []

    return [
        int(segment_id)
        for segment_id in raw_segment_ids
    ]


def _row_to_chunk(row: Row) -> TranscriptChunk:
    return TranscriptChunk(
        id=row["id"],
        course_id=row["course_id"],
        job_id=row["job_id"],
        chunk_index=row["chunk_index"],
        start_seconds=row["start_seconds"],
        end_seconds=row["end_seconds"],
        text=row["text"],
        segment_ids=_segment_ids_from_json(row["segment_ids"]),
        chunker_version=row["chunker_version"],
        created_at=_datetime_from_text(row["created_at"]),
    )


def create_chunks(chunks: list[TranscriptChunk]) -> None:
    ensure_db()

    if not chunks:
        return

    with connect() as conn:
        conn.executemany(
            """
            INSERT INTO transcript_chunks (
                id,
                course_id,
                job_id,
                chunk_index,
                start_seconds,
                end_seconds,
                text,
                segment_ids,
                chunker_version,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    chunk.id,
                    chunk.course_id,
                    chunk.job_id,
                    chunk.chunk_index,
                    chunk.start_seconds,
                    chunk.end_seconds,
                    chunk.text,
                    _segment_ids_to_json(chunk),
                    chunk.chunker_version,
                    _datetime_to_text(chunk.created_at),
                )
                for chunk in chunks
            ],
        )


def replace_chunks_for_job(
    job_id: str,
    chunks: list[TranscriptChunk],
) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            "DELETE FROM transcript_chunks WHERE job_id = ?",
            (job_id,),
        )

        if not chunks:
            return

        conn.executemany(
            """
            INSERT INTO transcript_chunks (
                id,
                course_id,
                job_id,
                chunk_index,
                start_seconds,
                end_seconds,
                text,
                segment_ids,
                chunker_version,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    chunk.id,
                    chunk.course_id,
                    chunk.job_id,
                    chunk.chunk_index,
                    chunk.start_seconds,
                    chunk.end_seconds,
                    chunk.text,
                    _segment_ids_to_json(chunk),
                    chunk.chunker_version,
                    _datetime_to_text(chunk.created_at),
                )
                for chunk in chunks
            ],
        )


def list_chunks_for_job(job_id: str) -> list[TranscriptChunk]:
    ensure_db()

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM transcript_chunks
            WHERE job_id = ?
            ORDER BY chunk_index ASC
            """,
            (job_id,),
        ).fetchall()

    return [
        _row_to_chunk(row)
        for row in rows
    ]


def list_chunks_for_course(course_id: str) -> list[TranscriptChunk]:
    ensure_db()

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM transcript_chunks
            WHERE course_id = ?
            ORDER BY job_id ASC, chunk_index ASC
            """,
            (course_id,),
        ).fetchall()

    return [
        _row_to_chunk(row)
        for row in rows
    ]


def delete_chunks_for_job(job_id: str) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            "DELETE FROM transcript_chunks WHERE job_id = ?",
            (job_id,),
        )


def clear_chunks() -> None:
    ensure_db()

    with connect() as conn:
        conn.execute("DELETE FROM transcript_chunks")
