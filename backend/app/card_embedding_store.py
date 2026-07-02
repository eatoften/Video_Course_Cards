from __future__ import annotations

from datetime import datetime
from sqlite3 import Row

from .card_embedding import (
    CardEmbedding,
    CardEmbeddingInfo,
    vector_from_blob,
    vector_to_blob,
)
from .db import connect, ensure_db


def _datetime_to_text(value: datetime) -> str:
    return value.isoformat()


def _datetime_from_text(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _row_to_embedding(row: Row) -> CardEmbedding:
    return CardEmbedding(
        card_id=row["card_id"],
        model=row["model"],
        dimension=row["dimension"],
        text_hash=row["text_hash"],
        vector=vector_from_blob(
            row["vector"],
            dimension=row["dimension"],
        ),
        created_at=_datetime_from_text(row["created_at"]),
        updated_at=_datetime_from_text(row["updated_at"]),
    )


def _row_to_info(row: Row) -> CardEmbeddingInfo:
    return CardEmbeddingInfo(
        card_id=row["card_id"],
        model=row["model"],
        dimension=row["dimension"],
        text_hash=row["text_hash"],
        created_at=_datetime_from_text(row["created_at"]),
        updated_at=_datetime_from_text(row["updated_at"]),
    )


def upsert_card_embedding(embedding: CardEmbedding) -> None:
    upsert_card_embeddings([embedding])


def upsert_card_embeddings(embeddings: list[CardEmbedding]) -> None:
    ensure_db()

    if not embeddings:
        return

    with connect() as conn:
        conn.executemany(
            """
            INSERT INTO card_embeddings (
                card_id,
                model,
                dimension,
                text_hash,
                vector,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(card_id) DO UPDATE SET
                model = excluded.model,
                dimension = excluded.dimension,
                text_hash = excluded.text_hash,
                vector = excluded.vector,
                updated_at = excluded.updated_at
            """,
            [
                (
                    embedding.card_id,
                    embedding.model,
                    embedding.dimension,
                    embedding.text_hash,
                    vector_to_blob(embedding.vector),
                    _datetime_to_text(embedding.created_at),
                    _datetime_to_text(embedding.updated_at),
                )
                for embedding in embeddings
            ],
        )


def get_card_embedding(card_id: str) -> CardEmbedding | None:
    ensure_db()

    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM card_embeddings WHERE card_id = ?",
            (card_id,),
        ).fetchone()

    if row is None:
        return None

    return _row_to_embedding(row)


def get_card_embedding_info(card_id: str) -> CardEmbeddingInfo | None:
    ensure_db()

    with connect() as conn:
        row = conn.execute(
            """
            SELECT card_id, model, dimension, text_hash, created_at, updated_at
            FROM card_embeddings
            WHERE card_id = ?
            """,
            (card_id,),
        ).fetchone()

    if row is None:
        return None

    return _row_to_info(row)


def list_card_embeddings_for_job(
    job_id: str,
) -> list[CardEmbedding]:
    ensure_db()

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT card_embeddings.*
            FROM card_embeddings
            INNER JOIN knowledge_cards
                ON knowledge_cards.id = card_embeddings.card_id
            WHERE knowledge_cards.job_id = ?
            ORDER BY knowledge_cards.source_start_seconds ASC,
                     knowledge_cards.created_at ASC
            """,
            (job_id,),
        ).fetchall()

    return [
        _row_to_embedding(row)
        for row in rows
    ]


def list_card_embedding_infos_for_job(
    job_id: str,
) -> list[CardEmbeddingInfo]:
    ensure_db()

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
                card_embeddings.card_id,
                card_embeddings.model,
                card_embeddings.dimension,
                card_embeddings.text_hash,
                card_embeddings.created_at,
                card_embeddings.updated_at
            FROM card_embeddings
            INNER JOIN knowledge_cards
                ON knowledge_cards.id = card_embeddings.card_id
            WHERE knowledge_cards.job_id = ?
            ORDER BY knowledge_cards.source_start_seconds ASC,
                     knowledge_cards.created_at ASC
            """,
            (job_id,),
        ).fetchall()

    return [
        _row_to_info(row)
        for row in rows
    ]


def list_card_embeddings_for_course(
    course_id: str,
) -> list[CardEmbedding]:
    ensure_db()

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT card_embeddings.*
            FROM card_embeddings
            INNER JOIN knowledge_cards
                ON knowledge_cards.id = card_embeddings.card_id
            INNER JOIN jobs ON jobs.id = knowledge_cards.job_id
            WHERE jobs.course_id = ?
            ORDER BY jobs.created_at ASC,
                     knowledge_cards.source_start_seconds ASC,
                     knowledge_cards.created_at ASC
            """,
            (course_id,),
        ).fetchall()

    return [
        _row_to_embedding(row)
        for row in rows
    ]


def list_card_embedding_infos_for_course(
    course_id: str,
) -> list[CardEmbeddingInfo]:
    ensure_db()

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
                card_embeddings.card_id,
                card_embeddings.model,
                card_embeddings.dimension,
                card_embeddings.text_hash,
                card_embeddings.created_at,
                card_embeddings.updated_at
            FROM card_embeddings
            INNER JOIN knowledge_cards
                ON knowledge_cards.id = card_embeddings.card_id
            INNER JOIN jobs ON jobs.id = knowledge_cards.job_id
            WHERE jobs.course_id = ?
            ORDER BY jobs.created_at ASC,
                     knowledge_cards.source_start_seconds ASC,
                     knowledge_cards.created_at ASC
            """,
            (course_id,),
        ).fetchall()

    return [
        _row_to_info(row)
        for row in rows
    ]


def delete_card_embedding(card_id: str) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            "DELETE FROM card_embeddings WHERE card_id = ?",
            (card_id,),
        )


def delete_card_embeddings_for_job(job_id: str) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            """
            DELETE FROM card_embeddings
            WHERE card_id IN (
                SELECT id FROM knowledge_cards WHERE job_id = ?
            )
            """,
            (job_id,),
        )


def delete_card_embeddings_for_course(course_id: str) -> None:
    ensure_db()

    with connect() as conn:
        conn.execute(
            """
            DELETE FROM card_embeddings
            WHERE card_id IN (
                SELECT knowledge_cards.id
                FROM knowledge_cards
                INNER JOIN jobs ON jobs.id = knowledge_cards.job_id
                WHERE jobs.course_id = ?
            )
            """,
            (course_id,),
        )


def clear_card_embeddings() -> None:
    ensure_db()

    with connect() as conn:
        conn.execute("DELETE FROM card_embeddings")
