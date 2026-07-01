from __future__ import annotations

import struct
from datetime import datetime

from pydantic import BaseModel, Field

from .embedding import EmbeddingVector
from .job import utc_now


class CardEmbedding(BaseModel):
    card_id: str
    model: str
    dimension: int = Field(ge=1)
    text_hash: str
    vector: EmbeddingVector = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class CardEmbeddingInfo(BaseModel):
    card_id: str
    model: str
    dimension: int
    text_hash: str
    created_at: datetime
    updated_at: datetime


class CardEmbeddingStatus(BaseModel):
    total_cards: int
    embedded_cards: int
    missing_cards: int
    stale_cards: int
    model: str
    dimension: int | None = None


class CardEmbeddingBatchResult(BaseModel):
    total_cards: int
    embedded_cards: int
    skipped_cards: int
    model: str
    dimension: int | None = None


class CardEmbeddingError(ValueError):
    pass


def vector_to_blob(vector: EmbeddingVector) -> bytes:
    if not vector:
        raise CardEmbeddingError("Embedding vector cannot be empty.")

    return struct.pack(
        f"<{len(vector)}f",
        *[
            float(value)
            for value in vector
        ],
    )


def vector_from_blob(
    value: bytes,
    *,
    dimension: int,
) -> EmbeddingVector:
    if dimension <= 0:
        raise CardEmbeddingError("Embedding dimension must be positive.")

    expected_size = dimension * 4

    if len(value) != expected_size:
        raise CardEmbeddingError(
            "Embedding blob size does not match the stored dimension."
        )

    return [
        float(item)
        for item in struct.unpack(f"<{dimension}f", value)
    ]


def embedding_to_info(embedding: CardEmbedding) -> CardEmbeddingInfo:
    return CardEmbeddingInfo(
        card_id=embedding.card_id,
        model=embedding.model,
        dimension=embedding.dimension,
        text_hash=embedding.text_hash,
        created_at=embedding.created_at,
        updated_at=embedding.updated_at,
    )
