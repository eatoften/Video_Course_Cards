from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import Protocol

from .transcription import TranscriptSegment


DEFAULT_EMBEDDING_MODEL = os.getenv(
    "VCC_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
DEFAULT_EMBEDDING_BATCH_SIZE = 32

EmbeddingVector = list[float]


class EmbeddingError(RuntimeError):
    pass


class TextEmbedder(Protocol):
    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> list[EmbeddingVector]:
        pass


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._model: object | None = None

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> list[EmbeddingVector]:
        cleaned_texts = [text.strip() for text in texts]

        if not cleaned_texts:
            return []

        if any(not text for text in cleaned_texts):
            raise EmbeddingError("Embedding text cannot be empty.")

        model = self._get_model()
        encoded = model.encode(
            cleaned_texts,
            batch_size=batch_size or self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

        return [
            [float(value) for value in vector]
            for vector in encoded
        ]

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed. "
                    "Run `uv add sentence-transformers` in the backend "
                    "directory."
                ) from exc

            self._model = SentenceTransformer(self.model_name)

        return self._model


def build_segment_context_texts(
    segments: Sequence[TranscriptSegment],
    *,
    radius: int = 1,
) -> list[str]:
    if radius < 0:
        raise ValueError("radius must be greater than or equal to 0.")

    context_texts: list[str] = []

    for index in range(len(segments)):
        start_index = max(0, index - radius)
        end_index = min(len(segments), index + radius + 1)
        text = " ".join(
            segment.text.strip()
            for segment in segments[start_index:end_index]
            if segment.text.strip()
        )
        context_texts.append(text)

    return context_texts


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimension.")

    if not left:
        raise ValueError("Vectors cannot be empty.")

    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right)
    )
    left_norm = math.sqrt(
        sum(value * value for value in left)
    )
    right_norm = math.sqrt(
        sum(value * value for value in right)
    )

    if left_norm == 0 or right_norm == 0:
        return 0.0

    similarity = dot_product / (left_norm * right_norm)

    return max(-1.0, min(1.0, similarity))


def cosine_distance(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    return 1.0 - cosine_similarity(left, right)


def adjacent_cosine_distances(
    embeddings: Sequence[Sequence[float]],
) -> list[float]:
    return [
        cosine_distance(embeddings[index], embeddings[index + 1])
        for index in range(len(embeddings) - 1)
    ]
