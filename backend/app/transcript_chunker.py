from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

from .embedding import (
    TextEmbedder,
    adjacent_cosine_distances,
    build_segment_context_texts,
)
from .job import utc_now
from .transcript_chunk import DEFAULT_CHUNKER_VERSION, TranscriptChunk
from .transcription import TranscriptSegment


@dataclass(frozen=True)
class SemanticChunkerConfig:
    context_radius: int = 1
    min_chunk_seconds: float = 120.0
    max_chunk_seconds: float = 360.0
    boundary_percentile: float = 90.0
    chunker_version: str = DEFAULT_CHUNKER_VERSION

    def __post_init__(self) -> None:
        if self.context_radius < 0:
            raise ValueError("context_radius must be greater than or equal to 0.")

        if self.min_chunk_seconds < 0:
            raise ValueError(
                "min_chunk_seconds must be greater than or equal to 0."
            )

        if self.max_chunk_seconds <= 0:
            raise ValueError("max_chunk_seconds must be greater than 0.")

        if self.max_chunk_seconds < self.min_chunk_seconds:
            raise ValueError(
                "max_chunk_seconds must be greater than or equal to "
                "min_chunk_seconds."
            )

        if not 0 <= self.boundary_percentile <= 100:
            raise ValueError("boundary_percentile must be between 0 and 100.")


def chunk_transcript_segments(
    segments: Sequence[TranscriptSegment],
    *,
    embedder: TextEmbedder,
    config: SemanticChunkerConfig,
    course_id: str,
    job_id: str,
) -> list[TranscriptChunk]:
    indexed_segments = [
        (segment_index, segment)
        for segment_index, segment in enumerate(segments)
        if segment.text.strip()
    ]

    if not indexed_segments:
        return []

    clean_segments = [
        segment
        for _, segment in indexed_segments
    ]
    context_texts = build_segment_context_texts(
        clean_segments,
        radius=config.context_radius,
    )
    embeddings = embedder.embed_texts(context_texts)

    if len(embeddings) != len(clean_segments):
        raise ValueError("Embedder returned a different number of vectors.")

    if len(clean_segments) == 1:
        return [
            _build_chunk(
                indexed_segments=indexed_segments,
                local_start_index=0,
                local_end_index=0,
                chunk_index=0,
                course_id=course_id,
                job_id=job_id,
                chunker_version=config.chunker_version,
            )
        ]

    distances = adjacent_cosine_distances(embeddings)
    boundary_threshold = _percentile(
        distances,
        config.boundary_percentile,
    )

    chunks: list[TranscriptChunk] = []
    chunk_start_index = 0

    for distance_index, distance in enumerate(distances):
        current_start = clean_segments[chunk_start_index]
        current_end = clean_segments[distance_index]
        current_duration = (
            current_end.end_seconds
            - current_start.start_seconds
        )

        if _should_cut_after_segment(
            distance=distance,
            distance_index=distance_index,
            distances=distances,
            current_duration=current_duration,
            boundary_threshold=boundary_threshold,
            config=config,
        ):
            chunks.append(
                _build_chunk(
                    indexed_segments=indexed_segments,
                    local_start_index=chunk_start_index,
                    local_end_index=distance_index,
                    chunk_index=len(chunks),
                    course_id=course_id,
                    job_id=job_id,
                    chunker_version=config.chunker_version,
                )
            )
            chunk_start_index = distance_index + 1

    if chunk_start_index < len(clean_segments):
        chunks.append(
            _build_chunk(
                indexed_segments=indexed_segments,
                local_start_index=chunk_start_index,
                local_end_index=len(clean_segments) - 1,
                chunk_index=len(chunks),
                course_id=course_id,
                job_id=job_id,
                chunker_version=config.chunker_version,
            )
        )

    return chunks


def _should_cut_after_segment(
    *,
    distance: float,
    distance_index: int,
    distances: Sequence[float],
    current_duration: float,
    boundary_threshold: float,
    config: SemanticChunkerConfig,
) -> bool:
    if current_duration >= config.max_chunk_seconds:
        return True

    if current_duration < config.min_chunk_seconds:
        return False

    return (
        distance >= boundary_threshold
        and _is_local_peak(distances, distance_index)
    )


def _build_chunk(
    *,
    indexed_segments: Sequence[tuple[int, TranscriptSegment]],
    local_start_index: int,
    local_end_index: int,
    chunk_index: int,
    course_id: str,
    job_id: str,
    chunker_version: str,
) -> TranscriptChunk:
    selected = indexed_segments[local_start_index:local_end_index + 1]
    segment_ids = [
        segment_index
        for segment_index, _ in selected
    ]
    chunk_segments = [
        segment
        for _, segment in selected
    ]
    text = " ".join(
        segment.text.strip()
        for segment in chunk_segments
        if segment.text.strip()
    )

    return TranscriptChunk(
        id=uuid4().hex,
        course_id=course_id,
        job_id=job_id,
        chunk_index=chunk_index,
        start_seconds=chunk_segments[0].start_seconds,
        end_seconds=chunk_segments[-1].end_seconds,
        text=text,
        segment_ids=segment_ids,
        chunker_version=chunker_version,
        created_at=utc_now(),
    )


def _is_local_peak(
    values: Sequence[float],
    index: int,
) -> bool:
    left = values[index - 1] if index > 0 else -math.inf
    right = values[index + 1] if index < len(values) - 1 else -math.inf

    return values[index] >= left and values[index] >= right


def _percentile(
    values: Sequence[float],
    percentile: float,
) -> float:
    if not values:
        return math.inf

    sorted_values = sorted(values)

    if percentile <= 0:
        return sorted_values[0]

    if percentile >= 100:
        return sorted_values[-1]

    rank = (percentile / 100) * (len(sorted_values) - 1)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)

    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    fraction = rank - lower_index

    return lower_value + (upper_value - lower_value) * fraction
