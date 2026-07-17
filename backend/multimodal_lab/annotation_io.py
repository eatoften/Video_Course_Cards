from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .metrics import POSITIVE_TRANSITION_TYPES
from .schemas import (
    AnnotationBundleSummary,
    LectureDatasetManifest,
    SlideTransitionAnnotation,
    StablePageReference,
)


ModelType = TypeVar("ModelType", bound=BaseModel)


class AnnotationFileError(ValueError):
    pass


def load_jsonl(path: str | Path, model: type[ModelType]) -> list[ModelType]:
    resolved_path = Path(path)
    if not resolved_path.is_file():
        raise AnnotationFileError(f"Annotation file does not exist: {resolved_path}")

    records: list[ModelType] = []

    with resolved_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                payload = json.loads(line)
                records.append(model.model_validate(payload))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise AnnotationFileError(
                    f"Invalid {model.__name__} at {resolved_path}:{line_number}: "
                    f"{exc}"
                ) from exc

    return records


def write_jsonl(path: str | Path, records: Iterable[BaseModel]) -> None:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
        for record in records
    ]
    content = "\n".join(serialized)
    if content:
        content += "\n"
    resolved_path.write_text(content, encoding="utf-8")


def validate_annotation_bundle(
    manifests: Sequence[LectureDatasetManifest],
    transitions: Sequence[SlideTransitionAnnotation],
    references: Sequence[StablePageReference],
) -> AnnotationBundleSummary:
    manifests_by_lecture = _index_unique(
        manifests,
        key=lambda item: item.lecture_id,
        label="lecture_id in dataset manifests",
    )
    _index_unique(
        manifests,
        key=lambda item: item.job_id,
        label="job_id in dataset manifests",
    )
    transitions_by_id = _index_unique(
        transitions,
        key=lambda item: item.event_id,
        label="event_id in transition annotations",
    )
    _index_unique(
        references,
        key=lambda item: item.page_event_id,
        label="page_event_id in stable page references",
    )

    for transition in transitions:
        manifest = _require_manifest(
            manifests_by_lecture,
            transition.lecture_id,
        )
        if transition.change_start_seconds < manifest.pilot_start_seconds:
            raise AnnotationFileError(
                f"Transition {transition.event_id} starts before the pilot interval."
            )
        if transition.stable_at_seconds > manifest.pilot_end_seconds:
            raise AnnotationFileError(
                f"Transition {transition.event_id} exceeds the pilot interval."
            )
        if transition.stable_at_seconds > manifest.duration_seconds:
            raise AnnotationFileError(
                f"Transition {transition.event_id} exceeds lecture duration."
            )

    transitions_by_lecture = _group_transitions_by_lecture(transitions)

    for reference in references:
        manifest = _require_manifest(
            manifests_by_lecture,
            reference.lecture_id,
        )
        event = transitions_by_id.get(reference.page_event_id)
        if event is None:
            raise AnnotationFileError(
                "Stable page reference points to an unknown event: "
                f"{reference.page_event_id}"
            )
        if event.event_type not in POSITIVE_TRANSITION_TYPES:
            raise AnnotationFileError(
                "Stable page reference must point to a meaningful transition: "
                f"{reference.page_event_id}"
            )
        if event.lecture_id != reference.lecture_id:
            raise AnnotationFileError(
                f"Reference {reference.page_event_id} belongs to the wrong lecture."
            )
        if reference.page_number != event.to_page:
            raise AnnotationFileError(
                f"Reference {reference.page_event_id} page_number does not match "
                "the event to_page."
            )
        if reference.stable_frame_timestamp < event.stable_at_seconds:
            raise AnnotationFileError(
                f"Reference {reference.page_event_id} occurs before stable_at."
            )
        if reference.stable_frame_timestamp > manifest.pilot_end_seconds:
            raise AnnotationFileError(
                f"Reference {reference.page_event_id} exceeds the pilot interval."
            )
        if reference.stable_frame_timestamp > manifest.duration_seconds:
            raise AnnotationFileError(
                f"Reference {reference.page_event_id} exceeds lecture duration."
            )

        next_change_start = _next_change_start(
            transitions_by_lecture.get(reference.lecture_id, []),
            event,
        )
        if (
            next_change_start is not None
            and reference.stable_frame_timestamp >= next_change_start
        ):
            raise AnnotationFileError(
                f"Reference {reference.page_event_id} occurs at/after the next event."
            )

    return AnnotationBundleSummary(
        lecture_count=len(manifests),
        transition_event_count=len(transitions),
        positive_transition_count=sum(
            transition.event_type in POSITIVE_TRANSITION_TYPES
            for transition in transitions
        ),
        stable_page_reference_count=len(references),
    )


def _index_unique(items, *, key, label: str):
    indexed = {}
    for item in items:
        value = key(item)
        if value in indexed:
            raise AnnotationFileError(f"Duplicate {label}: {value}")
        indexed[value] = item
    return indexed


def _require_manifest(
    manifests_by_lecture: dict[str, LectureDatasetManifest],
    lecture_id: str,
) -> LectureDatasetManifest:
    manifest = manifests_by_lecture.get(lecture_id)
    if manifest is None:
        raise AnnotationFileError(
            f"Annotation references unknown lecture_id: {lecture_id}"
        )
    return manifest


def _group_transitions_by_lecture(
    transitions: Sequence[SlideTransitionAnnotation],
) -> dict[str, list[SlideTransitionAnnotation]]:
    grouped: dict[str, list[SlideTransitionAnnotation]] = {}
    for transition in transitions:
        grouped.setdefault(transition.lecture_id, []).append(transition)
    for lecture_transitions in grouped.values():
        lecture_transitions.sort(key=lambda item: item.change_start_seconds)
    return grouped


def _next_change_start(
    transitions: Sequence[SlideTransitionAnnotation],
    current: SlideTransitionAnnotation,
) -> float | None:
    return next(
        (
            transition.change_start_seconds
            for transition in transitions
            if transition.event_type in POSITIVE_TRANSITION_TYPES
            and transition.change_start_seconds > current.change_start_seconds
        ),
        None,
    )
