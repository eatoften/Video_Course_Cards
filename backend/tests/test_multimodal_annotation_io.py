import json

import pytest

from multimodal_lab.annotation_io import (
    AnnotationFileError,
    load_jsonl,
    validate_annotation_bundle,
    write_jsonl,
)
from multimodal_lab.schemas import (
    DatasetSplit,
    LectureDatasetManifest,
    SlideTransitionAnnotation,
    StablePageReference,
    TransitionEventType,
)
from multimodal_lab.validate_annotations import main


def make_manifest() -> LectureDatasetManifest:
    return LectureDatasetManifest(
        lecture_id="cs231n-2025-lecture-2",
        job_id="job-2",
        course_id="uncategorized",
        title="Image Classification with Linear Classifiers",
        video_path="data/uploads/job-2.mp4",
        transcript_path="data/transcripts/job-2.json",
        source_deck_urls=["https://example.test/lecture-2.pdf"],
        duration_seconds=4021.4,
        pilot_start_seconds=0,
        pilot_end_seconds=900,
        split=DatasetSplit.smoke,
    )


def make_transition() -> SlideTransitionAnnotation:
    return SlideTransitionAnnotation(
        event_id="lecture-2-event-001",
        lecture_id="cs231n-2025-lecture-2",
        change_start_seconds=12.0,
        stable_at_seconds=12.8,
        from_page=1,
        to_page=2,
        event_type=TransitionEventType.page_change,
    )


def make_reference() -> StablePageReference:
    return StablePageReference(
        page_event_id="lecture-2-event-001",
        lecture_id="cs231n-2025-lecture-2",
        stable_frame_timestamp=13.0,
        image_path="frames/lecture-2-event-001.png",
        page_number=2,
        gold_text="Image classification with linear classifiers",
        technical_terms=["linear classifier"],
        gold_concepts=["linear classification"],
    )


def test_jsonl_round_trip_and_bundle_validation(tmp_path):
    manifest_path = tmp_path / "manifests.jsonl"
    transition_path = tmp_path / "transitions.jsonl"
    reference_path = tmp_path / "references.jsonl"
    write_jsonl(manifest_path, [make_manifest()])
    write_jsonl(transition_path, [make_transition()])
    write_jsonl(reference_path, [make_reference()])

    manifests = load_jsonl(manifest_path, LectureDatasetManifest)
    transitions = load_jsonl(transition_path, SlideTransitionAnnotation)
    references = load_jsonl(reference_path, StablePageReference)
    summary = validate_annotation_bundle(manifests, transitions, references)

    assert manifests == [make_manifest()]
    assert transitions == [make_transition()]
    assert references == [make_reference()]
    assert summary.model_dump() == {
        "lecture_count": 1,
        "transition_event_count": 1,
        "positive_transition_count": 1,
        "stable_page_reference_count": 1,
    }


def test_jsonl_loader_reports_the_failing_line(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text(
        json.dumps(make_manifest().model_dump(mode="json"))
        + "\n{not-json}\n",
        encoding="utf-8",
    )

    with pytest.raises(AnnotationFileError, match=r"broken\.jsonl:2"):
        load_jsonl(path, LectureDatasetManifest)


def test_bundle_rejects_reference_before_stable_frame():
    reference = make_reference().model_copy(
        update={"stable_frame_timestamp": 12.5}
    )

    with pytest.raises(AnnotationFileError, match="before stable_at"):
        validate_annotation_bundle(
            [make_manifest()],
            [make_transition()],
            [reference],
        )


def test_bundle_rejects_transition_outside_pilot_interval():
    transition = make_transition().model_copy(
        update={"change_start_seconds": 899.5, "stable_at_seconds": 900.5}
    )

    with pytest.raises(AnnotationFileError, match="pilot interval"):
        validate_annotation_bundle([make_manifest()], [transition], [])


def test_bundle_rejects_reference_page_mismatch():
    reference = make_reference().model_copy(update={"page_number": 3})

    with pytest.raises(AnnotationFileError, match="does not match"):
        validate_annotation_bundle(
            [make_manifest()],
            [make_transition()],
            [reference],
        )


def test_bundle_ignores_non_semantic_motion_when_bounding_a_reference():
    cursor_motion = SlideTransitionAnnotation(
        event_id="cursor-motion",
        lecture_id="cs231n-2025-lecture-2",
        change_start_seconds=12.9,
        stable_at_seconds=12.9,
        event_type=TransitionEventType.non_semantic_motion,
    )
    reference = make_reference().model_copy(
        update={"stable_frame_timestamp": 13.0}
    )

    summary = validate_annotation_bundle(
        [make_manifest()],
        [make_transition(), cursor_motion],
        [reference],
    )

    assert summary.transition_event_count == 2
    assert summary.positive_transition_count == 1


def test_validation_cli_prints_a_machine_readable_summary(tmp_path, capsys):
    manifest_path = tmp_path / "manifests.jsonl"
    transition_path = tmp_path / "transitions.jsonl"
    reference_path = tmp_path / "references.jsonl"
    write_jsonl(manifest_path, [make_manifest()])
    write_jsonl(transition_path, [make_transition()])
    write_jsonl(reference_path, [make_reference()])

    exit_code = main(
        [
            "--manifests",
            str(manifest_path),
            "--transitions",
            str(transition_path),
            "--references",
            str(reference_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["transition_event_count"] == 1
