from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from multimodal_lab.annotation_io import load_jsonl
from multimodal_lab.reader_benchmark_protocol import (
    ReaderBenchmarkLectureSpec,
    ReaderBenchmarkPageSpec,
    ReaderBenchmarkProtocol,
    audit_reader_benchmark_protocol,
    best_source_line_match,
    load_reader_benchmark_protocol,
    materialize_protocol_references,
    normalize_alignment_text,
)
from multimodal_lab.schemas import DatasetSplit, StablePageReference


def test_checked_in_protocol_freezes_five_lecture_split() -> None:
    path = (
        Path(__file__).parents[1]
        / "multimodal_lab"
        / "configs"
        / "reader_benchmark_v2_protocol.json"
    )

    protocol = load_reader_benchmark_protocol(path)

    assert [lecture.lecture_id for lecture in protocol.lectures] == [
        "cs231n-2025-lecture-01",
        "cs231n-2025-lecture-02",
        "cs231n-2026-lecture-03",
        "cs231n-2025-lecture-04",
        "cs231n-2025-lecture-05",
    ]
    assert [lecture.split for lecture in protocol.lectures] == [
        DatasetSplit.train,
        DatasetSplit.train,
        DatasetSplit.train,
        DatasetSplit.validation,
        DatasetSplit.test,
    ]
    assert sum(
        lecture.reference_mode == "materialize"
        for lecture in protocol.lectures
    ) == 2


def test_protocol_refuses_multiple_test_lectures(tmp_path: Path) -> None:
    protocol = _protocol_fixture(tmp_path)
    payload = protocol.model_dump(mode="json")
    payload["lectures"][3]["split"] = "test"

    with pytest.raises(ValidationError, match="one validation lecture"):
        ReaderBenchmarkProtocol.model_validate(payload)


def test_alignment_normalizes_source_punctuation_and_split_lines() -> None:
    assert normalize_alignment_text("Evolution\u2019s Big Bang") == (
        "evolution's big bang"
    )
    score = best_source_line_match(
        "Deep Learning",
        ["Deep ", "Learning"],
    )

    assert score == 1.0


def test_protocol_audit_verifies_assets_and_source_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol_fixture(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        protocol.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "multimodal_lab.reader_benchmark_protocol._extract_source_pages",
        lambda path: [["Official line"]],
    )

    audit = audit_reader_benchmark_protocol(
        protocol,
        protocol_path=protocol_path,
        project_root=tmp_path,
    )

    assert audit.passed
    assert not audit.problems
    assert len(audit.page_records) == 2
    assert all(record.minimum_gold_line_match == 1.0 for record in audit.page_records)
    assert audit.split_lecture_ids == {
        "train": ["lecture-1", "lecture-2", "lecture-3"],
        "validation": ["lecture-4"],
        "test": ["lecture-5"],
    }


def test_materialization_writes_references_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol_fixture(tmp_path)

    def fake_extract_frame(
        executable: str,
        *,
        video_path: Path,
        timestamp_seconds: float,
        output_path: Path,
    ) -> None:
        assert executable == "test-ffmpeg"
        output_path.write_bytes(f"frame-{timestamp_seconds}".encode())

    monkeypatch.setattr(
        "multimodal_lab.reader_benchmark_protocol._extract_frame",
        fake_extract_frame,
    )

    result = materialize_protocol_references(
        protocol,
        project_root=tmp_path,
        ffmpeg_executable="test-ffmpeg",
    )

    assert len(result["materialized_lectures"]) == 2
    for lecture_number in (1, 5):
        base = tmp_path / f"lecture-{lecture_number}"
        references = load_jsonl(
            base / "references.jsonl",
            StablePageReference,
        )
        assert len(references) == 1
        assert references[0].gold_text == "Official line"
        provenance = json.loads(
            (base / "input_provenance_v2.json").read_text(encoding="utf-8")
        )
        assert provenance["model_predictions_used_for_selection"] is False


def _protocol_fixture(tmp_path: Path) -> ReaderBenchmarkProtocol:
    shared = tmp_path / "shared.bin"
    shared.write_bytes(b"frozen")
    digest = hashlib.sha256(b"frozen").hexdigest()
    lectures: list[ReaderBenchmarkLectureSpec] = []
    assignments = [
        DatasetSplit.train,
        DatasetSplit.train,
        DatasetSplit.train,
        DatasetSplit.validation,
        DatasetSplit.test,
    ]
    for index, split in enumerate(assignments, start=1):
        materialize = index in (1, 5)
        references = tmp_path / f"lecture-{index}" / "references.jsonl"
        if not materialize:
            references.parent.mkdir(parents=True, exist_ok=True)
            references.write_bytes(b"frozen")
        lectures.append(
            ReaderBenchmarkLectureSpec(
                lecture_id=f"lecture-{index}",
                split=split,
                job_id=f"job-{index}",
                video_path=str(shared),
                video_sha256=digest,
                transcript_path=str(shared),
                transcript_sha256=digest,
                source_deck_path=str(shared),
                source_deck_sha256=digest,
                source_deck_url=f"https://example.test/lecture-{index}.pdf",
                reference_mode=("materialize" if materialize else "existing"),
                references_path=str(references),
                references_sha256=(None if materialize else digest),
                event_prefix=f"l{index}",
                evaluation_interval_seconds=900,
                selection_protocol="fixed grid",
                pages=(
                    [
                        ReaderBenchmarkPageSpec(
                            timestamp_seconds=30,
                            source_page_number=1,
                            gold_lines=["Official line"],
                        )
                    ]
                    if materialize
                    else []
                ),
            )
        )
    return ReaderBenchmarkProtocol(
        protocol_id="test-protocol",
        historical_cnn_v1_commit="4b59b2c",
        historical_cnn_v1_result_sha256="a" * 64,
        split_policy="three train, one validation, one test",
        test_access_policy="one frozen evaluation",
        source_alignment_policy="official source text",
        lectures=lectures,
    )
