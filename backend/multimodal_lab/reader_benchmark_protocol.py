from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator
from pypdf import PdfReader

from .annotation_io import write_jsonl
from .page_reading import sha256_file
from .schemas import DatasetSplit, GoldTextScope, StablePageReference


class ReaderBenchmarkProtocolError(ValueError):
    pass


class ReaderBenchmarkPageSpec(BaseModel):
    timestamp_seconds: float = Field(ge=0)
    source_page_number: int = Field(ge=1)
    gold_lines: list[str] = Field(min_length=1)
    technical_terms: list[str] = Field(default_factory=list)
    gold_concepts: list[str] = Field(default_factory=list)

    @field_validator("gold_lines", "technical_terms", "gold_concepts")
    @classmethod
    def validate_text_list(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = value.strip()
            if not text or "\n" in text or "\r" in text:
                raise ValueError("Protocol text entries must be non-empty lines.")
            if text not in seen:
                cleaned.append(text)
                seen.add(text)
        return cleaned


class ReaderBenchmarkLectureSpec(BaseModel):
    lecture_id: str = Field(min_length=1)
    split: DatasetSplit
    job_id: str = Field(min_length=1)
    video_path: str = Field(min_length=1)
    video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transcript_path: str = Field(min_length=1)
    transcript_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_deck_path: str = Field(min_length=1)
    source_deck_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_deck_url: str = Field(min_length=1)
    reference_mode: Literal["existing", "materialize"]
    references_path: str = Field(min_length=1)
    references_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    event_prefix: str = Field(min_length=1)
    evaluation_interval_seconds: float = Field(gt=0)
    selection_protocol: str = Field(min_length=1)
    pages: list[ReaderBenchmarkPageSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_reference_mode(self) -> Self:
        if self.split is DatasetSplit.smoke:
            raise ValueError("A formal protocol cannot assign a smoke split.")
        if self.reference_mode == "existing":
            if self.references_sha256 is None or self.pages:
                raise ValueError(
                    "Existing references require a hash and no page materialization."
                )
        else:
            if self.references_sha256 is not None or not self.pages:
                raise ValueError(
                    "Materialized references require page specs and no prior hash."
                )
            timestamps = [page.timestamp_seconds for page in self.pages]
            if timestamps != sorted(set(timestamps)):
                raise ValueError(
                    "Materialized page timestamps must be unique and sorted."
                )
            if timestamps[-1] > self.evaluation_interval_seconds:
                raise ValueError("A page timestamp exceeds the frozen interval.")
        return self


class ReaderBenchmarkProtocol(BaseModel):
    schema_version: str = "1.0"
    protocol_id: str = Field(min_length=1)
    historical_cnn_v1_commit: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    historical_cnn_v1_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_policy: str = Field(min_length=1)
    test_access_policy: str = Field(min_length=1)
    source_alignment_policy: str = Field(min_length=1)
    lectures: list[ReaderBenchmarkLectureSpec] = Field(min_length=5)

    @model_validator(mode="after")
    def validate_split_protocol(self) -> Self:
        lecture_ids = [lecture.lecture_id for lecture in self.lectures]
        job_ids = [lecture.job_id for lecture in self.lectures]
        if len(set(lecture_ids)) != len(lecture_ids):
            raise ValueError("Protocol lecture IDs must be unique.")
        if len(set(job_ids)) != len(job_ids):
            raise ValueError("Protocol job IDs must be unique.")
        split_counts = {
            split: sum(lecture.split is split for lecture in self.lectures)
            for split in (
                DatasetSplit.train,
                DatasetSplit.validation,
                DatasetSplit.test,
            )
        }
        if split_counts[DatasetSplit.train] < 3:
            raise ValueError("The v2 protocol requires at least three train lectures.")
        if split_counts[DatasetSplit.validation] != 1:
            raise ValueError("The v2 protocol requires one validation lecture.")
        if split_counts[DatasetSplit.test] != 1:
            raise ValueError("The v2 protocol requires one held-out test lecture.")
        materialized = [
            lecture
            for lecture in self.lectures
            if lecture.reference_mode == "materialize"
        ]
        if len(materialized) < 2:
            raise ValueError("At least two independent lectures must be added.")
        return self


class ProtocolAssetRecord(BaseModel):
    lecture_id: str
    asset_kind: str
    path: str
    expected_sha256: str
    actual_sha256: str
    matched: bool


class ProtocolPageAudit(BaseModel):
    lecture_id: str
    source_page_number: int
    timestamp_seconds: float
    minimum_gold_line_match: float
    gold_line_count: int
    passed: bool


class ReaderBenchmarkProtocolAudit(BaseModel):
    schema_version: str = "1.0"
    protocol_id: str
    protocol_sha256: str
    asset_records: list[ProtocolAssetRecord]
    page_records: list[ProtocolPageAudit]
    split_lecture_ids: dict[str, list[str]]
    problems: list[str]
    passed: bool


def load_reader_benchmark_protocol(
    path: str | Path,
) -> ReaderBenchmarkProtocol:
    return ReaderBenchmarkProtocol.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def audit_reader_benchmark_protocol(
    protocol: ReaderBenchmarkProtocol,
    *,
    protocol_path: str | Path,
    project_root: str | Path,
    minimum_source_match: float = 0.88,
) -> ReaderBenchmarkProtocolAudit:
    root = Path(project_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    asset_records: list[ProtocolAssetRecord] = []
    page_records: list[ProtocolPageAudit] = []
    problems: list[str] = []

    for lecture in protocol.lectures:
        assets = (
            ("video", lecture.video_path, lecture.video_sha256),
            ("transcript", lecture.transcript_path, lecture.transcript_sha256),
            ("source_deck", lecture.source_deck_path, lecture.source_deck_sha256),
        )
        if lecture.reference_mode == "existing":
            assert lecture.references_sha256 is not None
            assets += (
                (
                    "references",
                    lecture.references_path,
                    lecture.references_sha256,
                ),
            )
        for kind, configured_path, expected_hash in assets:
            path = _resolve(configured_path, root=root)
            actual_hash = sha256_file(path) if path.is_file() else "0" * 64
            matched = actual_hash == expected_hash
            asset_records.append(
                ProtocolAssetRecord(
                    lecture_id=lecture.lecture_id,
                    asset_kind=kind,
                    path=_portable(path, root=root),
                    expected_sha256=expected_hash,
                    actual_sha256=actual_hash,
                    matched=matched,
                )
            )
            if not matched:
                problems.append(
                    f"{lecture.lecture_id} {kind} hash mismatch: {path}"
                )

        if lecture.reference_mode == "materialize":
            deck_path = _resolve(lecture.source_deck_path, root=root)
            if not deck_path.is_file():
                continue
            source_pages = _extract_source_pages(deck_path)
            for page in lecture.pages:
                if page.source_page_number > len(source_pages):
                    problems.append(
                        f"{lecture.lecture_id} source page "
                        f"{page.source_page_number} exceeds deck length."
                    )
                    continue
                source_lines = source_pages[page.source_page_number - 1]
                scores = [
                    best_source_line_match(line, source_lines)
                    for line in page.gold_lines
                ]
                minimum = min(scores)
                passed = minimum >= minimum_source_match
                page_records.append(
                    ProtocolPageAudit(
                        lecture_id=lecture.lecture_id,
                        source_page_number=page.source_page_number,
                        timestamp_seconds=page.timestamp_seconds,
                        minimum_gold_line_match=minimum,
                        gold_line_count=len(page.gold_lines),
                        passed=passed,
                    )
                )
                if not passed:
                    problems.append(
                        f"{lecture.lecture_id} page {page.source_page_number} "
                        f"has source match {minimum:.3f}."
                    )

    split_lecture_ids = {
        split.value: sorted(
            lecture.lecture_id
            for lecture in protocol.lectures
            if lecture.split is split
        )
        for split in (
            DatasetSplit.train,
            DatasetSplit.validation,
            DatasetSplit.test,
        )
    }
    return ReaderBenchmarkProtocolAudit(
        protocol_id=protocol.protocol_id,
        protocol_sha256=sha256_file(protocol_file),
        asset_records=asset_records,
        page_records=page_records,
        split_lecture_ids=split_lecture_ids,
        problems=problems,
        passed=not problems,
    )


def materialize_protocol_references(
    protocol: ReaderBenchmarkProtocol,
    *,
    project_root: str | Path,
    ffmpeg_executable: str = "ffmpeg",
    force: bool = False,
) -> dict[str, object]:
    root = Path(project_root).resolve()
    lecture_results: list[dict[str, object]] = []
    for lecture in protocol.lectures:
        if lecture.reference_mode != "materialize":
            continue
        video_path = _resolve(lecture.video_path, root=root)
        references_path = _resolve(lecture.references_path, root=root)
        frame_dir = references_path.parent / "stable_frames_v2"
        frame_dir.mkdir(parents=True, exist_ok=True)
        references: list[StablePageReference] = []
        frame_hashes: dict[str, str] = {}
        for page in lecture.pages:
            timestamp_tag = f"{round(page.timestamp_seconds):04d}"
            frame_path = frame_dir / f"t_{timestamp_tag}.png"
            if force or not frame_path.is_file():
                _extract_frame(
                    ffmpeg_executable,
                    video_path=video_path,
                    timestamp_seconds=page.timestamp_seconds,
                    output_path=frame_path,
                )
            frame_hashes[_portable(frame_path, root=root)] = sha256_file(frame_path)
            references.append(
                StablePageReference(
                    page_event_id=(
                        f"{lecture.event_prefix}-p{page.source_page_number:03d}"
                        f"-t{timestamp_tag}"
                    ),
                    lecture_id=lecture.lecture_id,
                    stable_frame_timestamp=page.timestamp_seconds,
                    image_path=_portable(frame_path, root=root),
                    page_number=page.source_page_number,
                    gold_text="\n".join(page.gold_lines),
                    gold_text_scope=GoldTextScope.verbatim_content,
                    technical_terms=page.technical_terms,
                    gold_concepts=page.gold_concepts,
                )
            )
        references_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(references_path, references)
        reference_hash = sha256_file(references_path)
        provenance = {
            "schema_version": "1.0",
            "protocol_id": protocol.protocol_id,
            "lecture_id": lecture.lecture_id,
            "split": lecture.split.value,
            "job_id": lecture.job_id,
            "video_sha256": lecture.video_sha256,
            "transcript_sha256": lecture.transcript_sha256,
            "source_deck_sha256": lecture.source_deck_sha256,
            "source_deck_url": lecture.source_deck_url,
            "selection_protocol": lecture.selection_protocol,
            "references_sha256": reference_hash,
            "frame_sha256": frame_hashes,
            "model_predictions_used_for_selection": False,
        }
        provenance_path = references_path.parent / "input_provenance_v2.json"
        provenance_path.write_text(
            json.dumps(provenance, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        lecture_results.append(
            {
                "lecture_id": lecture.lecture_id,
                "split": lecture.split.value,
                "page_count": len(references),
                "references_path": _portable(references_path, root=root),
                "references_sha256": reference_hash,
                "provenance_path": _portable(provenance_path, root=root),
                "provenance_sha256": sha256_file(provenance_path),
            }
        )
    return {
        "schema_version": "1.0",
        "protocol_id": protocol.protocol_id,
        "materialized_lectures": lecture_results,
    }


def best_source_line_match(gold_line: str, source_lines: list[str]) -> float:
    gold = normalize_alignment_text(gold_line)
    normalized_lines = [normalize_alignment_text(line) for line in source_lines]
    normalized_lines = [line for line in normalized_lines if line]
    candidates = normalized_lines.copy()
    for window in (2, 3, 4):
        candidates.extend(
            " ".join(normalized_lines[index : index + window])
            for index in range(len(normalized_lines) - window + 1)
        )
    page_text = " ".join(normalized_lines)
    if gold and gold in page_text:
        return 1.0
    return max(
        (SequenceMatcher(None, gold, candidate).ratio() for candidate in candidates),
        default=0.0,
    )


def normalize_alignment_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(
        str.maketrans(
            {
                "\u2018": "'",
                "\u2019": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u2013": "-",
                "\u2014": "-",
                "\u00d7": "x",
                "\u00bd": "1/2",
                "\u2022": "",
            }
        )
    )
    normalized = re.sub(r"\s+", " ", normalized).strip().casefold()
    return normalized


def _extract_source_pages(path: Path) -> list[list[str]]:
    reader = PdfReader(path)
    return [
        [line.strip() for line in (page.extract_text() or "").splitlines() if line.strip()]
        for page in reader.pages
    ]


def _extract_frame(
    ffmpeg_executable: str,
    *,
    video_path: Path,
    timestamp_seconds: float,
    output_path: Path,
) -> None:
    command = [
        ffmpeg_executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp_seconds:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-compression_level",
        "3",
        "-y",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not output_path.is_file():
        raise ReaderBenchmarkProtocolError(
            f"FFmpeg frame extraction failed: {completed.stderr.strip()}"
        )


def _resolve(path: str | Path, *, root: Path) -> Path:
    configured = Path(path)
    return (configured if configured.is_absolute() else root / configured).resolve()


def _portable(path: Path, *, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def protocol_payload_sha256(protocol: ReaderBenchmarkProtocol) -> str:
    payload = protocol.model_dump_json(indent=None)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
