from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from app.card_service import CardDraftResponse

from .annotation_io import load_jsonl
from .ctc_text import CharacterTokenizer
from .line_crop_dataset import partition_by_lecture_split
from .page_reading import sha256_file
from .reader_comparison import (
    ReaderComparisonReport,
    rapidocr_predictions_for_samples,
)
from .schemas import (
    DatasetSplit,
    LectureSplitManifest,
    LineCropReviewRecord,
    LineCropSample,
    ReaderEvaluationReport,
    ReaderPrediction,
    StablePageReference,
)


ReaderCardSystemName = Literal["cnn_v2", "vit_v1", "rapidocr_stored"]


class ReaderCardTextSource(BaseModel):
    system_name: ReaderCardSystemName
    source_kind: Literal["reader_evaluation", "rapidocr_reviews"]
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ReaderCardCascadeProtocol(BaseModel):
    schema_version: str = "1.0"
    protocol_id: str = Field(min_length=1)
    source_comparison_protocol_id: str = Field(min_length=1)
    source_comparison_path: str = Field(min_length=1)
    source_comparison_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_path: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_path: str = Field(min_length=1)
    split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_lecture_id: str = Field(min_length=1)
    references_path: str = Field(min_length=1)
    references_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    card_service_path: str = Field(min_length=1)
    card_service_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    text_sources: list[ReaderCardTextSource] = Field(min_length=3, max_length=3)
    provider: Literal["ollama"] = "ollama"
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    model_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    temperature: Literal[0.0] = 0.0
    max_tokens: int = Field(ge=1)
    timeout_seconds: float = Field(ge=1)
    reasoning_effort: Literal["default", "none"] = "default"
    card_count: Literal[1] = 1
    focus: str = Field(min_length=1, max_length=500)
    schedule: Literal["cyclic_page_order"] = "cyclic_page_order"
    review_method: str = Field(min_length=1)
    citation_similarity_threshold: float = Field(ge=0, le=1)
    bootstrap_seed: int = Field(ge=0)
    bootstrap_iterations: int = Field(ge=1000)
    confidence_level: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def validate_sources(self) -> Self:
        names = [source.system_name for source in self.text_sources]
        if len(set(names)) != 3 or set(names) != {
            "cnn_v2",
            "vit_v1",
            "rapidocr_stored",
        }:
            raise ValueError("Card cascade requires CNN, ViT, and RapidOCR.")
        kinds = {source.system_name: source.source_kind for source in self.text_sources}
        if kinds["rapidocr_stored"] != "rapidocr_reviews":
            raise ValueError("RapidOCR must use the frozen review records.")
        if kinds["cnn_v2"] != "reader_evaluation":
            raise ValueError("CNN must use its frozen evaluation report.")
        if kinds["vit_v1"] != "reader_evaluation":
            raise ValueError("ViT must use its frozen evaluation report.")
        return self


class ReaderCardInputCase(BaseModel):
    protocol_id: str = Field(min_length=1)
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    system_name: ReaderCardSystemName
    page_event_id: str = Field(min_length=1)
    lecture_id: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    stable_frame_timestamp: float = Field(ge=0)
    schedule_position: int = Field(ge=0, le=2)
    sample_ids: list[str]
    lines: list[str]
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def key(self) -> tuple[str, str]:
        return self.system_name, self.page_event_id


class ReaderCardGenerationStatus(str, Enum):
    succeeded = "succeeded"
    failed = "failed"


class ReaderCardLLMCall(BaseModel):
    call_index: int = Field(ge=0)
    messages: list[dict[str, str]]
    model: str
    temperature: float
    max_tokens: int
    reasoning_effort: str | None = None
    response_format: dict[str, object] | None = None
    elapsed_seconds: float = Field(ge=0)
    raw_output: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        has_output = self.raw_output is not None
        has_error = self.error_type is not None or self.error_message is not None
        if has_output == has_error:
            raise ValueError("An LLM call must have exactly one outcome.")
        if has_error and (not self.error_type or not self.error_message):
            raise ValueError("An LLM call error requires type and message.")
        return self


class ReaderCardGenerationRecord(BaseModel):
    schema_version: str = "1.0"
    protocol_id: str = Field(min_length=1)
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str = Field(min_length=1)
    model_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    system_name: ReaderCardSystemName
    page_event_id: str = Field(min_length=1)
    lecture_id: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    stable_frame_timestamp: float = Field(ge=0)
    schedule_position: int = Field(ge=0, le=2)
    sample_ids: list[str]
    input_lines: list[str]
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: ReaderCardGenerationStatus
    generated_at: datetime
    elapsed_seconds: float = Field(ge=0)
    llm_calls: list[ReaderCardLLMCall] = Field(default_factory=list)
    response: CardDraftResponse | None = None
    error_type: str | None = None
    error_message: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.system_name, self.page_event_id

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.status is ReaderCardGenerationStatus.succeeded:
            if self.response is None or self.error_type or self.error_message:
                raise ValueError("A successful generation requires only a response.")
        else:
            if self.response is not None or not self.error_type or not self.error_message:
                raise ValueError("A failed generation requires only an error.")
        return self


class ReaderCardGenerationManifest(BaseModel):
    schema_version: str = "1.0"
    protocol_id: str = Field(min_length=1)
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str = Field(min_length=1)
    model_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_count: int = Field(ge=1)
    expected_record_count: int = Field(ge=1)
    completed_record_count: int = Field(ge=0)
    succeeded_record_count: int = Field(ge=0)
    failed_record_count: int = Field(ge=0)
    records_path: str = Field(min_length=1)
    records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    code_fingerprints: dict[str, str]
    completed_at: datetime

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.completed_record_count != (
            self.succeeded_record_count + self.failed_record_count
        ):
            raise ValueError("Generation manifest outcome counts do not add up.")
        if self.completed_record_count > self.expected_record_count:
            raise ValueError("Generation produced more records than expected.")
        return self


@dataclass(frozen=True)
class FrozenReaderCardInputs:
    references: list[StablePageReference]
    cases: list[ReaderCardInputCase]


def load_reader_card_cascade_protocol(
    path: str | Path,
) -> ReaderCardCascadeProtocol:
    return ReaderCardCascadeProtocol.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def load_frozen_reader_card_inputs(
    protocol: ReaderCardCascadeProtocol,
    *,
    protocol_sha256: str,
    project_root: str | Path,
) -> FrozenReaderCardInputs:
    root = Path(project_root).resolve()
    comparison_path = _verified_path(
        root,
        protocol.source_comparison_path,
        protocol.source_comparison_sha256,
        label="source comparison",
    )
    comparison = ReaderComparisonReport.model_validate_json(
        comparison_path.read_text(encoding="utf-8")
    )
    if comparison.protocol_id != protocol.source_comparison_protocol_id:
        raise ValueError("Source comparison protocol ID changed.")
    if comparison.dataset_sha256 != protocol.dataset_sha256:
        raise ValueError("Source comparison dataset hash changed.")
    if comparison.split_sha256 != protocol.split_sha256:
        raise ValueError("Source comparison split hash changed.")
    if comparison.test_lecture_id != protocol.test_lecture_id:
        raise ValueError("Source comparison test lecture changed.")

    dataset_path = _verified_path(
        root,
        protocol.dataset_path,
        protocol.dataset_sha256,
        label="line-crop dataset",
    )
    split_path = _verified_path(
        root,
        protocol.split_path,
        protocol.split_sha256,
        label="lecture split",
    )
    references_path = _verified_path(
        root,
        protocol.references_path,
        protocol.references_sha256,
        label="stable-page references",
    )
    _verified_path(
        root,
        protocol.card_service_path,
        protocol.card_service_sha256,
        label="card service",
    )

    samples = load_jsonl(dataset_path, LineCropSample)
    split = LectureSplitManifest.model_validate_json(
        split_path.read_text(encoding="utf-8")
    )
    partitions = partition_by_lecture_split(samples, split)
    test_samples = partitions[DatasetSplit.test]
    if split.test_lecture_ids != [protocol.test_lecture_id]:
        raise ValueError("Cascade protocol does not name the frozen test lecture.")
    if {sample.lecture_id for sample in test_samples} != {protocol.test_lecture_id}:
        raise ValueError("Test samples contain an unexpected lecture.")

    tokenizer = CharacterTokenizer.fit(
        sample.normalized_text for sample in partitions[DatasetSplit.train]
    )
    if tokenizer.spec.sha256 != protocol.vocabulary_sha256:
        raise ValueError("Frozen training vocabulary changed.")

    references = load_jsonl(references_path, StablePageReference)
    references = sorted(
        references,
        key=lambda item: (item.stable_frame_timestamp, item.page_event_id),
    )
    if {reference.lecture_id for reference in references} != {
        protocol.test_lecture_id
    }:
        raise ValueError("Stable references contain an unexpected lecture.")
    if any(len(reference.gold_concepts) != 1 for reference in references):
        raise ValueError("The frozen card study requires one concept per page.")

    predictions_by_system: dict[str, list[ReaderPrediction]] = {}
    for source in protocol.text_sources:
        source_path = _verified_path(
            root,
            source.path,
            source.sha256,
            label=f"{source.system_name} text source",
        )
        if source.source_kind == "reader_evaluation":
            report = ReaderEvaluationReport.model_validate_json(
                source_path.read_text(encoding="utf-8")
            )
            if report.split is not DatasetSplit.test:
                raise ValueError(f"{source.system_name} is not a test report.")
            predictions_by_system[source.system_name] = report.predictions
        else:
            reviews = load_jsonl(source_path, LineCropReviewRecord)
            predictions_by_system[source.system_name] = (
                rapidocr_predictions_for_samples(
                    test_samples,
                    reviews,
                    tokenizer=tokenizer,
                )
            )

    samples_by_page: dict[str, list[LineCropSample]] = defaultdict(list)
    for sample in test_samples:
        samples_by_page[sample.page_event_id].append(sample)
    for page_samples in samples_by_page.values():
        page_samples.sort(key=lambda item: item.source_block_order)
    reference_ids = {reference.page_event_id for reference in references}
    if set(samples_by_page) != reference_ids:
        raise ValueError("Test line crops and stable pages do not align exactly.")

    ordered_systems: tuple[ReaderCardSystemName, ...] = (
        "cnn_v2",
        "vit_v1",
        "rapidocr_stored",
    )
    prediction_maps = {
        system_name: _prediction_map(
            predictions_by_system[system_name],
            test_samples,
            system_name=system_name,
        )
        for system_name in ordered_systems
    }
    cases: list[ReaderCardInputCase] = []
    for page_index, reference in enumerate(references):
        rotated = ordered_systems[page_index % 3 :] + ordered_systems[: page_index % 3]
        page_samples = samples_by_page[reference.page_event_id]
        for schedule_position, system_name in enumerate(rotated):
            lines = [
                prediction_maps[system_name][sample.sample_id].prediction.strip()
                for sample in page_samples
            ]
            lines = [line for line in lines if line]
            sample_ids = [sample.sample_id for sample in page_samples]
            cases.append(
                ReaderCardInputCase(
                    protocol_id=protocol.protocol_id,
                    protocol_sha256=protocol_sha256,
                    system_name=system_name,
                    page_event_id=reference.page_event_id,
                    lecture_id=reference.lecture_id,
                    page_number=reference.page_number,
                    stable_frame_timestamp=reference.stable_frame_timestamp,
                    schedule_position=schedule_position,
                    sample_ids=sample_ids,
                    lines=lines,
                    input_sha256=card_input_sha256(
                        system_name,
                        reference.page_event_id,
                        sample_ids,
                        lines,
                    ),
                )
            )
    return FrozenReaderCardInputs(references=references, cases=cases)


def card_input_sha256(
    system_name: str,
    page_event_id: str,
    sample_ids: Sequence[str],
    lines: Sequence[str],
) -> str:
    payload = json.dumps(
        {
            "system_name": system_name,
            "page_event_id": page_event_id,
            "sample_ids": list(sample_ids),
            "lines": list(lines),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_generation_records(
    path: str | Path,
) -> list[ReaderCardGenerationRecord]:
    resolved_path = Path(path)
    if not resolved_path.exists():
        return []
    records = load_jsonl(resolved_path, ReaderCardGenerationRecord)
    keys = [record.key for record in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Generation records contain duplicate system/page keys.")
    return records


def write_generation_records_atomic(
    path: str | Path,
    records: Sequence[ReaderCardGenerationRecord],
) -> None:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = resolved_path.with_suffix(resolved_path.suffix + ".tmp")
    content = "".join(
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n"
        for record in records
    )
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(resolved_path)


def validate_generation_records(
    cases: Sequence[ReaderCardInputCase],
    records: Sequence[ReaderCardGenerationRecord],
    *,
    protocol: ReaderCardCascadeProtocol,
    protocol_sha256: str,
) -> None:
    cases_by_key = {case.key: case for case in cases}
    if len(cases_by_key) != len(cases):
        raise ValueError("Input cases contain duplicate system/page keys.")
    for record in records:
        case = cases_by_key.get(record.key)
        if case is None:
            raise ValueError(f"Generation record has unknown key: {record.key}.")
        if record.protocol_id != protocol.protocol_id:
            raise ValueError("Generation record protocol ID changed.")
        if record.protocol_sha256 != protocol_sha256:
            raise ValueError("Generation record protocol hash changed.")
        if record.model != protocol.model:
            raise ValueError("Generation record model changed.")
        if record.model_digest != protocol.model_digest:
            raise ValueError("Generation record model digest changed.")
        if record.input_sha256 != case.input_sha256:
            raise ValueError(f"Generation input changed for {record.key}.")


def _prediction_map(
    predictions: Sequence[ReaderPrediction],
    samples: Sequence[LineCropSample],
    *,
    system_name: str,
) -> Mapping[str, ReaderPrediction]:
    prediction_map = {prediction.sample_id: prediction for prediction in predictions}
    sample_ids = {sample.sample_id for sample in samples}
    if len(prediction_map) != len(predictions) or set(prediction_map) != sample_ids:
        raise ValueError(f"{system_name} does not cover the frozen test samples.")
    return prediction_map


def _verified_path(
    root: Path,
    relative_path: str,
    expected_sha256: str,
    *,
    label: str,
) -> Path:
    path = (root / relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} hash changed: expected {expected_sha256}, got {actual_sha256}."
        )
    return path
