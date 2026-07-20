from __future__ import annotations

import json
import math
import random
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

import torch
from pydantic import BaseModel, Field, model_validator
from torch import nn

from .annotation_io import load_jsonl
from .ctc_text import CharacterTokenizer, normalize_line_text
from .line_crop_dataset import LineCropBatch
from .metrics import levenshtein_distance
from .models import build_reader_model
from .page_reading import sha256_file
from .reader_config import (
    ReaderExperimentConfig,
    load_reader_experiment_config,
)
from .reader_coverage_audit import (
    ReaderCoverageCategory,
    classify_reader_text,
)
from .schemas import (
    LineCropReviewRecord,
    LineCropSample,
    LineReviewDecision,
    ReaderEvaluationReport,
    ReaderPrediction,
)
from .training.reader_checkpoint import load_frozen_reader_checkpoint
from .training.reader_protocol import forward_reader_model


class ReaderComparisonModelSpec(BaseModel):
    name: str = Field(min_length=1)
    model_kind: Literal["cnn_ctc", "vit_ctc"]
    config_path: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_path: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameter_count: int = Field(gt=0)


class ReaderComparisonProtocol(BaseModel):
    schema_version: str = "1.0"
    protocol_id: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_lecture_id: str = Field(min_length=1)
    models: list[ReaderComparisonModelSpec] = Field(min_length=2, max_length=2)
    rapidocr_reviews_path: str = Field(min_length=1)
    rapidocr_reviews_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_access_ledger_path: str = Field(min_length=1)
    cpu_thread_count: int = Field(gt=0)
    latency_warmup_runs: int = Field(ge=1)
    latency_timed_runs: int = Field(ge=3)
    bootstrap_seed: int = Field(ge=0)
    bootstrap_iterations: int = Field(ge=1000)
    confidence_level: float = Field(gt=0, lt=1)
    short_text_maximum: int = Field(ge=1)
    long_text_minimum: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_comparison(self) -> Self:
        names = [model.name for model in self.models]
        kinds = [model.model_kind for model in self.models]
        if len(set(names)) != len(names) or set(kinds) != {"cnn_ctc", "vit_ctc"}:
            raise ValueError("Comparison requires one uniquely named CNN and ViT.")
        if self.long_text_minimum <= self.short_text_maximum:
            raise ValueError("Long-text threshold must exceed short-text threshold.")
        return self


class ReaderSliceMetrics(BaseModel):
    sample_count: int = Field(ge=1)
    character_edits: int = Field(ge=0)
    character_count: int = Field(ge=1)
    character_error_rate: float = Field(ge=0)
    word_edits: int = Field(ge=0)
    word_count: int = Field(ge=1)
    word_error_rate: float = Field(ge=0)
    exact_match_count: int = Field(ge=0)
    exact_match_rate: float = Field(ge=0, le=1)


class ReaderLatencySummary(BaseModel):
    warmup_runs: int = Field(ge=1)
    timed_runs: int = Field(ge=3)
    batch_size: int = Field(ge=1)
    sample_count: int = Field(ge=1)
    total_seconds: list[float]
    median_total_seconds: float = Field(gt=0)
    mean_total_seconds: float = Field(gt=0)
    p95_total_seconds: float = Field(gt=0)
    median_milliseconds_per_line: float = Field(gt=0)
    scope: str


class ReaderSystemComparison(BaseModel):
    system_name: str
    parameter_count: int | None = Field(default=None, gt=0)
    mean_ctc_loss: float | None = Field(default=None, ge=0)
    overall: ReaderSliceMetrics
    slices: dict[str, ReaderSliceMetrics]
    latency: ReaderLatencySummary | None = None


class PairedBootstrapDifference(BaseModel):
    system_a: str
    system_b: str
    metric: Literal["character_error_rate"] = "character_error_rate"
    point_difference_a_minus_b: float
    confidence_level: float = Field(gt=0, lt=1)
    lower_bound: float
    upper_bound: float
    iterations: int = Field(ge=1000)
    seed: int = Field(ge=0)
    probability_a_lower_than_b: float = Field(ge=0, le=1)


class ReaderComparisonReport(BaseModel):
    schema_version: str = "1.0"
    protocol_id: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_lecture_id: str
    sample_count: int = Field(ge=1)
    systems: dict[str, ReaderSystemComparison]
    paired_bootstrap: list[PairedBootstrapDifference]
    rapidocr_scope_warning: str
    test_access_count: Literal[1] = 1


@dataclass(frozen=True)
class PreparedReaderModel:
    spec: ReaderComparisonModelSpec
    config: ReaderExperimentConfig
    model: nn.Module


def load_reader_comparison_protocol(
    path: str | Path,
) -> ReaderComparisonProtocol:
    return ReaderComparisonProtocol.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def prepare_reader_comparison(
    protocol: ReaderComparisonProtocol,
    *,
    project_root: str | Path,
    tokenizer: CharacterTokenizer,
    device: torch.device | str,
) -> dict[str, PreparedReaderModel]:
    root = Path(project_root).resolve()
    prepared: dict[str, PreparedReaderModel] = {}
    shared_data_identity: tuple[str, str, str, str] | None = None
    for model_spec in protocol.models:
        config_path = _verified_path(
            root,
            model_spec.config_path,
            model_spec.config_sha256,
            label=f"{model_spec.name} config",
        )
        checkpoint_path = _verified_path(
            root,
            model_spec.checkpoint_path,
            model_spec.checkpoint_sha256,
            label=f"{model_spec.name} checkpoint",
        )
        config = load_reader_experiment_config(config_path)
        if config.model.kind != model_spec.model_kind:
            raise ValueError(f"{model_spec.name} model kind changed.")
        if config.data.dataset_sha256 != protocol.dataset_sha256:
            raise ValueError(f"{model_spec.name} dataset hash changed.")
        if config.data.split_sha256 != protocol.split_sha256:
            raise ValueError(f"{model_spec.name} split hash changed.")
        identity = (
            config.data.dataset_sha256,
            config.data.split_sha256,
            config.data.expected_vocabulary_sha256,
            config.data.augmentation_sha256 or "disabled",
        )
        if shared_data_identity is None:
            shared_data_identity = identity
        elif identity != shared_data_identity:
            raise ValueError("Reader models do not share one frozen data policy.")
        if tokenizer.spec.sha256 != config.data.expected_vocabulary_sha256:
            raise ValueError("Comparison tokenizer does not match the model config.")
        model = build_reader_model(
            config.model,
            tokenizer.vocabulary_size,
            blank_id=tokenizer.blank_id,
        ).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != model_spec.parameter_count:
            raise ValueError(f"{model_spec.name} parameter count changed.")
        load_frozen_reader_checkpoint(
            checkpoint_path,
            model=model,
            config=config,
            tokenizer=tokenizer,
            experiment_config_sha256=model_spec.config_sha256,
            device=device,
        )
        model.eval()
        prepared[model_spec.name] = PreparedReaderModel(
            spec=model_spec,
            config=config,
            model=model,
        )

    _verified_path(
        root,
        protocol.rapidocr_reviews_path,
        protocol.rapidocr_reviews_sha256,
        label="RapidOCR review records",
    )
    return prepared


def benchmark_reader_models(
    models: Mapping[str, nn.Module],
    batches: Sequence[LineCropBatch],
    *,
    vocabulary_size: int,
    warmup_runs: int,
    timed_runs: int,
    device: torch.device | str,
) -> dict[str, ReaderLatencySummary]:
    if not batches:
        raise ValueError("Latency benchmark needs at least one batch.")
    configured_device = torch.device(device)
    batch_size = max(len(batch.texts) for batch in batches)
    sample_count = sum(len(batch.texts) for batch in batches)
    names = list(models)
    timings = {name: [] for name in names}

    with torch.inference_mode():
        for name in names:
            for _ in range(warmup_runs):
                _run_model_batches(
                    models[name],
                    batches,
                    vocabulary_size=vocabulary_size,
                    device=configured_device,
                )
        for repeat in range(timed_runs):
            order = names if repeat % 2 == 0 else list(reversed(names))
            for name in order:
                _synchronize(configured_device)
                started_at = time.perf_counter()
                _run_model_batches(
                    models[name],
                    batches,
                    vocabulary_size=vocabulary_size,
                    device=configured_device,
                )
                _synchronize(configured_device)
                timings[name].append(time.perf_counter() - started_at)

    return {
        name: _latency_summary(
            values,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
            batch_size=batch_size,
            sample_count=sample_count,
        )
        for name, values in timings.items()
    }


def rapidocr_predictions_for_samples(
    samples: Sequence[LineCropSample],
    reviews: Sequence[LineCropReviewRecord],
    *,
    tokenizer: CharacterTokenizer,
) -> list[ReaderPrediction]:
    included_reviews = {
        (review.page_event_id, review.source_block_order): review
        for review in reviews
        if review.decision is LineReviewDecision.include
    }
    predictions: list[ReaderPrediction] = []
    for sample in samples:
        key = (sample.page_event_id, sample.source_block_order)
        review = included_reviews.get(key)
        if review is None:
            raise ValueError(f"RapidOCR review is missing for sample {sample.sample_id}.")
        reference_ids = tokenizer.encode(sample.normalized_text)
        detected_text = normalize_line_text(review.detected_text)
        prediction_ids = tokenizer.encode(detected_text)
        predictions.append(
            ReaderPrediction(
                sample_id=sample.sample_id,
                reference=sample.normalized_text,
                prediction=detected_text,
                scored_reference=_scored_text(reference_ids, tokenizer),
                scored_prediction=_scored_text(prediction_ids, tokenizer),
                exact_match=reference_ids == prediction_ids,
            )
        )
    return predictions


def build_system_comparison(
    system_name: str,
    predictions: Sequence[ReaderPrediction],
    samples_by_id: Mapping[str, LineCropSample],
    *,
    tokenizer: CharacterTokenizer,
    parameter_count: int | None = None,
    mean_ctc_loss: float | None = None,
    latency: ReaderLatencySummary | None = None,
    short_text_maximum: int,
    long_text_minimum: int,
) -> ReaderSystemComparison:
    prediction_by_id = {prediction.sample_id: prediction for prediction in predictions}
    if set(prediction_by_id) != set(samples_by_id):
        raise ValueError(f"{system_name} predictions do not cover test samples.")

    slice_ids: dict[str, list[str]] = {
        "short": [],
        "long": [],
        "numeric": [],
        "punctuation": [],
        "code_or_formula": [],
        "contains_unknown": [],
        "known_characters_only": [],
    }
    category_names = {
        ReaderCoverageCategory.short: "short",
        ReaderCoverageCategory.long: "long",
        ReaderCoverageCategory.numeric: "numeric",
        ReaderCoverageCategory.punctuation: "punctuation",
        ReaderCoverageCategory.code_or_formula: "code_or_formula",
    }
    for sample_id, sample in samples_by_id.items():
        categories = classify_reader_text(
            sample.normalized_text,
            short_text_maximum=short_text_maximum,
            long_text_minimum=long_text_minimum,
        )
        for category, name in category_names.items():
            if category in categories:
                slice_ids[name].append(sample_id)
        encoded = tokenizer.encode(sample.normalized_text)
        unknown_slice = (
            "contains_unknown"
            if tokenizer.unknown_id in encoded
            else "known_characters_only"
        )
        slice_ids[unknown_slice].append(sample_id)

    ordered = [prediction_by_id[sample_id] for sample_id in samples_by_id]
    slices = {
        name: score_reader_predictions(
            [prediction_by_id[sample_id] for sample_id in sample_ids]
        )
        for name, sample_ids in slice_ids.items()
        if sample_ids
    }
    return ReaderSystemComparison(
        system_name=system_name,
        parameter_count=parameter_count,
        mean_ctc_loss=mean_ctc_loss,
        overall=score_reader_predictions(ordered),
        slices=slices,
        latency=latency,
    )


def score_reader_predictions(
    predictions: Sequence[ReaderPrediction],
) -> ReaderSliceMetrics:
    if not predictions:
        raise ValueError("Cannot score an empty prediction slice.")
    character_edits = sum(
        levenshtein_distance(item.scored_reference, item.scored_prediction)
        for item in predictions
    )
    character_count = sum(len(item.scored_reference) for item in predictions)
    word_edits = sum(
        levenshtein_distance(
            item.scored_reference.split(),
            item.scored_prediction.split(),
        )
        for item in predictions
    )
    word_count = sum(len(item.scored_reference.split()) for item in predictions)
    exact_match_count = sum(item.exact_match for item in predictions)
    return ReaderSliceMetrics(
        sample_count=len(predictions),
        character_edits=character_edits,
        character_count=character_count,
        character_error_rate=character_edits / character_count,
        word_edits=word_edits,
        word_count=word_count,
        word_error_rate=word_edits / word_count,
        exact_match_count=exact_match_count,
        exact_match_rate=exact_match_count / len(predictions),
    )


def paired_bootstrap_cer_difference(
    system_a: str,
    predictions_a: Sequence[ReaderPrediction],
    system_b: str,
    predictions_b: Sequence[ReaderPrediction],
    *,
    seed: int,
    iterations: int,
    confidence_level: float,
) -> PairedBootstrapDifference:
    rows_a = {item.sample_id: item for item in predictions_a}
    rows_b = {item.sample_id: item for item in predictions_b}
    if rows_a.keys() != rows_b.keys():
        raise ValueError("Paired bootstrap requires identical sample IDs.")
    sample_ids = sorted(rows_a)
    reference_lengths = [len(rows_a[sample_id].scored_reference) for sample_id in sample_ids]
    edits_a = [
        levenshtein_distance(
            rows_a[sample_id].scored_reference,
            rows_a[sample_id].scored_prediction,
        )
        for sample_id in sample_ids
    ]
    edits_b = [
        levenshtein_distance(
            rows_b[sample_id].scored_reference,
            rows_b[sample_id].scored_prediction,
        )
        for sample_id in sample_ids
    ]
    point = sum(edits_a) / sum(reference_lengths) - sum(edits_b) / sum(
        reference_lengths
    )
    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(iterations):
        indexes = [rng.randrange(len(sample_ids)) for _ in sample_ids]
        denominator = sum(reference_lengths[index] for index in indexes)
        differences.append(
            sum(edits_a[index] for index in indexes) / denominator
            - sum(edits_b[index] for index in indexes) / denominator
        )
    differences.sort()
    alpha = (1 - confidence_level) / 2
    return PairedBootstrapDifference(
        system_a=system_a,
        system_b=system_b,
        point_difference_a_minus_b=point,
        confidence_level=confidence_level,
        lower_bound=_percentile(differences, alpha),
        upper_bound=_percentile(differences, 1 - alpha),
        iterations=iterations,
        seed=seed,
        probability_a_lower_than_b=(
            sum(difference < 0 for difference in differences) / iterations
        ),
    )


def claim_sealed_test_access(
    ledger_path: str | Path,
    *,
    protocol_id: str,
    protocol_sha256: str,
    run_id: str,
) -> dict[str, object]:
    path = Path(ledger_path)
    if path.exists():
        raise RuntimeError(
            f"Sealed test has already been opened; ledger exists at {path}."
        )
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha256,
        "run_id": run_id,
        "test_access_count": 1,
        "status": "opened",
        "opened_at": datetime.now(UTC).isoformat(),
        "completed_at": None,
        "failure": None,
    }
    _write_json(path, payload)
    return payload


def finalize_test_access(
    ledger_path: str | Path,
    payload: Mapping[str, object],
    *,
    status: Literal["completed", "failed"],
    failure: str | None = None,
) -> None:
    updated = {
        **payload,
        "status": status,
        "completed_at": datetime.now(UTC).isoformat(),
        "failure": failure,
    }
    _write_json(Path(ledger_path), updated)


def _run_model_batches(
    model: nn.Module,
    batches: Sequence[LineCropBatch],
    *,
    vocabulary_size: int,
    device: torch.device,
) -> None:
    model.eval()
    for batch in batches:
        forward_reader_model(
            model,
            batch.images.to(device),
            batch.widths.to(device),
            vocabulary_size=vocabulary_size,
        )


def _latency_summary(
    values: Sequence[float],
    *,
    warmup_runs: int,
    timed_runs: int,
    batch_size: int,
    sample_count: int,
) -> ReaderLatencySummary:
    ordered = sorted(values)
    median = statistics.median(values)
    return ReaderLatencySummary(
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        batch_size=batch_size,
        sample_count=sample_count,
        total_seconds=list(values),
        median_total_seconds=median,
        mean_total_seconds=statistics.fmean(values),
        p95_total_seconds=ordered[math.ceil(0.95 * len(ordered)) - 1],
        median_milliseconds_per_line=1000 * median / sample_count,
        scope="model forward only; in-memory batches; decoding and image I/O excluded",
    )


def _scored_text(token_ids: Sequence[int], tokenizer: CharacterTokenizer) -> str:
    return "".join(
        "\ufffd" if token_id == tokenizer.unknown_id else tokenizer.decode([token_id])
        for token_id in token_ids
    )


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _verified_path(root: Path, configured: str, expected: str, *, label: str) -> Path:
    path = Path(configured)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Configured {label} does not exist: {resolved}")
    actual = sha256_file(resolved)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: expected {expected}, got {actual}.")
    return resolved


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
