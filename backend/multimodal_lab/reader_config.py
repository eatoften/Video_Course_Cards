from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from .experiment_protocol import ReaderDatasetAuditReport
from .page_reading import sha256_file
from .schemas import LectureSplitManifest


class ReaderDataConfig(BaseModel):
    dataset_path: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_path: str = Field(min_length=1)
    split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audit_path: str = Field(min_length=1)
    audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_root: str = Field(min_length=1)
    target_height: int = Field(gt=0)
    max_image_width: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    pin_memory: bool = True

    @model_validator(mode="after")
    def validate_image_shape(self) -> Self:
        if self.max_image_width < self.target_height:
            raise ValueError("max_image_width cannot be below target_height.")
        return self


class CnnCtcEncoderConfig(BaseModel):
    kind: Literal["cnn_ctc"] = "cnn_ctc"
    input_channels: Literal[1] = 1
    channels: list[int] = Field(min_length=2)
    kernel_size: int = Field(default=3, ge=1)
    temporal_downsample: int = Field(default=4, ge=1)
    output_features: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)
    normalization: Literal["channel_layer_norm"] = "channel_layer_norm"
    activation: Literal["gelu"] = "gelu"
    pooling: Literal["max_pool_2x2"] = "max_pool_2x2"
    height_reduction: Literal["mean"] = "mean"
    blank_logit_bias: float = -1.0

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, channels: list[int]) -> list[int]:
        if any(channel <= 0 for channel in channels):
            raise ValueError("CNN channels must be positive.")
        return channels

    @model_validator(mode="after")
    def validate_spatial_contract(self) -> Self:
        if self.kernel_size % 2 == 0:
            raise ValueError("The same-padding CNN requires an odd kernel size.")
        if self.temporal_downsample & (self.temporal_downsample - 1):
            raise ValueError("temporal_downsample must be a power of two.")
        pooling_stages = self.temporal_downsample.bit_length() - 1
        if pooling_stages > len(self.channels):
            raise ValueError(
                "temporal_downsample requires more pooling stages than blocks."
            )
        return self


class ReaderOptimizationConfig(BaseModel):
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    gradient_clip_norm: float = Field(gt=0)
    early_stopping_patience: int = Field(gt=0)
    mixed_precision: bool = True


class ReaderSelectionConfig(BaseModel):
    primary_metric: Literal["character_error_rate"] = "character_error_rate"
    mode: Literal["min"] = "min"
    tie_breaker: Literal["word_error_rate"] = "word_error_rate"
    evaluate_test_during_training: Literal[False] = False


class ReaderExperimentConfig(BaseModel):
    schema_version: str = "1.0"
    experiment_id: str = Field(min_length=1)
    seed: int = Field(ge=0)
    deterministic_algorithms: bool = True
    model: CnnCtcEncoderConfig
    data: ReaderDataConfig
    optimization: ReaderOptimizationConfig
    selection: ReaderSelectionConfig = Field(default_factory=ReaderSelectionConfig)

    @model_validator(mode="after")
    def validate_model_data_compatibility(self) -> Self:
        if self.data.target_height < self.model.temporal_downsample:
            raise ValueError(
                "target_height is too small for the configured pooling stages."
            )
        return self


class VerifiedReaderDataContract(BaseModel):
    dataset_path: Path
    split_path: Path
    audit_path: Path
    image_root: Path
    split: LectureSplitManifest
    audit: ReaderDatasetAuditReport


def load_reader_experiment_config(path: str | Path) -> ReaderExperimentConfig:
    config_path = Path(path)
    return ReaderExperimentConfig.model_validate_json(
        config_path.read_text(encoding="utf-8")
    )


def verify_reader_data_contract(
    config: ReaderExperimentConfig,
    *,
    project_root: str | Path,
) -> VerifiedReaderDataContract:
    root = Path(project_root).resolve()
    dataset_path = _verified_path(
        root,
        config.data.dataset_path,
        config.data.dataset_sha256,
        label="reader dataset",
    )
    split_path = _verified_path(
        root,
        config.data.split_path,
        config.data.split_sha256,
        label="lecture split",
    )
    audit_path = _verified_path(
        root,
        config.data.audit_path,
        config.data.audit_sha256,
        label="dataset audit",
    )
    image_root = _resolve(root, config.data.image_root)
    if not image_root.is_dir():
        raise FileNotFoundError(f"Reader image root does not exist: {image_root}")

    split = LectureSplitManifest.model_validate_json(
        split_path.read_text(encoding="utf-8")
    )
    audit = ReaderDatasetAuditReport.model_validate_json(
        audit_path.read_text(encoding="utf-8")
    )
    if split.dataset_sha256 != config.data.dataset_sha256:
        raise ValueError("Lecture split targets a different reader dataset.")
    if audit.dataset_sha256 != config.data.dataset_sha256:
        raise ValueError("Dataset audit targets a different reader dataset.")
    if audit.split_seed != split.seed:
        raise ValueError("Dataset audit and split use different seeds.")
    if not audit.passed:
        raise ValueError("The configured reader dataset did not pass its audit.")
    if audit.vocabulary_sha256 != config.data.expected_vocabulary_sha256:
        raise ValueError("Training-only character vocabulary hash changed.")
    return VerifiedReaderDataContract(
        dataset_path=dataset_path,
        split_path=split_path,
        audit_path=audit_path,
        image_root=image_root,
        split=split,
        audit=audit,
    )


def _verified_path(
    root: Path,
    configured_path: str,
    expected_sha256: str,
    *,
    label: str,
) -> Path:
    path = _resolve(root, configured_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configured {label} does not exist: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"Configured {label} hash mismatch: expected "
            f"{expected_sha256}, got {actual}."
        )
    return path


def _resolve(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return (path if path.is_absolute() else root / path).resolve()
