from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

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
    augmentation_path: str | None = None
    augmentation_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_image_shape(self) -> Self:
        if self.max_image_width < self.target_height:
            raise ValueError("max_image_width cannot be below target_height.")
        if (self.augmentation_path is None) != (
            self.augmentation_sha256 is None
        ):
            raise ValueError(
                "augmentation_path and augmentation_sha256 must be set together."
            )
        return self


class ReaderAugmentationConfig(BaseModel):
    schema_version: str = "1.0"
    policy_id: str = Field(min_length=1)
    enabled: bool = True
    seed: int = Field(ge=0)
    rotation_probability: float = Field(ge=0, le=1)
    maximum_rotation_degrees: float = Field(ge=0)
    contrast_probability: float = Field(ge=0, le=1)
    minimum_contrast: float = Field(gt=0)
    maximum_contrast: float = Field(gt=0)
    brightness_probability: float = Field(ge=0, le=1)
    minimum_brightness: float = Field(gt=0)
    maximum_brightness: float = Field(gt=0)
    blur_probability: float = Field(ge=0, le=1)
    maximum_blur_radius: float = Field(ge=0)
    noise_probability: float = Field(ge=0, le=1)
    maximum_noise_standard_deviation: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        if self.minimum_contrast > self.maximum_contrast:
            raise ValueError("Contrast bounds must be ordered.")
        if self.minimum_brightness > self.maximum_brightness:
            raise ValueError("Brightness bounds must be ordered.")
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


class VitCtcEncoderConfig(BaseModel):
    kind: Literal["vit_ctc"] = "vit_ctc"
    input_channels: Literal[1] = 1
    patch_height: int = Field(gt=0)
    patch_width: int = Field(gt=0)
    temporal_downsample: int = Field(gt=0)
    embedding_dim: int = Field(gt=0)
    depth: int = Field(gt=0)
    num_heads: int = Field(gt=0)
    mlp_dim: int = Field(gt=0)
    maximum_position_tokens: int = Field(gt=0)
    output_features: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)
    attention_dropout: float = Field(ge=0, lt=1)
    activation: Literal["gelu"] = "gelu"
    normalization: Literal["pre_layer_norm"] = "pre_layer_norm"
    blank_logit_bias: float = -1.0

    @model_validator(mode="after")
    def validate_attention_contract(self) -> Self:
        if self.temporal_downsample != self.patch_width:
            raise ValueError(
                "ViT temporal_downsample must equal horizontal patch width."
            )
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        return self


ReaderEncoderConfig = Annotated[
    CnnCtcEncoderConfig | VitCtcEncoderConfig,
    Field(discriminator="kind"),
]


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
    model: ReaderEncoderConfig
    data: ReaderDataConfig
    optimization: ReaderOptimizationConfig
    selection: ReaderSelectionConfig = Field(default_factory=ReaderSelectionConfig)

    @model_validator(mode="after")
    def validate_model_data_compatibility(self) -> Self:
        if isinstance(self.model, CnnCtcEncoderConfig) and (
            self.data.target_height < self.model.temporal_downsample
        ):
            raise ValueError(
                "target_height is too small for the configured pooling stages."
            )
        if isinstance(self.model, VitCtcEncoderConfig):
            if self.data.target_height != self.model.patch_height:
                raise ValueError(
                    "ViT patch_height must equal the transformed line height."
                )
            available_width = (
                self.model.maximum_position_tokens * self.model.patch_width
            )
            if self.data.max_image_width > available_width:
                raise ValueError(
                    "ViT position embeddings do not cover max_image_width."
                )
        return self


class VerifiedReaderDataContract(BaseModel):
    dataset_path: Path
    split_path: Path
    audit_path: Path
    image_root: Path
    split: LectureSplitManifest
    audit: ReaderDatasetAuditReport
    augmentation: ReaderAugmentationConfig | None = None


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
    augmentation = None
    if config.data.augmentation_path is not None:
        assert config.data.augmentation_sha256 is not None
        augmentation_path = _verified_path(
            root,
            config.data.augmentation_path,
            config.data.augmentation_sha256,
            label="reader augmentation policy",
        )
        augmentation = ReaderAugmentationConfig.model_validate_json(
            augmentation_path.read_text(encoding="utf-8")
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
        augmentation=augmentation,
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
