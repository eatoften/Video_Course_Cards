from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

from .ctc_text import CharacterTokenizer
from .line_crop_dataset import partition_by_lecture_split
from .schemas import (
    DatasetSplit,
    LectureSplitManifest,
    LineCropSample,
    LineLabelSource,
)


class ExperimentTask(str, Enum):
    ctc_overfit = "ctc_overfit"
    reader_benchmark = "reader_benchmark"


class ExperimentPhase(str, Enum):
    diagnostic = "diagnostic"
    formal = "formal"


ParameterValue = str | int | float | bool


class ExperimentRunSpec(BaseModel):
    """Frozen inputs and policies for one reproducible experiment run."""

    schema_version: str = "1.0"
    experiment_id: str = Field(min_length=1)
    task: ExperimentTask
    phase: ExperimentPhase
    model_variant: str = Field(min_length=1)
    seed: int = Field(ge=0)
    deterministic_algorithms: bool
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    primary_metric: str = Field(min_length=1)
    parameters: dict[str, ParameterValue] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    @field_validator("experiment_id", "model_variant", "primary_metric")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Experiment identifiers cannot be blank.")
        return stripped

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, tags: list[str]) -> list[str]:
        cleaned = [tag.strip() for tag in tags]
        if any(not tag for tag in cleaned):
            raise ValueError("Experiment tags cannot be blank.")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("Experiment tags must be unique.")
        return cleaned

    @model_validator(mode="after")
    def validate_phase_contract(self) -> Self:
        if self.task is ExperimentTask.ctc_overfit:
            if self.phase is not ExperimentPhase.diagnostic:
                raise ValueError("The CTC overfit gate is diagnostic only.")
            if self.split_sha256 is not None:
                raise ValueError("The CTC overfit gate must not claim a split.")
        elif self.phase is ExperimentPhase.formal and self.split_sha256 is None:
            raise ValueError("Formal experiments require a frozen split hash.")
        return self


class SplitCharacterCoverage(BaseModel):
    character_count: int = Field(ge=0)
    unknown_character_count: int = Field(ge=0)
    unknown_character_rate: float = Field(ge=0, le=1)
    unknown_characters: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.unknown_character_count > self.character_count:
            raise ValueError(
                "unknown_character_count cannot exceed character_count."
            )
        return self


class ReaderDatasetAuditReport(BaseModel):
    schema_version: str = "1.0"
    phase: ExperimentPhase
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_seed: int = Field(ge=0)
    sample_count: int = Field(ge=1)
    lecture_count: int = Field(ge=1)
    split_sample_counts: dict[DatasetSplit, int]
    split_lecture_ids: dict[DatasetSplit, list[str]]
    label_sources_by_split: dict[DatasetSplit, dict[LineLabelSource, int]]
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    character_coverage: dict[DatasetSplit, SplitCharacterCoverage]
    cross_split_crop_hashes: list[str] = Field(default_factory=list)
    cross_split_source_image_hashes: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    passed: bool

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        expected_splits = set(DatasetSplit) - {DatasetSplit.smoke}
        if set(self.split_sample_counts) != expected_splits:
            raise ValueError("Audit counts must cover train, validation, and test.")
        if set(self.split_lecture_ids) != expected_splits:
            raise ValueError("Audit lecture IDs must cover every formal split.")
        if set(self.character_coverage) != expected_splits:
            raise ValueError("Character coverage must cover every formal split.")
        if self.passed == bool(self.problems):
            raise ValueError("passed must be true exactly when problems is empty.")
        return self


def audit_reader_dataset(
    samples: Sequence[LineCropSample],
    split: LectureSplitManifest,
    *,
    dataset_sha256: str,
    phase: ExperimentPhase = ExperimentPhase.formal,
) -> ReaderDatasetAuditReport:
    """Audit split integrity and fit the shared vocabulary on train only."""

    partitions = partition_by_lecture_split(samples, split)
    formal_splits = (
        DatasetSplit.train,
        DatasetSplit.validation,
        DatasetSplit.test,
    )
    problems: list[str] = []
    warnings: list[str] = []

    if split.dataset_sha256 != dataset_sha256:
        problems.append(
            "The split manifest dataset hash does not match the dataset file."
        )

    sample_ids = [sample.sample_id for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        problems.append("Sample IDs are not globally unique.")

    cross_split_crop_hashes = _cross_split_hashes(
        partitions,
        key=lambda sample: sample.crop_sha256,
    )
    if cross_split_crop_hashes:
        problems.append("Identical line crops occur in more than one split.")

    cross_split_source_hashes = _cross_split_hashes(
        partitions,
        key=lambda sample: sample.source_image_sha256,
    )
    if cross_split_source_hashes:
        problems.append("Identical source pages occur in more than one split.")

    label_sources_by_split = {
        dataset_split: dict(
            Counter(sample.label_source for sample in partitions[dataset_split])
        )
        for dataset_split in formal_splits
    }
    if phase is ExperimentPhase.formal:
        _audit_formal_label_sources(
            label_sources_by_split,
            problems=problems,
        )

    train_texts = [
        sample.normalized_text
        for sample in partitions[DatasetSplit.train]
    ]
    tokenizer = CharacterTokenizer.fit(train_texts)
    character_coverage = {
        dataset_split: _character_coverage(
            partitions[dataset_split],
            tokenizer,
        )
        for dataset_split in formal_splits
    }
    for dataset_split in (DatasetSplit.validation, DatasetSplit.test):
        coverage = character_coverage[dataset_split]
        if coverage.unknown_character_count:
            warnings.append(
                f"{dataset_split.value} contains characters absent from the "
                "training vocabulary: "
                + ", ".join(repr(value) for value in coverage.unknown_characters)
            )

    return ReaderDatasetAuditReport(
        phase=phase,
        dataset_sha256=dataset_sha256,
        split_seed=split.seed,
        sample_count=len(samples),
        lecture_count=len({sample.lecture_id for sample in samples}),
        split_sample_counts={
            dataset_split: len(partitions[dataset_split])
            for dataset_split in formal_splits
        },
        split_lecture_ids={
            DatasetSplit.train: split.train_lecture_ids,
            DatasetSplit.validation: split.validation_lecture_ids,
            DatasetSplit.test: split.test_lecture_ids,
        },
        label_sources_by_split=label_sources_by_split,
        vocabulary_sha256=tokenizer.spec.sha256,
        character_coverage=character_coverage,
        cross_split_crop_hashes=cross_split_crop_hashes,
        cross_split_source_image_hashes=cross_split_source_hashes,
        problems=problems,
        warnings=warnings,
        passed=not problems,
    )


def _audit_formal_label_sources(
    sources: dict[DatasetSplit, dict[LineLabelSource, int]],
    *,
    problems: list[str],
) -> None:
    if sources[DatasetSplit.train].get(LineLabelSource.manual_exact_match, 0):
        problems.append(
            "Formal training cannot use OCR-selected exact-match crops; use "
            "human-corrected or synthetic-render labels."
        )

    for dataset_split in (DatasetSplit.validation, DatasetSplit.test):
        invalid_sources = set(sources[dataset_split]) - {
            LineLabelSource.human_corrected
        }
        if invalid_sources:
            problems.append(
                f"Formal {dataset_split.value} labels must be human-corrected "
                "and independent of the evaluated reader."
            )


def _cross_split_hashes(
    partitions: dict[DatasetSplit, list[LineCropSample]],
    *,
    key: Callable[[LineCropSample], str],
) -> list[str]:
    locations: dict[str, set[DatasetSplit]] = defaultdict(set)
    for dataset_split, samples in partitions.items():
        for sample in samples:
            locations[key(sample)].add(dataset_split)
    return sorted(
        value
        for value, dataset_splits in locations.items()
        if len(dataset_splits) > 1
    )


def _character_coverage(
    samples: Sequence[LineCropSample],
    tokenizer: CharacterTokenizer,
) -> SplitCharacterCoverage:
    characters = [
        character
        for sample in samples
        for character in sample.normalized_text
    ]
    known_characters = set(tokenizer.spec.characters)
    unknown = [
        character
        for character in characters
        if character not in known_characters
    ]
    return SplitCharacterCoverage(
        character_count=len(characters),
        unknown_character_count=len(unknown),
        unknown_character_rate=(len(unknown) / len(characters) if characters else 0),
        unknown_characters=sorted(set(unknown)),
    )
