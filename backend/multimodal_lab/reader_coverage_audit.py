from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, Field

from .ctc_text import CharacterTokenizer
from .line_crop_dataset import partition_by_lecture_split
from .schemas import DatasetSplit, LectureSplitManifest, LineCropSample


class ReaderCoverageCategory(StrEnum):
    all = "all"
    short = "short"
    long = "long"
    numeric = "numeric"
    punctuation = "punctuation"
    code_or_formula = "code_or_formula"


class ReaderCoverageSlice(BaseModel):
    sample_count: int = Field(ge=0)
    character_count: int = Field(ge=0)
    lecture_ids: list[str]
    label_sources: dict[str, int]


class ReaderCoverageSplitReport(BaseModel):
    sample_count: int = Field(ge=1)
    minimum_length: int = Field(ge=1)
    maximum_length: int = Field(ge=1)
    mean_length: float = Field(gt=0)
    length_bins: dict[str, int]
    categories: dict[ReaderCoverageCategory, ReaderCoverageSlice]
    characters: list[str]
    unknown_characters_from_train: list[str]
    unknown_character_occurrences: int = Field(ge=0)


class ReaderCoverageAuditReport(BaseModel):
    schema_version: str = "1.0"
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    short_text_maximum: int = Field(ge=1)
    long_text_minimum: int = Field(ge=2)
    code_formula_markers: str
    train_vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    splits: dict[DatasetSplit, ReaderCoverageSplitReport]


_CODE_FORMULA_PATTERN = re.compile(r"[=+*/_@#<>\[\]{}()]|->|::")
_LENGTH_BINS = (
    ("01-04", 1, 4),
    ("05-15", 5, 15),
    ("16-31", 16, 31),
    ("32-47", 32, 47),
    ("48-63", 48, 63),
    ("64+", 64, None),
)


def audit_reader_content_coverage(
    samples: Sequence[LineCropSample],
    split: LectureSplitManifest,
    *,
    dataset_sha256: str,
    split_sha256: str,
    short_text_maximum: int = 4,
    long_text_minimum: int = 48,
) -> ReaderCoverageAuditReport:
    if short_text_maximum <= 0:
        raise ValueError("short_text_maximum must be positive.")
    if long_text_minimum <= short_text_maximum:
        raise ValueError("long_text_minimum must exceed the short threshold.")

    partitions = partition_by_lecture_split(samples, split)
    train_tokenizer = CharacterTokenizer.fit(
        sample.normalized_text
        for sample in partitions[DatasetSplit.train]
    )
    train_characters = set(train_tokenizer.spec.characters)
    formal_splits = (
        DatasetSplit.train,
        DatasetSplit.validation,
        DatasetSplit.test,
    )
    reports = {
        dataset_split: _audit_split(
            partitions[dataset_split],
            train_characters=train_characters,
            short_text_maximum=short_text_maximum,
            long_text_minimum=long_text_minimum,
        )
        for dataset_split in formal_splits
    }
    return ReaderCoverageAuditReport(
        dataset_sha256=dataset_sha256,
        split_sha256=split_sha256,
        short_text_maximum=short_text_maximum,
        long_text_minimum=long_text_minimum,
        code_formula_markers=_CODE_FORMULA_PATTERN.pattern,
        train_vocabulary_sha256=train_tokenizer.spec.sha256,
        splits=reports,
    )


def classify_reader_text(
    text: str,
    *,
    short_text_maximum: int = 4,
    long_text_minimum: int = 48,
) -> set[ReaderCoverageCategory]:
    categories = {ReaderCoverageCategory.all}
    if len(text) <= short_text_maximum:
        categories.add(ReaderCoverageCategory.short)
    if len(text) >= long_text_minimum:
        categories.add(ReaderCoverageCategory.long)
    if any(character.isdigit() for character in text):
        categories.add(ReaderCoverageCategory.numeric)
    if any(unicodedata.category(character).startswith("P") for character in text):
        categories.add(ReaderCoverageCategory.punctuation)
    if _CODE_FORMULA_PATTERN.search(text):
        categories.add(ReaderCoverageCategory.code_or_formula)
    return categories


def _audit_split(
    samples: Sequence[LineCropSample],
    *,
    train_characters: set[str],
    short_text_maximum: int,
    long_text_minimum: int,
) -> ReaderCoverageSplitReport:
    if not samples:
        raise ValueError("Every formal coverage split must contain samples.")
    lengths = [len(sample.normalized_text) for sample in samples]
    category_samples: dict[ReaderCoverageCategory, list[LineCropSample]] = {
        category: [] for category in ReaderCoverageCategory
    }
    for sample in samples:
        for category in classify_reader_text(
            sample.normalized_text,
            short_text_maximum=short_text_maximum,
            long_text_minimum=long_text_minimum,
        ):
            category_samples[category].append(sample)

    split_characters = {
        character
        for sample in samples
        for character in sample.normalized_text
    }
    unknown_characters = split_characters - train_characters
    unknown_occurrences = sum(
        character in unknown_characters
        for sample in samples
        for character in sample.normalized_text
    )
    return ReaderCoverageSplitReport(
        sample_count=len(samples),
        minimum_length=min(lengths),
        maximum_length=max(lengths),
        mean_length=sum(lengths) / len(lengths),
        length_bins={
            label: sum(
                minimum <= length and (maximum is None or length <= maximum)
                for length in lengths
            )
            for label, minimum, maximum in _LENGTH_BINS
        },
        categories={
            category: _coverage_slice(category_samples[category])
            for category in ReaderCoverageCategory
        },
        characters=sorted(split_characters),
        unknown_characters_from_train=sorted(unknown_characters),
        unknown_character_occurrences=unknown_occurrences,
    )


def _coverage_slice(samples: Sequence[LineCropSample]) -> ReaderCoverageSlice:
    return ReaderCoverageSlice(
        sample_count=len(samples),
        character_count=sum(len(sample.normalized_text) for sample in samples),
        lecture_ids=sorted({sample.lecture_id for sample in samples}),
        label_sources=dict(
            sorted(Counter(sample.label_source.value for sample in samples).items())
        ),
    )
