from __future__ import annotations

from multimodal_lab.reader_coverage_audit import (
    ReaderCoverageCategory,
    audit_reader_content_coverage,
    classify_reader_text,
)
from multimodal_lab.schemas import (
    DatasetSplit,
    LectureSplitManifest,
    LineCropSample,
    LineLabelSource,
    PageReaderKind,
    PixelCrop,
)


def test_classify_reader_text_supports_overlapping_slices() -> None:
    categories = classify_reader_text("f(x_1) = 42;", long_text_minimum=12)

    assert categories == {
        ReaderCoverageCategory.all,
        ReaderCoverageCategory.long,
        ReaderCoverageCategory.numeric,
        ReaderCoverageCategory.punctuation,
        ReaderCoverageCategory.code_or_formula,
    }


def test_coverage_audit_uses_train_only_vocabulary() -> None:
    samples = [
        _sample("a" * 64, "train-lecture", "A1"),
        _sample("b" * 64, "validation-lecture", "B?"),
        _sample("c" * 64, "test-lecture", "C7"),
    ]
    split = LectureSplitManifest(
        dataset_sha256="d" * 64,
        seed=17,
        train_lecture_ids=["train-lecture"],
        validation_lecture_ids=["validation-lecture"],
        test_lecture_ids=["test-lecture"],
    )

    report = audit_reader_content_coverage(
        samples,
        split,
        dataset_sha256="d" * 64,
        split_sha256="e" * 64,
    )

    validation = report.splits[DatasetSplit.validation]
    test = report.splits[DatasetSplit.test]
    assert validation.unknown_characters_from_train == ["?", "B"]
    assert validation.unknown_character_occurrences == 2
    assert test.categories[ReaderCoverageCategory.numeric].sample_count == 1
    assert test.unknown_characters_from_train == ["7", "C"]


def _sample(sample_id: str, lecture_id: str, text: str) -> LineCropSample:
    return LineCropSample(
        sample_id=sample_id,
        lecture_id=lecture_id,
        page_event_id=f"{lecture_id}-page",
        page_number=1,
        stable_frame_timestamp=0,
        source_image_path=f"{sample_id}.png",
        source_image_sha256="1" * 64,
        crop_path=f"{sample_id}-crop.png",
        crop_sha256="2" * 64,
        bounding_box=PixelCrop(x=0, y=0, width=10, height=10),
        text=text,
        normalized_text=text,
        label_source=LineLabelSource.source_aligned,
        detector_reader=PageReaderKind.rapidocr,
        detector_version="test",
        detector_preprocessing_version="test",
        detector_cache_key="3" * 64,
        source_block_order=0,
    )
