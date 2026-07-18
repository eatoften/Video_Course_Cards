from pathlib import Path

from multimodal_lab.annotation_io import write_jsonl
from multimodal_lab.experiment_protocol import (
    ExperimentPhase,
    ExperimentRunSpec,
    ExperimentTask,
    audit_reader_dataset,
)
from multimodal_lab.page_reading import sha256_file
from multimodal_lab.run_formal_reader_dataset import main as dataset_cli_main
from multimodal_lab.schemas import (
    LectureSplitManifest,
    LineCropSample,
    LineLabelSource,
    PageReaderKind,
    PixelCrop,
)


DATASET_HASH = "d" * 64


def make_sample(
    index: int,
    *,
    lecture_id: str,
    text: str,
    label_source: LineLabelSource = LineLabelSource.human_corrected,
    crop_sha256: str | None = None,
    source_sha256: str | None = None,
) -> LineCropSample:
    return LineCropSample(
        sample_id=f"{index:064x}",
        lecture_id=lecture_id,
        page_event_id=f"{lecture_id}-event-{index}",
        page_number=1,
        stable_frame_timestamp=float(index),
        source_image_path=f"{lecture_id}-page-{index}.png",
        source_image_sha256=source_sha256 or f"{index + 100:064x}",
        crop_path=f"crop-{index}.png",
        crop_sha256=crop_sha256 or f"{index + 200:064x}",
        bounding_box=PixelCrop(x=0, y=0, width=40, height=20),
        text=text,
        normalized_text=text,
        label_source=label_source,
        detector_reader=PageReaderKind.rapidocr,
        detector_version="test",
        detector_preprocessing_version="test",
        detector_cache_key=f"{index + 300:064x}",
        source_block_order=0,
    )


def make_split() -> LectureSplitManifest:
    return LectureSplitManifest(
        dataset_sha256=DATASET_HASH,
        seed=42,
        train_lecture_ids=["lecture-train"],
        validation_lecture_ids=["lecture-validation"],
        test_lecture_ids=["lecture-test"],
    )


def test_formal_run_spec_requires_a_frozen_split():
    try:
        ExperimentRunSpec(
            experiment_id="reader-study",
            task=ExperimentTask.reader_benchmark,
            phase=ExperimentPhase.formal,
            model_variant="cnn_ctc",
            seed=42,
            deterministic_algorithms=True,
            dataset_sha256=DATASET_HASH,
            primary_metric="character_error_rate",
        )
    except ValueError as exc:
        assert "frozen split hash" in str(exc)
    else:
        raise AssertionError("A formal run without a split should be rejected.")


def test_reader_audit_fits_vocabulary_on_train_only():
    samples = [
        make_sample(1, lecture_id="lecture-train", text="abc"),
        make_sample(2, lecture_id="lecture-validation", text="abd"),
        make_sample(3, lecture_id="lecture-test", text="abe"),
    ]

    report = audit_reader_dataset(
        samples,
        make_split(),
        dataset_sha256=DATASET_HASH,
    )

    assert report.passed
    assert report.character_coverage["train"].unknown_character_rate == 0
    assert report.character_coverage["validation"].unknown_characters == ["d"]
    assert report.character_coverage["test"].unknown_characters == ["e"]
    assert len(report.warnings) == 2


def test_reader_audit_rejects_cross_split_image_leakage():
    duplicate_crop = "f" * 64
    samples = [
        make_sample(
            1,
            lecture_id="lecture-train",
            text="abc",
            crop_sha256=duplicate_crop,
        ),
        make_sample(2, lecture_id="lecture-validation", text="abc"),
        make_sample(
            3,
            lecture_id="lecture-test",
            text="abc",
            crop_sha256=duplicate_crop,
        ),
    ]

    report = audit_reader_dataset(
        samples,
        make_split(),
        dataset_sha256=DATASET_HASH,
    )

    assert not report.passed
    assert report.cross_split_crop_hashes == [duplicate_crop]
    assert any("Identical line crops" in problem for problem in report.problems)


def test_formal_reader_audit_rejects_ocr_selected_labels():
    samples = [
        make_sample(
            1,
            lecture_id="lecture-train",
            text="abc",
            label_source=LineLabelSource.manual_exact_match,
        ),
        make_sample(2, lecture_id="lecture-validation", text="abc"),
        make_sample(3, lecture_id="lecture-test", text="abc"),
    ]

    report = audit_reader_dataset(
        samples,
        make_split(),
        dataset_sha256=DATASET_HASH,
    )

    assert not report.passed
    assert any("OCR-selected" in problem for problem in report.problems)


def test_formal_reader_audit_accepts_source_aligned_evaluation_with_warning():
    samples = [
        make_sample(
            1,
            lecture_id="lecture-train",
            text="abc",
            label_source=LineLabelSource.synthetic_render,
        ),
        make_sample(
            2,
            lecture_id="lecture-validation",
            text="abc",
            label_source=LineLabelSource.source_aligned,
        ),
        make_sample(
            3,
            lecture_id="lecture-test",
            text="abc",
            label_source=LineLabelSource.source_aligned,
        ),
    ]

    report = audit_reader_dataset(
        samples,
        make_split(),
        dataset_sha256=DATASET_HASH,
    )

    assert report.passed
    assert len(report.warnings) == 2
    assert all("human spot-check" in warning for warning in report.warnings)


def test_dataset_cli_freezes_split_and_passing_audit(tmp_path: Path):
    samples = [
        make_sample(1, lecture_id="lecture-a", text="abc"),
        make_sample(2, lecture_id="lecture-b", text="abc"),
        make_sample(3, lecture_id="lecture-c", text="abc"),
    ]
    dataset_path = tmp_path / "samples.jsonl"
    split_path = tmp_path / "split.json"
    audit_path = tmp_path / "audit.json"
    write_jsonl(dataset_path, samples)

    exit_code = dataset_cli_main(
        [
            "audit",
            "--dataset",
            str(dataset_path),
            "--split-output",
            str(split_path),
            "--audit-output",
            str(audit_path),
            "--seed",
            "1",
        ]
    )

    split = LectureSplitManifest.model_validate_json(
        split_path.read_text(encoding="utf-8")
    )
    report = audit_reader_dataset(
        samples,
        split,
        dataset_sha256=sha256_file(dataset_path),
    )
    assert exit_code == 0
    assert split.dataset_sha256 == sha256_file(dataset_path)
    assert report.passed
    assert audit_path.is_file()
