from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw
from pydantic import ValidationError

from multimodal_lab.line_crop_dataset import (
    LineCropDataset,
    LineCropDatasetError,
    LineImageAugmentation,
    LineImageTransform,
    build_exact_match_line_crops,
    collate_line_crops,
    make_lecture_level_split,
    make_explicit_lecture_split,
    partition_by_lecture_split,
)
from multimodal_lab.metrics import normalize_ocr_text
from multimodal_lab.page_reading import sha256_file
from multimodal_lab.annotation_io import write_jsonl
from multimodal_lab.run_lecture_split import main as split_main
from multimodal_lab.schemas import (
    DatasetSplit,
    GoldTextScope,
    LectureSplitManifest,
    LineCropSample,
    LineLabelSource,
    PageContent,
    PageContentBlock,
    PageReaderKind,
    PixelCrop,
    StablePageReference,
)


def make_page_fixture(tmp_path: Path):
    image_path = tmp_path / "page.png"
    image = Image.new("RGB", (160, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 70, 30), fill="black")
    draw.rectangle((10, 40, 120, 65), fill="black")
    image.save(image_path)
    image_hash = sha256_file(image_path)
    reference = StablePageReference(
        page_event_id="event-1",
        lecture_id="lecture-1",
        stable_frame_timestamp=12.0,
        image_path=str(image_path),
        page_number=3,
        gold_text="Alpha\nBeta",
        gold_text_scope=GoldTextScope.verbatim_content,
    )
    blocks = [
        PageContentBlock(
            order=0,
            text="Alpha",
            polygon=[(10, 10), (70, 10), (70, 30), (10, 30)],
        ),
        PageContentBlock(
            order=1,
            text="Wrong",
            polygon=[(80, 10), (140, 10), (140, 30), (80, 30)],
        ),
        PageContentBlock(
            order=2,
            text="Beta",
            polygon=[(10, 40), (120, 40), (120, 65), (10, 65)],
        ),
    ]
    raw_text = "\n".join(block.text for block in blocks)
    content = PageContent(
        page_event_id=reference.page_event_id,
        lecture_id=reference.lecture_id,
        page_number=reference.page_number,
        stable_frame_timestamp=reference.stable_frame_timestamp,
        image_path=str(image_path),
        image_sha256=image_hash,
        reader=PageReaderKind.rapidocr,
        reader_version="rapidocr-test",
        preprocessing_version="test",
        cache_key="a" * 64,
        raw_text=raw_text,
        normalized_text=normalize_ocr_text(raw_text, case_sensitive=True),
        ordered_blocks=blocks,
        latency_seconds=0,
    )
    return reference, content


def make_sample(lecture_id: str, index: int) -> LineCropSample:
    return LineCropSample(
        sample_id=f"{index:064x}",
        lecture_id=lecture_id,
        page_event_id=f"event-{index}",
        page_number=1,
        stable_frame_timestamp=float(index),
        source_image_path="page.png",
        source_image_sha256="a" * 64,
        crop_path=f"crop-{index}.png",
        crop_sha256="b" * 64,
        bounding_box=PixelCrop(x=0, y=0, width=10, height=10),
        text=f"line {index}",
        normalized_text=f"line {index}",
        label_source=LineLabelSource.manual_exact_match,
        detector_reader=PageReaderKind.rapidocr,
        detector_version="test",
        detector_preprocessing_version="test",
        detector_cache_key="c" * 64,
        source_block_order=0,
    )


def test_exact_match_builder_crops_only_manually_grounded_lines(tmp_path):
    reference, content = make_page_fixture(tmp_path)

    samples = build_exact_match_line_crops(
        [reference],
        [content],
        image_root=tmp_path,
        output_dir=tmp_path / "crops",
        padding_pixels=2,
    )

    assert [sample.text for sample in samples] == ["Alpha", "Beta"]
    assert samples[0].bounding_box == PixelCrop(
        x=8,
        y=8,
        width=64,
        height=24,
    )
    assert samples[0].label_source is LineLabelSource.manual_exact_match
    assert (tmp_path / samples[0].crop_path).is_file()


def test_line_crop_dataset_preserves_aspect_ratio_and_pads_right(tmp_path):
    reference, content = make_page_fixture(tmp_path)
    samples = build_exact_match_line_crops(
        [reference],
        [content],
        image_root=tmp_path,
        output_dir=tmp_path / "crops",
    )
    dataset = LineCropDataset(
        samples,
        image_root=tmp_path,
        transform=LineImageTransform(target_height=16, max_width=128),
    )

    batch = collate_line_crops([dataset[0], dataset[1]], width_multiple=4)

    assert batch.images.shape[0:3] == (2, 1, 16)
    assert batch.images.shape[-1] % 4 == 0
    assert batch.widths[0] != batch.widths[1]
    assert torch.count_nonzero(batch.images[0]) > 0
    assert torch.count_nonzero(
        batch.images[0, :, :, batch.widths[0] :]
    ) == 0


def test_line_augmentation_is_deterministic_by_sample_and_epoch(tmp_path):
    reference, content = make_page_fixture(tmp_path)
    samples = build_exact_match_line_crops(
        [reference],
        [content],
        image_root=tmp_path,
        output_dir=tmp_path / "crops",
    )
    policy = LineImageAugmentation(
        enabled=True,
        policy_id="test-policy",
        seed=17,
        rotation_probability=1,
        maximum_rotation_degrees=1,
        contrast_probability=1,
        minimum_contrast=0.8,
        maximum_contrast=1.2,
        brightness_probability=1,
        minimum_brightness=0.9,
        maximum_brightness=1.1,
        blur_probability=1,
        maximum_blur_radius=0.5,
        noise_probability=1,
        maximum_noise_standard_deviation=0.02,
    )
    transform = LineImageTransform(
        target_height=16,
        max_width=128,
        augmentation=policy,
    )
    first_dataset = LineCropDataset(
        samples,
        image_root=tmp_path,
        transform=transform,
    )
    second_dataset = LineCropDataset(
        samples,
        image_root=tmp_path,
        transform=transform,
    )

    first_dataset.set_epoch(3)
    second_dataset.set_epoch(3)
    first = first_dataset[0].image
    repeated = first_dataset[0].image
    independent = second_dataset[0].image
    first_dataset.set_epoch(4)
    next_epoch = first_dataset[0].image

    assert torch.equal(first, repeated)
    assert torch.equal(first, independent)
    assert not torch.equal(first, next_epoch)


def test_lecture_level_split_is_deterministic_and_has_no_leakage():
    samples = [
        make_sample(f"lecture-{lecture}", lecture * 10 + item)
        for lecture in range(5)
        for item in range(2)
    ]

    first = make_lecture_level_split(
        samples,
        dataset_sha256="c" * 64,
        seed=7,
    )
    second = make_lecture_level_split(
        samples,
        dataset_sha256="c" * 64,
        seed=7,
    )
    partitions = partition_by_lecture_split(samples, first)

    assert first == second
    assert partitions[DatasetSplit.train]
    assert partitions[DatasetSplit.validation]
    assert partitions[DatasetSplit.test]
    lecture_sets = [
        {sample.lecture_id for sample in partition}
        for partition in partitions.values()
    ]
    assert lecture_sets[0].isdisjoint(lecture_sets[1])
    assert lecture_sets[0].isdisjoint(lecture_sets[2])
    assert lecture_sets[1].isdisjoint(lecture_sets[2])


def test_formal_split_rejects_a_single_lecture():
    samples = [make_sample("lecture-1", index) for index in range(3)]

    with pytest.raises(LineCropDatasetError, match="at least three lectures"):
        make_lecture_level_split(samples, dataset_sha256="c" * 64)


def test_explicit_split_freezes_declared_lecture_roles():
    samples = [
        make_sample(lecture_id, index)
        for index, lecture_id in enumerate(("train", "validation", "test"))
    ]

    split = make_explicit_lecture_split(
        samples,
        dataset_sha256="c" * 64,
        seed=17,
        train_lecture_ids=["train"],
        validation_lecture_ids=["validation"],
        test_lecture_ids=["test"],
    )

    assert split.train_lecture_ids == ["train"]
    assert split.validation_lecture_ids == ["validation"]
    assert split.test_lecture_ids == ["test"]


def test_split_schema_rejects_lecture_overlap():
    with pytest.raises(ValidationError, match="must be disjoint"):
        LectureSplitManifest(
            dataset_sha256="c" * 64,
            seed=1,
            train_lecture_ids=["lecture-1"],
            validation_lecture_ids=["lecture-1"],
            test_lecture_ids=["lecture-2"],
        )


def test_lecture_split_cli_writes_a_reusable_manifest(tmp_path):
    samples = [
        make_sample(f"lecture-{lecture}", lecture * 10 + item)
        for lecture in range(4)
        for item in range(2)
    ]
    samples_path = tmp_path / "samples.jsonl"
    output_path = tmp_path / "split.json"
    write_jsonl(samples_path, samples)

    exit_code = split_main(
        [
            "--samples",
            str(samples_path),
            "--output",
            str(output_path),
            "--seed",
            "3",
        ]
    )
    split = LectureSplitManifest.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )

    assert exit_code == 0
    assert split.dataset_sha256 == sha256_file(samples_path)
