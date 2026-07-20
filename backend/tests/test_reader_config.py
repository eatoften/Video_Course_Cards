from pathlib import Path

import pytest
from PIL import Image

from multimodal_lab.annotation_io import write_jsonl
from multimodal_lab.ctc_text import CharacterTokenizer
from multimodal_lab.experiment_protocol import audit_reader_dataset
from multimodal_lab.page_reading import sha256_file
from multimodal_lab.reader_config import (
    CnnCtcEncoderConfig,
    ReaderDataConfig,
    ReaderExperimentConfig,
    ReaderOptimizationConfig,
    load_reader_experiment_config,
    verify_reader_data_contract,
)
from multimodal_lab.schemas import (
    LectureSplitManifest,
    LineCropSample,
    LineLabelSource,
    PageReaderKind,
    PixelCrop,
)
from multimodal_lab.training.reader_data import (
    build_reader_test_data_bundle,
    build_reader_training_data_bundle,
)


CONFIG_PATH = (
    Path(__file__).parents[1]
    / "multimodal_lab"
    / "configs"
    / "reader_cnn_v1.json"
)


def test_checked_in_cnn_config_freezes_the_architecture_choices():
    config = load_reader_experiment_config(CONFIG_PATH)

    assert config.model.kind == "cnn_ctc"
    assert config.model.channels == [32, 64, 128]
    assert config.model.normalization == "channel_layer_norm"
    assert config.model.temporal_downsample == 4
    assert config.selection.evaluate_test_during_training is False


def test_reader_config_forbids_test_evaluation_during_training():
    payload = load_reader_experiment_config(CONFIG_PATH).model_dump(mode="json")
    payload["selection"]["evaluate_test_during_training"] = True

    with pytest.raises(ValueError, match="Input should be False"):
        ReaderExperimentConfig.model_validate(payload)


def test_checked_in_config_verifies_the_local_frozen_data_contract():
    config = load_reader_experiment_config(CONFIG_PATH)
    backend_root = Path(__file__).parents[1]
    if not (backend_root / config.data.dataset_path).is_file():
        pytest.skip("The frozen research dataset is local and ignored by Git.")

    verified = verify_reader_data_contract(config, project_root=backend_root)

    assert verified.audit.passed
    assert verified.split.train_lecture_ids == ["cs231n-2025-lecture-02"]
    assert verified.split.validation_lecture_ids == ["cs231n-2026-lecture-03"]
    assert verified.split.test_lecture_ids == ["cs231n-2025-lecture-04"]


def test_data_bundle_loads_only_after_a_three_lecture_audit(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    lecture_specs = (
        ("lecture-train", "ab", LineLabelSource.synthetic_render),
        ("lecture-validation", "ba", LineLabelSource.source_aligned),
        ("lecture-test", "aa", LineLabelSource.source_aligned),
    )
    samples: list[LineCropSample] = []
    for index, (lecture_id, text, label_source) in enumerate(
        lecture_specs,
        start=1,
    ):
        crop_path = image_root / f"crop-{index}.png"
        image = Image.new("L", (16 + index, 8), color=255)
        image.putpixel((index, 4), 0)
        image.save(crop_path)
        samples.append(
            LineCropSample(
                sample_id=f"{index:064x}",
                lecture_id=lecture_id,
                page_event_id=f"{lecture_id}-page",
                page_number=1,
                stable_frame_timestamp=float(index),
                source_image_path=f"source-{index}.png",
                source_image_sha256=f"{index + 100:064x}",
                crop_path=crop_path.name,
                crop_sha256=sha256_file(crop_path),
                bounding_box=PixelCrop(
                    x=0,
                    y=0,
                    width=image.width,
                    height=image.height,
                ),
                text=text,
                normalized_text=text,
                label_source=label_source,
                detector_reader=PageReaderKind.rapidocr,
                detector_version="test",
                detector_preprocessing_version="test",
                detector_cache_key=f"{index + 200:064x}",
                source_block_order=0,
            )
        )

    dataset_path = tmp_path / "samples.jsonl"
    write_jsonl(dataset_path, samples)
    dataset_sha256 = sha256_file(dataset_path)
    split = LectureSplitManifest(
        dataset_sha256=dataset_sha256,
        seed=7,
        train_lecture_ids=["lecture-train"],
        validation_lecture_ids=["lecture-validation"],
        test_lecture_ids=["lecture-test"],
    )
    split_path = tmp_path / "split.json"
    split_path.write_text(
        split.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    audit = audit_reader_dataset(
        samples,
        split,
        dataset_sha256=dataset_sha256,
    )
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        audit.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    vocabulary = CharacterTokenizer.fit(["ab"])
    config = ReaderExperimentConfig(
        experiment_id="temporary-data-contract-test",
        seed=7,
        model=CnnCtcEncoderConfig(
            channels=[8, 16],
            output_features=16,
            dropout=0,
        ),
        data=ReaderDataConfig(
            dataset_path=dataset_path.name,
            dataset_sha256=dataset_sha256,
            split_path=split_path.name,
            split_sha256=sha256_file(split_path),
            audit_path=audit_path.name,
            audit_sha256=sha256_file(audit_path),
            expected_vocabulary_sha256=vocabulary.spec.sha256,
            image_root=image_root.name,
            target_height=8,
            max_image_width=32,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        ),
        optimization=ReaderOptimizationConfig(
            epochs=1,
            learning_rate=0.001,
            weight_decay=0,
            gradient_clip_norm=1,
            early_stopping_patience=1,
            mixed_precision=False,
        ),
    )

    bundle = build_reader_training_data_bundle(config, project_root=tmp_path)

    assert bundle.contract.audit.passed
    assert bundle.tokenizer.spec.sha256 == vocabulary.spec.sha256
    assert bundle.sample_counts == {
        "train": 1,
        "validation": 1,
    }
    assert not hasattr(bundle, "test_loader")
    validation_batch = next(iter(bundle.validation_loader))
    assert validation_batch.texts == ("ba",)
    assert validation_batch.images.shape[0:3] == (1, 1, 8)

    test_bundle = build_reader_test_data_bundle(config, project_root=tmp_path)
    assert test_bundle.sample_count == 1
    assert next(iter(test_bundle.test_loader)).texts == ("aa",)


def test_cnn_config_rejects_non_power_of_two_downsampling():
    payload = load_reader_experiment_config(CONFIG_PATH).model_dump(mode="json")
    payload["model"]["temporal_downsample"] = 3

    with pytest.raises(ValueError, match="power of two"):
        ReaderExperimentConfig.model_validate(payload)
