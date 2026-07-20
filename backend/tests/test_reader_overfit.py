from multimodal_lab.reader_config import CnnCtcEncoderConfig
from multimodal_lab.schemas import (
    LineCropSample,
    LineLabelSource,
    PageReaderKind,
    PixelCrop,
)
from multimodal_lab.training.reader_overfit import (
    ReaderOverfitConfig,
    select_unique_train_samples,
)


def make_sample(index: int, *, lecture_id: str, text: str) -> LineCropSample:
    return LineCropSample(
        sample_id=f"{index:064x}",
        lecture_id=lecture_id,
        page_event_id=f"page-{index}",
        page_number=1,
        stable_frame_timestamp=float(index),
        source_image_path=f"crop-{index}.png",
        source_image_sha256=f"{index + 100:064x}",
        crop_path=f"crop-{index}.png",
        crop_sha256=f"{index + 200:064x}",
        bounding_box=PixelCrop(x=0, y=0, width=40, height=20),
        text=text,
        normalized_text=text,
        label_source=LineLabelSource.synthetic_render,
        detector_reader=PageReaderKind.gold_reference,
        detector_version="test",
        detector_preprocessing_version="test",
        detector_cache_key=f"{index + 300:064x}",
        source_block_order=index,
    )


def test_overfit_selection_uses_unique_labels_from_train_only():
    samples = [
        make_sample(1, lecture_id="train", text="alpha"),
        make_sample(2, lecture_id="train", text="alpha"),
        make_sample(3, lecture_id="train", text="beta"),
        make_sample(4, lecture_id="validation", text="gamma"),
    ]
    config = ReaderOverfitConfig(sample_count=2, max_label_length=10)

    selected = select_unique_train_samples(
        samples,
        train_lecture_ids=["train"],
        config=config,
    )

    assert {sample.normalized_text for sample in selected} == {"alpha", "beta"}
    assert {sample.lecture_id for sample in selected} == {"train"}


def test_small_cnn_config_is_constructible_for_overfit_tests():
    config = CnnCtcEncoderConfig(
        channels=[8, 16],
        temporal_downsample=2,
        output_features=16,
        dropout=0,
    )

    assert config.temporal_downsample == 2
