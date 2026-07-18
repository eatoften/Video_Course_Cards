import torch

from multimodal_lab.ctc_overfit import (
    CtcOverfitConfig,
    TinyCtcLineReader,
    select_overfit_samples,
)
from multimodal_lab.ctc_text import CharacterTokenizer, ctc_loss, pack_ctc_targets
from multimodal_lab.schemas import (
    LineCropSample,
    LineLabelSource,
    PageReaderKind,
    PixelCrop,
)


def make_sample(index: int, text: str) -> LineCropSample:
    return LineCropSample(
        sample_id=f"{index:064x}",
        lecture_id="lecture-1",
        page_event_id=f"event-{index}",
        page_number=1,
        stable_frame_timestamp=float(index),
        source_image_path="page.png",
        source_image_sha256="a" * 64,
        crop_path=f"crop-{index}.png",
        crop_sha256="b" * 64,
        bounding_box=PixelCrop(x=0, y=0, width=40, height=20),
        text=text,
        normalized_text=text,
        label_source=LineLabelSource.manual_exact_match,
        detector_reader=PageReaderKind.rapidocr,
        detector_version="test",
        detector_preprocessing_version="test",
        detector_cache_key="c" * 64,
        source_block_order=index,
    )


def test_tiny_ctc_probe_emits_batch_time_vocabulary_logits():
    model = TinyCtcLineReader(vocabulary_size=12)
    images = torch.randn(2, 1, 24, 80)

    logits = model(images)

    assert logits.shape == (2, 20, 12)
    assert model.output_lengths(torch.tensor([80, 79])).tolist() == [20, 19]


def test_one_ctc_training_step_changes_probe_parameters():
    tokenizer = CharacterTokenizer.fit(["cat", "dog"])
    model = TinyCtcLineReader(tokenizer.vocabulary_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    images = torch.randn(2, 1, 24, 96)
    widths = torch.tensor([96, 96])
    targets = pack_ctc_targets(["cat", "dog"], tokenizer)
    before = model.classifier.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = ctc_loss(logits, model.output_lengths(widths), targets)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert not torch.equal(before, model.classifier.weight)


def test_overfit_subset_is_seeded_and_respects_label_length():
    samples = [
        make_sample(index, f"line {index}")
        for index in range(40)
    ]
    config = CtcOverfitConfig(
        sample_count=32,
        max_label_length=7,
        seed=9,
    )

    first = select_overfit_samples(samples, config)
    second = select_overfit_samples(samples, config)

    assert [sample.sample_id for sample in first] == [
        sample.sample_id for sample in second
    ]
    assert len(first) == 32
    assert all(len(sample.normalized_text) <= 7 for sample in first)
