import pytest
import torch

from multimodal_lab.ctc_text import CharacterTokenizer, ctc_loss, pack_ctc_targets
from multimodal_lab.models.cnn_ctc import CnnCtcReader
from multimodal_lab.reader_config import CnnCtcEncoderConfig
from multimodal_lab.training.reader_protocol import forward_reader_model


def make_config(*, dropout: float = 0.0) -> CnnCtcEncoderConfig:
    return CnnCtcEncoderConfig(
        channels=[8, 16, 24],
        kernel_size=3,
        temporal_downsample=4,
        output_features=20,
        dropout=dropout,
        blank_logit_bias=-1.0,
    )


def test_cnn_reader_matches_the_shared_shape_and_length_contract():
    model = CnnCtcReader(make_config(), vocabulary_size=11)
    images = torch.rand(2, 1, 32, 128)
    widths = torch.tensor([128, 100])

    output = forward_reader_model(
        model,
        images,
        widths,
        vocabulary_size=11,
    )

    assert output.logits.shape == (2, 32, 11)
    assert output.input_lengths.tolist() == [32, 25]


def test_extra_right_padding_does_not_change_valid_logits():
    torch.manual_seed(7)
    model = CnnCtcReader(make_config(), vocabulary_size=9).eval()
    original = torch.rand(1, 1, 32, 100)
    padded = torch.zeros(1, 1, 32, 128)
    padded[:, :, :, :100] = original
    widths = torch.tensor([100])

    with torch.inference_mode():
        original_output = model(original, widths)
        padded_output = model(padded, widths)

    valid_steps = original_output.input_lengths.item()
    torch.testing.assert_close(
        original_output.logits[:, :valid_steps],
        padded_output.logits[:, :valid_steps],
        rtol=0,
        atol=1e-6,
    )


def test_cnn_reader_produces_finite_ctc_gradients():
    torch.manual_seed(11)
    tokenizer = CharacterTokenizer.fit(["ab", "ba"])
    model = CnnCtcReader(
        make_config(),
        vocabulary_size=tokenizer.vocabulary_size,
        blank_id=tokenizer.blank_id,
    )
    images = torch.rand(2, 1, 32, 64)
    widths = torch.tensor([64, 56])
    output = model(images, widths)
    targets = pack_ctc_targets(("ab", "ba"), tokenizer)

    loss = ctc_loss(
        output.logits,
        output.input_lengths,
        targets,
        blank_id=tokenizer.blank_id,
    )
    loss.backward()

    assert torch.isfinite(loss)
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_cnn_reader_rejects_widths_beyond_the_padded_tensor():
    model = CnnCtcReader(make_config(), vocabulary_size=8)

    with pytest.raises(ValueError, match="exceeds the padded"):
        model(
            torch.rand(1, 1, 32, 64),
            torch.tensor([65]),
        )


def test_ctc_head_starts_with_the_configured_blank_bias():
    model = CnnCtcReader(make_config(), vocabulary_size=8, blank_id=0)

    assert model.head.classifier.bias[0].item() == pytest.approx(-1.0)
    assert torch.count_nonzero(model.head.classifier.bias[1:]) == 0
