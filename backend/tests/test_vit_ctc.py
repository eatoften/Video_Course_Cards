from pathlib import Path

import pytest
import torch

from multimodal_lab.ctc_text import CharacterTokenizer, ctc_loss, pack_ctc_targets
from multimodal_lab.models import CnnCtcReader, VitCtcReader, build_reader_model
from multimodal_lab.reader_config import (
    VitCtcEncoderConfig,
    load_reader_experiment_config,
)
from multimodal_lab.training.reader_protocol import forward_reader_model


VIT_CONFIG_PATH = (
    Path(__file__).parents[1]
    / "multimodal_lab"
    / "configs"
    / "reader_vit_v1.json"
)
CNN_CONFIG_PATH = (
    Path(__file__).parents[1]
    / "multimodal_lab"
    / "configs"
    / "reader_cnn_v2.json"
)


def make_config(*, dropout: float = 0.0) -> VitCtcEncoderConfig:
    return VitCtcEncoderConfig(
        patch_height=32,
        patch_width=4,
        temporal_downsample=4,
        embedding_dim=32,
        depth=2,
        num_heads=4,
        mlp_dim=64,
        maximum_position_tokens=64,
        output_features=32,
        dropout=dropout,
        attention_dropout=dropout,
        blank_logit_bias=-1,
    )


def test_vit_reader_matches_shared_shape_and_length_contract() -> None:
    model = VitCtcReader(make_config(), vocabulary_size=11)
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


def test_vit_padding_mask_keeps_valid_logits_invariant() -> None:
    torch.manual_seed(7)
    model = VitCtcReader(make_config(), vocabulary_size=9).eval()
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


def test_vit_reader_produces_finite_ctc_gradients() -> None:
    torch.manual_seed(11)
    tokenizer = CharacterTokenizer.fit(["ab", "ba"])
    model = VitCtcReader(
        make_config(),
        vocabulary_size=tokenizer.vocabulary_size,
        blank_id=tokenizer.blank_id,
    )
    images = torch.rand(2, 1, 32, 64)
    widths = torch.tensor([64, 56])
    targets = pack_ctc_targets(("ab", "ba"), tokenizer)
    output = model(images, widths)

    loss = ctc_loss(
        output.logits,
        output.input_lengths,
        targets,
        blank_id=tokenizer.blank_id,
    )
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert torch.isfinite(loss)
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_vit_rejects_incompatible_patch_and_position_contracts() -> None:
    with pytest.raises(ValueError, match="must equal horizontal patch width"):
        VitCtcEncoderConfig(
            **{
                **make_config().model_dump(),
                "temporal_downsample": 2,
            }
        )


def test_reader_factory_builds_the_discriminated_model() -> None:
    config = make_config()

    model = build_reader_model(config, vocabulary_size=8)

    assert isinstance(model, VitCtcReader)


def test_formal_vit_is_within_ten_percent_of_cnn_parameter_count() -> None:
    vit_config = load_reader_experiment_config(VIT_CONFIG_PATH)
    cnn_config = load_reader_experiment_config(CNN_CONFIG_PATH)
    vit = build_reader_model(vit_config.model, vocabulary_size=85)
    cnn = build_reader_model(cnn_config.model, vocabulary_size=85)
    vit_count = sum(parameter.numel() for parameter in vit.parameters())
    cnn_count = sum(parameter.numel() for parameter in cnn.parameters())

    assert isinstance(vit, VitCtcReader)
    assert isinstance(cnn, CnnCtcReader)
    assert abs(vit_count - cnn_count) / cnn_count <= 0.1
