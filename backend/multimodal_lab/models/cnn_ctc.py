from __future__ import annotations

import torch
from torch import Tensor, nn

from ..reader_config import CnnCtcEncoderConfig
from ..training.reader_protocol import ReaderModelOutput
from .reader_layers import ChannelLayerNorm2d, CtcProjectionHead


class CnnFeatureBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int,
        downsample: bool,
    ) -> None:
        super().__init__()
        self.downsample = downsample
        self.convolution = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.normalization = ChannelLayerNorm2d(output_channels)
        self.activation = nn.GELU()
        self.pooling = (
            nn.MaxPool2d(kernel_size=2, stride=2)
            if downsample
            else nn.Identity()
        )

    def forward(self, features: Tensor) -> Tensor:
        convolved = self.convolution(features)
        normalized = self.normalization(convolved)
        activated = self.activation(normalized)
        return self.pooling(activated)


class CnnCtcReader(nn.Module):
    """Convert a variable-width line image into horizontal CTC logits."""

    def __init__(
        self,
        config: CnnCtcEncoderConfig,
        vocabulary_size: int,
        *,
        blank_id: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocabulary_size = vocabulary_size
        self.pooling_stages = config.temporal_downsample.bit_length() - 1

        blocks: list[nn.Module] = []
        input_channels = config.input_channels
        for block_index, output_channels in enumerate(config.channels):
            blocks.append(
                CnnFeatureBlock(
                    input_channels,
                    output_channels,
                    kernel_size=config.kernel_size,
                    downsample=block_index < self.pooling_stages,
                )
            )
            input_channels = output_channels
        self.encoder = nn.ModuleList(blocks)
        self.head = CtcProjectionHead(
            input_features=config.channels[-1],
            output_features=config.output_features,
            vocabulary_size=vocabulary_size,
            dropout=config.dropout,
            blank_id=blank_id,
            blank_logit_bias=config.blank_logit_bias,
        )

    def output_lengths(self, widths: Tensor) -> Tensor:
        if widths.ndim != 1:
            raise ValueError("widths must contain one value per line image.")
        if widths.is_floating_point():
            raise ValueError("widths must use an integer tensor dtype.")
        if (widths <= 0).any():
            raise ValueError("Line-image widths must be positive.")
        lengths = widths.to(dtype=widths.dtype)
        for _ in range(self.pooling_stages):
            lengths = lengths.div(2, rounding_mode="floor")
        if (lengths <= 0).any():
            raise ValueError("Pooling removed the complete horizontal sequence.")
        return lengths

    def forward(self, images: Tensor, widths: Tensor) -> ReaderModelOutput:
        self._validate_inputs(images, widths)
        feature_map = _mask_right_padding(images, widths)
        feature_widths = widths
        for block in self.encoder:
            feature_map = block(feature_map)
            if block.downsample:
                feature_widths = feature_widths.div(2, rounding_mode="floor")
            feature_map = _mask_right_padding(feature_map, feature_widths)
        sequence = feature_map.mean(dim=2).transpose(1, 2).contiguous()
        logits = self.head(sequence)
        input_lengths = feature_widths.to(device=logits.device)
        if (input_lengths > logits.shape[1]).any():
            raise ValueError("A valid line width exceeds the CNN time axis.")
        return ReaderModelOutput(
            logits=logits,
            input_lengths=input_lengths,
        )

    def _validate_inputs(self, images: Tensor, widths: Tensor) -> None:
        if images.ndim != 4 or images.shape[1] != self.config.input_channels:
            raise ValueError(
                "CNN line images must have shape [batch, 1, height, width]."
            )
        if widths.shape != (images.shape[0],):
            raise ValueError("widths must contain one value per line image.")
        if widths.device != images.device:
            raise ValueError("images and widths must be on the same device.")
        if (widths > images.shape[-1]).any():
            raise ValueError("A valid line width exceeds the padded image width.")
        minimum_size = 2**self.pooling_stages
        if images.shape[-2] < minimum_size or images.shape[-1] < minimum_size:
            raise ValueError("The line image is too small for configured pooling.")


def _mask_right_padding(features: Tensor, widths: Tensor) -> Tensor:
    positions = torch.arange(features.shape[-1], device=features.device)
    valid_columns = positions.unsqueeze(0) < widths.unsqueeze(1)
    return features.masked_fill(~valid_columns[:, None, None, :], 0)
