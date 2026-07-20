from __future__ import annotations

import torch
from torch import Tensor, nn


class ChannelLayerNorm2d(nn.Module):
    """Apply LayerNorm over channels independently at each spatial position."""

    def __init__(self, channels: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        self.channels = channels
        self.normalization = nn.LayerNorm(channels, eps=eps)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError(
                "ChannelLayerNorm2d expects [batch, channels, height, width]."
            )
        channels_last = features.permute(0, 2, 3, 1)
        normalized = self.normalization(channels_last)
        return normalized.permute(0, 3, 1, 2).contiguous()


class CtcProjectionHead(nn.Module):
    """Shared per-time-step projection for CNN and future ViT encoders."""

    def __init__(
        self,
        input_features: int,
        output_features: int,
        vocabulary_size: int,
        *,
        dropout: float,
        blank_id: int,
        blank_logit_bias: float,
    ) -> None:
        super().__init__()
        if input_features <= 0 or output_features <= 0:
            raise ValueError("CTC head feature dimensions must be positive.")
        if vocabulary_size <= 2:
            raise ValueError("The CTC vocabulary needs ordinary characters.")
        if not 0 <= blank_id < vocabulary_size:
            raise ValueError("blank_id is outside the CTC vocabulary.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        self.input_features = input_features
        self.projection = nn.Linear(input_features, output_features)
        self.normalization = nn.LayerNorm(output_features)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_features, vocabulary_size)
        with torch.no_grad():
            self.classifier.bias.zero_()
            self.classifier.bias[blank_id] = blank_logit_bias

    def forward(self, sequence: Tensor) -> Tensor:
        if sequence.ndim != 3 or sequence.shape[-1] != self.input_features:
            raise ValueError(
                "The CTC head expects [batch, time, input_features]."
            )
        projected = self.projection(sequence)
        normalized = self.normalization(projected)
        activated = self.activation(normalized)
        return self.classifier(self.dropout(activated))
