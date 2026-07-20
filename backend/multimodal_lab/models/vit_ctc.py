from __future__ import annotations

import torch
from torch import Tensor, nn

from ..reader_config import VitCtcEncoderConfig
from ..training.reader_protocol import ReaderModelOutput
from .reader_layers import CtcProjectionHead


class StripPatchEmbedding(nn.Module):
    """Turn a fixed-height line image into horizontal visual tokens."""

    def __init__(self, config: VitCtcEncoderConfig) -> None:
        super().__init__()
        self.patch_height = config.patch_height
        self.patch_width = config.patch_width
        self.projection = nn.Conv2d(
            config.input_channels,
            config.embedding_dim,
            kernel_size=(config.patch_height, config.patch_width),
            stride=(config.patch_height, config.patch_width),
        )

    def forward(self, images: Tensor) -> Tensor:
        patches = self.projection(images)
        if patches.shape[-2] != 1:
            raise ValueError("Strip patch embedding must produce one token row.")
        return patches.squeeze(2).transpose(1, 2).contiguous()


class MultiHeadSelfAttention(nn.Module):
    """Explicit scaled dot-product self-attention with a padding mask."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        *,
        attention_dropout: float,
        output_dropout: float,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, sequence: Tensor, valid_tokens: Tensor) -> Tensor:
        if sequence.ndim != 3:
            raise ValueError("Attention expects [batch, time, embedding].")
        batch_size, time_steps, embedding_dim = sequence.shape
        if embedding_dim != self.embedding_dim:
            raise ValueError("Attention received the wrong embedding dimension.")
        if valid_tokens.shape != (batch_size, time_steps):
            raise ValueError("Attention padding mask has the wrong shape.")

        qkv = self.qkv(sequence).reshape(
            batch_size,
            time_steps,
            3,
            self.num_heads,
            self.head_dim,
        )
        queries, keys, values = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(
            ~valid_tokens[:, None, None, :],
            torch.finfo(scores.dtype).min,
        )
        weights = self.attention_dropout(scores.softmax(dim=-1))
        attended = torch.matmul(weights, values)
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            time_steps,
            embedding_dim,
        )
        projected = self.output_dropout(self.output_projection(attended))
        return projected.masked_fill(~valid_tokens.unsqueeze(-1), 0)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, config: VitCtcEncoderConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = MultiHeadSelfAttention(
            config.embedding_dim,
            config.num_heads,
            attention_dropout=config.attention_dropout,
            output_dropout=config.dropout,
        )
        self.mlp_norm = nn.LayerNorm(config.embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embedding_dim, config.mlp_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, sequence: Tensor, valid_tokens: Tensor) -> Tensor:
        sequence = sequence + self.attention(
            self.attention_norm(sequence),
            valid_tokens,
        )
        sequence = sequence + self.mlp(self.mlp_norm(sequence))
        return sequence.masked_fill(~valid_tokens.unsqueeze(-1), 0)


class VitCtcReader(nn.Module):
    """A small strip-patch Vision Transformer with a shared CTC head."""

    def __init__(
        self,
        config: VitCtcEncoderConfig,
        vocabulary_size: int,
        *,
        blank_id: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocabulary_size = vocabulary_size
        self.patch_embedding = StripPatchEmbedding(config)
        self.position_embedding = nn.Parameter(
            torch.empty(
                1,
                config.maximum_position_tokens,
                config.embedding_dim,
            )
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.encoder = nn.ModuleList(
            TransformerEncoderBlock(config) for _ in range(config.depth)
        )
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.head = CtcProjectionHead(
            input_features=config.embedding_dim,
            output_features=config.output_features,
            vocabulary_size=vocabulary_size,
            dropout=config.dropout,
            blank_id=blank_id,
            blank_logit_bias=config.blank_logit_bias,
        )

    def output_lengths(self, widths: Tensor) -> Tensor:
        if widths.ndim != 1 or widths.is_floating_point():
            raise ValueError("widths must be a one-dimensional integer tensor.")
        if (widths < self.config.patch_width).any():
            raise ValueError("Every line must contain at least one full patch.")
        return widths.div(self.config.patch_width, rounding_mode="floor")

    def forward(self, images: Tensor, widths: Tensor) -> ReaderModelOutput:
        self._validate_inputs(images, widths)
        sequence = self.patch_embedding(images)
        input_lengths = self.output_lengths(widths).to(device=images.device)
        time_steps = sequence.shape[1]
        if time_steps > self.config.maximum_position_tokens:
            raise ValueError("Input exceeds the learned position embedding table.")
        positions = self.position_embedding[:, :time_steps]
        valid_tokens = (
            torch.arange(time_steps, device=images.device).unsqueeze(0)
            < input_lengths.unsqueeze(1)
        )
        sequence = self.embedding_dropout(sequence + positions)
        sequence = sequence.masked_fill(~valid_tokens.unsqueeze(-1), 0)
        for block in self.encoder:
            sequence = block(sequence, valid_tokens)
        sequence = self.final_norm(sequence)
        sequence = sequence.masked_fill(~valid_tokens.unsqueeze(-1), 0)
        logits = self.head(sequence)
        return ReaderModelOutput(logits=logits, input_lengths=input_lengths)

    def _validate_inputs(self, images: Tensor, widths: Tensor) -> None:
        expected_shape = (images.shape[0],)
        if images.ndim != 4 or images.shape[1] != self.config.input_channels:
            raise ValueError(
                "ViT line images must have shape [batch, 1, height, width]."
            )
        if images.shape[-2] != self.config.patch_height:
            raise ValueError("Line-image height must equal ViT patch_height.")
        if widths.shape != expected_shape:
            raise ValueError("widths must contain one value per line image.")
        if widths.device != images.device:
            raise ValueError("images and widths must be on the same device.")
        if (widths > images.shape[-1]).any():
            raise ValueError("A valid line width exceeds the padded image width.")
        maximum_width = (
            self.config.maximum_position_tokens * self.config.patch_width
        )
        if images.shape[-1] > maximum_width:
            raise ValueError("Padded image width exceeds the ViT position table.")
