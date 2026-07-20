from __future__ import annotations

from torch import nn

from ..reader_config import (
    CnnCtcEncoderConfig,
    ReaderEncoderConfig,
    VitCtcEncoderConfig,
)
from .cnn_ctc import CnnCtcReader
from .vit_ctc import VitCtcReader


def build_reader_model(
    config: ReaderEncoderConfig,
    vocabulary_size: int,
    *,
    blank_id: int = 0,
) -> nn.Module:
    if isinstance(config, CnnCtcEncoderConfig):
        return CnnCtcReader(
            config,
            vocabulary_size,
            blank_id=blank_id,
        )
    if isinstance(config, VitCtcEncoderConfig):
        return VitCtcReader(
            config,
            vocabulary_size,
            blank_id=blank_id,
        )
    raise TypeError(f"Unsupported reader model config: {type(config).__name__}")
