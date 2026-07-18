from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


class ReaderModelContractError(ValueError):
    pass


@dataclass(frozen=True)
class ReaderModelOutput:
    logits: Tensor
    input_lengths: Tensor


def forward_reader_model(
    model: nn.Module,
    images: Tensor,
    widths: Tensor,
    *,
    vocabulary_size: int,
) -> ReaderModelOutput:
    output = model(images, widths)
    if not isinstance(output, ReaderModelOutput):
        raise ReaderModelContractError(
            "A reader model must return ReaderModelOutput."
        )
    if output.logits.ndim != 3:
        raise ReaderModelContractError(
            "Reader logits must have shape [batch, time, vocabulary]."
        )
    batch_size, max_time, output_vocabulary = output.logits.shape
    if batch_size != images.shape[0]:
        raise ReaderModelContractError("Reader output batch size changed.")
    if output_vocabulary != vocabulary_size:
        raise ReaderModelContractError(
            "Reader output vocabulary does not match the tokenizer."
        )
    if output.input_lengths.shape != (batch_size,):
        raise ReaderModelContractError(
            "Reader input_lengths must contain one value per sample."
        )
    lengths = output.input_lengths.detach().to(device="cpu")
    if (lengths <= 0).any() or (lengths > max_time).any():
        raise ReaderModelContractError(
            "Reader input_lengths must lie within the time axis."
        )
    return output
