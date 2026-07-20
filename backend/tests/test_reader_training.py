from pathlib import Path

import pytest
import torch
from torch import nn

from multimodal_lab.ctc_text import CharacterTokenizer
from multimodal_lab.line_crop_dataset import LineCropBatch
from multimodal_lab.reader_config import ReaderOptimizationConfig
from multimodal_lab.schemas import DatasetSplit
from multimodal_lab.training.reader_evaluator import evaluate_reader
from multimodal_lab.training.reader_protocol import (
    ReaderModelContractError,
    ReaderModelOutput,
    forward_reader_model,
)
from multimodal_lab.training.reader_trainer import fit_reader


def make_batch(texts: tuple[str, ...]) -> LineCropBatch:
    return LineCropBatch(
        images=torch.zeros(len(texts), 1, 8, 12),
        widths=torch.full((len(texts),), 12, dtype=torch.long),
        texts=texts,
        sample_ids=tuple(f"sample-{index}" for index in range(len(texts))),
    )


class FixedPathReader(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, images: torch.Tensor, widths: torch.Tensor):
        batch_size = images.shape[0]
        logits = self.fixed_logits.expand(batch_size, -1, -1)
        return ReaderModelOutput(
            logits=logits,
            input_lengths=torch.full(
                (batch_size,),
                logits.shape[1],
                dtype=torch.long,
                device=images.device,
            ),
        )


class LearnablePathReader(nn.Module):
    def __init__(self, vocabulary_size: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(1, 3, vocabulary_size))

    def forward(self, images: torch.Tensor, widths: torch.Tensor):
        batch_size = images.shape[0]
        logits = self.logits.expand(batch_size, -1, -1)
        return ReaderModelOutput(
            logits=logits,
            input_lengths=torch.full(
                (batch_size,),
                logits.shape[1],
                dtype=torch.long,
                device=images.device,
            ),
        )


def test_evaluator_decodes_a_perfect_ctc_path():
    tokenizer = CharacterTokenizer.fit(["ab"])
    path = [0, tokenizer.encode("a")[0], 0, tokenizer.encode("b")[0], 0]
    logits = torch.full((1, len(path), tokenizer.vocabulary_size), -8.0)
    for timestep, token_id in enumerate(path):
        logits[0, timestep, token_id] = 8.0
    model = FixedPathReader(logits)

    report = evaluate_reader(
        model,
        [make_batch(("ab",))],
        tokenizer=tokenizer,
        split=DatasetSplit.validation,
        device="cpu",
    )

    assert report.metrics.character_error_rate == 0
    assert report.metrics.word_error_rate == 0
    assert report.metrics.exact_match_rate == 1
    assert report.predictions[0].prediction == "ab"
    assert report.predictions[0].scored_reference == "ab"


def test_evaluator_counts_unknown_as_one_character_token():
    tokenizer = CharacterTokenizer.fit(["a"])
    unknown_id = tokenizer.unknown_id
    path = [0, unknown_id, 0]
    logits = torch.full((1, len(path), tokenizer.vocabulary_size), -8.0)
    for timestep, token_id in enumerate(path):
        logits[0, timestep, token_id] = 8.0
    model = FixedPathReader(logits)

    report = evaluate_reader(
        model,
        [make_batch(("q",))],
        tokenizer=tokenizer,
        split=DatasetSplit.validation,
        device="cpu",
    )

    assert report.metrics.character_error_rate == 0
    assert report.metrics.unknown_reference_character_count == 1
    assert report.predictions[0].prediction == "<unk>"
    assert report.predictions[0].scored_reference == "\ufffd"
    assert report.predictions[0].scored_prediction == "\ufffd"


def test_evaluator_refuses_to_open_the_training_split():
    tokenizer = CharacterTokenizer.fit(["a"])
    model = LearnablePathReader(tokenizer.vocabulary_size)

    with pytest.raises(ValueError, match="validation or test"):
        evaluate_reader(
            model,
            [make_batch(("a",))],
            tokenizer=tokenizer,
            split=DatasetSplit.train,
            device="cpu",
        )


def test_reader_model_contract_rejects_an_unstructured_tensor():
    class InvalidReader(nn.Module):
        def forward(self, images, widths):
            return torch.zeros(images.shape[0], 2, 3)

    with pytest.raises(ReaderModelContractError, match="ReaderModelOutput"):
        forward_reader_model(
            InvalidReader(),
            torch.zeros(1, 1, 8, 8),
            torch.tensor([8]),
            vocabulary_size=3,
        )


def test_shared_trainer_updates_a_toy_model_and_saves_best_checkpoint(
    tmp_path: Path,
):
    tokenizer = CharacterTokenizer.fit(["a"])
    model = LearnablePathReader(tokenizer.vocabulary_size)
    initial = model.logits.detach().clone()
    checkpoint = tmp_path / "best.pt"
    optimization = ReaderOptimizationConfig(
        epochs=3,
        learning_rate=0.1,
        weight_decay=0,
        gradient_clip_norm=5,
        early_stopping_patience=2,
        mixed_precision=False,
    )

    report = fit_reader(
        model,
        [make_batch(("a", "a"))],
        [make_batch(("a", "a"))],
        tokenizer=tokenizer,
        optimization=optimization,
        device="cpu",
        checkpoint_path=checkpoint,
    )

    assert not torch.equal(initial, model.logits.detach())
    assert checkpoint.is_file()
    assert report.best_epoch >= 1
    assert report.epochs_completed == len(report.history)
