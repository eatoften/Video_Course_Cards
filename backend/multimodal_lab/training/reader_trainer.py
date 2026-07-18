from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..ctc_text import CharacterTokenizer, ctc_loss, pack_ctc_targets
from ..line_crop_dataset import LineCropBatch
from ..reader_config import ReaderOptimizationConfig
from ..schemas import (
    DatasetSplit,
    ReaderEpochRecord,
    ReaderTrainingReport,
)
from .reader_evaluator import evaluate_reader
from .reader_protocol import forward_reader_model


def configure_reproducibility(
    seed: int,
    *,
    deterministic_algorithms: bool,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)
    torch.backends.cudnn.benchmark = False


def train_reader_epoch(
    model: nn.Module,
    batches: Iterable[LineCropBatch],
    *,
    tokenizer: CharacterTokenizer,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    gradient_clip_norm: float,
    mixed_precision: bool,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    if gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive.")
    configured_device = torch.device(device)
    use_amp = mixed_precision and configured_device.type == "cuda"
    gradient_scaler = scaler or torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    loss_sum = 0.0
    sample_count = 0

    for batch in batches:
        images = batch.images.to(configured_device)
        widths = batch.widths.to(configured_device)
        targets = pack_ctc_targets(batch.texts, tokenizer)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=configured_device.type,
            enabled=use_amp,
        ):
            output = forward_reader_model(
                model,
                images,
                widths,
                vocabulary_size=tokenizer.vocabulary_size,
            )
            loss = ctc_loss(
                output.logits,
                output.input_lengths,
                targets,
                blank_id=tokenizer.blank_id,
            )
        gradient_scaler.scale(loss).backward()
        gradient_scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        gradient_scaler.step(optimizer)
        gradient_scaler.update()
        loss_sum += float(loss.detach().item()) * len(batch.texts)
        sample_count += len(batch.texts)

    if sample_count == 0:
        raise ValueError("Cannot train on an empty reader dataset.")
    return loss_sum / sample_count


def fit_reader(
    model: nn.Module,
    train_batches: Iterable[LineCropBatch],
    validation_batches: Iterable[LineCropBatch],
    *,
    tokenizer: CharacterTokenizer,
    optimization: ReaderOptimizationConfig,
    device: torch.device | str,
    checkpoint_path: str | Path,
) -> ReaderTrainingReport:
    configured_device = torch.device(device)
    model.to(configured_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimization.learning_rate,
        weight_decay=optimization.weight_decay,
    )
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    use_amp = optimization.mixed_precision and configured_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[ReaderEpochRecord] = []
    best_key: tuple[float, float] | None = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, optimization.epochs + 1):
        train_loss = train_reader_epoch(
            model,
            train_batches,
            tokenizer=tokenizer,
            optimizer=optimizer,
            device=configured_device,
            gradient_clip_norm=optimization.gradient_clip_norm,
            mixed_precision=optimization.mixed_precision,
            scaler=scaler,
        )
        validation = evaluate_reader(
            model,
            validation_batches,
            tokenizer=tokenizer,
            split=DatasetSplit.validation,
            device=configured_device,
        )
        history.append(
            ReaderEpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                validation=validation.metrics,
            )
        )
        key = (
            validation.metrics.character_error_rate,
            validation.metrics.word_error_rate,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": validation.metrics.model_dump(
                        mode="json"
                    ),
                    "vocabulary": tokenizer.spec.model_dump(mode="json"),
                },
                checkpoint,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= optimization.early_stopping_patience:
            break

    return ReaderTrainingReport(
        epochs_completed=len(history),
        best_epoch=best_epoch,
        stopped_early=len(history) < optimization.epochs,
        checkpoint_path=str(checkpoint),
        history=history,
    )
