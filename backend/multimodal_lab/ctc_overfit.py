from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from .ctc_text import (
    CharacterTokenizer,
    ctc_loss,
    greedy_ctc_decode,
    pack_ctc_targets,
)
from .line_crop_dataset import (
    LineCropDataset,
    LineImageTransform,
    collate_line_crops,
)
from .metrics import levenshtein_distance
from .schemas import CtcOverfitReport, LineCropSample


class CtcOverfitError(ValueError):
    pass


@dataclass(frozen=True)
class CtcOverfitConfig:
    sample_count: int = 32
    seed: int = 17
    max_label_length: int = 20
    max_steps: int = 1200
    learning_rate: float = 0.003
    check_every: int = 20
    target_height: int = 24
    max_image_width: int = 320

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive.")
        if self.max_label_length <= 0:
            raise ValueError("max_label_length must be positive.")
        if self.max_steps <= 0 or self.check_every <= 0:
            raise ValueError("Training step counts must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")


@dataclass(frozen=True)
class CtcOverfitRun:
    report: CtcOverfitReport
    tokenizer: CharacterTokenizer
    model: TinyCtcLineReader
    references: tuple[str, ...]
    predictions: tuple[str, ...]


class TinyCtcLineReader(nn.Module):
    """A small image-to-CTC probe, not the final CNN experiment model."""

    width_stride = 4

    def __init__(self, vocabulary_size: int, *, blank_id: int = 0) -> None:
        super().__init__()
        if vocabulary_size <= 2:
            raise ValueError("The CTC model needs characters beyond special tokens.")
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )
        self.classifier = nn.Linear(96, vocabulary_size)
        with torch.no_grad():
            self.classifier.bias.zero_()
            self.classifier.bias[blank_id] = -1.0

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("Line images must have shape [batch, 1, height, width].")
        features = self.features(images)
        sequence = features.mean(dim=2).transpose(1, 2)
        return self.classifier(sequence)

    def output_lengths(self, image_widths: Tensor) -> Tensor:
        lengths = torch.div(image_widths, 2, rounding_mode="floor")
        return torch.div(lengths, 2, rounding_mode="floor")


def select_overfit_samples(
    samples: Sequence[LineCropSample],
    config: CtcOverfitConfig,
) -> list[LineCropSample]:
    eligible = [
        sample
        for sample in samples
        if len(sample.normalized_text) <= config.max_label_length
    ]
    if len(eligible) < config.sample_count:
        raise CtcOverfitError(
            f"Need {config.sample_count} labels with at most "
            f"{config.max_label_length} characters; found {len(eligible)}."
        )
    ordered = sorted(eligible, key=lambda sample: sample.sample_id)
    return random.Random(config.seed).sample(ordered, config.sample_count)


def run_ctc_overfit_gate(
    samples: Sequence[LineCropSample],
    *,
    image_root: str | Path,
    dataset_sha256: str,
    config: CtcOverfitConfig | None = None,
    device: str = "cpu",
) -> CtcOverfitRun:
    resolved_config = config or CtcOverfitConfig()
    selected = select_overfit_samples(samples, resolved_config)
    tokenizer = CharacterTokenizer.fit(
        [sample.normalized_text for sample in selected]
    )
    dataset = LineCropDataset(
        selected,
        image_root=image_root,
        transform=LineImageTransform(
            target_height=resolved_config.target_height,
            max_width=resolved_config.max_image_width,
            verify_hashes=True,
        ),
    )
    batch = collate_line_crops(
        [dataset[index] for index in range(len(dataset))],
        width_multiple=TinyCtcLineReader.width_stride,
    )

    torch.manual_seed(resolved_config.seed)
    model = TinyCtcLineReader(
        tokenizer.vocabulary_size,
        blank_id=tokenizer.blank_id,
    ).to(device)
    images = batch.images.to(device)
    image_widths = batch.widths.to(device)
    input_lengths = model.output_lengths(image_widths)
    targets = pack_ctc_targets(batch.texts, tokenizer, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=resolved_config.learning_rate)

    previous_thread_count = torch.get_num_threads()
    torch.set_num_threads(1)
    started_at = time.perf_counter()
    steps_completed = 0
    predictions: list[str] = []
    try:
        model.train()
        with torch.no_grad():
            initial_logits = model(images)
            initial_loss = float(
                ctc_loss(
                    initial_logits,
                    input_lengths,
                    targets,
                    blank_id=tokenizer.blank_id,
                ).item()
            )

        for step in range(1, resolved_config.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = ctc_loss(
                logits,
                input_lengths,
                targets,
                blank_id=tokenizer.blank_id,
            )
            if not torch.isfinite(loss):
                raise CtcOverfitError("The CTC overfit loss became non-finite.")
            loss.backward()
            if any(
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise CtcOverfitError("The CTC overfit gradients became non-finite.")
            optimizer.step()
            steps_completed = step

            if step % resolved_config.check_every == 0:
                model.eval()
                with torch.no_grad():
                    check_logits = model(images)
                    predictions = greedy_ctc_decode(
                        check_logits,
                        input_lengths,
                        tokenizer,
                    )
                if predictions == list(batch.texts):
                    break
                model.train()

        model.eval()
        with torch.no_grad():
            final_logits = model(images)
            final_loss = float(
                ctc_loss(
                    final_logits,
                    input_lengths,
                    targets,
                    blank_id=tokenizer.blank_id,
                ).item()
            )
            predictions = greedy_ctc_decode(
                final_logits,
                input_lengths,
                tokenizer,
            )
    finally:
        torch.set_num_threads(previous_thread_count)

    references = list(batch.texts)
    exact_match_count = sum(
        reference == prediction
        for reference, prediction in zip(references, predictions)
    )
    total_reference_characters = sum(len(reference) for reference in references)
    total_character_edits = sum(
        levenshtein_distance(reference, prediction)
        for reference, prediction in zip(references, predictions)
    )
    exact_match_rate = exact_match_count / len(references)
    report = CtcOverfitReport(
        sample_ids=list(batch.sample_ids),
        sample_count=len(references),
        dataset_sha256=dataset_sha256,
        vocabulary_sha256=tokenizer.spec.sha256,
        seed=resolved_config.seed,
        device=device,
        model_parameter_count=sum(
            parameter.numel() for parameter in model.parameters()
        ),
        steps_completed=steps_completed,
        max_steps=resolved_config.max_steps,
        initial_loss=initial_loss,
        final_loss=final_loss,
        exact_match_count=exact_match_count,
        exact_match_rate=exact_match_rate,
        character_error_rate=total_character_edits / total_reference_characters,
        elapsed_seconds=time.perf_counter() - started_at,
        passed=exact_match_rate == 1.0 and final_loss < initial_loss,
    )
    return CtcOverfitRun(
        report=report,
        tokenizer=tokenizer,
        model=model,
        references=tuple(references),
        predictions=tuple(predictions),
    )
