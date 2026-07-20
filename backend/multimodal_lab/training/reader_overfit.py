from __future__ import annotations

import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from ..annotation_io import load_jsonl
from ..ctc_text import (
    CharacterTokenizer,
    ctc_loss,
    greedy_ctc_decode,
    pack_ctc_targets,
)
from ..line_crop_dataset import (
    LineCropDataset,
    LineImageTransform,
    collate_line_crops,
    partition_by_lecture_split,
)
from ..metrics import levenshtein_distance
from ..models import CnnCtcReader
from ..reader_config import ReaderExperimentConfig, verify_reader_data_contract
from ..schemas import CtcOverfitReport, DatasetSplit, LineCropSample
from .reader_trainer import configure_reproducibility


class ReaderOverfitError(ValueError):
    pass


@dataclass(frozen=True)
class ReaderOverfitConfig:
    sample_count: int = 32
    seed: int = 29
    max_label_length: int = 32
    max_steps: int = 1200
    learning_rate: float = 0.003
    check_every: int = 20
    cpu_thread_count: int = 1

    def __post_init__(self) -> None:
        if self.sample_count <= 0 or self.max_label_length <= 0:
            raise ValueError("Overfit sample and label limits must be positive.")
        if self.max_steps <= 0 or self.check_every <= 0:
            raise ValueError("Overfit step counts must be positive.")
        if self.learning_rate <= 0 or self.cpu_thread_count <= 0:
            raise ValueError("Overfit learning rate and thread count must be positive.")


@dataclass(frozen=True)
class ReaderOverfitRun:
    report: CtcOverfitReport
    tokenizer: CharacterTokenizer
    model: CnnCtcReader
    references: tuple[str, ...]
    predictions: tuple[str, ...]


def select_unique_train_samples(
    samples: Sequence[LineCropSample],
    *,
    train_lecture_ids: Sequence[str],
    config: ReaderOverfitConfig,
) -> list[LineCropSample]:
    allowed_lectures = set(train_lecture_ids)
    candidates_by_text: dict[str, list[LineCropSample]] = {}
    for sample in sorted(samples, key=lambda item: item.sample_id):
        if (
            sample.lecture_id in allowed_lectures
            and len(sample.normalized_text) <= config.max_label_length
        ):
            candidates_by_text.setdefault(sample.normalized_text, []).append(sample)

    if len(candidates_by_text) < config.sample_count:
        raise ReaderOverfitError(
            f"Need {config.sample_count} unique train labels with at most "
            f"{config.max_label_length} characters; found "
            f"{len(candidates_by_text)}."
        )

    rng = random.Random(config.seed)
    selected_texts = rng.sample(
        sorted(candidates_by_text),
        config.sample_count,
    )
    selected = [
        rng.choice(candidates_by_text[text])
        for text in selected_texts
    ]
    return sorted(selected, key=lambda item: item.sample_id)


def run_cnn_overfit_gate(
    experiment: ReaderExperimentConfig,
    *,
    project_root: str | Path,
    config: ReaderOverfitConfig | None = None,
    device: str = "cpu",
) -> ReaderOverfitRun:
    overfit = config or ReaderOverfitConfig()
    contract = verify_reader_data_contract(experiment, project_root=project_root)
    samples = load_jsonl(contract.dataset_path, LineCropSample)
    partitions = partition_by_lecture_split(samples, contract.split)
    train_samples = partitions[DatasetSplit.train]
    tokenizer = CharacterTokenizer.fit(
        [sample.normalized_text for sample in train_samples]
    )
    if tokenizer.spec.sha256 != experiment.data.expected_vocabulary_sha256:
        raise ReaderOverfitError("The frozen train vocabulary changed.")

    selected = select_unique_train_samples(
        train_samples,
        train_lecture_ids=contract.split.train_lecture_ids,
        config=overfit,
    )
    dataset = LineCropDataset(
        selected,
        image_root=contract.image_root,
        transform=LineImageTransform(
            target_height=experiment.data.target_height,
            max_width=experiment.data.max_image_width,
            verify_hashes=True,
        ),
    )
    batch = collate_line_crops(
        [dataset[index] for index in range(len(dataset))],
        width_multiple=experiment.model.temporal_downsample,
    )

    configure_reproducibility(
        overfit.seed,
        deterministic_algorithms=experiment.deterministic_algorithms,
    )
    configured_device = torch.device(device)
    model = CnnCtcReader(
        experiment.model,
        tokenizer.vocabulary_size,
        blank_id=tokenizer.blank_id,
    ).to(configured_device)
    images = batch.images.to(configured_device)
    widths = batch.widths.to(configured_device)
    targets = pack_ctc_targets(batch.texts, tokenizer, device=configured_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=overfit.learning_rate)

    previous_thread_count = torch.get_num_threads()
    if configured_device.type == "cpu":
        torch.set_num_threads(overfit.cpu_thread_count)
    started_at = time.perf_counter()
    steps_completed = 0
    predictions: list[str] = []
    try:
        model.eval()
        with torch.inference_mode():
            initial_output = model(images, widths)
            initial_loss = float(
                ctc_loss(
                    initial_output.logits,
                    initial_output.input_lengths,
                    targets,
                    blank_id=tokenizer.blank_id,
                ).item()
            )

        for step in range(1, overfit.max_steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(images, widths)
            loss = ctc_loss(
                output.logits,
                output.input_lengths,
                targets,
                blank_id=tokenizer.blank_id,
            )
            if not torch.isfinite(loss):
                raise ReaderOverfitError("The CNN overfit loss became non-finite.")
            loss.backward()
            if any(
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise ReaderOverfitError(
                    "The CNN overfit gradients became non-finite."
                )
            optimizer.step()
            steps_completed = step

            if step % overfit.check_every == 0:
                model.eval()
                with torch.inference_mode():
                    checked = model(images, widths)
                    predictions = greedy_ctc_decode(
                        checked.logits,
                        checked.input_lengths,
                        tokenizer,
                    )
                if predictions == list(batch.texts):
                    break

        model.eval()
        with torch.inference_mode():
            final_output = model(images, widths)
            final_loss = float(
                ctc_loss(
                    final_output.logits,
                    final_output.input_lengths,
                    targets,
                    blank_id=tokenizer.blank_id,
                ).item()
            )
            predictions = greedy_ctc_decode(
                final_output.logits,
                final_output.input_lengths,
                tokenizer,
            )
    finally:
        torch.set_num_threads(previous_thread_count)

    references = list(batch.texts)
    exact_match_count = sum(
        reference == prediction
        for reference, prediction in zip(references, predictions)
    )
    character_total = sum(len(reference) for reference in references)
    character_edits = sum(
        levenshtein_distance(reference, prediction)
        for reference, prediction in zip(references, predictions)
    )
    exact_match_rate = exact_match_count / len(references)
    report = CtcOverfitReport(
        sample_ids=list(batch.sample_ids),
        sample_count=len(references),
        dataset_sha256=experiment.data.dataset_sha256,
        vocabulary_sha256=tokenizer.spec.sha256,
        seed=overfit.seed,
        device=str(configured_device),
        model_parameter_count=sum(
            parameter.numel() for parameter in model.parameters()
        ),
        steps_completed=steps_completed,
        max_steps=overfit.max_steps,
        initial_loss=initial_loss,
        final_loss=final_loss,
        exact_match_count=exact_match_count,
        exact_match_rate=exact_match_rate,
        character_error_rate=character_edits / character_total,
        elapsed_seconds=time.perf_counter() - started_at,
        passed=exact_match_rate == 1.0 and final_loss < initial_loss,
    )
    return ReaderOverfitRun(
        report=report,
        tokenizer=tokenizer,
        model=model,
        references=tuple(references),
        predictions=tuple(predictions),
    )
