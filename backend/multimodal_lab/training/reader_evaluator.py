from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from ..ctc_text import (
    CharacterTokenizer,
    ctc_loss,
    greedy_ctc_decode,
    greedy_ctc_decode_token_ids,
    pack_ctc_targets,
)
from ..line_crop_dataset import LineCropBatch
from ..metrics import levenshtein_distance
from ..schemas import (
    DatasetSplit,
    ReaderEvaluationMetrics,
    ReaderEvaluationReport,
    ReaderPrediction,
)
from .reader_protocol import forward_reader_model


def evaluate_reader(
    model: nn.Module,
    batches: Iterable[LineCropBatch],
    *,
    tokenizer: CharacterTokenizer,
    split: DatasetSplit,
    device: torch.device | str,
) -> ReaderEvaluationReport:
    if split is DatasetSplit.train or split is DatasetSplit.smoke:
        raise ValueError("The shared evaluator is for validation or test only.")
    configured_device = torch.device(device)
    model.eval()
    predictions: list[ReaderPrediction] = []
    loss_sum = 0.0
    character_edits = 0
    character_total = 0
    word_edits = 0
    word_total = 0
    unknown_reference_characters = 0

    with torch.inference_mode():
        for batch in batches:
            images = batch.images.to(configured_device)
            widths = batch.widths.to(configured_device)
            output = forward_reader_model(
                model,
                images,
                widths,
                vocabulary_size=tokenizer.vocabulary_size,
            )
            targets = pack_ctc_targets(
                batch.texts,
                tokenizer,
                allow_unknown=True,
            )
            loss = ctc_loss(
                output.logits,
                output.input_lengths,
                targets,
                blank_id=tokenizer.blank_id,
            )
            decoded = greedy_ctc_decode(
                output.logits,
                output.input_lengths,
                tokenizer,
            )
            decoded_token_ids = greedy_ctc_decode_token_ids(
                output.logits,
                output.input_lengths,
                blank_id=tokenizer.blank_id,
                vocabulary_size=tokenizer.vocabulary_size,
            )
            loss_sum += float(loss.item()) * len(batch.texts)
            for sample_id, reference, prediction, prediction_ids in zip(
                batch.sample_ids,
                batch.texts,
                decoded,
                decoded_token_ids,
            ):
                reference_ids = tokenizer.encode(reference)
                scored_reference = _scored_text(reference_ids, tokenizer)
                scored_prediction = _scored_text(prediction_ids, tokenizer)
                reference_words = scored_reference.split()
                prediction_words = scored_prediction.split()
                character_edits += levenshtein_distance(
                    reference_ids,
                    prediction_ids,
                )
                character_total += len(reference_ids)
                word_edits += levenshtein_distance(
                    reference_words,
                    prediction_words,
                )
                word_total += len(reference_words)
                unknown_reference_characters += sum(
                    token_id == tokenizer.unknown_id
                    for token_id in tokenizer.encode(reference)
                )
                predictions.append(
                    ReaderPrediction(
                        sample_id=sample_id,
                        reference=reference,
                        prediction=prediction,
                        scored_reference=scored_reference,
                        scored_prediction=scored_prediction,
                        exact_match=reference_ids == prediction_ids,
                    )
                )

    if not predictions:
        raise ValueError("Cannot evaluate an empty reader dataset.")
    exact_matches = sum(item.exact_match for item in predictions)
    metrics = ReaderEvaluationMetrics(
        sample_count=len(predictions),
        mean_loss=loss_sum / len(predictions),
        character_error_rate=character_edits / character_total,
        word_error_rate=word_edits / word_total,
        exact_match_count=exact_matches,
        exact_match_rate=exact_matches / len(predictions),
        unknown_reference_character_count=unknown_reference_characters,
    )
    return ReaderEvaluationReport(
        split=split,
        metrics=metrics,
        predictions=predictions,
    )


def _scored_text(
    token_ids: list[int],
    tokenizer: CharacterTokenizer,
) -> str:
    return "".join(
        "\ufffd" if token_id == tokenizer.unknown_id else tokenizer.decode([token_id])
        for token_id in token_ids
    )
