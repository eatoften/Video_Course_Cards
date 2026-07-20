from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from .experiment_protocol import ExperimentPhase, ExperimentRunSpec, ExperimentTask
from .experiment_tracking import ExperimentRunRecorder, ExperimentRunStatus
from .models import CnnCtcReader
from .page_reading import sha256_file
from .reader_config import load_reader_experiment_config
from .schemas import DatasetSplit
from .training.reader_checkpoint import load_frozen_reader_checkpoint
from .training.reader_data import build_reader_test_data_bundle
from .training.reader_evaluator import evaluate_reader
from .training.reader_trainer import configure_reproducibility


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one frozen reader checkpoint on held-out test."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_device(args.device)
    expected_checkpoint_hash = args.expected_checkpoint_sha256.strip().lower()
    actual_checkpoint_hash = sha256_file(args.checkpoint)
    if actual_checkpoint_hash != expected_checkpoint_hash:
        raise ValueError(
            "Frozen checkpoint hash mismatch: expected "
            f"{expected_checkpoint_hash}, got {actual_checkpoint_hash}."
        )

    config = load_reader_experiment_config(args.config)
    config_sha256 = sha256_file(args.config)
    configure_reproducibility(
        config.seed,
        deterministic_algorithms=config.deterministic_algorithms,
    )
    data = build_reader_test_data_bundle(
        config,
        project_root=Path(__file__).resolve().parents[1],
    )
    model = CnnCtcReader(
        config.model,
        data.tokenizer.vocabulary_size,
        blank_id=data.tokenizer.blank_id,
    ).to(args.device)
    load_frozen_reader_checkpoint(
        args.checkpoint,
        model=model,
        config=config,
        tokenizer=data.tokenizer,
        experiment_config_sha256=config_sha256,
        device=args.device,
    )
    spec = ExperimentRunSpec(
        experiment_id=f"{config.experiment_id}-held-out-test",
        task=ExperimentTask.reader_benchmark,
        phase=ExperimentPhase.formal,
        model_variant=config.model.kind,
        seed=config.seed,
        deterministic_algorithms=config.deterministic_algorithms,
        dataset_sha256=config.data.dataset_sha256,
        split_sha256=config.data.split_sha256,
        primary_metric=config.selection.primary_metric,
        parameters={
            "config_sha256": config_sha256,
            "checkpoint_sha256": actual_checkpoint_hash,
            "test_sample_count": data.sample_count,
            "device": args.device,
        },
        tags=["cnn", "ctc", "held-out-test", "no-selection"],
    )
    recorder = ExperimentRunRecorder.start(
        spec,
        output_root=args.output_dir,
        code_root=Path(__file__).resolve().parents[2],
        device=args.device,
    )

    try:
        report = evaluate_reader(
            model,
            data.test_loader,
            tokenizer=data.tokenizer,
            split=DatasetSplit.test,
            device=args.device,
        )
        report_path = recorder.run_dir / "test_report.json"
        report_path.write_text(
            report.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        metrics = report.metrics
        recorder.complete(
            metrics={
                "test_loss": metrics.mean_loss,
                "test_cer": metrics.character_error_rate,
                "test_wer": metrics.word_error_rate,
                "test_exact_match_rate": metrics.exact_match_rate,
                "test_unknown_characters": float(
                    metrics.unknown_reference_character_count
                ),
            },
            artifacts={
                "test_report": report_path,
                "frozen_checkpoint": args.checkpoint,
            },
        )
    except BaseException as exc:
        if recorder.manifest.status is ExperimentRunStatus.running:
            recorder.fail(exc)
        raise

    print(f"run_dir={recorder.run_dir}")
    print(report.metrics.model_dump_json(indent=2))
    return 0


def _validate_device(device: str) -> None:
    configured = torch.device(device)
    if configured.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but this PyTorch build has no CUDA.")


if __name__ == "__main__":
    raise SystemExit(main())
