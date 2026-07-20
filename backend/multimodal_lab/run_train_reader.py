from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import torch

from .experiment_protocol import ExperimentPhase, ExperimentRunSpec, ExperimentTask
from .experiment_tracking import ExperimentRunRecorder, ExperimentRunStatus
from .models import build_reader_model
from .page_reading import sha256_file
from .reader_config import load_reader_experiment_config
from .schemas import DatasetSplit, ReaderEpochRecord
from .training.reader_checkpoint import (
    load_frozen_reader_checkpoint,
    reader_checkpoint_metadata,
)
from .training.reader_data import build_reader_training_data_bundle
from .training.reader_evaluator import evaluate_reader
from .training.reader_trainer import configure_reproducibility, fit_reader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a reader using train/validation lectures only."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-thread-count", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_device(args.device)
    _configure_cpu_threads(args.device, args.cpu_thread_count)
    config = load_reader_experiment_config(args.config)
    config_sha256 = sha256_file(args.config)
    configure_reproducibility(
        config.seed,
        deterministic_algorithms=config.deterministic_algorithms,
    )
    data = build_reader_training_data_bundle(
        config,
        project_root=Path(__file__).resolve().parents[1],
    )
    model = build_reader_model(
        config.model,
        data.tokenizer.vocabulary_size,
        blank_id=data.tokenizer.blank_id,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    spec = ExperimentRunSpec(
        experiment_id=config.experiment_id,
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
            "parameter_count": parameter_count,
            "vocabulary_sha256": data.tokenizer.spec.sha256,
            "augmentation_sha256": config.data.augmentation_sha256 or "disabled",
            "train_sample_count": data.sample_counts[DatasetSplit.train],
            "validation_sample_count": data.sample_counts[DatasetSplit.validation],
            "epochs": config.optimization.epochs,
            "learning_rate": config.optimization.learning_rate,
            "weight_decay": config.optimization.weight_decay,
            "batch_size": config.data.batch_size,
            "device": args.device,
            "cpu_thread_count": torch.get_num_threads(),
        },
        tags=[
            config.model.kind.removesuffix("_ctc"),
            "ctc",
            "train-validation-only",
        ],
    )
    recorder = ExperimentRunRecorder.start(
        spec,
        output_root=args.output_dir,
        code_root=Path(__file__).resolve().parents[2],
        device=args.device,
    )

    try:
        run_dir = recorder.run_dir
        checkpoint_path = run_dir / "best_reader_checkpoint.pt"
        config_path = run_dir / "reader_experiment_config.json"
        vocabulary_path = run_dir / "character_vocabulary.json"
        training_report_path = run_dir / "training_report.json"
        validation_report_path = run_dir / "validation_report.json"
        code_fingerprint_path = run_dir / "code_fingerprint.json"
        live_epochs_path = run_dir / "live_epochs.jsonl"

        config_path.write_text(
            config.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        data.tokenizer.save(vocabulary_path)
        _write_code_fingerprint(
            code_fingerprint_path,
            args.config,
            augmentation_path=config.data.augmentation_path,
        )
        training_report = fit_reader(
            model,
            data.train_loader,
            data.validation_loader,
            tokenizer=data.tokenizer,
            optimization=config.optimization,
            device=args.device,
            checkpoint_path=checkpoint_path,
            checkpoint_metadata=reader_checkpoint_metadata(
                config,
                experiment_config_sha256=config_sha256,
            ),
            epoch_callback=partial(
                _record_epoch,
                output_path=live_epochs_path,
            ),
        )
        load_frozen_reader_checkpoint(
            checkpoint_path,
            model=model,
            config=config,
            tokenizer=data.tokenizer,
            experiment_config_sha256=config_sha256,
            device=args.device,
        )
        validation_report = evaluate_reader(
            model,
            data.validation_loader,
            tokenizer=data.tokenizer,
            split=DatasetSplit.validation,
            device=args.device,
        )
        training_report_path.write_text(
            training_report.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        validation_report_path.write_text(
            validation_report.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        metrics = validation_report.metrics
        recorder.complete(
            metrics={
                "best_epoch": float(training_report.best_epoch),
                "epochs_completed": float(training_report.epochs_completed),
                "validation_loss": metrics.mean_loss,
                "validation_cer": metrics.character_error_rate,
                "validation_wer": metrics.word_error_rate,
                "validation_exact_match_rate": metrics.exact_match_rate,
                "validation_unknown_characters": float(
                    metrics.unknown_reference_character_count
                ),
            },
            artifacts={
                "config": config_path,
                "vocabulary": vocabulary_path,
                "checkpoint": checkpoint_path,
                "training_report": training_report_path,
                "validation_report": validation_report_path,
                "code_fingerprint": code_fingerprint_path,
                "epoch_log": live_epochs_path,
            },
        )
    except BaseException as exc:
        if recorder.manifest.status is ExperimentRunStatus.running:
            recorder.fail(exc)
        raise

    print(f"run_dir={recorder.run_dir}")
    print(f"checkpoint_sha256={sha256_file(checkpoint_path)}")
    print(validation_report.metrics.model_dump_json(indent=2))
    return 0


def _record_epoch(record: ReaderEpochRecord, *, output_path: Path) -> None:
    payload = {
        "epoch": record.epoch,
        "train_loss": record.train_loss,
        "validation_cer": record.validation.character_error_rate,
        "validation_wer": record.validation.word_error_rate,
        "validation_exact_match_rate": record.validation.exact_match_rate,
    }
    line = json.dumps(payload, ensure_ascii=False)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
    print(line, flush=True)


def _validate_device(device: str) -> None:
    configured = torch.device(device)
    if configured.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but this PyTorch build has no CUDA.")


def _configure_cpu_threads(
    device: str,
    cpu_thread_count: int | None,
) -> None:
    if cpu_thread_count is not None and cpu_thread_count <= 0:
        raise ValueError("cpu_thread_count must be positive.")
    if torch.device(device).type == "cpu" and cpu_thread_count is not None:
        torch.set_num_threads(cpu_thread_count)


def _write_code_fingerprint(
    output: Path,
    config_path: Path,
    *,
    augmentation_path: str | None,
) -> None:
    package_root = Path(__file__).resolve().parent
    source_paths = [
        Path(__file__).resolve(),
        package_root / "models" / "factory.py",
        package_root / "models" / "cnn_ctc.py",
        package_root / "models" / "vit_ctc.py",
        package_root / "models" / "reader_layers.py",
        package_root / "line_crop_dataset.py",
        package_root / "reader_config.py",
        package_root / "training" / "reader_data.py",
        package_root / "training" / "reader_trainer.py",
        package_root / "training" / "reader_evaluator.py",
        package_root / "ctc_text.py",
        config_path.resolve(),
    ]
    if augmentation_path is not None:
        source_paths.append(
            (package_root.parent / augmentation_path).resolve()
        )
    payload = {str(path): sha256_file(path) for path in source_paths}
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
