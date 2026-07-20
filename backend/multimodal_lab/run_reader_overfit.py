from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from .experiment_protocol import ExperimentPhase, ExperimentRunSpec, ExperimentTask
from .experiment_tracking import ExperimentRunRecorder, ExperimentRunStatus
from .page_reading import sha256_file
from .reader_config import load_reader_experiment_config
from .training.reader_overfit import (
    ReaderOverfitConfig,
    run_reader_overfit_gate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a reader architecture's 32-line overfit gate."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--max-label-length", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--check-every", type=int, default=20)
    parser.add_argument("--cpu-thread-count", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment = load_reader_experiment_config(args.config)
    config_sha256 = sha256_file(args.config)
    overfit = ReaderOverfitConfig(
        sample_count=args.sample_count,
        seed=args.seed,
        max_label_length=args.max_label_length,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        check_every=args.check_every,
        cpu_thread_count=args.cpu_thread_count,
    )
    spec = ExperimentRunSpec(
        experiment_id=f"{experiment.experiment_id}-overfit-gate",
        task=ExperimentTask.ctc_overfit,
        phase=ExperimentPhase.diagnostic,
        model_variant=experiment.model.kind,
        seed=overfit.seed,
        deterministic_algorithms=experiment.deterministic_algorithms,
        dataset_sha256=experiment.data.dataset_sha256,
        primary_metric="exact_match_rate",
        parameters={
            **asdict(overfit),
            "config_sha256": config_sha256,
            "model_kind": experiment.model.kind,
            "device": args.device,
        },
        tags=[
            experiment.model.kind.removesuffix("_ctc"),
            "ctc",
            "capacity-gate",
            "not-generalization",
        ],
    )
    recorder = ExperimentRunRecorder.start(
        spec,
        output_root=args.output_dir,
        code_root=Path(__file__).resolve().parents[2],
        device=args.device,
    )

    try:
        run = run_reader_overfit_gate(
            experiment,
            project_root=Path(__file__).resolve().parents[1],
            config=overfit,
            device=args.device,
            progress_callback=lambda payload: _record_progress(
                payload,
                output_path=recorder.run_dir / "live_overfit.jsonl",
            ),
        )
        run_dir = recorder.run_dir
        config_path = run_dir / "reader_experiment_config.json"
        vocabulary_path = run_dir / "character_vocabulary.json"
        report_path = run_dir / "reader_overfit_report.json"
        predictions_path = run_dir / "reader_overfit_predictions.json"
        checkpoint_path = run_dir / "reader_overfit_checkpoint.pt"
        progress_path = run_dir / "live_overfit.jsonl"

        config_path.write_text(
            experiment.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        run.tokenizer.save(vocabulary_path)
        report_path.write_text(
            run.report.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        predictions = [
            {
                "sample_id": sample_id,
                "reference": reference,
                "prediction": prediction,
                "exact_match": reference == prediction,
            }
            for sample_id, reference, prediction in zip(
                run.report.sample_ids,
                run.references,
                run.predictions,
            )
        ]
        predictions_path.write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        torch.save(
            {
                "model_state_dict": run.model.state_dict(),
                "model_config": experiment.model.model_dump(mode="json"),
                "vocabulary": run.tokenizer.spec.model_dump(mode="json"),
                "experiment_config_sha256": config_sha256,
                "report": run.report.model_dump(mode="json"),
            },
            checkpoint_path,
        )
        recorder.complete(
            metrics={
                "initial_loss": run.report.initial_loss,
                "final_loss": run.report.final_loss,
                "exact_match_rate": run.report.exact_match_rate,
                "character_error_rate": run.report.character_error_rate,
                "steps_completed": float(run.report.steps_completed),
                "elapsed_seconds": run.report.elapsed_seconds,
            },
            artifacts={
                "config": config_path,
                "vocabulary": vocabulary_path,
                "report": report_path,
                "predictions": predictions_path,
                "checkpoint": checkpoint_path,
                "progress": progress_path,
            },
        )
    except BaseException as exc:
        if recorder.manifest.status is ExperimentRunStatus.running:
            recorder.fail(exc)
        raise

    print(f"run_dir={recorder.run_dir}")
    print(run.report.model_dump_json(indent=2))
    return 0 if run.report.passed else 2


def _record_progress(
    payload: dict[str, float | int],
    *,
    output_path: Path,
) -> None:
    line = json.dumps(payload, ensure_ascii=False)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
    print(line, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
