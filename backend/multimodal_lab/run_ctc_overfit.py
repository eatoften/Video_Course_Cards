from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from .annotation_io import load_jsonl
from .ctc_overfit import CtcOverfitConfig, run_ctc_overfit_gate
from .experiment_protocol import (
    ExperimentPhase,
    ExperimentRunSpec,
    ExperimentTask,
)
from .experiment_tracking import ExperimentRunRecorder, ExperimentRunStatus
from .page_reading import sha256_file
from .schemas import LineCropSample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the shared 32-line image-to-CTC overfit gate."
    )
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--image-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-label-length", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--check-every", type=int, default=20)
    parser.add_argument("--target-height", type=int, default=24)
    parser.add_argument("--max-image-width", type=int, default=320)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_sha256 = sha256_file(args.samples)
    expected_hash = (args.expected_dataset_sha256 or "").strip().lower()
    if expected_hash and expected_hash != dataset_sha256:
        raise ValueError(
            "Frozen line-crop dataset hash mismatch: "
            f"expected {expected_hash}, got {dataset_sha256}."
        )

    samples = load_jsonl(args.samples, LineCropSample)
    config = CtcOverfitConfig(
        sample_count=args.sample_count,
        seed=args.seed,
        max_label_length=args.max_label_length,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        check_every=args.check_every,
        target_height=args.target_height,
        max_image_width=args.max_image_width,
    )
    spec = ExperimentRunSpec(
        experiment_id="assignment-3-ctc-overfit-gate",
        task=ExperimentTask.ctc_overfit,
        phase=ExperimentPhase.diagnostic,
        model_variant="tiny_cnn_ctc_probe",
        seed=config.seed,
        deterministic_algorithms=False,
        dataset_sha256=dataset_sha256,
        primary_metric="exact_match_rate",
        parameters={**asdict(config), "device": args.device},
        tags=["ctc", "plumbing-gate", "not-generalization"],
    )
    recorder = ExperimentRunRecorder.start(
        spec,
        output_root=args.output_dir,
        code_root=Path(__file__).resolve().parents[2],
        device=args.device,
    )

    try:
        run = run_ctc_overfit_gate(
            samples,
            image_root=args.image_root,
            dataset_sha256=dataset_sha256,
            config=config,
            device=args.device,
        )
        run_dir = recorder.run_dir
        vocabulary_path = run_dir / "character_vocabulary.json"
        report_path = run_dir / "ctc_overfit_report.json"
        predictions_path = run_dir / "ctc_overfit_predictions.json"
        checkpoint_path = run_dir / "ctc_overfit_probe.pt"

        run.tokenizer.save(vocabulary_path)
        report_path.write_text(
            run.report.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        comparisons = [
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
            json.dumps(comparisons, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        torch.save(
            {
                "state_dict": run.model.state_dict(),
                "vocabulary": run.tokenizer.spec.model_dump(mode="json"),
                "config": asdict(config),
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
                "vocabulary": vocabulary_path,
                "report": report_path,
                "predictions": predictions_path,
                "checkpoint": checkpoint_path,
            },
        )
    except BaseException as exc:
        if recorder.manifest.status is ExperimentRunStatus.running:
            recorder.fail(exc)
        raise

    print(f"run_dir={recorder.run_dir}")
    print(run.report.model_dump_json(indent=2))
    return 0 if run.report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
