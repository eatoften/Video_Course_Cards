from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from .annotation_io import load_jsonl
from .experiment_protocol import ExperimentPhase, ExperimentRunSpec, ExperimentTask
from .experiment_tracking import ExperimentRunRecorder, ExperimentRunStatus
from .line_crop_dataset import partition_by_lecture_split
from .page_reading import sha256_file
from .reader_comparison import (
    ReaderComparisonReport,
    benchmark_reader_models,
    build_system_comparison,
    claim_sealed_test_access,
    finalize_test_access,
    load_reader_comparison_protocol,
    paired_bootstrap_cer_difference,
    prepare_reader_comparison,
    rapidocr_predictions_for_samples,
)
from .reader_config import load_reader_experiment_config
from .schemas import (
    DatasetSplit,
    LineCropReviewRecord,
    LineCropSample,
)
from .training.reader_data import (
    build_reader_test_data_bundle,
    build_reader_training_data_bundle,
)
from .training.reader_evaluator import evaluate_reader
from .training.reader_trainer import configure_reproducibility


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight or execute the one-time reader test comparison."
    )
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--open-sealed-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol_path = args.protocol.resolve()
    protocol = load_reader_comparison_protocol(protocol_path)
    protocol_sha256 = sha256_file(protocol_path)
    project_root = Path(__file__).resolve().parents[1]
    configured_device = torch.device(args.device)
    if configured_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but this PyTorch build has no CUDA.")
    if configured_device.type == "cpu":
        torch.set_num_threads(protocol.cpu_thread_count)

    first_spec = protocol.models[0]
    first_config = load_reader_experiment_config(
        project_root / first_spec.config_path
    )
    configure_reproducibility(
        first_config.seed,
        deterministic_algorithms=first_config.deterministic_algorithms,
    )
    training_data = build_reader_training_data_bundle(
        first_config,
        project_root=project_root,
    )
    if training_data.contract.split.test_lecture_ids != [protocol.test_lecture_id]:
        raise ValueError("Comparison protocol does not name the frozen test lecture.")
    prepared = prepare_reader_comparison(
        protocol,
        project_root=project_root,
        tokenizer=training_data.tokenizer,
        device=configured_device,
    )
    ledger_path = project_root / protocol.test_access_ledger_path

    if args.preflight:
        print(
            json.dumps(
                {
                    "passed": True,
                    "protocol_id": protocol.protocol_id,
                    "protocol_sha256": protocol_sha256,
                    "dataset_sha256": protocol.dataset_sha256,
                    "split_sha256": protocol.split_sha256,
                    "test_lecture_id": protocol.test_lecture_id,
                    "model_checkpoint_sha256": {
                        name: item.spec.checkpoint_sha256
                        for name, item in prepared.items()
                    },
                    "test_access_ledger_exists": ledger_path.exists(),
                    "model_inference_performed": False,
                },
                indent=2,
            )
        )
        return 0

    spec = ExperimentRunSpec(
        experiment_id=protocol.protocol_id,
        task=ExperimentTask.reader_benchmark,
        phase=ExperimentPhase.formal,
        model_variant="cnn_ctc-vs-vit_ctc-vs-rapidocr",
        seed=protocol.bootstrap_seed,
        deterministic_algorithms=True,
        dataset_sha256=protocol.dataset_sha256,
        split_sha256=protocol.split_sha256,
        primary_metric="character_error_rate",
        parameters={
            "protocol_sha256": protocol_sha256,
            "test_lecture_id": protocol.test_lecture_id,
            "cpu_thread_count": protocol.cpu_thread_count,
            "latency_warmup_runs": protocol.latency_warmup_runs,
            "latency_timed_runs": protocol.latency_timed_runs,
            "bootstrap_iterations": protocol.bootstrap_iterations,
            "cnn_checkpoint_sha256": prepared["cnn_v2"].spec.checkpoint_sha256,
            "vit_checkpoint_sha256": prepared["vit_v1"].spec.checkpoint_sha256,
        },
        tags=["cnn", "vit", "ctc", "rapidocr", "sealed-test", "no-selection"],
    )
    recorder = ExperimentRunRecorder.start(
        spec,
        output_root=args.output_dir,
        code_root=project_root.parent,
        device=args.device,
    )
    ledger_payload = None
    try:
        ledger_payload = claim_sealed_test_access(
            ledger_path,
            protocol_id=protocol.protocol_id,
            protocol_sha256=protocol_sha256,
            run_id=recorder.manifest.run_id,
        )
        test_data = build_reader_test_data_bundle(
            first_config,
            project_root=project_root,
        )
        if test_data.tokenizer.spec != training_data.tokenizer.spec:
            raise ValueError("Test bundle tokenizer changed after preflight.")
        batches = list(test_data.test_loader)
        all_samples = load_jsonl(
            test_data.contract.dataset_path,
            LineCropSample,
        )
        test_samples = partition_by_lecture_split(
            all_samples,
            test_data.contract.split,
        )[DatasetSplit.test]
        samples_by_id = {sample.sample_id: sample for sample in test_samples}

        model_reports = {
            name: evaluate_reader(
                item.model,
                batches,
                tokenizer=test_data.tokenizer,
                split=DatasetSplit.test,
                device=configured_device,
            )
            for name, item in prepared.items()
        }
        model_report_paths = {}
        for name, report in model_reports.items():
            path = recorder.run_dir / f"{name}_test_report.json"
            path.write_text(
                report.model_dump_json(indent=2) + "\n",
                encoding="utf-8",
            )
            model_report_paths[name] = path

        reviews = load_jsonl(
            project_root / protocol.rapidocr_reviews_path,
            LineCropReviewRecord,
        )
        rapidocr_predictions = rapidocr_predictions_for_samples(
            test_samples,
            reviews,
            tokenizer=test_data.tokenizer,
        )
        predictions = {
            **{
                name: report.predictions
                for name, report in model_reports.items()
            },
            "rapidocr_stored": rapidocr_predictions,
        }
        latency = benchmark_reader_models(
            {name: item.model for name, item in prepared.items()},
            batches,
            vocabulary_size=test_data.tokenizer.vocabulary_size,
            warmup_runs=protocol.latency_warmup_runs,
            timed_runs=protocol.latency_timed_runs,
            device=configured_device,
        )
        systems = {
            name: build_system_comparison(
                name,
                report.predictions,
                samples_by_id,
                tokenizer=test_data.tokenizer,
                parameter_count=prepared[name].spec.parameter_count,
                mean_ctc_loss=report.metrics.mean_loss,
                latency=latency[name],
                short_text_maximum=protocol.short_text_maximum,
                long_text_minimum=protocol.long_text_minimum,
            )
            for name, report in model_reports.items()
        }
        systems["rapidocr_stored"] = build_system_comparison(
            "rapidocr_stored",
            rapidocr_predictions,
            samples_by_id,
            tokenizer=test_data.tokenizer,
            short_text_maximum=protocol.short_text_maximum,
            long_text_minimum=protocol.long_text_minimum,
        )
        pairs = (
            ("cnn_v2", "vit_v1"),
            ("cnn_v2", "rapidocr_stored"),
            ("vit_v1", "rapidocr_stored"),
        )
        bootstrap = [
            paired_bootstrap_cer_difference(
                system_a,
                predictions[system_a],
                system_b,
                predictions[system_b],
                seed=protocol.bootstrap_seed + pair_index,
                iterations=protocol.bootstrap_iterations,
                confidence_level=protocol.confidence_level,
            )
            for pair_index, (system_a, system_b) in enumerate(pairs)
        ]
        report = ReaderComparisonReport(
            protocol_id=protocol.protocol_id,
            protocol_sha256=protocol_sha256,
            dataset_sha256=protocol.dataset_sha256,
            split_sha256=protocol.split_sha256,
            test_lecture_id=protocol.test_lecture_id,
            sample_count=len(test_samples),
            systems=systems,
            paired_bootstrap=bootstrap,
            rapidocr_scope_warning=(
                "RapidOCR text comes from the same included detector polygons. "
                "Missed detections are absent, so this is not an end-to-end "
                "page-reading or latency comparison."
            ),
        )
        report_path = recorder.run_dir / "reader_comparison_report.json"
        report_path.write_text(
            report.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        code_fingerprint_path = recorder.run_dir / "code_fingerprint.json"
        _write_code_fingerprint(code_fingerprint_path, protocol_path)
        finalize_test_access(
            ledger_path,
            ledger_payload,
            status="completed",
        )
        recorder.complete(
            metrics={
                "cnn_test_cer": systems["cnn_v2"].overall.character_error_rate,
                "vit_test_cer": systems["vit_v1"].overall.character_error_rate,
                "rapidocr_test_cer": (
                    systems["rapidocr_stored"].overall.character_error_rate
                ),
                "cnn_test_exact": systems["cnn_v2"].overall.exact_match_rate,
                "vit_test_exact": systems["vit_v1"].overall.exact_match_rate,
            },
            artifacts={
                "protocol": protocol_path,
                "comparison_report": report_path,
                "cnn_test_report": model_report_paths["cnn_v2"],
                "vit_test_report": model_report_paths["vit_v1"],
                "cnn_checkpoint": project_root
                / prepared["cnn_v2"].spec.checkpoint_path,
                "vit_checkpoint": project_root
                / prepared["vit_v1"].spec.checkpoint_path,
                "rapidocr_reviews": project_root / protocol.rapidocr_reviews_path,
                "test_access_ledger": ledger_path,
                "code_fingerprint": code_fingerprint_path,
            },
        )
    except BaseException as exc:
        if ledger_payload is not None:
            finalize_test_access(
                ledger_path,
                ledger_payload,
                status="failed",
                failure=f"{type(exc).__name__}: {exc}",
            )
        if recorder.manifest.status is ExperimentRunStatus.running:
            recorder.fail(exc)
        raise

    print(f"run_dir={recorder.run_dir}")
    print(f"comparison_report_sha256={sha256_file(report_path)}")
    print(report.model_dump_json(indent=2))
    return 0


def _write_code_fingerprint(output: Path, protocol_path: Path) -> None:
    package_root = Path(__file__).resolve().parent
    paths = (
        Path(__file__).resolve(),
        package_root / "reader_comparison.py",
        package_root / "models" / "cnn_ctc.py",
        package_root / "models" / "vit_ctc.py",
        package_root / "training" / "reader_evaluator.py",
        protocol_path,
    )
    output.write_text(
        json.dumps(
            {str(path): sha256_file(path) for path in paths},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
