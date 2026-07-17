from __future__ import annotations

import argparse
import hashlib
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from .annotation_io import load_jsonl, write_jsonl
from .metrics import evaluate_transition_detection
from .run_transition_baseline import DEFAULT_CONFIG_PATH
from .schemas import (
    SlideTransitionAnnotation,
    SlideTransitionPrediction,
    TransitionBaselineConfig,
    TransitionBaselineEvaluationReport,
    TransitionComparisonReport,
    TransitionDetectorVariant,
    TransitionVariantRunResult,
)
from .transition_baseline import (
    decode_sampled_frames,
    detect_transition_predictions,
    extract_ffmpeg_scene_scores,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract transition features once, then compare the registered "
            "training-free detector variants."
        )
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start", required=True, type=float)
    parser.add_argument("--end", required=True, type=float)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--tolerance", type=float, default=1.0)
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument(
        "--expected-config-sha256",
        help="Fail before inference if the config no longer matches the freeze.",
    )
    parser.add_argument(
        "--held-out",
        action="store_true",
        help="Mark every report as held-out rather than calibration-only.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start < 0 or args.end <= args.start:
        raise ValueError("Require 0 <= start < end.")

    config_bytes = args.config.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    expected_hash = (args.expected_config_sha256 or "").strip().lower()
    if expected_hash and config_sha256 != expected_hash:
        raise ValueError(
            "Configuration hash does not match the preregistered freeze: "
            f"expected {expected_hash}, got {config_sha256}."
        )
    config = TransitionBaselineConfig.model_validate_json(config_bytes)
    annotations = load_jsonl(args.annotations, SlideTransitionAnnotation)
    interval_annotations = _annotations_for_interval(
        annotations,
        start_seconds=args.start,
        end_seconds=args.end,
    )

    feature_started_at = time.perf_counter()
    duration_seconds = args.end - args.start
    scene_scores = extract_ffmpeg_scene_scores(
        args.video,
        start_seconds=args.start,
        duration_seconds=duration_seconds,
        config=config,
        ffmpeg_binary=args.ffmpeg_binary,
        timeout_seconds=args.timeout,
    )
    sampled_frames = decode_sampled_frames(
        args.video,
        start_seconds=args.start,
        duration_seconds=duration_seconds,
        config=config,
        ffmpeg_binary=args.ffmpeg_binary,
        timeout_seconds=args.timeout,
    )
    feature_seconds = time.perf_counter() - feature_started_at

    args.output_dir.mkdir(parents=True, exist_ok=True)
    variant_results: dict[str, TransitionVariantRunResult] = {}
    for variant in TransitionDetectorVariant:
        detection_started_at = time.perf_counter()
        predictions = detect_transition_predictions(
            scene_scores,
            sampled_frames,
            config,
            variant=variant,
        )
        detection_seconds = time.perf_counter() - detection_started_at

        prediction_path = args.output_dir / f"{variant.value}_predictions.jsonl"
        evaluation_path = args.output_dir / f"{variant.value}_evaluation.json"
        write_jsonl(prediction_path, predictions)
        evaluation = _evaluate_variant(
            interval_annotations,
            predictions,
            start_seconds=args.start,
            duration_seconds=duration_seconds,
            tolerance_seconds=args.tolerance,
            calibration_only=not args.held_out,
        )
        evaluation_path.write_text(
            evaluation.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        variant_results[variant.value] = TransitionVariantRunResult(
            detector_variant=variant,
            detection_seconds=detection_seconds,
            prediction_path=str(prediction_path),
            evaluation_path=str(evaluation_path),
            evaluation=evaluation,
        )

    report = TransitionComparisonReport(
        calibration_only=not args.held_out,
        video_path=str(args.video),
        config_path=str(args.config),
        config_sha256=config_sha256,
        profile_name=config.profile_name,
        start_seconds=args.start,
        end_seconds=args.end,
        tolerance_seconds=args.tolerance,
        feature_extraction_seconds=feature_seconds,
        scene_score_count=len(scene_scores),
        sampled_frame_count=len(sampled_frames),
        variants=variant_results,
    )
    rendered = report.model_dump_json(indent=2)
    report_path = args.output_dir / "comparison_report.json"
    report_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


def _annotations_for_interval(
    annotations: list[SlideTransitionAnnotation],
    *,
    start_seconds: float,
    end_seconds: float,
) -> list[SlideTransitionAnnotation]:
    return [
        annotation
        for annotation in annotations
        if start_seconds <= annotation.change_start_seconds < end_seconds
    ]


def _evaluate_variant(
    annotations: list[SlideTransitionAnnotation],
    predictions: list[SlideTransitionPrediction],
    *,
    start_seconds: float,
    duration_seconds: float,
    tolerance_seconds: float,
    calibration_only: bool,
) -> TransitionBaselineEvaluationReport:
    relative_annotations = [
        annotation.model_copy(
            update={
                "change_start_seconds": (
                    annotation.change_start_seconds - start_seconds
                ),
                "stable_at_seconds": annotation.stable_at_seconds - start_seconds,
            }
        )
        for annotation in annotations
    ]
    relative_predictions = [
        prediction.model_copy(
            update={
                "timestamp_seconds": prediction.timestamp_seconds - start_seconds,
            }
        )
        for prediction in predictions
        if (
            start_seconds
            <= prediction.timestamp_seconds
            <= start_seconds + duration_seconds
        )
    ]
    counts = Counter(
        prediction.event_type.value if prediction.event_type else "unknown"
        for prediction in predictions
    )
    return TransitionBaselineEvaluationReport(
        calibration_only=calibration_only,
        interval_duration_seconds=duration_seconds,
        tolerance_seconds=tolerance_seconds,
        prediction_count=len(predictions),
        predictions_by_type=dict(sorted(counts.items())),
        relaxed=evaluate_transition_detection(
            relative_annotations,
            relative_predictions,
            video_duration_seconds=duration_seconds,
            tolerance_seconds=tolerance_seconds,
            require_event_type=False,
        ),
        typed=evaluate_transition_detection(
            relative_annotations,
            relative_predictions,
            video_duration_seconds=duration_seconds,
            tolerance_seconds=tolerance_seconds,
            require_event_type=True,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
