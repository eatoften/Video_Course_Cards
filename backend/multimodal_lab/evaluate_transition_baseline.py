from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from .annotation_io import load_jsonl
from .metrics import evaluate_transition_detection
from .schemas import (
    SlideTransitionAnnotation,
    SlideTransitionPrediction,
    TransitionBaselineEvaluationReport,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate transition predictions against a gold JSONL file."
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--duration", required=True, type=float)
    parser.add_argument("--tolerance", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--held-out",
        action="store_true",
        help="Mark the report as held-out evaluation rather than calibration.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    annotations = load_jsonl(args.annotations, SlideTransitionAnnotation)
    predictions = load_jsonl(args.predictions, SlideTransitionPrediction)
    counts = Counter(
        prediction.event_type.value if prediction.event_type else "unknown"
        for prediction in predictions
    )
    report = TransitionBaselineEvaluationReport(
        calibration_only=not args.held_out,
        interval_duration_seconds=args.duration,
        tolerance_seconds=args.tolerance,
        prediction_count=len(predictions),
        predictions_by_type=dict(sorted(counts.items())),
        relaxed=evaluate_transition_detection(
            annotations,
            predictions,
            video_duration_seconds=args.duration,
            tolerance_seconds=args.tolerance,
            require_event_type=False,
        ),
        typed=evaluate_transition_detection(
            annotations,
            predictions,
            video_duration_seconds=args.duration,
            tolerance_seconds=args.tolerance,
            require_event_type=True,
        ),
    )
    rendered = report.model_dump_json(indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
