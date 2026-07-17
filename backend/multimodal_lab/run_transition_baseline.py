from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from .annotation_io import write_jsonl
from .schemas import TransitionBaselineConfig, TransitionDetectorVariant
from .transition_baseline import detect_video_transitions


DEFAULT_CONFIG_PATH = (
    Path(__file__).parent / "configs" / "cs231n_2025_web.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the training-free slide-transition baseline."
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", required=True, type=float)
    parser.add_argument("--end", required=True, type=float)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument(
        "--variant",
        choices=[variant.value for variant in TransitionDetectorVariant],
        default=TransitionDetectorVariant.spatial_state.value,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TransitionBaselineConfig.model_validate_json(
        args.config.read_text(encoding="utf-8")
    )
    variant = TransitionDetectorVariant(args.variant)
    started_at = time.perf_counter()
    predictions = detect_video_transitions(
        args.video,
        start_seconds=args.start,
        end_seconds=args.end,
        config=config,
        variant=variant,
        ffmpeg_binary=args.ffmpeg_binary,
        timeout_seconds=args.timeout,
    )
    write_jsonl(args.output, predictions)

    counts = Counter(
        prediction.event_type.value if prediction.event_type else "unknown"
        for prediction in predictions
    )
    print(
        json.dumps(
            {
                "profile_name": config.profile_name,
                "detector_variant": variant.value,
                "start_seconds": args.start,
                "end_seconds": args.end,
                "prediction_count": len(predictions),
                "predictions_by_type": dict(sorted(counts.items())),
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
