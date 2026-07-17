from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .annotation_io import (
    AnnotationFileError,
    load_jsonl,
    validate_annotation_bundle,
)
from .schemas import (
    LectureDatasetManifest,
    SlideTransitionAnnotation,
    StablePageReference,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a multimodal lecture annotation bundle."
    )
    parser.add_argument("--manifests", required=True)
    parser.add_argument("--transitions", required=True)
    parser.add_argument("--references", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = validate_annotation_bundle(
            load_jsonl(args.manifests, LectureDatasetManifest),
            load_jsonl(args.transitions, SlideTransitionAnnotation),
            load_jsonl(args.references, StablePageReference),
        )
    except AnnotationFileError as exc:
        parser.exit(status=1, message=f"annotation validation failed: {exc}\n")

    print(json.dumps(summary.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
