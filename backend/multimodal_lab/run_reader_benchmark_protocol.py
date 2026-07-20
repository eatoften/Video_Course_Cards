from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .reader_benchmark_protocol import (
    audit_reader_benchmark_protocol,
    load_reader_benchmark_protocol,
    materialize_protocol_references,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit and materialize the frozen reader v2 protocol."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit")
    _add_shared_arguments(audit)
    audit.add_argument("--output", required=True, type=Path)
    audit.add_argument("--minimum-source-match", type=float, default=0.88)

    materialize = subparsers.add_parser("materialize")
    _add_shared_arguments(materialize)
    materialize.add_argument("--ffmpeg", default="ffmpeg")
    materialize.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = load_reader_benchmark_protocol(args.protocol)
    if args.command == "audit":
        result = audit_reader_benchmark_protocol(
            protocol,
            protocol_path=args.protocol,
            project_root=args.project_root,
            minimum_source_match=args.minimum_source_match,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            result.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        print(result.model_dump_json(indent=2))
        return 0 if result.passed else 2

    result = materialize_protocol_references(
        protocol,
        project_root=args.project_root,
        ffmpeg_executable=args.ffmpeg,
        force=args.force,
    )
    import json

    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())


if __name__ == "__main__":
    raise SystemExit(main())
