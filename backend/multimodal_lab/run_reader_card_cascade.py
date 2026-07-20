from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, Field

from app import card_service, job_service
from app.llm_client import LLMMessage, LocalLLMClient
from app.settings import LLMSettings
from app.transcription import TranscriptSegment

from .page_reading import sha256_file
from .reader_card_cascade import (
    ReaderCardCascadeProtocol,
    ReaderCardGenerationManifest,
    ReaderCardGenerationRecord,
    ReaderCardGenerationStatus,
    ReaderCardInputCase,
    ReaderCardLLMCall,
    load_frozen_reader_card_inputs,
    load_generation_records,
    load_reader_card_cascade_protocol,
    validate_generation_records,
    write_generation_records_atomic,
)
from .reader_card_evaluation import (
    apply_reader_card_decisions,
    build_reader_card_review_template,
    evaluate_reader_card_cascade,
    load_reader_card_decision_bundle,
    load_reader_card_reviews,
    write_reader_card_reviews,
)


class OllamaModelIdentity(BaseModel):
    name: str = Field(min_length=1)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    modified_at: str | None = None


class RecordingCardLLMClient:
    def __init__(self, delegate: LocalLLMClient) -> None:
        self._delegate = delegate
        self.settings = delegate.settings
        self.calls: list[ReaderCardLLMCall] = []

    def create_chat_completion(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, object] | None = None,
    ) -> str:
        selected_model = model or self.settings.model
        selected_temperature = (
            self.settings.temperature if temperature is None else temperature
        )
        selected_max_tokens = (
            self.settings.max_tokens if max_tokens is None else max_tokens
        )
        started_at = time.perf_counter()
        try:
            output = self._delegate.create_chat_completion(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        except Exception as exc:
            self.calls.append(
                ReaderCardLLMCall(
                    call_index=len(self.calls),
                    messages=[message.model_dump() for message in messages],
                    model=selected_model,
                    temperature=selected_temperature,
                    max_tokens=selected_max_tokens,
                    reasoning_effort=self.settings.reasoning_effort,
                    response_format=response_format,
                    elapsed_seconds=time.perf_counter() - started_at,
                    error_type=type(exc).__name__,
                    error_message=str(exc) or repr(exc),
                )
            )
            raise
        self.calls.append(
            ReaderCardLLMCall(
                call_index=len(self.calls),
                messages=[message.model_dump() for message in messages],
                model=selected_model,
                temperature=selected_temperature,
                max_tokens=selected_max_tokens,
                reasoning_effort=self.settings.reasoning_effort,
                response_format=response_format,
                elapsed_seconds=time.perf_counter() - started_at,
                raw_output=output,
            )
        )
        return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen OCR-to-card downstream comparison."
    )
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--generate", action="store_true")
    action.add_argument("--create-review-template", action="store_true")
    action.add_argument("--apply-review-decisions", action="store_true")
    action.add_argument("--evaluate", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--overwrite-review-template", action="store_true")
    parser.add_argument("--decisions", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol_path = args.protocol.resolve()
    protocol = load_reader_card_cascade_protocol(protocol_path)
    protocol_sha256 = sha256_file(protocol_path)
    project_root = Path(__file__).resolve().parents[1]
    frozen = load_frozen_reader_card_inputs(
        protocol,
        protocol_sha256=protocol_sha256,
        project_root=project_root,
    )
    output_dir = args.output_dir.resolve()
    records_path = output_dir / "generation_records.jsonl"
    reviews_path = output_dir / "source_audit_reviews.jsonl"
    report_path = output_dir / "card_cascade_report.json"

    if args.preflight:
        identity = fetch_ollama_model_identity(
            protocol.base_url,
            protocol.model,
            timeout_seconds=protocol.timeout_seconds,
        )
        _verify_model_identity(protocol, identity)
        print(
            json.dumps(
                {
                    "passed": True,
                    "protocol_id": protocol.protocol_id,
                    "protocol_sha256": protocol_sha256,
                    "model": identity.model_dump(),
                    "page_count": len(frozen.references),
                    "generation_case_count": len(frozen.cases),
                    "schedule_position_counts": dict(
                        Counter(
                            f"{case.system_name}@{case.schedule_position}"
                            for case in frozen.cases
                        )
                    ),
                    "gold_concepts_in_generation_cases": False,
                    "model_inference_performed": False,
                },
                indent=2,
            )
        )
        return 0

    if args.generate:
        return _generate(
            protocol,
            protocol_sha256=protocol_sha256,
            protocol_path=protocol_path,
            project_root=project_root,
            output_dir=output_dir,
            records_path=records_path,
            cases=frozen.cases,
            page_count=len(frozen.references),
            retry_failed=args.retry_failed,
        )

    records = load_generation_records(records_path)
    validate_generation_records(
        frozen.cases,
        records,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )
    if len(records) != len(frozen.cases):
        raise ValueError(
            f"Generation is incomplete: {len(records)} / {len(frozen.cases)}."
        )

    if args.create_review_template:
        if reviews_path.exists() and not args.overwrite_review_template:
            raise FileExistsError(
                f"Review file already exists: {reviews_path}. Use "
                "--overwrite-review-template only before manual review begins."
            )
        reviews = build_reader_card_review_template(
            records,
            frozen.references,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
        )
        write_reader_card_reviews(reviews_path, reviews)
        print(f"review_template={reviews_path}")
        print(f"review_count={len(reviews)}")
        print(f"review_sha256={sha256_file(reviews_path)}")
        return 0

    if args.apply_review_decisions:
        if args.decisions is None:
            raise ValueError("--apply-review-decisions requires --decisions.")
        pending_reviews = load_reader_card_reviews(reviews_path)
        bundle = load_reader_card_decision_bundle(args.decisions.resolve())
        completed_reviews = apply_reader_card_decisions(
            pending_reviews,
            bundle,
        )
        write_reader_card_reviews(reviews_path, completed_reviews)
        print(f"completed_reviews={reviews_path}")
        print(f"review_count={len(completed_reviews)}")
        print(f"review_sha256={sha256_file(reviews_path)}")
        print(f"decisions_sha256={sha256_file(args.decisions.resolve())}")
        return 0

    reviews = load_reader_card_reviews(reviews_path)
    report = evaluate_reader_card_cascade(
        records,
        reviews,
        frozen.references,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        generation_records_sha256=sha256_file(records_path),
        review_records_sha256=sha256_file(reviews_path),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"report={report_path}")
    print(f"report_sha256={sha256_file(report_path)}")
    print(report.model_dump_json(indent=2))
    return 0


def fetch_ollama_model_identity(
    base_url: str,
    model: str,
    *,
    timeout_seconds: float,
) -> OllamaModelIdentity:
    native_root = _ollama_native_root(base_url)
    with httpx.Client(timeout=timeout_seconds, trust_env=False) as client:
        response = client.get(f"{native_root}/api/tags")
        response.raise_for_status()
        payload = response.json()
    raw_models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(raw_models, list):
        raise ValueError("Ollama /api/tags did not return a model list.")
    for raw_model in raw_models:
        if not isinstance(raw_model, dict):
            continue
        names = {raw_model.get("name"), raw_model.get("model")}
        if model not in names:
            continue
        return OllamaModelIdentity(
            name=str(raw_model.get("name") or raw_model.get("model")),
            digest=str(raw_model.get("digest") or ""),
            size_bytes=int(raw_model.get("size") or 0),
            modified_at=(
                str(raw_model["modified_at"])
                if raw_model.get("modified_at") is not None
                else None
            ),
        )
    raise ValueError(f"Ollama model is not installed: {model}.")


def _generate(
    protocol: ReaderCardCascadeProtocol,
    *,
    protocol_sha256: str,
    protocol_path: Path,
    project_root: Path,
    output_dir: Path,
    records_path: Path,
    cases: Sequence[ReaderCardInputCase],
    page_count: int,
    retry_failed: bool,
) -> int:
    identity = fetch_ollama_model_identity(
        protocol.base_url,
        protocol.model,
        timeout_seconds=protocol.timeout_seconds,
    )
    _verify_model_identity(protocol, identity)
    settings = LLMSettings(
        provider=protocol.provider,
        base_url=protocol.base_url,
        model=protocol.model,
        api_key="local",
        temperature=protocol.temperature,
        max_tokens=protocol.max_tokens,
        timeout_seconds=protocol.timeout_seconds,
        reasoning_effort=(
            None
            if protocol.reasoning_effort == "default"
            else protocol.reasoning_effort
        ),
    )
    client = LocalLLMClient(settings)
    records = load_generation_records(records_path)
    validate_generation_records(
        cases,
        records,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )
    if retry_failed:
        records = [
            record
            for record in records
            if record.status is ReaderCardGenerationStatus.succeeded
        ]
        write_generation_records_atomic(records_path, records)
    records_by_key = {record.key: record for record in records}
    order_by_key = {case.key: index for index, case in enumerate(cases)}
    for case_index, case in enumerate(cases, start=1):
        if case.key in records_by_key:
            continue
        record = _generate_case(
            case,
            protocol=protocol,
            delegate=client,
        )
        records_by_key[record.key] = record
        records = sorted(
            records_by_key.values(),
            key=lambda item: order_by_key[item.key],
        )
        write_generation_records_atomic(records_path, records)
        succeeded = sum(
            item.status is ReaderCardGenerationStatus.succeeded for item in records
        )
        failed = len(records) - succeeded
        print(
            json.dumps(
                {
                    "case": f"{case_index}/{len(cases)}",
                    "system": case.system_name,
                    "page_event_id": case.page_event_id,
                    "status": record.status.value,
                    "elapsed_seconds": round(record.elapsed_seconds, 3),
                    "completed": len(records),
                    "succeeded": succeeded,
                    "failed": failed,
                }
            ),
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    succeeded_count = sum(
        record.status is ReaderCardGenerationStatus.succeeded for record in records
    )
    manifest = ReaderCardGenerationManifest(
        protocol_id=protocol.protocol_id,
        protocol_sha256=protocol_sha256,
        model=protocol.model,
        model_digest=protocol.model_digest,
        page_count=page_count,
        expected_record_count=len(cases),
        completed_record_count=len(records),
        succeeded_record_count=succeeded_count,
        failed_record_count=len(records) - succeeded_count,
        records_path=str(records_path),
        records_sha256=sha256_file(records_path),
        code_fingerprints=_code_fingerprints(
            project_root,
            protocol_path,
        ),
        completed_at=datetime.now(UTC),
    )
    manifest_path = output_dir / "generation_manifest.json"
    manifest_path.write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"generation_manifest={manifest_path}")
    print(f"generation_manifest_sha256={sha256_file(manifest_path)}")
    return 0


def _generate_case(
    case: ReaderCardInputCase,
    *,
    protocol: ReaderCardCascadeProtocol,
    delegate: LocalLLMClient,
) -> ReaderCardGenerationRecord:
    generated_at = datetime.now(UTC)
    started_at = time.perf_counter()
    recording_client = RecordingCardLLMClient(delegate)
    try:
        if not case.lines:
            raise ValueError("OCR system produced no text for this page.")
        end_seconds = case.stable_frame_timestamp + 1.0
        segments = [
            TranscriptSegment(
                start_seconds=case.stable_frame_timestamp,
                end_seconds=end_seconds,
                text=line,
            )
            for line in case.lines
        ]
        context = job_service.TranscriptContext(
            job_id=case.page_event_id,
            source_video=(
                f"{case.lecture_id}/page-{case.page_number or case.page_event_id}"
            ),
            start_seconds=case.stable_frame_timestamp,
            end_seconds=end_seconds,
            segments=segments,
            text="\n".join(case.lines),
        )
        request = card_service.CardDraftRequest(
            job_id=case.page_event_id,
            start_seconds=case.stable_frame_timestamp,
            end_seconds=end_seconds,
            card_count=protocol.card_count,
            focus=protocol.focus,
            model=protocol.model,
        )
        response = card_service.draft_knowledge_cards_from_context(
            request,
            context,
            llm_client=recording_client,
            evidence_kind="slide_ocr",
        )
        return ReaderCardGenerationRecord(
            protocol_id=protocol.protocol_id,
            protocol_sha256=case.protocol_sha256,
            model=protocol.model,
            model_digest=protocol.model_digest,
            system_name=case.system_name,
            page_event_id=case.page_event_id,
            lecture_id=case.lecture_id,
            page_number=case.page_number,
            stable_frame_timestamp=case.stable_frame_timestamp,
            schedule_position=case.schedule_position,
            sample_ids=case.sample_ids,
            input_lines=case.lines,
            input_sha256=case.input_sha256,
            status=ReaderCardGenerationStatus.succeeded,
            generated_at=generated_at,
            elapsed_seconds=time.perf_counter() - started_at,
            llm_calls=recording_client.calls,
            response=response,
        )
    except Exception as exc:
        return ReaderCardGenerationRecord(
            protocol_id=protocol.protocol_id,
            protocol_sha256=case.protocol_sha256,
            model=protocol.model,
            model_digest=protocol.model_digest,
            system_name=case.system_name,
            page_event_id=case.page_event_id,
            lecture_id=case.lecture_id,
            page_number=case.page_number,
            stable_frame_timestamp=case.stable_frame_timestamp,
            schedule_position=case.schedule_position,
            sample_ids=case.sample_ids,
            input_lines=case.lines,
            input_sha256=case.input_sha256,
            status=ReaderCardGenerationStatus.failed,
            generated_at=generated_at,
            elapsed_seconds=time.perf_counter() - started_at,
            llm_calls=recording_client.calls,
            error_type=type(exc).__name__,
            error_message=str(exc) or repr(exc),
        )


def _verify_model_identity(
    protocol: ReaderCardCascadeProtocol,
    identity: OllamaModelIdentity,
) -> None:
    if identity.name != protocol.model:
        raise ValueError(
            f"Ollama resolved {identity.name}, expected {protocol.model}."
        )
    if identity.digest != protocol.model_digest:
        raise ValueError(
            "Ollama model digest changed: "
            f"expected {protocol.model_digest}, got {identity.digest}."
        )


def _ollama_native_root(base_url: str) -> str:
    parsed = urlsplit(base_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")


def _code_fingerprints(
    project_root: Path,
    protocol_path: Path,
) -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().parent / "reader_card_cascade.py",
        Path(__file__).resolve().parent / "reader_card_evaluation.py",
        project_root / "app" / "card_service.py",
        protocol_path,
    )
    return {str(path): sha256_file(path) for path in paths}


if __name__ == "__main__":
    raise SystemExit(main())
