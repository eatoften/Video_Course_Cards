from collections.abc import Callable
from uuid import uuid4

from . import card_service
from . import job_service
from . import knowledge_card_service
from . import transcript_chunk_service
from .card_generation_run import (
    AutoCardGenerationRequest,
    CardGenerationRun,
    CardGenerationRunError,
)
from .card_generation_run_store import (
    create_run,
    get_run,
    list_runs_for_job,
    update_run,
)
from .job import utc_now
from .knowledge_card import KnowledgeCardCreate
from .knowledge_card_store import list_cards_for_job
from .transcript_chunk import TranscriptChunk
from .transcript_chunk_store import list_chunks_for_job


class AutoCardGenerationServiceError(Exception):
    pass


class CardGenerationRunNotFoundError(AutoCardGenerationServiceError):
    pass


class InvalidAutoCardGenerationRequestError(AutoCardGenerationServiceError):
    pass


def start_auto_card_generation(
    job_id: str,
    request: AutoCardGenerationRequest | None = None,
) -> CardGenerationRun:
    job = job_service.get_video_job(job_id)
    job_service.get_job_transcript(job.id)
    generation_request = request or AutoCardGenerationRequest()
    now = utc_now()
    run = CardGenerationRun(
        id=uuid4().hex,
        job_id=job.id,
        status="pending",
        model=_clean_optional_text(generation_request.model),
        card_count_per_chunk=generation_request.card_count_per_chunk,
        request=generation_request,
        created_at=now,
        updated_at=now,
    )

    create_run(run)

    return run


def get_card_generation_run(run_id: str) -> CardGenerationRun:
    run = get_run(run_id)

    if run is None:
        raise CardGenerationRunNotFoundError(
            "Card generation run not found."
        )

    return run


def list_job_card_generation_runs(job_id: str) -> list[CardGenerationRun]:
    job = job_service.get_video_job(job_id)

    return list_runs_for_job(job.id)


def run_auto_card_generation(
    run_id: str,
    llm_client_factory: Callable[[], card_service.CardLLMClient],
) -> None:
    try:
        run = get_card_generation_run(run_id)
    except CardGenerationRunNotFoundError:
        return

    _mark_running(run)

    try:
        chunks = _prepare_chunks(run)
        selected_chunks = _limit_chunks(chunks, run.request.max_chunks)

        run.total_chunks = len(selected_chunks)
        run.updated_at = utc_now()
        update_run(run)

        if not selected_chunks:
            run.status = "completed"
            run.completed_at = utc_now()
            run.updated_at = run.completed_at
            update_run(run)
            return

        llm_client = llm_client_factory()

        for chunk in selected_chunks:
            _process_chunk(
                run,
                chunk,
                llm_client=llm_client,
            )

        run.status = "completed"
        run.completed_at = utc_now()
        run.updated_at = run.completed_at
        update_run(run)
    except Exception as exc:
        run.status = "failed"
        run.error_message = str(exc)
        run.completed_at = utc_now()
        run.updated_at = run.completed_at
        update_run(run)


def _mark_running(run: CardGenerationRun) -> None:
    now = utc_now()
    run.status = "running"
    run.started_at = now
    run.updated_at = now
    update_run(run)


def _prepare_chunks(run: CardGenerationRun) -> list[TranscriptChunk]:
    chunks = list_chunks_for_job(run.job_id)

    if run.request.regenerate_chunks or not chunks:
        chunks = transcript_chunk_service.generate_job_chunks(
            run.job_id,
            run.request.chunking,
        )

    return chunks


def _limit_chunks(
    chunks: list[TranscriptChunk],
    max_chunks: int | None,
) -> list[TranscriptChunk]:
    if max_chunks is None:
        return chunks

    return chunks[:max_chunks]


def _process_chunk(
    run: CardGenerationRun,
    chunk: TranscriptChunk,
    *,
    llm_client: card_service.CardLLMClient,
) -> None:
    try:
        draft = card_service.draft_knowledge_cards(
            card_service.CardDraftRequest(
                job_id=chunk.job_id,
                start_seconds=chunk.start_seconds,
                end_seconds=chunk.end_seconds,
                card_count=run.request.card_count_per_chunk,
                focus=run.request.focus,
                model=run.request.model,
            ),
            llm_client=llm_client,
        )

        created_count = _save_new_cards_from_draft(run.job_id, draft)
        run.cards_created += created_count
        run.succeeded_chunks += 1
    except Exception as exc:
        run.failed_chunks += 1
        run.errors.append(
            CardGenerationRunError(
                chunk_id=chunk.id,
                chunk_index=chunk.chunk_index,
                message=str(exc),
            )
        )
    finally:
        run.completed_chunks += 1
        run.updated_at = utc_now()
        update_run(run)


def _save_new_cards_from_draft(
    job_id: str,
    draft: card_service.CardDraftResponse,
) -> int:
    existing_signatures = {
        _card_signature(
            title=card.title,
            source_start_seconds=card.source_start_seconds,
            source_end_seconds=card.source_end_seconds,
        )
        for card in list_cards_for_job(job_id)
    }
    created_count = 0

    for card in draft.cards:
        signature = _card_signature(
            title=card.title,
            source_start_seconds=card.source_start_seconds,
            source_end_seconds=card.source_end_seconds,
        )

        if signature in existing_signatures:
            continue

        knowledge_card_service.save_job_card(
            job_id,
            KnowledgeCardCreate(
                title=card.title,
                summary=card.summary,
                key_points=card.key_points,
                claims=card.claims,
                unsupported_terms=card.unsupported_terms,
                question=card.question,
                answer=card.answer,
                difficulty=card.difficulty,
                tags=[],
                review_state="draft",
                source_start_seconds=card.source_start_seconds,
                source_end_seconds=card.source_end_seconds,
                provider=draft.provider,
                model=draft.model,
            ),
        )
        existing_signatures.add(signature)
        created_count += 1

    return created_count


def _card_signature(
    *,
    title: str,
    source_start_seconds: float,
    source_end_seconds: float,
) -> tuple[str, float, float]:
    return (
        " ".join(title.lower().split()),
        round(source_start_seconds, 2),
        round(source_end_seconds, 2),
    )


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None

    stripped = value.strip()

    return stripped or None
