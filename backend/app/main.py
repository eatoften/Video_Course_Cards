from contextlib import asynccontextmanager
from functools import cache
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, UploadFile
from fastapi import status
from fastapi import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import auto_card_generation_service
from . import card_service
from . import course_service
from . import export_service
from . import job_service
from . import knowledge_card_service
from . import knowledge_card_note_service
from . import transcript_chunk_service
from .course import Course, CourseCreate, CourseUpdate
from .card_generation_run import AutoCardGenerationRequest, CardGenerationRun
from .db import init_db
from .job import VideoJob, VideoJobStatus
from .job_service import TranscriptContext
from .knowledge_card import (
    KnowledgeCard,
    KnowledgeCardCreate,
    KnowledgeCardIndexItem,
    KnowledgeCardUpdate,
)
from .knowledge_card_note import (
    KnowledgeCardNote,
    KnowledgeCardNoteCreate,
    KnowledgeCardNoteUpdate,
)
from .llm_client import LLMModelList, LLMStatus, LocalLLMClient
from .transcription import FasterWhisperTranscriber, TranscriptionResult
from .transcript_chunk import TranscriptChunk, TranscriptChunkGenerationRequest
from .video_pipeline import VideoPipeline


BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class VideoUploadResponse(BaseModel):
    id: str
    course_id: str
    filename: str
    stored_name: str
    size_bytes: int
    status: VideoJobStatus


def raise_http_error(exc: job_service.JobServiceError) -> None:
    if isinstance(exc, job_service.MissingFilenameError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if isinstance(
        exc,
        (
            job_service.UnsupportedVideoExtensionError,
            job_service.UnsupportedContentTypeError,
            job_service.InvalidVideoError,
        ),
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc

    if isinstance(
        exc,
        (
            job_service.JobNotFoundError,
            job_service.CourseNotFoundError,
        ),
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    if isinstance(
        exc,
        (
            job_service.InvalidJobStatusError,
            job_service.TranscriptNotReadyError,
        ),
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    if isinstance(exc, job_service.InvalidTranscriptContextError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unexpected job service error.",
    ) from exc


def raise_card_http_error(exc: card_service.CardServiceError) -> None:
    if isinstance(exc, card_service.InvalidCardDraftRequestError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if isinstance(exc, card_service.CardGenerationTimeoutError):
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        ) from exc

    if isinstance(exc, card_service.CardGenerationError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    if isinstance(exc, card_service.CardOutputParseError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unexpected card service error.",
    ) from exc


def raise_knowledge_card_http_error(
    exc: knowledge_card_service.KnowledgeCardServiceError,
) -> None:
    if isinstance(exc, knowledge_card_service.KnowledgeCardNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    if isinstance(exc, knowledge_card_service.InvalidKnowledgeCardError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unexpected knowledge card service error.",
    ) from exc


def raise_knowledge_card_note_http_error(
    exc: knowledge_card_note_service.KnowledgeCardNoteServiceError,
) -> None:
    if isinstance(
        exc,
        (
            knowledge_card_note_service.KnowledgeCardNoteNotFoundError,
            knowledge_card_note_service.KnowledgeCardForNoteNotFoundError,
        ),
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    if isinstance(
        exc,
        knowledge_card_note_service.InvalidKnowledgeCardNoteError,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unexpected knowledge card note service error.",
    ) from exc


def raise_course_http_error(exc: course_service.CourseServiceError) -> None:
    if isinstance(exc, course_service.CourseNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    if isinstance(
        exc,
        (
            course_service.InvalidCourseError,
            course_service.DefaultCourseDeleteError,
        ),
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unexpected course service error.",
    ) from exc


def raise_transcript_chunk_http_error(
    exc: transcript_chunk_service.TranscriptChunkServiceError,
) -> None:
    if isinstance(
        exc,
        transcript_chunk_service.InvalidTranscriptChunkConfigError,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if isinstance(
        exc,
        transcript_chunk_service.TranscriptChunkGenerationError,
    ):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unexpected transcript chunk service error.",
    ) from exc


def raise_auto_generation_http_error(
    exc: auto_card_generation_service.AutoCardGenerationServiceError,
) -> None:
    if isinstance(
        exc,
        auto_card_generation_service.CardGenerationRunNotFoundError,
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    if isinstance(
        exc,
        auto_card_generation_service.InvalidAutoCardGenerationRequestError,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unexpected auto card generation service error.",
    ) from exc


def archive_response(archive: export_service.MarkdownArchive) -> Response:
    return Response(
        content=archive.content,
        media_type=archive.media_type,
        headers={
            "Content-Disposition": (
                f'attachment; filename="{archive.filename}"'
            ),
        },
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    yield


app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["*"],
)


@cache
def get_video_pipeline() -> VideoPipeline:
    transcriber = FasterWhisperTranscriber(
        model_size="base",
        device="cpu",
        compute_type="int8",
    )

    return VideoPipeline(
        transcriber=transcriber,
    )


@cache
def get_llm_client() -> LocalLLMClient:
    return LocalLLMClient()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get(
    "/llm/status",
    response_model=LLMStatus,
)
def get_llm_status() -> LLMStatus:
    return get_llm_client().check_status()


@app.get(
    "/llm/models",
    response_model=LLMModelList,
)
def list_llm_models() -> LLMModelList:
    return get_llm_client().list_models()


@app.post("/videos/inspect")
async def inspect_video(video: UploadFile):
    return {
        "filename": video.filename,
        "content_type": video.content_type,
        "size_bytes": video.size,
    }


@app.post(
    "/videos",
    response_model=VideoUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_video(
    video: UploadFile,
    course_id: str | None = Form(default=None),
) -> VideoUploadResponse:
    try:
        job = job_service.create_video_job(
            video_file=video.file,
            original_filename=video.filename,
            content_type=video.content_type,
            upload_dir=UPLOAD_DIR,
            course_id=course_id,
        )
    except job_service.JobServiceError as exc:
        raise_http_error(exc)

    return VideoUploadResponse(
        id=job.id,
        course_id=job.course_id,
        filename=job.original_filename or "",
        stored_name=job.stored_name or "",
        size_bytes=job.size_bytes or 0,
        status=job.status,
    )


@app.get(
    "/courses",
    response_model=list[Course],
)
def list_courses() -> list[Course]:
    return course_service.list_video_courses()


@app.post(
    "/courses",
    response_model=Course,
    status_code=status.HTTP_201_CREATED,
)
def create_course(request: CourseCreate) -> Course:
    try:
        return course_service.create_video_course(request)
    except course_service.CourseServiceError as exc:
        raise_course_http_error(exc)


@app.patch(
    "/courses/{course_id}",
    response_model=Course,
)
def update_course(
    course_id: str,
    request: CourseUpdate,
) -> Course:
    try:
        return course_service.update_video_course(course_id, request)
    except course_service.CourseServiceError as exc:
        raise_course_http_error(exc)


@app.delete(
    "/courses/{course_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_course(course_id: str) -> Response:
    try:
        course_service.delete_video_course(course_id)
    except course_service.CourseServiceError as exc:
        raise_course_http_error(exc)

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get(
    "/courses/{course_id}/jobs",
    response_model=list[VideoJob],
)
def list_course_jobs(course_id: str) -> list[VideoJob]:
    try:
        return course_service.list_course_jobs(course_id)
    except course_service.CourseServiceError as exc:
        raise_course_http_error(exc)


@app.get(
    "/courses/{course_id}/cards",
    response_model=list[KnowledgeCard],
)
def list_course_cards(course_id: str) -> list[KnowledgeCard]:
    try:
        return course_service.list_course_cards(course_id)
    except course_service.CourseServiceError as exc:
        raise_course_http_error(exc)


@app.post(
    "/courses/{course_id}/chunks",
    response_model=list[TranscriptChunk],
)
def generate_course_transcript_chunks(
    course_id: str,
    request: TranscriptChunkGenerationRequest | None = None,
) -> list[TranscriptChunk]:
    try:
        return transcript_chunk_service.generate_course_chunks(
            course_id,
            request,
        )
    except course_service.CourseServiceError as exc:
        raise_course_http_error(exc)
    except job_service.JobServiceError as exc:
        raise_http_error(exc)
    except transcript_chunk_service.TranscriptChunkServiceError as exc:
        raise_transcript_chunk_http_error(exc)


@app.get(
    "/courses/{course_id}/chunks",
    response_model=list[TranscriptChunk],
)
def list_course_transcript_chunks(course_id: str) -> list[TranscriptChunk]:
    try:
        return transcript_chunk_service.list_course_transcript_chunks(
            course_id
        )
    except course_service.CourseServiceError as exc:
        raise_course_http_error(exc)


@app.get(
    "/courses/{course_id}/card-index",
    response_model=list[KnowledgeCardIndexItem],
)
def list_course_card_index(
    course_id: str,
) -> list[KnowledgeCardIndexItem]:
    try:
        return course_service.list_course_card_index(course_id)
    except course_service.CourseServiceError as exc:
        raise_course_http_error(exc)


@app.delete(
    "/courses/{course_id}/cards",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_course_cards(course_id: str) -> Response:
    try:
        course_service.delete_all_course_cards(course_id)
    except course_service.CourseServiceError as exc:
        raise_course_http_error(exc)

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get(
    "/jobs",
    response_model=list[VideoJob],
)
def list_jobs() -> list[VideoJob]:
    return job_service.list_video_jobs()


@app.get(
    "/jobs/{job_id}",
    response_model=VideoJob,
)
def get_job(job_id: str) -> VideoJob:
    try:
        job = job_service.get_video_job(job_id)
    except job_service.JobServiceError as exc:
        raise_http_error(exc)

    return job


@app.post(
    "/jobs/{job_id}/run",
    response_model=VideoJob,
    status_code=status.HTTP_202_ACCEPTED,
)
def run_job(
    job_id: str,
    background_tasks: BackgroundTasks,
) -> VideoJob:
    try:
        job = job_service.start_job(job_id)
    except job_service.JobServiceError as exc:
        raise_http_error(exc)

    background_tasks.add_task(
        job_service.run_job_pipeline,
        job.id,
        get_video_pipeline,
        DATA_DIR,
    )
    return job


@app.post(
    "/jobs/{job_id}/retry",
    response_model=VideoJob,
    status_code=status.HTTP_202_ACCEPTED,
)
def retry_job(
    job_id: str,
    background_tasks: BackgroundTasks,
) -> VideoJob:
    try:
        job = job_service.retry_job(job_id)
    except job_service.JobServiceError as exc:
        raise_http_error(exc)

    background_tasks.add_task(
        job_service.run_job_pipeline,
        job.id,
        get_video_pipeline,
        DATA_DIR,
    )
    return job


@app.delete(
    "/jobs/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_job(job_id: str) -> Response:
    try:
        job_service.delete_video_job(job_id, DATA_DIR)
    except job_service.JobServiceError as exc:
        raise_http_error(exc)

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get(
    "/jobs/{job_id}/transcript",
    response_model=TranscriptionResult,
)
def get_job_transcript(job_id: str) -> TranscriptionResult:
    try:
        return job_service.get_job_transcript(job_id)
    except job_service.JobServiceError as exc:
        raise_http_error(exc)


@app.get(
    "/jobs/{job_id}/context",
    response_model=TranscriptContext,
)
def get_transcript_context(
    job_id: str,
    start_seconds: float,
    end_seconds: float,
) -> TranscriptContext:
    try:
        return job_service.get_transcript_context(
            job_id,
            start_seconds,
            end_seconds,
        )
    except job_service.JobServiceError as exc:
        raise_http_error(exc)


@app.post(
    "/jobs/{job_id}/chunks",
    response_model=list[TranscriptChunk],
)
def generate_job_transcript_chunks(
    job_id: str,
    request: TranscriptChunkGenerationRequest | None = None,
) -> list[TranscriptChunk]:
    try:
        return transcript_chunk_service.generate_job_chunks(
            job_id,
            request,
        )
    except job_service.JobServiceError as exc:
        raise_http_error(exc)
    except transcript_chunk_service.TranscriptChunkServiceError as exc:
        raise_transcript_chunk_http_error(exc)


@app.get(
    "/jobs/{job_id}/chunks",
    response_model=list[TranscriptChunk],
)
def list_job_transcript_chunks(job_id: str) -> list[TranscriptChunk]:
    try:
        return transcript_chunk_service.list_job_chunks(job_id)
    except job_service.JobServiceError as exc:
        raise_http_error(exc)


@app.post(
    "/cards/draft",
    response_model=card_service.CardDraftResponse,
)
def draft_cards(
    request: card_service.CardDraftRequest,
) -> card_service.CardDraftResponse:
    try:
        return card_service.draft_knowledge_cards(
            request,
            llm_client=get_llm_client(),
        )
    except job_service.JobServiceError as exc:
        raise_http_error(exc)
    except card_service.CardServiceError as exc:
        raise_card_http_error(exc)


@app.post(
    "/jobs/{job_id}/cards/auto-generate",
    response_model=CardGenerationRun,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_auto_card_generation(
    job_id: str,
    background_tasks: BackgroundTasks,
    request: AutoCardGenerationRequest | None = None,
) -> CardGenerationRun:
    try:
        run = auto_card_generation_service.start_auto_card_generation(
            job_id,
            request,
        )
    except job_service.JobServiceError as exc:
        raise_http_error(exc)
    except auto_card_generation_service.AutoCardGenerationServiceError as exc:
        raise_auto_generation_http_error(exc)

    background_tasks.add_task(
        auto_card_generation_service.run_auto_card_generation,
        run.id,
        get_llm_client,
    )

    return run


@app.get(
    "/card-generation-runs/{run_id}",
    response_model=CardGenerationRun,
)
def get_card_generation_run(run_id: str) -> CardGenerationRun:
    try:
        return auto_card_generation_service.get_card_generation_run(run_id)
    except auto_card_generation_service.AutoCardGenerationServiceError as exc:
        raise_auto_generation_http_error(exc)


@app.get(
    "/jobs/{job_id}/card-generation-runs",
    response_model=list[CardGenerationRun],
)
def list_job_card_generation_runs(job_id: str) -> list[CardGenerationRun]:
    try:
        return auto_card_generation_service.list_job_card_generation_runs(
            job_id
        )
    except job_service.JobServiceError as exc:
        raise_http_error(exc)


@app.get("/jobs/{job_id}/cards/export/markdown")
def export_job_cards_markdown(job_id: str) -> Response:
    try:
        return archive_response(
            export_service.export_job_cards_markdown(job_id)
        )
    except job_service.JobServiceError as exc:
        raise_http_error(exc)


@app.post(
    "/jobs/{job_id}/cards/export/markdown/local",
    response_model=export_service.SavedMarkdownArchive,
)
def save_job_cards_markdown_export(job_id: str):
    try:
        archive = export_service.export_job_cards_markdown(job_id)
        return export_service.save_archive_to_disk(archive)
    except job_service.JobServiceError as exc:
        raise_http_error(exc)


@app.post(
    "/jobs/{job_id}/cards/export/markdown/folder",
    response_model=export_service.SavedMarkdownFolder,
)
def save_job_cards_markdown_folder_export(job_id: str):
    try:
        return export_service.save_job_cards_markdown_folder(job_id)
    except job_service.JobServiceError as exc:
        raise_http_error(exc)


@app.get(
    "/jobs/{job_id}/cards",
    response_model=list[KnowledgeCard],
)
def list_job_cards(job_id: str) -> list[KnowledgeCard]:
    try:
        return knowledge_card_service.list_job_cards(job_id)
    except job_service.JobServiceError as exc:
        raise_http_error(exc)


@app.post(
    "/jobs/{job_id}/cards",
    response_model=KnowledgeCard,
    status_code=status.HTTP_201_CREATED,
)
def save_job_card(
    job_id: str,
    request: KnowledgeCardCreate,
) -> KnowledgeCard:
    try:
        return knowledge_card_service.save_job_card(job_id, request)
    except job_service.JobServiceError as exc:
        raise_http_error(exc)
    except knowledge_card_service.KnowledgeCardServiceError as exc:
        raise_knowledge_card_http_error(exc)


@app.delete(
    "/jobs/{job_id}/cards",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_job_cards(job_id: str) -> Response:
    try:
        knowledge_card_service.delete_all_job_cards(job_id)
    except job_service.JobServiceError as exc:
        raise_http_error(exc)

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/cards/export/markdown")
def export_all_cards_markdown() -> Response:
    return archive_response(
        export_service.export_all_cards_markdown()
    )


@app.post(
    "/cards/export/markdown/local",
    response_model=export_service.SavedMarkdownArchive,
)
def save_all_cards_markdown_export():
    archive = export_service.export_all_cards_markdown()
    return export_service.save_archive_to_disk(archive)


@app.post(
    "/cards/export/markdown/folder",
    response_model=export_service.SavedMarkdownFolder,
)
def save_all_cards_markdown_folder_export():
    return export_service.save_all_cards_markdown_folder()


@app.get(
    "/cards/{card_id}",
    response_model=KnowledgeCard,
)
def get_saved_card(card_id: str) -> KnowledgeCard:
    try:
        return knowledge_card_service.get_saved_card(card_id)
    except knowledge_card_service.KnowledgeCardServiceError as exc:
        raise_knowledge_card_http_error(exc)


@app.get(
    "/cards/{card_id}/notes",
    response_model=list[KnowledgeCardNote],
)
def list_card_notes(card_id: str) -> list[KnowledgeCardNote]:
    try:
        return knowledge_card_note_service.list_card_notes(card_id)
    except knowledge_card_note_service.KnowledgeCardNoteServiceError as exc:
        raise_knowledge_card_note_http_error(exc)


@app.post(
    "/cards/{card_id}/notes",
    response_model=KnowledgeCardNote,
    status_code=status.HTTP_201_CREATED,
)
def save_card_note(
    card_id: str,
    request: KnowledgeCardNoteCreate,
) -> KnowledgeCardNote:
    try:
        return knowledge_card_note_service.save_card_note(card_id, request)
    except knowledge_card_note_service.KnowledgeCardNoteServiceError as exc:
        raise_knowledge_card_note_http_error(exc)


@app.patch(
    "/card-notes/{note_id}",
    response_model=KnowledgeCardNote,
)
def update_card_note(
    note_id: str,
    request: KnowledgeCardNoteUpdate,
) -> KnowledgeCardNote:
    try:
        return knowledge_card_note_service.update_card_note(note_id, request)
    except knowledge_card_note_service.KnowledgeCardNoteServiceError as exc:
        raise_knowledge_card_note_http_error(exc)


@app.delete(
    "/card-notes/{note_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_card_note(note_id: str) -> Response:
    try:
        knowledge_card_note_service.delete_card_note(note_id)
    except knowledge_card_note_service.KnowledgeCardNoteServiceError as exc:
        raise_knowledge_card_note_http_error(exc)

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.patch(
    "/cards/{card_id}",
    response_model=KnowledgeCard,
)
def update_saved_card(
    card_id: str,
    request: KnowledgeCardUpdate,
) -> KnowledgeCard:
    try:
        return knowledge_card_service.update_saved_card(card_id, request)
    except knowledge_card_service.KnowledgeCardServiceError as exc:
        raise_knowledge_card_http_error(exc)


@app.delete(
    "/cards/{card_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_saved_card(card_id: str) -> Response:
    try:
        knowledge_card_service.delete_saved_card(card_id)
    except knowledge_card_service.KnowledgeCardServiceError as exc:
        raise_knowledge_card_http_error(exc)

    return Response(status_code=status.HTTP_204_NO_CONTENT)
