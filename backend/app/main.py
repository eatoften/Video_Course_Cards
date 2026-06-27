from contextlib import asynccontextmanager
from functools import cache
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, status
from fastapi import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import card_service
from . import job_service
from . import knowledge_card_service
from .db import init_db
from .job import VideoJob, VideoJobStatus
from .job_service import TranscriptContext
from .knowledge_card import (
    KnowledgeCard,
    KnowledgeCardCreate,
    KnowledgeCardUpdate,
)
from .llm_client import LLMModelList, LLMStatus, LocalLLMClient
from .transcription import FasterWhisperTranscriber, TranscriptionResult
from .video_pipeline import VideoPipeline


BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class VideoUploadResponse(BaseModel):
    id: str
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

    if isinstance(exc, job_service.JobNotFoundError):
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
) -> VideoUploadResponse:
    try:
        job = job_service.create_video_job(
            video_file=video.file,
            original_filename=video.filename,
            content_type=video.content_type,
            upload_dir=UPLOAD_DIR,
        )
    except job_service.JobServiceError as exc:
        raise_http_error(exc)

    return VideoUploadResponse(
        id=job.id,
        filename=job.original_filename or "",
        stored_name=job.stored_name or "",
        size_bytes=job.size_bytes or 0,
        status=job.status,
    )


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
