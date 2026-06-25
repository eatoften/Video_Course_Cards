import shutil
from functools import cache
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .job import JOB_STORE, VideoJob, VideoJobStatus
from .media_probe import MediaProbeError, probe_video
from .transcription import FasterWhisperTranscriber
from .video_pipeline import VideoPipeline


BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class VideoUploadResponse(BaseModel):
    id: str
    filename: str
    stored_name: str
    size_bytes: int
    status: VideoJobStatus


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["GET", "POST"],
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


@app.get("/health")
def health_check():
    return {"status": "ok"}


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
    if not video.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename.",
        )

    original_name = video.filename
    suffix = Path(original_name).suffix.lower()

    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported video extension: "
                f"{suffix or 'none'}"
            ),
        )

    if (
        not video.content_type
        or not video.content_type.startswith("video/")
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported content type: "
                f"{video.content_type}"
            ),
        )

    video_id = uuid4().hex
    destination = UPLOAD_DIR / f"{video_id}{suffix}"

    try:
        with destination.open("wb") as output_file:
            shutil.copyfileobj(
                video.file,
                output_file,
            )

        probe_video(destination)

    except MediaProbeError as exc:
        destination.unlink(missing_ok=True)

        raise HTTPException(
            status_code=(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
            ),
            detail="Uploaded file is not a valid video.",
        ) from exc

    except Exception:
        destination.unlink(missing_ok=True)
        raise

    job = VideoJob(
        id=video_id,
        video_path=destination,
        status=VideoJobStatus.uploaded,
    )

    JOB_STORE[job.id] = job

    return VideoUploadResponse(
        id=video_id,
        filename=original_name,
        stored_name=destination.name,
        size_bytes=destination.stat().st_size,
        status=job.status,
    )


@app.get(
    "/jobs/{job_id}",
    response_model=VideoJob,
)
def get_job(job_id: str) -> VideoJob:
    job = JOB_STORE.get(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found.",
        )

    return job


@app.post(
    "/jobs/{job_id}/run",
    response_model=VideoJob,
)
def run_job(job_id: str) -> VideoJob:
    job = JOB_STORE.get(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found.",
        )

    if job.status != VideoJobStatus.uploaded:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Job cannot run from status: "
                f"{job.status.value}"
            ),
        )

    job.error_message = None

    try:
        pipeline = get_video_pipeline()

        pipeline.process(
            video_path=job.video_path,
            artifact_root=DATA_DIR,
            job=job,
        )

    except Exception as exc:
        job.status = VideoJobStatus.failed
        job.error_message = str(exc)

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail="Video processing failed.",
        ) from exc

    return job