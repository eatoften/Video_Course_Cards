import shutil
from pathlib import Path
from uuid import uuid4


from fastapi import FastAPI, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from .media_probe import MediaProbeError, probe_video
from .job import JOB_STORE, VideoJob, VideoJobStatus

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

class VideoUploadResponse(BaseModel):
    id:str
    filename: str
    stored_name: str
    size_bytes: int
    status: VideoJobStatus

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_methods=["GET","POST"],
    allow_headers=["*"]
)

@app.get("/health")
def health_check():
    return {"status":"ok"}


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



@app.post("/videos/inspect")
async def inspect_video(video:UploadFile):
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
def upload_video(video:UploadFile) -> VideoUploadResponse:
    if not video.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename.",
        )

    original_name = video.filename 
    suffix = Path(original_name).suffix.lower()

    video_id = uuid4().hex
    destination = UPLOAD_DIR / f"{video_id}{suffix}"

    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported video extension: {suffix or 'none'}",
        )

    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type: {video.content_type}",
        )
    try:
        with destination.open("wb") as output_file:
            shutil.copyfileobj(video.file,output_file)
        
        probe_video(destination)

    except MediaProbeError as exc:
        destination.unlink(missing_ok=True)

        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Uploaded file is not a valid video.",
        ) from exc
    
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    job = VideoJob(
        id = video_id,
        video_path=destination,
        status=VideoJobStatus.uploaded
    )

    JOB_STORE[job.id] = job

    return VideoUploadResponse(
        id = video_id,
        filename = original_name,
        stored_name = destination.name,
        size_bytes = destination.stat().st_size,
        status = job.status
    )