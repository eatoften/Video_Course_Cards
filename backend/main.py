from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

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


@app.post("/videos/inspect")
async def inspect_video(video:UploadFile):
    return {
        "filename": video.filename,
        "content_type": video.content_type,
        "size_bytes": video.size,
    }