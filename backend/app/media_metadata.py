from pydantic import BaseModel, ValidationError

from .media_probe import MediaProbeError


class VideoMetadata(BaseModel):
    duration_seconds: float
    width: int
    height: int
    video_codec: str
    has_audio: bool


def extract_video_metadata(data: dict) -> VideoMetadata:
    streams = data.get("streams", [])
    format_data = data.get("format", {})

    video_stream = None
    has_audio = False

    for stream in streams:
        stream_type = stream.get("codec_type")

        if stream_type == "video" and video_stream is None:
            video_stream = stream

        if stream_type == "audio":
            has_audio = True

    if video_stream is None:
        raise MediaProbeError("media file contains no video stream")

    try:
        return VideoMetadata(
            duration_seconds=format_data["duration"],
            width=video_stream["width"],
            height=video_stream["height"],
            video_codec=video_stream["codec_name"],
            has_audio=has_audio,
        )
    except (KeyError, ValidationError) as exc:
        raise MediaProbeError(
            "ffprobe returned incomplete video metadata"
        ) from exc