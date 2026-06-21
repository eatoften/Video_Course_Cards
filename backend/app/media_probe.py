import json
import subprocess
from pathlib import Path


class MediaProbeError(Exception):
    """ffprobe 无法正确解析媒体文件时抛出的异常。"""


def run_ffprobe(file_path: Path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        (
            "format=filename,format_name,duration,size,bit_rate:"
            "stream=index,codec_type,codec_name,width,height,"
            "avg_frame_rate,sample_rate,channels"
        ),
        "-of",
        "json",
        str(file_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    return result


def probe_media(file_path: Path) -> dict:
    result = run_ffprobe(file_path)

    if result.returncode != 0:
        error_message = result.stderr.strip()
        raise MediaProbeError(error_message or "ffprobe failed")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaProbeError("ffprobe returned invalid JSON") from exc

    return data


if __name__ == "__main__":
    video_path = Path(r"C:\Users\12245\Desktop\录屏.mp4")

    try:
        media_data = probe_media(video_path)

        print("解析成功")
        print(type(media_data))
        print(media_data["format"]["duration"])
        print(media_data["streams"][0]["codec_type"])

    except MediaProbeError as exc:
        print("解析失败：")
        print(exc)



def probe_video(file_path: Path) -> dict:
    data = probe_media(file_path)

    streams = data.get("streams", [])

    has_video_stream = any(
        stream.get("codec_type") == "video"
        for stream in streams
    )

    if not has_video_stream:
        raise MediaProbeError("media file contains no video stream")

    return data





