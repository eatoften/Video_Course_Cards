import subprocess
from pathlib import Path

class AudioExtractionError(Exception):
    """FFmpeg 无法从视频中提取音频时抛出"""



def extract_audio(video_path:Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    if result.returncode !=0:
        output_path.unlink(missing_ok=True)

        error_message = result.stderr.strip()

        raise AudioExtractionError(
            error_message or "FFmpeg failed to extract audio"
        )
    
    if not output_path.exists():
        raise AudioExtractionError(
            "FFmpeg reported success but did not create an output file"
        )
    
    return output_path



