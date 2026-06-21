from app.media_metadata import extract_video_metadata


def test_extract_video_metadata():
    raw_data = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
            },
        ],
        "format": {
            "duration": "10.5",
        },
    }

    metadata = extract_video_metadata(raw_data)

    assert metadata.duration_seconds == 10.5
    assert metadata.width == 1920
    assert metadata.height == 1080
    assert metadata.video_codec == "h264"
    assert metadata.has_audio is True