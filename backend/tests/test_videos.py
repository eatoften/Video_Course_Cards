from fastapi.testclient import TestClient

import app.main as main
import app.job_service as job_service


client = TestClient(main.app)


def test_upload_rejects_unsupported_extension():
    files = {
        "video": ("notes.txt", b"not a real video", "text/plain")
    }

    response = client.post("/videos", files=files)

    assert response.status_code == 415
    assert response.json() == {
        "detail": "Unsupported video extension: .txt"
    }

def test_upload_video_saves_file(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)

    def fake_probe_video(file_path):
        assert file_path.exists()
        return {"streams": [{"codec_type": "video"}]}

    monkeypatch.setattr(job_service, "probe_video", fake_probe_video)

    video_content = b"fake video content"
    files = {
        "video": ("lecture.mp4", video_content, "video/mp4")
    }

    response = client.post("/videos", files=files)

    assert response.status_code == 201

    data = response.json()

    assert data["filename"] == "lecture.mp4"
    assert data["size_bytes"] == len(video_content)
    assert data["id"]

    saved_file = tmp_path / data["stored_name"]

    assert saved_file.exists()
    assert saved_file.read_bytes() == video_content

def test_upload_rejects_invalid_video_content(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)

    def fake_probe_video(file_path):
        raise job_service.MediaProbeError("moov atom not found")

    monkeypatch.setattr(job_service, "probe_video", fake_probe_video)

    files = {
        "video": (
            "fake.mp4",
            b"not a real video",
            "video/mp4",
        )
    }

    response = client.post("/videos", files=files)

    assert response.status_code == 415
    assert response.json() == {
        "detail": "Uploaded file is not a valid video."
    }

    assert list(tmp_path.iterdir()) == []
