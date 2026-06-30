import json
from io import BytesIO
from zipfile import ZipFile

from fastapi.testclient import TestClient

import app.main as main
import app.export_service as export_service
from app.course import DEFAULT_COURSE_ID
from app.job import VideoJob, VideoJobStatus
from app.job_store import create_job
from app.knowledge_card import KnowledgeCard
from app.knowledge_card_store import create_card


client = TestClient(main.app)


def create_course(title: str = "MIT 18.065") -> dict:
    response = client.post(
        "/courses",
        json={
            "title": title,
            "description": "Matrix methods and deep learning",
        },
    )

    assert response.status_code == 201

    return response.json()


def create_uploaded_job(
    tmp_path,
    *,
    job_id: str,
    course_id: str = DEFAULT_COURSE_ID,
    filename: str = "lecture 01.mp4",
) -> VideoJob:
    job = VideoJob(
        id=job_id,
        course_id=course_id,
        video_path=tmp_path / filename,
        status=VideoJobStatus.completed,
        original_filename=filename,
        stored_name=f"{job_id}.mp4",
    )
    create_job(job)

    return job


def create_svd_card(job_id: str, card_id: str = "card-svd") -> KnowledgeCard:
    card = KnowledgeCard(
        id=card_id,
        job_id=job_id,
        title="Singular Value Decomposition",
        summary="SVD factors a matrix using orthogonal structure.",
        key_points=[
            "Orthogonal matrices are part of the factorization.",
            "The diagonal structure stores singular values.",
        ],
        claims=[
            {
                "text": (
                    "SVD factors a matrix using orthogonal and diagonal "
                    "structure."
                ),
                "evidence": [
                    {
                        "quote": (
                            "orthogonal times diagonal times orthogonal "
                            "matrix"
                        ),
                        "segment_start_seconds": 724.0,
                        "segment_end_seconds": 738.0,
                    }
                ],
            }
        ],
        unsupported_terms=[],
        question="What structures appear in SVD?",
        answer="Orthogonal and diagonal matrix structure.",
        tags=[
            "linear algebra",
            "svd",
        ],
        review_state="reviewed",
        source_start_seconds=724.0,
        source_end_seconds=738.0,
        provider="ollama",
        model="qwen3:4b",
    )
    create_card(card)

    return card


def open_zip(response) -> ZipFile:
    return ZipFile(BytesIO(response.content))


def test_export_job_cards_as_markdown_zip(tmp_path):
    job = create_uploaded_job(tmp_path, job_id="job-svd")
    create_svd_card(job.id)

    response = client.get(f"/jobs/{job.id}/cards/export/markdown")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["content-disposition"] == (
        'attachment; filename="job-job-svd-cards.zip"'
    )

    with open_zip(response) as archive:
        names = archive.namelist()

        assert names == [
            "README.md",
            "cards/0001-singular-value-decomposition.md",
        ]

        markdown = archive.read(names[1]).decode("utf-8")

    assert "# Singular Value Decomposition" in markdown
    assert "- Video: lecture 01.mp4" in markdown
    assert "- Time: 12:04 - 12:18" in markdown
    assert "## Claims" in markdown
    assert (
        "- Claim: SVD factors a matrix using orthogonal and diagonal "
        "structure."
    ) in markdown
    assert (
        '  - Evidence: "orthogonal times diagonal times orthogonal matrix"'
    ) in markdown
    assert "    Source: 12:04 - 12:18" in markdown
    assert "## Active Recall" in markdown
    assert "Q: What structures appear in SVD?" in markdown
    assert "A: Orthogonal and diagonal matrix structure." in markdown


def test_export_all_cards_uses_obsidian_friendly_layout(tmp_path):
    course = create_course("MIT 18.065 / Matrix Methods")
    course_job = create_uploaded_job(
        tmp_path,
        job_id="course-job",
        course_id=course["id"],
        filename="lecture 01.mp4",
    )
    default_job = create_uploaded_job(
        tmp_path,
        job_id="default-job",
        filename="intro.mp4",
    )
    create_svd_card(course_job.id, "course-card")
    create_svd_card(default_job.id, "default-card")

    response = client.get("/cards/export/markdown")

    assert response.status_code == 200
    assert response.headers["content-disposition"] == (
        'attachment; filename="video-course-cards-vault.zip"'
    )

    with open_zip(response) as archive:
        names = archive.namelist()

        assert "README.md" in names
        assert (
            "MIT 18.065 Matrix Methods/lecture 01.mp4/"
            "0001-singular-value-decomposition.md"
        ) in names
        assert (
            "Uncategorized/intro.mp4/"
            "0002-singular-value-decomposition.md"
        ) in names


def test_export_job_cards_returns_readme_when_job_has_no_cards(tmp_path):
    job = create_uploaded_job(tmp_path, job_id="empty-job")

    response = client.get(f"/jobs/{job.id}/cards/export/markdown")

    assert response.status_code == 200

    with open_zip(response) as archive:
        assert archive.namelist() == ["README.md"]
        readme = archive.read("README.md").decode("utf-8")

    assert "- Cards: 0" in readme


def test_save_job_cards_export_to_local_directory(monkeypatch, tmp_path):
    export_dir = tmp_path / "cards"
    monkeypatch.setattr(export_service, "DEFAULT_EXPORT_DIR", export_dir)
    job = create_uploaded_job(tmp_path, job_id="job-local")
    create_svd_card(job.id)

    response = client.post(f"/jobs/{job.id}/cards/export/markdown/local")

    assert response.status_code == 200

    data = response.json()
    saved_path = export_dir / "job-job-local-cards.zip"

    assert data == {
        "filename": "job-job-local-cards.zip",
        "path": str(saved_path),
        "byte_count": saved_path.stat().st_size,
    }
    assert saved_path.is_file()

    with ZipFile(saved_path) as archive:
        assert "cards/0001-singular-value-decomposition.md" in (
            archive.namelist()
        )


def test_save_all_cards_export_to_local_directory(monkeypatch, tmp_path):
    export_dir = tmp_path / "cards"
    monkeypatch.setattr(export_service, "DEFAULT_EXPORT_DIR", export_dir)
    job = create_uploaded_job(tmp_path, job_id="job-vault")
    create_svd_card(job.id)

    response = client.post("/cards/export/markdown/local")

    assert response.status_code == 200

    data = response.json()
    saved_path = export_dir / "video-course-cards-vault.zip"

    assert data["filename"] == "video-course-cards-vault.zip"
    assert data["path"] == str(saved_path)
    assert data["byte_count"] == saved_path.stat().st_size
    assert saved_path.is_file()


def test_save_job_cards_export_to_markdown_folder(monkeypatch, tmp_path):
    export_dir = tmp_path / "cards"
    monkeypatch.setattr(export_service, "DEFAULT_EXPORT_DIR", export_dir)
    job = create_uploaded_job(
        tmp_path,
        job_id="job-folder",
        filename="lecture 01.mp4",
    )
    create_svd_card(job.id)

    response = client.post(
        f"/jobs/{job.id}/cards/export/markdown/folder"
    )

    assert response.status_code == 200

    data = response.json()
    root_path = export_dir / "lecture 01.mp4"
    markdown_path = root_path / "0001-singular-value-decomposition.md"

    assert data == {
        "root_path": str(root_path),
        "file_count": 1,
        "files": [
            "0001-singular-value-decomposition.md",
        ],
    }
    assert markdown_path.is_file()

    markdown = markdown_path.read_text(encoding="utf-8")

    assert "# Singular Value Decomposition" in markdown
    assert "- Video: lecture 01.mp4" in markdown
    assert "Q: What structures appear in SVD?" in markdown


def test_save_all_cards_export_to_obsidian_folder(monkeypatch, tmp_path):
    export_dir = tmp_path / "cards"
    monkeypatch.setattr(export_service, "DEFAULT_EXPORT_DIR", export_dir)
    course = create_course("MIT 18.065 / Matrix Methods")
    course_job = create_uploaded_job(
        tmp_path,
        job_id="course-folder-job",
        course_id=course["id"],
        filename="lecture 01.mp4",
    )
    default_job = create_uploaded_job(
        tmp_path,
        job_id="default-folder-job",
        filename="intro.mp4",
    )
    create_svd_card(course_job.id, "course-folder-card")
    create_svd_card(default_job.id, "default-folder-card")

    response = client.post("/cards/export/markdown/folder")

    assert response.status_code == 200

    data = response.json()

    assert data["root_path"] == str(export_dir)
    assert data["file_count"] == 2
    assert (
        "MIT 18.065 Matrix Methods/lecture 01.mp4/"
        "0001-singular-value-decomposition.md"
    ) in data["files"]
    assert (
        "Uncategorized/intro.mp4/0002-singular-value-decomposition.md"
    ) in data["files"]
    assert (
        export_dir
        / "MIT 18.065 Matrix Methods"
        / "lecture 01.mp4"
        / "0001-singular-value-decomposition.md"
    ).is_file()
    assert (
        export_dir
        / "Uncategorized"
        / "intro.mp4"
        / "0002-singular-value-decomposition.md"
    ).is_file()


def test_folder_export_removes_previous_snapshot_files(
    monkeypatch,
    tmp_path,
):
    export_dir = tmp_path / "cards"
    monkeypatch.setattr(export_service, "DEFAULT_EXPORT_DIR", export_dir)
    job = create_uploaded_job(
        tmp_path,
        job_id="job-refresh",
        filename="lecture refresh.mp4",
    )
    create_svd_card(job.id, "refresh-card")

    first_response = client.post(
        f"/jobs/{job.id}/cards/export/markdown/folder"
    )

    assert first_response.status_code == 200

    old_path = (
        export_dir
        / "lecture refresh.mp4"
        / "old-generated-card.md"
    )
    old_path.write_text("stale generated snapshot", encoding="utf-8")
    manifest_path = (
        export_dir
        / "lecture refresh.mp4"
        / ".vcc-job-export-manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            {
                "files": [
                    "old-generated-card.md",
                ],
            }
        ),
        encoding="utf-8",
    )

    second_response = client.post(
        f"/jobs/{job.id}/cards/export/markdown/folder"
    )

    assert second_response.status_code == 200
    assert not old_path.exists()
    assert (
        export_dir
        / "lecture refresh.mp4"
        / "0001-singular-value-decomposition.md"
    ).is_file()


def test_export_job_cards_returns_404_for_missing_job():
    response = client.get("/jobs/missing-job/cards/export/markdown")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Job not found."
    }
