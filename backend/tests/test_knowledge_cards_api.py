from fastapi.testclient import TestClient

import app.main as main
from app.job import VideoJob, VideoJobStatus
from app.job_store import create_job


client = TestClient(main.app)


def create_uploaded_job(tmp_path, job_id: str = "job-123") -> VideoJob:
    job = VideoJob(
        id=job_id,
        video_path=tmp_path / "lecture.mp4",
        status=VideoJobStatus.completed,
        original_filename="lecture.mp4",
    )
    create_job(job)

    return job


def card_payload() -> dict:
    return {
        "title": "Linear Algebra",
        "summary": "Linear algebra is one of the course's core subjects.",
        "key_points": [
            "Symmetric matrices",
            "Orthogonal matrices",
        ],
        "question": "What is the first big subject?",
        "answer": "Linear algebra.",
        "difficulty": "easy",
        "source_start_seconds": 36.44,
        "source_end_seconds": 92.68,
        "provider": "ollama",
        "model": "qwen3:4b",
    }


def test_save_and_list_job_cards(tmp_path):
    job = create_uploaded_job(tmp_path)

    create_response = client.post(
        f"/jobs/{job.id}/cards",
        json=card_payload(),
    )

    assert create_response.status_code == 201

    created_card = create_response.json()

    assert created_card["id"]
    assert created_card["job_id"] == job.id
    assert created_card["title"] == "Linear Algebra"
    assert created_card["key_points"] == [
        "Symmetric matrices",
        "Orthogonal matrices",
    ]
    assert created_card["created_at"]
    assert created_card["updated_at"]

    list_response = client.get(
        f"/jobs/{job.id}/cards"
    )

    assert list_response.status_code == 200
    assert [
        card["id"]
        for card in list_response.json()
    ] == [created_card["id"]]


def test_update_saved_card(tmp_path):
    job = create_uploaded_job(tmp_path)
    created_card = client.post(
        f"/jobs/{job.id}/cards",
        json=card_payload(),
    ).json()

    response = client.patch(
        f"/cards/{created_card['id']}",
        json={
            "title": "Singular Value Decomposition",
            "summary": "SVD is a critical matrix factorization.",
            "key_points": [
                "Matrix factorization",
                "Orthogonal matrices",
            ],
            "difficulty": "medium",
        },
    )

    assert response.status_code == 200

    updated_card = response.json()

    assert updated_card["id"] == created_card["id"]
    assert updated_card["title"] == "Singular Value Decomposition"
    assert updated_card["summary"] == (
        "SVD is a critical matrix factorization."
    )
    assert updated_card["key_points"] == [
        "Matrix factorization",
        "Orthogonal matrices",
    ]
    assert updated_card["difficulty"] == "medium"
    assert updated_card["updated_at"] >= created_card["updated_at"]


def test_delete_saved_card(tmp_path):
    job = create_uploaded_job(tmp_path)
    created_card = client.post(
        f"/jobs/{job.id}/cards",
        json=card_payload(),
    ).json()

    delete_response = client.delete(
        f"/cards/{created_card['id']}"
    )

    assert delete_response.status_code == 204

    list_response = client.get(
        f"/jobs/{job.id}/cards"
    )

    assert list_response.status_code == 200
    assert list_response.json() == []


def test_save_card_returns_404_for_missing_job():
    response = client.post(
        "/jobs/missing-job/cards",
        json=card_payload(),
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Job not found."
    }


def test_save_card_returns_400_for_invalid_time_range(tmp_path):
    job = create_uploaded_job(tmp_path)
    payload = card_payload()
    payload["source_end_seconds"] = payload["source_start_seconds"]

    response = client.post(
        f"/jobs/{job.id}/cards",
        json=payload,
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Card source end must be greater than start."
    }


def test_update_card_returns_404_for_missing_card():
    response = client.patch(
        "/cards/missing-card",
        json={
            "title": "Missing",
        },
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Knowledge card not found."
    }


def test_list_cards_returns_404_for_missing_job():
    response = client.get(
        "/jobs/missing-job/cards"
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Job not found."
    }
