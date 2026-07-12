from fastapi.testclient import TestClient

import app.main as main
from app.job import VideoJob, VideoJobStatus
from app.job_store import create_job


client = TestClient(main.app)


def create_card(tmp_path) -> dict:
    job = VideoJob(
        id="review-job",
        video_path=tmp_path / "lecture.mp4",
        status=VideoJobStatus.completed,
    )
    create_job(job)
    return client.post(
        f"/jobs/{job.id}/cards",
        json={
            "title": "Gradient Descent",
            "summary": "Gradient descent minimizes a loss function.",
            "claims": [
                {
                    "text": "Gradient descent follows the negative gradient.",
                    "evidence": [
                        {
                            "quote": "move in the negative gradient direction",
                            "segment_start_seconds": 1.0,
                            "segment_end_seconds": 2.0,
                        }
                    ],
                }
            ],
            "source_start_seconds": 1.0,
            "source_end_seconds": 3.0,
        },
    ).json()


def test_review_item_crud(tmp_path):
    card = create_card(tmp_path)
    create_response = client.post(
        f"/cards/{card['id']}/review-items",
        json={
            "item_type": "explain",
            "prompt": "Why follow the negative gradient?",
            "expected_answer": "It points toward local loss reduction.",
            "source_claim_ids": [card["claims"][0]["id"]],
            "source": "manual",
        },
    )
    assert create_response.status_code == 201
    item = create_response.json()

    list_response = client.get(f"/cards/{card['id']}/review-items")
    assert list_response.status_code == 200
    assert [value["id"] for value in list_response.json()] == [item["id"]]

    patch_response = client.patch(
        f"/review-items/{item['id']}",
        json={"status": "disabled", "item_type": "apply"},
    )
    assert patch_response.status_code == 200
    assert patch_response.json()["status"] == "disabled"
    assert patch_response.json()["item_type"] == "apply"

    delete_response = client.delete(f"/review-items/{item['id']}")
    assert delete_response.status_code == 204
    assert client.get(f"/cards/{card['id']}/review-items").json() == []


def test_review_queue_and_fsrs_rating(tmp_path):
    card = create_card(tmp_path)
    item = client.post(
        f"/cards/{card['id']}/review-items",
        json={
            "item_type": "basic",
            "prompt": "What direction does gradient descent follow?",
            "expected_answer": "The negative gradient direction.",
            "source": "manual",
        },
    ).json()

    queue_response = client.get("/courses/uncategorized/review/queue")
    assert queue_response.status_code == 200
    queue = queue_response.json()
    assert queue["due_count"] == 1
    assert queue["new_count"] == 1
    assert queue["items"][0]["review_item"]["id"] == item["id"]
    assert queue["items"][0]["phase"] == "new"

    rating_response = client.post(
        f"/review-items/{item['id']}/rate",
        json={"rating": "good", "response_time_ms": 4200},
    )
    assert rating_response.status_code == 200
    result = rating_response.json()
    assert result["progress"]["review_count"] == 1
    assert result["progress"]["due_at"] > result["event"]["reviewed_at"]
    assert result["event"]["rating"] == "good"
    assert result["event"]["response_time_ms"] == 4200
