from fastapi.testclient import TestClient

import app.main as main
from app.card_embedding import CardEmbedding
from app.card_embedding_store import upsert_card_embeddings
from app.course import DEFAULT_COURSE_ID
from app.job import VideoJob, VideoJobStatus
from app.job_store import create_job
from app.settings import LLMSettings


client = TestClient(main.app)


class FakeRelationLLMClient:
    def __init__(self, output: str) -> None:
        self.output = output
        self.settings = LLMSettings(
            provider="ollama",
            base_url="http://127.0.0.1:11434/v1",
            api_key="ollama",
            model="qwen3:4b",
            timeout_seconds=30,
            temperature=0.0,
            max_tokens=2048,
        )

    def create_chat_completion(self, messages, **kwargs) -> str:
        return self.output


def create_uploaded_job(tmp_path, job_id: str = "job-123") -> VideoJob:
    job = VideoJob(
        id=job_id,
        video_path=tmp_path / "lecture.mp4",
        status=VideoJobStatus.completed,
        original_filename="lecture.mp4",
    )
    create_job(job)

    return job


def card_payload(title: str) -> dict:
    return {
        "title": title,
        "summary": f"{title} summary.",
        "key_points": [
            f"{title} key point.",
        ],
        "claims": [
            {
                "text": f"{title} claim.",
                "evidence": [
                    {
                        "quote": f"{title} evidence.",
                        "segment_start_seconds": 10.0,
                        "segment_end_seconds": 15.0,
                    }
                ],
            }
        ],
        "unsupported_terms": [],
        "question": f"What is {title}?",
        "answer": f"{title}.",
        "difficulty": "easy",
        "tags": [
            title.lower(),
        ],
        "source_start_seconds": 10.0,
        "source_end_seconds": 20.0,
        "provider": "ollama",
        "model": "qwen3:4b",
    }


def save_card(job_id: str, title: str) -> dict:
    response = client.post(
        f"/jobs/{job_id}/cards",
        json=card_payload(title),
    )

    assert response.status_code == 201

    return response.json()


def make_embedding(card_id: str, vector: list[float]) -> CardEmbedding:
    return CardEmbedding(
        card_id=card_id,
        model="sentence-transformers/all-MiniLM-L6-v2",
        dimension=len(vector),
        text_hash=f"hash-{card_id}",
        vector=vector,
    )


def test_recompute_card_relations_and_get_related_cards(tmp_path):
    job = create_uploaded_job(tmp_path)
    linear_card = save_card(job.id, "Linear Algebra")
    svd_card = save_card(job.id, "SVD")
    statistics_card = save_card(job.id, "Statistics")
    upsert_card_embeddings(
        [
            make_embedding(linear_card["id"], [1.0, 0.0]),
            make_embedding(svd_card["id"], [0.9, 0.1]),
            make_embedding(statistics_card["id"], [0.0, 1.0]),
        ]
    )

    recompute_response = client.post(
        f"/courses/{DEFAULT_COURSE_ID}/card-relations/recompute",
        json={
            "threshold": 0.7,
            "top_k": 1,
        },
    )

    assert recompute_response.status_code == 200
    assert recompute_response.json() == {
        "course_id": DEFAULT_COURSE_ID,
        "total_cards": 3,
        "embedded_cards": 3,
        "skipped_cards": 0,
        "relations_written": 2,
        "threshold": 0.7,
        "top_k": 1,
    }

    related_response = client.get(f"/cards/{linear_card['id']}/related")

    assert related_response.status_code == 200
    related_payload = related_response.json()
    assert related_payload["card_id"] == linear_card["id"]
    assert len(related_payload["related"]) == 1
    assert related_payload["related"][0]["card_id"] == svd_card["id"]
    assert related_payload["related"][0]["score"] > 0.99


def test_course_card_relations_graph_update_and_delete(tmp_path):
    job = create_uploaded_job(tmp_path)
    first_card = save_card(job.id, "Backpropagation")
    second_card = save_card(job.id, "Gradient Descent")
    upsert_card_embeddings(
        [
            make_embedding(first_card["id"], [1.0, 0.0]),
            make_embedding(second_card["id"], [0.9, 0.1]),
        ]
    )
    client.post(
        f"/courses/{DEFAULT_COURSE_ID}/card-relations/recompute",
        json={
            "threshold": 0.7,
            "top_k": 1,
        },
    )

    graph_response = client.get(
        f"/courses/{DEFAULT_COURSE_ID}/card-relations"
    )

    assert graph_response.status_code == 200
    graph_payload = graph_response.json()
    assert {
        node["id"]
        for node in graph_payload["nodes"]
    } == {
        first_card["id"],
        second_card["id"],
    }
    assert len(graph_payload["edges"]) == 2

    relation_id = graph_payload["edges"][0]["id"]
    patch_response = client.patch(
        f"/card-relations/{relation_id}",
        json={
            "status": "accepted",
            "explanation": "Both cards describe optimization mechanics.",
        },
    )

    assert patch_response.status_code == 200
    assert patch_response.json()["status"] == "accepted"
    assert (
        patch_response.json()["explanation"]
        == "Both cards describe optimization mechanics."
    )

    delete_response = client.delete(f"/card-relations/{relation_id}")

    assert delete_response.status_code == 204


def test_related_cards_returns_404_for_missing_card():
    response = client.get("/cards/missing-card/related")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Knowledge card not found."
    }


def test_create_manual_relation_and_reject_duplicate(tmp_path):
    job = create_uploaded_job(tmp_path)
    source_card = save_card(job.id, "Convolution")
    target_card = save_card(job.id, "Image Filter")
    payload = {
        "source_card_id": source_card["id"],
        "target_card_id": target_card["id"],
        "relation_type": "example_of",
        "explanation": "Convolution applies an image filter.",
    }

    response = client.post(
        f"/courses/{DEFAULT_COURSE_ID}/card-relations",
        json=payload,
    )

    assert response.status_code == 201
    assert response.json()["method"] == "manual"
    assert response.json()["status"] == "accepted"
    assert response.json()["score"] == 1.0

    duplicate_response = client.post(
        f"/courses/{DEFAULT_COURSE_ID}/card-relations",
        json=payload,
    )

    assert duplicate_response.status_code == 400
    assert duplicate_response.json() == {
        "detail": "This manual card relation already exists."
    }


def test_classify_similarity_relation_with_local_llm(tmp_path, monkeypatch):
    job = create_uploaded_job(tmp_path)
    source_card = save_card(job.id, "Linear Algebra")
    target_card = save_card(job.id, "Singular Value Decomposition")
    upsert_card_embeddings(
        [
            make_embedding(source_card["id"], [1.0, 0.0]),
            make_embedding(target_card["id"], [0.9, 0.1]),
        ]
    )
    client.post(
        f"/courses/{DEFAULT_COURSE_ID}/card-relations/recompute",
        json={"threshold": 0.7, "top_k": 1},
    )
    graph = client.get(
        f"/courses/{DEFAULT_COURSE_ID}/card-relations"
    ).json()
    source_relation = next(
        edge
        for edge in graph["edges"]
        if edge["source"] == source_card["id"]
    )
    fake_client = FakeRelationLLMClient(
        '{"relation_type":"prerequisite",'
        '"explanation":"Linear algebra is needed before studying SVD."}'
    )
    monkeypatch.setattr(main, "get_llm_client", lambda: fake_client)

    response = client.post(
        f"/card-relations/{source_relation['id']}/classify",
        json={"model": "qwen3:4b"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["classification"] == "prerequisite"
    assert result["relation"]["method"] == "local_llm"
    assert result["relation"]["status"] == "suggested"

    updated_graph = client.get(
        f"/courses/{DEFAULT_COURSE_ID}/card-relations"
    ).json()
    edges_by_id = {edge["id"]: edge for edge in updated_graph["edges"]}
    assert edges_by_id[source_relation["id"]]["status"] == "hidden"
    assert any(
        edge["relation_type"] == "prerequisite"
        and edge["method"] == "local_llm"
        for edge in updated_graph["edges"]
    )
