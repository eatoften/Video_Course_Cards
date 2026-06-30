from pathlib import Path

import pytest

from app.embedding import (
    EmbeddingError,
    resolve_sentence_transformer_model_source,
)


def _write_snapshot(snapshot: Path, *, complete: bool) -> None:
    snapshot.mkdir(parents=True)
    (snapshot / "modules.json").write_text("[]", encoding="utf-8")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    if complete:
        (snapshot / "model.safetensors").write_bytes(b"weights")


def test_resolves_complete_huggingface_snapshot(monkeypatch, tmp_path):
    hf_home = tmp_path / "hf"
    snapshots_root = (
        hf_home
        / "hub"
        / "models--sentence-transformers--all-MiniLM-L6-v2"
        / "snapshots"
    )
    incomplete_snapshot = snapshots_root / "incomplete"
    complete_snapshot = snapshots_root / "complete"
    _write_snapshot(incomplete_snapshot, complete=False)
    _write_snapshot(complete_snapshot, complete=True)
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)

    model_source = resolve_sentence_transformer_model_source(
        "sentence-transformers/all-MiniLM-L6-v2",
        local_files_only=True,
    )

    assert model_source == complete_snapshot


def test_explicit_incomplete_model_path_fails_fast(tmp_path):
    incomplete_snapshot = tmp_path / "incomplete"
    _write_snapshot(incomplete_snapshot, complete=False)

    with pytest.raises(EmbeddingError, match="incomplete"):
        resolve_sentence_transformer_model_source(
            "sentence-transformers/all-MiniLM-L6-v2",
            model_path=incomplete_snapshot,
            local_files_only=True,
        )


def test_missing_local_model_fails_fast(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)

    with pytest.raises(EmbeddingError, match="Local SentenceTransformer"):
        resolve_sentence_transformer_model_source(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True,
        )
