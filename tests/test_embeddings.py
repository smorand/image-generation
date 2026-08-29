"""Tests for textual inversion embedding loading (no model needed)."""

import pytest

from image_gen.embeddings import load_embeddings


class _FakePipeline:
    def __init__(self):
        self.calls: list[dict] = []

    def load_textual_inversion(self, path, token):
        self.calls.append({"path": path, "token": token})


def test_load_embeddings_noop_on_empty_list():
    pipeline = _FakePipeline()
    assert load_embeddings(pipeline, []) == []
    assert pipeline.calls == []


def test_load_embeddings_loads_each_and_returns_token_names(tmp_path):
    emb_a = tmp_path / "my_style.safetensors"
    emb_b = tmp_path / "another_token.pt"
    emb_a.write_bytes(b"fake")
    emb_b.write_bytes(b"fake")

    pipeline = _FakePipeline()
    tokens = load_embeddings(pipeline, [str(emb_a), str(emb_b)])

    assert tokens == ["my_style", "another_token"]
    assert pipeline.calls[0] == {"path": str(emb_a), "token": "my_style"}
    assert pipeline.calls[1] == {"path": str(emb_b), "token": "another_token"}


def test_load_embeddings_missing_file_raises(tmp_path):
    pipeline = _FakePipeline()
    with pytest.raises(FileNotFoundError, match="Embedding file not found"):
        load_embeddings(pipeline, [str(tmp_path / "missing.safetensors")])
