"""Tests for video generation metadata (JSON sidecar)."""

import json
from datetime import datetime

from video_gen.metadata import VideoMetadata, write_sidecar


def _meta(**kw):
    base = dict(
        prompt="a cat waking up",
        negative_prompt="blurry",
        backend="ltx",
        model="Lightricks/LTX-Video",
        source_image="seed.png",
        seed=42,
        width=768,
        height=512,
        num_frames=73,
        fps=24,
        duration_s=3.0,
        steps=30,
        guidance=3.0,
    )
    base.update(kw)
    return VideoMetadata(**base)


def test_to_json_is_single_line():
    meta = _meta(variables={"subject": "a cat"})
    js = meta.to_json()
    assert "\n" not in js
    data = json.loads(js)
    assert data["variables"]["subject"] == "a cat"
    assert data["seed"] == 42


def test_generated_at_is_stamped_automatically():
    meta = _meta()
    datetime.fromisoformat(meta.generated_at)


def test_generated_at_can_be_overridden_explicitly():
    meta = _meta(generated_at="2020-01-01T00:00:00")
    assert meta.generated_at == "2020-01-01T00:00:00"


def test_write_sidecar_includes_generated_at(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake")
    meta = _meta(generated_at="2026-08-29T14:30:05")

    sidecar = write_sidecar(clip, meta)

    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["generated_at"] == "2026-08-29T14:30:05"
