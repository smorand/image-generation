"""Tests for the JSONL generation log."""

import json
from datetime import datetime

from image_gen.logging_jsonl import append_generation_log, log_file_for
from image_gen.metadata import GenerationMetadata


def _meta(**over):
    base = dict(
        prompt="a girl, (petite:1.3)",
        negative_prompt="low quality, (muscular:1.3)",
        model="m.safetensors",
        vae=None,
        seed=42,
        width=1024,
        height=1024,
        steps=40,
        cfg_scale=5.0,
        scheduler="euler_a",
        clip_skip=2,
        lora=None,
        embedding=None,
        hires_fix=False,
        hires_scale=None,
        hires_steps=None,
        hires_denoising=None,
        variables={"sizefix": "(petite:1.3)", "massfix": "(muscular:1.3)"},
    )
    base.update(over)
    return GenerationMetadata(**base)


def test_log_file_name_is_daily():
    p = log_file_for("/tmp/logs", datetime(2025, 2, 3, 12, 0, 0))
    assert p.name == "generations-2025-02-03.jsonl"
    assert str(p.parent) == "/tmp/logs"


def test_append_writes_full_record(tmp_path):
    meta = _meta()
    path = append_generation_log(tmp_path, meta, tmp_path / "out.png", command="generate")
    assert path is not None and path.exists()
    line = path.read_text(encoding="utf-8").strip()
    rec = json.loads(line)
    # Traceability fields.
    assert rec["command"] == "generate"
    assert rec["output"].endswith("out.png")
    assert "timestamp" in rec
    # Full metadata is merged in (nothing lost).
    assert rec["prompt"] == "a girl, (petite:1.3)"
    assert rec["negative_prompt"] == "low quality, (muscular:1.3)"
    assert rec["seed"] == 42
    assert rec["cfg_scale"] == 5.0
    assert rec["scheduler"] == "euler_a"
    assert rec["variables"] == {"sizefix": "(petite:1.3)", "massfix": "(muscular:1.3)"}
    # None fields are dropped (vae/lora/embedding).
    assert "vae" not in rec
    assert "lora" not in rec


def test_append_is_appendonly(tmp_path):
    meta = _meta()
    append_generation_log(tmp_path, meta, tmp_path / "a.png", command="generate-var", number="0000000001")
    append_generation_log(tmp_path, meta, tmp_path / "b.png", command="generate-var", number="0000000002")
    path = log_file_for(tmp_path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["number"] == "0000000001"
    assert json.loads(lines[1])["number"] == "0000000002"


def test_none_log_dir_disables(tmp_path):
    result = append_generation_log(None, _meta(), tmp_path / "x.png", command="generate")
    assert result is None
    # No files created.
    assert list(tmp_path.iterdir()) == []
