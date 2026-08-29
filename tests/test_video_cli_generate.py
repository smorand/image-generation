"""End-to-end tests for `video-gen generate` with a fake pipeline (real MP4 encode)."""

import json

from PIL import Image
from typer.testing import CliRunner

import video_gen.pipeline as pipeline_mod
from video_gen.cli import app

runner = CliRunner()


class _FakePipeline:
    """Stand-in for VideoPipeline: returns solid-colour frames, no model."""

    def __init__(self, backend, repo_override=None, offload=False, precision=None):
        self.backend = backend
        self.repo = repo_override or f"fake/{backend}"
        self.device = "cpu"
        self.dtype = "bf16"
        self.calls = 0

    def load(self):
        pass

    def generate(self, config):
        self.calls += 1
        return [Image.new("RGB", (config.width, config.height), (10, 20, 30)) for _ in range(config.num_frames)]


def test_generate_writes_mp4_and_sidecar(tmp_path, monkeypatch):
    seed_img = tmp_path / "seed.png"
    Image.new("RGB", (320, 240), (10, 20, 30)).save(seed_img)
    out = tmp_path / "clip.mp4"

    fake = _FakePipeline("ltx")
    monkeypatch.setattr(pipeline_mod, "VideoPipeline", lambda **kw: fake)

    result = runner.invoke(
        app,
        [
            "generate",
            "-i",
            str(seed_img),
            "-p",
            "a cat waking up",
            "-o",
            str(out),
            "-d",
            "0.5",
            "-W",
            "128",
            "-H",
            "128",
            "--seed",
            "42",
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake.calls == 1
    assert out.exists()
    sidecar = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["seed"] == 42
    assert sidecar["backend"] == "ltx"


def test_generate_unknown_backend_rejected(tmp_path, monkeypatch):
    seed_img = tmp_path / "seed.png"
    Image.new("RGB", (128, 128), (0, 0, 0)).save(seed_img)
    out = tmp_path / "clip.mp4"

    result = runner.invoke(
        app,
        ["generate", "-i", str(seed_img), "-p", "move", "-o", str(out), "-b", "bogus"],
    )

    assert result.exit_code == 1
    assert "unknown backend" in result.output


def test_generate_unknown_precision_rejected(tmp_path, monkeypatch):
    seed_img = tmp_path / "seed.png"
    Image.new("RGB", (128, 128), (0, 0, 0)).save(seed_img)
    out = tmp_path / "clip.mp4"

    result = runner.invoke(
        app,
        ["generate", "-i", str(seed_img), "-p", "move", "-o", str(out), "--precision", "bogus"],
    )

    assert result.exit_code == 1
    assert "unknown precision" in result.output


def test_generate_random_seed_when_not_given(tmp_path, monkeypatch):
    seed_img = tmp_path / "seed.png"
    Image.new("RGB", (128, 128), (0, 0, 0)).save(seed_img)
    out = tmp_path / "clip.mp4"

    fake = _FakePipeline("ltx")
    monkeypatch.setattr(pipeline_mod, "VideoPipeline", lambda **kw: fake)

    result = runner.invoke(
        app,
        ["generate", "-i", str(seed_img), "-p", "move", "-o", str(out), "-d", "0.5", "-W", "128", "-H", "128"],
    )

    assert result.exit_code == 0, result.output
    assert "seed" in result.output
