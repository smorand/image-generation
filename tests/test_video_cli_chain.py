"""End-to-end test of `video-gen chain` with a fake pipeline (real MP4 encode)."""

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

    def empty_cache(self):
        pass

    def generate(self, config):
        self.calls += 1
        # One distinct colour per segment so the frames are real and same-sized.
        shade = (self.calls * 40) % 255
        return [Image.new("RGB", (config.width, config.height), (shade, 0, 0)) for _ in range(config.num_frames)]


def test_chain_command_writes_mp4_and_sidecar(tmp_path, monkeypatch):
    seed_img = tmp_path / "seed.png"
    Image.new("RGB", (320, 240), (10, 20, 30)).save(seed_img)
    out = tmp_path / "clip.mp4"

    fake = _FakePipeline("ltx")
    monkeypatch.setattr(pipeline_mod, "VideoPipeline", lambda **kw: fake)

    result = runner.invoke(
        app,
        [
            "chain",
            "-i",
            str(seed_img),
            "-p",
            "first move",
            "-p",
            "second move",
            "-o",
            str(out),
            "--seg-duration",
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
    assert fake.calls == 2  # one generate() per segment
    assert out.exists()

    sidecar = out.with_suffix(".json")
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["segments"] == 2
    assert meta["prompt_series"] == ["first move", "second move"]
    assert meta["seed"] == 42
    assert meta["backend"] == "ltx"


def test_chain_command_pads_prompts_to_segments(tmp_path, monkeypatch):
    seed_img = tmp_path / "seed.png"
    Image.new("RGB", (128, 128), (0, 0, 0)).save(seed_img)
    out = tmp_path / "clip.mp4"

    fake = _FakePipeline("ltx")
    monkeypatch.setattr(pipeline_mod, "VideoPipeline", lambda **kw: fake)

    result = runner.invoke(
        app,
        [
            "chain",
            "-i",
            str(seed_img),
            "-p",
            "only prompt",
            "--segments",
            "3",
            "-o",
            str(out),
            "--seg-duration",
            "0.5",
            "-W",
            "128",
            "-H",
            "128",
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake.calls == 3  # padded to 3 segments
    meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["prompt_series"] == ["only prompt", "only prompt", "only prompt"]
