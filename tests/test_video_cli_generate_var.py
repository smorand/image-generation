"""End-to-end tests for `video-gen generate-var` (dry-run, full loop, validation)."""

import random

from PIL import Image
from typer.testing import CliRunner

import video_gen.runner as runner_mod
from video_gen.backends import BACKENDS
from video_gen.cli import app

runner = CliRunner()


def _spec_yaml(out_dir, seed_path, loop=1, status="live"):
    return (
        f"status: {status}\n"
        f"loop: {loop}\n"
        f'template_output: "{out_dir}/clip_<number>.mp4"\n'
        f'template_input: "{seed_path}"\n'
        'template_prompt: "<subject>"\n'
        "defaults:\n"
        '  backend: "ltx"\n'
        "  duration: 0.2\n"
        "variables:\n"
        "  subject: [a cat]\n"
    )


class _FakePipeline:
    def __init__(self):
        self.spec = BACKENDS["ltx"]
        self.device = "cpu"
        self.dtype = "bf16"
        self.repo = "fake/ltx"
        self.calls = 0

    def generate(self, config):
        self.calls += 1
        return [Image.new("RGB", (config.width, config.height), (10, 20, 30)) for _ in range(config.num_frames)]

    def unload(self):
        pass


def test_generate_var_unknown_backend_rejected(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(_spec_yaml(tmp_path / "out", tmp_path / "seed.png"), encoding="utf-8")

    result = runner.invoke(app, ["generate-var", "-C", str(spec_path), "--backend", "bogus"])

    assert result.exit_code == 1
    assert "unknown backend" in result.output


def test_generate_var_unknown_precision_rejected(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(_spec_yaml(tmp_path / "out", tmp_path / "seed.png"), encoding="utf-8")

    result = runner.invoke(app, ["generate-var", "-C", str(spec_path), "--precision", "bogus"])

    assert result.exit_code == 1
    assert "unknown precision" in result.output


def test_generate_var_dry_run_prints_prompt(tmp_path):
    seed_img = tmp_path / "seed.png"
    Image.new("RGB", (64, 64), (0, 0, 0)).save(seed_img)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(_spec_yaml(tmp_path / "out", seed_img), encoding="utf-8")

    result = runner.invoke(app, ["generate-var", "-C", str(spec_path), "--dry-run", "--dry-run-count", "1"])

    assert result.exit_code == 0, result.output
    assert "a cat" in result.output


def test_generate_var_full_loop_with_fake_pipeline(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    seed_img = tmp_path / "seed.png"
    Image.new("RGB", (64, 64), (0, 0, 0)).save(seed_img)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(_spec_yaml(out_dir, seed_img, loop=2), encoding="utf-8")

    fake = _FakePipeline()
    monkeypatch.setattr(runner_mod, "_build_pipeline", lambda spec, ov, echo: fake)
    monkeypatch.setattr(random, "randint", lambda a, b: 1234)

    result = runner.invoke(app, ["generate-var", "-C", str(spec_path), "--poll", "1", "--var-seed", "0"])

    assert result.exit_code == 0, result.output
    assert "Generated 2 clip(s)" in result.output
    assert fake.calls == 2
    files = sorted(p.name for p in out_dir.glob("clip_*.mp4"))
    assert files == ["clip_0000000000.mp4", "clip_0000000001.mp4"]
