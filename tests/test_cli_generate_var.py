"""End-to-end tests for `image-gen generate-var` (status flags, dry-run, full loop)."""

import random

from PIL import Image
from typer.testing import CliRunner

import image_gen.runner as runner_mod
from image_gen.cli import app

runner = CliRunner()


def _spec_yaml(out_dir, loop=1, status="live"):
    return (
        f"status: {status}\n"
        f"loop: {loop}\n"
        f'template_output: "{out_dir}/img_<number>.png"\n'
        'template_prompt: "solo, <subject>"\n'
        "defaults:\n"
        '  model: "fake.safetensors"\n'
        "variables:\n"
        "  subject: [a cat]\n"
    )


class _FakePipeline:
    def __init__(self):
        self.calls = 0

    def generate(self, config):
        self.calls += 1
        return [Image.new("RGB", (8, 8), (self.calls % 255, 0, 0))]


def test_generate_var_pause_sets_status_and_exits(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(_spec_yaml(tmp_path / "out"), encoding="utf-8")

    result = runner.invoke(app, ["generate-var", "-C", str(spec_path), "--pause"])

    assert result.exit_code == 0, result.output
    assert "status set to 'pause'" in result.output
    assert "status: pause" in spec_path.read_text(encoding="utf-8")


def test_generate_var_mutually_exclusive_status_flags_rejected(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(_spec_yaml(tmp_path / "out"), encoding="utf-8")

    result = runner.invoke(app, ["generate-var", "-C", str(spec_path), "--pause", "--stop"])

    assert result.exit_code == 1
    assert "mutually exclusive" in result.output


def test_generate_var_unknown_scheduler_rejected(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(_spec_yaml(tmp_path / "out"), encoding="utf-8")

    result = runner.invoke(app, ["generate-var", "-C", str(spec_path), "--scheduler", "bogus"])

    assert result.exit_code == 1
    assert "Unknown scheduler" in result.output


def test_generate_var_dry_run_prints_prompt(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(_spec_yaml(tmp_path / "out"), encoding="utf-8")

    result = runner.invoke(app, ["generate-var", "-C", str(spec_path), "--dry-run", "--dry-run-count", "1"])

    assert result.exit_code == 0, result.output
    assert "a cat" in result.output


def test_generate_var_full_loop_with_fake_pipeline(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(_spec_yaml(out_dir, loop=2), encoding="utf-8")

    fake = _FakePipeline()
    monkeypatch.setattr(runner_mod, "_build_pipeline", lambda spec, ov, echo: (fake, [], None))
    monkeypatch.setattr(random, "randint", lambda a, b: 1234)

    result = runner.invoke(app, ["generate-var", "-C", str(spec_path), "--poll", "1", "--var-seed", "0"])

    assert result.exit_code == 0, result.output
    assert "Generated 2 image(s)" in result.output
    assert fake.calls == 2
    files = sorted(p.name for p in out_dir.glob("img_*.png"))
    assert files == ["img_0000000000.png", "img_0000000001.png"]
