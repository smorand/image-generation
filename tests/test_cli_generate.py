"""End-to-end tests for `image-gen generate` with a fake pipeline."""

import json

from PIL import Image
from typer.testing import CliRunner

import image_gen.cli as cli_mod
from image_gen.cli import app

runner = CliRunner()


class _FakePipeline:
    """Stand-in for SDXLPipeline: returns solid-colour images, no model."""

    def __init__(self, model_path, vae_path=None, scheduler_name="euler_a"):
        self.model_path = model_path
        self.vae_path = vae_path
        self.scheduler_name = scheduler_name
        self.calls = 0
        self.loaded_loras = None
        self.loaded_embeddings = None
        self.loaded_ip_adapter = None

    def load_loras(self, specs):
        self.loaded_loras = specs

    def load_embeddings(self, paths):
        self.loaded_embeddings = paths
        return [f"token_{i}" for i in range(len(paths))]

    def load_ip_adapter(self, preset, scale):
        self.loaded_ip_adapter = (preset, scale)
        return scale or 0.6

    def generate(self, config):
        self.calls += 1
        return [
            Image.new("RGB", (config.width, config.height), (self.calls * 10 % 255, 0, 0))
            for _ in range(config.batch_size)
        ]


def _make_model_file(tmp_path):
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"fake")
    return model


def test_generate_writes_single_image_with_default_negative(tmp_path, monkeypatch):
    model = _make_model_file(tmp_path)
    out = tmp_path / "output.jpg"
    fake = _FakePipeline(model)
    monkeypatch.setattr(cli_mod, "SDXLPipeline", lambda **kw: fake)

    result = runner.invoke(
        app,
        ["generate", "-m", str(model), "-p", "a cat", "-o", str(out), "--seed", "42"],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    assert fake.calls == 1
    assert "Using seed" not in result.output  # explicit --seed given, no random-seed message
    assert "Saved:" in result.output
    assert "Done!" in result.output


def test_generate_random_seed_when_not_given(tmp_path, monkeypatch):
    model = _make_model_file(tmp_path)
    out = tmp_path / "output.jpg"
    fake = _FakePipeline(model)
    monkeypatch.setattr(cli_mod, "SDXLPipeline", lambda **kw: fake)

    result = runner.invoke(app, ["generate", "-m", str(model), "-p", "a cat", "-o", str(out)])

    assert result.exit_code == 0, result.output
    assert "Using seed:" in result.output


def test_generate_batch_size_writes_numbered_files(tmp_path, monkeypatch):
    model = _make_model_file(tmp_path)
    out = tmp_path / "output.jpg"
    fake = _FakePipeline(model)
    monkeypatch.setattr(cli_mod, "SDXLPipeline", lambda **kw: fake)

    result = runner.invoke(
        app,
        ["generate", "-m", str(model), "-p", "a cat", "-o", str(out), "--batch-size", "3", "--seed", "1"],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "output_00.jpg").exists()
    assert (tmp_path / "output_01.jpg").exists()
    assert (tmp_path / "output_02.jpg").exists()


def test_generate_unknown_scheduler_rejected(tmp_path, monkeypatch):
    model = _make_model_file(tmp_path)
    out = tmp_path / "output.jpg"
    monkeypatch.setattr(cli_mod, "SDXLPipeline", lambda **kw: _FakePipeline(model))

    result = runner.invoke(
        app,
        ["generate", "-m", str(model), "-p", "a cat", "-o", str(out), "--scheduler", "bogus"],
    )

    assert result.exit_code == 1
    assert "Unknown scheduler" in result.output


def test_generate_unknown_ip_adapter_rejected(tmp_path, monkeypatch):
    model = _make_model_file(tmp_path)
    out = tmp_path / "output.jpg"
    monkeypatch.setattr(cli_mod, "SDXLPipeline", lambda **kw: _FakePipeline(model))

    result = runner.invoke(
        app,
        ["generate", "-m", str(model), "-p", "a cat", "-o", str(out), "--ip-adapter", "bogus"],
    )

    assert result.exit_code == 1
    assert "Unknown IP-Adapter" in result.output


def test_generate_loads_loras_and_embeddings(tmp_path, monkeypatch):
    model = _make_model_file(tmp_path)
    out = tmp_path / "output.jpg"
    lora_file = tmp_path / "style.safetensors"
    lora_file.write_bytes(b"fake")
    emb_file = tmp_path / "emb.safetensors"
    emb_file.write_bytes(b"fake")

    fake = _FakePipeline(model)
    monkeypatch.setattr(cli_mod, "SDXLPipeline", lambda **kw: fake)

    result = runner.invoke(
        app,
        [
            "generate",
            "-m",
            str(model),
            "-p",
            "a cat",
            "-o",
            str(out),
            "--lora",
            str(lora_file),
            "--embedding",
            str(emb_file),
            "--seed",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake.loaded_loras == [str(lora_file)]
    assert fake.loaded_embeddings == [str(emb_file)]
    assert "Available tokens: token_0" in result.output


def test_generate_writes_generation_log(tmp_path, monkeypatch):
    model = _make_model_file(tmp_path)
    out = tmp_path / "output.jpg"
    log_dir = tmp_path / "logs"
    fake = _FakePipeline(model)
    monkeypatch.setattr(cli_mod, "SDXLPipeline", lambda **kw: fake)

    result = runner.invoke(
        app,
        ["generate", "-m", str(model), "-p", "a cat", "-o", str(out), "--seed", "7", "--log-dir", str(log_dir)],
    )

    assert result.exit_code == 0, result.output
    assert "Logged:" in result.output
    log_files = list(log_dir.glob("*.jsonl"))
    assert len(log_files) == 1
    rec = json.loads(log_files[0].read_text(encoding="utf-8").strip().splitlines()[0])
    assert rec["command"] == "generate"
    assert rec["prompt"] == "a cat"
