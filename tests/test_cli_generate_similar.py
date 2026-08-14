"""End-to-end tests for `image-gen generate-similar` with a fake pipeline."""

import json

import piexif
from PIL import Image
from typer.testing import CliRunner

import image_gen.cli as cli_mod
from image_gen import llm_variation
from image_gen.cli import app
from image_gen.llm_variation import LLMConfig, VariationResult
from image_gen.metadata import GenerationMetadata

runner = CliRunner()


def _read_usercomment_json(path):
    exif_bytes = Image.open(path).info.get("exif")
    exif = piexif.load(exif_bytes) if exif_bytes else piexif.load(str(path))
    raw = exif["Exif"][piexif.ExifIFD.UserComment]
    return json.loads(bytes(raw)[8:].decode("utf-8"))


class _FakePipeline:
    """Stand-in for SDXLPipeline: returns solid-colour images, no model."""

    def __init__(self, model_path, vae_path=None, scheduler_name="euler_a"):
        self.model_path = model_path
        self.vae_path = vae_path
        self.scheduler_name = scheduler_name
        self.calls = 0

    def load_loras(self, specs):
        pass

    def load_embeddings(self, paths):
        return []

    def load_ip_adapter(self, preset, scale):
        return scale or 0.6

    def generate(self, config):
        self.calls += 1
        return [Image.new("RGB", (config.width, config.height), (self.calls * 10 % 255, 0, 0))]


def _make_source(tmp_path, **meta_overrides):
    base = dict(
        prompt="a girl in a garden",
        negative_prompt="blurry",
        model="fake_model.safetensors",
        vae=None,
        seed=999,
        width=512,
        height=512,
        steps=25,
        cfg_scale=5.0,
        scheduler="euler_a",
        clip_skip=2,
        lora=None,
        embedding=None,
        hires_fix=False,
        hires_scale=None,
        hires_steps=None,
        hires_denoising=None,
    )
    base.update(meta_overrides)
    source = tmp_path / "source.jpg"
    Image.new("RGB", (512, 512), (5, 5, 5)).save(
        source, "JPEG", exif=_exif_bytes(GenerationMetadata(**base))
    )
    return source


def _exif_bytes(metadata):
    from image_gen.metadata import _build_exif_bytes

    return _build_exif_bytes(metadata)


def _setup_model_dir(tmp_path, name="fake_model.safetensors"):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / name).write_bytes(b"fake")
    return model_dir


def test_generate_similar_writes_count_images_with_distinct_seeds(tmp_path, monkeypatch):
    source = _make_source(tmp_path)
    model_dir = _setup_model_dir(tmp_path)

    fake = _FakePipeline(model_dir / "fake_model.safetensors")
    monkeypatch.setattr(cli_mod, "SDXLPipeline", lambda **kw: fake)

    result = runner.invoke(
        app,
        [
            "generate-similar",
            str(source),
            "--count", "3",
            "--model-dir", str(model_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake.calls == 3

    outputs = [tmp_path / f"source_similar_{i:02d}.jpg" for i in range(3)]
    seeds = set()
    for out in outputs:
        assert out.exists()
        data = _read_usercomment_json(out)
        assert data["prompt"] == "a girl in a garden"
        assert data["steps"] == 25
        assert data["scheduler"] == "euler_a"
        assert data["source_image"] == "source.jpg"
        seeds.add(data["seed"])
    assert len(seeds) == 3  # all distinct


def test_generate_similar_applies_override(tmp_path, monkeypatch):
    source = _make_source(tmp_path)
    model_dir = _setup_model_dir(tmp_path)

    fake = _FakePipeline(model_dir / "fake_model.safetensors")
    monkeypatch.setattr(cli_mod, "SDXLPipeline", lambda **kw: fake)

    result = runner.invoke(
        app,
        [
            "generate-similar",
            str(source),
            "--count", "1",
            "--model-dir", str(model_dir),
            "--clip-skip", "4",
        ],
    )

    assert result.exit_code == 0, result.output
    out = tmp_path / "source_similar.jpg"
    assert out.exists()
    data = _read_usercomment_json(out)
    assert data["clip_skip"] == 4
    assert data["prompt"] == "a girl in a garden"


def test_generate_similar_missing_model_errors(tmp_path, monkeypatch):
    source = _make_source(tmp_path)
    empty_model_dir = tmp_path / "empty_models"
    empty_model_dir.mkdir()

    monkeypatch.setattr(cli_mod, "SDXLPipeline", lambda **kw: _FakePipeline(None))

    result = runner.invoke(
        app,
        [
            "generate-similar",
            str(source),
            "--count", "1",
            "--model-dir", str(empty_model_dir),
        ],
    )

    assert result.exit_code == 1
    assert "not found" in result.output


def test_generate_similar_missing_metadata_errors(tmp_path, monkeypatch):
    plain = tmp_path / "plain.jpg"
    Image.new("RGB", (16, 16)).save(plain, "JPEG")
    model_dir = _setup_model_dir(tmp_path)

    monkeypatch.setattr(cli_mod, "SDXLPipeline", lambda **kw: _FakePipeline(None))

    result = runner.invoke(
        app,
        [
            "generate-similar",
            str(plain),
            "--count", "1",
            "--model-dir", str(model_dir),
        ],
    )

    assert result.exit_code == 1
    assert "No generation metadata found" in result.output


def test_generate_similar_explicit_output_single(tmp_path, monkeypatch):
    source = _make_source(tmp_path)
    model_dir = _setup_model_dir(tmp_path)

    fake = _FakePipeline(model_dir / "fake_model.safetensors")
    monkeypatch.setattr(cli_mod, "SDXLPipeline", lambda **kw: fake)

    out = tmp_path / "custom" / "result.jpg"
    result = runner.invoke(
        app,
        [
            "generate-similar",
            str(source),
            "--count", "1",
            "--model-dir", str(model_dir),
            "--output", str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()


def test_generate_similar_explicit_output_multi_numbered(tmp_path, monkeypatch):
    source = _make_source(tmp_path)
    model_dir = _setup_model_dir(tmp_path)

    fake = _FakePipeline(model_dir / "fake_model.safetensors")
    monkeypatch.setattr(cli_mod, "SDXLPipeline", lambda **kw: fake)

    out = tmp_path / "result.jpg"
    result = runner.invoke(
        app,
        [
            "generate-similar",
            str(source),
            "--count", "2",
            "--model-dir", str(model_dir),
            "--output", str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "result_00.jpg").exists()
    assert (tmp_path / "result_01.jpg").exists()


def test_generate_similar_keep_seed_reuses_source_seed(tmp_path, monkeypatch):
    source = _make_source(tmp_path, seed=777)
    model_dir = _setup_model_dir(tmp_path)

    fake = _FakePipeline(model_dir / "fake_model.safetensors")
    monkeypatch.setattr(cli_mod, "SDXLPipeline", lambda **kw: fake)

    result = runner.invoke(
        app,
        [
            "generate-similar",
            str(source),
            "--count", "3",
            "--model-dir", str(model_dir),
            "--keep-seed",
        ],
    )

    assert result.exit_code == 0, result.output
    for i in range(3):
        data = _read_usercomment_json(tmp_path / f"source_similar_{i:02d}.jpg")
        assert data["seed"] == 777


def test_generate_similar_vocab_without_vary_errors(tmp_path, monkeypatch):
    source = _make_source(tmp_path)
    model_dir = _setup_model_dir(tmp_path)
    vocab = tmp_path / "vocab.yaml"
    vocab.write_text(
        "template_prompt: \"<x>\"\ntemplate_output: \"out/<number>.png\"\nvariables:\n  x: [a, b]\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(cli_mod, "SDXLPipeline", lambda **kw: _FakePipeline(None))

    result = runner.invoke(
        app,
        [
            "generate-similar",
            str(source),
            "--count", "1",
            "--model-dir", str(model_dir),
            "--vocab", str(vocab),
        ],
    )

    assert result.exit_code == 1
    assert "--vocab requires --vary" in result.output


def test_generate_similar_llm_missing_config_errors(tmp_path, monkeypatch):
    source = _make_source(tmp_path)
    model_dir = _setup_model_dir(tmp_path)

    monkeypatch.delenv("IMAGEGEN_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("IMAGEGEN_MODEL_NAME", raising=False)
    monkeypatch.setattr(cli_mod, "SDXLPipeline", lambda **kw: _FakePipeline(None))

    result = runner.invoke(
        app,
        [
            "generate-similar",
            str(source),
            "--count", "1",
            "--model-dir", str(model_dir),
            "--vary", "make it different",
        ],
    )

    assert result.exit_code == 1
    assert "requires an LLM endpoint" in result.output


def test_generate_similar_vary_produces_varied_prompts(tmp_path, monkeypatch):
    source = _make_source(tmp_path)
    model_dir = _setup_model_dir(tmp_path)

    fake = _FakePipeline(model_dir / "fake_model.safetensors")
    monkeypatch.setattr(cli_mod, "SDXLPipeline", lambda **kw: fake)
    monkeypatch.setattr(
        llm_variation,
        "resolve_llm_config",
        lambda base_url, api_key, model: LLMConfig(base_url="http://x", api_key="k", model="m"),
    )

    counter = {"n": 0}

    def _fake_generate_variation(config, base_prompt, base_negative, user_request, vocabulary, previous, temperature):
        counter["n"] += 1
        return VariationResult(prompt=f"{base_prompt}, variation {counter['n']}", negative_prompt=None)

    monkeypatch.setattr(llm_variation, "generate_variation", _fake_generate_variation)

    result = runner.invoke(
        app,
        [
            "generate-similar",
            str(source),
            "--count", "3",
            "--model-dir", str(model_dir),
            "--vary", "add variety",
        ],
    )

    assert result.exit_code == 0, result.output
    prompts = set()
    for i in range(3):
        data = _read_usercomment_json(tmp_path / f"source_similar_{i:02d}.jpg")
        assert data["llm_request"] == "add variety"
        assert data["prompt"].startswith("a girl in a garden, variation")
        prompts.add(data["prompt"])
    assert len(prompts) == 3
