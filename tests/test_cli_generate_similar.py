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


def _default_outputs(tmp_path, prefix, count, suffix=".jpg"):
    """Match the default '<prefix>_s<uuid>[_NN]<suffix>' naming, sorted.

    Each generate-similar run tags its default outputs with a random per-run
    id, so tests glob for the pattern instead of a fixed filename.
    """
    pattern = f"{prefix}_s*{suffix}" if count == 1 else f"{prefix}_s*_[0-9][0-9]{suffix}"
    matches = sorted(tmp_path.glob(pattern))
    assert len(matches) == count, f"expected {count} matches for {pattern!r}, got {matches}"
    return matches


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

    outputs = _default_outputs(tmp_path, "source", 3)
    seeds = set()
    for out in outputs:
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
    (out,) = _default_outputs(tmp_path, "source", 1)
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
    for out in _default_outputs(tmp_path, "source", 3):
        data = _read_usercomment_json(out)
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
    for out in _default_outputs(tmp_path, "source", 3):
        data = _read_usercomment_json(out)
        assert data["llm_request"] == "add variety"
        assert data["prompt"].startswith("a girl in a garden, variation")
        prompts.add(data["prompt"])
    assert len(prompts) == 3


def test_generate_similar_no_sources_is_noop(tmp_path, monkeypatch):
    model_dir = _setup_model_dir(tmp_path)
    monkeypatch.setattr(cli_mod, "SDXLPipeline", lambda **kw: _FakePipeline(None))

    result = runner.invoke(app, ["generate-similar", "--model-dir", str(model_dir)])

    assert result.exit_code == 0, result.output
    assert "nothing to do" in result.output.lower()


def test_generate_similar_multiple_sources_generates_count_each(tmp_path, monkeypatch):
    source_a = _make_source(tmp_path, seed=111)
    source_a.rename(tmp_path / "a.jpg")
    source_a = tmp_path / "a.jpg"
    source_b = _make_source(tmp_path, seed=222, prompt="a boy on a beach")
    source_b.rename(tmp_path / "b.jpg")
    source_b = tmp_path / "b.jpg"
    model_dir = _setup_model_dir(tmp_path)

    fake = _FakePipeline(model_dir / "fake_model.safetensors")
    monkeypatch.setattr(cli_mod, "SDXLPipeline", lambda **kw: fake)

    result = runner.invoke(
        app,
        [
            "generate-similar",
            str(source_a),
            str(source_b),
            "--count", "2",
            "--model-dir", str(model_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake.calls == 4

    for out in _default_outputs(tmp_path, "a", 2):
        data = _read_usercomment_json(out)
        assert data["source_image"] == "a.jpg"
        assert data["prompt"] == "a girl in a garden"

    for out in _default_outputs(tmp_path, "b", 2):
        data = _read_usercomment_json(out)
        assert data["source_image"] == "b.jpg"
        assert data["prompt"] == "a boy on a beach"


def test_generate_similar_multiple_sources_keep_seed_uses_each_own_seed(tmp_path, monkeypatch):
    source_a = _make_source(tmp_path, seed=111)
    source_a.rename(tmp_path / "a.jpg")
    source_a = tmp_path / "a.jpg"
    source_b = _make_source(tmp_path, seed=222)
    source_b.rename(tmp_path / "b.jpg")
    source_b = tmp_path / "b.jpg"
    model_dir = _setup_model_dir(tmp_path)

    fake = _FakePipeline(model_dir / "fake_model.safetensors")
    monkeypatch.setattr(cli_mod, "SDXLPipeline", lambda **kw: fake)

    result = runner.invoke(
        app,
        [
            "generate-similar",
            str(source_a),
            str(source_b),
            "--count", "2",
            "--model-dir", str(model_dir),
            "--keep-seed",
        ],
    )

    assert result.exit_code == 0, result.output
    for out in _default_outputs(tmp_path, "a", 2):
        assert _read_usercomment_json(out)["seed"] == 111
    for out in _default_outputs(tmp_path, "b", 2):
        assert _read_usercomment_json(out)["seed"] == 222


def test_generate_similar_multiple_sources_with_output_errors(tmp_path, monkeypatch):
    source_a = _make_source(tmp_path)
    source_a.rename(tmp_path / "a.jpg")
    source_a = tmp_path / "a.jpg"
    source_b = _make_source(tmp_path)
    source_b.rename(tmp_path / "b.jpg")
    source_b = tmp_path / "b.jpg"
    model_dir = _setup_model_dir(tmp_path)

    monkeypatch.setattr(cli_mod, "SDXLPipeline", lambda **kw: _FakePipeline(None))

    result = runner.invoke(
        app,
        [
            "generate-similar",
            str(source_a),
            str(source_b),
            "--count", "1",
            "--model-dir", str(model_dir),
            "--output", str(tmp_path / "out.jpg"),
        ],
    )

    assert result.exit_code == 1
    assert "--output cannot be used with multiple source images" in result.output


def test_generate_similar_multiple_sources_reuses_pipeline_when_config_unchanged(tmp_path, monkeypatch):
    source_a = _make_source(tmp_path)
    source_a.rename(tmp_path / "a.jpg")
    source_a = tmp_path / "a.jpg"
    source_b = _make_source(tmp_path)
    source_b.rename(tmp_path / "b.jpg")
    source_b = tmp_path / "b.jpg"
    model_dir = _setup_model_dir(tmp_path)

    load_count = {"n": 0}

    def _factory(**kw):
        load_count["n"] += 1
        return _FakePipeline(**kw)

    monkeypatch.setattr(cli_mod, "SDXLPipeline", _factory)

    result = runner.invoke(
        app,
        [
            "generate-similar",
            str(source_a),
            str(source_b),
            "--count", "1",
            "--model-dir", str(model_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert load_count["n"] == 1
    assert "Reusing already-loaded model" in result.output


def test_generate_similar_missing_source_warns_and_continues(tmp_path, monkeypatch):
    source_a = _make_source(tmp_path)
    source_a.rename(tmp_path / "a.jpg")
    source_a = tmp_path / "a.jpg"
    missing = tmp_path / "missing.jpg"
    model_dir = _setup_model_dir(tmp_path)

    fake = _FakePipeline(model_dir / "fake_model.safetensors")
    monkeypatch.setattr(cli_mod, "SDXLPipeline", lambda **kw: fake)

    result = runner.invoke(
        app,
        [
            "generate-similar",
            str(source_a),
            str(missing),
            "--count", "1",
            "--model-dir", str(model_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"'{missing}' does not exist, skipping" in result.output
    _default_outputs(tmp_path, "a", 1)
    assert fake.calls == 1


def test_generate_similar_all_sources_missing_is_clean(tmp_path, monkeypatch):
    model_dir = _setup_model_dir(tmp_path)
    monkeypatch.setattr(cli_mod, "SDXLPipeline", lambda **kw: _FakePipeline(None))

    result = runner.invoke(
        app,
        [
            "generate-similar",
            str(tmp_path / "missing1.jpg"),
            str(tmp_path / "missing2.jpg"),
            "--count", "1",
            "--model-dir", str(model_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output.count("does not exist, skipping") == 2
    assert "Generated 0 image(s)" in result.output
