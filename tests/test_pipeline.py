"""Tests for the SDXL pipeline wrapper (diffusers/compel mocked, no model needed)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

import image_gen.pipeline as pipeline_mod
from image_gen.pipeline import DEFAULT_NEGATIVE_PROMPT, GenerationConfig, SDXLPipeline


class _FakeDiffusersPipeline(MagicMock):
    """A MagicMock that also behaves like an object with fluent .to()."""

    def to(self, device):
        return self


class _FakePromptEncoder:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def get_embeddings_for_pipeline(self, prompt, negative_prompt=None):
        return {"prompt_embeds": f"embeds:{prompt}", "negative_prompt_embeds": f"neg:{negative_prompt}"}


def _patch_model_loading(monkeypatch):
    fake_pipeline = _FakeDiffusersPipeline()
    monkeypatch.setattr(
        pipeline_mod.StableDiffusionXLPipeline,
        "from_single_file",
        classmethod(lambda cls, *a, **kw: fake_pipeline),
    )
    monkeypatch.setattr(
        pipeline_mod.AutoencoderKL,
        "from_single_file",
        classmethod(lambda cls, *a, **kw: MagicMock()),
    )
    monkeypatch.setattr(pipeline_mod, "get_scheduler", lambda name, config: f"scheduler:{name}")
    monkeypatch.setattr(pipeline_mod, "SDXLPromptEncoder", _FakePromptEncoder)
    return fake_pipeline


def _make_model_file(tmp_path):
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"fake")
    return model


# --------------------------------------------------------------------------- #
# GenerationConfig
# --------------------------------------------------------------------------- #
def test_generation_config_defaults_negative_prompt_when_none():
    config = GenerationConfig(prompt="a cat")
    assert config.negative_prompt == DEFAULT_NEGATIVE_PROMPT


def test_generation_config_keeps_explicit_negative_prompt():
    config = GenerationConfig(prompt="a cat", negative_prompt="blurry")
    assert config.negative_prompt == "blurry"


# --------------------------------------------------------------------------- #
# SDXLPipeline construction / device+dtype autodetection
# --------------------------------------------------------------------------- #
def test_pipeline_uses_explicit_device_and_dtype(tmp_path):
    pipeline = SDXLPipeline(model_path=_make_model_file(tmp_path), device="cpu", dtype=torch.float32)
    assert pipeline.device == "cpu"
    assert pipeline.dtype == torch.float32


def test_pipeline_property_before_load_raises(tmp_path):
    pipeline = SDXLPipeline(model_path=_make_model_file(tmp_path), device="cpu")
    with pytest.raises(RuntimeError, match="Pipeline not loaded"):
        _ = pipeline.pipeline


def test_prompt_encoder_property_before_load_raises(tmp_path):
    pipeline = SDXLPipeline(model_path=_make_model_file(tmp_path), device="cpu")
    with pytest.raises(RuntimeError, match="Prompt encoder not initialized"):
        _ = pipeline.prompt_encoder


# --------------------------------------------------------------------------- #
# load()
# --------------------------------------------------------------------------- #
def test_load_sets_pipeline_and_prompt_encoder(tmp_path, monkeypatch):
    fake_pipeline = _patch_model_loading(monkeypatch)
    pipeline = SDXLPipeline(model_path=_make_model_file(tmp_path), device="cpu", scheduler_name="euler_a")

    pipeline.load()

    assert pipeline.pipeline is fake_pipeline
    assert isinstance(pipeline.prompt_encoder, _FakePromptEncoder)
    assert fake_pipeline.scheduler == "scheduler:euler_a"


def test_load_is_idempotent(tmp_path, monkeypatch):
    _patch_model_loading(monkeypatch)
    pipeline = SDXLPipeline(model_path=_make_model_file(tmp_path), device="cpu")

    pipeline.load()
    first = pipeline.pipeline
    pipeline.load()

    assert pipeline.pipeline is first


def test_load_with_custom_vae_sets_vae(tmp_path, monkeypatch):
    fake_pipeline = _patch_model_loading(monkeypatch)
    vae_file = tmp_path / "vae.safetensors"
    vae_file.write_bytes(b"fake")
    pipeline = SDXLPipeline(model_path=_make_model_file(tmp_path), vae_path=vae_file, device="cpu")

    pipeline.load()

    assert fake_pipeline.vae is not None


# --------------------------------------------------------------------------- #
# LoRA / embedding / IP-Adapter wrapper methods (delegate to their modules)
# --------------------------------------------------------------------------- #
def test_load_loras_delegates_and_records_specs(tmp_path, monkeypatch):
    _patch_model_loading(monkeypatch)
    calls = []
    monkeypatch.setattr(pipeline_mod, "load_loras", lambda pipeline, specs: calls.append((pipeline, specs)))
    pipeline = SDXLPipeline(model_path=_make_model_file(tmp_path), device="cpu")

    pipeline.load_loras(["a.safetensors:0.5"])

    assert calls[0][1] == ["a.safetensors:0.5"]
    assert pipeline._loaded_loras == ["a.safetensors:0.5"]


def test_load_embeddings_delegates_and_returns_tokens(tmp_path, monkeypatch):
    _patch_model_loading(monkeypatch)
    monkeypatch.setattr(pipeline_mod, "load_embeddings", lambda pipeline, paths: [f"tok_{p}" for p in paths])
    pipeline = SDXLPipeline(model_path=_make_model_file(tmp_path), device="cpu")

    tokens = pipeline.load_embeddings(["a.pt", "b.pt"])

    assert tokens == ["tok_a.pt", "tok_b.pt"]
    assert pipeline._loaded_embeddings == ["a.pt", "b.pt"]


def test_load_ip_adapter_delegates_and_stores_effective_scale(tmp_path, monkeypatch):
    _patch_model_loading(monkeypatch)
    monkeypatch.setattr(pipeline_mod, "_load_ip_adapter", lambda pipeline, preset, scale, device, dtype: 0.42)
    pipeline = SDXLPipeline(model_path=_make_model_file(tmp_path), device="cpu")

    effective = pipeline.load_ip_adapter("face", None)

    assert effective == 0.42
    assert pipeline._ip_adapter_preset == "face"
    assert pipeline._ip_adapter_scale == 0.42


# --------------------------------------------------------------------------- #
# free_cache
# --------------------------------------------------------------------------- #
def test_free_cache_noop_on_cpu(tmp_path, monkeypatch):
    _patch_model_loading(monkeypatch)
    pipeline = SDXLPipeline(model_path=_make_model_file(tmp_path), device="cpu")
    pipeline.free_cache()  # should not raise


def test_free_cache_calls_mps_empty_cache(tmp_path, monkeypatch):
    _patch_model_loading(monkeypatch)
    pipeline = SDXLPipeline(model_path=_make_model_file(tmp_path), device="mps")
    called = []
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: called.append(True))

    pipeline.free_cache()

    assert called == [True]


# --------------------------------------------------------------------------- #
# generate() (base resolution, no hi-res fix)
# --------------------------------------------------------------------------- #
def test_generate_calls_pipeline_and_returns_images(tmp_path, monkeypatch):
    fake_pipeline = _patch_model_loading(monkeypatch)
    img = Image.new("RGB", (8, 8))
    fake_pipeline.return_value = SimpleNamespace(images=[img])
    pipeline = SDXLPipeline(model_path=_make_model_file(tmp_path), device="cpu")

    result = pipeline.generate(GenerationConfig(prompt="a cat", seed=0))

    assert result == [img]
    call_kwargs = fake_pipeline.call_args.kwargs
    assert call_kwargs["prompt_embeds"] == "embeds:a cat"
    assert call_kwargs["width"] == 1024
    assert call_kwargs["num_inference_steps"] == 30


def test_generate_without_seed_passes_no_generator(tmp_path, monkeypatch):
    fake_pipeline = _patch_model_loading(monkeypatch)
    fake_pipeline.return_value = SimpleNamespace(images=[Image.new("RGB", (8, 8))])
    pipeline = SDXLPipeline(model_path=_make_model_file(tmp_path), device="cpu")

    pipeline.generate(GenerationConfig(prompt="a cat", seed=None))

    assert fake_pipeline.call_args.kwargs["generator"] is None


def test_generate_passes_ip_adapter_image_when_present(tmp_path, monkeypatch):
    fake_pipeline = _patch_model_loading(monkeypatch)
    fake_pipeline.return_value = SimpleNamespace(images=[Image.new("RGB", (8, 8))])
    pipeline = SDXLPipeline(model_path=_make_model_file(tmp_path), device="cpu")
    ref_img = Image.new("RGB", (16, 16))

    pipeline.generate(GenerationConfig(prompt="a cat", ip_adapter_images=[ref_img]))

    assert fake_pipeline.call_args.kwargs["ip_adapter_image"] is ref_img


def test_generate_clip_skip_1_passes_none(tmp_path, monkeypatch):
    fake_pipeline = _patch_model_loading(monkeypatch)
    fake_pipeline.return_value = SimpleNamespace(images=[Image.new("RGB", (8, 8))])
    pipeline = SDXLPipeline(model_path=_make_model_file(tmp_path), device="cpu")

    pipeline.generate(GenerationConfig(prompt="a cat", clip_skip=1))

    assert fake_pipeline.call_args.kwargs["clip_skip"] is None


def test_generate_clip_skip_2_passes_1(tmp_path, monkeypatch):
    fake_pipeline = _patch_model_loading(monkeypatch)
    fake_pipeline.return_value = SimpleNamespace(images=[Image.new("RGB", (8, 8))])
    pipeline = SDXLPipeline(model_path=_make_model_file(tmp_path), device="cpu")

    pipeline.generate(GenerationConfig(prompt="a cat", clip_skip=2))

    assert fake_pipeline.call_args.kwargs["clip_skip"] == 1


# --------------------------------------------------------------------------- #
# generate() with hires_fix (img2img second pass)
# --------------------------------------------------------------------------- #
def test_generate_with_hires_fix_runs_img2img_pass(tmp_path, monkeypatch):
    fake_pipeline = _patch_model_loading(monkeypatch)
    base_img = Image.new("RGB", (64, 64))
    fake_pipeline.return_value = SimpleNamespace(images=[base_img])

    fake_img2img = MagicMock()
    upscaled_img = Image.new("RGB", (96, 96))
    fake_img2img.return_value = SimpleNamespace(images=[upscaled_img])
    monkeypatch.setattr(
        "diffusers.StableDiffusionXLImg2ImgPipeline",
        lambda **kw: fake_img2img,
    )

    pipeline = SDXLPipeline(model_path=_make_model_file(tmp_path), device="cpu")

    result = pipeline.generate(GenerationConfig(prompt="a cat", width=64, height=64, hires_fix=True, hires_scale=1.5))

    assert result == [upscaled_img]
    img2img_call = fake_img2img.call_args.kwargs
    assert img2img_call["strength"] == 0.5  # default hires_denoising
    assert img2img_call["image"].size == (96, 96)  # 64 * 1.5
