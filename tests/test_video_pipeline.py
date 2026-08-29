"""Tests for the video pipeline wrapper (diffusers mocked, no model needed)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from video_gen.pipeline import VideoGenConfig, VideoPipeline, load_source_image


def _patch_load(monkeypatch, backend="ltx"):
    fake_pipe = MagicMock()
    fake_pipe.to.return_value = fake_pipe

    def _fake_load_ltx(self):
        self._pipe = fake_pipe

    def _fake_load_wan(self):
        self._pipe = fake_pipe

    monkeypatch.setattr(VideoPipeline, "_load_ltx", _fake_load_ltx)
    monkeypatch.setattr(VideoPipeline, "_load_wan", _fake_load_wan)
    return fake_pipe


# --------------------------------------------------------------------------- #
# load_source_image
# --------------------------------------------------------------------------- #
def test_load_source_image_reads_rgb(tmp_path):
    img_path = tmp_path / "seed.png"
    Image.new("RGB", (32, 32), (1, 2, 3)).save(img_path)
    loaded = load_source_image(img_path)
    assert loaded.mode == "RGB"
    assert loaded.size == (32, 32)


def test_load_source_image_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="source image not found"):
        load_source_image(tmp_path / "missing.png")


# --------------------------------------------------------------------------- #
# VideoPipeline construction / load
# --------------------------------------------------------------------------- #
def test_pipeline_construction_uses_explicit_device_and_precision():
    pipeline = VideoPipeline(backend="ltx", device="cpu", precision="fp32")
    assert pipeline.device == "cpu"
    import torch

    assert pipeline.dtype == torch.float32


def test_pipeline_repo_override_wins():
    pipeline = VideoPipeline(backend="ltx", repo_override="custom/repo", device="cpu")
    assert pipeline.repo == "custom/repo"


def test_pipeline_property_before_load_raises():
    pipeline = VideoPipeline(backend="ltx", device="cpu")
    with pytest.raises(RuntimeError, match="pipeline not loaded"):
        _ = pipeline.pipe


def test_load_places_pipeline_on_device(monkeypatch):
    fake_pipe = _patch_load(monkeypatch)
    pipeline = VideoPipeline(backend="ltx", device="cpu")

    pipeline.load()

    assert pipeline.pipe is fake_pipe
    fake_pipe.set_progress_bar_config.assert_called_once_with(disable=True)


def test_load_is_idempotent(monkeypatch):
    _patch_load(monkeypatch)
    pipeline = VideoPipeline(backend="ltx", device="cpu")
    pipeline.load()
    first = pipeline.pipe
    pipeline.load()
    assert pipeline.pipe is first


def test_load_with_offload_calls_sequential_cpu_offload(monkeypatch):
    fake_pipe = _patch_load(monkeypatch)
    pipeline = VideoPipeline(backend="ltx", device="cpu", offload=True)

    pipeline.load()

    fake_pipe.enable_sequential_cpu_offload.assert_called_once_with(device="cpu")
    fake_pipe.to.assert_not_called()


def test_unload_clears_pipe(monkeypatch):
    _patch_load(monkeypatch)
    pipeline = VideoPipeline(backend="ltx", device="cpu")
    pipeline.load()

    pipeline.unload()

    with pytest.raises(RuntimeError, match="pipeline not loaded"):
        _ = pipeline.pipe


# --------------------------------------------------------------------------- #
# prepare_image (center-crop-resize, pure PIL logic)
# --------------------------------------------------------------------------- #
def test_prepare_image_resizes_and_center_crops_to_exact_target():
    pipeline = VideoPipeline(backend="ltx", device="cpu")
    source = Image.new("RGB", (100, 50), (10, 20, 30))

    result = pipeline.prepare_image(source, width=64, height=64)

    assert result.size == (64, 64)


def test_prepare_image_converts_to_rgb():
    pipeline = VideoPipeline(backend="ltx", device="cpu")
    source = Image.new("L", (40, 40))  # grayscale

    result = pipeline.prepare_image(source, width=32, height=32)

    assert result.mode == "RGB"


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #
def test_generate_calls_pipe_and_returns_frames(monkeypatch):
    fake_pipe = _patch_load(monkeypatch)
    frames = [Image.new("RGB", (64, 64))]
    fake_pipe.return_value = SimpleNamespace(frames=[frames])
    pipeline = VideoPipeline(backend="ltx", device="cpu")

    config = VideoGenConfig(prompt="a cat waking up", image=Image.new("RGB", (64, 64)), seed=0)
    result = pipeline.generate(config)

    assert result == frames
    call_kwargs = fake_pipe.call_args.kwargs
    assert call_kwargs["prompt"] == "a cat waking up"
    assert call_kwargs["output_type"] == "pil"


def test_generate_without_seed_passes_no_generator(monkeypatch):
    fake_pipe = _patch_load(monkeypatch)
    fake_pipe.return_value = SimpleNamespace(frames=[[Image.new("RGB", (64, 64))]])
    pipeline = VideoPipeline(backend="ltx", device="cpu")

    pipeline.generate(VideoGenConfig(prompt="p", image=Image.new("RGB", (64, 64)), seed=None))

    assert fake_pipe.call_args.kwargs["generator"] is None


def test_generate_merges_backend_and_config_call_extra(monkeypatch):
    fake_pipe = _patch_load(monkeypatch)
    fake_pipe.return_value = SimpleNamespace(frames=[[Image.new("RGB", (64, 64))]])
    pipeline = VideoPipeline(backend="ltx", device="cpu")

    config = VideoGenConfig(
        prompt="p",
        image=Image.new("RGB", (64, 64)),
        call_extra={"extra_flag": True},
    )
    pipeline.generate(config)

    assert fake_pipe.call_args.kwargs["extra_flag"] is True
