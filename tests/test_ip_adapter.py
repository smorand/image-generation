"""Tests for IP-Adapter helpers (no model download required)."""

import pytest
from PIL import Image

from image_gen.ip_adapter import (
    IP_ADAPTER_PRESETS,
    IP_ADAPTER_REPO,
    IP_ADAPTER_REVISION,
    SUPPORTED_IP_ADAPTERS,
    build_ip_adapter_image,
    load_ip_adapter,
    load_reference_images,
)


def _make_image(path, color=(120, 80, 200)):
    Image.new("RGB", (32, 32), color).save(path)


def test_presets_are_exposed():
    assert "face" in SUPPORTED_IP_ADAPTERS
    assert set(SUPPORTED_IP_ADAPTERS) == set(IP_ADAPTER_PRESETS)


def test_presets_have_required_keys():
    for cfg in IP_ADAPTER_PRESETS.values():
        assert {"weight_subfolder", "weight_name", "encoder_subfolder", "default_scale"} <= cfg.keys()
        assert 0.0 <= cfg["default_scale"] <= 1.0


def test_build_single_reference_returns_image():
    img = Image.new("RGB", (16, 16))
    assert build_ip_adapter_image([img]) is img


def test_build_multiple_references_wraps_for_single_adapter():
    imgs = [Image.new("RGB", (16, 16)), Image.new("RGB", (16, 16))]
    result = build_ip_adapter_image(imgs)
    # diffusers expects a nested list: one entry per adapter, holding all refs.
    assert result == [imgs]


def test_load_reference_images_reads_rgb(tmp_path):
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    _make_image(p1)
    _make_image(p2, color=(0, 0, 0))
    images = load_reference_images([str(p1), str(p2)])
    assert len(images) == 2
    assert all(im.mode == "RGB" for im in images)


def test_load_reference_images_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_reference_images([str(tmp_path / "nope.png")])


class _FakeImageEncoder:
    def __init__(self):
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class _FakePipeline:
    def __init__(self):
        self.image_encoder = None
        self.ip_adapter_calls: list[dict] = []
        self.scale_set = None

    def load_ip_adapter(self, repo, revision, subfolder, weight_name, image_encoder_folder):
        self.ip_adapter_calls.append(
            {
                "repo": repo,
                "revision": revision,
                "subfolder": subfolder,
                "weight_name": weight_name,
                "image_encoder_folder": image_encoder_folder,
            }
        )

    def set_ip_adapter_scale(self, scale):
        self.scale_set = scale


def test_load_ip_adapter_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown IP-Adapter"):
        load_ip_adapter(_FakePipeline(), "not-a-preset", None, "cpu", None)


def test_load_ip_adapter_loads_encoder_and_weights_with_default_scale(monkeypatch):
    fake_encoder_cls = type(
        "FakeCLIPVisionModel",
        (),
        {"from_pretrained": staticmethod(lambda *a, **kw: _FakeImageEncoder())},
    )
    monkeypatch.setattr("transformers.CLIPVisionModelWithProjection", fake_encoder_cls)

    pipeline = _FakePipeline()
    effective_scale = load_ip_adapter(pipeline, "face", None, "cpu", None)

    assert effective_scale == IP_ADAPTER_PRESETS["face"]["default_scale"]
    assert pipeline.scale_set == effective_scale
    assert pipeline.image_encoder is not None
    assert pipeline.image_encoder.moved_to == "cpu"
    call = pipeline.ip_adapter_calls[0]
    assert call["repo"] == IP_ADAPTER_REPO
    assert call["revision"] == IP_ADAPTER_REVISION
    assert call["weight_name"] == IP_ADAPTER_PRESETS["face"]["weight_name"]
    assert call["image_encoder_folder"] is None


def test_load_ip_adapter_explicit_scale_overrides_default(monkeypatch):
    fake_encoder_cls = type(
        "FakeCLIPVisionModel",
        (),
        {"from_pretrained": staticmethod(lambda *a, **kw: _FakeImageEncoder())},
    )
    monkeypatch.setattr("transformers.CLIPVisionModelWithProjection", fake_encoder_cls)

    pipeline = _FakePipeline()
    effective_scale = load_ip_adapter(pipeline, "plus", 0.9, "cpu", None)

    assert effective_scale == 0.9
    assert pipeline.scale_set == 0.9
