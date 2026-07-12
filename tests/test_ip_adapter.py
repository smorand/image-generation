"""Tests for IP-Adapter helpers (no model download required)."""

import pytest
from PIL import Image

from image_gen.ip_adapter import (
    IP_ADAPTER_PRESETS,
    SUPPORTED_IP_ADAPTERS,
    build_ip_adapter_image,
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
