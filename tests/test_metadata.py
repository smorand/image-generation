"""Tests for metadata embedding (EXIF UserComment in JPEG and PNG)."""

import json

import piexif
from PIL import Image

from image_gen.metadata import GenerationMetadata, save_image_with_metadata


def _read_usercomment_json(path):
    """Extract the JSON metadata from a saved file's EXIF UserComment."""
    # piexif.load reads JPEG directly; for PNG, pull the eXIf bytes via PIL.
    exif_bytes = Image.open(path).info.get("exif")
    exif = piexif.load(exif_bytes) if exif_bytes else piexif.load(str(path))
    raw = exif["Exif"][piexif.ExifIFD.UserComment]
    # Strip the 8-byte charset prefix (b"ASCII\x00\x00\x00").
    return json.loads(raw[8:].decode("utf-8"))


def _meta(**kw):
    base = dict(
        prompt="a prompt",
        negative_prompt="bad",
        model="m.safetensors",
        vae=None,
        seed=123,
        width=1024,
        height=1024,
        steps=30,
        cfg_scale=4.0,
        scheduler="euler_a",
        clip_skip=2,
        lora=None,
        embedding=None,
        hires_fix=False,
        hires_scale=None,
        hires_steps=None,
        hires_denoising=None,
    )
    base.update(kw)
    return GenerationMetadata(**base)


def test_to_json_is_single_line_and_includes_variables():
    meta = _meta(variables={"eth": "inuit", "clothes": "a dress"})
    js = meta.to_json()
    assert "\n" not in js
    data = json.loads(js)
    assert data["variables"]["eth"] == "inuit"
    assert data["seed"] == 123


def test_save_png_embeds_metadata(tmp_path):
    img = Image.new("RGB", (16, 16), (10, 20, 30))
    out = tmp_path / "img_0000000001.png"
    saved = save_image_with_metadata(img, out, _meta(variables={"place": "beach"}))
    assert saved == out
    assert saved.exists()
    data = _read_usercomment_json(saved)
    assert data["variables"]["place"] == "beach"
    assert data["seed"] == 123


def test_save_jpeg_embeds_metadata(tmp_path):
    img = Image.new("RGB", (16, 16))
    out = tmp_path / "img.jpg"
    saved = save_image_with_metadata(img, out, _meta(variables={"eth": "inuit"}))
    assert saved.suffix == ".jpg"
    assert saved.exists()
    data = _read_usercomment_json(saved)
    assert data["variables"]["eth"] == "inuit"


def test_non_png_suffix_coerced_to_jpg(tmp_path):
    img = Image.new("RGB", (16, 16))
    out = tmp_path / "img.webp"
    saved = save_image_with_metadata(img, out, _meta())
    assert saved.suffix == ".jpg"
