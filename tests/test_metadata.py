"""Tests for metadata embedding (EXIF UserComment in JPEG and PNG)."""

import json
from datetime import datetime

import piexif
import pytest
from PIL import Image

from image_gen.metadata import GenerationMetadata, load_metadata, save_image_with_metadata


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


def test_generated_at_is_stamped_automatically():
    meta = _meta()
    # ISO 8601, second precision, no explicit arg needed at any call site.
    datetime.fromisoformat(meta.generated_at)


def test_generated_at_can_be_overridden_explicitly():
    meta = _meta(generated_at="2020-01-01T00:00:00")
    assert meta.generated_at == "2020-01-01T00:00:00"


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


def test_load_metadata_roundtrip_png(tmp_path):
    img = Image.new("RGB", (16, 16), (1, 2, 3))
    out = tmp_path / "img.png"
    save_image_with_metadata(img, out, _meta(source_image="orig.jpg"))
    data = load_metadata(out)
    assert data["prompt"] == "a prompt"
    assert data["seed"] == 123
    assert data["clip_skip"] == 2
    assert data["source_image"] == "orig.jpg"
    assert "vae" not in data


def test_load_metadata_roundtrip_jpeg(tmp_path):
    img = Image.new("RGB", (16, 16))
    out = tmp_path / "img.jpg"
    save_image_with_metadata(img, out, _meta())
    data = load_metadata(out)
    assert data["prompt"] == "a prompt"
    assert data["scheduler"] == "euler_a"


def test_load_metadata_missing_exif_raises(tmp_path):
    out = tmp_path / "plain.jpg"
    Image.new("RGB", (16, 16)).save(out, "JPEG")
    with pytest.raises(ValueError, match="No generation metadata found"):
        load_metadata(out)


def test_load_metadata_missing_file_raises(tmp_path):
    with pytest.raises(ValueError):
        load_metadata(tmp_path / "does-not-exist.jpg")


def test_save_writes_standard_exif_date_tags(tmp_path):
    """Finder/Preview/exiftool read the standard EXIF date tags, not our JSON
    blob in UserComment, so they must carry the generation date too."""
    img = Image.new("RGB", (16, 16))
    out = tmp_path / "img.jpg"
    save_image_with_metadata(img, out, _meta(generated_at="2026-08-29T14:30:05"))

    exif = piexif.load(str(out))
    expected = b"2026:08:29 14:30:05"
    assert exif["Exif"][piexif.ExifIFD.DateTimeOriginal] == expected
    assert exif["Exif"][piexif.ExifIFD.DateTimeDigitized] == expected
    assert exif["0th"][piexif.ImageIFD.DateTime] == expected


def test_load_metadata_roundtrip_includes_generated_at(tmp_path):
    img = Image.new("RGB", (16, 16))
    out = tmp_path / "img.png"
    save_image_with_metadata(img, out, _meta(generated_at="2026-08-29T14:30:05"))

    data = load_metadata(out)

    assert data["generated_at"] == "2026-08-29T14:30:05"
