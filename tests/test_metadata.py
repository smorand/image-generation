"""Tests for metadata embedding (EXIF UserComment in JPEG and PNG)."""

import json
from datetime import datetime

import piexif
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from image_gen.metadata import GenerationMetadata, load_metadata, save_image_with_metadata

RAW_PROFILE_KEYWORD = b"Raw profile type exif"


def _encode_raw_profile_exif(exif_bytes: bytes) -> bytes:
    """Build the ImageMagick-style "exif" text-chunk payload that
    ``little_exif`` (and ImageMagick itself) write for PNG: a literal
    "\nexif\n" header, the byte length as an 8-char space-padded decimal
    string, then the EXIF bytes hex-encoded two ASCII chars per byte,
    terminated by a literal "00" and a trailing newline. This is the exact
    format ``PIL.Image.Exif.getexif()``'s ``"Raw profile type exif"``
    fallback (see ``PIL/Image.py``) decodes back with
    ``bytes.fromhex("".join(text.split("\n")[3:]))``.
    """
    length_str = str(len(exif_bytes)).rjust(8)
    hex_body = exif_bytes.hex()
    return b"\nexif\n" + length_str.encode("ascii") + b"\n" + hex_body.encode("ascii") + b"00\n"


def _save_png_with_raw_profile_chunks(path, exif_payloads, chunk_types):
    """Save a plain PNG whose EXIF lives only in one or more legacy
    ``zTXt``/``tEXt`` "Raw profile type exif" chunks (never a real ``eXIf``
    chunk), mirroring how ``little_exif``-based writers (e.g.
    image-sec-gallery's ``download``) produce PNGs today.

    ``exif_payloads``/``chunk_types`` are parallel lists; chunks are added,
    and therefore end up in the file, in list order.
    """
    png_info = PngInfo()
    for exif_bytes, chunk_type in zip(exif_payloads, chunk_types, strict=True):
        payload = _encode_raw_profile_exif(exif_bytes)
        png_info.add_text(RAW_PROFILE_KEYWORD, payload, zip=(chunk_type == "zTXt"))
    img = Image.new("RGB", (16, 16), (1, 2, 3))
    img.save(path, "PNG", pnginfo=png_info)


def _user_comment_exif_bytes(json_payload: bytes) -> bytes:
    """Build raw TIFF/EXIF bytes carrying ``json_payload`` (already including
    the 8-byte ASCII charset prefix) as the UserComment tag, via piexif, the
    same way ``little_exif``/image-gen's own writer would."""
    exif_dict = {"0th": {}, "Exif": {piexif.ExifIFD.UserComment: json_payload}, "GPS": {}, "1st": {}, "thumbnail": None}
    return piexif.dump(exif_dict)


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


def test_load_metadata_reads_imagemagick_raw_profile_ztxt_png(tmp_path):
    """PNGs written by little_exif-based tools (e.g. image-sec-gallery's
    `download`) carry EXIF only in a legacy zTXt "Raw profile type exif"
    chunk, never a real eXIf chunk. load_metadata must still read them, via
    Pillow's own getexif() fallback for this convention."""
    out = tmp_path / "img.png"
    payload = b"ASCII\x00\x00\x00" + json.dumps({"prompt": "a raw-profile prompt", "seed": 99}).encode("utf-8")
    exif_bytes = _user_comment_exif_bytes(payload)
    _save_png_with_raw_profile_chunks(out, [exif_bytes], ["zTXt"])

    data = load_metadata(out)
    assert data["prompt"] == "a raw-profile prompt"
    assert data["seed"] == 99


def test_load_metadata_reads_imagemagick_raw_profile_text_png(tmp_path):
    """Same convention, but as an uncompressed tEXt chunk instead of zTXt:
    little_exif's own reader (and ImageMagick) support both, so we must
    too."""
    out = tmp_path / "img.png"
    payload = b"ASCII\x00\x00\x00" + json.dumps({"prompt": "a tEXt prompt", "seed": 7}).encode("utf-8")
    exif_bytes = _user_comment_exif_bytes(payload)
    _save_png_with_raw_profile_chunks(out, [exif_bytes], ["tEXt"])

    data = load_metadata(out)
    assert data["prompt"] == "a tEXt prompt"
    assert data["seed"] == 7


def _stringified_payload(**overrides):
    """A UserComment JSON payload with every typed field stringified, as
    image-sec-gallery's download hands them back after its SIV-encrypted
    metadata round trip (every field, including numbers/bools/arrays, is
    stored and re-read as plain text; see flatten.rs's flatten_json)."""
    base = {
        "prompt": "a cat",
        "seed": "2081615060",
        "width": "1024",
        "height": "768",
        "steps": "40",
        "cfg_scale": "5.0",
        "clip_skip": "2",
        "hires_fix": "false",
        "hires_scale": "1.5",
        "hires_steps": "15",
        "hires_denoising": "0.5",
        "ip_adapter_scale": "0.8",
        "lora": '["style.safetensors", "face.safetensors"]',
        "embedding": '["emb1"]',
        "ip_adapter_images": '["ref.png"]',
    }
    base.update(overrides)
    return base


def test_load_metadata_coerces_stringified_numeric_and_bool_fields(tmp_path):
    """After an image-sec-gallery upload/download round trip, every
    GenerationMetadata field comes back as a plain string (its SIV metadata
    model has no type schema). load_metadata must coerce known fields back
    to real int/float/bool/list so downstream consumers (generate-similar's
    GenerationConfig) work unmodified."""
    out = tmp_path / "img.png"
    payload = b"ASCII\x00\x00\x00" + json.dumps(_stringified_payload()).encode("utf-8")
    exif_bytes = _user_comment_exif_bytes(payload)
    _save_png_with_raw_profile_chunks(out, [exif_bytes], ["zTXt"])

    data = load_metadata(out)
    assert data["seed"] == 2081615060 and isinstance(data["seed"], int)
    assert data["width"] == 1024 and isinstance(data["width"], int)
    assert data["height"] == 768 and isinstance(data["height"], int)
    assert data["steps"] == 40 and isinstance(data["steps"], int)
    assert data["clip_skip"] == 2 and isinstance(data["clip_skip"], int)
    assert data["hires_steps"] == 15 and isinstance(data["hires_steps"], int)
    assert data["cfg_scale"] == 5.0 and isinstance(data["cfg_scale"], float)
    assert data["hires_scale"] == 1.5 and isinstance(data["hires_scale"], float)
    assert data["hires_denoising"] == 0.5 and isinstance(data["hires_denoising"], float)
    assert data["ip_adapter_scale"] == 0.8 and isinstance(data["ip_adapter_scale"], float)
    assert data["hires_fix"] is False
    assert data["lora"] == ["style.safetensors", "face.safetensors"]
    assert data["embedding"] == ["emb1"]
    assert data["ip_adapter_images"] == ["ref.png"]


def test_load_metadata_coerces_stringified_true_bool(tmp_path):
    out = tmp_path / "img.png"
    payload = b"ASCII\x00\x00\x00" + json.dumps(_stringified_payload(hires_fix="true")).encode("utf-8")
    exif_bytes = _user_comment_exif_bytes(payload)
    _save_png_with_raw_profile_chunks(out, [exif_bytes], ["zTXt"])

    data = load_metadata(out)
    assert data["hires_fix"] is True


def test_load_metadata_leaves_already_typed_fields_untouched(tmp_path):
    """Files written directly by image-gen's own save_image_with_metadata
    (never round-tripped through image-sec-gallery) already carry real
    JSON types; the coercion step must be a no-op for them."""
    img = Image.new("RGB", (16, 16))
    out = tmp_path / "img.png"
    save_image_with_metadata(img, out, _meta(hires_fix=True, hires_scale=1.5, lora=["a.safetensors"]))

    data = load_metadata(out)
    assert data["seed"] == 123 and isinstance(data["seed"], int)
    assert data["hires_fix"] is True
    assert data["lora"] == ["a.safetensors"]


def test_load_metadata_malformed_numeric_field_raises(tmp_path):
    out = tmp_path / "img.png"
    payload = b"ASCII\x00\x00\x00" + json.dumps(_stringified_payload(seed="not-a-number")).encode("utf-8")
    exif_bytes = _user_comment_exif_bytes(payload)
    _save_png_with_raw_profile_chunks(out, [exif_bytes], ["zTXt"])

    with pytest.raises(ValueError, match="malformed 'seed' field"):
        load_metadata(out)


def test_load_metadata_last_raw_profile_chunk_wins_on_duplicates(tmp_path):
    """If a file somehow ends up with more than one "Raw profile type
    exif" text chunk (same or mixed zTXt/tEXt/iTXt), Pillow's PNG chunk
    reader resolves it deterministically: the last chunk in file order
    wins, silently. This locks in that accepted behavior as a regression
    guard; image-gen itself never produces duplicates."""
    out = tmp_path / "img.png"
    first_payload = b"ASCII\x00\x00\x00" + json.dumps({"prompt": "first (stale)", "seed": 1}).encode("utf-8")
    second_payload = b"ASCII\x00\x00\x00" + json.dumps({"prompt": "second (wins)", "seed": 2}).encode("utf-8")
    exif_bytes_list = [_user_comment_exif_bytes(first_payload), _user_comment_exif_bytes(second_payload)]
    _save_png_with_raw_profile_chunks(out, exif_bytes_list, ["zTXt", "tEXt"])

    data = load_metadata(out)
    assert data["prompt"] == "second (wins)"
    assert data["seed"] == 2


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
