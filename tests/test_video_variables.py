"""Tests for the video-gen spec loader and template_input resolution."""

import random

import pytest

from image_gen.variables import Option
from video_gen.variables import (
    VideoVarSpec,
    load_video_spec,
    render_input,
    resolve_prompt,
)


def _spec(**kw):
    base = dict(
        template_prompt="<subject>",
        template_output="out/clip_<number>.mp4",
        template_input="seeds/cat.png",
        variables={"subject": [Option("a cat")]},
    )
    base.update(kw)
    return VideoVarSpec(**base)


# --------------------------------------------------------------------------- #
# load_video_spec
# --------------------------------------------------------------------------- #
SPEC_YAML = """\
status: live
loop: 2
template_prompt: "<subject>, cinematic"
template_output: "out/clip_<number>.mp4"
template_input: "seeds/<number>.png"
negative_prompt: "blurry"
defaults:
  backend: ltx
  fps: 24
variables:
  subject: ["a cat", "a dog"]
"""


def _write(tmp_path, text):
    p = tmp_path / "spec.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_video_spec_parses_header(tmp_path):
    spec = load_video_spec(_write(tmp_path, SPEC_YAML))
    assert spec.status == "live"
    assert spec.loop == 2
    assert spec.template_input == "seeds/<number>.png"
    assert spec.negative_prompt == "blurry"
    assert spec.defaults == {"backend": "ltx", "fps": 24}


def test_load_video_spec_missing_key_raises(tmp_path):
    text = SPEC_YAML.replace('template_input: "seeds/<number>.png"\n', "")
    with pytest.raises(ValueError, match="missing required key 'template_input'"):
        load_video_spec(_write(tmp_path, text))


def test_load_video_spec_invalid_status_raises(tmp_path):
    text = SPEC_YAML.replace("status: live", "status: bogus")
    with pytest.raises(ValueError, match="status must be one of"):
        load_video_spec(_write(tmp_path, text))


def test_load_video_spec_negative_loop_raises(tmp_path):
    text = SPEC_YAML.replace("loop: 2", "loop: -1")
    with pytest.raises(ValueError, match="loop must be >= 0"):
        load_video_spec(_write(tmp_path, text))


def test_load_video_spec_no_negative_prompt_defaults_to_none(tmp_path):
    text = SPEC_YAML.replace('negative_prompt: "blurry"\n', "")
    spec = load_video_spec(_write(tmp_path, text))
    assert spec.negative_prompt is None


def test_validate_rejects_undefined_template_input_placeholder():
    with pytest.raises(ValueError, match="template_input references"):
        _spec(template_input="<unknown>.png")


def test_validate_rejects_undefined_template_prompt_placeholder():
    with pytest.raises(ValueError, match="template_prompt references undefined"):
        _spec(template_prompt="<missing>")


# --------------------------------------------------------------------------- #
# resolve_prompt (re-exported, exercised here against a VideoVarSpec)
# --------------------------------------------------------------------------- #
def test_resolve_prompt_works_on_video_var_spec():
    spec = _spec(negative_prompt="low quality")
    prompt, negative, chosen = resolve_prompt(spec, random.Random(0))
    assert prompt == "a cat"
    assert negative == "low quality"
    assert chosen == {"subject": "a cat"}


# --------------------------------------------------------------------------- #
# render_input
# --------------------------------------------------------------------------- #
def test_render_input_plain_path_returned_as_is(tmp_path):
    img = tmp_path / "cat.png"
    img.write_bytes(b"fake-png")
    result = render_input(str(img), number=0, seed=0, rng=random.Random(0))
    assert result == img


def test_render_input_plain_path_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="input image not found"):
        render_input(str(tmp_path / "missing.png"), number=0, seed=0, rng=random.Random(0))


def test_render_input_counter_template_resolves(tmp_path):
    img = tmp_path / "img_0000000003.png"
    img.write_bytes(b"fake-png")
    template = str(tmp_path / "img_<number>.png")
    result = render_input(template, number=3, seed=0, rng=random.Random(0))
    assert result == img


def test_render_input_counter_template_missing_raises(tmp_path):
    template = str(tmp_path / "img_<number>.png")
    with pytest.raises(FileNotFoundError, match="input image not found"):
        render_input(template, number=0, seed=0, rng=random.Random(0))


def test_render_input_glob_picks_an_existing_match(tmp_path):
    (tmp_path / "a.png").write_bytes(b"a")
    (tmp_path / "b.png").write_bytes(b"b")
    template = str(tmp_path / "*.png")
    result = render_input(template, number=0, seed=0, rng=random.Random(0))
    assert result in {tmp_path / "a.png", tmp_path / "b.png"}


def test_render_input_glob_no_match_raises(tmp_path):
    template = str(tmp_path / "*.png")
    with pytest.raises(FileNotFoundError, match="no input images match glob"):
        render_input(template, number=0, seed=0, rng=random.Random(0))


def test_render_input_directory_picks_an_image(tmp_path):
    (tmp_path / "a.png").write_bytes(b"a")
    (tmp_path / "notes.txt").write_bytes(b"not an image")
    result = render_input(str(tmp_path), number=0, seed=0, rng=random.Random(0))
    assert result == tmp_path / "a.png"


def test_render_input_directory_no_images_raises(tmp_path):
    (tmp_path / "notes.txt").write_bytes(b"not an image")
    with pytest.raises(FileNotFoundError, match="no input images in directory"):
        render_input(str(tmp_path), number=0, seed=0, rng=random.Random(0))
