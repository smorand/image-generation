"""Tests for the generate-var variable engine (no model needed)."""

import random

import pytest

from image_gen.variables import (
    clean_prompt,
    load_spec,
    render_output,
    resolve_prompt,
)

SPEC_YAML = """\
status: live
loop: 0
template_output: "out/img_<number>.png"
template_prompt: "solo, <eth> girl, <hair>, <clothes>, <place>, photography"
negative_prompt: "low quality"
defaults:
  model: "m.safetensors"
  steps: 25
variables:
  eth: [inuit]
  hair: ["long black hair", "short black hair"]
  clothes:
    - "wearing a dress"
    - value: "wearing <top> with <bottom>"
      weight: 3
      variables:
        top:
          - value: "<color> tank top"
            variables:
              color:
                - {value: "", weight: 5}
                - blue
                - red
        bottom: ["jeans", "a skirt"]
  place: ["in a meadow", "on a beach"]
"""


def _write(tmp_path, text):
    p = tmp_path / "spec.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_spec_parses_header(tmp_path):
    spec = load_spec(_write(tmp_path, SPEC_YAML))
    assert spec.status == "live"
    assert spec.loop == 0
    assert spec.template_output == "out/img_<number>.png"
    assert spec.defaults["steps"] == 25
    assert set(spec.variables) == {"eth", "hair", "clothes", "place"}


def test_resolve_produces_valid_prompt(tmp_path):
    spec = load_spec(_write(tmp_path, SPEC_YAML))
    rng = random.Random(0)
    prompt, _neg, chosen = resolve_prompt(spec, rng)
    assert prompt.startswith("solo, inuit girl,")
    assert "  " not in prompt  # no double spaces
    assert ", ," not in prompt  # no double commas
    assert chosen["eth"] == "inuit"
    assert chosen["hair"] in {"long black hair", "short black hair"}


def test_empty_option_is_cleaned(tmp_path):
    # Force the empty-color branch: weight 5 on "" dominates over many draws.
    spec = load_spec(_write(tmp_path, SPEC_YAML))
    rng = random.Random(1)
    saw_empty_color = False
    for _ in range(200):
        prompt, _neg, chosen = resolve_prompt(spec, rng)
        if chosen.get("color") == "":
            saw_empty_color = True
            assert "tank top" in prompt
            assert "  " not in prompt
            assert ", ," not in prompt
    assert saw_empty_color


def test_weights_bias_selection(tmp_path):
    spec = load_spec(_write(tmp_path, SPEC_YAML))
    rng = random.Random(42)
    empty = colored = 0
    for _ in range(2000):
        _, _neg, chosen = resolve_prompt(spec, rng)
        if "color" in chosen:
            if chosen["color"] == "":
                empty += 1
            else:
                colored += 1
    # "" has weight 5 vs blue(1)+red(1): expect roughly 5:2, so clearly more empty.
    assert empty > colored


def test_recursive_subvariables(tmp_path):
    spec = load_spec(_write(tmp_path, SPEC_YAML))
    rng = random.Random(7)
    for _ in range(100):
        prompt, _neg, chosen = resolve_prompt(spec, rng)
        if "top" in chosen:
            # recursive: clothes -> top(<color> tank top) -> color
            assert "tank top" in chosen["top"]
            assert "with" in prompt
            assert chosen["bottom"] in {"jeans", "a skirt"}


def test_negative_prompt_placeholders_resolved(tmp_path):
    """Placeholders in negative_prompt are drawn, substituted, and cleaned."""
    from image_gen.variables import VarSpec, Option

    spec = VarSpec(
        template_prompt="girl, <sizefix>, end",
        template_output="o_<number>.png",
        negative_prompt="bad quality, <massfix>",
        variables={
            "sizefix": [Option("(petite:1.3)")],
            "massfix": [Option("(muscular:1.3), (broad shoulders:1.3)")],
        },
    )
    prompt, negative, chosen = resolve_prompt(spec, random.Random(0))
    assert prompt == "girl, (petite:1.3), end"
    assert negative == "bad quality, (muscular:1.3), (broad shoulders:1.3)"
    assert chosen["massfix"] == "(muscular:1.3), (broad shoulders:1.3)"
    # An empty negative option must be cleaned (no dangling comma).
    spec_empty = VarSpec(
        template_prompt="girl",
        template_output="o_<number>.png",
        negative_prompt="bad quality, <massfix>",
        variables={"massfix": [Option("")]},
    )
    _, neg_empty, _ = resolve_prompt(spec_empty, random.Random(0))
    assert neg_empty == "bad quality"


def test_variable_shared_between_prompt_and_negative():
    """A variable used in both prompt and negative draws once (shared memo)."""
    from image_gen.variables import VarSpec, Option

    spec = VarSpec(
        template_prompt="a <col> shirt",
        template_output="o_<number>.png",
        negative_prompt="not a <col> shirt",
        variables={"col": [Option("blue"), Option("red")]},
    )
    for seed in range(20):
        prompt, negative, chosen = resolve_prompt(spec, random.Random(seed))
        col = chosen["col"]
        assert prompt == f"a {col} shirt"
        assert negative == f"not a {col} shirt"


def test_negative_unknown_placeholder_rejected():
    from image_gen.variables import VarSpec, Option

    with pytest.raises(ValueError, match="negative_prompt references undefined <typo>"):
        VarSpec(
            template_prompt="a <col> b",
            template_output="o_<number>.png",
            negative_prompt="bad, <typo>",
            variables={"col": [Option("blue")]},
        )


def test_repeated_variable_reused():
    from image_gen.variables import VarSpec, Option

    spec = VarSpec(
        template_prompt="<hair> and again <hair>",
        template_output="o_<number>.png",
        variables={"hair": [Option("black"), Option("blonde")]},
    )
    for seed in range(20):
        prompt, _neg, chosen = resolve_prompt(spec, random.Random(seed))
        color = chosen["hair"]
        assert prompt == f"{color} and again {color}"


def test_unknown_placeholder_rejected():
    from image_gen.variables import VarSpec, Option

    with pytest.raises(ValueError, match="undefined <missing>"):
        VarSpec(
            template_prompt="a <missing> b",
            template_output="o_<number>.png",
            variables={"eth": [Option("inuit")]},
        )


def test_output_template_only_allows_builtins():
    from image_gen.variables import VarSpec, Option

    with pytest.raises(ValueError, match="template_output references <foo>"):
        VarSpec(
            template_prompt="<eth>",
            template_output="o_<foo>.png",
            variables={"eth": [Option("inuit")]},
        )


def test_render_output_pads_number():
    assert render_output("out/img_<number>.png", 7, 123) == "out/img_0000000007.png"
    assert render_output("img_<number>_<seed>.png", 42, 999) == "img_0000000042_999.png"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("solo,  , natural", "solo, natural"),
        ("a, , , b", "a, b"),
        (", leading, trailing,", "leading, trailing"),
        ("wearing  tank top", "wearing tank top"),
        ("a ,b ,c", "a, b, c"),
        ("keep, one, comma", "keep, one, comma"),
    ],
)
def test_clean_prompt(raw, expected):
    assert clean_prompt(raw) == expected
