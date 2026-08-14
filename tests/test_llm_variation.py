"""Tests for LLM-driven prompt variation (generate-similar --vary)."""

import json

import pytest

from image_gen import llm_variation
from image_gen.llm_variation import (
    LLMConfig,
    format_vocabulary,
    generate_variation,
    resolve_llm_config,
)


# --------------------------------------------------------------------------- #
# resolve_llm_config
# --------------------------------------------------------------------------- #
def test_resolve_llm_config_cli_overrides_win(monkeypatch):
    monkeypatch.setenv("IMAGEGEN_MODEL_BASE_URL", "http://env-url/v1")
    monkeypatch.setenv("IMAGEGEN_MODEL_NAME", "env-model")
    monkeypatch.setenv("IMAGEGEN_MODEL_API_KEY", "env-key")

    config = resolve_llm_config("http://cli-url/v1", "cli-key", "cli-model")

    assert config == LLMConfig(base_url="http://cli-url/v1", api_key="cli-key", model="cli-model")


def test_resolve_llm_config_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("IMAGEGEN_MODEL_BASE_URL", "http://env-url/v1")
    monkeypatch.setenv("IMAGEGEN_MODEL_NAME", "env-model")
    monkeypatch.delenv("IMAGEGEN_MODEL_API_KEY", raising=False)

    config = resolve_llm_config(None, None, None)

    assert config.base_url == "http://env-url/v1"
    assert config.model == "env-model"
    assert config.api_key == "not-needed"


def test_resolve_llm_config_missing_base_url_and_model_raises(monkeypatch):
    monkeypatch.delenv("IMAGEGEN_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("IMAGEGEN_MODEL_NAME", raising=False)

    with pytest.raises(ValueError, match="IMAGEGEN_MODEL_BASE_URL"):
        resolve_llm_config(None, None, None)


# --------------------------------------------------------------------------- #
# format_vocabulary
# --------------------------------------------------------------------------- #
def _write_spec(tmp_path):
    spec = tmp_path / "vocab.yaml"
    spec.write_text(
        """
template_prompt: "<subject>, <light>"
template_output: "out/<number>.png"
variables:
  subject: ["a cat", "a dog", "a cat"]
  light: [{value: "golden hour", weight: 2}, "soft light"]
""",
        encoding="utf-8",
    )
    return spec


def test_format_vocabulary_lists_deduplicated_options(tmp_path):
    spec = _write_spec(tmp_path)
    text = format_vocabulary(spec)
    assert "- subject: a cat; a dog" in text
    assert "- light: golden hour; soft light" in text


def test_format_vocabulary_truncates(tmp_path):
    spec = _write_spec(tmp_path)
    text = format_vocabulary(spec, max_chars=10)
    assert len(text) <= 10 + len("\n...[truncated]")
    assert text.endswith("...[truncated]")


# --------------------------------------------------------------------------- #
# generate_variation
# --------------------------------------------------------------------------- #
class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    """Stand-in for client.chat.completions with scripted responses."""

    def __init__(self, responses):
        # responses: list of either a string (success) or an Exception instance to raise.
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item)


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeOpenAI:
    def __init__(self, responses):
        self.chat = _FakeChat(_FakeCompletions(responses))

    def __call__(self, **kwargs):
        # Mimic OpenAI(...) constructor: return self so tests can inspect calls.
        return self


def _config():
    return LLMConfig(base_url="http://x/v1", api_key="k", model="m")


def test_generate_variation_parses_clean_json(monkeypatch):
    fake = _FakeOpenAI([json.dumps({"prompt": "a cat in a hat", "negative_prompt": None})])
    monkeypatch.setattr(llm_variation, "OpenAI", fake)

    result = generate_variation(_config(), "a cat", "blurry", "add a hat", None, [])

    assert result.prompt == "a cat in a hat"
    assert result.negative_prompt is None


def test_generate_variation_extracts_json_from_surrounding_text(monkeypatch):
    content = 'Sure! Here is the variation:\n{"prompt": "a dog running", "negative_prompt": "static"}\nEnjoy.'
    fake = _FakeOpenAI([content])
    monkeypatch.setattr(llm_variation, "OpenAI", fake)

    result = generate_variation(_config(), "a dog", None, "make it run", None, [])

    assert result.prompt == "a dog running"
    assert result.negative_prompt == "static"


def test_generate_variation_retries_without_response_format(monkeypatch):
    fake = _FakeOpenAI(
        [
            RuntimeError("response_format not supported"),
            json.dumps({"prompt": "a cat variation"}),
        ]
    )
    monkeypatch.setattr(llm_variation, "OpenAI", fake)

    result = generate_variation(_config(), "a cat", None, "vary it", None, [])

    assert result.prompt == "a cat variation"
    calls = fake.chat.completions.calls
    assert len(calls) == 2
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


def test_generate_variation_invalid_json_raises(monkeypatch):
    fake = _FakeOpenAI(["not json at all, sorry"])
    monkeypatch.setattr(llm_variation, "OpenAI", fake)

    with pytest.raises(ValueError, match="did not return valid JSON"):
        generate_variation(_config(), "a cat", None, "vary it", None, [])


def test_generate_variation_missing_prompt_falls_back_to_base(monkeypatch):
    fake = _FakeOpenAI([json.dumps({"negative_prompt": "blurry"})])
    monkeypatch.setattr(llm_variation, "OpenAI", fake)

    result = generate_variation(_config(), "a cat", None, "vary it", None, [])

    assert result.prompt == "a cat"
    assert result.negative_prompt == "blurry"


def test_generate_variation_without_openai_installed_raises(monkeypatch):
    monkeypatch.setattr(llm_variation, "OpenAI", None)

    with pytest.raises(RuntimeError, match="openai' package is required"):
        generate_variation(_config(), "a cat", None, "vary it", None, [])
