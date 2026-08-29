"""Tests for LLM-driven prompt variation (generate-similar --vary)."""

import json

import pytest

from image_gen import llm_variation
from image_gen.llm_variation import (
    LLMConfig,
    VariationResult,
    format_vocabulary,
    generate_variation,
    generate_variation_with_retry,
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


def test_generate_variation_non_string_negative_prompt_raises(monkeypatch):
    content = json.dumps({"prompt": "a cat in a hat", "negative_prompt": {"tags": ["blurry"]}})
    fake = _FakeOpenAI([content])
    monkeypatch.setattr(llm_variation, "OpenAI", fake)

    with pytest.raises(ValueError, match="non-string"):
        generate_variation(_config(), "a cat", "blurry", "add a hat", None, [])


def test_generate_variation_non_string_prompt_raises(monkeypatch):
    content = json.dumps({"prompt": ["a", "cat"], "negative_prompt": "blurry"})
    fake = _FakeOpenAI([content])
    monkeypatch.setattr(llm_variation, "OpenAI", fake)

    with pytest.raises(ValueError, match="non-string"):
        generate_variation(_config(), "a cat", None, "vary it", None, [])


def test_generate_variation_with_retry_recovers_from_non_string_negative_prompt(monkeypatch):
    bad = json.dumps({"prompt": "a cat in a hat", "negative_prompt": {"tags": ["blurry"]}})
    good = json.dumps({"prompt": "a cat in a hat", "negative_prompt": "blurry, low quality"})
    fake = _FakeOpenAI([bad, good])
    monkeypatch.setattr(llm_variation, "OpenAI", fake)
    monkeypatch.setattr(llm_variation.time, "sleep", lambda _seconds: None)

    result = generate_variation_with_retry(_config(), "a cat", "blurry", "add a hat", None, [])

    assert result.prompt == "a cat in a hat"
    assert result.negative_prompt == "blurry, low quality"
    assert len(fake.chat.completions.calls) == 2


def test_generate_variation_salvages_truncated_json(monkeypatch, capsys):
    content = '```json\n{"prompt": "a cat wearing a tiny hat, detailed fur, high'
    fake = _FakeOpenAI([content])
    monkeypatch.setattr(llm_variation, "OpenAI", fake)

    result = generate_variation(_config(), "a cat", None, "vary it", None, [])

    assert result.prompt == "a cat wearing a tiny hat, detailed fur, high"
    assert result.negative_prompt is None
    assert "truncated" in capsys.readouterr().err


def test_generate_variation_salvages_truncated_json_with_negative(monkeypatch):
    content = '{"prompt": "a dog", "negative_prompt": "blur'
    fake = _FakeOpenAI([content])
    monkeypatch.setattr(llm_variation, "OpenAI", fake)

    result = generate_variation(_config(), "a dog", None, "vary it", None, [])

    assert result.prompt == "a dog"
    assert result.negative_prompt == "blur"


def test_generate_variation_sends_max_tokens(monkeypatch):
    fake = _FakeOpenAI([json.dumps({"prompt": "a cat variation"})])
    monkeypatch.setattr(llm_variation, "OpenAI", fake)

    generate_variation(_config(), "a cat", None, "vary it", None, [])

    assert fake.chat.completions.calls[0]["max_tokens"] == llm_variation._DEFAULT_MAX_TOKENS


def test_generate_variation_without_openai_installed_raises(monkeypatch):
    monkeypatch.setattr(llm_variation, "OpenAI", None)

    with pytest.raises(RuntimeError, match="openai' package is required"):
        generate_variation(_config(), "a cat", None, "vary it", None, [])


# --------------------------------------------------------------------------- #
# generate_variation_with_retry
# --------------------------------------------------------------------------- #
def test_generate_variation_with_retry_succeeds_after_failures(monkeypatch):
    calls = {"n": 0}
    sleeps = []

    def _fake(config, base_prompt, base_negative, user_request, vocabulary, previous, temperature):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("LLM did not return valid JSON. Response excerpt: 'oops'")
        return VariationResult(prompt="ok prompt", negative_prompt=None)

    monkeypatch.setattr(llm_variation, "generate_variation", _fake)
    monkeypatch.setattr(llm_variation.time, "sleep", lambda s: sleeps.append(s))

    result = generate_variation_with_retry(_config(), "a cat", None, "vary it", None, [])

    assert result.prompt == "ok prompt"
    assert calls["n"] == 3
    assert sleeps == [1, 2]


def test_generate_variation_with_retry_backoff_capped(monkeypatch):
    sleeps = []

    def _always_fails(config, base_prompt, base_negative, user_request, vocabulary, previous, temperature):
        raise ValueError("bad json")

    monkeypatch.setattr(llm_variation, "generate_variation", _always_fails)
    monkeypatch.setattr(llm_variation.time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(ValueError, match="after 20 attempts"):
        generate_variation_with_retry(_config(), "a cat", None, "vary it", None, [])

    assert len(sleeps) == 19
    assert max(sleeps) == 15
    assert sleeps[:5] == [1, 2, 4, 8, 15]


def test_generate_variation_with_retry_does_not_retry_runtime_error(monkeypatch):
    calls = {"n": 0}

    def _fake(config, base_prompt, base_negative, user_request, vocabulary, previous, temperature):
        calls["n"] += 1
        raise RuntimeError("no openai package")

    monkeypatch.setattr(llm_variation, "generate_variation", _fake)

    with pytest.raises(RuntimeError, match="no openai package"):
        generate_variation_with_retry(_config(), "a cat", None, "vary it", None, [])

    assert calls["n"] == 1
