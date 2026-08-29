"""Tests for LLM-driven prompt variation (generate-similar --vary)."""

import json

import httpx
import pytest

from image_gen import llm_variation
from image_gen.llm_variation import (
    LLMConfig,
    LLMFormatError,
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
# generate_variation (fakes: tool_calls=None means "no tool call", falls back
# to plain message content, exactly like a backend that ignored `tools`).
# --------------------------------------------------------------------------- #
class _FakeFunction:
    def __init__(self, arguments):
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, arguments):
        self.function = _FakeFunction(arguments)


class _FakeMessage:
    def __init__(self, content=None, tool_call_arguments=None):
        self.content = content
        self.tool_calls = [_FakeToolCall(tool_call_arguments)] if tool_call_arguments is not None else None


class _FakeChoice:
    def __init__(self, content=None, tool_call_arguments=None):
        self.message = _FakeMessage(content, tool_call_arguments)


class _FakeResponse:
    def __init__(self, content=None, tool_call_arguments=None):
        self.choices = [_FakeChoice(content, tool_call_arguments)]


class _FakeCompletions:
    """Stand-in for client.chat.completions with scripted responses.

    Each scripted item is either:
    - a plain string: a text-content response (as if `tools` were ignored),
    - a ("tool", json_string) tuple: a tool-call response,
    - an Exception instance: the call raises it (simulating a backend that
      rejects `tools` or `response_format` outright).
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, tuple) and item[0] == "tool":
            return _FakeResponse(tool_call_arguments=item[1])
        return _FakeResponse(content=item)


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


def test_generate_variation_uses_tool_call_response_by_default(monkeypatch):
    fake = _FakeOpenAI([("tool", json.dumps({"prompt": "a cat in a hat", "negative_prompt": None}))])
    monkeypatch.setattr(llm_variation, "OpenAI", fake)

    result = generate_variation(_config(), "a cat", "blurry", "add a hat", None, [])

    assert result.prompt == "a cat in a hat"
    assert result.negative_prompt is None
    assert "tools" in fake.chat.completions.calls[0]
    assert fake.chat.completions.calls[0]["tool_choice"]["function"]["name"] == "submit_prompt_variation"


def test_generate_variation_parses_clean_json_from_content(monkeypatch):
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


def test_generate_variation_falls_back_through_tools_then_response_format_then_plain(monkeypatch):
    fake = _FakeOpenAI(
        [
            RuntimeError("tools not supported"),
            RuntimeError("response_format not supported"),
            json.dumps({"prompt": "a cat variation"}),
        ]
    )
    monkeypatch.setattr(llm_variation, "OpenAI", fake)

    result = generate_variation(_config(), "a cat", None, "vary it", None, [])

    assert result.prompt == "a cat variation"
    calls = fake.chat.completions.calls
    assert len(calls) == 3
    assert "tools" in calls[0]
    assert "response_format" in calls[1] and "tools" not in calls[1]
    assert "tools" not in calls[2] and "response_format" not in calls[2]


def test_generate_variation_invalid_json_raises_llm_format_error(monkeypatch):
    fake = _FakeOpenAI(["not json at all, sorry"])
    monkeypatch.setattr(llm_variation, "OpenAI", fake)

    with pytest.raises(LLMFormatError, match="did not return valid JSON"):
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

    with pytest.raises(LLMFormatError, match="non-string"):
        generate_variation(_config(), "a cat", "blurry", "add a hat", None, [])


def test_generate_variation_non_string_prompt_raises(monkeypatch):
    content = json.dumps({"prompt": ["a", "cat"], "negative_prompt": "blurry"})
    fake = _FakeOpenAI([content])
    monkeypatch.setattr(llm_variation, "OpenAI", fake)

    with pytest.raises(LLMFormatError, match="non-string"):
        generate_variation(_config(), "a cat", None, "vary it", None, [])


def test_generate_variation_passes_extra_messages_through(monkeypatch):
    fake = _FakeOpenAI([json.dumps({"prompt": "a cat variation", "negative_prompt": None})])
    monkeypatch.setattr(llm_variation, "OpenAI", fake)
    extra = [{"role": "assistant", "content": "oops"}, {"role": "user", "content": "fix it"}]

    generate_variation(_config(), "a cat", None, "vary it", None, [], extra_messages=extra)

    sent_messages = fake.chat.completions.calls[0]["messages"]
    assert sent_messages[-2:] == extra


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
    # The retry's message history carries the bad response plus a corrective note.
    second_call_messages = fake.chat.completions.calls[1]["messages"]
    assert second_call_messages[-2] == {"role": "assistant", "content": bad}
    assert "1st time" in second_call_messages[-1]["content"]


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
def test_generate_variation_with_retry_succeeds_after_format_failures(monkeypatch):
    calls = {"n": 0}
    sleeps = []

    def _fake(config, base_prompt, base_negative, user_request, vocabulary, previous, temperature, extra_messages=None):
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


def test_generate_variation_with_retry_format_budget_capped_by_default(monkeypatch):
    sleeps = []

    def _always_fails(
        config, base_prompt, base_negative, user_request, vocabulary, previous, temperature, extra_messages=None
    ):
        raise ValueError("bad json")

    monkeypatch.setattr(llm_variation, "generate_variation", _always_fails)
    monkeypatch.setattr(llm_variation.time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(ValueError, match="5 format-corrective attempts"):
        generate_variation_with_retry(_config(), "a cat", None, "vary it", None, [])

    # Small, fast-failing budget: format mistakes rarely self-correct by
    # brute-force repetition, so this must NOT burn the full 20x/120s budget
    # reserved for genuinely transient connection errors.
    assert len(sleeps) == 4  # max_format_attempts=5 -> 4 sleeps before giving up
    assert max(sleeps) == 5  # capped at max_format_backoff=5s
    assert sleeps == [1, 2, 4, 5]


def test_generate_variation_with_retry_sends_corrective_feedback_with_growing_count(monkeypatch):
    calls = []

    def _always_fails(
        config, base_prompt, base_negative, user_request, vocabulary, previous, temperature, extra_messages=None
    ):
        # Snapshot: extra_messages is the same list object mutated in place
        # across attempts, so record a copy, not a reference.
        calls.append(list(extra_messages) if extra_messages is not None else None)
        raise LLMFormatError("bad shape", raw_content="{'bad': true}")

    monkeypatch.setattr(llm_variation, "generate_variation", _always_fails)
    monkeypatch.setattr(llm_variation.time, "sleep", lambda s: None)

    with pytest.raises(ValueError):
        generate_variation_with_retry(_config(), "a cat", None, "vary it", None, [])

    assert calls[0] is None  # first attempt: no history yet
    # Each retry carries the growing conversation: prior (assistant, user) pairs.
    assert len(calls[1]) == 2
    assert len(calls[2]) == 4
    assert len(calls[3]) == 6
    assert calls[1][0] == {"role": "assistant", "content": "{'bad': true}"}
    assert "1st time" in calls[1][1]["content"]
    assert "2nd time" in calls[2][3]["content"]
    assert "3rd time" in calls[3][5]["content"]
    assert "times" not in calls[2][3]["content"]  # "2nd time", not "2nd times"


def test_generate_variation_with_retry_does_not_retry_runtime_error(monkeypatch):
    calls = {"n": 0}

    def _fake(config, base_prompt, base_negative, user_request, vocabulary, previous, temperature, extra_messages=None):
        calls["n"] += 1
        raise RuntimeError("no openai package")

    monkeypatch.setattr(llm_variation, "generate_variation", _fake)

    with pytest.raises(RuntimeError, match="no openai package"):
        generate_variation_with_retry(_config(), "a cat", None, "vary it", None, [])


def _connection_error(message):
    request = httpx.Request("POST", "http://x/v1/chat/completions")
    return llm_variation.APIConnectionError(message=message, request=request)


def test_generate_variation_with_retry_recovers_from_connection_error(monkeypatch):
    calls = {"n": 0}
    sleeps = []

    def _fake(config, base_prompt, base_negative, user_request, vocabulary, previous, temperature, extra_messages=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _connection_error("dropped connection")
        return VariationResult(prompt="ok prompt", negative_prompt=None)

    monkeypatch.setattr(llm_variation, "generate_variation", _fake)
    monkeypatch.setattr(llm_variation.time, "sleep", lambda s: sleeps.append(s))

    result = generate_variation_with_retry(_config(), "a cat", None, "vary it", None, [])

    assert result.prompt == "ok prompt"
    assert calls["n"] == 3
    assert sleeps == [1, 2]


def test_generate_variation_with_retry_connection_error_keeps_full_budget(monkeypatch):
    """Connection errors are unrelated to prompt content: no corrective message,
    and the full 20-attempt/120s budget applies (unlike format errors)."""
    sleeps = []

    def _always_fails(
        config, base_prompt, base_negative, user_request, vocabulary, previous, temperature, extra_messages=None
    ):
        assert extra_messages is None  # never grows the conversation for connection errors
        raise _connection_error("still down")

    monkeypatch.setattr(llm_variation, "generate_variation", _always_fails)
    monkeypatch.setattr(llm_variation.time, "sleep", lambda s: sleeps.append(s))

    # Unlike the format-error case, a connection error is re-raised as-is
    # (wrapping it as a ValueError would misrepresent what actually failed).
    with pytest.raises(llm_variation.APIConnectionError, match="still down"):
        generate_variation_with_retry(_config(), "a cat", None, "vary it", None, [])

    assert len(sleeps) == 19
    assert max(sleeps) == 120


def test_generate_variation_with_retry_default_budgets():
    import inspect

    sig = inspect.signature(generate_variation_with_retry)
    assert sig.parameters["max_attempts"].default == 20
    assert sig.parameters["max_backoff"].default == 120.0
    assert sig.parameters["max_format_attempts"].default == 5
    assert sig.parameters["max_format_backoff"].default == 5.0
