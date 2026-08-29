"""LLM-driven prompt variation for `generate-similar --vary`.

Calls an OpenAI-compatible chat completion endpoint to produce a new prompt
(and optionally negative prompt) per generated image, guided by a base prompt,
an explicit user instruction, and an optional vocabulary of thematic options
drawn from a `generate-var`-format spec file.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    from openai import APIConnectionError, OpenAI
except ImportError:  # pragma: no cover - exercised only when the SDK is absent
    OpenAI = None  # type: ignore[assignment,misc]

    class APIConnectionError(Exception):  # type: ignore[no-redef]
        """Placeholder so generate_variation_with_retry can still reference this
        type when the openai SDK isn't installed; generate_variation() would
        already have raised RuntimeError before ever reaching that code."""


if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from .variables import load_spec

ENV_BASE_URL = "IMAGEGEN_MODEL_BASE_URL"
ENV_API_KEY = "IMAGEGEN_MODEL_API_KEY"
ENV_MODEL_NAME = "IMAGEGEN_MODEL_NAME"

# Placeholder used by OpenAI-compatible backends that don't check the key at all.
_NO_KEY_PLACEHOLDER = "not-needed"

# Matches the first brace-delimited JSON object in a larger text blob.
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# Best-effort salvage for JSON truncated mid-response (e.g. hit the token
# limit): pull the "prompt"/"negative_prompt" string values directly, without
# requiring a closing brace or even a closing quote.
_PROMPT_FIELD_RE = re.compile(r'"prompt"\s*:\s*"((?:\\.|[^"\\])*)"?', re.DOTALL)
_NEGATIVE_FIELD_RE = re.compile(r'"negative_prompt"\s*:\s*"((?:\\.|[^"\\])*)"?', re.DOTALL)
# Default output budget: SDXL prompts are often long, dense tag lists; a low
# max_tokens truncates the LLM's JSON response mid-string.
_DEFAULT_MAX_TOKENS = 4096

_SYSTEM_PROMPT = (
    "You are a prompt-variation assistant for an SDXL image generation tool. "
    "Given a base prompt, an optional base negative prompt, an optional "
    "vocabulary of thematic options, and a user request describing the kind "
    "of variation wanted, produce ONE new variation.\n\n"
    "Rules:\n"
    "- Keep the same subject, character identity, and overall style unless the "
    "user request explicitly asks to change them.\n"
    "- Only vary what the user request asks for; do not rewrite unrelated parts "
    "of the prompt.\n"
    "- If a vocabulary is provided, prefer drawing replacement terms from it for "
    "consistency with the rest of the project's prompt style.\n"
    "- Avoid repeating any variation already listed as a previous variation.\n"
    "- Respond with STRICT JSON ONLY, no markdown fences, no commentary, "
    'matching exactly: {"prompt": "...", "negative_prompt": "..." or null}. '
    "Set negative_prompt to null when it should stay unchanged."
)

# Structured-output tool: forcing the call through this schema (instead of the
# looser response_format={"type": "json_object"}) makes backends that support
# tool calling enforce prompt/negative_prompt as plain strings, which is the
# single most common way an OpenAI-compatible backend violates the format
# (returning a nested object/array for negative_prompt instead of a string).
_VARIATION_TOOL_NAME = "submit_prompt_variation"
_VARIATION_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": _VARIATION_TOOL_NAME,
        "description": "Submit the new prompt variation.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "The new positive prompt."},
                "negative_prompt": {
                    "type": ["string", "null"],
                    "description": "The new negative prompt, or null to keep the base one unchanged.",
                },
            },
            "required": ["prompt", "negative_prompt"],
            "additionalProperties": False,
        },
    },
}


class LLMFormatError(ValueError):
    """The LLM's response violated the expected prompt/negative_prompt shape.

    A ValueError subclass, so existing `except ValueError` handling keeps
    working unchanged. Carries the raw offending content so
    generate_variation_with_retry can echo it back to the model as part of a
    corrective follow-up message on retry.
    """

    def __init__(self, message: str, raw_content: str):
        super().__init__(message)
        self.raw_content = raw_content


@dataclass
class LLMConfig:
    """Resolved connection settings for the OpenAI-compatible endpoint."""

    base_url: str
    api_key: str
    model: str


@dataclass
class VariationResult:
    """One LLM-produced prompt variation."""

    prompt: str
    negative_prompt: str | None


def resolve_llm_config(
    base_url: str | None,
    api_key: str | None,
    model: str | None,
) -> LLMConfig:
    """Resolve the LLM endpoint config: CLI overrides win over environment.

    Args:
        base_url: CLI override for the API base URL, or None.
        api_key: CLI override for the API key, or None.
        model: CLI override for the model name, or None.

    Returns:
        A fully resolved LLMConfig.

    Raises:
        ValueError: If base_url or model is neither given nor set in the
            environment.
    """
    resolved_base_url = base_url or os.environ.get(ENV_BASE_URL)
    resolved_model = model or os.environ.get(ENV_MODEL_NAME)
    resolved_api_key = api_key or os.environ.get(ENV_API_KEY) or _NO_KEY_PLACEHOLDER

    missing = []
    if not resolved_base_url:
        missing.append(f"--llm-base-url or ${ENV_BASE_URL}")
    if not resolved_model:
        missing.append(f"--llm-model or ${ENV_MODEL_NAME}")
    if missing:
        raise ValueError("generate-similar --vary requires an LLM endpoint. Set: " + "; ".join(missing))

    # `missing` is empty here, so both are guaranteed non-empty strings; narrow for mypy
    # without using `assert` (reserved for tests per project conventions).
    if resolved_base_url is None or resolved_model is None:
        raise RuntimeError("unreachable: missing was empty but a value is still None")
    return LLMConfig(base_url=resolved_base_url, api_key=resolved_api_key, model=resolved_model)


def format_vocabulary(vocab_path: str | Path, max_chars: int = 6000) -> str:
    """Format a generate-var spec's top-level variables as LLM inspiration text.

    Reuses the generate-var YAML parser (same spec format), then lists each
    top-level variable's deduplicated option values (not descending into
    nested sub-variables, to keep the result compact).

    Args:
        vocab_path: Path to a generate-var-format YAML spec.
        max_chars: Maximum length of the returned text; truncated with a
            trailing marker if exceeded.

    Returns:
        A compact, human-readable vocabulary listing.
    """
    spec = load_spec(vocab_path)
    lines = []
    for name, options in spec.variables.items():
        seen: list[str] = []
        for opt in options:
            if opt.value and opt.value not in seen:
                seen.append(opt.value)
        if seen:
            lines.append(f"- {name}: {'; '.join(seen)}")

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n...[truncated]"
    return text


def _build_user_message(
    base_prompt: str,
    base_negative: str | None,
    user_request: str,
    vocabulary: str | None,
    previous_prompts: list[str],
) -> str:
    parts = [f"BASE_PROMPT:\n{base_prompt}"]
    if base_negative:
        parts.append(f"BASE_NEGATIVE_PROMPT:\n{base_negative}")
    parts.append(f"USER_REQUEST:\n{user_request}")
    if vocabulary:
        parts.append(f"VOCABULARY:\n{vocabulary}")
    if previous_prompts:
        joined = "\n".join(f"- {p}" for p in previous_prompts)
        parts.append(f"PREVIOUS_VARIATIONS (avoid repeating these):\n{joined}")
    return "\n\n".join(parts)


def _corrective_message(attempt: int, exc: Exception) -> str:
    """Follow-up user message pointing out a repeated format mistake.

    Sent alongside an echo of the LLM's own (invalid) previous response, so
    the model can see exactly what it got wrong instead of blindly repeating
    the same mistake on an unchanged prompt.
    """
    return (
        f"That response is invalid: {exc}\n\n"
        f"This is the {attempt}{_ordinal_suffix(attempt)} time you have made this exact kind of mistake. "
        'Re-validate your output carefully before answering: "prompt" and "negative_prompt" MUST be plain '
        "strings (or null for negative_prompt), never an object, array, or any other JSON type. Respond with "
        "the corrected JSON only, no markdown fences, no commentary."
    )


def _ordinal_suffix(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _unescape_json_string(raw: str) -> str:
    """Best-effort unescape of a JSON string fragment (may be unterminated)."""
    try:
        return json.loads(f'"{raw}"')
    except json.JSONDecodeError:
        # Likely cut off mid-escape (trailing lone backslash); drop it and retry once.
        try:
            return json.loads(f'"{raw.rstrip(chr(92))}"')
        except json.JSONDecodeError:
            return raw


def _salvage_truncated_json(content: str) -> dict | None:
    """Recover prompt/negative_prompt from a JSON response cut off mid-string.

    Returns None if no "prompt" field can be found at all.
    """
    match = _PROMPT_FIELD_RE.search(content)
    if not match:
        return None
    prompt = _unescape_json_string(match.group(1))

    negative = None
    neg_match = _NEGATIVE_FIELD_RE.search(content)
    if neg_match:
        negative = _unescape_json_string(neg_match.group(1))

    print(
        "Warning: LLM response was truncated (likely hit the output token limit); using the partial prompt as-is.",
        file=sys.stderr,
    )
    return {"prompt": prompt, "negative_prompt": negative}


def _extract_json(content: str) -> dict:
    """Parse strict JSON, falling back to extracting the first {...} block,
    then to salvaging a truncated response."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = _JSON_BLOCK_RE.search(content)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    salvaged = _salvage_truncated_json(content)
    if salvaged is not None:
        return salvaged

    excerpt = content[:500]
    raise LLMFormatError(f"LLM did not return valid JSON. Response excerpt: {excerpt!r}", raw_content=content)


def _call_llm(
    client: OpenAI,
    config: LLMConfig,
    messages: list[ChatCompletionMessageParam],
    temperature: float,
) -> str:
    """Call chat.completions.create, preferring structured tool calling.

    Tries, in order, until one succeeds at the HTTP level:
    1. Forced tool calling against _VARIATION_TOOL (schema-enforced strings).
    2. response_format={"type": "json_object"} (valid JSON, no schema).
    3. Plain completion (relies on _extract_json's regex fallback).

    Backends that don't support tools/response_format typically reject the
    request outright (e.g. 400), which is what triggers each fallback step;
    a request that *is* accepted is trusted and not retried at this level.

    Returns the raw text to parse: the tool call's JSON arguments if the
    model used the tool, else the message content.
    """
    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=_DEFAULT_MAX_TOKENS,
            tools=[_VARIATION_TOOL],
            tool_choice={"type": "function", "function": {"name": _VARIATION_TOOL_NAME}},
        )
    except Exception:
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=temperature,
                max_tokens=_DEFAULT_MAX_TOKENS,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=temperature,
                max_tokens=_DEFAULT_MAX_TOKENS,
            )

    message = response.choices[0].message
    if message.tool_calls:
        # We only ever request our one function tool (_VARIATION_TOOL), so the
        # first tool call is always the function variant; the SDK's union
        # type also allows a "custom" tool call, which we never trigger.
        tool_call: Any = message.tool_calls[0]
        return tool_call.function.arguments
    return message.content or ""


def generate_variation(
    config: LLMConfig,
    base_prompt: str,
    base_negative: str | None,
    user_request: str,
    vocabulary: str | None,
    previous_prompts: list[str],
    temperature: float = 1.0,
    extra_messages: list[ChatCompletionMessageParam] | None = None,
) -> VariationResult:
    """Call the LLM once to produce one new prompt (and optional negative) variation.

    Args:
        config: Resolved LLM endpoint config.
        base_prompt: The source image's prompt.
        base_negative: The source image's negative prompt, if any.
        user_request: Explicit instruction on the kind of variation wanted.
        vocabulary: Optional vocabulary text (see format_vocabulary), or None.
        previous_prompts: Prompts already produced earlier in this run, so the
            LLM can avoid repeating them.
        temperature: Sampling temperature (diversity).
        extra_messages: Additional messages appended after the initial user
            message (e.g. a previous invalid response plus a corrective
            follow-up), so a retry can continue the same conversation instead
            of repeating an identical request. None on a first attempt.

    Returns:
        The new prompt and negative prompt (None means "keep base_negative").

    Raises:
        RuntimeError: If the `openai` package is not installed.
        LLMFormatError: If the LLM response violates the expected shape
            (invalid JSON, or a non-string prompt/negative_prompt).
    """
    if OpenAI is None:
        raise RuntimeError(
            "The 'openai' package is required for --vary. Run 'uv sync' (it is "
            "declared in pyproject.toml) or 'pip install openai'."
        )

    client = OpenAI(base_url=config.base_url, api_key=config.api_key)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_user_message(base_prompt, base_negative, user_request, vocabulary, previous_prompts),
        },
    ]
    if extra_messages:
        messages.extend(extra_messages)

    content = _call_llm(client, config, messages, temperature)
    data = _extract_json(content)

    raw_prompt = data.get("prompt")
    if raw_prompt is not None and not isinstance(raw_prompt, str):
        # Some backends don't strictly follow the requested JSON schema and
        # return a nested object/array instead of a plain string. Passing
        # that through would eventually crash the prompt encoder (it expects
        # str), so treat it like malformed JSON: raise to trigger a retry.
        raise LLMFormatError(f'LLM returned a non-string "prompt" ({type(raw_prompt).__name__}).', raw_content=content)
    prompt = raw_prompt or base_prompt

    raw_negative = data.get("negative_prompt")
    if raw_negative is not None and not isinstance(raw_negative, str):
        raise LLMFormatError(
            f'LLM returned a non-string "negative_prompt" ({type(raw_negative).__name__}).', raw_content=content
        )
    return VariationResult(prompt=prompt, negative_prompt=raw_negative)


def generate_variation_with_retry(
    config: LLMConfig,
    base_prompt: str,
    base_negative: str | None,
    user_request: str,
    vocabulary: str | None,
    previous_prompts: list[str],
    temperature: float = 1.0,
    max_attempts: int = 20,
    max_backoff: float = 120.0,
    max_format_attempts: int = 5,
    max_format_backoff: float = 5.0,
) -> VariationResult:
    """Call generate_variation, retrying on connection or format failures.

    Two independent retry budgets, because the two failure kinds have very
    different odds of a retry helping:

    - APIConnectionError (dropped connection, DNS failure, timeout, ...) is
      genuinely transient and unrelated to what was asked, so it gets the
      full budget: up to `max_attempts` tries with exponential backoff
      (1s, 2s, 4s, ... capped at `max_backoff`).
    - A format violation (invalid JSON, or a non-string prompt/negative_prompt)
      is a mistake by the model itself; blindly repeating the identical
      request rarely fixes it. It gets a much smaller budget
      (`max_format_attempts`, short `max_format_backoff`), and each retry
      appends the model's own invalid response plus a corrective follow-up
      message calling out the repeated mistake, so the model can actually
      self-correct instead of just rolling the dice again.

    Other failures (e.g. RuntimeError for a missing SDK, auth errors)
    propagate immediately since retrying them can't help.

    Args:
        config: Resolved LLM endpoint config.
        base_prompt: The source image's prompt.
        base_negative: The source image's negative prompt, if any.
        user_request: Explicit instruction on the kind of variation wanted.
        vocabulary: Optional vocabulary text, or None.
        previous_prompts: Prompts already produced earlier in this run.
        temperature: Sampling temperature (diversity).
        max_attempts: Maximum connection-error attempts before giving up
            (default: 20).
        max_backoff: Cap, in seconds, on the connection-error backoff delay
            (default: 120s / 2 minutes).
        max_format_attempts: Maximum format-error attempts before giving up
            (default: 5).
        max_format_backoff: Cap, in seconds, on the format-error backoff
            delay (default: 5s).

    Returns:
        The new prompt and negative prompt.

    Raises:
        ValueError: If every format-error attempt fails.
        APIConnectionError: If every connection attempt fails.
        RuntimeError: If the `openai` package is not installed.
    """
    last_error: ValueError | APIConnectionError | None = None
    extra_messages: list[ChatCompletionMessageParam] = []
    connection_attempt = 0
    format_attempt = 0

    while True:
        try:
            return generate_variation(
                config,
                base_prompt,
                base_negative,
                user_request,
                vocabulary,
                previous_prompts,
                temperature=temperature,
                extra_messages=extra_messages or None,
            )
        except APIConnectionError as exc:
            last_error = exc
            connection_attempt += 1
            if connection_attempt >= max_attempts:
                break
            delay = min(2 ** (connection_attempt - 1), max_backoff)
            print(
                f"Warning: LLM variation attempt {connection_attempt}/{max_attempts} failed "
                f"(connection: {exc}); retrying in {delay:.0f}s...",
                file=sys.stderr,
            )
            time.sleep(delay)
        except ValueError as exc:
            last_error = exc
            format_attempt += 1
            if format_attempt >= max_format_attempts:
                break
            delay = min(2 ** (format_attempt - 1), max_format_backoff)
            print(
                f"Warning: LLM variation attempt {format_attempt}/{max_format_attempts} failed "
                f"(format: {exc}); retrying in {delay:.0f}s...",
                file=sys.stderr,
            )
            if isinstance(exc, LLMFormatError):
                # Echo the invalid response back and call out the repeated
                # mistake, instead of just resending the identical request.
                extra_messages.append({"role": "assistant", "content": exc.raw_content})
                extra_messages.append({"role": "user", "content": _corrective_message(format_attempt, exc)})
            time.sleep(delay)

    if last_error is None:
        # unreachable: both budgets are >= 1, so the loop above always runs
        # at least once and sets last_error before falling through here.
        raise RuntimeError("unreachable: no attempt was made")
    if isinstance(last_error, APIConnectionError):
        # Re-raise as-is: it's already an accurate, specific error (unlike
        # the format-error case below, wrapping would just lose information).
        raise last_error
    raise ValueError(
        f"LLM did not return a valid response after {format_attempt} format-corrective attempts "
        f"(plus {connection_attempt} connection retries). Last error: {last_error}"
    ) from last_error
