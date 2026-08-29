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

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only when the SDK is absent
    OpenAI = None  # type: ignore[assignment,misc]

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
    raise ValueError(f"LLM did not return valid JSON. Response excerpt: {excerpt!r}")


def generate_variation(
    config: LLMConfig,
    base_prompt: str,
    base_negative: str | None,
    user_request: str,
    vocabulary: str | None,
    previous_prompts: list[str],
    temperature: float = 1.0,
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

    Returns:
        The new prompt and negative prompt (None means "keep base_negative").

    Raises:
        RuntimeError: If the `openai` package is not installed.
        ValueError: If the LLM response cannot be parsed as the expected JSON.
    """
    if OpenAI is None:
        raise RuntimeError(
            "The 'openai' package is required for --vary. Run 'uv sync' (it is "
            "declared in pyproject.toml) or 'pip install openai'."
        )

    client = OpenAI(base_url=config.base_url, api_key=config.api_key)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_user_message(base_prompt, base_negative, user_request, vocabulary, previous_prompts),
        },
    ]

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=_DEFAULT_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
    except Exception:
        # Not every OpenAI-compatible backend supports response_format; retry
        # without it and rely on _extract_json's regex fallback.
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=_DEFAULT_MAX_TOKENS,
        )

    content = response.choices[0].message.content or ""
    data = _extract_json(content)

    raw_prompt = data.get("prompt")
    if raw_prompt is not None and not isinstance(raw_prompt, str):
        # Some backends don't strictly follow the requested JSON schema and
        # return a nested object/array instead of a plain string. Passing
        # that through would eventually crash the prompt encoder (it expects
        # str), so treat it like malformed JSON: raise to trigger a retry.
        raise ValueError(f'LLM returned a non-string "prompt" ({type(raw_prompt).__name__}).')
    prompt = raw_prompt or base_prompt

    raw_negative = data.get("negative_prompt")
    if raw_negative is not None and not isinstance(raw_negative, str):
        raise ValueError(f'LLM returned a non-string "negative_prompt" ({type(raw_negative).__name__}).')
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
    max_backoff: float = 15.0,
) -> VariationResult:
    """Call generate_variation, retrying on invalid-JSON failures.

    Retries only on ValueError (invalid/unparsable JSON response), with
    exponential backoff (1s, 2s, 4s, ... capped at max_backoff). Other
    failures (e.g. RuntimeError for a missing SDK) propagate immediately.

    Args:
        config: Resolved LLM endpoint config.
        base_prompt: The source image's prompt.
        base_negative: The source image's negative prompt, if any.
        user_request: Explicit instruction on the kind of variation wanted.
        vocabulary: Optional vocabulary text, or None.
        previous_prompts: Prompts already produced earlier in this run.
        temperature: Sampling temperature (diversity).
        max_attempts: Maximum number of attempts before giving up.
        max_backoff: Cap, in seconds, on the exponential backoff delay.

    Returns:
        The new prompt and negative prompt.

    Raises:
        ValueError: If every attempt fails to produce valid JSON.
        RuntimeError: If the `openai` package is not installed.
    """
    last_error: ValueError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return generate_variation(
                config,
                base_prompt,
                base_negative,
                user_request,
                vocabulary,
                previous_prompts,
                temperature=temperature,
            )
        except ValueError as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            delay = min(2 ** (attempt - 1), max_backoff)
            print(
                f"Warning: LLM variation attempt {attempt}/{max_attempts} failed ({exc}); retrying in {delay:.0f}s...",
                file=sys.stderr,
            )
            time.sleep(delay)

    assert last_error is not None
    raise ValueError(
        f"LLM did not return valid JSON after {max_attempts} attempts. Last error: {last_error}"
    ) from last_error
