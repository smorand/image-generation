# generate-similar --vary: LLM-driven prompt variation

`image-gen generate-similar <source> --count N --vary "instruction"` calls an
OpenAI-compatible chat completion endpoint once per generated image to produce
a fresh prompt (and optionally negative prompt) variation, instead of reusing
the source image's prompt verbatim.

Module: `src/image_gen/llm_variation.py`.

## Configuration

| Env var | CLI override | Required | Default |
|---------|---------------|----------|---------|
| `IMAGEGEN_MODEL_BASE_URL` | `--llm-base-url` | yes | none |
| `IMAGEGEN_MODEL_NAME` | `--llm-model` | yes | none |
| `IMAGEGEN_MODEL_API_KEY` | `--llm-api-key` | no | `"not-needed"` |

CLI flags win over env vars. `resolve_llm_config()` raises a `ValueError`
listing exactly which env var/flag is missing if `base_url` or `model` can't
be resolved. `api_key` falls back to `"not-needed"` because most local
OpenAI-compatible backends (vLLM, llama.cpp server, LM Studio, Ollama's `/v1`)
don't check it, but the `openai` SDK requires a non-empty string.

The `openai` package is imported with a module-level `try/except ImportError`
(`OpenAI = None` if absent), so `generate-similar` without `--vary` keeps
working even if the package isn't installed. Only an actual `--vary` call
without the package raises a clear `RuntimeError` telling the user to
`uv sync`.

## Vocabulary (`--vocab`)

Optional. Points to a **generate-var format spec** (same parser,
`image_gen.variables.load_spec`, see `.agent_docs/generate-var.md`). Only the
top-level `variables:` block is used: for each variable, `format_vocabulary()`
lists its deduplicated top-level option values (not descending into nested
sub-variables, to stay compact) as `"- name: value1; value2; ..."`. The result
is truncated to `max_chars` (default 6000) with a trailing `"...[truncated]"`
marker. This text is injected as inspiration, not resolved as `<placeholder>`
templates.

`--vocab` without `--vary` is a CLI error (`--vocab requires --vary`).

## LLM call contract

One independent chat completion call per image (no multi-turn session kept).
System prompt fixes the role and output contract; user message injects:

```
BASE_PROMPT: ...
BASE_NEGATIVE_PROMPT: ...        (if the source has one)
USER_REQUEST: ...                (the --vary text)
VOCABULARY: ...                  (if --vocab given)
PREVIOUS_VARIATIONS (avoid repeating these):
- ...                            (prompts already generated earlier in this run)
```

Expected response: strict JSON, `{"prompt": "...", "negative_prompt": "..." | null}`.

- `prompt` missing/empty → falls back to `BASE_PROMPT`.
- `negative_prompt` missing/`null` → the source's negative prompt is kept
  unchanged for that image.

### `response_format` fallback

The first call passes `response_format={"type": "json_object"}` (OpenAI-style
strict JSON mode). If the backend rejects it (any exception — some
OpenAI-compatible servers don't support the parameter), a second call is made
without it. Either way, the response content is parsed with `_extract_json()`:
try `json.loads` directly, then fall back to a regex extraction of the first
`{...}` block (handles chatty backends that wrap JSON in prose). If both fail,
raises `ValueError` with a 500-char excerpt of the raw response for debugging.

## `--keep-seed`

Independent of `--vary`. Reuses the source image's exact original seed
(`meta["seed"]`) for every generated image instead of a fresh random seed per
image. Useful to isolate the effect of a prompt variation on identical noise;
also usable standalone without `--vary`.

## Metadata

`GenerationMetadata.llm_request` stores the `--vary` text (omitted from the
JSON, like every other optional field, when unset). Lets you filter
LLM-varied generations in the JSONL log or via `get_info.sh`.

## Failure handling

`generate_variation_with_retry()` (cli.py calls this, not `generate_variation`
directly) retries only on `ValueError` (invalid/unparsable JSON response): up
to 20 attempts, exponential backoff `2**(attempt-1)`s capped at 15s. Each
retry re-issues the same request (fresh sampling can yield valid JSON next
time). A `RuntimeError` (missing `openai` package) or any other exception
still aborts immediately, no retry. If all 20 attempts fail, raises
`ValueError` and the command aborts with `exit 1`. Images already generated
earlier in the same run stay on disk (no rollback), consistent with the rest
of `generate-similar`.
