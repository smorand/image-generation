# generate-var: variable-driven continuous generation

`image-gen generate-var --config spec.yaml` runs a hot-reloadable loop that
resolves a templated prompt from a tree of weighted variables and generates one
image per iteration.

Modules:
- `variables.py` — YAML parsing, the resolution engine, prompt cleanup.
- `runner.py` — the control loop (status, loop count, hot-reload, counter, manifest).
- `metadata.py` — EXIF UserComment embedding (JPEG and PNG), incl. the resolved
  variables sub-dict.
- `cli.py` — the `generate-var` command (overrides for the YAML `defaults:` block).

## Spec format

```yaml
status: live                 # live | pause | stop  (forced to "live" on startup)
loop: 0                      # 0 = infinite, N > 0 = stop after N images
# log_dir: /path/to/logs     # optional; daily JSONL log of every image.
#                              defaults to the output directory.
template_output: "out/img_<number>.png"
template_prompt: "<season> landscape, <biome>, <light>, <weather>, photography"
negative_prompt: "low quality, deformed"
defaults:                    # generation params; every CLI flag overrides these
  model: "models/x.safetensors"
  steps: 30
  cfg_scale: 4.0
  width: 1024
  height: 1024
  scheduler: euler_a
  clip_skip: 2
  # optional: vae, lora (list of "path:weight"), embedding (list),
  #           ip_adapter, ip_adapter_image (list), ip_adapter_scale,
  #           hires_fix, hires_scale, hires_steps, hires_denoising
variables:
  season: [spring, summer, autumn, winter]
  biome: ["dense forest", "mountain valley", "coastal cliffs", "desert dunes"]
  light:
    - "golden hour"                            # plain string, weight 1
    - value: "<time> light"                    # mapping: value + weight + nested vars
      weight: 2
      variables:
        time: [morning, midday, dusk]
    - value: "<cover> sky"
      variables:
        cover:
          - value: ""
            weight: 2                           # 2x more likely: no sky qualifier
          - overcast
          - stormy
  weather:
    - clear                                     # plain string, weight 1
    - foggy
    - "light rain"
```

### Required keys
`template_prompt`, `template_output`, `variables`. Everything else is optional
(`status` defaults `live`, `loop` defaults `0`, `defaults` defaults `{}`).

## Placeholders `<name>`

- Prompt slots use `<name>` (chosen over `{{name}}` so values need no YAML
  quoting; `{` starts a YAML flow mapping).
- `negative_prompt` supports the same `<name>` slots as `template_prompt`. They
  resolve against the same variable tree and share the same per-variable draw:
  a variable used in both the prompt and the negative prompt is drawn once and
  reused in both. Undefined placeholders in `negative_prompt` are rejected at
  load time too.
- `template_output` supports two builtins only: `<number>` (10-digit
  zero-padded counter) and `<seed>` (the diffusion seed).
- Undefined placeholders are rejected at load time (typo protection).

## Variable resolution

- A variable is a list of **options**. An option is a plain string (weight 1) or
  a mapping `{value, weight?, variables?}`.
- `value` may contain nested `<sub>` placeholders resolved against the option's
  local `variables`, overlaid on the parent scope (recursive, any depth; a
  50-level guard trips on cycles).
- **Weights** are relative among sibling options (default 1). `{value: "",
  weight: 4}` makes the empty choice 4x more likely than a weight-1 sibling.
- A named variable is **drawn once per prompt and reused** wherever it appears
  again. So `<hair> ... <hair>` yields the same value in both spots (including
  when one occurrence is in `template_prompt` and the other in
  `negative_prompt`).
- After substitution the prompt is cleaned: runs of whitespace collapse, spaces
  before commas drop, repeated commas merge to one, a single space follows each
  comma, and leading/trailing commas are stripped. This absorbs empty slots
  (`solo, <x>, natural` with empty `x` -> `solo, natural`).

## Control model

- **status** (read fresh from the file every poll):
  - `live` — keep generating (subject to `loop`).
  - `pause` — finish the current image, then wait; keeps polling the file, so
    setting it back to `live` resumes (with any edits made during the pause).
  - `stop` — finish the current image, then exit.
- On startup the runner **rewrites the `status:` line to `live`** (surgical
  regex edit, comments and formatting preserved), so a run always begins live.
- **loop**: `0` = infinite; `N` = exit after N images this run.

### Setting status without generating

`generate-var` accepts three exclusive flags that **only edit the `status:` line
and exit** (no model load, no generation):

```bash
image-gen generate-var --config prompts.yaml --pause   # -> status: pause
image-gen generate-var --config prompts.yaml --live    # -> status: live
image-gen generate-var --config prompts.yaml --stop    # -> status: stop
```

Same surgical regex edit as startup (rest of the file untouched). Passing more
than one is an error. A running loop picks up the change on its next poll, so
these are the easy way to pause/resume/stop a background run from another shell.

## Hot reload

The config file's mtime is checked every `--poll` seconds (default 5, and
between every image). On change the whole spec reloads live. If a
pipeline-affecting key changed (`model`, `vae`, `scheduler`, `clip_skip`,
`lora`, `embedding`, `ip_adapter*`), the pipeline is rebuilt; otherwise only the
prompt/variables/loop/status/output are swapped.

**Resilient reload:** any bad edit is caught, logged
(`Reload failed, keeping previous configuration unchanged: ...`), and the loop
keeps running with the **previous** config unchanged. This covers all three
failure modes:
- unreadable YAML (`yaml.YAMLError`: bad indentation, stray tab, dangling colon),
- an invalid value that fails validation (bad `status`, undefined placeholder…),
- a pipeline-affecting change that blows up the rebuild (a `model` path that does
  not exist, an unknown `scheduler`, a missing LoRA/embedding).

The rebuild is atomic: on failure both the pipeline **and** the spec are kept as
they were, so config never ends up half-applied. A broken file is retried only
after the next save (its mtime is consumed), not on every poll. Fix the file and
save again to resume.

## Output and metadata

- Filenames come from `template_output`; `<number>` is zero-padded to 10 digits
  for clean alphanumeric sorting. On startup the counter continues after the
  highest existing `<number>` in the output directory (never overwrites).
- Each iteration uses a fresh random seed (recorded in metadata).
- Metadata is embedded as an EXIF **UserComment** (single-line JSON) for both
  `.png` (eXIf chunk) and `.jpg`. It includes all generation params plus a
  `variables` sub-dict mapping each variable name to its chosen value.
- A JSONL generation log is appended, one line per generated image, to
  `<log_dir>/generations-YYYY-MM-DD.jsonl` (daily rotation). Each line carries a
  `timestamp`, the `command` ("generate-var"), the `number`, the `output` path,
  and the **full** metadata (prompt, negative_prompt, all sampling params, and
  the `variables` sub-dict). Nothing is lost if the image is later moved or
  deleted, so downstream stats stay exhaustive.
- `log_dir` resolves in this order: CLI `--log-dir`, then the spec's top-level
  `log_dir:`, then `defaults.log_dir`, else the output directory. There is no
  separate `manifest.jsonl` anymore; this log supersedes it.

### Reading metadata back

`~/.local/bin/get_info.sh <file>` prints the JSON (works for PNG and JPG):

```bash
get_info.sh out/img_0000000042.png | jq '.variables'
```

> The nested `variables` object requires `get_info.sh` to strip from the first
> `{` (`sed 's/^[^{]*//'`), not the last `: {`. The greedy original
> (`s/.*: {/{/`) breaks on any nested JSON object; it has been fixed.

## CLI overrides

Every generation param in `defaults:` can be overridden on the command line
(applied only when passed): `--model`, `--negative-prompt`, `--steps`,
`--cfg-scale`, `--width`, `--height`, `--scheduler`, `--clip-skip`, `--lora`,
`--vae`. Other flags: `--dry-run` / `--dry-run-count` (print prompts, no model),
`--poll` (reload interval), `--var-seed` (reproducible variable draws).

## Gotchas

- Values containing `<` are treated as placeholders. Literal `<`/`>` in a prompt
  is not supported (rare for prompts).
- `weight` must be `>= 0`; all-zero weights in a group fall back to uniform.
- The pipeline seed RNG and the variable RNG are separate; `--var-seed` fixes
  only the prompt draws, image seeds stay random.
