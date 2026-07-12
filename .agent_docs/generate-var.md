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
template_output: "out/img_<number>.png"
template_prompt: "solo, <eth> girl, <hair>, <clothes>, <place>, photography"
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
  eth: [inuit]
  hair: ["long black hair", "short black hair"]
  clothes:
    - "wearing a dress"                       # plain string, weight 1
    - value: "wearing a <color> bikini"       # mapping: value + weight + nested vars
      weight: 2
      variables:
        color: [red, blue, white]
    - value: "wearing <top> with <bottom>"
      variables:
        top:
          - value: "a <color> tank top"
            variables:
              color:
                - {value: "", weight: 4}       # 4x more likely: no color word
                - white
                - black
        bottom: ["jeans", "a skirt"]
```

### Required keys
`template_prompt`, `template_output`, `variables`. Everything else is optional
(`status` defaults `live`, `loop` defaults `0`, `defaults` defaults `{}`).

## Placeholders `<name>`

- Prompt slots use `<name>` (chosen over `{{name}}` so values need no YAML
  quoting; `{` starts a YAML flow mapping).
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
  again. So `<hair> ... <hair>` yields the same value in both spots.
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

## Hot reload

The config file's mtime is checked every `--poll` seconds (default 5, and
between every image). On change the whole spec reloads live. If a
pipeline-affecting key changed (`model`, `vae`, `scheduler`, `clip_skip`,
`lora`, `embedding`, `ip_adapter*`), the pipeline is rebuilt; otherwise only the
prompt/variables/loop/status/output are swapped. A reload that fails to parse is
logged and ignored (the previous spec keeps running).

## Output and metadata

- Filenames come from `template_output`; `<number>` is zero-padded to 10 digits
  for clean alphanumeric sorting. On startup the counter continues after the
  highest existing `<number>` in the output directory (never overwrites).
- Each iteration uses a fresh random seed (recorded in metadata).
- Metadata is embedded as an EXIF **UserComment** (single-line JSON) for both
  `.png` (eXIf chunk) and `.jpg`. It includes all generation params plus a
  `variables` sub-dict mapping each variable name to its chosen value.
- A `manifest.jsonl` is appended in the output directory: one line per image
  with `number`, `output`, `seed`, `prompt`, `variables`, `timestamp`.

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
