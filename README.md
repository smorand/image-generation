# Image Generation CLI

CLI tool for SDXL image generation with safetensors models, LoRA and textual inversion embedding support.

## Installation

```bash
# Using uv
uv sync
```

## Quick Start

```bash
# Basic generation
image-gen generate \
  --model /path/to/sdxl-model.safetensors \
  --prompt "a beautiful mountain landscape at sunset, masterpiece" \
  --output landscape.png

# With LoRA and custom settings
image-gen generate \
  --model /path/to/model.safetensors \
  --prompt "portrait of a woman, professional photography" \
  --negative-prompt "ugly, deformed, blurry" \
  --width 832 --height 1216 \
  --steps 35 \
  --cfg-scale 7.5 \
  --scheduler dpm++_2m_sde_karras \
  --lora /path/to/style-lora.safetensors:0.8 \
  --seed 12345 \
  --output portrait.png
```

## Commands

### `generate`

Generate images using SDXL safetensors model.

### `generate-var`

Continuous, variable-driven generation from a YAML spec. The spec defines a
prompt template with `<placeholder>` slots and a tree of weighted, recursive
variables. Each loop iteration draws one value per variable, cleans the prompt,
generates one image, and saves it under a zero-padded counter. The same
`<placeholder>` slots also work in `negative_prompt` (resolved with the same
per-variable draw as the positive prompt). The spec is
hot-reloadable: edit it while the loop runs to change prompts, variables, the
loop count, or the run status.

```bash
# Preview resolved prompts without loading the model
image-gen generate-var --config spec.yaml --dry-run --dry-run-count 8

# Run the loop (Ctrl-C or status: stop to end)
image-gen generate-var --config spec.yaml
```

See [`.agent_docs/generate-var.md`](.agent_docs/generate-var.md) for the full
spec format, control model (live/pause/stop), and examples. A ready-to-edit
starter spec lives at [`spec.example.yaml`](spec.example.yaml).

### `info`

Display available schedulers and default settings.

```bash
image-gen info
```

## Generation Options

### Model & Output

| Option | Default | Description |
|--------|---------|-------------|
| `--model`, `-m` | required | Path to safetensors SDXL checkpoint |
| `--output`, `-o` | `./output.png` | Output image path |
| `--vae` | bundled | Path to custom VAE safetensors |
| `--log-dir` | disabled | Directory for a JSONL log of every generated image (one line per image, daily-rotated `generations-YYYY-MM-DD.jsonl`, full params + variables). Omit to disable. |

### Prompt

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt`, `-p` | required | Positive prompt |
| `--negative-prompt`, `-n` | quality defaults | Negative prompt |

### Image Settings

| Option | Default | Range | Description |
|--------|---------|-------|-------------|
| `--width`, `-W` | 1024 | 512-2048 | Image width |
| `--height`, `-H` | 1024 | 512-2048 | Image height |
| `--steps`, `-s` | 30 | 1-150 | Sampling steps |
| `--cfg-scale`, `-c` | 7.0 | 1-30 | Guidance scale |
| `--seed` | random | any int | Random seed |
| `--clip-skip` | 1 | 1-4 | CLIP layers to skip |
| `--batch-size`, `-b` | 1 | 1-8 | Images per batch |

### Scheduler

| Option | Default | Description |
|--------|---------|-------------|
| `--scheduler` | `dpm++_2m_karras` | Sampling algorithm |

**Supported Schedulers:**
- `dpm++_2m_karras` (default, best quality/speed balance)
- `dpm++_2m_sde_karras` (more detail)
- `dpm++_sde_karras`
- `euler_a` (creative)
- `euler`
- `ddim`
- `unipc`

### Hi-Res Fix (Upscaling)

| Option | Default | Description |
|--------|---------|-------------|
| `--hires-fix` | false | Enable 2-pass upscaling |
| `--hires-scale` | 1.5 | Upscale factor |
| `--hires-steps` | 15 | Second pass steps |
| `--hires-denoising` | 0.5 | Denoising strength |

### Extensions

| Option | Format | Description |
|--------|--------|-------------|
| `--lora` | `path:weight` | Load LoRA (repeatable) |
| `--embedding` | path | Load textual inversion (repeatable) |
| `--ip-adapter-image` | path | Reference image for IP-Adapter (repeatable) |
| `--ip-adapter` | preset | IP-Adapter preset: `face` (default), `plus`, `standard` |
| `--ip-adapter-scale` | float 0-1 | Conditioning strength (default 0.6) |

### IP-Adapter (Consistent Identity / Style)

IP-Adapter conditions a generation on one or more **reference images** so the
same face or style carries across seeds and prompts, without training a LoRA.
It is the fastest path to a recurring model/character.

- Provide `--ip-adapter-image` to activate it (this is the switch).
- `--ip-adapter face` (default) uses the plus-face SDXL adapter, best for
  identity. `plus` favors general subject/style, `standard` is lighter.
- Weights download once from `h94/IP-Adapter` on first use (cached by HuggingFace).
- Pass `--ip-adapter-image` multiple times to average several references of the
  same person (different angles) for a stronger, more stable identity.
- Combine with `--lora` (style) and it still works; IP-Adapter drives identity,
  LoRA drives style. For precise pose control, add ControlNet (not yet wired).

**Face identity vs pose:** IP-Adapter fixes *who* the person is, not their pose.
Higher `--ip-adapter-scale` sticks closer to the reference (less prompt freedom);
lower lets the prompt reshape the scene. 0.5-0.7 is the useful range.

## Examples

### Basic Generation

```bash
image-gen generate \
  --model ~/models/sdxl-base.safetensors \
  --prompt "a cat sitting on a windowsill, masterpiece, best quality" \
  --output cat.png
```

### With Multiple LoRAs

```bash
image-gen generate \
  --model ~/models/sdxl.safetensors \
  --prompt "a cyberpunk city at night" \
  --lora ~/loras/cyberpunk-style.safetensors:0.9 \
  --lora ~/loras/neon-lights.safetensors:0.6 \
  --steps 40 \
  --output cyberpunk.png
```

### Hi-Res Fix for Large Images

```bash
image-gen generate \
  --model ~/models/sdxl.safetensors \
  --prompt "detailed landscape painting" \
  --width 1024 --height 1024 \
  --hires-fix \
  --hires-scale 1.5 \
  --hires-denoising 0.4 \
  --output landscape-hires.png
```

### Batch Generation

```bash
image-gen generate \
  --model ~/models/sdxl.safetensors \
  --prompt "abstract art, colorful" \
  --batch-size 4 \
  --seed 42 \
  --output batch.png
# Creates: batch_00.png, batch_01.png, batch_02.png, batch_03.png
```

### Consistent Model with IP-Adapter

```bash
# Same face across different scenes/prompts
image-gen generate \
  --model ~/models/sdxl.safetensors \
  --prompt "professional headshot, studio lighting, business attire" \
  --ip-adapter-image ~/refs/model-face.jpg \
  --ip-adapter face \
  --ip-adapter-scale 0.6 \
  --output headshot.png

# Stronger identity from multiple reference angles
image-gen generate \
  --model ~/models/sdxl.safetensors \
  --prompt "outdoor portrait, golden hour" \
  --ip-adapter-image ~/refs/face-front.jpg \
  --ip-adapter-image ~/refs/face-side.jpg \
  --ip-adapter-scale 0.7 \
  --output outdoor.png
```

### With Custom VAE and Embeddings

```bash
image-gen generate \
  --model ~/models/sdxl.safetensors \
  --vae ~/models/sdxl-vae-fp16-fix.safetensors \
  --embedding ~/embeddings/easynegative.pt \
  --prompt "beautiful portrait, easynegative" \
  --output portrait.png
```

## Video generation (`video-gen`)

Sibling CLI that animates a still image into an MP4 clip with a local
image-to-video model. Same design as `image-gen`: a `generate` command, a
hot-reloadable `generate-var` batch loop, and the identical `<placeholder>`
variable engine. Two switchable backends, both bfloat16 on Apple Silicon (MPS):

| Backend | Model | Speed | Quality | Apple Silicon |
|---------|-------|-------|---------|---------------|
| `ltx` (default) | `Lightricks/LTX-Video` | Fast (~6 min / 3s) | Coherent | Works |
| `wan` | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | Slow (~35 min / 3s) | Higher on CUDA | **Broken on MPS** (noise after frame 1) |

On Apple Silicon use `ltx`. Wan 2.2 TI2V-5B is wired and downloads fine, but its
denoising diverges to noise on Metal/MPS in bfloat16 (first frame clean, rest
noise) and 512px OOMs on 32 GB. It stays available for CUDA machines. See
[`.agent_docs/video-gen.md`](.agent_docs/video-gen.md) for measured timings.

```bash
# One clip from an image (LTX, 3 seconds)
video-gen generate -i seeds/cat.png -p "kitten blinking, slow zoom in" -b ltx -d 3 -o out/cat.mp4

# Higher quality with Wan (slower; add --offload on tight memory)
video-gen generate -i seeds/cat.png -p "kitten turning its head" -b wan -d 3 -o out/cat_wan.mp4

# Long clip past the native ceiling: chain short segments, one -p per segment
video-gen chain -i seeds/cat.png \
  -p "kitten blinking" -p "kitten looks left" -p "kitten yawns" -d 3 -o out/cat_long.mp4

# Backends and defaults
video-gen info

# Variable-driven batch (edit the spec while it runs)
video-gen generate-var --config video-spec.example.yaml --dry-run
video-gen generate-var --config video-spec.example.yaml
```

`--duration` is snapped to each backend's frame arithmetic (LTX `8k+1`, Wan
`4k+1`). A single long clip denoises every frame in one tensor, so past the
native max (~5-10s) it OOMs Metal (`Failed to allocate private MTLBuffer ...`).
Use `video-gen chain` instead: it generates a **series of short segments**, each
seeded by the last frame of the previous one and driven by its own `-p` prompt,
then concatenates them into one MP4. Peak memory stays flat regardless of total
length. Provenance is embedded in the MP4 `comment` tag and a `<name>.json`
sidecar (with the full `prompt_series` for chained clips).

**Mac reality:** clips are minutes, not seconds. LTX fast preset is the "minutes"
path; Wan is the quality path and slower. FP8 is unusable on Metal, so both run
bfloat16. See [`.agent_docs/video-gen.md`](.agent_docs/video-gen.md) and the
starter [`video-spec.example.yaml`](video-spec.example.yaml).

**Stable Video Diffusion (SVD) is intentionally not a backend**: it conditions on
the image only (no text prompt), so it cannot drive the `<placeholder>` prompt
engine. Wan 2.2 14B and LTX-2 19B are excluded too: 15-80 min per clip on Mac and
NaN issues on MPS.

## Model storage

HuggingFace models (SDXL base, Wan, LTX) default to **`~/.cache/models/hf`**.
This is set in code (`image_gen`/`video_gen` `__init__`) only when neither
`HF_HUB_CACHE` nor `HF_HOME` is already set, so an explicit override still wins.
Local safetensors checkpoints passed with `--model` are read from their given
path (e.g. the flat files in `~/.cache/models/`).

## Requirements

- Python 3.11+
- CUDA-capable GPU recommended (MPS/CPU fallback available)
- ~10GB VRAM for SDXL models; video backends run in bfloat16 (no FP8 on Metal)
- Disk: LTX-Video and Wan 2.2 TI2V-5B are ~15-20GB each (cached on first use)

## Tech Stack

- **diffusers** - HuggingFace pipeline management
- **transformers** - CLIP tokenizers
- **torch** - GPU acceleration
- **safetensors** - Model loading
- **typer** - CLI interface
- **Pillow** - Image handling
