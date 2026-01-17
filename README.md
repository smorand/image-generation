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

### With Custom VAE and Embeddings

```bash
image-gen generate \
  --model ~/models/sdxl.safetensors \
  --vae ~/models/sdxl-vae-fp16-fix.safetensors \
  --embedding ~/embeddings/easynegative.pt \
  --prompt "beautiful portrait, easynegative" \
  --output portrait.png
```

## Requirements

- Python 3.11+
- CUDA-capable GPU recommended (MPS/CPU fallback available)
- ~10GB VRAM for SDXL models

## Tech Stack

- **diffusers** - HuggingFace pipeline management
- **transformers** - CLIP tokenizers
- **torch** - GPU acceleration
- **safetensors** - Model loading
- **typer** - CLI interface
- **Pillow** - Image handling
