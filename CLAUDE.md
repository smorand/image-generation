# Image Generation CLI

## Overview

CLI tool for SDXL image generation using safetensors models with LoRA and embedding support.

**Tech Stack:** Python 3.11+, diffusers, torch, typer, safetensors

## Key Commands

```bash
# Install
uv sync

# Generate image
image-gen generate --model path.safetensors --prompt "text" --output out.png

# Show info
image-gen info
```

## Project Structure

```
src/image_gen/
├── __init__.py       # Package init
├── cli.py            # Typer CLI entry point
├── pipeline.py       # SDXLPipeline class
├── schedulers.py     # Scheduler factory (7 schedulers)
├── lora.py           # LoRA loading utilities
└── embeddings.py     # Textual inversion support
```

## Conventions

- **Entry point:** `image-gen` command (defined in pyproject.toml)
- **Pipeline:** `SDXLPipeline` class wraps diffusers `StableDiffusionXLPipeline`
- **Config:** `GenerationConfig` dataclass holds all generation parameters
- **Device auto-detection:** CUDA > MPS > CPU

## Documentation Index

| File | Topic |
|------|-------|
| `.agent_docs/sdxl-pipeline.md` | SDXL pipeline details, optimizations, troubleshooting |
