# Image Generation CLI

## Overview

CLI tools for local generation: **`image-gen`** (SDXL images, safetensors + LoRA
+ embeddings) and **`video-gen`** (image-to-video, Wan 2.2 / LTX-Video).

**Tech Stack:** Python 3.11+, diffusers, torch, typer, safetensors, imageio-ffmpeg

## Key Commands

```bash
# Install
uv sync

# Generate image
image-gen generate --model path.safetensors --prompt "text" --output out.png

# Re-generate variations of an existing image (same params, fresh seed each time)
image-gen generate-similar out.png --count 5 --clip-skip 4

# Same, but with LLM-driven prompt variation per image (see IMAGEGEN_MODEL_* env vars)
image-gen generate-similar out.png --count 5 --vary "more casual outfit" --keep-seed

# Variable-driven continuous generation (YAML spec, hot-reloadable)
image-gen generate-var --config spec.yaml --dry-run   # preview prompts
image-gen generate-var --config spec.yaml             # run the loop
image-gen generate-var --config spec.yaml --pause     # set status only (also --live/--stop), no generation

# Image-to-video (backend: ltx fast | wan quality)
video-gen generate -i seeds/cat.png -p "kitten blinking, slow zoom" -b ltx -d 3 -o out/cat.mp4
# Long clip past the native ceiling: chain short segments (one -p per segment)
video-gen chain -i seeds/cat.png -p "blinking" -p "looks left" -p "yawns" -d 3 -o out/long.mp4
video-gen generate-var --config video-spec.example.yaml   # hot-reloadable batch
video-gen info

# Show info
image-gen info
```

## Project Structure

```
src/image_gen/
├── __init__.py       # Package init
├── cli.py            # Typer CLI entry point (generate, generate-var, info)
├── pipeline.py       # SDXLPipeline class
├── schedulers.py     # Scheduler factory (7 schedulers)
├── lora.py           # LoRA loading utilities
├── ip_adapter.py     # IP-Adapter presets + reference-image loading
├── embeddings.py     # Textual inversion support
├── variables.py      # generate-var: YAML spec + variable resolution engine
├── runner.py         # generate-var: hot-reloadable control loop
└── metadata.py       # EXIF UserComment embedding (JPEG + PNG)

src/video_gen/        # image-to-video CLI (reuses image_gen.variables engine)
├── backends.py       # Wan/LTX registry, frame arithmetic, defaults
├── pipeline.py       # VideoPipeline: load backend, generate -> [PIL frames]
├── chain.py          # long-clip chaining: prompt series + last-frame carry-over
├── encode.py         # frames -> H.264 MP4 (imageio-ffmpeg)
├── metadata.py       # MP4 comment tag + <name>.json sidecar
├── variables.py      # VideoVarSpec (adds template_input)
├── runner.py         # video generate-var loop (hot-reload)
└── cli.py            # video-gen: generate, generate-var, info

scripts/               # One-off maintenance scripts (not part of the CLI)
└── backfill_generated_at.py  # add generated_at to images saved before that
                              # field existed (uses file mtime). Dry-run by
                              # default; see its docstring for --apply/--limit.
```

## Conventions

- **Model cache:** HuggingFace models default to `~/.cache/models/hf` (set in
  `image_gen`/`video_gen` `__init__`, only if `HF_HUB_CACHE`/`HF_HOME` unset).
  Flat local checkpoints live in `~/.cache/models/` and are passed via `--model`.
- **generate-similar model/VAE resolution:** EXIF metadata stores `model`/`vae`
  as filenames only (no path). `generate-similar` looks them up by exact
  filename in `--model-dir` (default `~/.cache/models`, non-recursive); pass
  `--model`/`--vae` explicitly to bypass the lookup.
- **generate-similar --vary (LLM):** env vars `IMAGEGEN_MODEL_BASE_URL`,
  `IMAGEGEN_MODEL_NAME`, `IMAGEGEN_MODEL_API_KEY` (optional, defaults to
  `not-needed`) configure any OpenAI-compatible endpoint. See
  `.agent_docs/generate-similar-llm.md`.
- **Entry point:** `image-gen` command (defined in pyproject.toml)
- **Pipeline:** `SDXLPipeline` class wraps diffusers `StableDiffusionXLPipeline`
- **Config:** `GenerationConfig` dataclass holds all generation parameters
- **Device auto-detection:** CUDA > MPS > CPU
- **Tests:** run with `uv run python -m pytest` (not `uv run pytest`, which may
  resolve a pytest outside the venv)
- **generate-var spec:** starter file at `spec.example.yaml`; placeholders use
  `<name>`; metadata (incl. resolved `variables` sub-dict) read via
  `~/.local/bin/get_info.sh`

## Documentation Index

| File | Topic |
|------|-------|
| `.agent_docs/sdxl-pipeline.md` | SDXL pipeline details, optimizations, troubleshooting |
| `.agent_docs/generate-var.md` | generate-var spec format, control model, metadata |
| `.agent_docs/video-gen.md` | video-gen backends (Wan/LTX), spec, Mac perf, metadata |
| `.agent_docs/generate-similar-llm.md` | `generate-similar --vary` LLM contract, env vars, fallback behavior |
| `.agent_docs/metadata-reading.md` | `load_metadata` PNG EXIF containers (real `eXIf` vs ImageMagick `zTXt`/`tEXt`/`iTXt` raw-profile), duplicate-chunk resolution |
