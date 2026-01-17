# SDXL Pipeline Details

## Architecture

### SDXLPipeline Class (`pipeline.py`)

The main wrapper around HuggingFace's `StableDiffusionXLPipeline`:

- **Loading:** Uses `from_single_file()` for safetensors checkpoints
- **Dual Text Encoders:** SDXL uses CLIP-L + OpenCLIP-G (handled automatically)
- **VAE:** Can load custom VAE via `--vae` flag

### Device & Dtype Selection

```python
# Auto-detection priority
1. CUDA -> float16
2. MPS (Apple Silicon) -> float16
3. CPU -> float32
```

### Memory Optimizations

Enabled by default:
- `enable_model_cpu_offload()` - CUDA only
- `enable_vae_slicing()` - Process VAE in slices
- `enable_vae_tiling()` - Tile large images

## Schedulers

| Name | Class | Best For |
|------|-------|----------|
| `dpm++_2m_karras` | DPMSolverMultistepScheduler | General use (default) |
| `dpm++_2m_sde_karras` | DPMSolverMultistepScheduler | More detail |
| `dpm++_sde_karras` | DPMSolverSinglestepScheduler | Quality |
| `euler_a` | EulerAncestralDiscreteScheduler | Creative/varied |
| `euler` | EulerDiscreteScheduler | Deterministic |
| `ddim` | DDIMScheduler | Fast/simple |
| `unipc` | UniPCMultistepScheduler | Fast quality |

## LoRA Loading

- Format: `path:weight` or `path` (default weight: 0.8)
- Multiple LoRAs supported via repeated `--lora` flags
- Uses diffusers' `load_lora_weights()` with PEFT backend

## Hi-Res Fix

Two-pass generation:
1. Generate at base resolution
2. Upscale with LANCZOS
3. img2img pass with denoising

Parameters:
- `hires_scale`: Upscale factor (default 1.5)
- `hires_steps`: Second pass steps (default 15)
- `hires_denoising`: Strength 0-1 (default 0.5)

## CLIP Skip

- `clip_skip=1`: Use all layers (default)
- `clip_skip=2+`: Skip N-1 layers from end
- Passed to diffusers pipeline as `clip_skip` parameter

## Troubleshooting

### Out of Memory

- Reduce `--width` / `--height`
- Disable `--hires-fix`
- Reduce `--batch-size`
- Try different model (pruned/fp16)

### Slow Generation

- Use `dpm++_2m_karras` scheduler
- Reduce `--steps` (20-30 usually sufficient)
- Ensure CUDA/MPS is being used (check logs)

### Black/Corrupted Images

- Try different `--vae` (some models need specific VAE)
- Reduce `--cfg-scale` (7-9 recommended)
- Check model compatibility (must be SDXL)
