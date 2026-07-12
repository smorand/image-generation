# video-gen: local image-to-video

Sibling CLI to `image-gen`. Animates a still image into an MP4 clip using a
local diffusers video pipeline. Two switchable backends, both bfloat16 on MPS.

## Backends

| Backend | Model | Class | Frames | Native fps | Notes |
|---------|-------|-------|--------|-----------|-------|
| `ltx` (default) | `Lightricks/LTX-Video` | `LTXImageToVideoPipeline` | 8k+1 | 24 | Fast, lighter, lower coherence |
| `wan` | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | `WanImageToVideoPipeline` | 4k+1 | 24 | Higher quality, slower, big text encoder |

Registry lives in `src/video_gen/backends.py` (`BACKENDS`). Each `BackendSpec`
declares the frame arithmetic (`num_frames = frame_multiple * k + 1`), native
fps, dim multiple (32), and Mac-friendly defaults.

## Why these and not others

- **FP8 fails on Metal.** Both backends run bfloat16 only. No FP8 checkpoints.
- **SVD excluded**: image-only conditioning, no text prompt, so it cannot use
  the `<placeholder>` prompt engine. Wan 2.2 14B / LTX-2 19B excluded: 15-80 min
  per clip on Mac, NaN issues on MPS.

## Architecture (mirrors image-gen)

```
src/video_gen/
├── backends.py    # BackendSpec registry, resolve_frames(duration -> frame count)
├── pipeline.py    # VideoPipeline: load per backend, prepare_image, generate -> [PIL frames]
├── chain.py       # long-clip chaining: prompt series, stitch_segments, generate_chain
├── encode.py      # encode_frames -> H.264 MP4 via imageio-ffmpeg, comment tag
├── metadata.py    # VideoMetadata: MP4 comment tag + <name>.json sidecar
├── variables.py   # VideoVarSpec, reuses image_gen.variables engine + template_input
├── runner.py      # generate-var loop: hot-reload, counter, manifest_video.jsonl
└── cli.py         # video-gen: generate, chain, generate-var, info
```

The variable resolution engine is imported wholesale from
`image_gen.variables` (weighted, recursive `<placeholder>` draws). Video adds
`template_input` for the source image.

## Commands

```bash
# Single clip
video-gen generate -i seeds/cat.png -p "kitten blinking, slow zoom" -b ltx -d 3 -o out/cat.mp4

# Long clip: chain short segments, one -p per segment (see "Long clips" below)
video-gen chain -i seeds/cat.png \
  -p "kitten blinking" -p "kitten looks left" -p "kitten yawns" -d 3 -o out/cat_long.mp4

# Backends + defaults
video-gen info

# Variable-driven batch (hot-reloadable)
video-gen generate-var --config video-spec.example.yaml --dry-run
video-gen generate-var --config video-spec.example.yaml
```

## Spec format

Adds to the image-gen spec: `template_input` and video `defaults` (backend,
duration/num_frames, fps, guidance, offload). `template_input` resolves as:

- plain path `seeds/cat.png` -> same image every clip
- `seeds/img_<number>.png` -> pairs with the loop counter (animate a batch)
- `seeds/*.png` or a directory -> random existing file per clip

`template_output` may carry any suffix; the runner writes `.mp4` and the
counter scan matches `.mp4`.

## Frame count arithmetic

`--duration` is snapped to a valid count for the backend:
`num_frames = frame_multiple * round((duration*fps - 1)/frame_multiple) + 1`,
clamped to `max_frames`. Pass `--num-frames` to set it explicitly.

## Long clips: chaining (`video-gen chain`)

A single generation denoises the **whole clip in one latent tensor**, so peak
memory grows with the frame count. On a 32 GB Mac an LTX `-d 10` (241 frames)
tries to allocate a ~27 GB Metal buffer and dies with:
`failed assertion Failed to allocate private MTLBuffer for size 26791142400`.
There is no automatic fallback to a shorter clip; it allocates or it crashes.

`video-gen chain` gets past the native ceiling by generating a **series of short
segments**: each segment is a normal i2v clip **seeded by the last frame of the
previous one**, and each gets its own prompt so the action can evolve. Segments
are concatenated into one MP4 (the duplicated seam frame is dropped). Because
each segment is small, **peak memory stays flat** regardless of total length.

```bash
video-gen chain -i seeds/girl.png \
  -p "she turns her head slowly" \
  -p "she smiles and looks up" \
  -p "a breeze moves her hair" \
  --seg-duration 3 -o out/long.mp4
```

- `-p` is repeatable: one prompt per segment. `--segments N` overrides the count
  (pads by repeating the last prompt, or truncates). `--seg-duration/-d` is the
  length of **each** segment (default 3 s, capped at 6 s to stay within memory).
- Seeds: segment `i` uses `base_seed + i` (reproducible, varied).
- Total length ~= `segments * seg_duration`. Metadata records `segments` and the
  full `prompt_series`.
- **Seam limitation (inherent to image-conditioned i2v):** the model sees only
  one frame between segments, so motion *velocity* resets at each seam and a
  slight hitch is possible. Keep segments a few seconds long to minimise it.
  For `generate-var` continuous batches, each clip is still a single generation.

Core logic is model-free and unit-tested in `chain.py` (`build_prompt_series`,
`stitch_segments`, `generate_chain`); the CLI wires the pipeline in and frees the
MPS cache (`VideoPipeline.empty_cache`) between segments.

## Mac performance (measured, M5 32 GB)

Real timings from the cat demo (3.04s clip, 73 frames @ 24 fps, from seeds/cat.png):

| Backend | Settings | Time | Memory | Result |
|---------|----------|------|--------|--------|
| LTX | 704x480, 40 steps | **5m51s** | resident, fine | coherent, kitten turns its head |
| Wan | 512x512, 30 steps | 35m14s | **OOM resident, offload forced** | **broken: frame 1 clean, then noise** |

**Wan 2.2 TI2V-5B is not viable on MPS as of this build.** The denoising diverges:
the first frame (the VAE-encoded conditioning image) is clean, every subsequent
frame collapses to structured grid-noise. Diagnosis (all on the cat seed):

- Not offload: broken both resident and with sequential CPU offload.
- **Not precision: bf16, fp16 and fp32 all produce identical noise** (tested
  384x384, 13 frames). So it is an op-level MPS incompatibility, not numerical
  divergence. The regular grid pattern points at patchify / rotary-embedding /
  attention in the Wan transformer computing wrong values on Metal.
- The RoPE MPS float32 fix (diffusers #10986) is already present in 0.36, so that
  is not the cause here.
- The `expand_timesteps` I2V denoising loop (5B path) is mathematically correct.

Fixing it means bisecting MPS ops inside a 5B transformer, which the community
has not solved ("Wan needs Mac-native Metal optimization"). Not worth it while
LTX works. Additionally, 512x512 resident OOMs on 32 GB (~42 GB requested), so
the demo falls back to CPU offload (~6x slower). `_load_wan` emits a warning on
MPS. Wan stays wired and correct on CUDA.

`--precision fp16|bf16|fp32` overrides the dtype (default bf16 on GPU). It does
not fix Wan on MPS but is useful on CUDA and for experimentation.

**Practical guidance on Apple Silicon: use `ltx`.** Wan stays wired for CUDA
machines (where it works and is the quality option) and for a future Metal-fixed
build. Do not default Mac users to Wan.

- Clips are minutes, not seconds. LTX fast preset is the only "minutes" path;
  Wan is the quality path and slower (and currently broken on MPS).
- bfloat16 only; `PYTORCH_ENABLE_MPS_FALLBACK=1` is set in `cli.py` so
  unsupported ops fall back to CPU instead of raising.
- Memory: VAE tiling/slicing always on. Wan's UMT5 text encoder is large; pass
  `--offload` (sequential CPU offload) if you hit pressure. Default keeps the
  pipeline resident (faster when it fits in unified memory).
- Backend switch in a running generate-var loop calls `VideoPipeline.unload()`
  (`torch.mps.empty_cache()`) before rebuilding.

## Metadata

MP4 has no EXIF. Provenance is written twice: the container `comment` tag (JSON)
and a one-line `<name>.json` sidecar next to the clip. Includes prompt, backend,
seed, frames/fps/duration, and resolved generate-var `variables`.

## Gotchas

- `num_frames` must satisfy the backend arithmetic or the VAE errors. Always go
  through `resolve_frames` / `--num-frames` with a valid value.
- Width/height are snapped down to a multiple of 32 (`snap_dim`).
- MPS has no native `torch.Generator`; seeds use a CPU generator (works across
  devices, reproducible enough).
- `macro_block_size=16` in the encoder needs even-ish dims; the 32-multiple
  snapping already satisfies it.
