# Metadata reading (`load_metadata`)

## PNG EXIF: two containers exist, `load_metadata` must read both

`load_metadata` (`src/image_gen/metadata.py`) reads the EXIF `UserComment`
tag via `PIL.Image.getexif()` + `.get_ifd(ExifTags.IFD.Exif)`, not `piexif`.
`piexif` has zero PNG support (only JPEG/TIFF/WebP container detection, see
its `_load.py`), so it can't read either PNG case below. Pillow's own
`getexif()` already transparently supports both:

- A real PNG `eXIf` chunk (what `save_image_with_metadata` writes here).
- The legacy ImageMagick `zTXt`/`tEXt`/`iTXt` "Raw profile type exif"
  text-chunk convention: `\nexif\n<8-space-padded-length>\n<hex bytes>00\n`.
  Some external EXIF writers only ever produce this for PNG, notably the
  `little_exif` Rust crate (confirmed bug as of 0.6.23: its PNG write path
  ignores the `as_zTXt_chunk` flag entirely and always writes `zTXt`, never
  a real `eXIf` chunk) — used by `image-sec-gallery`'s `download` CLI. Files
  downloaded from there are unreadable by `piexif`-based tools but fine via
  `exiftool` (which also understands the legacy convention).

Discovered 2025 while debugging `generate-similar` crashing with
`Error: No generation metadata found ... Given file is neither JPEG nor
TIFF` on PNGs pulled from `image-sec-gallery`. `piexif.load(str(path))`
was hitting its final fallback branch (raw file bytes, magic-byte check),
which only recognizes JPEG/TIFF/WebP.

`piexif` is still used for **writing** (`_build_exif_bytes` /
`save_image_with_metadata`); only the read path changed.

## Duplicate "Raw profile type exif" chunks: last one in file order wins

If a PNG somehow contains more than one text chunk with that keyword
(same or mixed `zTXt`/`tEXt`/`iTXt`), Pillow's PNG chunk reader parses
chunks in file order into a plain dict keyed by keyword
(`self.im_info[k_str] = v_str`), so the **last chunk in file order**
silently overwrites earlier ones before `getexif()` ever sees it. No
merge/error/warning logic was added on our side: this is Pillow's own
existing, deterministic behavior, and no writer we control or interoperate
with (`little_exif`'s clear-then-write, our own EXIF replace) produces
duplicates in the first place. Regression-guarded by
`test_load_metadata_last_raw_profile_chunk_wins_on_duplicates` in
`tests/test_metadata.py`.
