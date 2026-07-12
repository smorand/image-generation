"""Local image-to-video generation CLI (Wan 2.2 / LTX-Video)."""

import os as _os
from pathlib import Path as _Path

# Default HuggingFace model cache to ~/.cache/models/hf (the user's model
# library convention). Only applied if the user has not already chosen a cache
# location via HF_HUB_CACHE or HF_HOME, so an explicit override still wins.
if not _os.environ.get("HF_HUB_CACHE") and not _os.environ.get("HF_HOME"):
    _os.environ["HF_HUB_CACHE"] = str(_Path.home() / ".cache" / "models" / "hf")

__version__ = "0.1.0"
