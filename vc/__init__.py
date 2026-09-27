"""Voice Clone Studio: consent-based voice transcription and regeneration."""
import os
from pathlib import Path

__version__ = "0.2.0"

# torch.compile needs triton / a C++ toolchain on PATH, neither of which exists on stock Windows.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Installed copy: the installer ships every model in <install>\models, so nothing is downloaded.
BUNDLED_MODELS = Path(__file__).resolve().parent.parent / "models"
if (BUNDLED_MODELS / "bundle.json").exists():
    os.environ.setdefault("VC_MODELS_DIR", str(BUNDLED_MODELS))
    os.environ.setdefault("HF_HOME", str(BUNDLED_MODELS / "hf"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("PKUSEG_HOME", str(BUNDLED_MODELS / "pkuseg"))
    os.environ.setdefault("AUDIOSEAL_CACHE_DIR", str(BUNDLED_MODELS))      # the loader appends "audioseal"
