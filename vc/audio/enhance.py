"""DeepFilterNet3 denoiser (MIT/Apache-2.0), full-band 48 kHz. Only called for noisy uploads;
see `preprocess.maybe_enhance`."""
from __future__ import annotations

import logging
import threading

import numpy as np

from vc.audio.io import resample
from vc.registry import Capabilities, registry

log = logging.getLogger("vc.enhance")
_lock = threading.Lock()
_state: tuple | None = None


def _load():
    global _state
    with _lock:
        if _state is None:
            from df.enhance import init_df

            import os
            from pathlib import Path

            bundled = Path(os.environ.get("VC_MODELS_DIR", "")) / "deepfilternet" / "DeepFilterNet3"
            base = str(bundled) if (bundled / "config.ini").exists() else None
            # DeepFilterNet picks its own device (2M parameters, ~0.2 GB VRAM on CUDA)
            model, df_state, _ = init_df(model_base_dir=base, log_level="WARNING", log_file=None)
            _state = (model.eval(), df_state)
        return _state


@registry.register(Capabilities(name="deepfilternet3", kind="enhancer", license="MIT / Apache-2.0",
                                commercial_ok=True, languages=["*"], native_sr=48000, approx_vram_gb=0.0,
                                notes="discriminative full-band denoiser; CPU"))
def create_enhancer(**_):
    return denoise


def denoise(audio: np.ndarray, sr: int, block_s: float = 30.0, atten_lim_db: float | None = 24.0) -> np.ndarray:
    """Denoise in 30 s blocks with 0.5 s crossfaded overlap. `atten_lim_db` caps the suppression so
    breaths and room tone are reduced rather than gated to digital silence."""
    import torch
    from df.enhance import enhance

    model, df_state = _load()
    df_sr = df_state.sr()
    x = resample(audio, sr, df_sr)
    block, ov = int(block_s * df_sr), int(0.5 * df_sr)
    out = np.zeros_like(x)
    pos = 0
    while pos < len(x):
        seg = x[pos: pos + block + ov]
        with torch.no_grad():
            y = enhance(model, df_state, torch.from_numpy(seg).unsqueeze(0), atten_lim_db=atten_lim_db)
        y = y.squeeze(0).cpu().numpy()[: len(seg)]
        if pos > 0:
            k = min(ov, len(y))
            fade = np.linspace(0.0, 1.0, k, dtype=np.float32)
            out[pos: pos + k] = out[pos: pos + k] * (1 - fade) + y[:k] * fade
            out[pos + k: pos + len(y)] = y[k:]
        else:
            out[: len(y)] = y
        pos += block
    return resample(out, df_sr, sr).astype(np.float32)[: len(audio)]
