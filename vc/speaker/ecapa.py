"""SpeechBrain ECAPA-TDNN speaker encoder (Apache-2.0, 192-d, VoxCeleb EER ~0.8%).
Used for: speaker-consistency check on uploads, reference-clip ranking, consent speaker verification,
and the post-generation similarity gate. ReDimNet2 / WeSpeaker can be dropped in behind the same
interface for higher accuracy."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from vc.registry import Capabilities, registry

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


@registry.register(Capabilities(
    name="ecapa", kind="speaker", license="Apache-2.0", commercial_ok=True, languages=["*"],
    approx_vram_gb=0.3, notes="speechbrain/spkrec-ecapa-voxceleb"))
class EcapaEncoder:
    name = "ecapa"
    dim = 192

    def __init__(self, device: str = "cuda", savedir: str | None = None):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        self.device = device if torch.cuda.is_available() else "cpu"
        # A complete local copy (shipped by the installer, or left by an earlier run) is used as is
        local = savedir and all((Path(savedir) / f).exists() for f in
                                ("hyperparams.yaml", "embedding_model.ckpt", "mean_var_norm_emb.ckpt",
                                 "classifier.ckpt", "label_encoder.ckpt"))
        # COPY instead of the default SYMLINK: symlinks need admin / Developer Mode on Windows.
        self.model = EncoderClassifier.from_hparams(
            source=savedir if local else "speechbrain/spkrec-ecapa-voxceleb", savedir=savedir,
            local_strategy=LocalStrategy.COPY, run_opts={"device": self.device})

    def embed(self, audio16k: np.ndarray) -> np.ndarray:
        import torch

        wav = torch.from_numpy(np.ascontiguousarray(audio16k, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            emb = self.model.encode_batch(wav).squeeze().float().cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-9)

    def unload(self) -> None:
        self.model = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
