from __future__ import annotations

from typing import Protocol

import numpy as np


class SpeakerEncoder(Protocol):
    name: str
    dim: int

    def embed(self, audio16k: np.ndarray) -> np.ndarray:
        """L2-normalised embedding for one utterance (16 kHz mono)."""
        ...

    def unload(self) -> None: ...


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def robust_centroid(embs: np.ndarray, iters: int = 3) -> np.ndarray:
    """Mean embedding after iteratively dropping the 20% least similar clips (guards against a
    second speaker or a noisy clip pulling the centroid)."""
    c = embs.mean(axis=0)
    keep = np.ones(len(embs), dtype=bool)
    for _ in range(iters):
        sims = embs @ (c / (np.linalg.norm(c) + 1e-9))
        thr = np.percentile(sims[keep], 20) if keep.sum() > 5 else -1
        keep = sims >= thr
        c = embs[keep].mean(axis=0)
    return c / (np.linalg.norm(c) + 1e-9)


def count_speakers(embs: np.ndarray, split_threshold: float = 0.35) -> tuple[int, float]:
    """Very small agglomerative check: if the clips form two clusters whose centroids are far apart
    (cosine < split_threshold) we report 2 speakers. Returns (n_speakers, consistency)."""
    if len(embs) < 4:
        return 1, 1.0
    c = robust_centroid(embs)
    sims = embs @ c
    consistency = float(np.mean(sims))
    low = embs[sims < np.percentile(sims, 30)]
    high = embs[sims >= np.percentile(sims, 30)]
    if len(low) >= 2 and len(high) >= 2:
        c_low = low.mean(axis=0); c_low /= np.linalg.norm(c_low) + 1e-9
        c_high = high.mean(axis=0); c_high /= np.linalg.norm(c_high) + 1e-9
        if float(c_low @ c_high) < split_threshold:
            return 2, consistency
    return 1, consistency
