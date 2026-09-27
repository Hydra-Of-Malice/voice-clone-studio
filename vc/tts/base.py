from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class StylePreset:
    name: str
    exaggeration: float
    cfg_weight: float
    temperature: float


STYLE_PRESETS: dict[str, StylePreset] = {
    "natural": StylePreset("natural", 0.5, 0.5, 0.8),
    "conversational": StylePreset("conversational", 0.6, 0.4, 0.8),
    "professional": StylePreset("professional", 0.4, 0.6, 0.7),
    "energetic": StylePreset("energetic", 0.8, 0.3, 0.9),
}


@dataclass
class ReferencePrompt:
    audio_path: str          # decrypted temp WAV at the engine's expected rate
    transcript: str
    language: str


class TTSEngine(Protocol):
    name: str
    sample_rate: int

    def synthesize(self, text: str, language: str, ref: ReferencePrompt, style: StylePreset,
                   seed: int | None = None) -> np.ndarray:
        """Return mono float32 audio at `sample_rate` for one text chunk."""
        ...

    def supports_language(self, language: str) -> bool: ...
    def unload(self) -> None: ...
