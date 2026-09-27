"""Silero VAD wrapper (MIT). Works on 16 kHz mono float audio."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass
class SpeechSpan:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@lru_cache(maxsize=1)
def _model():
    from silero_vad import load_silero_vad
    return load_silero_vad()


def speech_spans(audio16k: np.ndarray, threshold: float = 0.5, min_silence_ms: int = 250,
                 min_speech_ms: int = 200, pad_ms: int = 60) -> list[SpeechSpan]:
    import torch
    from silero_vad import get_speech_timestamps

    ts = get_speech_timestamps(torch.from_numpy(audio16k), _model(), sampling_rate=16000,
                               threshold=threshold, min_silence_duration_ms=min_silence_ms,
                               min_speech_duration_ms=min_speech_ms, speech_pad_ms=pad_ms,
                               return_seconds=True)
    return [SpeechSpan(float(t["start"]), float(t["end"])) for t in ts]


def speech_mask(audio16k: np.ndarray, spans: list[SpeechSpan]) -> np.ndarray:
    mask = np.zeros(len(audio16k), dtype=bool)
    for s in spans:
        mask[int(s.start * 16000): int(s.end * 16000)] = True
    return mask
