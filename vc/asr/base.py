from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Protocol

import numpy as np

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")


def script_language(text: str, fallback: str = "en") -> str:
    """Language guess from script composition. Hinglish (mixed) is reported as 'hi' when Devanagari
    dominates, 'hi-Latn' when romanised Hindi is likely, 'en' otherwise."""
    # Word-level so that long English loanwords do not outvote short Hindi function words
    words = text.split()
    dev = sum(1 for w in words if _DEVANAGARI.search(w))
    lat = sum(1 for w in words if _LATIN.search(w) and not _DEVANAGARI.search(w))
    if dev == 0 and lat == 0:
        return fallback
    if dev / (dev + lat) >= 0.3:
        return "hi"
    return "en"


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class Segment:
    start: float
    end: float
    text: str
    confidence: float
    language: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    transcript: str
    language: str
    language_confidence: float
    confidence: float
    segments: list[Segment]
    engine: str

    def to_dict(self) -> dict:
        return {
            "transcript": self.transcript,
            "language": self.language,
            "language_confidence": round(self.language_confidence, 3),
            "confidence": round(self.confidence, 3),
            "engine": self.engine,
            "segments": [
                {**{k: v for k, v in asdict(s).items() if k != "words"},
                 "words": [asdict(w) for w in s.words]} for s in self.segments
            ],
        }

    @property
    def word_count(self) -> int:
        return len(self.transcript.split())

    def text_between(self, start: float, end: float) -> str:
        """Words whose midpoint falls inside [start, end] — used to build reference-clip transcripts."""
        words = []
        for seg in self.segments:
            if seg.words:
                for w in seg.words:
                    mid = (w.start + w.end) / 2
                    if start <= mid <= end:
                        words.append(w.word.strip())
            elif seg.start >= start - 0.2 and seg.end <= end + 0.2:
                words.append(seg.text.strip())
        return " ".join(words).strip()


class ASREngine(Protocol):
    name: str

    def transcribe(self, audio16k: np.ndarray, language: str | None = None) -> Transcript: ...
    def unload(self) -> None: ...
