"""Language-routing ASR: Whisper language ID first, then the best engine for that language.
Hindi (and Hinglish) -> Trelis whisper-hinglish; everything else -> faster-whisper."""
from __future__ import annotations

import logging

import numpy as np

from vc.asr.base import Transcript
from vc.audio.vad import SpeechSpan
from vc.registry import Capabilities, registry

log = logging.getLogger("vc.asr.router")
# Whisper often labels Hindi speech as one of its close neighbours
_HINDI_LIKE = {"hi", "ur", "mr", "ne", "pa", "sa"}


@registry.register(Capabilities(
    name="auto", kind="asr", license="MIT + Apache-2.0", commercial_ok=True, languages=["*"],
    approx_vram_gb=3.5, notes="routes hi/Hinglish to whisper-hinglish, other languages to faster-whisper"))
class RoutedASR:
    name = "auto"

    def __init__(self, model: str = "large-v3-turbo", device: str = "cuda", download_root: str | None = None,
                 hindi_model: str | None = None, **_):
        self.device = device
        self.download_root = download_root
        self.general_model = model
        self.hindi_model = hindi_model
        self._general = None
        self._hindi = None

    def general(self):
        if self._general is None:
            from vc.asr.whisper_fw import FasterWhisperASR
            self._general = FasterWhisperASR(self.general_model, self.device, download_root=self.download_root)
        return self._general

    def hindi(self):
        if self._hindi is None:
            from vc.asr.whisper_hf import WhisperHinglishASR
            self._hindi = WhisperHinglishASR(self.hindi_model, self.device, download_root=self.download_root)
        return self._hindi

    def detect_language(self, audio16k: np.ndarray, chunks: list[SpeechSpan] | None = None) -> tuple[str, float]:
        """Majority vote over up to three 30 s windows of speech."""
        fw = self.general().model
        starts = [c.start for c in chunks] if chunks else [0.0]
        total = len(audio16k) / 16000
        picks = [starts[0], starts[len(starts) // 2], starts[-1]] if len(starts) >= 3 else starts[:1]
        votes: dict[str, float] = {}
        for t in dict.fromkeys(picks):
            seg = audio16k[int(t * 16000): int(min(total, t + 30) * 16000)]
            if len(seg) < 16000:
                continue
            lang, prob, all_probs = fw.detect_language(seg)
            for l, p in (all_probs or [(lang, prob)])[:5]:
                votes[l] = votes.get(l, 0.0) + float(p)
        if not votes:
            return "en", 0.0
        n = max(1, len(set(picks)))
        hindi_like = sum(p for l, p in votes.items() if l in _HINDI_LIKE) / n
        best = max(votes, key=votes.get)
        if hindi_like >= 0.5 or best in _HINDI_LIKE:
            return "hi", round(min(1.0, hindi_like), 3)
        return best, round(min(1.0, votes[best] / n), 3)

    def transcribe(self, audio16k: np.ndarray, language: str | None = None,
                   chunks: list[SpeechSpan] | None = None) -> Transcript:
        if language:
            lang, prob = language.split("-")[0], 1.0
        else:
            lang, prob = self.detect_language(audio16k, chunks)
        log.info("ASR language: %s (%.2f)", lang, prob)
        if lang == "hi":
            if self._general is not None:            # free VRAM before loading the 1.5B model
                self._general.unload()
                self._general = None
            tr = self.hindi().transcribe(audio16k, "hi", chunks)
            tr.language_confidence = prob
            return tr
        if self._hindi is not None:
            self._hindi.unload()
            self._hindi = None
        tr = self.general().transcribe(audio16k, lang)
        tr.language_confidence = prob if not language else tr.language_confidence
        return tr

    def unload(self) -> None:
        for e in (self._general, self._hindi):
            if e is not None:
                e.unload()
        self._general = self._hindi = None
