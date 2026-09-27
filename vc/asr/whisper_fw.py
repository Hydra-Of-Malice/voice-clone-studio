"""faster-whisper ASR engine. Default model `large-v3-turbo`; set VC_ASR_MODEL to a CTranslate2
model directory to use a Hindi/Hinglish fine-tune (e.g. a converted Vaani-Whisper / Trelis checkpoint).

We always force task=transcribe (never translate) and keep the original script."""
from __future__ import annotations

import math

import numpy as np

from vc.asr.base import ASREngine, Segment, Transcript, Word, script_language
from vc.registry import Capabilities, registry


@registry.register(Capabilities(
    name="faster-whisper", kind="asr", license="MIT (code) / MIT (Whisper weights)", commercial_ok=True,
    languages=["*"], approx_vram_gb=2.0,
    notes="int8_float16 large-v3-turbo ~2 GB VRAM; swap VC_ASR_MODEL for Hindi fine-tunes"))
class FasterWhisperASR:
    name = "faster-whisper"

    def __init__(self, model: str = "large-v3-turbo", device: str = "cuda", compute_type: str | None = None,
                 download_root: str | None = None, **_):
        from faster_whisper import WhisperModel

        if compute_type is None:
            compute_type = "int8_float16" if device == "cuda" else "int8"
        self.model_name = model
        self.model = WhisperModel(model, device=device, compute_type=compute_type, download_root=download_root)

    def transcribe(self, audio16k: np.ndarray, language: str | None = None, chunks=None) -> Transcript:
        segments_iter, info = self.model.transcribe(
            audio16k.astype(np.float32), language=language, task="transcribe", beam_size=5,
            word_timestamps=True, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 200},
            condition_on_previous_text=False)
        segs: list[Segment] = []
        for s in segments_iter:
            words = [Word(float(w.start), float(w.end), w.word, float(w.probability)) for w in (s.words or [])]
            if words:
                conf = float(np.mean([w.probability for w in words]))
            else:
                conf = float(math.exp(s.avg_logprob)) if s.avg_logprob is not None else 0.5
            # Penalise hallucination markers
            if s.compression_ratio and s.compression_ratio > 2.4:
                conf *= 0.5
            if s.no_speech_prob and s.no_speech_prob > 0.6:
                conf *= 0.5
            text = s.text.strip()
            if not text:
                continue
            segs.append(Segment(float(s.start), float(s.end), text, round(min(conf, 1.0), 3),
                                script_language(text, info.language), words))
        transcript = " ".join(s.text for s in segs).strip()
        if segs:
            weights = np.array([max(s.end - s.start, 0.1) for s in segs])
            overall = float(np.average([s.confidence for s in segs], weights=weights))
        else:
            overall = 0.0
        return Transcript(transcript=transcript, language=info.language,
                          language_confidence=float(info.language_probability or 0.0),
                          confidence=overall, segments=segs, engine=f"faster-whisper:{self.model_name}")

    def unload(self) -> None:
        self.model = None
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
