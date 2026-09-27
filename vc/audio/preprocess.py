"""Sample preprocessing: gentle level normalisation, conditional enhancement, and slicing the long
recording into chunks at natural pause boundaries (keeping the speaker's pauses).

Enhancement is deliberately conservative: denoising a clean prompt measurably lowers cloned speaker
similarity, so it only runs below the SNR threshold, and the result is kept only if it improves SNR
without moving the speaker embedding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from vc.audio.vad import SpeechSpan


@dataclass
class Clip:
    start: float
    end: float
    audio: np.ndarray   # at the working sample rate
    sr: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def normalize_level(audio: np.ndarray, target_dbfs: float = -20.0, max_gain_db: float = 3.0) -> np.ndarray:
    """Emilia-style: move RMS toward target but clamp the gain so the noise floor is not lifted."""
    rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
    gain_db = np.clip(target_dbfs - 20 * np.log10(rms), -max_gain_db * 4, max_gain_db)
    out = audio * (10 ** (gain_db / 20))
    peak = np.max(np.abs(out))
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def maybe_enhance(audio: np.ndarray, sr: int, snr_db: float, threshold_db: float = 20.0,
                  snr_fn: Callable[[np.ndarray], float] | None = None,
                  identity_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
                  min_identity: float = 0.75, min_gain_db: float = 2.0) -> tuple[np.ndarray, dict]:
    """Returns (audio, report). `snr_fn(audio)` re-measures SNR, `identity_fn(raw, enhanced)` returns
    the speaker-embedding cosine between the two versions."""
    report: dict = {"applied": False, "snr_before_db": round(float(snr_db), 1)}
    if snr_db >= threshold_db:
        report["reason"] = f"skipped: already clean (SNR >= {threshold_db:.0f} dB)"
        return audio, report
    try:
        from vc.audio.enhance import denoise
        enhanced = denoise(audio, sr)
    except Exception as e:  # noqa: BLE001
        report["reason"] = f"skipped: enhancer unavailable ({type(e).__name__}: {e})"
        return audio, report
    report["engine"] = "deepfilternet3"
    if snr_fn is not None:
        after = snr_fn(enhanced)
        report["snr_after_db"] = round(float(after), 1)
        if after - snr_db < min_gain_db:
            report["reason"] = "reverted: no meaningful SNR gain"
            return audio, report
    if identity_fn is not None:
        ident = identity_fn(audio, enhanced)
        report["identity_cosine"] = round(float(ident), 3)
        if ident < min_identity:
            report["reason"] = "reverted: enhancement changed the voice"
            return audio, report
    report["applied"] = True
    report["reason"] = "applied: noisy upload"
    return enhanced, report


def merge_spans(spans: list[SpeechSpan], max_s: float = 12.0, max_gap_s: float = 1.5) -> list[SpeechSpan]:
    """Group VAD spans into chunks of at most `max_s` seconds, cutting only in pauses. Every span ends
    up in exactly one chunk, so the chunks cover all speech (used for ASR and for reference clips)."""
    chunks: list[SpeechSpan] = []
    cur: SpeechSpan | None = None
    for span in spans:
        # A single span longer than max_s is split into equal windows
        if span.duration > max_s:
            if cur is not None:
                chunks.append(cur)
                cur = None
            n = int(np.ceil(span.duration / max_s))
            step = span.duration / n
            chunks.extend(SpeechSpan(span.start + i * step, span.start + (i + 1) * step) for i in range(n))
            continue
        if cur is None:
            cur = SpeechSpan(span.start, span.end)
        elif span.start - cur.end > max_gap_s or span.end - cur.start > max_s:
            chunks.append(cur)
            cur = SpeechSpan(span.start, span.end)
        else:
            cur = SpeechSpan(cur.start, span.end)
    if cur is not None:
        chunks.append(cur)
    return chunks


def slice_clips(audio: np.ndarray, sr: int, spans: list[SpeechSpan], min_s: float = 6.0, max_s: float = 12.0,
                tail_silence_s: float = 0.4) -> list[Clip]:
    """Reference-clip candidates: the chunks from `merge_spans` that are at least `min_s` long, with a
    little trailing silence kept for TTS prompts."""
    clips: list[Clip] = []
    for ch in merge_spans(spans, max_s):
        if ch.duration < min_s:
            continue
        s = int(ch.start * sr)
        e = int(min(len(audio), (ch.end + tail_silence_s) * sr))
        clips.append(Clip(ch.start, e / sr, audio[s:e].copy(), sr))
    return clips
