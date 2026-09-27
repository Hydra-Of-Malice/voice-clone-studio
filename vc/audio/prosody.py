"""Prosody / speaking-style profiling: pitch statistics, energy, speaking rate and pause behaviour.
Uses librosa's pYIN (installed with chatterbox) so no extra dependency is needed; swap in
SwiftF0/RMVPE via the same function signature if higher accuracy is wanted."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from vc.audio.vad import SpeechSpan


@dataclass
class ProsodyProfile:
    f0_median_hz: float
    f0_mean_hz: float
    f0_std_semitones: float
    f0_range_semitones: float
    energy_rms_db: float
    energy_std_db: float
    speaking_rate_wps: float          # words per second of speech (from ASR word count)
    articulation_rate_sps: float      # syllable-nuclei estimate per second of speech
    pause_count: int
    pause_mean_s: float
    pause_ratio: float
    style_label: str

    def to_dict(self) -> dict:
        return asdict(self)


def _f0_track(audio16k: np.ndarray) -> np.ndarray:
    import librosa
    f0, voiced, _ = librosa.pyin(audio16k, fmin=60, fmax=500, sr=16000, frame_length=1024, hop_length=160)
    f0 = f0[voiced & np.isfinite(f0)] if f0 is not None else np.array([])
    return f0


def _syllable_nuclei_rate(audio16k: np.ndarray, spans: list[SpeechSpan]) -> float:
    """Rough De Jong & Wempe style count: intensity peaks >2 dB above the local dip within speech."""
    hop = 160
    frame = 400
    n = (len(audio16k) - frame) // hop
    if n <= 0:
        return 0.0
    idx = np.arange(n)[:, None] * hop + np.arange(frame)[None, :]
    env = 20 * np.log10(np.sqrt((audio16k[idx] ** 2).mean(axis=1)) + 1e-6)
    # smooth ~50 ms
    k = np.ones(5) / 5
    env = np.convolve(env, k, mode="same")
    thr = np.median(env[env > env.max() - 40]) - 2
    peaks = 0
    for i in range(1, len(env) - 1):
        if env[i] > env[i - 1] and env[i] >= env[i + 1] and env[i] > thr:
            # require a dip of 2 dB within the next 15 frames
            if env[i] - env[i + 1: i + 16].min(initial=env[i]) > 2:
                peaks += 1
    speech_s = sum(s.duration for s in spans) or 1.0
    return peaks / speech_s


def analyze(audio16k: np.ndarray, spans: list[SpeechSpan], word_count: int | None = None) -> ProsodyProfile:
    speech_s = sum(s.duration for s in spans) or 1e-6
    total_s = len(audio16k) / 16000

    # Pitch on speech only (cap at 3 minutes for speed)
    mask_idx = np.concatenate([np.arange(int(s.start * 16000), int(s.end * 16000)) for s in spans]) if spans else np.arange(len(audio16k))
    speech_audio = audio16k[mask_idx[: 16000 * 180]]
    f0 = _f0_track(speech_audio)
    if len(f0) > 10:
        semis = 12 * np.log2(f0 / 100.0)
        f0_median, f0_mean = float(np.median(f0)), float(np.mean(f0))
        f0_std_st = float(np.std(semis))
        f0_range_st = float(np.percentile(semis, 95) - np.percentile(semis, 5))
    else:
        f0_median = f0_mean = 0.0
        f0_std_st = f0_range_st = 0.0

    # Energy
    hop, frame = 160, 400
    n = (len(speech_audio) - frame) // hop
    if n > 0:
        idx = np.arange(n)[:, None] * hop + np.arange(frame)[None, :]
        rms_db = 20 * np.log10(np.sqrt((speech_audio[idx] ** 2).mean(axis=1)) + 1e-6)
        energy_db, energy_std = float(np.mean(rms_db)), float(np.std(rms_db))
    else:
        energy_db = energy_std = 0.0

    # Pauses between speech spans
    gaps = [b.start - a.end for a, b in zip(spans, spans[1:]) if b.start - a.end >= 0.15]
    pause_count = len(gaps)
    pause_mean = float(np.mean(gaps)) if gaps else 0.0
    pause_ratio = float(sum(gaps) / total_s) if total_s else 0.0

    art_rate = _syllable_nuclei_rate(audio16k, spans)
    wps = (word_count / speech_s) if word_count else 0.0

    # Coarse style label from rate + pitch variability + energy variability
    if art_rate > 5.2 and f0_std_st > 3.0:
        style = "energetic"
    elif f0_std_st < 1.8 and art_rate < 4.0:
        style = "calm / measured"
    elif pause_ratio > 0.25:
        style = "deliberate / professional"
    else:
        style = "conversational"

    return ProsodyProfile(f0_median_hz=round(f0_median, 1), f0_mean_hz=round(f0_mean, 1),
                          f0_std_semitones=round(f0_std_st, 2), f0_range_semitones=round(f0_range_st, 2),
                          energy_rms_db=round(energy_db, 1), energy_std_db=round(energy_std, 2),
                          speaking_rate_wps=round(wps, 2), articulation_rate_sps=round(art_rate, 2),
                          pause_count=pause_count, pause_mean_s=round(pause_mean, 3),
                          pause_ratio=round(pause_ratio, 3), style_label=style)
