"""No-reference quality assessment of an uploaded voice sample.

Physics-based metrics (SNR, clipping, bandwidth, loudness, speech ratio, duration) are weighted above
learned MOS predictors on purpose: they are language-agnostic, which matters for Hindi/Hinglish where
English-trained predictors are unvalidated. The result is a 0-100 score plus human-readable fixes."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pyloudnorm as pyln

from vc.audio.vad import SpeechSpan, speech_mask


@dataclass
class QualityReport:
    duration_s: float
    speech_s: float
    speech_ratio: float
    snr_db: float
    clipping_ratio: float
    bandwidth_hz: float
    loudness_lufs: float
    peak_dbfs: float
    noise_level: str
    score: int
    speakers_detected: int = 1
    speaker_consistency: float = 1.0
    accepted: bool = True
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_snr_db(audio16k: np.ndarray, spans: list[SpeechSpan]) -> float:
    """Speech-vs-pause power ratio. Falls back to a percentile-based estimate when the VAD finds
    almost no pauses (continuous speech)."""
    mask = speech_mask(audio16k, spans)
    frame = 400  # 25 ms
    hop = 160
    n = (len(audio16k) - frame) // hop
    if n <= 0:
        return 0.0
    idx = np.arange(n)[:, None] * hop + np.arange(frame)[None, :]
    frames = audio16k[idx]
    power = (frames ** 2).mean(axis=1) + 1e-10
    frame_mask = mask[idx].mean(axis=1) > 0.5
    speech_p = power[frame_mask]
    noise_p = power[~frame_mask]
    if len(noise_p) < 20 or len(speech_p) < 20:
        # Continuous speech: assume the quietest 10% of frames approximate the noise floor
        srt = np.sort(power)
        noise_p = srt[: max(20, len(srt) // 10)]
        speech_p = srt[len(srt) // 2:]
    # Median for the noise floor: VAD padding leaks speech onsets/offsets into the pause frames
    snr = 10 * np.log10(np.mean(speech_p) / np.median(noise_p))
    return float(np.clip(snr, -10, 80))


def effective_bandwidth_hz(audio: np.ndarray, sr: int, threshold_db: float = -50.0) -> float:
    """Highest frequency whose long-term spectrum stays within `threshold_db` of the peak
    (URGENT-challenge style). Detects upsampled / band-limited recordings."""
    n_fft = 2048
    if len(audio) < n_fft:
        return sr / 2
    hop = n_fft // 2
    n = (len(audio) - n_fft) // hop
    idx = np.arange(min(n, 4000)) * max(1, n // 4000)
    win = np.hanning(n_fft)
    spec = np.zeros(n_fft // 2 + 1)
    for i in idx:
        seg = audio[i * hop: i * hop + n_fft] * win
        spec += np.abs(np.fft.rfft(seg)) ** 2
    spec_db = 10 * np.log10(spec / spec.max() + 1e-12)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    above = np.where(spec_db > threshold_db)[0]
    return float(freqs[above[-1]]) if len(above) else 0.0


def assess(audio: np.ndarray, sr: int, audio16k: np.ndarray, spans: list[SpeechSpan],
           speakers_detected: int = 1, speaker_consistency: float = 1.0,
           min_seconds: float = 60.0, rec_min: float = 300.0, rec_max: float = 600.0,
           min_score: int = 55) -> QualityReport:
    duration = len(audio) / sr
    speech_s = sum(s.duration for s in spans)
    speech_ratio = speech_s / duration if duration else 0.0
    snr = estimate_snr_db(audio16k, spans)
    clipping = float(np.mean(np.abs(audio) >= 0.985))
    bandwidth = effective_bandwidth_hz(audio, sr)
    peak_dbfs = float(20 * np.log10(np.max(np.abs(audio)) + 1e-9))
    try:
        loudness = float(pyln.Meter(sr).integrated_loudness(audio))
    except Exception:
        loudness = -70.0

    issues: list[str] = []
    tips: list[str] = []
    score = 100.0

    # --- SNR (35%) --------------------------------------------------------------------------
    if snr >= 30:
        snr_pts = 35
    elif snr >= 20:
        snr_pts = 35 * (0.6 + 0.4 * (snr - 20) / 10)
    elif snr >= 10:
        snr_pts = 35 * (0.25 + 0.35 * (snr - 10) / 10)
        issues.append("Noticeable background noise")
        tips.append("Record in a quieter room, closer to the microphone (15-20 cm), and switch off fans/AC.")
    else:
        snr_pts = 35 * 0.25 * max(snr, 0) / 10
        issues.append("Heavy background noise")
        tips.append("The recording is too noisy to clone well. Please re-record in a quiet space.")
    score -= 35 - snr_pts

    # --- Clipping (10%) ---------------------------------------------------------------------
    if clipping > 0.002:
        score -= 10
        issues.append("Clipping / distortion detected")
        tips.append("Lower the input gain or move slightly away from the mic so loud words do not distort.")
    elif clipping > 0.0003:
        score -= 4

    # --- Bandwidth (10%) --------------------------------------------------------------------
    if bandwidth < 4000:
        score -= 10
        issues.append("Telephone-quality bandwidth")
        tips.append("Use a proper microphone or lossless format instead of a phone call / low-bitrate voice note.")
    elif bandwidth < 7000:
        score -= 5
        issues.append("Limited high-frequency content")

    # --- Levels (10%) -----------------------------------------------------------------------
    if loudness < -35:
        score -= 10
        issues.append("Recording is very quiet")
        tips.append("Increase the microphone gain or speak closer to the mic.")
    elif loudness < -28:
        score -= 4
    if peak_dbfs > -0.5 and clipping <= 0.0003:
        score -= 2

    # --- Speech ratio (10%) -----------------------------------------------------------------
    if speech_ratio < 0.4:
        score -= 10
        issues.append("Long silences (less than 40% speech)")
        tips.append("Trim long pauses or keep talking naturally; a little pause between sentences is fine.")
    elif speech_ratio < 0.55:
        score -= 4

    # --- Duration (10%) ---------------------------------------------------------------------
    if speech_s < min_seconds:
        score -= 10
        issues.append(f"Too short: only {speech_s/60:.1f} min of speech")
        tips.append(f"Please provide at least {rec_min/60:.0f} minutes of speech (5-10 minutes is ideal).")
    elif speech_s < rec_min:
        score -= 6
        issues.append(f"Short sample ({speech_s/60:.1f} min of speech)")
        tips.append("5-10 minutes of natural speech with varied sentences gives the best result.")
    elif speech_s > rec_max * 1.5:
        score -= 2

    # --- Speaker purity (15%) ---------------------------------------------------------------
    if speakers_detected > 1:
        score -= 15
        issues.append(f"{speakers_detected} speakers detected")
        tips.append("Use a recording with a single speaker and no overlapping voices.")
    elif speaker_consistency < 0.6:
        score -= 6
        issues.append("Voice or microphone changes during the recording")
        tips.append("Keep the same microphone, distance and speaking style throughout.")

    noise_level = "Low" if snr >= 25 else "Medium" if snr >= 15 else "High"
    score_i = int(np.clip(round(score), 0, 100))
    hard_reject = speech_s < min_seconds or speakers_detected > 1 or snr < 5
    accepted = score_i >= min_score and not hard_reject
    if not accepted and not tips:
        tips.append("Please upload a cleaner, longer single-speaker recording.")

    return QualityReport(duration_s=round(duration, 2), speech_s=round(speech_s, 2),
                         speech_ratio=round(speech_ratio, 3), snr_db=round(snr, 1),
                         clipping_ratio=round(clipping, 5), bandwidth_hz=round(bandwidth),
                         loudness_lufs=round(loudness, 1), peak_dbfs=round(peak_dbfs, 1),
                         noise_level=noise_level, score=score_i, speakers_detected=speakers_detected,
                         speaker_consistency=round(speaker_consistency, 3), accepted=accepted,
                         issues=issues, suggestions=tips)
