"""Post-synthesis processing: join chunks with crossfades and speaker-matched pauses, optional
speed change (WSOLA), loudness normalisation with true-peak ceiling, resample to 48 kHz, artifact
checks, and export to 24-bit WAV + 320 kbps MP3 carrying an AI-generated provenance notice."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pyloudnorm as pyln

from vc.audio.io import resample, save_mp3, save_wav

PROVENANCE_NOTE = "AI-generated synthetic speech (Voice Clone Studio). Created with the speaker's recorded consent."


@dataclass
class OutputMetrics:
    duration_s: float
    loudness_lufs: float
    peak_dbfs: float
    clipping_ratio: float
    chunks: int

    def to_dict(self) -> dict:
        return asdict(self)


def _fade(n: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n, dtype=np.float32)


def join_chunks(chunks: list[np.ndarray], pauses_s: list[float], sr: int, crossfade_s: float = 0.03) -> np.ndarray:
    """Concatenate chunk audio. Leading/trailing silences of each chunk are trimmed to ~80 ms so the
    inserted pause length reflects the speaker's own pause statistics."""
    out = np.zeros(0, dtype=np.float32)
    xf = int(crossfade_s * sr)
    for i, chunk in enumerate(chunks):
        chunk = _trim_silence(chunk, sr)
        if len(out) == 0:
            out = chunk
        else:
            pause = np.zeros(int(max(pauses_s[i - 1], 0.0) * sr), dtype=np.float32)
            if len(pause) > 0 or xf == 0 or len(out) < xf or len(chunk) < xf:
                out = np.concatenate([out, pause, chunk])
            else:
                f = _fade(xf)
                out[-xf:] = out[-xf:] * (1 - f) + chunk[:xf] * f
                out = np.concatenate([out, chunk[xf:]])
    return out


def _trim_silence(audio: np.ndarray, sr: int, thresh_db: float = -45.0, keep_s: float = 0.08) -> np.ndarray:
    if len(audio) == 0:
        return audio
    hop = int(0.01 * sr)
    n = max(1, len(audio) // hop)
    env = np.array([np.sqrt(np.mean(audio[i * hop:(i + 1) * hop] ** 2)) for i in range(n)])
    db = 20 * np.log10(env + 1e-9)
    loud = np.where(db > thresh_db)[0]
    if len(loud) == 0:
        return audio
    s = max(0, loud[0] * hop - int(keep_s * sr))
    e = min(len(audio), (loud[-1] + 1) * hop + int(keep_s * sr))
    return audio[s:e]


def change_speed(audio: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Time-stretch without pitch change (WSOLA). speed 1.2 = 20% faster."""
    if abs(speed - 1.0) < 0.02:
        return audio
    try:
        from audiotsm import wsola
        from audiotsm.io.array import ArrayReader, ArrayWriter

        reader = ArrayReader(audio.reshape(1, -1).astype(np.float32))
        writer = ArrayWriter(1)
        tsm = wsola(1, speed=speed)
        tsm.run(reader, writer)
        return writer.data.reshape(-1).astype(np.float32)
    except Exception:
        import librosa
        return librosa.effects.time_stretch(audio, rate=speed).astype(np.float32)


def normalize_loudness(audio: np.ndarray, sr: int, target_lufs: float = -16.0, true_peak_dbtp: float = -1.0) -> np.ndarray:
    meter = pyln.Meter(sr)
    try:
        loud = meter.integrated_loudness(audio)
    except Exception:
        loud = -70.0
    if np.isfinite(loud) and loud > -60:
        audio = audio * (10 ** ((target_lufs - loud) / 20))
    audio = _limit(audio, sr, 10 ** ((true_peak_dbtp - 0.3) / 20))
    # True-peak approximation: measure on 4x oversampled signal
    over = resample(audio, sr, sr * 4) if len(audio) > 0 else audio
    peak = np.max(np.abs(over)) if len(over) else 0.0
    ceiling = 10 ** (true_peak_dbtp / 20)
    if peak > ceiling:
        audio = audio * (ceiling / peak)
    return audio.astype(np.float32)


def _limit(audio: np.ndarray, sr: int, ceiling: float) -> np.ndarray:
    """Look-ahead peak limiter: gain reduction only around peaks above the ceiling (5 ms look-ahead,
    ~60 ms smoothing), so loudness can reach the target without clipping or audible pumping."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d

    mag = np.abs(audio)
    if len(audio) == 0 or mag.max() <= ceiling:
        return audio
    gain = np.minimum(1.0, ceiling / np.maximum(mag, 1e-9))
    gain = minimum_filter1d(gain, size=max(3, int(0.010 * sr)) | 1)
    gain = uniform_filter1d(gain, size=max(3, int(0.060 * sr)) | 1)
    gain = np.minimum(gain, minimum_filter1d(np.minimum(1.0, ceiling / np.maximum(mag, 1e-9)), size=3))
    return (audio * gain).astype(np.float32)


def light_denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """Remove DC offset and sub-50 Hz rumble; anything heavier would colour the cloned voice."""
    from scipy.signal import butter, sosfiltfilt

    audio = audio - np.mean(audio)
    sos = butter(2, 50, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def finalize(chunks: list[np.ndarray], pauses_s: list[float], sr_in: int, out_dir: Path, base_name: str,
             speed: float = 1.0, out_sr: int = 48000, target_lufs: float = -16.0, true_peak_dbtp: float = -1.0,
             mp3_bitrate: int = 320, audioseal_payload: int | None = None, signer=None,
             manifest_details: dict | None = None) -> tuple[Path, Path, OutputMetrics, dict]:
    audio = join_chunks(chunks, pauses_s, sr_in)
    audio = change_speed(audio, sr_in, speed)
    audio = light_denoise(audio, sr_in)
    audio = resample(audio, sr_in, out_sr)
    audio = normalize_loudness(audio, out_sr, target_lufs, true_peak_dbtp)
    prov: dict = {"audioseal": None, "c2pa": None}
    if audioseal_payload is not None:
        try:
            from vc.provenance import embed_audioseal
            audio = embed_audioseal(audio, out_sr, audioseal_payload)
            prov["audioseal"] = {"embedded": True, "payload": audioseal_payload}
        except Exception as e:  # noqa: BLE001  (never lose the audio because a watermark failed)
            prov["audioseal"] = {"embedded": False, "error": f"{type(e).__name__}: {e}"}
    # short fades to avoid clicks at the edges
    n = int(0.01 * out_sr)
    if len(audio) > 2 * n:
        audio[:n] *= _fade(n)
        audio[-n:] *= _fade(n)[::-1]

    wav_path = out_dir / f"{base_name}.wav"
    mp3_path = out_dir / f"{base_name}.mp3"
    save_wav(wav_path, audio, out_sr, subtype="PCM_24", comment=PROVENANCE_NOTE)
    save_mp3(mp3_path, audio, out_sr, bitrate=mp3_bitrate, comment=PROVENANCE_NOTE)
    if signer is not None:
        try:
            for p in (wav_path, mp3_path):
                prov["c2pa"] = signer.sign_file(p, f"Synthetic speech {base_name}", manifest_details or {})
        except Exception as e:  # noqa: BLE001
            prov["c2pa"] = {"signed": False, "error": f"{type(e).__name__}: {e}"}

    try:
        loud = float(pyln.Meter(out_sr).integrated_loudness(audio))
    except Exception:
        loud = -70.0
    metrics = OutputMetrics(duration_s=round(len(audio) / out_sr, 2), loudness_lufs=round(loud, 1),
                            peak_dbfs=round(float(20 * np.log10(np.max(np.abs(audio)) + 1e-9)), 2),
                            clipping_ratio=float(np.mean(np.abs(audio) >= 0.999)), chunks=len(chunks))
    return wav_path, mp3_path, metrics, prov
