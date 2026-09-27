"""Audio I/O without a system ffmpeg: soundfile for WAV/FLAC/OGG/MP3, PyAV (bundled with
faster-whisper) as a fallback decoder for anything else (webm/opus, m4a, ...)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr


def load_audio(path: str | Path, target_sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    """Return float32 audio in [-1, 1] and its sample rate."""
    path = str(path)
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        data, sr = _load_with_av(path)
    if mono and data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    audio = data[:, 0] if mono else data
    if target_sr and sr != target_sr:
        audio = resample(audio, sr, target_sr)
        sr = target_sr
    return np.ascontiguousarray(audio, dtype=np.float32), sr


def _load_with_av(path: str) -> tuple[np.ndarray, int]:
    import av  # provided by faster-whisper

    frames = []
    with av.open(path) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=stream.rate or 48000)
        for frame in container.decode(stream):
            for rf in resampler.resample(frame):
                frames.append(rf.to_ndarray().reshape(-1))
        sr = stream.rate or 48000
    if not frames:
        raise ValueError("No audio decoded")
    return np.concatenate(frames).reshape(-1, 1).astype(np.float32), sr


def resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio
    return soxr.resample(audio, sr, target_sr, quality="VHQ").astype(np.float32)


def save_wav(path: str | Path, audio: np.ndarray, sr: int, subtype: str = "PCM_16",
             comment: str | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(audio, -1.0, 1.0)
    with sf.SoundFile(str(path), "w", samplerate=sr, channels=1, subtype=subtype) as f:
        if comment:
            f.comment = comment
            f.software = "Voice Clone Studio"
        f.write(audio.astype(np.float32))


def save_mp3(path: str | Path, audio: np.ndarray, sr: int, bitrate: int = 320, comment: str | None = None) -> None:
    import lameenc

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    enc = lameenc.Encoder()
    enc.set_bit_rate(bitrate)
    enc.set_in_sample_rate(sr)
    enc.set_channels(1)
    enc.set_quality(2)
    mp3 = enc.encode(pcm.tobytes()) + enc.flush()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        if comment:
            f.write(_id3v2(comment))
        f.write(mp3)


def _id3v2(comment: str) -> bytes:
    """Minimal ID3v2.3 tag with a COMM frame carrying the provenance notice."""
    text = comment.encode("utf-8")
    body = b"\x03" + b"eng" + b"\x00" + text          # UTF-8 encoding byte, language, empty short desc
    frame = b"COMM" + len(body).to_bytes(4, "big") + b"\x00\x00" + body
    size = len(frame)
    syncsafe = bytes([(size >> 21) & 0x7F, (size >> 14) & 0x7F, (size >> 7) & 0x7F, size & 0x7F])
    return b"ID3\x03\x00\x00" + syncsafe + frame


def duration_seconds(path: str | Path) -> float:
    try:
        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        audio, sr = load_audio(path)
        return len(audio) / sr
