"""Speaker profile construction and loading.

A profile is engine-agnostic: curated reference clips (6-12 s, cut at pauses) with their transcripts,
a generic speaker embedding (centroid) for verification, per-clip embeddings, prosody statistics and
the language. Per-engine artifacts can be cached later without re-enrolment. Clips and embeddings are
stored encrypted; `open_reference()` decrypts one clip to a temp file for the TTS engine."""
from __future__ import annotations

import base64
import json
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from vc.asr.base import Transcript, script_language
from vc.audio.io import resample, save_wav
from vc.audio.preprocess import Clip
from vc.audio.prosody import ProsodyProfile
from vc.security import KeyStore, ProfileCipher
from vc.speaker.base import SpeakerEncoder, cosine, robust_centroid


@dataclass
class RefClip:
    file: str
    start: float
    end: float
    duration: float
    transcript: str
    language: str
    centroid_cosine: float
    rate_sps: float        # articulation-rate proxy for style bucketing
    energy_db: float
    f0_median_hz: float
    style_bucket: str


@dataclass
class SpeakerProfile:
    speaker_id: str
    name: str
    language: str
    embedding_engine: str
    voice_embedding_b64: str
    speaking_rate: dict
    pitch_statistics: dict
    energy_profile: dict
    style_features: dict
    quality_score: int
    clips: list[RefClip] = field(default_factory=list)
    intra_speaker_cosine_mean: float = 0.0
    intra_speaker_cosine_std: float = 0.0
    created_at: float = field(default_factory=time.time)
    version: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def voice_embedding(self) -> np.ndarray:
        return np.frombuffer(base64.b64decode(self.voice_embedding_b64), dtype=np.float32)

    def best_clip(self, style: str = "natural", language: str | None = None) -> RefClip:
        """Identity is anchored to centroid-nearest clips; style requests pick among the top clips."""
        cands = sorted(self.clips, key=lambda c: -c.centroid_cosine)
        if language:
            same = [c for c in cands if c.language.split("-")[0] == language.split("-")[0]]
            if len(same) >= 2:
                cands = same
        top = cands[: max(3, len(cands) // 3)]
        if style == "energetic":
            return max(top, key=lambda c: (c.rate_sps, c.energy_db))
        if style == "professional":
            return min(top, key=lambda c: c.rate_sps)
        if style == "conversational":
            mid = sorted(top, key=lambda c: c.rate_sps)
            return mid[len(mid) // 2]
        return top[0]


def _clip_prosody(audio16k: np.ndarray) -> tuple[float, float, float]:
    from vc.audio.prosody import _f0_track, _syllable_nuclei_rate
    from vc.audio.vad import SpeechSpan

    dur = len(audio16k) / 16000
    rate = _syllable_nuclei_rate(audio16k, [SpeechSpan(0.0, dur)])
    rms_db = float(20 * np.log10(np.sqrt(np.mean(audio16k ** 2)) + 1e-6))
    f0 = _f0_track(audio16k[: 16000 * 12])
    f0_med = float(np.median(f0)) if len(f0) > 5 else 0.0
    return rate, rms_db, f0_med


def build_profile(profile_id: str, name: str, clips: list[Clip], sr: int, transcript: Transcript,
                  prosody: ProsodyProfile, encoder: SpeakerEncoder, quality_score: int, profiles_dir: Path,
                  keystore: KeyStore, bank_size: int = 12) -> tuple[SpeakerProfile, bytes]:
    """Embed every clip, compute the robust centroid, rank clips, store the top `bank_size` encrypted.
    Returns the profile and the wrapped data key (to be stored in the DB row)."""
    pdir = profiles_dir / profile_id
    pdir.mkdir(parents=True, exist_ok=True)
    key, wrapped = keystore.new_profile_key()
    cipher = ProfileCipher(key)

    clips16 = [resample(c.audio, sr, 16000) for c in clips]
    embs = np.stack([encoder.embed(a) for a in clips16])
    centroid = robust_centroid(embs)
    sims = embs @ centroid
    order = np.argsort(-sims)

    # Prefer clips whose transcript is usable and which sit near the centroid
    chosen: list[int] = []
    for i in order:
        text = transcript.text_between(clips[i].start, clips[i].end)
        if len(text.split()) < 3 and len(chosen) > bank_size // 2:
            continue
        chosen.append(int(i))
        if len(chosen) >= bank_size:
            break

    rates = []
    ref_clips: list[RefClip] = []
    for rank, i in enumerate(chosen):
        c = clips[i]
        text = transcript.text_between(c.start, c.end)
        rate, energy_db, f0_med = _clip_prosody(clips16[i])
        rates.append(rate)
        fname = f"clip_{rank:02d}.wav.enc"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            save_wav(tmp_path, c.audio, sr)
            cipher.encrypt_file(tmp_path, pdir / fname)
        finally:
            tmp_path.unlink(missing_ok=True)
        ref_clips.append(RefClip(file=fname, start=round(c.start, 3), end=round(c.end, 3),
                                 duration=round(c.duration, 3), transcript=text,
                                 language=script_language(text, transcript.language),
                                 centroid_cosine=round(float(sims[i]), 4), rate_sps=round(rate, 2),
                                 energy_db=round(energy_db, 1), f0_median_hz=round(f0_med, 1), style_bucket=""))
    if rates:
        lo, hi = np.percentile(rates, 33), np.percentile(rates, 66)
        for rc in ref_clips:
            rc.style_bucket = "slow" if rc.rate_sps <= lo else "fast" if rc.rate_sps >= hi else "medium"

    # Encrypted per-clip embeddings for later engine artifacts / verification
    (pdir / "embeddings.npy.enc").write_bytes(cipher.encrypt_bytes(embs[chosen].astype(np.float32).tobytes()))

    pair = embs @ embs.T
    iu = np.triu_indices(len(embs), 1)
    prof = SpeakerProfile(
        speaker_id=profile_id, name=name, language=transcript.language, embedding_engine=encoder.name,
        voice_embedding_b64=base64.b64encode(centroid.astype(np.float32).tobytes()).decode(),
        speaking_rate={"words_per_second": prosody.speaking_rate_wps,
                       "articulation_rate_sps": prosody.articulation_rate_sps},
        pitch_statistics={"median_hz": prosody.f0_median_hz, "mean_hz": prosody.f0_mean_hz,
                          "std_semitones": prosody.f0_std_semitones, "range_semitones": prosody.f0_range_semitones},
        energy_profile={"rms_db": prosody.energy_rms_db, "std_db": prosody.energy_std_db},
        style_features={"label": prosody.style_label, "pause_count": prosody.pause_count,
                        "pause_mean_s": prosody.pause_mean_s, "pause_ratio": prosody.pause_ratio},
        quality_score=quality_score, clips=ref_clips,
        intra_speaker_cosine_mean=round(float(pair[iu].mean()), 4) if len(iu[0]) else 1.0,
        intra_speaker_cosine_std=round(float(pair[iu].std()), 4) if len(iu[0]) else 0.0)
    (pdir / "profile.json").write_text(json.dumps(prof.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return prof, wrapped


def load_profile(profiles_dir: Path, profile_id: str) -> SpeakerProfile:
    d = json.loads((profiles_dir / profile_id / "profile.json").read_text(encoding="utf-8"))
    d["clips"] = [RefClip(**c) for c in d["clips"]]
    return SpeakerProfile(**d)


class ReferenceOpener:
    """Context manager that decrypts one reference clip to a temp WAV for the TTS engine."""

    def __init__(self, profiles_dir: Path, profile: SpeakerProfile, clip: RefClip, key: bytes):
        self.path = profiles_dir / profile.speaker_id / clip.file
        self.cipher = ProfileCipher(key)
        self.tmp: Path | None = None

    def __enter__(self) -> Path:
        data = self.cipher.decrypt_bytes(self.path.read_bytes())
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(data)
            self.tmp = Path(tmp.name)
        return self.tmp

    def __exit__(self, *exc) -> None:
        if self.tmp:
            self.tmp.unlink(missing_ok=True)


def similarity_to_profile(profile: SpeakerProfile, encoder: SpeakerEncoder, audio16k: np.ndarray) -> float:
    return cosine(profile.voice_embedding, encoder.embed(audio16k))
