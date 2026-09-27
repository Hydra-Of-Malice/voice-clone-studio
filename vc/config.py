"""Central settings. Everything is overridable through VC_* environment variables so the same
code runs as a local desktop server and as a cloud worker."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"VC_{name}", default)


def _opt(name: str) -> str | None:
    return os.environ.get(f"VC_{name}") or None


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(Path.home() / ".voice-clone"))))
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(_env("PORT", "8765")))
    device: str = field(default_factory=lambda: _env("DEVICE", "cuda"))
    # "spawn": the API starts and supervises a worker process; "external": a worker is run separately
    worker_mode: str = field(default_factory=lambda: _env("WORKER_MODE", "spawn"))

    # Module selection (registry keys). Swap engines here without touching the pipeline.
    asr_engine: str = field(default_factory=lambda: _env("ASR", "auto"))
    asr_model: str = field(default_factory=lambda: _env("ASR_MODEL", "large-v3-turbo"))
    asr_hindi_model: str | None = field(default_factory=lambda: _opt("ASR_HINDI_MODEL"))
    speaker_engine: str = field(default_factory=lambda: _env("SPEAKER", "ecapa"))
    tts_engine: str = field(default_factory=lambda: _env("TTS", "chatterbox-multilingual"))

    # Sample requirements
    min_sample_seconds: float = 60.0          # hard reject below this
    recommended_min_seconds: float = 300.0    # 5 minutes
    recommended_max_seconds: float = 600.0    # 10 minutes
    max_upload_seconds: float = 1200.0
    min_quality_score: int = 55               # below this we ask for a better sample
    enhance_below_snr_db: float = 20.0        # denoise only noisy uploads

    # Reference bank
    ref_clip_min_s: float = 6.0
    ref_clip_max_s: float = 12.0
    ref_bank_size: int = 12

    # Speaker verification thresholds (ECAPA cosine; calibrate per encoder)
    same_speaker_cosine: float = 0.45
    consent_min_text_match: float = 0.6

    # Output
    output_sr: int = 48000
    target_lufs: float = -16.0
    true_peak_dbtp: float = -1.0
    mp3_bitrate: int = 320

    # Provenance
    audioseal: bool = field(default_factory=lambda: _env("AUDIOSEAL", "1") == "1")
    c2pa: bool = field(default_factory=lambda: _env("C2PA", "1") == "1")
    c2pa_cert: str | None = field(default_factory=lambda: _opt("C2PA_CERT"))
    c2pa_key: str | None = field(default_factory=lambda: _opt("C2PA_KEY"))
    c2pa_timestamp_url: str | None = field(default_factory=lambda: _opt("C2PA_TSA"))

    # Fine-tuning (LoRA on the TTS text-to-token transformer)
    ft_rank: int = 16
    ft_alpha: int = 32
    ft_lr: float = 1e-4
    ft_epochs: int = 12
    ft_max_steps: int = 600
    ft_min_clip_confidence: float = 0.8
    ft_min_similarity_gain: float = 0.01      # adapter must beat zero-shot by this much ...
    ft_max_cer_regression: float = 0.03       # ... without losing intelligibility

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def profiles_dir(self) -> Path:
        return self.data_dir / "profiles"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def models_dir(self) -> Path:
        override = os.environ.get("VC_MODELS_DIR")
        return Path(override) if override else self.data_dir / "models"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "voice-clone.db"

    def ensure_dirs(self) -> None:
        for d in (self.uploads_dir, self.profiles_dir, self.outputs_dir):
            d.mkdir(parents=True, exist_ok=True)
        if not os.environ.get("VC_MODELS_DIR"):
            self.models_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
