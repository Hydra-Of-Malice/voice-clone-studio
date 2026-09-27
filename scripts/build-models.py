"""Assembles build\\models: every model the app needs, laid out so the installed copy runs offline.

Run with the development venv after the app has been used once (the models are then in the local
caches). The Hinglish ASR model is stored in half precision (3 GB instead of 6 GB); it is loaded in
half precision on the GPU anyway.

    .venv\\Scripts\\python scripts\\build-models.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "models"
HOME = Path.home()
DEV_MODELS = Path(os.environ.get("VC_DATA_DIR", str(HOME / ".voice-clone"))) / "models"
HF_HUB = Path(os.environ.get("HF_HOME", str(HOME / ".cache" / "huggingface"))) / "hub"


def copy_tree(src: Path, dst: Path, ignore=None) -> None:
    if not src.exists():
        sys.exit(f"missing: {src} (use the app once so the model is downloaded)")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore, symlinks=False)
    size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 2 ** 30
    print(f"  {size:6.2f} GB  {dst.relative_to(OUT)}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    skip_locks = shutil.ignore_patterns(".locks", "*.lock", "*.incomplete")
    print("== Chatterbox (TTS)")
    copy_tree(HF_HUB / "models--ResembleAI--chatterbox", OUT / "hf" / "hub" / "models--ResembleAI--chatterbox", skip_locks)
    print("== Whisper turbo (general ASR)")
    name = "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo"
    copy_tree(DEV_MODELS / "asr" / name, OUT / "asr" / name, skip_locks)
    print("== Speaker encoder")
    copy_tree(DEV_MODELS / "speaker", OUT / "speaker")
    print("== Denoiser")
    copy_tree(Path(os.environ["LOCALAPPDATA"]) / "DeepFilterNet" / "DeepFilterNet" / "Cache" / "DeepFilterNet3",
              OUT / "deepfilternet" / "DeepFilterNet3", shutil.ignore_patterns("*.log"))
    print("== Watermark")
    copy_tree(HOME / ".cache" / "audioseal", OUT / "audioseal")
    print("== Tokenizer data")
    # the .zip must ship too: the library re-downloads the data when the archive is missing
    copy_tree(HOME / ".pkuseg", OUT / "pkuseg", shutil.ignore_patterns("temp"))

    print("== Hinglish ASR (half precision)")
    dst = OUT / "asr" / "whisper-hinglish-preview"
    if not (dst / "model.safetensors").exists():
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        src = DEV_MODELS / "asr" / "whisper-hinglish-preview"
        if dst.exists():
            shutil.rmtree(dst)
        model = WhisperForConditionalGeneration.from_pretrained(str(src), torch_dtype=torch.float16)
        model.save_pretrained(str(dst), safe_serialization=True, max_shard_size="5GB")
        WhisperProcessor.from_pretrained(str(src)).save_pretrained(str(dst))
        for extra in ("README.md",):
            if (src / extra).exists():
                shutil.copy2(src / extra, dst / extra)
    size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 2 ** 30
    print(f"  {size:6.2f} GB  {dst.relative_to(OUT)}")

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 2 ** 30
    (OUT / "bundle.json").write_text(json.dumps({"built": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                 "size_gb": round(total, 2)}, indent=1), encoding="utf-8")
    print(f"== models ready: {OUT} ({total:.2f} GB)")


if __name__ == "__main__":
    main()
