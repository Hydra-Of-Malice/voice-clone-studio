"""Diagnostics: `python -m vc.diagnose` (Start menu: "Voice Clone Studio diagnostics").

Checks the graphics card, the driver, the GPU runtime and the bundled models, runs a short
computation on the GPU, and writes the report to <data>\\logs\\diagnostics.txt so it can be sent
along when something does not work."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from vc import BUNDLED_MODELS, __version__
from vc.config import settings

LINES: list[str] = []


def out(text: str = "") -> None:
    LINES.append(text)
    print(text, flush=True)


def check(name: str, fn) -> bool:
    try:
        result = fn()
        out(f"[ OK ] {name}: {result}")
        return True
    except Exception as e:  # noqa: BLE001
        out(f"[FAIL] {name}: {type(e).__name__}: {e}")
        out("       " + traceback.format_exc().strip().splitlines()[-1])
        return False


def nvidia_smi() -> str:
    smi = shutil.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"
    r = subprocess.run([smi, "--query-gpu=name,memory.total,memory.used,compute_cap,driver_version",
                        "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "nvidia-smi failed")
    return r.stdout.strip()


def torch_info() -> str:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"torch {torch.__version__} cannot use the GPU (CUDA {torch.version.cuda})")
    cap = torch.cuda.get_device_capability(0)
    arch = f"sm_{cap[0]}{cap[1]}"
    supported = torch.cuda.get_arch_list()
    # A build for sm_86 also runs on sm_89: same generation (major), equal or older revision (minor)
    builds = [(int(a[3:-1]), int(a[-1])) for a in supported if a.startswith("sm_")]
    if not any(major == cap[0] and minor <= cap[1] for major, minor in builds):
        raise RuntimeError(f"this graphics card ({arch}) is not supported by the GPU runtime {supported}")
    return (f"torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}, "
            f"{arch} supported")


def gpu_compute() -> str:
    import torch
    t0 = time.time()
    x = torch.randn(2048, 2048, device="cuda")
    y = (x @ x).float().mean().item()
    h = torch.randn(1, 1, 16000, device="cuda", dtype=torch.float16)
    conv = torch.nn.Conv1d(1, 8, 9).cuda().half()
    z = conv(h).float().abs().mean().item()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (f"matrix and convolution ran in {time.time() - t0:.2f} s (values {y:.3f}, {z:.3f}); "
            f"{free / 2 ** 30:.1f} of {total / 2 ** 30:.1f} GB free")


def models() -> str:
    root = settings.models_dir
    needed = ["asr/whisper-hinglish-preview/model.safetensors", "speaker/embedding_model.ckpt",
              "deepfilternet/DeepFilterNet3/config.ini", "pkuseg/spacy_ontonotes.zip"]
    missing = [n for n in needed if not (root / n).exists()]
    if missing and (BUNDLED_MODELS / "bundle.json").exists():
        raise RuntimeError(f"missing in {root}: {missing}")
    size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 2 ** 30 if root.exists() else 0
    return f"{root} ({size:.1f} GB){' - models download on first use' if missing else ''}"


def asr_engine() -> str:
    import numpy as np
    from faster_whisper import WhisperModel
    m = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16",
                     download_root=str(settings.models_dir / "asr"))
    t0 = time.time()
    segs, info = m.transcribe(np.zeros(16000 * 3, dtype=np.float32), language="en")
    list(segs)
    return f"speech recognition ran on the GPU in {time.time() - t0:.1f} s"


def main() -> int:
    out(f"Voice Clone Studio {__version__} diagnostics - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    out(f"Windows {platform.version()}, Python {sys.version.split()[0]}, install {Path(__file__).resolve().parent.parent}")
    out()
    results = [check("Graphics card", nvidia_smi), check("GPU runtime", torch_info),
               check("GPU computation", gpu_compute), check("Models", models),
               check("Speech recognition engine", asr_engine)]
    out()
    ok = all(results)
    out("RESULT: everything needed to generate speech is working." if ok else
        "RESULT: something is not working. Please send this file to the person who gave you the app.")
    try:
        path = settings.data_dir / "logs" / "diagnostics.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(LINES) + "\n", encoding="utf-8")
        out(f"\nSaved to {path}")
    except OSError as e:
        out(f"could not save the report: {e}")
    if os.environ.get("VC_DIAGNOSE_PAUSE", "1") == "1" and sys.stdin and sys.stdin.isatty():
        input("\nPress Enter to close this window.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
