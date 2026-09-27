"""Build a ~6 minute single-speaker test sample from LibriSpeech dev-clean (one speaker, one chapter).
Usage: python scripts/make_test_sample.py <librispeech_root> <speaker_id> <out.wav>"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

root, spk, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
files = sorted((root / spk).rglob("*.flac"))
parts, total = [], 0.0
for f in files:
    a, sr = sf.read(f, dtype="float32")
    parts.append(a)
    parts.append(np.zeros(int(0.5 * sr), dtype=np.float32))   # natural-ish pause between utterances
    total += len(a) / sr + 0.5
    if total >= 360:
        break
audio = np.concatenate(parts)
out.parent.mkdir(parents=True, exist_ok=True)
sf.write(out, audio, sr)
print(f"wrote {out} ({total/60:.1f} min, {len(files)} files available, sr={sr})")
