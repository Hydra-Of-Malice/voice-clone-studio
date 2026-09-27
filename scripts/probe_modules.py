"""Developer probe for the individual modules (no server needed).
Usage: python scripts/probe_modules.py translit|denoise|provenance|asr"""
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
what = sys.argv[1]
SAMPLE = Path(__file__).resolve().parent.parent / "testdata" / "sample_1272.wav"
OUT = Path(os.path.expanduser("~/.voice-clone/outputs"))

if what == "translit":
    from vc.text.hinglish import romanized_to_mixed
    from vc.text.normalize import prepare_script
    for s in ["Mujhe samjhaunga ki yeh technology kaise kaam karti hai.",
              "Rahul ne bola ki barish mein padhai karna mushkil hai, lekin zindagi khubsurat hai.",
              "Kal hum office jayenge aur meeting mein project ka status discuss karenge.",
              "Tumhari awaaz bahut sundar hai, ekdum pyaari.",
              "Dosto, aaj ki video mein hum seekhenge ki machine learning kya hoti hai.",
              "Please send me the report by Monday, we need to finalise the budget."]:
        print(s, "\n  ->", romanized_to_mixed(s)[0])
    p = prepare_script("Is mahine 25000 rupaye kharch hue, yaani 12.5% zyada.", "auto")
    print(p["language"], "|", p["text"], "|", p["transliteration"])

elif what == "denoise":
    from vc.audio.enhance import denoise
    from vc.audio.io import load_audio, resample, save_wav
    from vc.audio.quality import estimate_snr_db
    from vc.audio.vad import speech_spans
    a, sr = load_audio(SAMPLE)
    a = a[: sr * 60]
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(len(a)).astype(np.float32)
    noise = np.convolve(noise, np.ones(8) / 8, mode="same")            # coloured noise
    sp = np.sqrt(np.mean(a ** 2)); npow = np.sqrt(np.mean(noise ** 2))
    noisy = (a + noise * (sp / npow) * 10 ** (-10 / 20)).astype(np.float32)   # 10 dB SNR
    t = time.time(); den = denoise(noisy, sr); dt_ = time.time() - t
    for name, x in [("clean", a), ("noisy", noisy), ("denoised", den)]:
        x16 = resample(x, sr, 16000)
        print(f"{name:9s} SNR {estimate_snr_db(x16, speech_spans(x16)):5.1f} dB")
    print(f"denoise time {dt_:.1f}s for 60 s audio")
    from vc.speaker.ecapa import EcapaEncoder
    from vc.speaker.base import cosine
    enc = EcapaEncoder(device="cpu", savedir=os.path.expanduser("~/.voice-clone/models/speaker"))
    e = lambda x: enc.embed(resample(x, sr, 16000)[: 16000 * 20])
    print("identity cosine clean-vs-noisy %.3f  clean-vs-denoised %.3f" % (cosine(e(a), e(noisy)), cosine(e(a), e(den))))
    save_wav(SAMPLE.with_name("sample_noisy.wav"), np.concatenate([noisy] * 6), sr)

elif what == "provenance":
    from vc import provenance as pv
    from vc.audio.io import load_audio, save_mp3, save_wav
    src = sorted(OUT.glob("*.wav"))[-1]
    a, sr = load_audio(src)
    print("source", src.name, sr, "Hz", round(len(a) / sr, 1), "s")
    print("before:", pv.detect_audioseal(a, sr), "| perth:", pv.detect_perth(a, sr))
    payload = pv.payload_for("au_test_123")
    t = time.time(); w = pv.embed_audioseal(a, sr, payload); print("embed %.2fs payload %d" % (time.time() - t, payload))
    print("SNR of watermark: %.1f dB" % (10 * np.log10(np.sum(a ** 2) / np.sum((w - a) ** 2))))
    print("after :", pv.detect_audioseal(w, sr), "| perth:", pv.detect_perth(w, sr))
    d = Path(tempfile.mkdtemp())
    wav, mp3 = d / "t.wav", d / "t.mp3"
    save_wav(wav, w, sr, subtype="PCM_24", comment="AI-generated"); save_mp3(mp3, w, sr, 320, comment="AI-generated")
    signer = pv.C2PASigner(Path(os.path.expanduser("~/.voice-clone")))
    for f in (wav, mp3):
        print(f.suffix, "sign:", signer.sign_file(f, "test", {"audio_id": "au_test_123", "audioseal_payload": payload}))
        b, sr2 = load_audio(f)
        print(f.suffix, "read:", pv.read_c2pa(f))
        print(f.suffix, "audioseal after export:", pv.detect_audioseal(b, sr2), "| perth:", pv.detect_perth(b, sr2))
    save_mp3(d / "low.mp3", w, sr, 64)
    b, sr2 = load_audio(d / "low.mp3")
    print("64 kbps mp3 audioseal:", pv.detect_audioseal(b, sr2))

elif what == "asr":
    from vc.asr.router import RoutedASR
    from vc.audio.io import load_audio
    from vc.audio.preprocess import merge_spans
    from vc.audio.vad import speech_spans
    import sqlite3, json
    root = os.path.expanduser("~/.voice-clone/models/asr")
    asr = RoutedASR(device="cuda", download_root=root)
    con = sqlite3.connect(os.path.expanduser("~/.voice-clone/voice-clone.db"))
    rows = con.execute("select g.wav_path, j.script from generated_audio g join generation_jobs j on j.id=g.job_id "
                       "order by g.created_at desc limit 4").fetchall()
    for path, script in rows:
        a, _ = load_audio(path, target_sr=16000)
        t = time.time()
        tr = asr.transcribe(a, chunks=merge_spans(speech_spans(a)))
        print(f"\nSCRIPT: {script}\nHEARD : {tr.transcript}\n  lang={tr.language} ({tr.language_confidence}) conf={tr.confidence:.2f} "
              f"engine={tr.engine} segs={len(tr.segments)} {time.time() - t:.1f}s")
