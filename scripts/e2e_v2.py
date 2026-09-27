"""End-to-end test of the v0.2 features against a running server.

Stages (pick with argv[1], default 'all'):
  noisy     upload a 10 dB SNR sample  -> denoiser must engage, profile is built
  consent   a synthetic reading of the consent statement must be REJECTED (watermark), wrong speaker too
  generate  generation carries AudioSeal + C2PA; /api/verify-audio recognises the file
  hindi     a Hindi/Hinglish recording is routed to the Hinglish ASR (mixed-script transcript)
  preview   text preview / transliteration endpoint
  finetune  LoRA fine-tune with the acceptance gate

Consent cannot be spoken by a script, so after the rejection checks the profile is marked verified
directly in the database (test only)."""
import io
import json
import os
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")
BASE = "http://127.0.0.1:8765"
ROOT = Path(__file__).resolve().parent.parent
DB = os.path.join(os.environ.get("VC_DATA_DIR") or os.path.join(os.path.expanduser("~"), ".voice-clone"),
                  "voice-clone.db")
STATE = ROOT / "testdata" / "e2e_state.json"
stage = sys.argv[1] if len(sys.argv) > 1 else "all"


def req(method, path, data=None, files=None):
    if files:
        b = "----vcb"
        body = b""
        for k, v in (data or {}).items():
            body += f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
        for k, (name, blob) in files.items():
            body += (f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{name}\"\r\n"
                     f"Content-Type: application/octet-stream\r\n\r\n").encode() + blob + b"\r\n"
        body += f"--{b}--\r\n".encode()
        r = urllib.request.Request(BASE + path, data=body, method=method,
                                   headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    else:
        r = urllib.request.Request(BASE + path, data=json.dumps(data).encode() if data is not None else None,
                                   method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=900) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} on {path}: {e.read().decode()[:300]}")


def wait(job_id, quiet=False, allow_fail=False):
    last = None
    while True:
        j = req("GET", f"/api/jobs/{job_id}")
        if not quiet and j["message"] != last:
            print(f"    [{j['status']}] {j['progress']:.0%} {j['message']}", flush=True)
            last = j["message"]
        if j["status"] == "completed":
            return j["result"]
        if j["status"] == "failed":
            if allow_fail:
                return {"failed": j["error"]}
            raise SystemExit("JOB FAILED: " + str(j["error"]))
        time.sleep(1.0)


def force_consent(pid):
    con = sqlite3.connect(DB)
    con.execute("UPDATE voice_profiles SET consent_status='verified' WHERE id=?", (pid,))
    con.commit()
    con.close()


def generate(pid, text, lang="auto", style="natural", fine_tuned=True):
    g = req("POST", "/api/generate", {"profile_id": pid, "text": text, "language": lang, "style": style,
                                       "speed": 1.0, "use_fine_tuned": fine_tuned})
    return wait(g["job_id"], quiet=True)


def audio_bytes(url):
    return urllib.request.urlopen(BASE + url).read()


state = json.loads(STATE.read_text()) if STATE.exists() else {}
run = lambda s: stage in ("all", s)
t00 = time.time()
st = req("GET", "/api/status")
print("worker:", {k: st["worker"].get(k) for k in ("alive", "pid")}, "gpu:", st["gpu"], "mode:", st["worker_mode"])

if run("noisy"):
    print("\n== NOISY UPLOAD (10 dB SNR) ==")
    up = req("POST", "/api/upload", files={"file": ("noisy.wav", (ROOT / "testdata" / "sample_noisy.wav").read_bytes())})
    res = wait(up["job_id"])
    q, qb = res["quality"], res["quality_before_enhancement"]
    print("  enhancement:", res["enhancement"])
    print(f"  quality before: {qb and qb['score']} (SNR {qb and qb['snr_db']})  after: {q['score']} (SNR {q['snr_db']}, noise {q['noise_level']})")
    print(f"  accepted={q['accepted']} issues={q['issues']}")
    print(f"  language={res['transcript']['language']} engine={res['transcript']['engine']} conf={res['transcript']['confidence']}")
    print("  transcript:", res["transcript"]["transcript"][:150], "...")
    assert res["enhancement"]["applied"], "denoiser did not engage"
    assert q["accepted"]
    prof = wait(req("POST", "/api/create-profile", {"sample_id": up["sample_id"], "name": "E2E v2 voice"})["job_id"], quiet=True)
    print("  profile:", prof["profile_id"], "clips", prof["clips"])
    state.update(profile_id=prof["profile_id"], sample_id=up["sample_id"])
    STATE.write_text(json.dumps(state))

pid = state.get("profile_id")

if run("consent"):
    print("\n== CONSENT CHECKS ==")
    old = [p for p in req("GET", "/api/profiles") if p["consent_status"] == "verified" and p["id"] != pid]
    stt = req("POST", "/api/consent/statement", {"profile_id": pid, "language": "en"})
    print("  statement:", stt["statement"])
    if old:
        # The same speaker's *cloned* voice reads the statement perfectly: must be refused as synthetic
        g = generate(old[0]["id"], stt["statement"], "en")
        r = wait(req("POST", "/api/consent/verify", data={"profile_id": pid, "consent_id": stt["consent_id"]},
                     files={"file": ("c.wav", audio_bytes(g["wav_url"]))})["job_id"], quiet=True)
        print(f"  cloned reading -> verified={r['verified']} text={r['text_match']} nonce={r['nonce_ok']} "
              f"speaker={r['speaker_match']} synthetic={r['synthetic_marks']}")
        assert not r["verified"] and r["synthetic_marks"], "synthetic consent was not rejected"
        assert r["text_match"] > 0.8 and r["nonce_ok"], "text matching failed on a perfect reading"
    a, sr = sf.read(ROOT / "testdata" / "sample_1462.wav", dtype="float32")
    buf = io.BytesIO(); sf.write(buf, a[: sr * 12], sr, format="WAV")
    stt2 = req("POST", "/api/consent/statement", {"profile_id": pid, "language": "en"})
    r = wait(req("POST", "/api/consent/verify", data={"profile_id": pid, "consent_id": stt2["consent_id"]},
                 files={"file": ("c.wav", buf.getvalue())})["job_id"], quiet=True)
    print(f"  other speaker  -> verified={r['verified']} speaker={r['speaker_match']} text={r['text_match']}")
    assert not r["verified"] and not r["speaker_match"]
    try:
        urllib.request.urlopen(urllib.request.Request(BASE + "/api/generate", method="POST",
                               data=json.dumps({"profile_id": pid, "text": "hello"}).encode(),
                               headers={"Content-Type": "application/json"}))
        raise SystemExit("generation was allowed without consent")
    except urllib.error.HTTPError as e:
        print("  generate without consent -> HTTP", e.code)
        assert e.code == 403
    force_consent(pid)
    print("  (consent forced in the database for the remaining tests)")

if run("preview"):
    print("\n== TEXT PREVIEW ==")
    for t in ["Kal hum office jayenge aur Rahul se milenge, barish ho ya na ho.",
              "इस साल 12.5% growth हुई और revenue ₹25,000 करोड़ रहा।",
              "Dr. Smith will present the API on 3 March."]:
        p = req("POST", "/api/text/preview", {"text": t, "language": "auto"})
        print(f"  [{p['language']}] {p['text']}   ({p['transliteration']})")

if run("generate"):
    print("\n== GENERATE + PROVENANCE ==")
    g = generate(pid, "आज हम voice cloning के बारे में बात करेंगे। This sentence is in English.", "auto")
    m = g["metrics"]
    print(f"  similarity={m['speaker_similarity']} (band >= {m['similarity_band_low']}) lufs={m['loudness_lufs']} provenance={m['provenance']}")
    state["audio_id"] = g["audio_id"]
    STATE.write_text(json.dumps(state))
    for fmt, url in (("wav", g["wav_url"]), ("mp3", g["mp3_url"])):
        v = wait(req("POST", "/api/verify-audio", files={"file": (f"x.{fmt}", audio_bytes(url))})["job_id"], quiet=True)
        print(f"  verify {fmt}: {v['verdict']} | match={v['matched_output'] and v['matched_output']['audio_id']} "
              f"audioseal={v['audioseal']['detected']} perth={v['perth'].get('detected')} "
              f"c2pa={v['c2pa'].get('present')} state={v['c2pa'].get('validation_state')} issuer={v['c2pa'].get('issuer')}")
        assert v["matched_output"] and v["matched_output"]["audio_id"] == g["audio_id"]
        assert v["c2pa"]["present"] and v["c2pa"]["ai_generated"]
    a, sr = sf.read(ROOT / "testdata" / "sample_1272.wav", dtype="float32")
    buf = io.BytesIO(); sf.write(buf, a[: sr * 20], sr, format="WAV")
    v = wait(req("POST", "/api/verify-audio", files={"file": ("real.wav", buf.getvalue())})["job_id"], quiet=True)
    print(f"  verify real recording: {v['verdict']} (ai_generated={v['ai_generated']})")
    assert not v["ai_generated"]

if run("hindi"):
    print("\n== HINDI / HINGLISH ASR ROUTING ==")
    paras = ["नमस्ते दोस्तों, आज हम artificial intelligence के बारे में बात करेंगे। यह technology हमारी ज़िंदगी को बदल रही है।",
             "सबसे पहले हम समझेंगे कि machine learning क्या होती है और यह कैसे काम करती है। Data बहुत ज़रूरी होता है।",
             "मेरा मानना है कि आने वाले सालों में हर company को इस तकनीक का इस्तेमाल करना पड़ेगा।",
             "अगर आपको यह video पसंद आया हो तो please like और share ज़रूर कीजिए। धन्यवाद।"]
    import numpy as np
    parts = []
    for ptxt in paras * 3:
        g = generate(pid, ptxt, "hi")
        a, sr = sf.read(io.BytesIO(audio_bytes(g["wav_url"])), dtype="float32")
        parts += [a, np.zeros(int(0.6 * sr), dtype=np.float32)]
    buf = io.BytesIO(); sf.write(buf, np.concatenate(parts), sr, format="WAV")
    print(f"  built a {sum(len(p) for p in parts) / sr:.0f} s Hindi recording")
    res = wait(req("POST", "/api/upload", files={"file": ("hindi.wav", buf.getvalue())})["job_id"], quiet=True)
    t = res["transcript"]
    print(f"  language={t['language']} ({t['language_confidence']}) engine={t['engine']} conf={t['confidence']}")
    print("  transcript:", t["transcript"][:260])
    print("  synthetic marks:", res["synthetic_marks"], "accepted:", res["quality"]["accepted"])
    assert "hinglish" in t["engine"] and t["language"] == "hi"
    assert res["synthetic_marks"] and not res["quality"]["accepted"], "cloned audio was accepted as a voice sample"

if run("finetune"):
    print("\n== FINE-TUNE (LoRA) ==")
    f = req("POST", "/api/fine-tune", {"profile_id": pid})
    r = wait(f["job_id"], allow_fail=True)
    print(json.dumps(r, indent=1, ensure_ascii=False))
    if not r.get("failed"):
        g = generate(pid, "Today we are going to learn how artificial intelligence works.", "en")
        print("  generation after fine-tune: fine_tuned =", g["metrics"]["fine_tuned"], "similarity", g["metrics"]["speaker_similarity"])

print(f"\nDONE ({stage}) in {time.time() - t00:.0f}s")
