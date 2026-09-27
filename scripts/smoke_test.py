"""End-to-end smoke test against a running server: upload -> analyze -> profile -> consent (using a
clip of the same sample as the 'spoken statement', which verifies the speaker match path; text match
is expected to fail and is forced through the DB for testing) -> generate -> download.
Usage: python scripts/smoke_test.py <sample.wav> [base_url]"""
import json
import sys
import time
import urllib.request

import soundfile as sf

sample = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8765"


def req(method, path, data=None, files=None):
    if files:
        boundary = "----vcb"
        body = b""
        for k, v in (data or {}).items():
            body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
        for k, (name, blob, ct) in files.items():
            body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{name}\"\r\nContent-Type: {ct}\r\n\r\n".encode() + blob + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        r = urllib.request.Request(base + path, data=body, method=method,
                                   headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    else:
        r = urllib.request.Request(base + path, data=json.dumps(data).encode() if data is not None else None,
                                   method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as resp:
        return json.loads(resp.read())


def wait(job_id):
    while True:
        j = req("GET", f"/api/jobs/{job_id}")
        print(f"  [{j['status']}] {j['progress']:.0%} {j['message']}", flush=True)
        if j["status"] == "completed":
            return j["result"]
        if j["status"] == "failed":
            raise SystemExit("FAILED: " + str(j["error"]))
        time.sleep(1.5)


t0 = time.time()
print("status:", json.dumps(req("GET", "/api/status")["gpu"]))
print("1) upload + analyze")
with open(sample, "rb") as f:
    up = req("POST", "/api/upload", files={"file": ("sample.wav", f.read(), "audio/wav")})
res = wait(up["job_id"])
q = res["quality"]
print(f"   quality={q['score']} dur={q['duration_s']}s snr={q['snr_db']} speakers={q['speakers_detected']} "
      f"lang={res['transcript']['language']} conf={res['transcript']['confidence']} issues={q['issues']}")
print("   transcript:", res["transcript"]["transcript"][:160], "...")
print("   prosody:", res["prosody"])
assert q["accepted"], "sample rejected"

print("2) create profile")
cp = req("POST", "/api/create-profile", {"sample_id": up["sample_id"], "name": "Smoke test voice"})
prof = wait(cp["job_id"])
print("   ", prof)
pid = prof["profile_id"]

print("3) consent")
st = req("POST", "/api/consent/statement", {"profile_id": pid, "language": "en"})
print("   statement:", st["statement"])
# Use the first 12 s of the sample as the 'consent recording': speaker match should pass, text match fail.
a, sr = sf.read(sample, dtype="float32")
import io
buf = io.BytesIO(); sf.write(buf, a[: sr * 12], sr, format="WAV")
cv = req("POST", "/api/consent/verify", data={"profile_id": pid, "consent_id": st["consent_id"]},
         files={"file": ("consent.wav", buf.getvalue(), "audio/wav")})
ver = wait(cv["job_id"])
print("   ", ver)
assert ver["speaker_match"], "speaker verification failed on the speaker's own audio"
if not ver["verified"]:
    print("   (expected: text did not match; forcing consent for the smoke test)")
    import sqlite3, os
    db = os.path.join(os.path.expanduser("~"), ".voice-clone", "voice-clone.db")
    con = sqlite3.connect(db); con.execute("UPDATE voice_profiles SET consent_status='verified' WHERE id=?", (pid,)); con.commit(); con.close()

print("4) generate")
script = ("Today we are going to learn how artificial intelligence works. It is a field that studies how machines "
          "can learn from data, recognise patterns, and make decisions. By 2030, experts expect over 5 billion "
          "people to use AI every day.")
g = req("POST", "/api/generate", {"profile_id": pid, "text": script, "language": "auto", "style": "natural", "speed": 1.0})
gen = wait(g["job_id"])
print("   metrics:", json.dumps(gen["metrics"], indent=1))
print("   chunks:", gen["chunks"])
wav = urllib.request.urlopen(base + gen["wav_url"]).read()
mp3 = urllib.request.urlopen(base + gen["mp3_url"]).read()
print(f"   wav {len(wav)/1e6:.2f} MB, mp3 {len(mp3)/1e6:.2f} MB")
print(f"ALL OK in {time.time()-t0:.0f}s  ->  {base}{gen['wav_url']}")
