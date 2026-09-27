"""Generate with an existing (consent-verified) profile. Usage: python scripts/gen_test.py [profile_id] [language] [style]"""
import json
import sys
import time
import urllib.request

base = "http://127.0.0.1:8765"
J = lambda p, d=None: json.loads(urllib.request.urlopen(urllib.request.Request(
    base + p, data=json.dumps(d).encode() if d is not None else None,
    headers={"Content-Type": "application/json"}), timeout=600).read())

pid = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else J("/api/profiles")[0]["id"]
lang = sys.argv[2] if len(sys.argv) > 2 else "auto"
style = sys.argv[3] if len(sys.argv) > 3 else "natural"
texts = {
    "en": ("Today we are going to learn how artificial intelligence works. It is a field that studies how machines "
           "can learn from data, recognise patterns, and make decisions. By 2030, experts expect over 5 billion "
           "people to use AI every day."),
    "hi": "आज हम artificial intelligence के बारे में discuss करेंगे। यह तकनीक 2030 तक 5 अरब लोगों की ज़िंदगी बदल देगी।",
    "hi-Latn": "Aaj hum artificial intelligence ke baare mein discuss karenge. Yeh bahut interesting topic hai.",
}
text = texts.get(lang, texts["en"])
t0 = time.time()
g = J("/api/generate", {"profile_id": pid, "text": text, "language": lang, "style": style, "speed": 1.0})
while True:
    j = J(f"/api/jobs/{g['job_id']}")
    print(f"  [{j['status']}] {j['progress']:.0%} {j['message']}", flush=True)
    if j["status"] == "completed":
        break
    if j["status"] == "failed":
        raise SystemExit("FAILED: " + str(j["error"]))
    time.sleep(2)
r = j["result"]
print(json.dumps(r["metrics"], indent=1, ensure_ascii=False))
print("chunks:", r["chunks"])
print(f"done in {time.time()-t0:.0f}s -> {base}{r['wav_url']}")
