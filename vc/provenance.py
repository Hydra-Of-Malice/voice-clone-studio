"""Provenance for generated audio: an inaudible AudioSeal watermark carrying a 16-bit payload, a
C2PA Content Credentials manifest signed into the exported WAV/MP3, and detection of both (plus
Resemble's Perth watermark that Chatterbox embeds).

Metadata is stripped by re-encoding and watermarks do not survive neural codecs, so the two layers
are complementary; neither is a guarantee."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import shutil
import threading
from pathlib import Path

import numpy as np

from vc import __version__
from vc.audio.io import resample

log = logging.getLogger("vc.provenance")
_lock = threading.Lock()
_gen = None
_det = None

AI_SOURCE_TYPE = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"


# ------------------------------------------------------------------ AudioSeal ----------------
def payload_for(audio_id: str) -> int:
    """Stable 16-bit payload derived from the output id (stored in the DB for look-up)."""
    return int.from_bytes(hashlib.sha256(audio_id.encode()).digest()[:2], "big")


def _bits(payload: int):
    import torch
    return torch.tensor([[(payload >> (15 - i)) & 1 for i in range(16)]], dtype=torch.int32)


def _generator():
    global _gen
    with _lock:
        if _gen is None:
            from audioseal import AudioSeal
            _gen = AudioSeal.load_generator("audioseal_wm_16bits").eval()
        return _gen


def _detector():
    global _det
    with _lock:
        if _det is None:
            from audioseal import AudioSeal
            _det = AudioSeal.load_detector("audioseal_detector_16bits").eval()
        return _det


def embed_audioseal(audio: np.ndarray, sr: int, payload: int, strength: float = 1.0) -> np.ndarray:
    """Adds the watermark computed at 16 kHz and upsampled to `sr` (CPU, ~real-time x50)."""
    import torch

    x16 = torch.from_numpy(resample(audio, sr, 16000)).float().view(1, 1, -1)
    with torch.inference_mode():
        wm16 = _generator().get_watermark(x16, 16000, message=_bits(payload)).view(-1).numpy()
    wm = resample(wm16, 16000, sr)
    n = min(len(wm), len(audio))
    out = audio.copy()
    out[:n] += strength * wm[:n]
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def detect_audioseal(audio: np.ndarray, sr: int) -> dict:
    import torch

    x16 = torch.from_numpy(resample(audio, sr, 16000)).float().view(1, 1, -1)
    with torch.inference_mode():
        prob, msg = _detector().detect_watermark(x16, 16000)
    bits = [int(b) for b in msg.view(-1).round().int().tolist()]
    payload = 0
    for b in bits:
        payload = (payload << 1) | b
    return {"probability": round(float(prob), 4), "detected": float(prob) >= 0.5, "payload": payload}


def detect_perth(audio: np.ndarray, sr: int) -> dict:
    try:
        import perth
        wm = perth.PerthImplicitWatermarker()
        score = float(wm.get_watermark(audio, sample_rate=sr))
        return {"confidence": round(score, 3), "detected": score >= 0.5}
    except Exception as e:  # noqa: BLE001
        return {"detected": False, "error": f"{type(e).__name__}: {e}"}


# ------------------------------------------------------------------ C2PA ---------------------
class C2PASigner:
    """Signs exported files with Content Credentials. Uses the certificate chain + key in
    `<data_dir>/c2pa/` (VC_C2PA_CERT / VC_C2PA_KEY to override). When none exists, a local
    development CA is generated: manifests are then cryptographically valid but validators report
    the issuer as untrusted until a certificate from a C2PA trust-list CA is installed."""

    def __init__(self, data_dir: Path, cert_path: str | None = None, key_path: str | None = None,
                 ta_url: str | None = None):
        self.dir = data_dir / "c2pa"
        self.cert_path = Path(cert_path) if cert_path else self.dir / "chain.pem"
        self.key_path = Path(key_path) if key_path else self.dir / "signing.key"
        self.ta_url = ta_url
        self.development = not (cert_path and key_path)
        if not (self.cert_path.exists() and self.key_path.exists()):
            self._make_dev_credentials()

    def _make_dev_credentials(self) -> None:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

        self.dir.mkdir(parents=True, exist_ok=True)
        now = dt.datetime.now(dt.timezone.utc)

        def name(cn: str) -> x509.Name:
            return x509.Name([x509.NameAttribute(NameOID.COUNTRY_NAME, "IN"),
                              x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Voice Clone Studio (development)"),
                              x509.NameAttribute(NameOID.COMMON_NAME, cn)])

        ca_key = ec.generate_private_key(ec.SECP256R1())
        ca = (x509.CertificateBuilder().subject_name(name("Voice Clone Studio Dev Root")).issuer_name(name("Voice Clone Studio Dev Root"))
              .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
              .not_valid_before(now - dt.timedelta(days=1)).not_valid_after(now + dt.timedelta(days=3650))
              .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
              .add_extension(x509.KeyUsage(False, False, False, False, False, True, True, False, False), critical=True)
              .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
              .sign(ca_key, hashes.SHA256()))
        key = ec.generate_private_key(ec.SECP256R1())
        leaf = (x509.CertificateBuilder().subject_name(name("Voice Clone Studio Signer")).issuer_name(ca.subject)
                .public_key(key.public_key()).serial_number(x509.random_serial_number())
                .not_valid_before(now - dt.timedelta(days=1)).not_valid_after(now + dt.timedelta(days=825))
                .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                .add_extension(x509.KeyUsage(True, False, False, False, False, False, False, False, False), critical=True)
                .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=False)
                .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
                .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
                .sign(ca_key, hashes.SHA256()))
        pem = serialization.Encoding.PEM
        self.cert_path.write_bytes(leaf.public_bytes(pem) + ca.public_bytes(pem))
        self.key_path.write_bytes(key.private_bytes(pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        log.info("Generated development C2PA credentials in %s", self.dir)

    def manifest(self, title: str, details: dict) -> dict:
        return {
            "claim_generator_info": [{"name": "Voice Clone Studio", "version": __version__}],
            "title": title,
            "assertions": [
                {"label": "c2pa.actions", "data": {"actions": [{
                    "action": "c2pa.created", "digitalSourceType": AI_SOURCE_TYPE,
                    "softwareAgent": {"name": "Voice Clone Studio", "version": __version__},
                    "description": "Synthetic speech generated from a consented voice profile"}]}},
                {"label": "com.voiceclonestudio.generation", "data": details},
            ],
        }

    def sign_file(self, path: Path, title: str, details: dict) -> dict:
        """Embeds a signed manifest into `path` in place. Returns a short report."""
        import c2pa

        info = c2pa.C2paSignerInfo(alg=b"es256", sign_cert=self.cert_path.read_bytes(),
                                   private_key=self.key_path.read_bytes(),
                                   ta_url=self.ta_url.encode() if self.ta_url else None)
        tmp = path.with_name(path.stem + ".c2pa-tmp" + path.suffix)
        try:
            with c2pa.Signer.from_info(info) as signer, c2pa.Builder(self.manifest(title, details)) as builder:
                builder.sign_file(str(path), str(tmp), signer)
            shutil.move(str(tmp), str(path))
        finally:
            tmp.unlink(missing_ok=True)
        return {"signed": True, "development_certificate": self.development,
                "timestamped": bool(self.ta_url)}


def read_c2pa(path: Path) -> dict:
    import c2pa

    try:
        with c2pa.Reader(str(path)) as reader:
            data = json.loads(reader.json())
    except Exception as e:  # noqa: BLE001
        return {"present": False, "detail": f"{type(e).__name__}: {e}"[:200]}
    active = data.get("manifests", {}).get(data.get("active_manifest", ""), {})
    ai = False
    details = {}
    for a in active.get("assertions", []):
        if a.get("label", "").startswith("c2pa.actions"):
            ai = ai or any(act.get("digitalSourceType") == AI_SOURCE_TYPE for act in a.get("data", {}).get("actions", []))
        if a.get("label") == "com.voiceclonestudio.generation":
            details = a.get("data", {})
    sig = active.get("signature_info", {})
    return {"present": True, "ai_generated": ai, "generator": active.get("claim_generator_info", [{}])[0].get("name"),
            "title": active.get("title"), "issuer": sig.get("issuer"), "signed_at": sig.get("time"),
            "validation_state": data.get("validation_state"),
            "validation_issues": [s.get("code") for s in data.get("validation_status", [])][:6],
            "details": details}
