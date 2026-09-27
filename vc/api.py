"""FastAPI application: the HTTP surface from the spec plus consent, provenance and fine-tuning.
The API process never loads a model: long-running work is queued as jobs for the GPU worker
process, and clients poll GET /api/jobs/{id} (or stream /api/jobs/{id}/events)."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vc import __version__, consent as consent_mod
from vc.config import Settings
from vc.db import Database, new_id
from vc.tts.base import STYLE_PRESETS

log = logging.getLogger("vc.api")
ROOT = Path(__file__).resolve().parent.parent
UI_DIRS = [ROOT / "web" / "dist", ROOT / "ui"]


class GenerateRequest(BaseModel):
    profile_id: str
    text: str = Field(min_length=1, max_length=20000)
    language: str = "auto"
    style: str = "natural"
    speed: float = Field(default=1.0, ge=0.8, le=1.2)
    use_fine_tuned: bool = True


class PreviewRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    language: str = "auto"


class ProfileRequest(BaseModel):
    sample_id: str
    name: str = Field(min_length=1, max_length=80)


class SampleRef(BaseModel):
    sample_id: str


class ProfileRef(BaseModel):
    profile_id: str


class ConsentStatementRequest(BaseModel):
    profile_id: str
    language: str | None = None


class AdapterToggle(BaseModel):
    active: bool


class WorkerSupervisor:
    """Starts `python -m vc.worker` and restarts it if it dies (e.g. a CUDA crash), so a model
    failure can never take the API down."""

    def __init__(self, settings: Settings):
        self.s = settings
        self.proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self.restarts = 0
        self._thread = threading.Thread(target=self._run, name="worker-supervisor", daemon=True)

    def start(self) -> None:
        if self.s.worker_mode == "spawn":
            self._thread.start()

    def _spawn(self) -> subprocess.Popen:
        logs = self.s.data_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        out = open(logs / "worker.log", "a", buffering=1, encoding="utf-8", errors="replace")
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        return subprocess.Popen([sys.executable, "-m", "vc.worker"], cwd=str(ROOT), env=env, stdout=out,
                                stderr=subprocess.STDOUT, creationflags=flags)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.proc = self._spawn()
            log.info("Worker process started (pid %d)", self.proc.pid)
            while self.proc.poll() is None and not self._stop.is_set():
                self._stop.wait(1.0)
            if self._stop.is_set():
                break
            self.restarts += 1
            log.error("Worker exited with code %s; restarting in 3 s", self.proc.returncode)
            self._stop.wait(3.0)

    def stop(self) -> None:
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def create_app(settings: Settings) -> FastAPI:
    settings.ensure_dirs()
    db = Database(settings.db_path)
    supervisor = WorkerSupervisor(settings)

    app = FastAPI(title="Voice Clone Studio", version=__version__)
    app.state.db = db

    @app.on_event("startup")
    def _start() -> None:
        supervisor.start()

    @app.on_event("shutdown")
    def _stop() -> None:
        supervisor.stop()

    def job_view(j: dict) -> dict:
        return {"job_id": j["id"], "kind": j["kind"], "status": j["status"], "progress": j["progress"],
                "message": j["message"], "result": json.loads(j["result_json"]) if j["result_json"] else None,
                "error": j["error"]}

    async def save_upload(file: UploadFile, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)

    # ---------------------------------------------------------------- status ----------------
    @app.get("/api/status")
    def status() -> dict:
        from vc.registry import load_builtin_engines, registry
        w = db.worker_status()
        try:
            load_builtin_engines()
            caps = [c.__dict__ for c in registry.capabilities()]
        except Exception as e:  # noqa: BLE001
            caps = [{"error": str(e)}]
        return {"version": __version__, "worker": w, "gpu": w.get("gpu", {}), "models": w.get("models", {}),
                "worker_mode": settings.worker_mode, "worker_restarts": supervisor.restarts,
                "engines": {"asr": settings.asr_engine, "asr_model": settings.asr_model,
                            "speaker": settings.speaker_engine, "tts": settings.tts_engine},
                "provenance": {"audioseal": settings.audioseal, "c2pa": settings.c2pa,
                               "c2pa_development_certificate": not (settings.c2pa_cert and settings.c2pa_key)},
                "registry": caps, "styles": list(STYLE_PRESETS)}

    # ---------------------------------------------------------------- upload ----------------
    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...), analyze: bool = Form(True)) -> dict:
        sample_id = new_id("vs_")
        suffix = Path(file.filename or "audio.wav").suffix.lower() or ".wav"
        dest = settings.uploads_dir / f"{sample_id}{suffix}"
        await save_upload(file, dest)
        db.insert("voice_samples", dict(id=sample_id, user_id="local", original_name=file.filename,
                                        path=str(dest), created_at=time.time()))
        out = {"sample_id": sample_id}
        if analyze:
            out["job_id"] = db.create_job("analyze", {"sample_id": sample_id})
            out["status"] = "queued"
        return out

    @app.post("/api/transcribe")
    def transcribe(req: SampleRef) -> dict:
        return _analyze(req.sample_id)

    @app.post("/api/analyze-voice")
    def analyze_voice(req: SampleRef) -> dict:
        return _analyze(req.sample_id)

    def _analyze(sample_id: str) -> dict:
        if not db.get("voice_samples", sample_id):
            raise HTTPException(404, "Unknown sample")
        return {"job_id": db.create_job("analyze", {"sample_id": sample_id}), "status": "queued"}

    @app.get("/api/samples/{sample_id}")
    def get_sample(sample_id: str) -> dict:
        row = db.get("voice_samples", sample_id)
        if not row:
            raise HTTPException(404, "Unknown sample")
        row["analysis"] = json.loads(row.pop("analysis_json") or "null")
        row.pop("path", None)
        return row

    # ---------------------------------------------------------------- profiles --------------
    @app.post("/api/create-profile")
    def create_profile(req: ProfileRequest) -> dict:
        if not db.get("voice_samples", req.sample_id):
            raise HTTPException(404, "Unknown sample")
        return {"job_id": db.create_job("create_profile", {"sample_id": req.sample_id, "name": req.name}),
                "status": "queued"}

    def adapters_of(profile_id: str) -> list[dict]:
        out = []
        for a in db.all("voice_adapters", "profile_id=?", (profile_id,)):
            out.append({"id": a["id"], "engine": a["engine"], "status": a["status"], "active": bool(a["active"]),
                        "metrics": json.loads(a["metrics_json"] or "{}"), "created_at": a["created_at"]})
        return out

    @app.get("/api/profiles")
    def list_profiles() -> list[dict]:
        out = []
        for r in db.all("voice_profiles"):
            p = json.loads(r.pop("profile_json") or "{}")
            r.pop("embedding_reference", None)
            r["clips"] = len(p.get("clips", []))
            r["style"] = p.get("style_features", {})
            r["pitch"] = p.get("pitch_statistics", {})
            r["speaking_rate"] = p.get("speaking_rate", {})
            r["adapters"] = adapters_of(r["id"])
            out.append(r)
        return out

    @app.get("/api/profiles/{profile_id}")
    def get_profile(profile_id: str) -> dict:
        r = db.get("voice_profiles", profile_id)
        if not r:
            raise HTTPException(404, "Unknown profile")
        p = json.loads(r.pop("profile_json") or "{}")
        p.pop("voice_embedding_b64", None)          # never expose raw biometrics
        r.pop("embedding_reference", None)
        r["profile"] = p
        r["adapters"] = adapters_of(profile_id)
        r["consents"] = [{k: v for k, v in c.items() if k not in ("audio_path",)}
                         for c in db.all("consent_records", "profile_id=?", (profile_id,))]
        return r

    @app.delete("/api/profiles/{profile_id}")
    def delete_profile(profile_id: str) -> dict:
        """Revocation: removes the encrypted clips, embeddings, adapters and the wrapped key."""
        r = db.get("voice_profiles", profile_id)
        if not r:
            raise HTTPException(404, "Unknown profile")
        shutil.rmtree(settings.profiles_dir / profile_id, ignore_errors=True)
        with db.connect() as con:
            con.execute("DELETE FROM voice_profiles WHERE id=?", (profile_id,))
            con.execute("DELETE FROM voice_adapters WHERE profile_id=?", (profile_id,))
            con.execute("UPDATE consent_records SET verified=0, audio_path=NULL WHERE profile_id=?", (profile_id,))
        return {"deleted": profile_id}

    # ---------------------------------------------------------------- consent ---------------
    @app.post("/api/consent/statement")
    def consent_statement(req: ConsentStatementRequest) -> dict:
        r = db.get("voice_profiles", req.profile_id)
        if not r:
            raise HTTPException(404, "Unknown profile")
        text, nonce = consent_mod.make_statement(req.language or r["language"] or "en")
        cid = new_id("cs_")
        db.insert("consent_records", dict(id=cid, profile_id=req.profile_id, user_id=r["user_id"],
                                          statement_text=text, nonce=nonce, verified=0,
                                          terms_version=consent_mod.TERMS_VERSION, created_at=time.time()))
        return {"consent_id": cid, "statement": text, "terms_version": consent_mod.TERMS_VERSION}

    @app.post("/api/consent/verify")
    async def consent_verify(request: Request, profile_id: str = Form(...), consent_id: str = Form(...),
                             file: UploadFile = File(...)) -> dict:
        rec = db.get("consent_records", consent_id)
        if not rec or rec["profile_id"] != profile_id:
            raise HTTPException(404, "Unknown consent record")
        suffix = Path(file.filename or "consent.wav").suffix or ".wav"
        dest = settings.profiles_dir / profile_id / f"consent_{consent_id}{suffix}"
        await save_upload(file, dest)
        client = json.dumps({"ip": request.client.host if request.client else None,
                             "ua": request.headers.get("user-agent", "")})
        return {"job_id": db.create_job("verify_consent", {"profile_id": profile_id, "consent_id": consent_id,
                                                           "audio_path": str(dest), "client_info": client}),
                "status": "queued"}

    # ---------------------------------------------------------------- text ------------------
    @app.post("/api/text/preview")
    def text_preview(req: PreviewRequest) -> dict:
        """Exactly what will be spoken: detected language, transliteration, normalised text, chunks."""
        from vc.text.normalize import prepare_script
        p = prepare_script(req.text, req.language)
        return {"language": p["language"], "engine_language": p["engine_language"], "text": p["text"],
                "transliteration": p["transliteration"],
                "chunks": [{"text": c.text, "language": c.language, "pause_after_s": round(c.pause_after_s, 2)}
                           for c in p["chunks"]]}

    # ---------------------------------------------------------------- generate --------------
    @app.post("/api/generate")
    def generate(req: GenerateRequest) -> dict:
        r = db.get("voice_profiles", req.profile_id)
        if not r:
            raise HTTPException(404, "Unknown profile")
        if r["consent_status"] != "verified":
            raise HTTPException(403, "Consent for this voice profile has not been verified")
        if req.style not in STYLE_PRESETS:
            raise HTTPException(400, f"Unknown style; choose one of {list(STYLE_PRESETS)}")
        gid = new_id("gen_")
        now = time.time()
        db.insert("generation_jobs", dict(id=gid, user_id=r["user_id"], voice_profile_id=req.profile_id,
                                          script=req.text, language=req.language, style=req.style,
                                          speed=req.speed, status="queued", created_at=now, updated_at=now,
                                          adapter_id=None if req.use_fine_tuned else "none"))
        jid = db.create_job("generate", {"generation_job_id": gid})
        return {"job_id": jid, "generation_id": gid, "status": "queued"}

    # ---------------------------------------------------------------- fine-tune -------------
    @app.post("/api/fine-tune")
    def fine_tune(req: ProfileRef) -> dict:
        r = db.get("voice_profiles", req.profile_id)
        if not r:
            raise HTTPException(404, "Unknown profile")
        if r["consent_status"] != "verified":
            raise HTTPException(403, "Consent for this voice profile has not been verified")
        if db.all("voice_adapters", "profile_id=? AND status='training'", (req.profile_id,)):
            raise HTTPException(409, "A fine-tuning run is already in progress for this profile")
        aid = new_id("ad_")
        db.insert("voice_adapters", dict(id=aid, profile_id=req.profile_id, engine=settings.tts_engine,
                                         path=str(settings.profiles_dir / req.profile_id / "adapters" / aid),
                                         status="training", active=0, created_at=time.time()))
        jid = db.create_job("fine_tune", {"profile_id": req.profile_id, "adapter_id": aid})
        return {"job_id": jid, "adapter_id": aid, "status": "queued"}

    @app.post("/api/adapters/{adapter_id}")
    def toggle_adapter(adapter_id: str, req: AdapterToggle) -> dict:
        a = db.get("voice_adapters", adapter_id)
        if not a:
            raise HTTPException(404, "Unknown adapter")
        if req.active and a["status"] != "accepted":
            raise HTTPException(400, "Only adapters that passed the quality gate can be activated")
        with db.connect() as con:
            con.execute("UPDATE voice_adapters SET active=0 WHERE profile_id=?", (a["profile_id"],))
            con.execute("UPDATE voice_adapters SET active=? WHERE id=?", (int(req.active), adapter_id))
        return {"adapter_id": adapter_id, "active": req.active}

    # ---------------------------------------------------------------- provenance ------------
    @app.post("/api/verify-audio")
    async def verify_audio(file: UploadFile = File(...)) -> dict:
        suffix = Path(file.filename or "audio.wav").suffix.lower() or ".wav"
        dest = settings.uploads_dir / "verify" / f"{new_id('vf_')}{suffix}"
        await save_upload(file, dest)
        return {"job_id": db.create_job("verify_audio", {"path": str(dest)}), "status": "queued"}

    # ---------------------------------------------------------------- jobs / audio ----------
    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict:
        j = db.get("jobs", job_id)
        if not j:
            raise HTTPException(404, "Unknown job")
        return job_view(j)

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str):
        if not db.get("jobs", job_id):
            raise HTTPException(404, "Unknown job")

        async def stream():
            last = None
            while True:
                view = job_view(db.get("jobs", job_id))
                key = (view["status"], view["progress"], view["message"])
                if key != last:
                    last = key
                    yield f"event: {view['status']}\ndata: {json.dumps(view, ensure_ascii=False)}\n\n"
                if view["status"] in ("completed", "failed"):
                    return
                await asyncio.sleep(0.4)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/audio/{audio_id}")
    def audio(audio_id: str, fmt: str = "wav", download: bool = False):
        r = db.get("generated_audio", audio_id)
        if not r:
            raise HTTPException(404, "Unknown audio")
        path = r["mp3_path"] if fmt == "mp3" else r["wav_path"]
        if not path or not Path(path).exists():
            raise HTTPException(404, "File missing")
        media = "audio/mpeg" if fmt == "mp3" else "audio/wav"
        headers = {"Content-Disposition": f'{"attachment" if download else "inline"}; filename="{audio_id}.{fmt}"'}
        return FileResponse(path, media_type=media, headers=headers)

    @app.get("/api/history")
    def history() -> list[dict]:
        rows = db.all("generated_audio")[:50]
        out = []
        for r in rows:
            gj = db.get("generation_jobs", r["job_id"]) or {}
            out.append({"id": r["id"], "profile_id": r["profile_id"], "duration_s": r["duration_s"],
                        "created_at": r["created_at"], "script": (gj.get("script") or "")[:200],
                        "metrics": json.loads(r["metrics_json"] or "{}")})
        return out

    @app.exception_handler(Exception)
    async def _err(_: Request, exc: Exception):
        log.exception("Unhandled error")
        return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

    for ui in UI_DIRS:
        if (ui / "index.html").exists():
            app.mount("/", StaticFiles(directory=str(ui), html=True), name="ui")
            log.info("Serving UI from %s", ui)
            break
    return app
