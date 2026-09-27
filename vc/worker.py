"""GPU worker process: `python -m vc.worker`.

Claims jobs from the shared database, runs them one at a time on the GPU and publishes a heartbeat
(`worker_status` table). The API server starts and supervises one worker locally; in a cloud
deployment run any number of these against the same database (VC_WORKER_MODE=external on the API)."""
from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
import traceback

from vc.config import settings
from vc.db import Database
from vc.jobs import HANDLERS, ModelManager, Pipeline
from vc.registry import load_builtin_engines

log = logging.getLogger("vc.worker")


def _gpu() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return {"name": torch.cuda.get_device_name(0), "free_gb": round(free / 2 ** 30, 2),
                    "total_gb": round(total / 2 ** 30, 2)}
        return {"name": "CPU"}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings.ensure_dirs()
    load_builtin_engines()
    db = Database(settings.db_path)
    db.requeue_stale()
    models = ModelManager(settings)
    pipeline = Pipeline(settings, db, models)
    stop = threading.Event()
    current: dict = {"job": None}

    def heartbeat() -> None:
        while not stop.is_set():
            try:
                db.set_worker_status(os.getpid(), current["job"], {"gpu": _gpu(), "models": models.status()})
            except Exception:  # noqa: BLE001
                log.exception("heartbeat failed")
            stop.wait(2.0)

    threading.Thread(target=heartbeat, name="heartbeat", daemon=True).start()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_: stop.set())
        except Exception:  # noqa: BLE001
            pass
    log.info("Worker %d ready (device=%s, tts=%s, asr=%s)", os.getpid(), settings.device, settings.tts_engine,
             settings.asr_engine)

    while not stop.is_set():
        job = db.claim_next_job()
        if not job:
            stop.wait(0.5)
            continue
        current["job"] = job["id"]
        payload = json.loads(job["payload_json"])
        log.info("Job %s (%s) started", job["id"], job["kind"])
        t0 = time.time()

        def prog(frac: float, msg: str, _jid=job["id"]) -> None:
            db.job_progress(_jid, frac, msg)

        try:
            result = HANDLERS[job["kind"]](pipeline, payload, prog)
            db.job_done(job["id"], result)
            log.info("Job %s completed in %.1fs", job["id"], time.time() - t0)
        except Exception as e:  # noqa: BLE001
            log.error("Job %s failed: %s\n%s", job["id"], e, traceback.format_exc())
            db.job_failed(job["id"], f"{type(e).__name__}: {e}")
            if job["kind"] == "generate":
                db.update("generation_jobs", payload["generation_job_id"], status="failed", error=str(e),
                          updated_at=time.time())
            if job["kind"] == "fine_tune":
                db.update("voice_adapters", payload["adapter_id"], status="failed",
                          metrics_json=json.dumps({"error": str(e)}))
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        finally:
            current["job"] = None
    log.info("Worker stopped")


if __name__ == "__main__":
    main()
