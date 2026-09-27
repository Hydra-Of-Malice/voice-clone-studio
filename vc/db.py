"""SQLite persistence (WAL mode). The schema mirrors the spec: users, voice_samples, transcriptions,
voice_profiles, consent_records, generation_jobs, generated_audio, plus a generic jobs table that the
worker polls. Swap for Postgres by re-implementing `connect()` and the few SQL strings below."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    display_name TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS voice_samples (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    original_name TEXT,
    path TEXT NOT NULL,
    duration_s REAL,
    sample_rate INTEGER,
    analysis_json TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS transcriptions (
    id TEXT PRIMARY KEY,
    sample_id TEXT NOT NULL,
    engine TEXT,
    language TEXT,
    confidence REAL,
    transcript TEXT,
    segments_json TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS voice_profiles (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    name TEXT NOT NULL,
    embedding_reference TEXT,
    language TEXT,
    quality_score REAL,
    consent_status TEXT NOT NULL DEFAULT 'pending',
    profile_json TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS consent_records (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    statement_text TEXT NOT NULL,
    nonce TEXT NOT NULL,
    audio_path TEXT,
    asr_text TEXT,
    text_match REAL,
    speaker_cosine REAL,
    verified INTEGER NOT NULL DEFAULT 0,
    terms_version TEXT NOT NULL,
    client_info TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS generation_jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    voice_profile_id TEXT NOT NULL,
    script TEXT NOT NULL,
    language TEXT,
    style TEXT,
    speed REAL,
    status TEXT NOT NULL,
    output_id TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS generated_audio (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    wav_path TEXT NOT NULL,
    mp3_path TEXT,
    duration_s REAL,
    metrics_json TEXT,
    watermark TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    message TEXT,
    payload_json TEXT NOT NULL,
    result_json TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at);
CREATE TABLE IF NOT EXISTS voice_adapters (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    engine TEXT NOT NULL,
    path TEXT NOT NULL,
    status TEXT NOT NULL,              -- training | accepted | rejected | failed
    active INTEGER NOT NULL DEFAULT 0,
    metrics_json TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS worker_status (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    pid INTEGER,
    heartbeat REAL,
    current_job TEXT,
    status_json TEXT
);
"""

# Columns added after the first release: (table, column, declaration)
_MIGRATIONS = [
    ("generated_audio", "audioseal_payload", "INTEGER"),
    ("generated_audio", "provenance_json", "TEXT"),
    ("generation_jobs", "adapter_id", "TEXT"),
]


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:16]}"


class Database:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(_SCHEMA)
            for table, column, decl in _MIGRATIONS:
                cols = {r["name"] for r in con.execute(f"PRAGMA table_info({table})")}
                if column not in cols:
                    con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
            con.execute(
                "INSERT OR IGNORE INTO users(id, display_name, created_at) VALUES('local', 'Local user', ?)",
                (time.time(),),
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            con = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            con.row_factory = sqlite3.Row
            try:
                yield con
                con.commit()
            finally:
                con.close()

    # ---- generic helpers -------------------------------------------------------------------
    def insert(self, table: str, row: dict[str, Any]) -> None:
        cols = ", ".join(row)
        qs = ", ".join("?" for _ in row)
        with self.connect() as con:
            con.execute(f"INSERT INTO {table}({cols}) VALUES({qs})", tuple(row.values()))

    def update(self, table: str, id_: str, **fields: Any) -> None:
        sets = ", ".join(f"{k}=?" for k in fields)
        with self.connect() as con:
            con.execute(f"UPDATE {table} SET {sets} WHERE id=?", (*fields.values(), id_))

    def get(self, table: str, id_: str) -> dict[str, Any] | None:
        with self.connect() as con:
            row = con.execute(f"SELECT * FROM {table} WHERE id=?", (id_,)).fetchone()
            return dict(row) if row else None

    def all(self, table: str, where: str = "", params: tuple = (), order: str = "created_at DESC") -> list[dict]:
        sql = f"SELECT * FROM {table}" + (f" WHERE {where}" if where else "") + f" ORDER BY {order}"
        with self.connect() as con:
            return [dict(r) for r in con.execute(sql, params).fetchall()]

    # ---- jobs --------------------------------------------------------------------------------
    def create_job(self, kind: str, payload: dict[str, Any]) -> str:
        jid = new_id("job_")
        now = time.time()
        self.insert("jobs", dict(id=jid, kind=kind, status="queued", progress=0.0, message="Queued",
                                 payload_json=json.dumps(payload), created_at=now, updated_at=now))
        return jid

    def claim_next_job(self) -> dict[str, Any] | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
            if not row:
                return None
            con.execute("UPDATE jobs SET status='processing', updated_at=? WHERE id=?", (time.time(), row["id"]))
            job = dict(row)
            job["status"] = "processing"
            return job

    def job_progress(self, jid: str, progress: float, message: str) -> None:
        self.update("jobs", jid, progress=float(progress), message=message, updated_at=time.time())

    def job_done(self, jid: str, result: dict[str, Any]) -> None:
        self.update("jobs", jid, status="completed", progress=1.0, message="Completed",
                    result_json=json.dumps(result), updated_at=time.time())

    def job_failed(self, jid: str, error: str) -> None:
        self.update("jobs", jid, status="failed", message="Failed", error=error, updated_at=time.time())

    # ---- worker status ----------------------------------------------------------------------
    def set_worker_status(self, pid: int, current_job: str | None, status: dict[str, Any]) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO worker_status(id, pid, heartbeat, current_job, status_json) VALUES(1, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET pid=excluded.pid, heartbeat=excluded.heartbeat, "
                "current_job=excluded.current_job, status_json=excluded.status_json",
                (pid, time.time(), current_job, json.dumps(status)))

    def worker_status(self) -> dict[str, Any]:
        with self.connect() as con:
            row = con.execute("SELECT * FROM worker_status WHERE id=1").fetchone()
        if not row:
            return {"alive": False}
        out = json.loads(row["status_json"] or "{}")
        out.update(pid=row["pid"], current_job=row["current_job"],
                   alive=(time.time() - (row["heartbeat"] or 0)) < 15,
                   heartbeat_age_s=round(time.time() - (row["heartbeat"] or 0), 1))
        return out

    def requeue_stale(self) -> None:
        """Jobs left 'processing' by a crashed worker are marked failed on startup."""
        with self.connect() as con:
            con.execute("UPDATE jobs SET status='failed', error='Worker restarted', updated_at=? "
                        "WHERE status='processing'", (time.time(),))
