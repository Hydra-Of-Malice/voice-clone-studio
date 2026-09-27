"""Entry point: `python -m vc.main` starts the API server (which supervises the GPU worker
process) and serves the UI."""
from __future__ import annotations

import logging

import uvicorn

from vc.api import create_app
from vc.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = create_app(settings)

if __name__ == "__main__":
    print(f"Voice Clone Studio -> http://{settings.host}:{settings.port}  (data: {settings.data_dir})")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
