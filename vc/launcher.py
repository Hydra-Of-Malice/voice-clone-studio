"""Desktop launcher (`pythonw -m vc.launcher`): starts the local server, opens the app in its own
window and shuts everything down when that window is closed. Standard library only.

The window is Microsoft Edge in app mode (present on every Windows 10/11 PC): unlike an embedded
web view it shows the normal microphone permission prompt, which recording needs."""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path

APP_TITLE = "Voice Clone Studio"
INSTALL_DIR = Path(__file__).resolve().parent.parent
HOST = os.environ.get("VC_HOST", "127.0.0.1")
PORT = int(os.environ.get("VC_PORT", "8765"))
URL = f"http://{HOST}:{PORT}/"
CREATE_NO_WINDOW = 0x08000000
MIN_VRAM_MB = 7000


def data_dir() -> Path:
    d = Path(os.environ.get("VC_DATA_DIR", str(Path.home() / ".voice-clone")))
    (d / "logs").mkdir(parents=True, exist_ok=True)
    return d


def log(msg: str) -> None:
    try:
        with open(data_dir() / "logs" / "launcher.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")
    except OSError:
        pass


def message(text: str, error: bool = False) -> None:
    log(f"[dialog] {text}")
    try:
        ctypes.windll.user32.MessageBoxW(None, text, APP_TITLE, (0x10 if error else 0x30) | 0x10000)
    except Exception:  # noqa: BLE001
        pass


def server_status() -> dict | None:
    try:
        with urllib.request.urlopen(URL + "api/status", timeout=2) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001
        return None


def check_gpu() -> None:
    smi = shutil.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"
    if not Path(smi).exists():
        message("No NVIDIA graphics driver was found.\n\nVoice Clone Studio needs an NVIDIA GPU with 8 GB of memory. "
                "It will start, but it cannot generate speech on this computer.")
        return
    try:
        out = subprocess.run([smi, "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15, creationflags=CREATE_NO_WINDOW).stdout
        name, mem, cap = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        log(f"GPU: {name}, {mem} MiB, compute capability {cap}")
        if float(cap) >= 10:
            message(f"{name} (RTX 50 series) is not supported by this release; speech generation will fail.\n\n"
                    "A later release will add support.")
        elif int(float(mem)) < MIN_VRAM_MB:
            message(f"{name} has {int(float(mem)) / 1024:.0f} GB of memory; 8 GB is recommended.\n\n"
                    "Generation may fail with an out-of-memory error. Close other apps that use the GPU.")
    except Exception as e:  # noqa: BLE001
        log(f"GPU check failed: {e}")


def python_exe(windowed: bool) -> str:
    name = "pythonw.exe" if windowed else "python.exe"
    for base in (INSTALL_DIR / "runtime", INSTALL_DIR / ".venv" / "Scripts"):
        if (base / name).exists():
            return str(base / name)
    return sys.executable


def start_server() -> subprocess.Popen:
    env = dict(os.environ, PYTHONPATH=str(INSTALL_DIR), PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    out = open(data_dir() / "logs" / "server.log", "w", encoding="utf-8", errors="replace")
    return subprocess.Popen([python_exe(windowed=False), "-m", "vc.main"], cwd=str(INSTALL_DIR), env=env,
                            stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                            creationflags=CREATE_NO_WINDOW)


def find_edge() -> str | None:
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
        if base:
            p = Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe"
            if p.exists():
                return str(p)
    return shutil.which("msedge")


def open_window() -> subprocess.Popen | None:
    edge = find_edge()
    if not edge:
        webbrowser.open(URL)
        return None
    profile = data_dir() / "window"
    return subprocess.Popen([edge, f"--app={URL}", f"--user-data-dir={profile}", "--no-first-run",
                             "--no-default-browser-check", "--window-size=1100,900"])


def kill_tree(pid: int) -> None:
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=CREATE_NO_WINDOW)


def main() -> int:
    log(f"launcher start; install dir {INSTALL_DIR}")
    if server_status():                       # already running: just show the window
        open_window()
        return 0
    check_gpu()
    server = start_server()
    deadline = time.time() + 90
    while time.time() < deadline and not server_status():
        if server.poll() is not None:
            tail = ""
            try:
                tail = "\n".join((data_dir() / "logs" / "server.log").read_text("utf-8", "replace").splitlines()[-10:])
            except OSError:
                pass
            message(f"{APP_TITLE} stopped while starting.\n\n{tail}", error=True)
            return 1
        time.sleep(0.5)
    if not server_status():
        kill_tree(server.pid)
        message(f"{APP_TITLE} did not start within 90 seconds. See {data_dir() / 'logs'}.", error=True)
        return 1
    window = open_window()
    if window is None:
        message(f"{APP_TITLE} is open in your web browser.\n\nClick OK here when you are finished to stop it.")
    else:
        window.wait()
    log("window closed; stopping the server")
    kill_tree(server.pid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
