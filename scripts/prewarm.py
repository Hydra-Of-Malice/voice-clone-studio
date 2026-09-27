"""Warm the antivirus scan cache before compiling the installer.

Windows Defender scans every file the first time a process opens it and then remembers the verdict
until the file changes. The Inno Setup compiler opens the tens of thousands of runtime files one at a
time and waits on every scan; reading them first from a thread pool lets the scans run on all cores.

    python scripts\\prewarm.py build\\runtime build\\models [--threads 12]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def collect(roots: list[str]) -> list[str]:
    files: list[str] = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            files.extend(os.path.join(dirpath, f) for f in sorted(filenames) if not f.endswith(".pyc"))
    return files


def read_all(path: str) -> None:
    try:
        with open(path, "rb") as fh:
            while fh.read(8 << 20):
                pass
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("roots", nargs="+")
    p.add_argument("--threads", type=int, default=12)
    a = p.parse_args(argv)
    files = collect([r for r in a.roots if os.path.isdir(r)])
    t0 = time.perf_counter()
    last = t0
    done = 0
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        for _ in ex.map(read_all, files, chunksize=64):
            done += 1
            if time.perf_counter() - last > 15:
                last = time.perf_counter()
                print(f"  {done}/{len(files)} files after {last - t0:.0f} s", flush=True)
    print(f"  read {done} files in {time.perf_counter() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
