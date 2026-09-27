"""Writes a multi-size .ico (a waveform on an indigo rounded square) using only the standard library."""
from __future__ import annotations

import struct
import sys
import zlib


def png(size: int) -> bytes:
    bars = [0.30, 0.55, 0.85, 0.50, 0.95, 0.60, 0.35]
    rows = bytearray()
    r = size * 0.22
    n = len(bars)
    slot = size * 0.72 / n
    x0 = size * 0.14
    for y in range(size):
        rows.append(0)
        for x in range(size):
            # rounded-square mask
            cx = min(max(x + 0.5, r), size - r)
            cy = min(max(y + 0.5, r), size - r)
            inside = (x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 <= r * r
            if not inside:
                rows += b"\x00\x00\x00\x00"
                continue
            col = (79, 70, 229, 255)
            i = int((x + 0.5 - x0) // slot)
            if 0 <= i < n and (x + 0.5 - x0) - i * slot < slot * 0.62:
                h = bars[i] * size * 0.62
                if abs(y + 0.5 - size / 2) <= h / 2:
                    col = (255, 255, 255, 255)
            rows += bytes(col)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(rows), 9)) + chunk(b"IEND", b""))


def main(path: str) -> None:
    sizes = [16, 24, 32, 48, 64, 128, 256]
    images = [png(s) for s in sizes]
    out = struct.pack("<HHH", 0, 1, len(sizes))
    offset = 6 + 16 * len(sizes)
    for s, img in zip(sizes, images):
        out += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(img), offset)
        offset += len(img)
    with open(path, "wb") as f:
        f.write(out + b"".join(images))
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1])
