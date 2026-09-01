"""Small browser-safe height codec.

``encode_rg16`` stores a height as two 8-bit grayscale channels: red is the
high byte and green the low byte of a uint16 quantized to ``[minimum, maximum]``.
The PNG is therefore lossless, deterministic, and easy to decode in WebGL or
JavaScript.  Quantization error is at most one half of one 16-bit step.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _write_png(path: Path, pixels: np.ndarray, color_type: int) -> None:
    pixels = np.asarray(pixels, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = b"".join(b"\x00" + row.tobytes() for row in pixels)
    h, w = pixels.shape[:2]
    header = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header) + _png_chunk(b"IDAT", zlib.compress(rows)) + _png_chunk(b"IEND", b""))


def encode_rg16(values: np.ndarray, path: str | Path, minimum: float, maximum: float) -> dict[str, float | str]:
    """Write packed RG16 PNG and return manifest codec metadata."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        raise ValueError("values must be 2D and maximum must exceed minimum")
    normalized = np.clip((np.nan_to_num(array, nan=minimum) - minimum) / (maximum - minimum), 0, 1)
    packed = np.rint(normalized * 65535).astype(np.uint16)
    channels = np.stack([(packed >> 8).astype(np.uint8), (packed & 255).astype(np.uint8)], axis=-1)
    _write_png(Path(path), channels, 4)
    return {"encoding": "rg16-linear", "minimum_m": float(minimum), "maximum_m": float(maximum)}
