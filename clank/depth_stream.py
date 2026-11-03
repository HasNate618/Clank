"""Read the Clank live depth stream, standard library only.

The Vulkan layer (spike/e5) publishes 480x270 mono metres plus the matching
colour into a seqlock-protected mmap file. Everything here is plain Python so a
single frame can be read without entering the nix develop shell.
"""

from __future__ import annotations

import array
import json
import mmap
import os
import struct
import sys
import time
import zlib
from pathlib import Path

HEADER = struct.Struct("<12I3Q2f16I")
HEADER_SIZE = HEADER.size  # 144
MAGIC = 0x434C4B44  # "CLKD"
MAX_SLOTS = 3

_KEYS = (
    "magic",
    "version",
    "width",
    "height",
    "dtype",
    "eye",
    "color_width",
    "color_height",
    "color_dtype",
    "slot",
    "seq",
    "reserved0",
    "frame_id",
    "inject_ns",
    "publish_ns",
    "near_m",
    "far_m",
)


def stream_path() -> Path:
    override = os.environ.get("DEPTH_STREAM_PATH")
    if override:
        return Path(override)
    return Path.home() / ".local/state/Clank/depth-stream.bin"


def _flip_depth(depth: array.array, width: int, height: int) -> array.array:
    out = array.array("f")
    for y in range(height - 1, -1, -1):
        out.extend(depth[y * width : (y + 1) * width])
    return out


def _flip_color(color: bytes, width: int, height: int) -> bytes:
    stride = width * 4
    return b"".join(color[y * stride : (y + 1) * stride] for y in range(height - 1, -1, -1))


class Stream:
    """A read-only view of the stream file. Reusable between reads."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else stream_path()
        self._fh = open(self.path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self) -> None:
        self._mm.close()
        self._fh.close()

    def __enter__(self) -> "Stream":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _header(self) -> dict:
        return dict(zip(_KEYS, HEADER.unpack_from(self._mm, 0)))

    def read(self, timeout: float = 5.0, after_seq: int = 0, flip: bool = True) -> dict | None:
        """Wait for a stable frame newer than ``after_seq``.

        The layer samples in Vulkan's bottom-up order, so the default flips the
        rows to the top-down order every consumer (PNG, dashboard, depth maps)
        presents.
        """
        deadline = time.monotonic() + timeout
        while True:
            h = self._header()
            seq = int(h["seq"])
            if seq != 0 and seq % 2 == 0 and seq > after_seq:
                w, hgt = int(h["width"]), int(h["height"])
                cw, ch = int(h["color_width"]), int(h["color_height"])
                pix = w * hgt
                slot_bytes = pix * 4 + (cw * ch * 4 if cw and ch else 0)
                off = HEADER_SIZE + int(h["slot"]) * slot_bytes
                depth = array.array("f")
                depth.frombytes(self._mm[off : off + pix * 4])
                if sys.byteorder != "little":
                    depth.byteswap()
                color = None
                if cw and ch:
                    start = off + pix * 4
                    color = bytes(self._mm[start : start + cw * ch * 4])
                if int(self._header()["seq"]) == seq:
                    if flip:
                        depth = _flip_depth(depth, w, hgt)
                        if color is not None:
                            color = _flip_color(color, cw, ch)
                    h["depth"] = depth
                    h["color"] = color
                    h["read_ns"] = time.monotonic_ns()
                    return h
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.0005)


def latest(
    max_age: float = 2.0, path: Path | str | None = None, flip: bool = True
) -> dict | None:
    """Return the newest frame only if it was published within ``max_age`` seconds.

    Rows are flipped to top-down by default, like `Stream.read`; pass
    ``flip=False`` only for a consumer that presents the layer order itself.
    """
    try:
        stream = Stream(path)
    except OSError:
        return None
    try:
        frame = stream.read(timeout=0.05, flip=flip)
    except (OSError, ValueError):
        frame = None
    finally:
        stream.close()
    if frame is None:
        return None
    age = time.monotonic_ns() - int(frame["publish_ns"])
    if age < -1_000_000_000 or age > max_age * 1_000_000_000:
        return None
    return frame


def _png(width: int, height: int, bit_depth: int, color_type: int, rows: list[bytes]) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + row for row in rows)
    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def depth_png(depth: array.array | list, width: int, height: int, scale: float = 10.0) -> bytes:
    rows = []
    for y in range(height):
        row = array.array("H")
        base = y * width
        for x in range(width):
            v = depth[base + x]
            if not (v == v) or v in (float("inf"), float("-inf")):
                v = 0.0
            v = 0.0 if v < 0.0 else (1.0 if v > scale else v / scale)
            row.append(int(v * 65535.0))
        if sys.byteorder == "little":
            row.byteswap()
        rows.append(row.tobytes())
    return _png(width, height, 16, 0, rows)


def _rgb_rows(rgba: bytes, width: int, height: int) -> list[bytes]:
    rows = []
    for y in range(height):
        src = y * width * 4
        row = bytearray(width * 3)
        for x in range(width):
            i = src + x * 4
            j = x * 3
            row[j] = rgba[i]
            row[j + 1] = rgba[i + 1]
            row[j + 2] = rgba[i + 2]
        rows.append(bytes(row))
    return rows


def color_png(rgba: bytes, width: int, height: int) -> bytes:
    return _png(width, height, 8, 2, _rgb_rows(rgba, width, height))


def bmp(rgba: bytes, width: int, height: int) -> bytes:
    """A 24-bit bottom-up BMP, which every browser decodes without a helper."""
    stride = (width * 3 + 3) & ~3
    pixels = bytearray(stride * height)
    for y in range(height):
        src = (height - 1 - y) * width * 4
        dst = y * stride
        for x in range(width):
            i = src + x * 4
            j = dst + x * 3
            pixels[j] = rgba[i + 2]
            pixels[j + 1] = rgba[i + 1]
            pixels[j + 2] = rgba[i]
    header = struct.pack(
        "<2sIHHI",
        b"BM",
        14 + 40 + len(pixels),
        0,
        0,
        14 + 40,
    )
    info = struct.pack(
        "<IiiHHIIiiII",
        40,
        width,
        height,
        1,
        24,
        0,
        len(pixels),
        2835,
        2835,
        0,
        0,
    )
    return header + info + bytes(pixels)


def _npy_f32(values: array.array, width: int, height: int) -> bytes:
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }" % (height, width)
    prefix = b"\x93NUMPY\x01\x00"
    pad = (64 - (len(prefix) + 2 + len(header) + 1) % 64) % 64
    header = header + " " * pad + "\n"
    return prefix + struct.pack("<H", len(header)) + header.encode("ascii") + values.tobytes()


def _pgm(depth: array.array, width: int, height: int, scale: float = 10.0) -> bytes:
    u16 = array.array("H")
    for v in depth:
        if not (v == v) or v in (float("inf"), float("-inf")):
            v = 0.0
        v = 0.0 if v < 0.0 else (1.0 if v > scale else v / scale)
        u16.append(int(v * 65535.0))
    if sys.byteorder == "little":
        u16.byteswap()
    return f"P5\n{width} {height}\n65535\n".encode("ascii") + u16.tobytes()


def _ppm(rgba: bytes, width: int, height: int) -> bytes:
    rows = _rgb_rows(rgba, width, height)
    return f"P6\n{width} {height}\n255\n".encode("ascii") + b"".join(rows)


def save_frame(frame: dict, outdir: Path | str, near_hint: float | None = None) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    width, height = int(frame["width"]), int(frame["height"])
    depth = frame["depth"]
    color = frame["color"]

    (outdir / "depth.npy").write_bytes(_npy_f32(depth, width, height))
    (outdir / "depth.pgm").write_bytes(_pgm(depth, width, height))
    (outdir / "depth.png").write_bytes(depth_png(depth, width, height))
    if color is not None:
        cw, ch = int(frame["color_width"]), int(frame["color_height"])
        (outdir / "color.ppm").write_bytes(_ppm(color, cw, ch))
        (outdir / "color.png").write_bytes(color_png(color, cw, ch))

    valid = [v for v in depth if v > 0.0]
    meta = {k: frame[k] for k in _KEYS}
    meta.update(
        {
            "read_ns": int(frame["read_ns"]),
            "layer_ns": int(frame["publish_ns"]) - int(frame["inject_ns"]),
            "transport_ns": int(frame["read_ns"]) - int(frame["publish_ns"]),
            "age_ns": int(frame["read_ns"]) - int(frame["inject_ns"]),
            "near_hint": near_hint,
            "valid_frac": len(valid) / float(width * height) if depth else 0.0,
            "min_m": min(valid) if valid else None,
            "median_m": sorted(valid)[len(valid) // 2] if valid else None,
            "max_m": max(valid) if valid else None,
        }
    )
    (outdir / "frame.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def _edges(values: list[float], width: int, height: int) -> list[float]:
    out = [0.0] * (width * height)
    for y in range(1, height - 1):
        row = y * width
        up = row - width
        down = row + width
        for x in range(1, width - 1):
            gx = (values[row + x + 1] - values[row + x - 1]) * 0.5
            gy = (values[down + x] - values[up + x]) * 0.5
            out[row + x] = (gx * gx + gy * gy) ** 0.5
    return out


def verify_frame(frame: dict, outdir: Path | str) -> float:
    """Correlate colour and depth edges, and write an overlay PNG."""
    outdir = Path(outdir)
    width, height = int(frame["width"]), int(frame["height"])
    color = frame["color"]
    depth = frame["depth"]
    if color is None:
        raise SystemExit("no colour channel in the stream")

    cw, ch = int(frame["color_width"]), int(frame["color_height"])
    gray = [0.0] * (width * height)
    rgb = bytearray(width * height * 3)
    if cw == width and ch == height:
        for i in range(width * height):
            src = i * 4
            gray[i] = (color[src] + color[src + 1] + color[src + 2]) / 765.0
            rgb[i * 3 : i * 3 + 3] = color[src : src + 3]
    else:
        # The colour is a larger render; sample it on the depth grid.
        for y in range(height):
            cy = min(ch - 1, int((y + 0.5) * ch / height))
            for x in range(width):
                cx = min(cw - 1, int((x + 0.5) * cw / width))
                src = (cy * cw + cx) * 4
                dst = (y * width + x) * 3
                gray[y * width + x] = (color[src] + color[src + 1] + color[src + 2]) / 765.0
                rgb[dst : dst + 3] = color[src : src + 3]
    depth_f = [0.0 if (v != v or v in (float("inf"), float("-inf"))) else v for v in depth]

    e_col = _edges(gray, width, height)
    e_dep = _edges(depth_f, width, height)

    mean_col = sum(e_col) / len(e_col)
    mean_dep = sum(e_dep) / len(e_dep)
    a = [v - mean_col for v in e_col]
    b = [v - mean_dep for v in e_dep]
    denom = (sum(v * v for v in a) ** 0.5) * (sum(v * v for v in b) ** 0.5)
    corr = sum(x * y for x, y in zip(a, b)) / denom if denom else 0.0

    nz = sorted(v for v in e_dep if v > 0.0)
    thr = nz[int(len(nz) * 0.98)] if nz else 0.0
    overlay = bytearray(width * height * 3)
    for i in range(width * height):
        dst = i * 3
        if thr > 0.0 and e_dep[i] >= thr:
            overlay[dst], overlay[dst + 1], overlay[dst + 2] = 255, 0, 0
        else:
            overlay[dst], overlay[dst + 1], overlay[dst + 2] = rgb[dst], rgb[dst + 1], rgb[dst + 2]

    stride = width * 3
    overlay_bytes = bytes(overlay)
    rows = [overlay_bytes[y * stride : (y + 1) * stride] for y in range(height)]
    (outdir / "alignment.png").write_bytes(_png(width, height, 8, 2, rows))
    return corr
