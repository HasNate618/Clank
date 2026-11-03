#!/usr/bin/env python3
"""Read the Clank live depth stream (480x270 mono, metres) from shared memory.

The layer writes a seqlock-protected mmap file; this reader waits for a stable
frame, reports latency, and can save depth/colour plus an alignment check.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HEADER = struct.Struct("<12I3Q2f16I")
HEADER_SIZE = HEADER.size  # 144


def stream_path() -> Path:
    p = os.environ.get("DEPTH_STREAM_PATH")
    if p:
        return Path(p)
    return Path.home() / ".local/state/Clank/depth-stream.bin"


def _fields(values) -> dict:
    keys = [
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
    ]
    return dict(zip(keys, values[: len(keys)]))


class Stream:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else stream_path()
        self._fh = open(self.path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self) -> None:
        self._mm.close()
        self._fh.close()

    def _header(self) -> dict:
        return _fields(HEADER.unpack_from(self._mm, 0))

    def read(self, timeout: float = 5.0, after_seq: int = 0) -> dict | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            h = self._header()
            seq = int(h["seq"])
            if seq == 0 or seq % 2 != 0 or seq <= after_seq:
                time.sleep(0.0005)
                continue
            slot = int(h["slot"])
            w, hgt = int(h["width"]), int(h["height"])
            cw, ch = int(h["color_width"]), int(h["color_height"])
            pix = w * hgt
            slot_bytes = pix * 4 + (cw * ch * 4 if cw and ch else 0)
            off = HEADER_SIZE + slot * slot_bytes
            depth = np.frombuffer(self._mm, dtype="<f4", count=pix, offset=off).reshape(hgt, w).copy()
            color = None
            if cw and ch:
                color = (
                    np.frombuffer(
                        self._mm, dtype=np.uint8, count=cw * ch * 4, offset=off + pix * 4
                    )
                    .reshape(ch, cw, 4)
                    .copy()
                )
            again = int(self._header()["seq"])
            if again == seq:
                h["depth"] = depth
                h["color"] = color
                h["read_ns"] = time.monotonic_ns()
                return h
            time.sleep(0.0002)
        return None


def _to_png(pgm_or_ppm: Path) -> Path | None:
    try:
        sys.path.insert(0, str(REPO / "spike/e4"))
        from depth_post import to_png  # type: ignore

        return to_png(pgm_or_ppm)
    except Exception:
        return None


def _save(frame: dict, outdir: Path, near_hint: float | None = None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    depth = frame["depth"]
    color = frame["color"]
    np.save(outdir / "depth.npy", depth)
    scale = 10.0
    vis = np.clip(np.nan_to_num(depth, nan=0.0, posinf=scale) / scale, 0.0, 1.0)
    h, w = vis.shape
    pgm = outdir / "depth.pgm"
    with open(pgm, "wb") as fh:
        fh.write(f"P5\n{w} {h}\n65535\n".encode())
        fh.write((vis * 65535).astype(">u2").tobytes())
    if color is not None:
        ppm = outdir / "color.ppm"
        with open(ppm, "wb") as fh:
            fh.write(f"P6\n{w} {h}\n255\n".encode())
            fh.write(color[..., :3].astype(np.uint8).tobytes())
    meta = {
        k: frame[k]
        for k in (
            "width",
            "height",
            "eye",
            "frame_id",
            "inject_ns",
            "publish_ns",
            "read_ns",
            "near_m",
        )
    }
    meta["layer_ns"] = int(frame["publish_ns"]) - int(frame["inject_ns"])
    meta["transport_ns"] = int(frame["read_ns"]) - int(frame["publish_ns"])
    meta["age_ns"] = int(frame["read_ns"]) - int(frame["inject_ns"])
    meta["near_hint"] = near_hint
    meta["valid_frac"] = float(np.mean(depth > 0.0))
    meta["min_m"] = float(np.min(depth[depth > 0.0])) if np.any(depth > 0.0) else None
    meta["median_m"] = float(np.median(depth[depth > 0.0])) if np.any(depth > 0.0) else None
    meta["max_m"] = float(np.max(depth[np.isfinite(depth)])) if np.any(np.isfinite(depth)) else None
    (outdir / "frame.json").write_text(json.dumps(meta, indent=2) + "\n")
    _to_png(pgm)


def _verify(frame: dict, outdir: Path) -> None:
    depth = frame["depth"]
    color = frame["color"]
    if color is None:
        raise SystemExit("no colour channel in the stream")
    gray = color[..., :3].astype(np.float32).mean(axis=2) / 255.0
    gz = np.nan_to_num(depth, nan=0.0, posinf=0.0)
    # Sobel-ish edge magnitude on both images, then a correlation of the maps.
    def edges(a: np.ndarray) -> np.ndarray:
        gx = np.zeros_like(a)
        gy = np.zeros_like(a)
        gx[:, 1:-1] = (a[:, 2:] - a[:, :-2]) * 0.5
        gy[1:-1, :] = (a[2:, :] - a[:-2, :]) * 0.5
        return np.hypot(gx, gy)

    e_col = edges(gray)
    e_dep = edges(gz)
    a = e_col.ravel()
    b = e_dep.ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    corr = float(np.dot(a, b) / denom) if denom > 0 else 0.0
    h, w = depth.shape
    over = color[..., :3].copy()
    nz = e_dep[e_dep > 0.0]
    thr = float(np.quantile(nz, 0.98)) if nz.size else 0.0
    mask = e_dep >= thr if thr > 0.0 else np.zeros_like(e_dep, dtype=bool)
    over[mask] = (255, 0, 0)
    ppm = outdir / "alignment.ppm"
    with open(ppm, "wb") as fh:
        fh.write(f"P6\n{w} {h}\n255\n".encode())
        fh.write(over.astype(np.uint8).tobytes())
    _to_png(ppm)
    print(f"edge correlation (colour vs depth): {corr:.3f}")
    print(f"wrote {ppm}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="depth_stream")
    parser.add_argument("--path", default=None)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--out", default="research/evidence/depth-stream")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args(argv)

    stream = Stream(Path(args.path) if args.path else None)
    outdir = Path(args.out)

    if args.watch:
        last = 0
        intervals = []
        layers = []
        transports = []
        ages = []
        t_end = time.monotonic() + args.seconds
        prev_pub = None
        while time.monotonic() < t_end:
            f = stream.read(timeout=min(1.0, max(0.0, t_end - time.monotonic())), after_seq=last)
            if f is None:
                continue
            last = int(f["seq"])
            now = f["read_ns"]
            layer = int(f["publish_ns"]) - int(f["inject_ns"])
            transport = now - int(f["publish_ns"])
            age = now - int(f["inject_ns"])
            layers.append(layer)
            transports.append(transport)
            ages.append(age)
            if prev_pub is not None:
                intervals.append(int(f["publish_ns"]) - prev_pub)
            prev_pub = int(f["publish_ns"])
            print(
                f"seq={last} frame={f['frame_id']} "
                f"layer={layer/1e6:6.2f}ms transport={transport/1e6:6.2f}ms "
                f"age={age/1e6:6.2f}ms"
            )
        if intervals:
            fps = 1e9 / float(np.median(intervals))
            print(f"frames={len(ages)} fps={fps:.1f}")
        for name, vals in (("layer", layers), ("transport", transports), ("age", ages)):
            if vals:
                print(
                    f"{name:9s} min={np.min(vals)/1e6:.2f}ms "
                    f"median={np.median(vals)/1e6:.2f}ms p95={np.percentile(vals,95)/1e6:.2f}ms"
                )
        stream.close()
        return 0

    frame = stream.read(timeout=args.timeout)
    if frame is None:
        raise SystemExit("no depth frame appeared (is the game running with the layer?)")
    _save(frame, outdir)
    if args.verify:
        _verify(frame, outdir)
    print(f"frame {frame['frame_id']} seq={frame['seq']} near={frame['near_m']:.6f} -> {outdir}")
    print(
        f"layer={(frame['publish_ns']-frame['inject_ns'])/1e6:.2f}ms "
        f"age={(frame['read_ns']-frame['inject_ns'])/1e6:.2f}ms"
    )
    stream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
