#!/usr/bin/env python3
"""Read the Clank live depth stream (480x270 mono, metres) from shared memory.

The layer writes a seqlock-protected mmap file; this reader waits for a stable
frame, reports latency, and can save depth/colour plus an alignment check. It is
standard-library only, so `python3 spike/e5/depth_stream.py --once` works without
entering the nix develop shell.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from clank.depth_stream import Stream, save_frame, verify_frame  # noqa: E402


def _percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return float(ordered[index])


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

    stream = Stream(args.path)
    outdir = Path(args.out)

    if args.watch:
        last = 0
        intervals: list[int] = []
        layers: list[int] = []
        transports: list[int] = []
        ages: list[int] = []
        t_end = time.monotonic() + args.seconds
        prev_pub = None
        while time.monotonic() < t_end:
            frame = stream.read(
                timeout=min(1.0, max(0.0, t_end - time.monotonic())), after_seq=last
            )
            if frame is None:
                continue
            last = int(frame["seq"])
            now = frame["read_ns"]
            layer = int(frame["publish_ns"]) - int(frame["inject_ns"])
            transport = now - int(frame["publish_ns"])
            age = now - int(frame["inject_ns"])
            layers.append(layer)
            transports.append(transport)
            ages.append(age)
            if prev_pub is not None:
                intervals.append(int(frame["publish_ns"]) - prev_pub)
            prev_pub = int(frame["publish_ns"])
            print(
                f"seq={last} frame={frame['frame_id']} "
                f"layer={layer / 1e6:6.2f}ms transport={transport / 1e6:6.2f}ms "
                f"age={age / 1e6:6.2f}ms"
            )
        if intervals:
            ordered = sorted(intervals)
            print(f"frames={len(ages)} fps={1e9 / ordered[len(ordered) // 2]:.1f}")
        for name, values in (("layer", layers), ("transport", transports), ("age", ages)):
            if values:
                print(
                    f"{name:9s} min={min(values) / 1e6:.2f}ms "
                    f"median={_percentile(values, 0.5) / 1e6:.2f}ms "
                    f"p95={_percentile(values, 0.95) / 1e6:.2f}ms"
                )
        stream.close()
        return 0

    frame = stream.read(timeout=args.timeout)
    if frame is None:
        raise SystemExit("no depth frame appeared (is the game running with the layer?)")
    save_frame(frame, outdir)
    if args.verify:
        corr = verify_frame(frame, outdir)
        print(f"edge correlation (colour vs depth): {corr:.3f}")
    print(
        f"frame {frame['frame_id']} seq={frame['seq']} "
        f"near={frame['near_m']:.6f} -> {outdir}"
    )
    print(
        f"layer={(frame['publish_ns'] - frame['inject_ns']) / 1e6:.2f}ms "
        f"age={(frame['read_ns'] - frame['inject_ns']) / 1e6:.2f}ms"
    )
    stream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
