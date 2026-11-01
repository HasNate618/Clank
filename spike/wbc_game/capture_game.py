#!/usr/bin/env python3
"""Capture the live game view, headless, and prove the frame is a rendered scene.

This is the spike that exercises the real transport end to end:

    spike/e3/.venv/bin/python spike/wbc_game/capture_game.py \
        --runtime-dir /run/user/1000 \
        --wayland-display clank-27abb1 \
        --out /tmp/game.png

It calls `clank.capture.capture_frame`, which runs `weston-screenshooter` against
the compositor that owns the game. No browser, no VNC client, no password.

Measured 2026-11-01 on session `20261101T223603Z-27abb1`: 1920x1080 RGB PNG,
~1.4 MB, ~0.4 s per shot. The PNG comes out **vertically mirrored** because the
GL renderer reports `screen capture uses y-flip: yes`; this script flips it back
with Pillow before publishing, and prints `flipped: true` when it did. Use
`--no-flip` to keep the compositor's raw orientation.

The numbers printed at the end are the evidence that the frame is a scene and not
a cleared window: a mostly-blank frame has a low luma standard deviation and few
distinct colours, while a rendered one has thousands of colours and neighbouring
pixels strongly correlated. `--check` turns that into an exit code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from clank.capture import capture_frame  # noqa: E402

DESCRIPTION = "Capture the live game view, headless, and check that it is rendered."
DEFAULT_WORK_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "clank" / "capture"


def analyse(path: Path) -> dict:
    """Pillow/numpy measurements, imported late so `--help` needs no venv."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        pixels = np.asarray(rgb, dtype=np.uint8)
    flat = pixels.reshape(-1, 3)
    luma = flat.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    horizontal = np.corrcoef(luma.reshape(height, width)[:, :-1].ravel(), luma.reshape(height, width)[:, 1:].ravel())[0, 1]
    vertical = np.corrcoef(luma.reshape(height, width)[:-1, :].ravel(), luma.reshape(height, width)[1:, :].ravel())[0, 1]
    return {
        "width": width,
        "height": height,
        "bytes": path.stat().st_size,
        "mean_rgb": [round(float(channel), 1) for channel in flat.mean(axis=0)],
        "luma_mean": round(float(luma.mean()), 1),
        "luma_std": round(float(luma.std()), 1),
        "distinct_colours": int(len(np.unique(flat, axis=0))),
        "corr_horizontal": round(float(horizontal), 4),
        "corr_vertical": round(float(vertical), 4),
        "bright_fraction": round(float((luma > 250).mean()), 4),
        "dark_fraction": round(float((luma < 5).mean()), 4),
    }


def rendered_scene(report: dict, min_std: float, min_colours: int) -> bool:
    """A cheap blank-frame test, not a scene classifier.

    A cleared or black window fails on luma spread and distinct colours. A real
    frame passes both with a wide margin; the thresholds are deliberately far
    below the measured 1920x1080 game values (std ~58, ~138k colours).
    """
    return report["luma_std"] >= min_std and report["distinct_colours"] >= min_colours


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--runtime-dir", default="/run/user/1000")
    parser.add_argument("--wayland-display", required=True, help="e.g. clank-27abb1")
    parser.add_argument("--screenshooter", default="weston-screenshooter")
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--out", required=True, help="where to publish the frame")
    parser.add_argument(
        "--raw-out",
        help="keep the compositor's raw-orientation PNG here instead of discarding it",
    )
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="keep the compositor's vertically mirrored orientation",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--check", action="store_true", help="exit 1 unless the frame looks rendered")
    parser.add_argument("--min-std", type=float, default=10.0)
    parser.add_argument("--min-colours", type=int, default=1000)
    args = parser.parse_args(argv)

    started = time.monotonic()
    raw = Path(args.out).with_name(Path(args.out).stem + ".raw.png")
    capture_frame(
        screenshooter=args.screenshooter,
        runtime_dir=args.runtime_dir,
        wayland_display=args.wayland_display,
        work_dir=args.work_dir,
        destination=raw,
        timeout=args.timeout,
    )
    elapsed = time.monotonic() - started

    flipped = not args.no_flip
    if flipped:
        from PIL import Image, ImageOps

        with Image.open(raw) as image:
            ImageOps.flip(image).save(args.out)
    else:
        raw.replace(args.out)

    if flipped:
        if args.raw_out:
            Path(args.raw_out).parent.mkdir(parents=True, exist_ok=True)
            raw.replace(args.raw_out)
        else:
            raw.unlink(missing_ok=True)

    report = analyse(Path(args.out))
    report.update(
        {
            "path": str(Path(args.out).resolve()),
            "raw_path": str(Path(args.raw_out).resolve()) if args.raw_out and flipped else None,
            "flipped": flipped,
            "seconds": round(elapsed, 3),
        }
    )
    report["rendered"] = rendered_scene(report, args.min_std, args.min_colours)
    report["verdict"] = (
        "rendered scene"
        if report["rendered"]
        else "blank or near-blank frame: the compositor produced no scene"
    )
    print(json.dumps(report, indent=2))
    if not flipped:
        print(
            "note: raw orientation is vertically mirrored under this compositor; "
            "pass no flag to flip it",
            file=sys.stderr,
        )
    if args.check and not report["rendered"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
