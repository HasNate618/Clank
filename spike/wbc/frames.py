#!/usr/bin/env python3
"""Frames on this side of the fence, so `clank/` can stay standard-library only.

`clank/framediff.py` and `clank/glow.py` do the arithmetic over nested lists of
`(r, g, b)` ints. Something has to turn a PNG into those lists, and Pillow is not
allowed inside `clank/`. So decoding lives here, in the spike, with the venv.

Capture goes through `clank.capture.capture_frame`, which knows the two traps found
the hard way: the tool ignores its output argument and writes a *timestamped*
filename into the current directory, and the game only appears when its window is
forward. Measured 2026-11-01: the compositor's PNG is **vertically mirrored**
(GL renderer reports `screen capture uses y-flip: yes`), so absolute screen
coordinates need `load_upright()` rather than `load()`. The dashboard/VNC canvas is
upright; do not mix the two orientations in one comparison.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from clank.capture import capture_frame  # noqa: E402


@dataclass(frozen=True)
class Frame:
    """Pixels as nested `[y][x] = (r, g, b)` ints, which is what the stdlib expects."""

    width: int
    height: int
    rows: list[list[tuple[int, int, int]]]

    def as_rows(self) -> list[list[tuple[int, int, int]]]:
        return self.rows


def load(path: str | Path) -> Frame:
    """Decode a PNG (or anything Pillow reads) into a `Frame`, in RGB.

    This preserves the file's orientation. A `weston-screenshooter` frame from the
    live VNC session is vertically mirrored; use `load_upright` before reading
    absolute screen coordinates from it.
    """
    from PIL import Image  # imported late so --help works without the venv

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        # `getdata()` is typed as an opaque ImagingCore in Pillow's stubs, so
        # `list(...)` reads as a protocol violation. `tobytes()` is plain bytes,
        # which indexes to ints and needs no suppression - and it is faster.
        raw = rgb.tobytes()
    stride = width * 3
    rows = [
        [(raw[i], raw[i + 1], raw[i + 2]) for i in range(y * stride, (y + 1) * stride, 3)]
        for y in range(height)
    ]
    return Frame(width=width, height=height, rows=rows)


def downscale(frame: Frame, step: int) -> Frame:
    """Box-average by an integer `step`, e.g. 4 turns 1920x1080 into 480x270.

    Averaging rather than nearest-neighbour because this feeds a difference against a
    noise baseline: averaging shrinks sensor and VNC grain, which is the thing the
    baseline is trying to characterise. The trailing remainder (if the size is not a
    multiple of `step`) is dropped, and the resulting size is reported so no caller
    has to guess it.
    """
    if step < 1:
        raise ValueError("step must be >= 1")
    if step == 1:
        return frame
    out: list[list[tuple[int, int, int]]] = []
    for y in range(0, frame.height - step + 1, step):
        row: list[tuple[int, int, int]] = []
        for x in range(0, frame.width - step + 1, step):
            r = g = b = 0
            for dy in range(step):
                source = frame.rows[y + dy]
                for dx in range(step):
                    pr, pg, pb = source[x + dx]
                    r += pr
                    g += pg
                    b += pb
            n = step * step
            row.append((r // n, g // n, b // n))
        out.append(row)
    width = len(out[0]) if out else 0
    return Frame(width=width, height=len(out), rows=out)


def load_upright(path: str | Path) -> Frame:
    """Decode a compositor screenshot and flip it to the game's real orientation.

    The live session's GL/VNC compositor writes captured frames mirrored across the
    horizontal axis (signs read upside down but left-to-right). Reversing the row
    order restores upright; tests cannot check this without a session, so the
    evidence is the readable signage in `research/capture.md`.
    """
    frame = load(path)
    return Frame(width=frame.width, height=frame.height, rows=list(reversed(frame.rows)))


def capture(
    *,
    screenshooter: str,
    runtime_dir: str | Path,
    wayland_display: str,
    work_dir: str | Path,
    destination: str | Path,
) -> Path:
    """Capture one frame and publish it to `destination`.

    Returns the published path. Raises `CaptureError` when the compositor produced
    no usable PNG, which is also how a dead compositor shows up. The published file
    is in the compositor's orientation (vertically mirrored); see `load_upright`.
    """
    return capture_frame(
        screenshooter=screenshooter,
        runtime_dir=runtime_dir,
        wayland_display=wayland_display,
        work_dir=work_dir,
        destination=destination,
    )
