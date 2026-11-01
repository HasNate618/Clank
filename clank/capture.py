"""One game frame, with no browser and no password in the path.

The game renders inside Weston's VNC output, so the framebuffer is only reachable
through the compositor. Weston 16 ships `weston-screenshooter`, a native Wayland
client that asks the compositor to copy the output; it needs no VNC credential and
no human. Measured 2026-11-01 on the live Job Simulator session: 1920x1080 PNG,
1.4 MB, ~0.4 s per shot with the binary on PATH.

Two traps, both found the hard way, both handled here:

1. **The tool ignores its output argument.** Whatever path you pass, it writes
   `wayland-screenshot-<timestamp>.png` into its *current working directory*. So the
   shot is taken in a fresh directory and the single PNG that appears there is the
   result. Passing a path and then looking for that path finds nothing even though
   the tool exited 0 - which is exactly how this path was once declared broken.
2. **The pixels are vertically mirrored.** The live compositor logs
   `screen capture uses y-flip: yes` (GL renderer, VNC backend), and its PNGs are
   upside down: readable signs appear mirrored across the horizontal axis, not
   rotated. The VNC client path is upright, so a frame from this module must be
   flipped vertically before it is compared with, or calibrated against, a frame
   from the dashboard's canvas. `spike/wbc_game/capture_game.py` does that flip.
   `clank/` stays standard-library only and therefore cannot decode the PNG; it
   hands back the raw file and names the orientation here.

This is deliberately a one-shot primitive, not a loop. `clank.preview.PreviewCapture`
is the loop/crop wrapper and now calls this function for each shot; a measurement
wants one frame and an exception when the frame did not arrive.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

#: The 8-byte signature every PNG starts with. A screenshooter that exits 0 but
#: leaves an empty or truncated file would otherwise look like a success.
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class CaptureError(RuntimeError):
    """The compositor produced no usable frame."""


def _command_error(label: str, result: subprocess.CompletedProcess) -> str:
    stderr = result.stderr
    if isinstance(stderr, bytes):
        detail = stderr.decode(errors="replace")
    else:
        detail = str(stderr or "")
    detail = detail.strip()[-400:]
    if detail:
        return f"{label} exited {result.returncode}: {detail}"
    return f"{label} exited {result.returncode}"


def _newest_png(directory: Path) -> Path | None:
    shots: list[tuple[int, Path]] = []
    for path in directory.glob("*.png"):
        try:
            shots.append((path.stat().st_mtime_ns, path))
        except OSError:
            continue
    if not shots:
        return None
    return max(shots, key=lambda item: item[0])[1]


def capture_frame(
    *,
    screenshooter: str,
    runtime_dir: str | Path,
    wayland_display: str,
    work_dir: str | Path,
    destination: str | Path,
    timeout: float = 5.0,
    run: Callable = subprocess.run,
) -> Path:
    """Capture one frame from `wayland_display` and publish it to `destination`.

    Returns the published path. Raises `CaptureError` when the screenshooter fails,
    writes nothing, or writes something that is not a PNG. The destination is only
    replaced on success, so a failed shot leaves the previous frame intact.

    `run` is injectable the same way `PreviewCapture.run` is, which is how the unit
    tests exercise the whole path with no compositor and no session.

    The `screenshooter` argument is a path or a name resolved on PATH; the
    orchestrator's `weston_screenshooter_bin` config supplies it.
    """
    source_dir = Path(work_dir)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    # A fresh directory is what makes "did this run write a file?" answerable. It
    # also neutralises the timestamped name: there is exactly one shot in here.
    frame_dir = Path(tempfile.mkdtemp(prefix="clank-capture-", dir=str(source_dir)))
    try:
        env = os.environ.copy()
        env["XDG_RUNTIME_DIR"] = str(runtime_dir)
        env["WAYLAND_DISPLAY"] = wayland_display
        # WAYLAND_SOCKET outranks WAYLAND_DISPLAY in libwayland. An inherited
        # socket would silently capture a different compositor than the one asked
        # for, which is worse than failing.
        env.pop("WAYLAND_SOCKET", None)

        result = run(
            [screenshooter],
            cwd=str(frame_dir),
            env=env,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise CaptureError(_command_error("screenshooter", result))

        shot = _newest_png(frame_dir)
        if shot is None:
            raise CaptureError(
                "screenshooter exited 0 but wrote no wayland-screenshot-*.png"
            )
        try:
            head = shot.read_bytes()[: len(PNG_SIGNATURE)]
        except OSError as exc:
            raise CaptureError(f"cannot read captured frame {shot.name}: {exc}") from exc
        if head != PNG_SIGNATURE:
            raise CaptureError(f"screenshooter wrote a non-PNG file: {shot.name}")

        os.replace(shot, target)
        return target
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)
