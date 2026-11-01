"""Frame sources for the brain.

The brain sees **one still per turn**. That much is common to the live game and to
a recording. What differs is where the still comes from, and this module keeps
that difference behind one interface so the brain itself never learns which it is.

    FrameSource.frame(index) -> Frame

Why a *stepped* source rather than a video player (the discontinuity the operator
flagged): in the live session the robot freezes its head while the model thinks,
so the frame it reasoned about is still the frame it grounds on when the answer
arrives. A recording has no head to freeze, so the harness does the equivalent
thing at the source: the clip is decomposed into stills once, and time only
advances when the caller asks for the next index. Nothing ever plays past a
thought.

`ReplaySource` is a development fixture. It shells out to `ffmpeg` to decompose a
clip, which is a binary, not a Python dependency: `clank` still imports nothing
outside the standard library, and the live path below needs no ffmpeg at all.

`crop_jpeg` is the same trade for the container crop: GROUND is asked about a
crop, so somebody has to cut one, and ffmpeg does it on stdin/stdout without the
brain learning an image library.
"""

from __future__ import annotations

import base64
import pathlib
import subprocess
from dataclasses import dataclass

#: The width the replay stills are extracted at. The grounder consumes a resized
#: still, not grid cells, so this is a detail/bandwidth trade: wide enough that a
#: container crop still resolves, small enough that a turn's base64 stays a few
#: hundred kilobytes. `crop_jpeg` picks its own width.
DEFAULT_WIDTH = 960


def _frac(value: float) -> str:
    """A normalised crop fraction as ffmpeg reads it, without trailing zero noise."""
    return f"{round(value, 3):g}"


@dataclass(frozen=True)
class Frame:
    """One still, plus enough provenance to cite it in a trace."""

    index: int
    t_s: float
    jpeg: bytes
    source: str

    def data_url(self) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(self.jpeg).decode("ascii")


class FrameSource:
    """What the brain is allowed to assume about the world it is looking at."""

    name: str = "source"

    def __len__(self) -> int:  # pragma: no cover - protocol
        raise NotImplementedError

    def frame(self, index: int) -> Frame:  # pragma: no cover - protocol
        raise NotImplementedError


class ReplaySource(FrameSource):
    """A clip decomposed into stills, advancing only on request.

    `prepare()` is idempotent: it extracts once into `cache_dir` and reuses it, so
    stepping back and forth is a file read rather than another decode. That
    matters because the dashboard lets a human scrub the same clip while the brain
    is mid-call.
    """

    name = "replay"

    def __init__(
        self,
        video: pathlib.Path | str,
        cache_dir: pathlib.Path | str,
        fps: float = 1.0,
        width: int = DEFAULT_WIDTH,
    ) -> None:
        self.video = pathlib.Path(video)
        self.cache_dir = pathlib.Path(cache_dir)
        self.fps = fps
        self.width = width
        self._paths: list[pathlib.Path] = []

    def prepare(self) -> int:
        """Extract every still once. Returns the number of frames available."""
        if self._paths:
            return len(self._paths)
        if not self.video.is_file():
            raise FileNotFoundError(f"no such clip: {self.video}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.cache_dir.glob("f_*.jpg"))
        if not existing:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(self.video),
                    "-vf", f"fps={self.fps},scale={self.width}:-1",
                    str(self.cache_dir / "f_%05d.jpg"),
                ],
                check=True,
            )
            existing = sorted(self.cache_dir.glob("f_*.jpg"))
        if not existing:
            raise RuntimeError(f"ffmpeg produced no frames from {self.video}")
        self._paths = existing
        return len(existing)

    def __len__(self) -> int:
        return len(self._paths) if self._paths else self.prepare()

    def frame(self, index: int) -> Frame:
        n = len(self)
        if not 0 <= index < n:
            raise IndexError(f"frame {index} outside 0..{n - 1}")
        path = self._paths[index]
        # The clip is decomposed at a known rate, so wall time is index / fps.
        return Frame(
            index=index,
            t_s=index / self.fps,
            jpeg=path.read_bytes(),
            source=f"replay:{self.video.name}",
        )


def crop_jpeg(
    jpeg: bytes,
    box: tuple[float, float, float, float],
    width: int = 1024,
) -> bytes:
    """Cut `box` out of a JPEG and resize it to `width`, through ffmpeg.

    Spec §10. GROUND is asked about a crop of the container, not the whole frame:
    measured 2026-11-01, the identical name->quadrant question scored 44/44 on a
    tray crop and 9/12 on the full frame, and accuracy tracks how much of the
    frame the container fills. `box` is normalised `(u0, v0, u1, v1)` - the same
    tuple `brain.ground()` is handed, so the answer is composed back into
    full-frame coordinates rather than crop coordinates.

    The still goes in on stdin and the crop comes back on stdout, so no temporary
    file is written and nothing in `clank` learns ffmpeg's output format. ffmpeg is
    a binary, not a Python dependency: this module still imports nothing outside
    the standard library.

    A degenerate box is rejected before ffmpeg runs, and an ffmpeg failure or an
    empty crop is a `RuntimeError` carrying ffmpeg's own message, never a bare
    `subprocess` error: `run_turn` catches it and grounds the whole frame instead,
    so a broken crop has to arrive as one catchable error rather than two.
    """
    u0, v0, u1, v1 = box
    if u0 >= u1 or v0 >= v1:
        raise RuntimeError(f"crop box {box} is degenerate: need u0 < u1 and v0 < v1")
    vf = (
        f"crop=iw*{_frac(u1 - u0)}:ih*{_frac(v1 - v0)}"
        f":iw*{_frac(u0)}:ih*{_frac(v0)},scale={width}:-1"
    )
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", "pipe:0",
                "-vf", vf,
                "-f", "mjpeg", "pipe:1",
            ],
            input=jpeg,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg could not crop {box}: {detail}") from exc
    except OSError as exc:
        raise RuntimeError(f"ffmpeg could not run for crop {box}: {exc}") from exc
    stderr_text = result.stderr.decode("utf-8", "replace").strip()
    if not result.stdout:
        raise RuntimeError(f"ffmpeg produced no crop for {box}: {stderr_text}")
    return result.stdout


class VncSource(FrameSource):
    """Live Weston framebuffer. Not wired yet - it needs the running session.

    The RFB client already exists and is standard library only at
    `spike/vnc_grab.py`. Promoting it here is a separate change because it can
    only be exercised against a live Weston, and this module is exercised against
    a fixture. Guessing at an untested socket protocol would put a plausible
    failure in the same file as the tested path.
    """

    name = "vnc"

    def __len__(self) -> int:
        raise NotImplementedError("VncSource is not wired yet")

    def frame(self, index: int) -> Frame:
        raise NotImplementedError("VncSource is not wired yet")
