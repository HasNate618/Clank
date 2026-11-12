"""Turn a live run into one watchable file: the pure parts.

A run produces three encoded parts: the game's picture, the robot's picture,
and the game's audio. This module holds what can be decided about turning them
into a single MP4 without touching the disk — the argv for one ffmpeg pass and
the HTTP ``Range`` parser the dashboard's video route answers with. Both are
pure, so the recorder and the composite that run them are tested around them.
"""

from __future__ import annotations

from pathlib import Path

GAME_WIDTH = 960
GAME_HEIGHT = 540
ROBOT_WIDTH = 720
HEIGHT = GAME_HEIGHT
WIDTH = GAME_WIDTH + ROBOT_WIDTH


def composite_argv(
    *,
    game: Path,
    robot: Path | None,
    audio: Path | None,
    out: Path,
    fps: int,
    crf: int,
    preset: str,
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    """The ffmpeg pass that stacks the parts side by side into `out`.

    A part that is missing is black padding of the same size, not a narrower
    file: the layout is fixed so every recording looks the same to a player.
    """
    inputs = ["-i", str(game)]
    input_count = 1
    audio_index: int | None = None
    if robot is not None:
        inputs += ["-i", str(robot)]
        input_count += 1
    if audio is not None:
        audio_index = input_count
        inputs += ["-i", str(audio)]

    left = f"[0:v]scale={GAME_WIDTH}:{GAME_HEIGHT}[g]"
    if robot is None:
        # An endless black pad, so the game's picture decides where the stack ends.
        right = f"color=black:s={ROBOT_WIDTH}x{HEIGHT}:r={fps}[r]"
        stack = "[g][r]hstack=inputs=2:shortest=1[v]"
    else:
        right = f"[1:v]scale={ROBOT_WIDTH}:{HEIGHT}[r]"
        stack = "[g][r]hstack=inputs=2[v]"

    argv = [
        ffmpeg, "-y", *inputs,
        "-filter_complex", ";".join([left, right, stack]),
        "-map", "[v]",
    ]
    if audio_index is not None:
        argv += ["-map", f"{audio_index}:a"]
    argv += [
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-r", str(fps),
    ]
    if audio_index is not None:
        argv += ["-c:a", "aac", "-b:a", "96k", "-shortest"]
    argv += ["-movflags", "+faststart", str(out)]
    return argv


def parse_range(value: str, size: int) -> tuple[int, int] | None:
    """Inclusive (start, end) for a single ``bytes=`` range, or None for all.

    None means the header names no range this module serves as one: it is
    absent, in another unit, or a multi-range, and answering with the whole
    file is always legal. A malformed or unsatisfiable range raises ValueError
    so the caller can answer 416 rather than guess at what was meant.
    """
    value = value.strip()
    if not value.startswith("bytes="):
        return None
    spec = value[len("bytes="):].strip()
    if "," in spec:
        return None
    first, dash, last = spec.partition("-")
    if not dash:
        raise ValueError(f"not a byte range: {value!r}")

    if not first:
        if not last.isdigit():
            raise ValueError(f"unparsable suffix length: {value!r}")
        suffix = int(last)
        if suffix == 0 or size <= 0:
            raise ValueError(f"unsatisfiable range: {value!r}")
        return (max(size - suffix, 0), size - 1)

    if not first.isdigit() or (last and not last.isdigit()):
        raise ValueError(f"unparsable range: {value!r}")
    start = int(first)
    end = int(last) if last else size - 1
    if start >= size:
        raise ValueError(f"unsatisfiable range: {value!r}")
    if end < start:
        raise ValueError(f"empty range: {value!r}")
    return (start, min(end, size - 1))
