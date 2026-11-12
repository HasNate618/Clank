"""How big the emulated player is, and where that size is measured from.

`clank/pose_relay.py` can size the player it reports to the game without touching the
simulated body: what it sends is `pivot + scale * (point - pivot)`. The game then lays its
adult-sized room out around a taller, longer-armed player, and the G1's own physics stays
exactly as the policy was trained for it.

Size is invisible, so it has to be *declared*. Every number derived from the depth stream
is in game space, and a caller that forgets to divide comes back 1.43x too far and aims
past the object - which reads as a grounding failure rather than a units one. The relay
writes what it used beside its own state file, and anything that unprojects a pixel reads
it back instead of assuming the default.

Standard library only, like the rest of `clank/`.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

#: Where the relay leaves the declaration, beside `body_state.json`.
FILENAME = "player_scale.json"

#: The state directory the spike helpers already hardcode.
DEFAULT_DIR = Path.home() / ".local" / "state" / "Clank" / "sessions" / "body"


@dataclass(frozen=True)
class PlayerScale:
    """The scale the relay is reporting the player at, and the pivot it is measured from.

    `head_scaled` is separate from `scale` because the two configurations are not the
    same measurement. With the head scaled, the game's camera really is taller, so the
    stream is game space and both the unprojection and the target need the scale. With
    the head left alone, the camera is where every measurement in `research/` was taken,
    so only the hands need it - see `research/g1-scale.md`.
    """

    scale: float = 1.0
    pivot: tuple[float, float, float] = (0.0, 0.0, 0.0)
    head_scaled: bool = True
    yaw_offset_deg: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError(f"scale must be finite and positive, got {self.scale!r}")
        pivot = tuple(float(c) for c in self.pivot)
        if len(pivot) != 3 or not all(math.isfinite(c) for c in pivot):
            raise ValueError(f"pivot must be three finite numbers, got {self.pivot!r}")
        object.__setattr__(self, "pivot", (pivot[0], pivot[1], pivot[2]))

    @property
    def camera_scale(self) -> float:
        """What to compose onto the camera before unprojecting; 1.0 when it is unscaled."""
        return self.scale if self.head_scaled else 1.0

    def pixel_point(self, frame, u: float, v: float, head, fov, **kwargs):
        """`clank.pixel3d.point_from_pixel` with this declaration applied.

        One call so a caller cannot apply the camera half and forget the hand half: that
        omission is exactly the 1.43x error this module exists to prevent.
        """
        from clank.pixel3d import point_from_pixel

        return point_from_pixel(
            frame, u, v, head, fov,
            head_scale=self.camera_scale, pivot=self.pivot, hands_scale=self.scale,
            **kwargs,
        )

    def as_json(self) -> dict:
        return {
            "scale": self.scale,
            "pivot": list(self.pivot),
            "head_scaled": self.head_scaled,
            "yaw_offset_deg": self.yaw_offset_deg,
        }


def path(state_dir: os.PathLike | str | None = None) -> Path:
    """The declaration's location, beside whatever state file the relay is using."""
    return Path(DEFAULT_DIR if state_dir is None else state_dir) / FILENAME


def read(state_dir: os.PathLike | str | None = None) -> PlayerScale:
    """What the relay declared, or the unscaled default when it declared nothing.

    A *malformed* file raises rather than falling back: assuming 1.0 when the game was
    told 1.43 would aim every reach 1.43x too far, and a silent wrong answer is worse
    than a stopped one. An absent file is different - that is a session that never
    declared a scale, and 1.0 is then the truth.
    """
    where = path(state_dir)
    try:
        raw = where.read_text()
    except FileNotFoundError:
        return PlayerScale()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{where} must hold an object, got {type(data).__name__}")
    try:
        wanted = [float(c) for c in data.get("pivot", (0.0, 0.0, 0.0))]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: bad pivot {data.get('pivot')!r}") from exc
    if len(wanted) != 3:
        raise ValueError(f"{where}: pivot must be three numbers, got {wanted!r}")
    pivot = (wanted[0], wanted[1], wanted[2])
    return PlayerScale(
        scale=float(data.get("scale", 1.0)),
        pivot=pivot,
        head_scaled=bool(data.get("head_scaled", True)),
        yaw_offset_deg=float(data.get("yaw_offset_deg", 0.0)),
    )


def write(value: PlayerScale, state_dir: os.PathLike | str | None = None) -> Path:
    """Record what the relay is using, atomically so a reader never sees half a file."""
    where = path(state_dir)
    where.parent.mkdir(parents=True, exist_ok=True)
    scratch = where.with_suffix(".json.tmp")
    scratch.write_text(json.dumps(value.as_json(), indent=2) + "\n")
    scratch.replace(where)
    return where


def apply(point: Sequence[float], value: PlayerScale) -> tuple[float, float, float]:
    """A game-space point as a robot-frame one: `pivot + (point - pivot) / scale`."""
    pivot = value.pivot
    return (
        pivot[0] + (float(point[0]) - pivot[0]) / value.scale,
        pivot[1] + (float(point[1]) - pivot[1]) / value.scale,
        pivot[2] + (float(point[2]) - pivot[2]) / value.scale,
    )
