"""Turn a pixel into a 3D point, using the live depth stream.

`clank/depth_stream.py` publishes linear metres from the same eye as its colour plane, so
the depth is measured rather than guessed and a marked pixel needs **one lookup**, not a
servo that hunts: `research/live-pixel-servo.md` records what a guessed depth cost, and
`research/g1-borrowable-code.md` §"Concrete borrow #2" states the rule this replaces -
without depth, only apparent size can give a range, and that needs a known object size.

Standard library only, like the rest of `clank/`. The sampling rule matters as much as the
arithmetic: a single texel is unreliable on an edge, and averaging across a discontinuity
invents a surface that is not there. The card is thin enough for that to be a real error,
so a patch is reduced to the **cluster around the centre texel**, and disagreement is
treated as "no answer" rather than averaged away.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from clank.body import Pose
from clank.projection import Fov, unproject

#: Default half-width of the sampling window, in texels, as a window side length.
DEFAULT_PATCH = 3
#: How far a neighbour may differ from the centre texel and still be the same surface.
DEFAULT_TOL_M = 0.10


def sample_depth(
    depth: Sequence[float],
    width: int,
    height: int,
    u: float,
    v: float,
    patch: int = DEFAULT_PATCH,
    tol_m: float = DEFAULT_TOL_M,
    nearest: bool = False,
    min_m: float = 0.0,
) -> float | None:
    """Depth at normalised `(u, v)`, or None when there is no usable surface there.

    `(u, v)` are normalised against the **depth** grid; a caller holding a colour pixel
    must scale it first. The centre texel seeds the answer, then neighbours within
    `tol_m` of it are collected and the median returned. Neighbours further away are a
    different surface - typically the background behind a thin object - and are dropped
    rather than averaged in.

    With `nearest=True` the **closest** surface in the patch wins instead of the median
    around the centre texel, and the tolerance does not filter anything out. That is the
    right question for a thin object: the JobBot card is a plane a few millimetres thick, so
    its pixel's centre texel lands on the wall behind it as often as on the card. Measured
    2026-11-08: the same card pixel unprojected to 0.635 m with the median (beyond the arm)
    and 0.50 m with the nearest (inside it), while a reach at the median saturated 120 mm
    short. The card is nearer than the wall; take the near thing.

    `min_m` is what keeps that honest: the robot's own hand and forearm are in the frame at
    about 0.2 m, so "nearest" without a floor answers "your own arm" - measured, a goal at
    z = -0.816, below the floor of the room. Surfaces nearer than `min_m` are skipped.

    Returns None for the stream's "nothing there" (0.0), for non-finite values, and for a
    pixel outside the unit square. None is a real answer and callers must not substitute
    a default depth for it.
    """
    if not (math.isfinite(u) and math.isfinite(v)):
        raise ValueError(f"pixel must be finite, got {(u, v)!r}")
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        raise ValueError(f"pixel must be inside [0, 1], got {(u, v)!r}")
    if width <= 0 or height <= 0:
        raise ValueError(f"bad depth grid {width}x{height}")
    if len(depth) < width * height:
        raise ValueError(
            f"depth holds {len(depth)} values, needs {width * height} for {width}x{height}"
        )

    cx = min(width - 1, max(0, int(u * width)))
    cy = min(height - 1, max(0, int(v * height)))
    ref = float(depth[cy * width + cx])
    if not math.isfinite(ref) or ref <= 0.0:
        return None

    radius = max(0, patch // 2)
    if nearest:
        nearest_value = ref if ref > min_m else math.inf
        for y in range(max(0, cy - radius), min(height, cy + radius + 1)):
            base = y * width
            for x in range(max(0, cx - radius), min(width, cx + radius + 1)):
                value = float(depth[base + x])
                if math.isfinite(value) and min_m < value < nearest_value:
                    nearest_value = value
        return nearest_value

    near: list[float] = []
    for y in range(max(0, cy - radius), min(height, cy + radius + 1)):
        base = y * width
        for x in range(max(0, cx - radius), min(width, cx + radius + 1)):
            value = float(depth[base + x])
            if math.isfinite(value) and value > 0.0 and abs(value - ref) <= tol_m:
                near.append(value)
    if not near:
        return None
    near.sort()
    return near[len(near) // 2]


def point_from_pixel(
    frame,
    u: float,
    v: float,
    head: Pose,
    fov: Fov,
    patch: int = DEFAULT_PATCH,
    tol_m: float = DEFAULT_TOL_M,
    head_scale: float = 1.0,
    nearest: bool = False,
    min_m: float = 0.0,
) -> tuple[float, float, float] | None:
    """The root-relative point under normalised `(u, v)`, or None.

    `frame` is a dict from `clank.depth_stream` (`depth`, `width`, `height`). The depth it
    carries is the distance along the view axis, which is exactly what `unproject` takes;
    passing a straight-line range instead would bend every off-axis point outward, most
    at the frame edge where a reach is most likely to be aimed.

    `head_scale` is the scale `clank.wire.body_state_to_xr` applied to the **head**, and
    only the head. When the camera itself is enlarged, the stream measures game space and
    every distance comes back `head_scale` times too large, so dividing is required. When
    only the hands are scaled (`player_scale_head=False`) the camera sits exactly where the
    robot's head does and the stream is already the robot frame - dividing there would pull
    every target 1.43x too close and make an arm that can reach look short.

    This parameter used to be called `scale` and was passed the *player* scale, which is
    the same number in one configuration and a silent 1.43x error in the other. It is
    named for what it actually undoes.

    Returns None when the depth has no answer there, so a caller cannot silently place a
    hand on a guess.
    """
    if not math.isfinite(head_scale) or head_scale <= 0.0:
        raise ValueError(f"head_scale must be finite and positive, got {head_scale!r}")
    width = int(frame["width"])
    height = int(frame["height"])
    depth = sample_depth(
        frame["depth"], width, height, u, v, patch=patch, tol_m=tol_m, nearest=nearest,
        min_m=min_m,
    )
    if depth is None or not math.isfinite(depth):
        return None
    return unproject((float(u), float(v)), head, fov, depth / head_scale)
