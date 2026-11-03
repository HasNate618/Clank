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
) -> float | None:
    """Depth at normalised `(u, v)`, or None when there is no usable surface there.

    `(u, v)` are normalised against the **depth** grid; a caller holding a colour pixel
    must scale it first. The centre texel seeds the answer, then neighbours within
    `tol_m` of it are collected and the median returned. Neighbours further away are a
    different surface - typically the background behind a thin object - and are dropped
    rather than averaged in.

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
    scale: float = 1.0,
) -> tuple[float, float, float] | None:
    """The root-relative point under normalised `(u, v)`, or None.

    `frame` is a dict from `clank.depth_stream` (`depth`, `width`, `height`). The depth it
    carries is the distance along the view axis, which is exactly what `unproject` takes;
    passing a straight-line range instead would bend every off-axis point outward, most
    at the frame edge where a reach is most likely to be aimed.

    `scale` is the emulated-player scale that `clank.wire.body_state_to_xr` applies to the
    poses it sends the game, and it has to be undone here or every distance comes back
    `scale` times too large. The stream measures **game space**; the game's camera sits at
    `scale` times the robot's head, so if `d` is the depth along the gaze then

    .. code-block:: text

        d_game = scale * d_robot        ->        d_robot = d_game / scale

    and the ray direction is unchanged, because a uniform scale does not rotate anything.
    At the default of 1.0 the two frames are the same and this is a no-op; with the scale
    enabled it is the difference between a 0.34 m reach and a 0.49 m one - and 0.49 m is
    outside the arm, which is what made the body step.

    Returns None when the depth has no answer there, so a caller cannot silently place a
    hand on a guess.
    """
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {scale!r}")
    width = int(frame["width"])
    height = int(frame["height"])
    depth = sample_depth(
        frame["depth"], width, height, u, v, patch=patch, tol_m=tol_m
    )
    if depth is None:
        return None
    return unproject((float(u), float(v)), head, fov, depth / scale)
