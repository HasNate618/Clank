#!/usr/bin/env python3
"""How a hand travelled, as numbers instead of an impression.

The reach this replaces was rejected for its *shape* - it stuttered, stopped, and corrected in
steps - while every gate measured only where it ended up. A residual cannot see the difference
between a clean arc and a zig-zag that arrives, so path shape gets its own measurements here.

Definitions are the ones already on record, so a new number can be compared to an old one:

* **Reversals** - the projection of the path onto the start-to-end axis, counting sign changes
  of steps above a deadband. `reach_smooth` and `move_path` both count this way, and the cruise
  figures this is compared against (2 reversals for a smooth path, 4 with two corrections,
  13-27 for the hand-written sweep) are that definition.
* **Straightness** - path length over net displacement, i.e. how much of the travel was
  wasted. 1.0 is a straight line; 2.0 is out and back.

Pure and standard-library only: the points come from a probe either way.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

#: Steps shorter than this are noise, not intent - the same 4 mm `reach_smooth` uses.
DEADBAND_M = 0.004


def path_shape(
    points: Sequence[Sequence[float]], *, deadband_m: float = DEADBAND_M
) -> dict:
    """The shape of a hand path: length, net travel, straightness and reversals.

    `points` are positions in one frame, in time order. A path shorter than two points has no
    shape, so it reports zeros with `straightness` 1.0 rather than raising: a body that never
    moved during a clip is a real observation, not an error.
    """
    path = [tuple(float(v) for v in point) for point in points]
    if len(path) < 2:
        return {
            "samples": len(path),
            "path_m": 0.0,
            "net_m": 0.0,
            "straightness": 1.0,
            "reversals": 0,
        }
    length = math.fsum(math.dist(a, b) for a, b in itertools.pairwise(path))
    net = math.dist(path[0], path[-1])
    return {
        "samples": len(path),
        "path_m": round(length, 4),
        "net_m": round(net, 4),
        "straightness": round(net / length, 3) if length > 0.0 else 1.0,
        "reversals": reversals(path, deadband_m=deadband_m),
    }


def reversals(path: Sequence[Sequence[float]], *, deadband_m: float = DEADBAND_M) -> int:
    """Direction changes along the path, measured on one axis.

    Projecting onto a single axis is what makes this comparable to `reach_smooth`'s count: a hand
    that arcs cleanly forward has a monotone projection even while it moves sideways, so only a
    hand that stops and comes back registers. The axis is the net displacement when there is one,
    and the widest-spread axis when the path returns to where it started - which is exactly the
    stuttering case, so it cannot be the one case that reports nothing.
    """
    if len(path) < 3:
        return 0
    unit = motion_axis(path)
    projected = [
        math.fsum((point[k] - path[0][k]) * unit[k] for k in range(3)) for point in path
    ]
    count = 0
    sign = 0
    for near, far in itertools.pairwise(projected):
        step = far - near
        if abs(step) < deadband_m:
            continue
        direction = 1 if step > 0.0 else -1
        if sign and direction != sign:
            count += 1
        sign = direction
    return count


def motion_axis(path: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    """A unit axis to measure progress along: net travel, or the widest spread if none."""
    net = [path[-1][k] - path[0][k] for k in range(3)]
    length = math.sqrt(math.fsum(component * component for component in net))
    if length > 0.0:
        return (net[0] / length, net[1] / length, net[2] / length)
    spread = [
        max(point[k] for point in path) - min(point[k] for point in path) for k in range(3)
    ]
    across = spread.index(max(spread))
    unit = [0.0, 0.0, 0.0]
    unit[across] = 1.0
    return (unit[0], unit[1], unit[2])
