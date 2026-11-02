"""The measured right-wrist workspace, and the clamp that keeps the servo inside it.

`clank.projection.servo_step` is deliberately stateless: it knows nothing about joint
limits or reach, so that it does not become a second, drifting copy of those rules.
The cost of that design is that it will happily command a point the arm cannot occupy,
and then the loop's error grows instead of shrinking. That is not a hypothesis - it is
what the live body did (`research/live-pixel-servo.md`): one target settled to 0.09 of
a frame, another oscillated between 0.16 and 0.36 while asking for
`forward -0.50, lateral +0.64`, a point behind the torso.

So the missing piece is this module: one measured box, in root-relative metres, that
the adapter clamps against before it ever speaks to the body.

## Provenance

Every bound is a measurement. `[MEASURED 2026-11-01]`

* `x` (forward). `research/sonic-reach-envelope.md` section 1. The standing rest is
  0.079 m and the first four reach rows track to 0.587 m at residual 0.184. Rows 0.60
  and 0.70 are excluded: they lift and centre the wrist rather than press forward, and
  their residual is 0.32-0.38 m - "the wrist translates to x ~ 0.70 m and stops", not
  a positioned wrist. The floor is the **standing rest**, not the first reach row,
  because the box has to contain the idle pose: a floor of 0.20 would make the adapter
  clamp its own rest forward and the arm would never stand still.
* `y` (lateral, + is the robot's left). `spike/wbc_game/side_envelope_sonic.py`. Swept
  `lateral_m` 0.00 .. 0.60 at `forward_m` 0.45: achieved y ran -0.280 .. +0.419, with
  the residual growing 0.087 -> 0.187. No fall risk at any target.
* `z` (up). Same probe, `up_m` -0.10 .. 0.60 at `forward_m` 0.45, `lateral_m` 0.20:
  achieved z -0.149 .. +0.449, residual 0.119 .. 0.214. The standing rest's -0.228
  sets the floor.

## What the box does and does not promise

It is a **stability** limit, not a guarantee of arrival. Residuals inside it run
0.05-0.19 m, and `FLAG_SATURATED` is set on essentially every target at the contract's
0.02 m tolerance, so "inside the box" means "the arm goes in the right direction",
never "the hand got there".

It is also not a full 3D characterisation. Each sweep held the other two axes fixed,
so the corners of the box - say `(0.60, 0.42, 0.45)` - were never visited. The box is
the axis-aligned envelope of three one-dimensional sweeps, which is the honest reading
of the evidence, and no more.
"""

import math
from dataclasses import dataclass

__all__ = ["Box", "WRIST_BOX", "box_bounds", "clamp_wrist", "in_workspace"]


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in the root frame: +X forward, +Y robot-left, +Z up."""

    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    def bounds(self) -> tuple[tuple[float, float], ...]:
        return (self.x, self.y, self.z)


#: Measured trackable right-wrist workspace, metres, root-relative. See provenance.
WRIST_BOX = Box(
    x=(0.079, 0.60),
    y=(-0.31, 0.42),
    z=(-0.23, 0.45),
)


def box_bounds() -> tuple[tuple[float, float], ...]:
    """`((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi))`. The adapter clamps to these."""
    return WRIST_BOX.bounds()


def _validated(point: tuple[float, ...]) -> tuple[float, float, float]:
    if len(point) != 3:
        raise ValueError(f"a wrist point has three coordinates, got {len(point)}")
    x, y, z = point
    for v in (x, y, z):
        if not math.isfinite(v):
            raise ValueError(f"wrist coordinate must be finite, got {v!r}")
    return x, y, z


def clamp_wrist(point: tuple[float, ...]) -> tuple[float, float, float]:
    """Pull `point` into the measured workspace, moving only what violates it."""
    x, y, z = _validated(point)
    out = []
    for v, (lo, hi) in zip((x, y, z), WRIST_BOX.bounds()):
        out.append(min(max(v, lo), hi))
    return out[0], out[1], out[2]


def in_workspace(point: tuple[float, ...]) -> bool:
    """Is `point` inside the measured trackable box? Agreement with `clamp_wrist` is
    guaranteed and tested."""
    return clamp_wrist(point) == _validated(point)
