#!/usr/bin/env python3
"""Turn a generated pose36 clip into the 50 Hz motion arrays SONIC's encoder reads.

Two things happen here, and both have a silent failure mode that reaches the policy.

**Upsample.** The generator runs at 25 Hz (ARDY-G1-RP-25FPS-Horizon52) and SONIC runs at
50 Hz, so half the rows don't exist. Interpolation of the hinges and the root position is
honest, but the root orientation must slerp: a lerped quaternion is not a unit quaternion,
and SONIC's observation carries per-frame joint velocities built from these rows, so the
artefact arrives as motion nobody asked for. Slerp is the shortest arc (a negated endpoint
is the same rotation).

**Flat hold.** SONIC's encoder reads a 10-frame lookahead at stride 5 - 0.9 s ahead at
50 Hz - and clamps it to the last frame. A clip that simply ends therefore leaves every one
of the ten lookahead slots on the final rows, and an index that has run past the end clamps
there anyway; the measured hold that does not walk is to repeat the last row in all ten
slots (`research/c3m-results.md`, H1f), which is what `lookahead_idx` with `flattened=True`
does. `clip_tick` decides when that starts.

The `jp` array is the easy one to get wrong: SONIC is in policy/CSV order, an interleaved
left/right permutation, so the conversion is `g1_pack.jp_from_pose36`, never index math
re-derived here. Pelvis row 0 is the only body that carries anything: the root xyz, and the
root quaternion in MuJoCo `wxyz` order.

Standard library only: `tests/` imports this module and must stay stdlib. The values are
plain lists so the adapter can wrap them in NumPy when it writes the npz.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from pathlib import Path

# The sibling modules are imported as top-level names from their own directory, the same
# way every other spike under spike/wbc/ reaches them.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import g1_pack  # noqa: E402

#: ARDY-G1-Horizon52 generates at 25 fps; the earlier MotionBricks clips were 30.
SRC_HZ = 25
#: The tracking policy's tick rate, and the rate its encoder observations are built at.
DST_HZ = 50
#: Lookahead slots the encoder reads, and the stride between them.
LOOKAHEAD = 10
LOOKAHEAD_STEP = 5

#: Joints in a `jp`/`jv` row: the 29 hinges in policy/CSV order.
JOINTS = len(g1_pack.CSV2MJ)
#: Bodies in SONIC's `bp`/`bq` arrays; only the pelvis (index 0) is filled from a pose36.
BODIES = 14
#: The pelvis' index in those arrays.
BODY_PELVIS = 0


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation; exact at both ends."""
    return a + (b - a) * t


def slerp_xyzw(
    q0: Sequence[float], q1: Sequence[float], t: float
) -> tuple[float, float, float, float]:
    """Shortest-arc interpolation between two `xyzw` quaternions, normalised.

    A negated endpoint is the same rotation, so the dot product picks the arc that is at
    most half a turn; the blend is then normalised back onto the unit sphere.
    """
    a = tuple(float(v) for v in q0)
    b = tuple(float(v) for v in q1)
    if sum(a[i] * b[i] for i in range(4)) < 0.0:
        b = tuple(-v for v in b)
    if t <= 0.0:
        return (a[0], a[1], a[2], a[3])
    if t >= 1.0:
        return (b[0], b[1], b[2], b[3])
    mixed = [lerp(a[i], b[i], t) for i in range(4)]
    norm = math.sqrt(sum(v * v for v in mixed))
    return (mixed[0] / norm, mixed[1] / norm, mixed[2] / norm, mixed[3] / norm)


def _blend(a: Sequence[float], b: Sequence[float], t: float) -> tuple[float, ...]:
    """One pose between two, slerping the root quaternion and lerping everything else."""
    if t == 0.0 or tuple(a) == tuple(b):
        return tuple(a)
    return (
        tuple(lerp(a[k], b[k], t) for k in range(3))
        + slerp_xyzw(a[3:7], b[3:7], t)
        + tuple(lerp(a[k], b[k], t) for k in range(7, len(a)))
    )


def resample_pose36(
    frames: Sequence[Sequence[float]],
    src_hz: float = SRC_HZ,
    dst_hz: float = DST_HZ,
) -> list[tuple[float, ...]]:
    """Resample a pose36 clip from `src_hz` to `dst_hz`.

    A clip of `n` frames at `src_hz` lasts `n / src_hz` seconds, so the output count is
    `round(n * dst_hz / src_hz)`; 25 frames of ARDY become 50 rows at 50 Hz. Output row `j`
    sits at `j / dst_hz` in the clip, clamped to the last input row - the tail is a hold,
    not a read past the end.
    """
    if not frames:
        raise ValueError("clip is empty")
    n = len(frames)
    count = max(1, round(n * dst_hz / src_hz))
    out: list[tuple[float, ...]] = []
    for j in range(count):
        pos = min(j * src_hz / dst_hz, float(n - 1))
        i = min(int(pos), n - 1)
        out.append(_blend(frames[i], frames[i + 1] if i + 1 < n else frames[i], pos - i))
    return out


def sonic_dict(frames50: Sequence[Sequence[float]]) -> dict:
    """The clip as SONIC's motion arrays: `jp`, `jv`, `bp`, `bq`, `n`.

    `jv` is a backwards finite difference of `jp` scaled by `DST_HZ`, with the last row
    copied because there is no row after it. `bp`/`bq` are the released 14-body arrays with
    every body but the pelvis left at the origin and the identity (`wxyz`).
    """
    if not frames50:
        raise ValueError("clip is empty")
    n = len(frames50)
    jp = [list(g1_pack.jp_from_pose36(pose)) for pose in frames50]
    jv: list[list[float]] = []
    for i in range(n):
        if i + 1 < n:
            jv.append([(jp[i + 1][k] - jp[i][k]) * DST_HZ for k in range(JOINTS)])
        else:
            jv.append(list(jv[-1]) if jv else [0.0] * JOINTS)
    bp = [[[0.0, 0.0, 0.0] for _ in range(BODIES)] for _ in range(n)]
    bq = [[[1.0, 0.0, 0.0, 0.0] for _ in range(BODIES)] for _ in range(n)]
    for i, pose in enumerate(frames50):
        bp[i][BODY_PELVIS] = [float(pose[0]), float(pose[1]), float(pose[2])]
        bq[i][BODY_PELVIS] = list(g1_pack.xyzw_to_wxyz(pose[3:7]))
    return {"jp": jp, "jv": jv, "bp": bp, "bq": bq, "n": n}


def lookahead_idx(frame: int, n: int, flattened: bool) -> tuple[int, ...]:
    """The ten clip frames the encoder reads from `frame`.

    `flattened` is the end-of-clip hold: all ten slots are the last row, so the lookahead
    holds still instead of promising motion the clip no longer has. Otherwise the slots run
    at `LOOKAHEAD_STEP` and clamp to the last row.
    """
    last = n - 1
    if flattened:
        return (last,) * LOOKAHEAD
    return tuple(min(frame + LOOKAHEAD_STEP * i, last) for i in range(LOOKAHEAD))


def clip_tick(frame: int, n: int) -> tuple[int, bool]:
    """Advance one 50 Hz tick: `(next frame, flattened)`.

    The frame stops on the last row and raises the flat hold rather than running off the
    end.
    """
    last = n - 1
    if frame >= last:
        return last, True
    return frame + 1, False
