"""Which still pose serves a grab: planted feet, or a stepped/pivoted stand.

The object is given in the pelvis frame (`p_root`): +X forward, +Y the robot's left, metres.
Two gates decide the answer, and both are needed.

- **Yaw.** The arm can physically crane up to ~90 deg off forward, but using it that way
  is the "dance" the design rejects. Past ``PLANTED_YAW_RAD`` the root steps instead, even
  when mink's residual would have been acceptable.
- **Arm residual.** Mink's planted solve reports how far the hand ended from the object.
  At or above ``PLANTED_RESIDUAL_M`` the object is out of reach from the current feet, so
  the root steps even though the direction was fine.

`step_root` is the placement arithmetic of the design spec §6.4 steps 1-2, pure planar
geometry: turn until the preferred hand direction points at the object, then put the root
behind it so that the preferred hand point lands on the object.

Nothing here imports NumPy, MuJoCo or mink: `classify` and `step_root` are stdlib, and the
solves are the only places a heavy import belongs.
"""

from __future__ import annotations

import math

PLANTED_YAW_RAD = math.radians(35)
PLANTED_RESIDUAL_M = 0.05
TRACK_SUCCESS_M = 0.08
CLASP_START_M = 0.08
MAX_STEP_M = 1.5
MAX_STEP_GEN = 4
STEP_CHAIN_S = 8.0

# The measured comfortable hand point in the root frame, from the in-box cruise target.
HAND_PREF_RIGHT = (0.35, -0.28, 0.15)
HAND_PREF_LEFT = (0.35, 0.28, 0.15)
HAND_PREF = {"right": HAND_PREF_RIGHT, "left": HAND_PREF_LEFT}


def yaw_error(p_root: tuple[float, float, float]) -> float:
    """Angle of the object off the pelvis +X axis, in radians, positive to the robot's left."""
    return math.atan2(p_root[1], p_root[0])


def classify(p_root: tuple[float, float, float], planted_residual_m: float | None) -> str:
    """"planted", "step", or "need_planted_solve".

    ``planted_residual_m`` is ``None`` until the planted solve has run. A candidate whose
    yaw is inside the gate is then reported as ``"need_planted_solve"``: the residual, not
    the direction, is what is missing.
    """
    if abs(yaw_error(p_root)) > PLANTED_YAW_RAD:
        return "step"
    if planted_residual_m is None:
        return "need_planted_solve"
    if planted_residual_m < PLANTED_RESIDUAL_M:
        return "planted"
    return "step"


def step_root(
    obj_xy: tuple[float, float],
    side: str,
    root_xy_guess: tuple[float, float],
) -> tuple[tuple[float, float], float]:
    """Root xy and yaw of a stand whose preferred hand sits on the object.

    ``root_xy_guess`` is the current root position, used once for the facing direction;
    the placement itself does not depend on it. The caller checks the distance with
    `step_too_far` — this function deliberately does not clamp, because a 3 m unprojection
    is a bad measurement, not a walk.
    """
    try:
        pref = HAND_PREF[side]
    except KeyError:
        raise ValueError(f"unknown hand side {side!r}") from None
    yaw = math.atan2(obj_xy[1] - root_xy_guess[1], obj_xy[0] - root_xy_guess[0]) - math.atan2(
        pref[1], pref[0]
    )
    c, s = math.cos(yaw), math.sin(yaw)
    hand_x = c * pref[0] - s * pref[1]
    hand_y = s * pref[0] + c * pref[1]
    return (obj_xy[0] - hand_x, obj_xy[1] - hand_y), yaw


def step_too_far(root_xy: tuple[float, float], guess_xy: tuple[float, float]) -> bool:
    """True when the placement asks the root to travel further than one reach may step."""
    return math.hypot(root_xy[0] - guess_xy[0], root_xy[1] - guess_xy[1]) > MAX_STEP_M


def solve_planted(*args: object, **kwargs: object) -> tuple[float, ...]:
    """Mink still frame on the current feet. Task 5 fills this in."""
    raise NotImplementedError("solve_planted belongs to the mink still-frame task")


def solve_stepped(*args: object, **kwargs: object) -> tuple[float, ...]:
    """Mink still frame on a placed stand. Task 5 fills this in."""
    raise NotImplementedError("solve_stepped belongs to the mink still-frame task")
