"""Which still pose serves a grab: planted feet, or a stepped/pivoted stand.

The object is given in the pelvis frame (`p_root`): +X forward, +Y the robot's left, metres.
Two gates decide the answer, and both are needed.

- **Yaw.** The arm can physically crane up to ~90 deg off forward, but using it that way
  is the "dance" the design rejects. Past ``PLANTED_YAW_RAD`` the root steps instead, even
  when mink's residual would have been acceptable.
- **Arm residual.** Mink's planted solve reports how far the hand ended from the object.
  At or above ``PLANTED_RESIDUAL_M`` the object is out of reach from the current feet, so
  the root steps even though the direction was fine.

`step_root` is that root-goal placement, pure planar geometry: turn until the preferred hand
direction points at the object, then put the root behind it so that the preferred hand point
lands on the object. Under the ARDY adapter the goal is a constraint handed to the generator,
not a pose authored here, and `decide_next` is the policy that spends it: one root goal, then
stop with the best residual reached.

Nothing here imports NumPy, MuJoCo or mink: `classify`, `step_root` and `decide_next` are
stdlib, and the solves are the only places a heavy import belongs.
"""

from __future__ import annotations

import math

PLANTED_YAW_RAD = math.radians(35)
PLANTED_RESIDUAL_M = 0.05
TRACK_SUCCESS_M = 0.08
CLASP_START_M = 0.08
ROOT_GOAL_RESIDUAL_M = 0.10
STEP_WORTH_M = 0.10
STEP_WORTH_RAD = math.radians(20)
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
    """Root xy and yaw that put the preferred hand on the object: one root goal.

    ``root_xy_guess`` is the current root position, and it sets the facing direction, so the
    placement does depend on it: two different guesses put the same object at two different
    roots. Applying the function again from its own result does return that same root, but a
    single pass can move the root further than ``|HAND_PREF_xy|``: the displacement is
    ``|root - guess| = | |obj - guess| - |HAND_PREF_xy| |``, so it grows with the object's
    *distance*, not with the turn - the yaw rotation is what *minimises* the move.
    ``HAND_PREF_xy`` is the 2-D norm, ``hypot(0.35, 0.28) = 0.4482`` m; the 3-D norm,
    0.4727, is not what the code uses. The caller checks the distance with `step_too_far` —
    this function deliberately does not clamp, because a 3 m unprojection is a bad
    measurement, not a walk.
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


def step_worthwhile(travel_m: float, heading_rad: float) -> bool:
    """Whether a root goal that travels `travel_m` and turns `heading_rad` is worth a generation.

    A root goal buys two things: distance the arm cannot cover, and a body facing where the
    hand has to go. Measured on the ARDY chain, a yaw-dominant object already inside the arm's
    reach asked for 0.02-0.06 m of travel, and spending a generation on it moved the achieved
    hand 111.4 -> 76.4 and 109.6 -> 65.7 mm - inside the 33-80 mm spread the same generator
    shows unseeded, so that generation bought a resample, not a correction. A goal has to buy
    a real walk (``STEP_WORTH_M``) or a real turn (``STEP_WORTH_RAD``) to be worth one.

    ``heading_rad`` is `yaw_error`: how far the body has to turn to point at the object. The
    turn and the walk are separate buys - a 53.5 deg object 0.41 m away needs no walk at all
    and still has to be turned to, and the placement's own yaw is that angle minus the hand's
    39 deg offset, so it understates the turn the body has to make.

    This is not `step_too_far`: that refuses a placement so far away it can only be a bad
    measurement, this refuses one too short (and too square-on) to change anything.
    """
    return float(travel_m) >= STEP_WORTH_M or abs(float(heading_rad)) >= STEP_WORTH_RAD


def step_too_far(root_xy: tuple[float, float], guess_xy: tuple[float, float]) -> bool:
    """True when the placement asks the root to travel further than one reach may step."""
    return math.hypot(root_xy[0] - guess_xy[0], root_xy[1] - guess_xy[1]) > MAX_STEP_M


def decide_next(
    *,
    kind: str,
    residual_m: float,
    escalated: bool,
    gen: int,
    elapsed_s: float,
    generated: int,
) -> str:
    """"escalate", "stop" or "saturated" for a generation whose hand residual is known.

    ``kind`` is `classify`'s answer for the object, ``residual_m`` the distance from the
    achieved hand to the object, ``escalated`` whether a root goal has already been spent,
    ``generated`` how many clips the chain has produced, ``elapsed_s`` how long it has run and
    ``gen`` the attempt index the caller prints; the budget rule reads ``generated``.

    Success is checked first. A hand on the object stops even when `classify` called the pose
    a step: walking away from a grab that works is the one failure the chain cannot recover
    from. Otherwise an object outside the planted envelope, or a residual over
    ``ROOT_GOAL_RESIDUAL_M``, wants a root goal — but one, so a spent root goal, a spent
    generation budget or a spent chain timeout answers "saturated" and the caller reports the
    best residual it reached instead of asking again. Anything left is "stop": a planted
    reach a few centimetres short is inside the generator's own spread, and chasing it would
    spend a generation on sampling noise.

    Design spec §6.4's chain vocabulary also includes "generate"; v1 has no rule that emits
    it. The first generation is unconditional and this decision is only asked after one.
    """
    if residual_m <= TRACK_SUCCESS_M:
        return "stop"
    if kind == "step" or residual_m > ROOT_GOAL_RESIDUAL_M:
        if escalated or generated >= MAX_STEP_GEN or elapsed_s >= STEP_CHAIN_S:
            return "saturated"
        return "escalate"
    return "stop"


def solve_planted(*args: object, **kwargs: object) -> tuple[float, ...]:
    """Mink still frame on the current feet. Task 5 fills this in."""
    raise NotImplementedError("solve_planted belongs to the mink still-frame task")


def solve_stepped(*args: object, **kwargs: object) -> tuple[float, ...]:
    """Mink still frame on a placed stand. Task 5 fills this in."""
    raise NotImplementedError("solve_stepped belongs to the mink still-frame task")
