"""Wrist -> pixel, through the camera we ourselves are telling the game to use.

This is the arithmetic that makes a slow brain sufficient. We emit the HMD pose,
so the camera is not an unknown we must estimate: a wrist from `mjData` and a
pixel named by a writer become comparable with a pinhole projection. No detector,
no depth, no 3D reconstruction.

Two facts this module depends on, both worth stating because they are easy to
break later:

1. **The root cancels.** `BodyState.head` and the wrist poses are all root-relative,
   so the projection subtracts one root-relative point from another. Composing the
   root first is not merely unnecessary, it would reintroduce the ordering trap
   that already cost a session (`root_q` is `(w, x, y, z)`, not `(x, y, z, w)`).
2. **The FOV is an input, not a constant.** `wire.fill` sets `per_view_data_valid =
   False` and writes no FOV, so the compositor's live field of view is currently
   unknown here. Tests pass a fixture; production code must pass a measured value.
   Inventing a default would make every projection a consistent lie.
3. **The frustum is rectilinear.** An OpenXR field of view is a tangent-space
   frustum, so a pixel is proportional to `tan(angle)`, not to the angle itself.
   The two coincide at the edges and at the centre of a symmetric frustum, which is
   why the linear version looked right, but they diverge in between: at 35 deg
   off-axis in a 140 deg field the difference is 0.12 of the frame, about 236 px of
   1920. Measured 2026-11-01 (`research/camera-equals-hmd-pose.md` section 7).

Frame conventions. Positions are MuJoCo's Z-up frame with +X forward, so in the
head's own frame the gaze is +X, up is +Z, and the robot's left is +Y. The output
is normalised to the HMD view, origin top-left, matching `clank.intent/1`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation only; `from __future__` makes it a string at runtime.
    # Kept out of the runtime import graph so this module can also be imported
    # bare - `wire.py` reaches it that way when the pose relay runs as a script.
    from .body import Pose

#: Head-frame axes. ASSUMPTION: the derived head pose is the clip's head body
#: orientation, whose forward axis is MuJoCo's conventional +X. If the live
#: pose->pixel measurement disagrees, change these three vectors and nothing else.
FORWARD = (1.0, 0.0, 0.0)
UP = (0.0, 0.0, 1.0)
LEFT = (0.0, 1.0, 0.0)

#: Below this forward component a point is at or behind the image plane. It is not
#: projected to a mirrored pixel, because a mirror would drive a servo away.
_MIN_FORWARD = 1e-6


def root_offset(
    forward_m: float = 0.0, up_m: float = 0.0, lateral_m: float = 0.0
) -> tuple[float, float, float]:
    """A named offset expressed in the model's root frame.

    The single place a caller's words become a vector. Every builder that takes a
    "forward", "up" or "lateral" distance should come through here rather than
    restating the mapping, because restating it is exactly how `hand_target` came
    to claim "+Y is forward, +X is the robot's left" while this module and
    `dodge_target` used the opposite. Measured 2026-10-30 against the clip's wrist
    positions (left y = +0.190, right y = -0.180), the model is +X forward,
    +Y left, +Z up.

    `lateral_m` is positive toward the robot's **left**, matching the sign of yaw
    that turns left and the axis `dodge_target` is measured on.
    """
    fx, fy, fz = FORWARD
    lx, ly, lz = LEFT
    ux, uy, uz = UP
    return (
        forward_m * fx + lateral_m * lx + up_m * ux,
        forward_m * fy + lateral_m * ly + up_m * uy,
        forward_m * fz + lateral_m * lz + up_m * uz,
    )


@dataclass(frozen=True)
class Fov:
    """An OpenXR-style asymmetric frustum, in radians.

    Magnitudes are used at projection time: implementations disagree about whether
    `angle_right` is signed, and a symmetric frustum must project identically either
    way, so a caller's sign convention cannot change the answer.
    """

    angle_left: float
    angle_right: float
    angle_up: float
    angle_down: float

    @staticmethod
    def from_degrees(left: float, right: float, up: float, down: float) -> Fov:
        angles = (left, right, up, down)
        if not all(isinstance(a, (int, float)) and math.isfinite(a) for a in angles):
            raise ValueError("fov angles must be finite")
        if abs(left) + abs(right) <= 0.0:
            raise ValueError("horizontal fov must be non-zero")
        if abs(up) + abs(down) <= 0.0:
            raise ValueError("vertical fov must be non-zero")
        return Fov(*[math.radians(abs(a)) for a in angles])

    @property
    def horizontal(self) -> float:
        return self.angle_left + self.angle_right

    @property
    def vertical(self) -> float:
        return self.angle_up + self.angle_down


def _angle_to_pixel(
    angle: float, positive_half: float, negative_half: float
) -> float:
    """Normalised pixel for `angle`, in a rectilinear frustum.

    `positive_half` is the half-angle on the side that maps to pixel 0, so for a
    horizontal axis that is `angle_left`. The mapping is

        pixel = (tan(positive_half) - tan(angle)) / (tan(positive_half) + tan(negative_half))

    which sends `angle = +positive_half` to 0, `angle = -negative_half` to 1, and
    the symmetric case to 0.5 by construction.
    """
    near = math.tan(positive_half)
    far = math.tan(negative_half)
    return (near - math.tan(angle)) / (near + far)


def _pixel_to_angle(
    pixel: float, positive_half: float, negative_half: float
) -> float:
    """The exact inverse of `_angle_to_pixel`.

    Clamped in tangent space by construction, not in angle space: `pixel = 0`
    returns exactly `positive_half` rather than a value near it.
    """
    near = math.tan(positive_half)
    far = math.tan(negative_half)
    return math.atan(near - pixel * (near + far))


def _rotate_inverse(q_xyzw: Sequence[float], v: Sequence[float]) -> tuple[float, float, float]:
    """Rotate `v` by the inverse of unit quaternion `q` (world -> body frame)."""
    x, y, z, w = q_xyzw
    vx, vy, vz = v
    # Inverse of a unit quaternion is its conjugate: (-x, -y, -z, w).
    tx = 2.0 * (-y * vz + z * vy)
    ty = 2.0 * (-z * vx + x * vz)
    tz = 2.0 * (-x * vy + y * vx)
    return (
        vx + w * tx + (-y * tz + z * ty),
        vy + w * ty + (-z * tx + x * tz),
        vz + w * tz + (-x * ty + y * tx),
    )


def yaw_of(q_xyzw: Sequence[float]) -> float:
    """Heading about +Z, in radians, from a unit quaternion in `(x, y, z, w)`.

    Positive is the robot's left, matching `look_target` and `root_offset`'s
    lateral-is-left convention. Pitch and roll contribute nothing, which is the
    point: carrying a hand with the head is a rotation about the vertical axis.

    Wraps to `(-pi, pi]` because a heading of 270 degrees right is 90 degrees left.
    """
    x, y, z, w = q_xyzw
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rotate_z(vec: Sequence[float], yaw_rad: float) -> tuple[float, float, float]:
    """`vec` rotated about +Z by `yaw_rad`, height untouched.

    A zero yaw returns the input **exactly**, by early return rather than by float
    luck, because `frame="head"` guarantees that an untwisted body reproduces the
    poses measured before that frame existed.
    """
    x, y, z = vec
    if yaw_rad == 0.0:
        return (x, y, z)
    cos_t = math.cos(yaw_rad)
    sin_t = math.sin(yaw_rad)
    return (x * cos_t - y * sin_t, x * sin_t + y * cos_t, z)


def carry_with_head(
    point: Sequence[float], pivot: Sequence[float], delta_yaw: float
) -> tuple[float, float, float]:
    """Carry `point` around the vertical axis through `pivot` by `delta_yaw`.

    This is "the hand follows the head": when the torso twists, a wrist held at a
    fixed offset from the body stays in the same place relative to the gaze. `point`
    standing exactly on `pivot` never moves, and the radius about the pivot is
    preserved - a hand out in front swings, one at the axis does not.

    Zero twist is an exact identity, not an approximate one (see `rotate_z`).
    """
    if delta_yaw == 0.0:
        return (float(point[0]), float(point[1]), float(point[2]))
    dx = point[0] - pivot[0]
    dy = point[1] - pivot[1]
    rx, ry, rz = rotate_z((dx, dy, point[2] - pivot[2]), delta_yaw)
    return (rx + pivot[0], ry + pivot[1], rz + pivot[2])


def project_point(
    head: Pose, point_p: Iterable[float], fov: Fov
) -> tuple[float, float] | None:
    """Normalised `(u, v)` of a root-relative point seen from `head`, or None.

    None means "not in this view": behind the camera, or on the image plane where
    the direction is undefined. Callers must treat it as "no answer", never as a
    pixel near the centre.
    """
    px, py, pz = point_p
    dx = px - head.p[0]
    dy = py - head.p[1]
    dz = pz - head.p[2]
    fx, fy, fz = _rotate_inverse(head.q_xyzw, (dx, dy, dz))

    if fx <= _MIN_FORWARD:
        return None

    yaw = math.atan2(fy, fx)      # + toward the robot's left
    pitch = math.atan2(fz, fx)    # + up

    u = _angle_to_pixel(yaw, fov.angle_left, fov.angle_right)
    v = _angle_to_pixel(pitch, fov.angle_up, fov.angle_down)
    return (u, v)


def pixel_error(target: tuple[float, float], actual: tuple[float, float]) -> float:
    """Euclidean distance in normalised pixels. The servo drives this to zero."""
    return math.hypot(target[0] - actual[0], target[1] - actual[1])


def _rotate(q_xyzw: Sequence[float], v: Sequence[float]) -> tuple[float, float, float]:
    """Rotate `v` by unit quaternion `q` (body -> world). The exact inverse of
    `_rotate_inverse`, which is what makes `unproject` round-trip."""
    x, y, z, w = q_xyzw
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def unproject(
    pixel: tuple[float, float],
    head: Pose,
    fov: Fov,
    depth_m: float,
) -> tuple[float, float, float] | None:
    """The root-relative point that `pixel` looks at, `depth_m` in front.

    `project_point` answers "which pixel is that point in". This answers the
    question a servo actually needs: **where do I put the hand so that it lands on
    that pixel**. Without it a pixel is unusable - the `reach` verb takes offsets in
    metres, and nothing in `clank/` turned an intent's pixel into one.

    `depth_m` is the **forward component along the head's gaze**, not the
    straight-line distance. That choice makes the transform exact and cheap (it is
    the inverse of the projection, not an approximation of it) at the cost of a
    caller having to say how far forward it means. There is no depth sensor here:
    the caller supplies a working distance, and the loop closes on the game's own
    grab-range highlight rather than on a reconstructed 3D point.

    Returns `None` when the geometry has no answer - at or behind the image plane.
    Raises `ValueError` for a pixel outside `[0, 1]` or a non-finite argument: that
    is a caller bug, and returning `None` would make a servo retry it forever.
    """
    u, v = pixel
    if not (math.isfinite(u) and math.isfinite(v)):
        raise ValueError(f"pixel must be finite, got {pixel!r}")
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        raise ValueError(f"pixel must be inside [0, 1], got {pixel!r}")
    if not math.isfinite(depth_m):
        raise ValueError(f"depth_m must be finite, got {depth_m!r}")
    if depth_m <= _MIN_FORWARD:
        return None

    # Undo the two tangent maps in `project_point`, in the opposite order.
    yaw = _pixel_to_angle(u, fov.angle_left, fov.angle_right)
    pitch = _pixel_to_angle(v, fov.angle_up, fov.angle_down)
    fx = depth_m
    fy = fx * math.tan(yaw)      # + toward the robot's left
    fz = fx * math.tan(pitch)    # + up

    rx, ry, rz = _rotate(head.q_xyzw, (fx, fy, fz))
    return (rx + head.p[0], ry + head.p[1], rz + head.p[2])


def servo_step(
    target_pixel: tuple[float, float],
    wrist_pixel: tuple[float, float],
    head: Pose,
    fov: Fov,
    depth_m: float,
    deadband: float = 0.0,
) -> tuple[tuple[float, float, float] | None, float]:
    """One iteration: the wrist target that puts the hand on `target_pixel`.

    Returns `(point, error)`. `point` is `None` when the servo should stop - either
    the wrist is already within `deadband` of the target pixel, or the geometry has
    no answer. `error` is always reported, including on the `None` paths, so a
    caller can tell "arrived" from "cannot answer" instead of seeing a silent stop.

    This is deliberately stateless and unlatched: it does not know about joint
    limits, reach, or the game's grab range, so it will happily name a point the arm
    cannot occupy. The adapter clamps against `clank.validate` and terminates on the
    blue highlight; a limit check here would make the pure function a second,
    differently-wrong copy of those rules.
    """
    if not math.isfinite(deadband) or deadband < 0.0:
        raise ValueError(f"deadband must be finite and >= 0, got {deadband!r}")
    error = pixel_error(target_pixel, wrist_pixel)
    if error <= deadband:
        return None, error
    return unproject(target_pixel, head, fov, depth_m), error
