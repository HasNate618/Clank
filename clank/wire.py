"""OpenXR remote-driver wire format, and the MuJoCo -> OpenXR frame map.

The layout is the `r_remote_data` struct the Monado `remote` driver reads: 376
bytes, little-endian, header `mndrmt3\\0`. It is duplicated from
`spike/c0/c0_client.py` on purpose - that harness is frozen evidence for C0, and
`assert_layout()` in both places means a size drift fails loudly instead of
silently sending garbage to the compositor.

Frame map, measured in E3 and reused here: MuJoCo is Z-up with +X forward, the
driver wants Y-up with -Z forward, so

    xr_x = -mj_y     xr_y = mj_z     xr_z = -mj_x

Positions are metres. Quaternions travel as (x, y, z, w); MuJoCo's own root_q is
(w, x, y, z), so conversions are explicit at every boundary.
"""

from __future__ import annotations

import ctypes as C
import math
import struct
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# The axis convention has exactly one owner. `rotate_z` is imported rather than
# restated because a second copy of "which way is forward" is precisely how
# `hand_target` came to disagree with `dodge_target`. There is no cycle:
# projection -> body, and body imports nothing from here.
#
# Both import shapes have to work. The pose relay is launched as a FILE
# (`python clank/pose_relay.py`), so its own `from . import wire` falls back to a
# bare `import wire`, and a bare module has no package to be relative to. Importing
# this relatively broke body mode the moment it was added.
try:
    from .projection import rotate_z
except ImportError:  # pragma: no cover - only when executed as a script
    from projection import rotate_z  # type: ignore[no-redef]

# ---------------------------------------------------------------- wire format


class Quat(C.Structure):
    _fields_ = [("x", C.c_float), ("y", C.c_float), ("z", C.c_float), ("w", C.c_float)]


class Vec3(C.Structure):
    _fields_ = [("x", C.c_float), ("y", C.c_float), ("z", C.c_float)]


class Vec2(C.Structure):
    _fields_ = [("x", C.c_float), ("y", C.c_float)]


class Vec1(C.Structure):
    _fields_ = [("x", C.c_float)]


class Pose(C.Structure):
    _fields_ = [("orientation", Quat), ("position", Vec3)]


class Fov(C.Structure):
    _fields_ = [
        ("angle_left", C.c_float),
        ("angle_right", C.c_float),
        ("angle_up", C.c_float),
        ("angle_down", C.c_float),
    ]


class View(C.Structure):
    _fields_ = [("fov", Fov), ("pose", Pose), ("_pad", C.c_uint32)]


class HeadData(C.Structure):
    _fields_ = [
        ("views", View * 2),
        ("center", Pose),
        ("per_view_data_valid", C.c_bool),
        ("_pad0", C.c_bool),
        ("_pad1", C.c_bool),
        ("_pad2", C.c_bool),
    ]


class ControllerData(C.Structure):
    _fields_ = [
        ("pose", Pose),
        ("linear_velocity", Vec3),
        ("angular_velocity", Vec3),
        ("hand_curl", C.c_float * 5),
        ("trigger_value", Vec1),
        ("squeeze_value", Vec1),
        ("squeeze_force", Vec1),
        ("thumbstick", Vec2),
        ("trackpad_force", Vec1),
        ("trackpad", Vec2),
        ("hand_tracking_active", C.c_bool),
        ("active", C.c_bool),
        ("system_click", C.c_bool),
        ("system_touch", C.c_bool),
        ("a_click", C.c_bool),
        ("a_touch", C.c_bool),
        ("b_click", C.c_bool),
        ("b_touch", C.c_bool),
        ("trigger_click", C.c_bool),
        ("trigger_touch", C.c_bool),
        ("thumbstick_click", C.c_bool),
        ("thumbstick_touch", C.c_bool),
        ("trackpad_touch", C.c_bool),
        ("_pad0", C.c_bool),
        ("_pad1", C.c_bool),
        ("_pad2", C.c_bool),
    ]


class RemoteData(C.Structure):
    _fields_ = [
        ("header", C.c_uint64),
        ("head", HeadData),
        ("left", ControllerData),
        ("right", ControllerData),
    ]


HEADER = struct.unpack("<Q", b"mndrmt3\0")[0]
SIZE = C.sizeof(RemoteData)

#: Sizes the driver expects. A mismatch means this file has drifted.
EXPECTED = {"r_head_data": (C.sizeof(HeadData), 128),
            "r_remote_controller_data": (C.sizeof(ControllerData), 120),
            "r_remote_data": (C.sizeof(RemoteData), 376)}


def assert_layout() -> None:
    """Raise when the struct no longer matches the driver's ABI."""
    bad = {k: (size, want) for k, (size, want) in EXPECTED.items() if size != want}
    if bad:
        raise ValueError(f"remote data layout mismatch: {bad}")


# ----------------------------------------------------------------- quaternions
# Internally w, x, y, z to match MuJoCo; converted to x, y, z, w at the packet.


def quat_mul(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_conj(q: Sequence[float]) -> tuple[float, float, float, float]:
    return (q[0], -q[1], -q[2], -q[3])


def quat_normalize(q: Sequence[float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(c * c for c in q))
    if norm == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm)


def quat_rotate(q: Sequence[float], v: Sequence[float]) -> tuple[float, float, float]:
    """Rotate vector v by unit quaternion q (w, x, y, z)."""
    qw, qx, qy, qz = q
    vx, vy, vz = v
    # t = 2 * (q_vec x v); v' = v + qw * t + q_vec x t
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def quat_from_matrix(m: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    """Rotation matrix -> unit quaternion (w, x, y, z), Shepperd's method."""
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = (0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        q = ((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        q = ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s)
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        q = ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s)
    return quat_normalize(q)


#: xr_x = -mj_y, xr_y = mj_z, xr_z = -mj_x, as a matrix.
MJ_TO_XR_M = ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0))
MJ_TO_XR_Q = quat_from_matrix(MJ_TO_XR_M)


def mj_to_xr_vec(v: Sequence[float]) -> tuple[float, float, float]:
    return (-v[1], v[2], -v[0])


def mj_to_xr_quat(q_wxyz: Sequence[float]) -> tuple[float, float, float, float]:
    """Re-express an orientation in the driver's frame; returns (x, y, z, w)."""
    q = quat_normalize(q_wxyz)
    out = quat_mul(quat_mul(MJ_TO_XR_Q, q), quat_conj(MJ_TO_XR_Q))
    return (out[1], out[2], out[3], out[0])


# ------------------------------------------------------- BodyState -> XR poses


def _xyzw_to_wxyz(q: Iterable[float]) -> tuple[float, float, float, float]:
    x, y, z, w = q
    return (w, x, y, z)


def body_state_to_xr(
    state: Mapping[str, Any],
    *,
    yaw_offset_rad: float = 0.0,
    scale: float = 1.0,
    pivot: Sequence[float] = (0.0, 0.0, 0.0),
    scale_head: bool = True,
) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]]:
    """BodyState dict -> absolute OpenXR poses for head and both hands.

    `BodyState` poses are root-relative and the root itself carries a rotation
    (the hold clip starts near -90 deg about Z), so composing the root first is
    not optional: skipping it leaves the heading wrong while the position looks
    plausible, which is exactly the mistake that once hid a working turn.

    `yaw_offset_rad` chooses the robot's **heading in the virtual room**: every pose
    is rotated by it about the world up axis. The desk needs a 90 deg left turn
    (operator, 2026-10-30) and this body cannot twist that far - a commanded left
    look reaches -28.6 deg against +54.3 right, measured at the join - so rather
    than turn the robot we place it in the room already facing the desk.

    Rotating the camera *and* both hands together is the whole point: the hand's
    position in the head's own frame is unchanged, so the arms still point where the
    view looks, while the desk's bearing relative to the robot changes. Rotating the
    head alone would have left the hands 90 deg off the view and no reach would have
    been possible. Declared caveat: the facing is chosen, not walked.

    `scale` sizes the emulated **player** against the room, about the play-space origin
    (the floor beneath the robot, since the root spawns at x=y=0). It exists because Job
    Simulator is laid out for an adult and the G1 is 1.32 m with ~0.45 m of arm travel:
    `research/g1-scale.md` measured the apparent-size error as roughly half, and the card
    on JobBot's counter sits 0.486 m from the wrist where the arm alone cannot go, so the
    body was stepping to balance and dragging the camera with it - which then invalidated
    every pixel marked before the move (`research/wrist-response.md`).

    Scaling the head and both hands together is again the whole point: the game-space arm
    length becomes `scale` times the real one, so the reach required for a given object
    falls by the same factor - 0.486 m of game distance is 0.34 m of real arm travel at
    1.43 - and the reach stays inside the arm-only envelope. The robot's own dimensions are
    untouched and the sim view still shows a 1.32 m humanoid; only the pose handed to the
    game is scaled, so the honest sentence is "the emulated player is adult-sized", not
    "the G1 reaches the counter". Orientation is unaffected, so the hands still point where
    the view looks.

    `pivot` is what the scale is measured from, and it decides whether any of that is true.
    Measured 2026-11-03, same body pose, card depth on the stream:

    .. code-block:: text

        pivot at the floor   card 0.527 m, floor 0.549 m   (scale on)
        pivot at the floor   card 0.342 m, floor 0.311 m   (scale off)

    so the game really does lift the camera, and scaling about the floor made the reach
    *longer*, not shorter: raising the pivot by 0.43 x 0.58 m moves the hand 0.25 m up
    while lengthening the arm only 0.43 x 0.45 = 0.19 m. For an object that sits half a
    metre away and below the shoulder, the rise costs more than the longer arm earns.

    With the pivot at the resting wrist the two effects stop fighting:

    .. code-block:: text

        pivot at the floor  travel = |O/k - H|  = 0.566 m   (outside the arm)
        pivot at the wrist  travel = |O - H|/k  = 0.340 m   (inside the clamp)

    So the caller should pass the wrist rest point, not the origin. The feet then sit below
    the floor by `(scale - 1)` times the wrist height, which is invisible: the game is
    first person and does not simulate the player's body.
    """
    if not math.isfinite(scale) or scale <= 0.0:
        # A negative or zero scale mirrors the world or collapses it to the origin, and
        # both would look like a tracking fault rather than the argument error they are.
        raise ValueError(f"scale must be finite and positive, got {scale!r}")
    pivot_v = tuple(float(c) for c in pivot)
    if len(pivot_v) != 3 or not all(math.isfinite(c) for c in pivot_v):
        raise ValueError(f"pivot must be three finite numbers, got {pivot!r}")
    root_p = tuple(float(c) for c in state["root_p"])
    # BodyState.root_q is (w, x, y, z). It is NOT the same order as Pose.q_xyzw,
    # and converting it as if it were turns an identity root into a 180 deg spin
    # about X: the heading comes out wrong while the positions still look sane.
    root_q = quat_normalize(tuple(float(c) for c in state["root_q"]))

    def compose(pose: dict[str, Any], scale_this: float = 1.0):
        p_rel = tuple(float(c) for c in pose["p"])
        q_rel = quat_normalize(_xyzw_to_wxyz(pose["q_xyzw"]))
        rotated = quat_rotate(root_q, p_rel)
        p_world = (root_p[0] + rotated[0], root_p[1] + rotated[1], root_p[2] + rotated[2])
        q_world = quat_mul(root_q, q_rel)
        if scale_this != 1.0:
            # About `pivot`, not the origin: the origin is the floor, and scaling there
            # lifts the hand faster than it lengthens the arm. This runs BEFORE the yaw
            # because the pivot is documented and measured in the body's world frame
            # (post root rotation, pre yaw offset). Uniform scaling commutes with a
            # rotation only about that rotation's own origin, so scaling after the yaw
            # pivots about a *rotated* pivot and displaces everything by
            # (I - R_yaw) x pivot - 0.18 m at this pivot and a 90 deg yaw offset.
            p_world = tuple(
                pivot_v[k] + (p_world[k] - pivot_v[k]) * scale_this for k in range(3)
            )
        if yaw_offset_rad != 0.0:
            # Rotate about world up. `rotate_z` is borrowed from projection rather
            # than restated, because a second copy of an axis convention is how
            # hand_target came to disagree with dodge_target about which way is
            # forward. Pre-multiplying turns the orientation in the world frame.
            p_world = rotate_z(p_world, yaw_offset_rad)
            half = yaw_offset_rad / 2.0
            q_world = quat_mul(
                (math.cos(half), 0.0, 0.0, math.sin(half)), q_world
            )
        return mj_to_xr_vec(p_world), mj_to_xr_quat(q_world)

    # `scale_head=False` is the configuration that actually shortens a reach, and the
    # reason is measured: the card on JobBot's counter sits 0.45 m ABOVE the resting hand
    # and only 0.12 m away from it horizontally. Scaling the head lifts the camera 0.29 m,
    # which changes the depth the camera reports, which moves the object's apparent
    # position in the robot frame and cancels most of the longer arm - 1.43 bought 0.500 ->
    # 0.472 m, four per cent. Leaving the head alone keeps the camera where the delivery
    # was measured and the depth valid, while the hands still travel 1.43x per metre.
    head_scale = scale if scale_head else 1.0
    return {
        "head": compose(state["head"], head_scale),
        "left": compose(state["left_hand"], scale),
        "right": compose(state["right_hand"], scale),
    }


#: Measured G1 geometry, in the driver's frame: E3 stood the HMD at 1.221 m and
#: the WBC probe put the wrists at (+-0.222, 0.721, 0.017). Used only before the
#: body has published a state, so the game sees a standing humanoid, not a black
#: view at the origin.
REST_POSES = {
    "head": ((0.0, 1.221, 0.0), (0.0, 0.0, 0.0, 1.0)),
    "left": ((-0.222, 0.721, 0.017), (0.0, 0.0, 0.0, 1.0)),
    "right": ((0.222, 0.721, 0.017), (0.0, 0.0, 0.0, 1.0)),
}


def blank_packet() -> RemoteData:
    d = RemoteData()
    d.header = HEADER
    d.head.per_view_data_valid = False
    for c in (d.left, d.right):
        c.active = True
        c.hand_tracking_active = True
        c.pose.orientation.w = 1.0
    return d


def fill(
    d: RemoteData,
    poses: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    *,
    grip: tuple[float, float] = (0.0, 0.0),
    curl: tuple[float, float] = (0.0, 0.0),
    trigger: tuple[float, float] = (0.0, 0.0),
    trigger_click: tuple[bool, bool] = (False, False),
    system_click: tuple[bool, bool] = (False, False),
    a_click: tuple[bool, bool] = (False, False),
    fov: tuple[float, float] | None = None,
    ipd_m: float = 0.063,
) -> RemoteData:
    """Write `{"head"|"left"|"right": (position, quaternion_xyzw)}` into a packet.

    Button tuples are (left, right). Grip, curl and trigger are separate on
    purpose: a tracker's squeeze is not a finger curl, and which of them a game
    reads is a question only the game can answer - so they travel independently
    and the caller decides.

    `trigger_click` is an edge, not a level. The previous version derived it from
    `trigger > 0.0`, which re-clicked on every packet while the button was held;
    the caller now owns the rising edge.

    `fov` is `(horizontal_half_angle, vertical_half_angle)` in radians and declares
    a symmetric frustum via `head.views[].fov` + `head.per_view_data_valid`.

    MEASURED 2026-10-31: this build does act on it - the view visibly widened the
    moment the relay declared 110 x 78. That same run then died, and the reason is
    the rest of this paragraph.

    `per_view_data_valid` covers the whole per-view entry, `{fov, pose, _pad}` - not
    just the frustum. While the flag is False those poses are zeroed and the driver
    ignores them, which is why it did not matter before. Setting it True makes them
    AUTHORITATIVE, so a caller declaring a FOV must also state the eyes: both at the
    origin with an orientation of (0, 0, 0, 0) is a zero-norm quaternion, not a
    rotation, and it gives a client degenerate transforms. Hence `ipd_m`.
    """
    if fov is not None:
        half_h, half_v = float(fov[0]), float(fov[1])
        if not (math.isfinite(half_h) and math.isfinite(half_v)):
            raise ValueError("fov angles must be finite")
        if half_h <= 0.0 or half_v <= 0.0:
            raise ValueError("fov half-angles must be positive")
        ipd = float(ipd_m)
        if not math.isfinite(ipd) or not 0.0 < ipd < 0.2:
            raise ValueError("ipd_m must be a plausible finite eye separation in metres")
        for i, side in enumerate((-1.0, 1.0)):
            view = d.head.views[i]
            view.fov.angle_left = half_h
            view.fov.angle_right = half_h
            view.fov.angle_up = half_v
            view.fov.angle_down = half_v
            # Eye pose relative to the head centre. OpenXR eye space: +X is the
            # viewer's right, +Y up, -Z forward. Level and symmetric - no toe-in,
            # since this body's eyes are a choice, not a lens design.
            view.pose.position.x = side * ipd / 2.0
            view.pose.position.y = 0.0
            view.pose.position.z = 0.0
            view.pose.orientation.x = 0.0
            view.pose.orientation.y = 0.0
            view.pose.orientation.z = 0.0
            view.pose.orientation.w = 1.0
        d.head.per_view_data_valid = True

    p, q = poses["head"]
    d.head.center.position.x, d.head.center.position.y, d.head.center.position.z = p
    d.head.center.orientation.x, d.head.center.orientation.y, \
        d.head.center.orientation.z, d.head.center.orientation.w = q
    for i, side in enumerate(("left", "right")):
        pos, quat = poses[side]
        c = getattr(d, side)
        c.pose.position.x, c.pose.position.y, c.pose.position.z = pos
        c.pose.orientation.x, c.pose.orientation.y, c.pose.orientation.z, c.pose.orientation.w = quat
        c.squeeze_value.x = grip[i]
        c.squeeze_force.x = grip[i]
        for k in range(5):
            c.hand_curl[k] = curl[i]
        c.trigger_value.x = trigger[i]
        c.trigger_touch = trigger[i] > 0.0
        c.trigger_click = bool(trigger_click[i])
        # The menu button. The remote driver maps this bit to XRT_INPUT_INDEX_SYSTEM_CLICK
        # (`r_device.c`, inputs[0]), which is the Index controller's menu click - the button
        # JobBot asks a player to press and the one thing here that cannot come from the body.
        c.system_click = bool(system_click[i])
        c.a_click = bool(a_click[i])
        c.thumbstick.x = 0.0
        c.thumbstick.y = 0.0
    return d
