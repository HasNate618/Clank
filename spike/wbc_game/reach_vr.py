"""Put the emulated hand on a point measured from the depth stream.

This replaces the streaming integrator (`reach_ho.py`, `grab_ho.py`), which oscillated:
`offset += 0.7 * error` at 6-8 Hz against a plant that settles in 0.77-1.05 s with +75 %
overshoot on the up axis is an oscillator by construction, and the resulting hunting is what
looked like the robot dancing.

What has to be right, each measured today:

* **The camera is not the robot's head.** The relay turns the head by `--yaw-offset-deg`
  (90 here) before sending it, so a pixel unprojected with the raw head names a point 90 deg
  away. With the raw head the reach still "converged" to 14 mm - on the wrong side of the
  room.
* **`vr_pos` is the physical wrist link**, and `_achieved` reports that link plus a 0.18 m
  `VR_OFFSETS` lever rotated by the wrist. The policy rotates the wrist as it reaches, so the
  lever swings up to 0.09 m: that, not a gain, is most of the residual.
* **The goal is pinned in the world.** A lean moves the root 2-4 cm, which makes a root-frame
  goal stale by exactly that much; a pixel is valid only for the camera that saw it, but the
  surface it names does not move. Pinned, a lean costs nothing and the guard can be set where
  a *step* is (>0.15 m) rather than where a lean is.
* **Pin the head to its current pose, not the clip's.** The resting head sits ~21 deg off the
  clip; asking for the clip rotates the camera 21 deg mid-reach and walks the target out of
  the frame.
* **Sample the depth more than once.** A single read occasionally comes back torn - the same
  card pixel read 0.29 m and 1.20 m a minute apart.
* **Correct with a damped gain.** The response is slightly over unity and the lever adds
  ~0.05 m of noise, so a full correction overshoots (152 -> 88 -> 56 -> 42 mm at gain 1.0,
  against 179 -> 55 -> 33 -> 35 -> 26 mm at 0.6).

Measured 2026-11-03, left hand, from rest: 0.41 m reach converged to **26 mm in 5 commands**;
0.52 m converged to 14 mm in 2. Root lean 2-3 cm, no step. The card on JobBot's counter is
**0.57 m** from the resting left hand, i.e. past the ~0.40 m arm-only envelope: reaching it
needs the body to walk closer, not a better loop.

Usage:
    reach_vr.py <px> <py> [--side left|right] [--tol 0.03] [--dwell 1.1]
                [--corrections 3] [--gain 0.6] [--guard 0.15] [--yaw DEG]
                [--target X Y Z | --world X Y Z] [--dry-run] [--no-rest]

`px`/`py` are colour-frame pixels (960x540), as marked in the game window. The tool rests
first, then reads the frame, so a pixel marked at rest stays valid.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from clank.body import Pose  # noqa: E402
from clank.depth_stream import Stream  # noqa: E402
from clank.pixel3d import point_from_pixel  # noqa: E402
from clank.projection import Fov  # noqa: E402

SOCK = Path.home() / ".local/state/Clank/sessions/body/control.sock"
STATE = Path.home() / ".local/state/Clank/sessions/body/body_state.json"
FOV = Fov.from_degrees(70.0, 70.0, 57.11, 57.11)

FLAG_FALL_RISK = 1 << 2
FLAG_SATURATED = 1 << 0

#: Arm-only envelope, from `clank/workspace.MAX_OFFSET_M`. Past it the *legs* answer the
#: reach and the root walks, which invalidates the pixel and everything after it.
ARM_ENVELOPE_M = 0.40
KEEPALIVE_S = 0.5

#: A lean is ~0.03-0.06 m and is normal balancing; a *step* is what took the body 1.33 m and
#: then 4.7 m in one session. The goal is pinned in the world so a lean does not invalidate
#: it, which leaves this guard to catch a real step rather than a shuffle.
STEP_GUARD_M = 0.15

#: A pixel whose point is further than this from the resting hand is not the object being
#: reached for. The arm's whole envelope is ~0.70 m; 1.0 m leaves room for a stretched pose
#: while refusing a stale pixel.
MAX_PLAUSIBLE_M = 1.0

#: `VR_OFFSETS[1]`: the lever between the physical wrist link and the point the game
#: draws the hand at. Measured today: the policy drives the link onto `vr_pos` to within
#: 2 mm, and `_achieved` then adds this lever rotated by the wrist quaternion (0.182 m).
VR_OFFSET = (0.18, 0.025, 0.0)


def send(**kw: object) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(2.0)
        s.connect(str(SOCK))
        s.sendall((json.dumps(kw) + "\n").encode())
        try:
            return json.loads(s.recv(255).decode())
        except (TimeoutError, json.JSONDecodeError):
            return {}


def state() -> dict:
    return json.loads(STATE.read_text())


def wrist(side: str = "right") -> tuple[float, float, float]:
    return tuple(state()[f"{side}_hand"]["p"])


def root() -> tuple[float, float, float]:
    return tuple(state()["root_p"])


def _quat_rotate(q, v) -> tuple[float, float, float]:
    """Rotate `v` by a (w, x, y, z) quaternion."""
    w, x, y, z = q
    t = (2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0]))
    return (
        v[0] + w * t[0] + (y * t[2] - z * t[1]),
        v[1] + w * t[1] + (z * t[0] - x * t[2]),
        v[2] + w * t[2] + (x * t[1] - y * t[0]),
    )


def _median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _xyzw_to_wxyz(q):
    x, y, z, w = q
    return (w, x, y, z)


def _wxyz_to_xyzw(q):
    w, x, y, z = q
    return (x, y, z, w)


def to_world(p, st: dict) -> tuple[float, float, float]:
    r = _quat_rotate(st["root_q"], p)
    root = st["root_p"]
    return (root[0] + r[0], root[1] + r[1], root[2] + r[2])


def to_rel(p, st: dict) -> tuple[float, float, float]:
    """Exact inverse of `to_world` for a unit quaternion."""
    root = st["root_p"]
    d = (p[0] - root[0], p[1] - root[1], p[2] - root[2])
    w, x, y, z = st["root_q"]
    return _quat_rotate((w, -x, -y, -z), d)


def lever(q) -> tuple[float, float, float]:
    """The VR offset rotated into the world by the wrist quaternion (xyzw)."""
    x, y, z, w = q
    t = (2 * (y * VR_OFFSET[2] - z * VR_OFFSET[1]),
         2 * (z * VR_OFFSET[0] - x * VR_OFFSET[2]),
         2 * (x * VR_OFFSET[1] - y * VR_OFFSET[0]))
    return (
        VR_OFFSET[0] + w * t[0] + (y * t[2] - z * t[1]),
        VR_OFFSET[1] + w * t[1] + (z * t[0] - x * t[2]),
        VR_OFFSET[2] + w * t[2] + (x * t[1] - y * t[0]),
    )


def link(st: dict, side: str = "right") -> tuple[float, float, float]:
    """The physical wrist: the reported point minus the lever it is carrying."""
    p = st[f"{side}_hand"]["p"]
    d = lever(st[f"{side}_hand"]["q_xyzw"])
    return tuple(p[k] - d[k] for k in range(3))


def fmt(p) -> str:
    return "[" + ", ".join(f"{v:+.3f}" for v in p) + "]"


def command(target, yaw_deg: float, side: str = "right") -> dict:
    return send(verb="reach", side=side, target=[round(v, 4) for v in target],
                yaw_deg=yaw_deg)


def hold(target, yaw_deg: float, seconds: float, guard: float, root0,
         side: str = "right") -> str:
    """Keep one target alive for `seconds`, watching the root. Returns "" or a reason."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        command(target, yaw_deg, side)
        time.sleep(KEEPALIVE_S)
        st = state()
        if int(st.get("flags") or 0) & FLAG_FALL_RISK:
            return "FALL RISK"
        drift = math.dist(root(), root0)
        if drift > guard:
            return f"root walked {drift:.3f} m"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("px", type=float, nargs="?", default=None)
    ap.add_argument("py", type=float, nargs="?", default=None)
    ap.add_argument("--side", choices=("left", "right"), default="right")
    ap.add_argument(
        "--yaw-offset-deg",
        type=float,
        default=90.0,
        help="the relay's --yaw-offset-deg: the game camera is the robot's head turned "
        "this far, so a pixel cannot be unprojected with the raw head",
    )
    ap.add_argument(
        "--target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="pelvis-frame point (at start) instead of a pixel; for calibration",
    )
    ap.add_argument(
        "--world",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="absolute world point instead of a pixel; the honest way to repeat a reach, "
        "because a world point never goes stale when the body leans",
    )
    ap.add_argument("--tol", type=float, default=0.03, help="stop when this close (m)")
    ap.add_argument("--dwell", type=float, default=1.1, help="settle time per command (s)")
    ap.add_argument("--corrections", type=int, default=3)
    ap.add_argument(
        "--gain",
        type=float,
        default=0.6,
        help="how much of the measured error to apply per correction; the response is "
        "slightly over unity and the wrist lever adds ~0.05 m of noise, so a full "
        "correction overshoots",
    )
    ap.add_argument(
        "--guard",
        type=float,
        default=STEP_GUARD_M,
        help="abort if the root moves this far (m); a lean is ~0.03-0.06, a step >0.10",
    )
    ap.add_argument("--yaw", type=float, default=0.0, help="wrist yaw about Z (deg)")
    ap.add_argument("--dry-run", action="store_true", help="measure only, send nothing")
    ap.add_argument(
        "--no-rest",
        action="store_true",
        help="skip the rest before reaching; skipping it starts from wherever the last "
        "reach left the arm, which is how one run began 0.10 m from its goal and stalled",
    )
    args = ap.parse_args()
    if args.target is None and (args.px is None or args.py is None):
        ap.error("give a pixel or --target X Y Z")

    if not args.dry_run and not args.no_rest:
        send(verb="rest")
        time.sleep(4.0)
    st = state()
    root0 = root()
    rest = wrist(args.side)
    head = Pose(p=tuple(st["head"]["p"]), q_xyzw=tuple(st["head"]["q_xyzw"]))

    if args.world is not None:
        goal_world = tuple(args.world)
        goal = to_rel(goal_world, st)
        source = "--world"
    elif args.target is not None:
        goal = tuple(args.target)
        goal_world = to_world(goal, st)
        source = "--target"
    else:
        # Sample several frames and take the median point. A single read occasionally comes
        # back torn: today the same card pixel read 0.29 m and 1.20 m a minute apart, and the
        # 1.20 m frame would have asked the arm for a metre it does not have. The spread
        # across frames is reported so a bad sample is visible rather than merely survived.
        with Stream() as stream:
            frame = stream.read(timeout=2.0)
        if frame is None:
            print("no depth frame")
            return 1
        cw, ch = int(frame["color_width"]), int(frame["color_height"])
        # The camera is NOT the robot's head: the relay turns it by its yaw offset before
        # sending it (`wire.body_state_to_xr` applies `yaw_offset_rad` to the head pose), so
        # the pixel must be unprojected with the turned head or every ray comes back 90 deg
        # wrong. Measured today: with the raw head the reach still "converged" to 14 mm -
        # on a point 90 deg away from the card, self-consistent and completely wrong.
        head_world = to_world(head.p, st)
        q_world = _quat_mul(st["root_q"], _xyzw_to_wxyz(head.q_xyzw))
        yaw = math.radians(args.yaw_offset_deg)
        cam_q = _quat_mul((math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)), q_world)
        cam_p = (
            math.cos(yaw) * head_world[0] - math.sin(yaw) * head_world[1],
            math.sin(yaw) * head_world[0] + math.cos(yaw) * head_world[1],
            head_world[2],
        )
        camera = Pose(p=cam_p, q_xyzw=_wxyz_to_xyzw(cam_q))
        points = []
        with Stream() as stream:
            for _ in range(3):
                f = stream.read(timeout=2.0)
                if f is None:
                    continue
                p3 = point_from_pixel(f, args.px / cw, args.py / ch, camera, FOV)
                if p3 is not None:
                    points.append(p3)
                time.sleep(0.15)
        source = f"({args.px:.0f}, {args.py:.0f})"
        if not points:
            print(f"no surface at {source}")
            return 1
        world_point = tuple(
            _median([q[k] for q in points]) for k in range(3)
        )
        spread = max(math.dist(q, world_point) for q in points)
        print(f"depth samples {len(points)}  spread {spread * 1000:.0f} mm")
        goal = to_rel(world_point, st)
    need = math.dist(rest, goal)
    if need > MAX_PLAUSIBLE_M:
        # A reach target is an object within arm's length of the rest pose. Beyond this the
        # pixel is not on the object - the camera moved since the pixel was marked, or the
        # depth sampled the far wall - and commanding it would ask the arm for a metre it
        # does not have. Measured today: a 15 deg camera shift turned a 0.56 m card into a
        # 2.21 m "goal", and the only thing that stopped it was the intent validator.
        print(f"goal  {fmt(goal)}  |goal-rest| {need:.3f} m from {source}  "
              f"** not an object: re-mark the pixel in a fresh frame")
        return 1
    print(f"root  {fmt(root0)}   head {fmt(head.p)}")
    print(f"rest  {fmt(rest)}  ({args.side} VR point, i.e. where the game draws the hand)")
    print(f"goal  {fmt(goal)}  |goal-rest| {need:.3f} m  from {source}", end="")
    if need > ARM_ENVELOPE_M:
        print(f"  ** past the {ARM_ENVELOPE_M:.2f} m arm envelope: the legs will answer")
    else:
        print("  inside the arm envelope")

    if args.world is None:
        goal_world = to_world(goal, st)
    if args.dry_run:
        return 0

    issued = 0
    achieved = rest
    # Command the reported point itself and correct by the measured error. The tempting
    # alternative - aim the physical link at `goal - lever`, since `_achieved` adds a
    # `VR_OFFSETS[side]` lever rotated by the wrist - is measured *worse*: it assumes the
    # policy puts the link where it is asked, and on a raised target it does not (the link
    # sat 0.17 m above the command while the lever swung 0.20 m), so the correction would
    # chase a model instead of the measurement. This converges 452 -> 150 -> 310 mm in the
    # bad case and 100 -> 14 mm in the good one, i.e. it is only as good as the lever is
    # stable, which is a property of the target, not of the loop.
    target = list(goal)
    for attempt in range(max(0, args.corrections) + 1):
        issued += 1
        # The world point is the authority; the root frame is just how the executor talks.
        goal_now = to_rel(goal_world, state())
        if issued > 1:
            target = [
                target[k] + args.gain * (goal_now[k] - achieved[k]) for k in range(3)
            ]
        command(target, args.yaw, args.side)
        reason = hold(target, args.yaw, args.dwell, args.guard, root0, args.side)
        st = state()
        goal_now = to_rel(goal_world, st)
        achieved = tuple(st[f"{args.side}_hand"]["p"])
        residual = math.dist(achieved, goal_now)
        d = lever(st[f"{args.side}_hand"]["q_xyzw"])
        d_now = d
        print(f"cmd {issued}: target {fmt(target)} -> hand {fmt(achieved)}  "
              f"residual {residual * 1000:5.1f} mm  lever {fmt(d)}")
        if reason:
            print(f"  ABORT after cmd {issued}: {reason}")
            return 3 if "root" in reason else 4
        if residual <= args.tol:
            break
        # The next iteration applies the correction at its top, from a fresh state read.
        # Not servo gain: the policy puts the physical link on `target` (measured 2 mm in x
        # and y) and then reports the link plus a lever whose direction follows whatever
        # wrist orientation it picked - 28.5 deg from rest on the first command, swinging
        # the lever up to 9 cm. The correction subtracts a lever we could not predict, and
        # one is normally enough because the wrist has already turned by the time we
        # measure.

    st = state()
    flags = int(st.get("flags") or 0)
    drift = math.dist(root(), root0)
    goal_now = to_rel(goal_world, st)
    print(f"\n{args.side} hand {fmt(achieved)}  (link {fmt(link(st, args.side))})")
    print(f"goal  {fmt(goal_now)}  (pinned in the world)")
    print(f"error {math.dist(achieved, goal_now) * 1000:.1f} mm after {issued} command(s)")
    print(f"root drift {drift * 1000:.1f} mm   flags {flags}"
          + ("  SATURATED" if flags & FLAG_SATURATED else ""))
    return 0 if math.dist(achieved, goal_now) <= args.tol else 2


if __name__ == "__main__":
    raise SystemExit(main())
