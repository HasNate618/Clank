"""Put the emulated hand on a point measured from the depth stream.

This replaces the streaming integrator (`reach_ho.py`, `grab_ho.py`), which oscillated:
`offset += 0.7 * error` at 6-8 Hz against a plant that settles in 0.77-1.05 s with +75 %
overshoot on the up axis is an oscillator by construction, and the resulting hunting is what
looked like the robot dancing.

What has to be right, each measured today:

* **The relay's yaw offset cancels, and applying it mirrors the target.** The relay rotates
  the whole player about the world origin, and the depth image is rendered from that rotated
  camera, so the object has not moved relative to the robot. Applying the +90 deg offset to
  the camera put the card at the robot's LEFT: the left hand looked 0.54 m away when the
  right hand was 0.44 m away and the card was on the right the whole time. Every left-hand
  reach in this session aimed across the body at a mirrored point.
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
import os
import socket
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from clank.body import Pose  # noqa: E402
from clank.depth_stream import Stream  # noqa: E402
from clank.workspace import clamp_wrist
from clank.pixel3d import point_from_pixel  # noqa: E402
from clank.projection import Fov, project_point  # noqa: E402

SOCK = Path.home() / ".local/state/Clank/sessions/body/control.sock"
STATE = Path.home() / ".local/state/Clank/sessions/body/body_state.json"
FOV = Fov.from_degrees(70.0, 70.0, 57.11, 57.11)

FLAG_FALL_RISK = 1 << 2
FLAG_SATURATED = 1 << 0

#: Arm-only envelope, from `clank/workspace.MAX_OFFSET_M`. Past it the *legs* answer the
#: reach and the root walks, which invalidates the pixel and everything after it.
#: Trace every command to stdout. Off unless REACH_DEBUG is set.
DEBUG = bool(os.environ.get("REACH_DEBUG"))

ARM_ENVELOPE_M = 0.40
KEEPALIVE_S = 0.5

#: How often the reference is re-sent, in Hz. The plant is a learned tracker with a
#: 0.77-1.05 s settle and +75 % vertical overshoot, so what it needs is a reference that
#: moves *continuously* at a speed it can follow - not a sequence of steps.
STREAM_HZ = 20

#: How fast the reference is allowed to travel, in m/s. This is the number that matters and
#: it is a property of the plant, not of the path: 12 waypoints in 1.2 s is 0.30 m/s, twice
#: what the arm can track, and the gap between reference and hand is what the correction
#: loop then fights (measured: 28 direction reversals over three reaches). Cruise below the
#: plant's bandwidth and the gap never opens.
CRUISE_MPS = 0.26

#: Stop the stream this far short of the goal: the arm coasts past where the reference stops
#: (measured +0.06 to +0.11 m in z on a raised target), so aiming short is the cheapest way to
#: cancel it. Swept: stopping 12 mm short left 162 mm, 45 mm short left 123, 75 mm short left 97.
STREAM_STOP_M = 0.045

#: Deceleration distance: the reference slows over the last stretch instead of stopping dead.
#: Swept with a 0.045 m stop: no ramp 148 mm in 1.9 s, 0.18 m ramp 93 mm in 2.8 s, 0.26 m ramp
#: 83 mm in 3.3 s. 0.20 is the knee.
DECEL_M = 0.20

#: How long to keep commanding the target before measuring. Swept 0.55 / 1.05 / 1.5 s on a raised
#: target: pass-1 residual 160 / 120 / 146 mm, so waiting longer does not buy the last centimetres
#: and the arm does not settle back on its own. The correction pass is what closes it.
SETTLE_S = 0.55

#: Give up on the stream if it has not closed the distance in this long (s).
STREAM_TIMEOUT_S = 9.0

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
    # The body writes this file atomically, but the reader can still catch it missing: a
    # reset deletes and recreates it. "No state" is not an exception here, it is a reason
    # to wait, so the caller sees a plain failure instead of a traceback in the middle of a
    # reach.
    for _ in range(20):
        try:
            return json.loads(STATE.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.05)
    raise SystemExit(f"no body state at {STATE} (is the body running?)")


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


def head7(head) -> list[float]:
    """The camera pose as the intent carries it: position then quaternion, 7 numbers."""
    return [round(v, 5) for v in (*head.p, *head.q_xyzw)]


def command(target, yaw_deg: float, side: str = "right", camera=None) -> dict:
    """One reach. `camera` is the head pose the pixel was marked from, and it is resent
    every time: the executor pins the head to whatever it is given, so giving it the
    current pose each round re-locks the drift instead of stopping it."""
    extra = {} if camera is None else {"head": head7(camera)}
    return send(verb="reach", side=side, target=[round(v, 4) for v in target],
                yaw_deg=yaw_deg, **extra)


def card_pixel(st: dict, goal_rel, camera) -> tuple[float, float] | None:
    """Where the pinned goal lands in the *current* view, as a fraction of the frame.

    This is the check the loop was missing: a reach can converge to millimetres on a
    point the camera is no longer looking at. Projecting the goal back into the live
    camera says whether the marked pixel still means the same surface."""
    head = Pose(p=tuple(st["head"]["p"]), q_xyzw=tuple(st["head"]["q_xyzw"]))
    p = project_point(head, goal_rel, FOV)
    return None if p is None else p


def hold(target, yaw_deg: float, seconds: float, guard: float, root0,
         side: str = "right", camera=None) -> str:
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
    ap.add_argument("--corrections", type=int, default=1)
    ap.add_argument(
        "--no-smooth",
        action="store_true",
        help="use the old step-and-correct loop. Kept only to reproduce the measured "
        "behaviour; the smooth path is the default because the step loop is what made the "
        "arm jerk for ten seconds",
    )
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
    ap.add_argument(
        "--feedforward",
        dest="feedforward",
        action="store_true",
        help="aim at goal minus the measured wrist lever, to place the drawn hand in one motion. "
        "MEASURED BAD: the lever's direction follows the wrist, and the wrist is what the command "
        "moves, so this closes a loop with gain about one and inverts - a 0.15 m target drove the "
        "hand to z = -0.18, back to the rest pose, 481 mm out. Off by default; kept so the "
        "measurement can be repeated",
    )
    ap.add_argument(
        "--speed",
        type=float,
        default=CRUISE_MPS,
        help="reference speed in m/s; the plant tracks ~0.13-0.30 depending on the direction, "
        "and past its bandwidth the gap shows up as overshoot",
    )
    ap.add_argument("--dry-run", action="store_true", help="measure only, send nothing")
    ap.add_argument(
        "--grab",
        action="store_true",
        help="after the last command, close the hand *in place* with no gap, then retract "
        "and report where the object ended up",
    )
    ap.add_argument(
        "--retract",
        nargs=3,
        type=float,
        default=[0.16, -0.20, 0.05],
        metavar=("X", "Y", "Z"),
        help="pelvis-frame point to withdraw to after grabbing",
    )
    ap.add_argument(
        "--no-rest",
        action="store_true",
        help="skip the rest before reaching; skipping it starts from wherever the last "
        "reach left the arm, which is how one run began 0.10 m from its goal and stalled",
    )
    args = ap.parse_args()
    if args.world is None and args.target is None and (args.px is None or args.py is None):
        ap.error("give a pixel, --target X Y Z or --world X Y Z")

    if not args.dry_run and not args.no_rest:
        send(verb="rest")
        time.sleep(4.0)
    st = state()
    root0 = root()
    rest = wrist(args.side)
    head = Pose(p=tuple(st["head"]["p"]), q_xyzw=tuple(st["head"]["q_xyzw"]))

    camera = head
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
        # The relay's yaw offset CANCELS and must not be applied here. The relay rotates the
        # whole player by `yaw_offset_rad` about the world origin, so the game's camera sits
        # at R(head) looking along R(q_head); the depth image is rendered from there, and the
        # object it shows has not moved. Unprojecting with the rotated camera puts the point
        # in the rotated world, R(p_true), and converting that to the pelvis frame without
        # undoing R mirrors it. Measured: applying +90 deg put the card at the robot's LEFT
        # (y = +0.33, 0.94 m from the left hand, 0.44 m from the right hand mirrored) when it
        # is at y = -0.52, i.e. **0.44 m from the right hand** and 0.94 m from the left.
        # The raw head is the correct camera: R^-1(R(head) + R(q) d n) = head + q d n.
        camera = head
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
        point_rel = tuple(_median([q[k] for q in points]) for k in range(3))
        spread = max(math.dist(q, point_rel) for q in points)
        print(f"depth samples {len(points)}  spread {spread * 1000:.0f} mm")
        # `point_from_pixel` returns a PELVIS-frame point (it unprojects from a root-relative
        # head), so converting it with `to_rel` applies the root transform a second time.
        # Measured: that turned a card at (0.356, -0.524, 0.125) - 0.44 m from the right hand
        # - into (0.546, 0.345, -0.672), i.e. a point near the floor on the robot's left.
        goal = point_rel
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

    # Clamp the goal itself, once, before any motion. Measured 2026-11-04: a goal 0.04 m outside
    # `WRIST_BOX` in y made every waypoint clamp, so the reference walked past the wall forever,
    # the commanded point stopped advancing, and a 0.4 m reach ground on for nine seconds without
    # arriving. The clamp is a measured safety wall; a goal outside it is a goal that will not be
    # reached, and saying so once is better than fighting it 90 times.
    goal_clamped = clamp_wrist(goal)
    if goal_clamped != tuple(goal):
        print(f"goal  {fmt(goal)} is outside the measured workspace; "
              f"using {fmt(goal_clamped)} ({math.dist(goal, goal_clamped) * 1000:.0f} mm closer in)")
        goal = goal_clamped
        goal_world = to_world(goal, st)
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
    # ---- the reach: one continuous stream, corrections folded in ------------------------
    #
    # Everything below commands at STREAM_HZ and never stops between "the path" and "a
    # correction". The old shape - stream, then up to four separate commands each with its own
    # 1.1 s dwell - is what made the arm visibly stop, nudge, stop, nudge at the end, and it is
    # also why a correction landed on top of a still-settling arm.
    #
    # The reference walks toward the target at CRUISE_MPS. When it arrives, the hand is measured
    # once, and if it is short the target moves by the measured error and the reference starts
    # walking again - from wherever the hand actually is, not from where it was asked to be.
    dt = 1.0 / STREAM_HZ
    signs: list[int | None] = [None, None, None]
    reversals = 0
    clamp_hits = 0
    ticks = 0
    stalled_ticks = 0
    target_world = list(goal)
    best_residual = float("inf")
    best_target = list(target_world)
    # Frame rule: with feedforward the command is a LINK point, so the reference must start at the
    # link too (`reported - lever`); without it the command is the reported point and the
    # reference starts at the reported point. Mixing the two is a one-lever (0.18 m) error and was
    # measured to drive a 0.15 m target to z = -0.18, back at the rest pose.
    _q0 = st[f"{args.side}_hand"]["q_xyzw"]
    _off0 = lever(_q0) if args.feedforward else (0.0, 0.0, 0.0)
    ref_rel = [wrist(args.side)[k] - _off0[k] for k in range(3)]
    target = list(goal)                     # the legacy step loop's commanded point
    t_reach = time.monotonic()
    reason = None

    for attempt in range(max(0, args.corrections) + 1):
        issued += 1
        if args.no_smooth:                      # legacy, kept to reproduce the old behaviour
            goal_now = to_rel(goal_world, state())
            if issued > 1:
                target = [
                    target[k] + args.gain * (goal_now[k] - achieved[k]) for k in range(3)
                ]
            else:
                target = list(goal)
            command(target, args.yaw, args.side, camera)
            reason = hold(target, args.yaw, args.dwell, args.guard, root0, args.side, camera)
        else:
            command(clamp_wrist(tuple(ref_rel)), args.yaw, args.side, camera)  # pin once
            stalled = 0
            last = wrist(args.side)
            while True:
                # One state read per tick. Reading it twice (once for to_rel, once for wrist)
                # cost the loop a third of its rate: measured 14 Hz against a 20 Hz target.
                st_now = state()
                # Feedforward, and the reason this is not a "dance": the game draws the hand at
                # `link + lever`, where the lever is the 0.18 m VR offset rotated by the wrist's
                # own quaternion - measured swinging 0.05 -> 0.18 m as the wrist turns. The policy
                # puts the *link* on whatever it is told, to 2 mm, so telling it `goal - lever`
                # puts the drawn hand on `goal` in one motion. Without this the loop has to
                # discover the lever by measurement, one pass at a time.
                off = (
                    lever(st_now[f"{args.side}_hand"]["q_xyzw"])
                    if args.feedforward
                    else (0.0, 0.0, 0.0)
                )
                aim_rel = tuple(target_world[k] - off[k] for k in range(3))
                err = [aim_rel[k] - ref_rel[k] for k in range(3)]
                left = math.sqrt(sum(e * e for e in err))
                if left <= STREAM_STOP_M:
                    break
                # Slow down over the last DECEL_M instead of stopping dead at cruise speed.
                speed = args.speed * min(1.0, left / DECEL_M) if DECEL_M > 0 else args.speed
                advance = min(left, max(speed, 0.04) * dt)
                ref_rel = [ref_rel[k] + err[k] / left * advance for k in range(3)]
                way = clamp_wrist(tuple(ref_rel))
                if way != tuple(ref_rel):
                    clamp_hits += 1
                # No head pin on the stream. Measured: re-sending a constant head pose 20 times a
                # second makes the torso twist against it, which moves the root frame the target
                # lives in - the same 0.35 m reach landed at y = +0.15 instead of -0.28 and 500 mm
                # out, and dropping the pin from the stream took the same motion to 37 mm.
                command(way, args.yaw, args.side)
                ticks += 1
                if DEBUG:
                    st_d = state()
                    print(f"    [{ticks}] ref {fmt(tuple(ref_rel))} way {fmt(way)} "
                          f"hand {fmt(tuple(st_d[f'{args.side}_hand']['p']))} "
                          f"left {left * 1000:.0f} mm", flush=True)
                time.sleep(dt)
                now = tuple(st_now[f"{args.side}_hand"]["p"])
                for k in range(3):
                    delta = now[k] - last[k]
                    if abs(delta) > 0.003:
                        sign = 1 if delta > 0 else -1
                        if signs[k] is not None and sign != signs[k]:
                            reversals += 1
                        signs[k] = sign
                # The hand has stopped following the reference: the arm is saturated or the goal
                # is past the envelope. Walking the reference further just leans the body after
                # it, which is how one reach took the root 1.33 m and then 4.7 m.
                stalled = stalled + 1 if math.dist(now, last) < 0.0015 else 0
                last = now
                if stalled > int(0.7 / dt):
                    print(f"  reference stalled with {left * 1000:.0f} mm to go ({ticks} ticks)")
                    stalled_ticks += 1
                    break
                if time.monotonic() - t_reach > STREAM_TIMEOUT_S:
                    print(f"  stream timed out with {left * 1000:.0f} mm to go")
                    break
            # Settle *on the target*, still commanding at full rate. Commanding the raw goal here
            # instead of the corrected target is what produced a 0.25 m "error" that was the arm
            # still moving, and a lever that had swung somewhere else.
            for _ in range(int(SETTLE_S / dt)):
                st_now = state()
                off = (
                    lever(st_now[f"{args.side}_hand"]["q_xyzw"])
                    if args.feedforward
                    else (0.0, 0.0, 0.0)
                )
                # `target_world` is a PELVIS-frame point (it is `goal` plus corrections in that
                # frame). Running it through `to_rel` again subtracts the root transform a second
                # time - 0.787 m in z - and drove the arm down at the end of every pass. Measured:
                # the cruise tracked to 162 mm and the settle then threw it to 613 mm.
                settle_rel = tuple(target_world[k] - off[k] for k in range(3))
                command(clamp_wrist(settle_rel), args.yaw, args.side)
                ticks += 1
                time.sleep(dt)
            drift = math.dist(root(), root0)
            if drift > args.guard:
                reason = f"root moved {drift * 1000:.0f} mm"
        st = state()
        goal_now = to_rel(goal_world, st)
        achieved = tuple(st[f"{args.side}_hand"]["p"])
        residual = math.dist(achieved, goal_now)
        d = lever(st[f"{args.side}_hand"]["q_xyzw"])
        seen = card_pixel(st, goal_now, camera)
        drift_px = ""
        if seen is not None and args.px is not None:
            drift_px = (f"  card at px ({seen[0] * cw:4.0f}, {seen[1] * ch:4.0f})"
                        f" vs ({args.px:.0f}, {args.py:.0f})")
        print(f"pass {attempt + 1}: target {fmt(tuple(target_world))} -> "
              f"hand {fmt(achieved)}  residual {residual * 1000:5.1f} mm  lever {fmt(d)}{drift_px}")
        if reason:
            print(f"  ABORT after pass {attempt + 1}: {reason}")
            return 3 if "root" in reason else 4
        if residual <= args.tol:
            break
        if residual < best_residual:
            best_residual, best_target = residual, list(target_world)
        else:
            # The residual grew. Adding more of the same is how the commanded point walked to
            # -0.92 m; go back to the best point and stop instead of chasing.
            print(f"  residual grew to {residual * 1000:.1f} mm "
                  f"(best {best_residual * 1000:.1f} mm): backing off")
            target_world = best_target
            break
        # Fold the correction into the world target, and restart the reference from the hand's
        # measured position so the next pass is a continuation and not a jump.
        here_rel = to_rel(goal_world, st)
        target_world = [
            target_world[k] + args.gain * (here_rel[k] - achieved[k]) for k in range(3)
        ]
        # Restart in the same frame the aim uses.
        off = (
            lever(st[f"{args.side}_hand"]["q_xyzw"]) if args.feedforward else (0.0, 0.0, 0.0)
        )
        ref_rel = [achieved[k] - off[k] for k in range(3)]

    if not args.no_smooth:
        elapsed = time.monotonic() - t_reach
        print(f"  {ticks} commands over {elapsed:.2f} s "
              f"({ticks / max(1e-6, elapsed):.0f} Hz), "
              f"hand direction changes {reversals}"
              + (f", {clamp_hits} clamped into the workspace" if clamp_hits else "")
              + ("  [arm saturated]" if stalled_ticks else ""))

    if args.grab:
        # No gap. Measured 2026-11-04: closing the hand in a separate process let ~2 s pass
        # between the last reach command and the close, and the body wandered 0.09 m in that
        # window - the hand left the object before the fingers shut. `close_hand` already
        # pins the hand to its achieved pose, so the pose is not the problem; the gap is.
        st = state()
        held = tuple(st[f"{args.side}_hand"]["p"])
        send(verb=f"close_{args.side}")
        for _ in range(3):
            time.sleep(0.35)
            send(verb=f"close_{args.side}")
        st = state()
        print(f"\nGRAB: closed at {fmt(held)} -> hand {fmt(st[f'{args.side}_hand']['p'])} "
              f"closure {st['closure']}")
        for _ in range(10):
            send(verb="reach", side=args.side,
                 target=[round(v, 4) for v in args.retract], ttl_ms=2500)
            time.sleep(0.3)
        time.sleep(0.5)
        st = state()
        after = st[f"{args.side}_hand"]["p"]
        print(f"GRAB: retracted to {fmt(after)}  closure {st['closure']}")
        print(f"GRAB: the object was at {fmt(goal)} in the pelvis frame before the grab; "
              f"if it came with the hand it is now within a hand's width of {fmt(after)}")

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
