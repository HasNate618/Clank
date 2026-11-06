"""Reach an object with the emulated player's arms scaled up, streaming continuously.

Two things are combined here, both from measurement rather than preference:

* **Hands-only scaling.** The head is not scaled, so the camera stays exactly where every
  measurement in `research/` was taken and the stream's depth is already in the robot frame.
  The hand still travels `scale` metres of room per real metre, so the offset the arm has to
  cover is `(O - H) / scale` - the card's 0.485 m becomes 0.339 m.
* **A continuous stream instead of staged holds.** Holding each step for 1.05 s let the
  body's 0.4 s oscillation play out between every correction, which is what made one reach
  look like ten unrelated movements. At 6 Hz every tick is a small step toward a target that
  has moved slightly, so the arm tracks a path.

The lateral axis is why the original envelope could not do this: sideways gain here is ~0.2,
so a 0.20 m sideways correction needs a command several times larger. The integrator finds
that number by accumulating rather than by being told, and `MAX_LATERAL_M` is raised because
`touch_card.py` already survived 0.34 m lateral with a fall-risk check at every step.
"""

from __future__ import annotations

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
from clank.workspace import clamp_offset  # noqa: E402

SOCK = Path.home() / ".local/state/Clank/sessions/body/control.sock"
STATE = Path.home() / ".local/state/Clank/sessions/body/body_state.json"
FOV = Fov.from_degrees(70.0, 70.0, 57.11, 57.11)
FLAG_FALL_RISK = 1 << 2
MAX_M = 0.56
MAX_LATERAL_M = 0.34


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


def wrist() -> tuple[float, float, float]:
    return tuple(state()["right_hand"]["p"])


def to_world(p, st) -> tuple[float, float, float]:
    """Root-relative -> world, composing the root the same way the relay does."""
    w, x, y, z = st["root_q"]
    t = (2 * (y * p[2] - z * p[1]), 2 * (z * p[0] - x * p[2]), 2 * (x * p[1] - y * p[0]))
    rot = (
        p[0] + w * t[0] + (y * t[2] - z * t[1]),
        p[1] + w * t[1] + (z * t[0] - x * t[2]),
        p[2] + w * t[2] + (x * t[1] - y * t[0]),
    )
    root = st["root_p"]
    return (root[0] + rot[0], root[1] + rot[1], root[2] + rot[2])


def main() -> int:
    px, py = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (800, 410)
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.43
    duration = float(sys.argv[4]) if len(sys.argv) > 4 else 2.2
    hz = float(sys.argv[5]) if len(sys.argv) > 5 else 8.0

    send(verb="look", yaw_deg=0.0)
    time.sleep(2.5)
    send(verb="reach", side="right", forward_m=0.0, lateral_m=0.0, up_m=0.0,
         look_yaw_deg=0.0)
    time.sleep(2.5)
    rest = wrist()

    with Stream() as stream:
        frame = stream.read(timeout=2.0)
    if frame is None:
        print("no depth frame")
        return 1
    cw, ch = int(frame["color_width"]), int(frame["color_height"])
    head = Pose(p=tuple(state()["head"]["p"]), q_xyzw=tuple(state()["head"]["q_xyzw"]))
    # head not scaled -> the stream is already the robot frame
    obj = point_from_pixel(frame, px / cw, py / ch, head, FOV)
    if obj is None:
        print("no surface there")
        return 1
    need = math.dist(rest, obj)
    goal = [rest[k] + (obj[k] - rest[k]) / scale for k in range(3)]
    print(f"object {[round(v, 3) for v in obj]}  |O-H| {need:.3f} m  "
          f"-> needs {need / scale:.3f} m of arm at scale {scale}")

    # Run until the wrist is on the target rather than for a fixed window: the first two
    # attempts stopped mid-flight at 2.2 s and 4.0 s, 0.207 m and 0.174 m out, both times
    # with the arm still travelling. `duration` is now the ceiling.
    dt = 1.0 / hz
    offset = [0.0, 0.0, 0.0]
    tol = 0.045
    t0 = time.monotonic()
    best = float("inf")
    while time.monotonic() - t0 < duration:
        if bool(int(state().get("flags") or 0) & FLAG_FALL_RISK):
            print("  FALL RISK - stopping")
            break
        got = wrist()
        err = math.dist(got, goal)
        best = min(best, err)
        if err < tol:
            print(f"  converged: {err:.3f} m from the goal after "
                  f"{time.monotonic() - t0:.1f} s")
            break
        offset = [offset[k] + 0.7 * (goal[k] - got[k]) for k in range(3)]
        offset = list(
            clamp_offset(*offset, max_m=MAX_M, max_lateral_m=MAX_LATERAL_M)
        )
        send(verb="reach", side="right", forward_m=round(offset[0], 4),
             lateral_m=round(offset[1], 4), up_m=round(offset[2], 4),
             look_yaw_deg=0.0)
        time.sleep(dt)
    print(f"  best {best:.3f} m from the goal")
    # hold the last command so the body settles rather than drifting back to the clip
    for _ in range(int(2.0 / 0.5)):
        send(verb="reach", side="right", forward_m=round(offset[0], 4),
             lateral_m=round(offset[1], 4), up_m=round(offset[2], 4),
             look_yaw_deg=0.0)
        time.sleep(0.5)

    st = state()
    got = tuple(st["right_hand"]["p"])
    hand_world = to_world(got, st)
    obj_world = to_world(obj, st)
    print(f"  final offset {[round(v, 3) for v in offset]} "
          f"(magnitude {math.dist((0, 0, 0), offset):.3f})")
    print(f"  sim wrist {[round(v, 3) for v in got]}, travel {math.dist(rest, got):.3f} m")
    print(f"  sim gap to object {math.dist(got, obj):.3f} m")
    print(f"  emulated hand (game) {[round(v, 3) for v in hand_world]}, "
          f"gap {math.dist(hand_world, obj_world):.3f} m")
    print(f"  flags {st.get('flags')}, {time.monotonic() - t0:.1f} s")

    from PIL import Image

    import numpy as np

    with Stream() as stream:
        after = stream.read(timeout=2.0)
    cw, ch = int(after["color_width"]), int(after["color_height"])
    head = Pose(p=tuple(state()["head"]["p"]), q_xyzw=tuple(state()["head"]["q_xyzw"]))
    from clank.projection import project_point

    o_px = project_point(head, obj, FOV)
    rgba = np.frombuffer(bytes(after["color"]), dtype=np.uint8).reshape(ch, cw, 4)
    img = Image.fromarray(rgba[..., :3])
    img.save("/tmp/ho_reach_full.jpg", quality=93)
    if o_px is not None:
        ox, oy = int(o_px[0] * cw), int(o_px[1] * ch)
        img.crop((max(0, ox - 260), max(0, oy - 200), min(cw, ox + 260),
                  min(ch, oy + 220))).resize((520, 420)).save("/tmp/ho_reach_crop.jpg",
                                                             quality=93)
    print("  wrote /tmp/ho_reach_full.jpg and /tmp/ho_reach_crop.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
