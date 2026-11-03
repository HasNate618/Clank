"""Reach a point as one continuous motion: interpolate the target, correct every tick.

Staging held the arm at each step for 1.05 s, long enough for the body's own 0.4 s
oscillation to play out and for the weak lateral axis to spring back - which is what made
the reach look like a series of unrelated movements. Streaming the same corrections at 6 Hz
removes the gaps: every tick is a small step toward a target that has moved slightly, so
the arm tracks a path instead of answering a sequence of questions.

No Jacobian: a linear inverse is unusable here (measured twice - it asked for 2.9 m of
offset for a 0.47 m reach). The integrator only needs the sign to be right and the damping
to be under 1, and it measures the coupling instead of predicting it.
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


def main() -> int:
    px, py = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (782, 432)
    up_bias = float(sys.argv[3]) if len(sys.argv) > 3 else 0.07
    duration = float(sys.argv[4]) if len(sys.argv) > 4 else 1.6
    hz = float(sys.argv[5]) if len(sys.argv) > 5 else 6.0

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
    point = point_from_pixel(frame, px / cw, py / ch, head, FOV)
    if point is None:
        print("no surface there")
        return 1
    dist = math.dist(rest, point)
    unit = [(point[k] - rest[k]) / dist for k in range(3)]
    goal = [point[k] - 0.08 * unit[k] for k in range(3)]
    goal[2] += up_bias
    print(f"goal ({goal[0]:+.3f},{goal[1]:+.3f},{goal[2]:+.3f}) "
          f"{math.dist(rest, goal):.3f} m away, over {duration:.1f} s at {hz:.0f} Hz")

    n = max(2, int(duration * hz))
    dt = duration / n
    offset = [0.0, 0.0, 0.0]
    trace: list[tuple[float, tuple[float, float, float]]] = []
    t0 = time.monotonic()
    for i in range(1, n + 1):
        if bool(int(state().get("flags") or 0) & FLAG_FALL_RISK):
            print("  FALL RISK - stop")
            break
        frac = i / n
        want = [rest[k] + frac * (goal[k] - rest[k]) for k in range(3)]
        got = wrist()
        offset = [offset[k] + 0.6 * (want[k] - got[k]) for k in range(3)]
        offset = list(
            clamp_offset(*offset, max_m=MAX_M, max_lateral_m=MAX_LATERAL_M)
        )
        send(verb="reach", side="right", forward_m=round(offset[0], 4),
             lateral_m=round(offset[1], 4), up_m=round(offset[2], 4),
             look_yaw_deg=0.0)
        for _ in range(max(1, int(round(dt / 0.02)))):
            trace.append((time.monotonic() - t0, wrist()))
            time.sleep(0.02)
    for _ in range(70):
        trace.append((time.monotonic() - t0, wrist()))
        time.sleep(0.02)

    final = wrist()
    err = math.dist(goal, final)
    print(f"  final ({final[0]:+.3f},{final[1]:+.3f},{final[2]:+.3f}) error {err:.3f} m "
          f"in {time.monotonic() - t0:.1f} s")

    # Reversals along the overall direction: a stuttering reach changes its mind.
    axis = [goal[k] - rest[k] for k in range(3)]
    length = math.dist(rest, goal)
    u_ = [a / length for a in axis]
    proj = [sum((p[k] - rest[k]) * u_[k] for k in range(3)) for _, p in trace]
    rev = 0
    sign = 0
    for a, b in zip(proj, proj[1:]):
        d = b - a
        if abs(d) < 0.004:
            continue
        s = 1 if d > 0 else -1
        if sign and s != sign:
            rev += 1
        sign = s
    print(f"  reversals along the path: {rev}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
