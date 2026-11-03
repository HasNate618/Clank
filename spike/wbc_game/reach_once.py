"""One command to a target, using the measured Jacobian - no staging, no corrections.

The previous approach staged the motion and corrected it up to twice per stage, which is
10-12 commands over ~15 s. The body is not slow (63 % of a move arrives in 0.10 s, settling
in ~1.05 s); the wandering is the corrections fighting the lateral axis, which overshoots
and springs back. Each correction therefore drags the arm somewhere before the next one
pulls it home, and that is what "weird movements" looks like.

`research/wrist-response.md` measured the 3x3 mapping from commanded offset to achieved
displacement. That makes placement a single pre-computed command:

    offset = J^-1 * (target - rest)

with a damping factor and the envelope clamp. The trade is accuracy for cleanliness: one
motion, and whatever residual the linear model leaves. This script prints both, and the
number of commands, so the trade is visible rather than argued about.
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

#: Measured at the standing rest posture, `spike/wbc_game/jacobian.py`, 2026-11-03.
#: Rows are achieved axes, columns are commanded axes.
J = [[0.72, 0.50, 0.03], [0.67, 0.71, 0.90], [0.11, 0.10, 0.86]]
#: Below 1 so an over-large inverse cannot drive the whole vector; the clamp backs it up.
LAMBDA = 0.85
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


def risky() -> bool:
    return bool(int(state().get("flags") or 0) & FLAG_FALL_RISK)


def inv3(m: list[list[float]]) -> list[list[float]]:
    (a, b, c), (d, e, f), (g, h, i) = m
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return [
        [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
        [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
        [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
    ]


def main() -> int:
    px, py = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (782, 432)
    up_bias = float(sys.argv[3]) if len(sys.argv) > 3 else 0.07

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
    u, v = px / cw, py / ch
    head = Pose(p=tuple(state()["head"]["p"]), q_xyzw=tuple(state()["head"]["q_xyzw"]))
    point = point_from_pixel(frame, u, v, head, FOV)
    if point is None:
        print("no surface there")
        return 1
    dist = math.dist(rest, point)
    unit = [(point[k] - rest[k]) / dist for k in range(3)]
    goal = [point[k] - 0.08 * unit[k] for k in range(3)]
    goal[2] += up_bias
    delta = [goal[k] - rest[k] for k in range(3)]
    print(f"target ({goal[0]:+.3f},{goal[1]:+.3f},{goal[2]:+.3f}), "
          f"{math.dist(rest, goal):.3f} m from rest")

    Jinv = inv3(J)
    offset = [LAMBDA * sum(Jinv[r][c] * delta[c] for c in range(3)) for r in range(3)]
    clamped = clamp_offset(*offset, max_m=MAX_M, max_lateral_m=MAX_LATERAL_M)
    if any(abs(clamped[k] - offset[k]) > 1e-9 for k in range(3)):
        print(f"  J^-1 asked ({offset[0]:+.3f},{offset[1]:+.3f},{offset[2]:+.3f}), "
              f"clamped to ({clamped[0]:+.3f},{clamped[1]:+.3f},{clamped[2]:+.3f})")

    t0 = time.monotonic()
    send(verb="reach", side="right", forward_m=round(clamped[0], 4),
         lateral_m=round(clamped[1], 4), up_m=round(clamped[2], 4), look_yaw_deg=0.0)
    time.sleep(2.2)
    got = wrist()
    print(f"ONE command -> wrist ({got[0]:+.3f},{got[1]:+.3f},{got[2]:+.3f})")
    print(f"  error {math.dist(goal, got):.3f} m, took {time.monotonic() - t0:.1f} s, "
          f"flags {state().get('flags')}")
    if risky():
        print("  FALL RISK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
