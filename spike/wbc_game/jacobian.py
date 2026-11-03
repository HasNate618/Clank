"""Measure the 3x3 Jacobian from commanded wrist offset to achieved displacement.

Per-axis gains were not enough: streaming a purely forward path bowed 0.162 m off a
0.300 m line, because pushing forward also drags the wrist +0.138 in y and +0.085 in z
(measured 2026-11-02). That is an off-diagonal term, and no diagonal correction can
remove it.

So measure the whole thing: command a known offset on each axis from a common rest and
read where the wrist actually went. Column k of J is the achieved displacement per metre
of offset on axis k. Placement then becomes `offset = J^-1 * desired`, one command.

J is a local linearisation around one posture. It is not a model - report its condition
number and do not trust a large `J^-1` entry.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clank.workspace import clamp_offset  # noqa: E402

SOCK = Path.home() / ".local/state/Clank/sessions/body/control.sock"
STATE = Path.home() / ".local/state/Clank/sessions/body/body_state.json"
AXES = ("fwd", "lat", "up")


def send(**kw: object) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(2.0)
        s.connect(str(SOCK))
        s.sendall((json.dumps(kw) + "\n").encode())
        try:
            return json.loads(s.recv(255).decode())
        except (TimeoutError, json.JSONDecodeError):
            return {}


def wrist() -> tuple[float, float, float]:
    return tuple(json.loads(STATE.read_text())["right_hand"]["p"])


def go(offset, yaw: float, settle: float) -> tuple[float, float, float]:
    send(verb="reach", side="right", forward_m=round(offset[0], 4),
         lateral_m=round(offset[1], 4), up_m=round(offset[2], 4), look_yaw_deg=yaw)
    time.sleep(settle)
    return wrist()


def mat_inv3(m: list[list[float]]) -> list[list[float]] | None:
    (a, b, c), (d, e, f), (g, h, i) = m
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-9:
        return None
    return [
        [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
        [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
        [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=float, default=0.15, help="probe offset per axis")
    ap.add_argument("--settle", type=float, default=1.05)
    ap.add_argument("--yaw", type=float, default=0.0)
    args = ap.parse_args()

    send(verb="look", yaw_deg=args.yaw)
    time.sleep(2.5)
    rest = go((0.0, 0.0, 0.0), args.yaw, args.settle + 1.5)
    print(f"rest ({rest[0]:+.3f},{rest[1]:+.3f},{rest[2]:+.3f}), "
          f"probing {args.probe:.2f} m per axis")

    cols: list[list[float]] = []
    for k in range(3):
        offset = [0.0, 0.0, 0.0]
        offset[k] = args.probe
        got = go(offset, args.yaw, args.settle)
        col = [(got[j] - rest[j]) / args.probe for j in range(3)]
        cols.append(col)
        print(f"  {AXES[k]} {args.probe:+.2f} -> achieved "
              f"({got[0] - rest[0]:+.3f},{got[1] - rest[1]:+.3f},{got[2] - rest[2]:+.3f})"
              f"  column ({col[0]:+.2f},{col[1]:+.2f},{col[2]:+.2f})")

    # J maps offsets to displacement: displacement = J @ offset, so J's columns are cols.
    J = [[cols[c][r] for c in range(3)] for r in range(3)]
    print("J (rows = achieved axis, cols = commanded axis):")
    for row in J:
        print("   [" + " ".join(f"{v:+.2f}" for v in row) + "]")
    Jinv = mat_inv3(J)
    if Jinv is None:
        print("J is singular - no usable inverse")
        return 1
    norm = max(abs(v) for row in Jinv for v in row)
    print(f"J^-1 largest entry {norm:.2f}"
          + ("  <- large; a small command error becomes a big miss" if norm > 5 else ""))

    # Validate: ask for a diagonal move and use J^-1 to get there in one command.
    want = [0.20, 0.10, 0.12]
    offset = [sum(Jinv[r][c] * want[c] for c in range(3)) for r in range(3)]
    safe = clamp_offset(*offset)
    print(f"for a {want} move, J^-1 asks for "
          f"({offset[0]:+.3f},{offset[1]:+.3f},{offset[2]:+.3f})")
    if any(abs(safe[k] - offset[k]) > 1e-9 for k in range(3)):
        print(f"  REFUSING to send: clamped to ({safe[0]:+.3f},{safe[1]:+.3f},"
              f"{safe[2]:+.3f}). The raw inverse asks for more offset than the arm has")
        print("  been measured to survive; an unclamped vector of this shape put the body")
        print("  on the floor on 2026-11-03 (FALL detected). Iterate with a damped")
        print("  correction instead - see place_wrist.py.")
        return 0
    offset = safe
    got = go(offset, args.yaw, args.settle + 0.4)
    err = math.dist(got, [rest[j] + want[j] for j in range(3)])
    print(f"  achieved ({got[0]:+.3f},{got[1]:+.3f},{got[2]:+.3f})  error {err:.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
