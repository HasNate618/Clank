"""Place the right wrist at a point, using the measured per-axis gains.

The 3D corrector this replaces added the raw error each iteration and slept 1.6 s. Both
halves were wrong. `research/wrist-response.md` measured what the axes actually do:

    forward  gain 1.10, settles 1.05 s    up    gain 1.00, 75 % transient overshoot
    lateral  gain 0.15-0.32, springs back  -> the corrector was asking for lateral moves
                                             the arm does not deliver, and large lateral
                                             requests made the body lean the wrist the
                                             other way (0.46 m swing on an "up" command)

So: divide each axis by its gain, sleep the measured settle time, then correct once from
the measured error - again gain-scaled. Two commands instead of six.

Refuses to ask for lateral offsets beyond `MAX_LATERAL`, because commanding 4x the desired
move to beat a 0.25 gain is exactly what provokes the lean.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

SOCK = Path.home() / ".local/state/Clank/sessions/body/control.sock"
STATE = Path.home() / ".local/state/Clank/sessions/body/body_state.json"

#: Fallback only. The response is **posture-dependent**, so this is measured in situ
#: rather than trusted: `research/wrist-response.md` recorded lateral 0.15-0.32 from a
#: tucked arm, while a fixed matrix built from that made an extended-arm loop diverge 5x
#: (a +0.300 lateral command produced +0.399 of motion there, gain ~1.3).
GAIN = (1.10, 1.00, 1.00)
#: Share of the remaining error to correct per step once the gain is known. Below 1 it
#: stays stable for any positive gain, which matters because the gain is not constant.
LAMBDA = 0.7
#: A lateral request beyond this is asking the arm to lean, not to move.
MAX_LATERAL = 0.30


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


def dist(a, b) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.0, help="forward, metres from rest")
    ap.add_argument("--dy", type=float, default=0.0, help="lateral, + is robot left")
    ap.add_argument("--dz", type=float, default=0.0, help="up")
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--settle", type=float, default=1.05)
    ap.add_argument("--tries", type=int, default=5)
    args = ap.parse_args()

    delta = (args.dx, args.dy, args.dz)
    send(verb="look", yaw_deg=args.yaw)
    time.sleep(2.5)
    send(verb="reach", side="right", forward_m=0.0, lateral_m=0.0, up_m=0.0,
         look_yaw_deg=args.yaw)
    time.sleep(args.settle + 1.5)
    rest = wrist()
    target = tuple(rest[k] + delta[k] for k in range(3))
    print(f"rest ({rest[0]:+.3f},{rest[1]:+.3f},{rest[2]:+.3f}) -> "
          f"target ({target[0]:+.3f},{target[1]:+.3f},{target[2]:+.3f}), "
          f"{dist(rest, target):.3f} m")

    gain = list(GAIN)
    offset = list(delta)
    t0 = time.monotonic()
    for i in range(args.tries):
        want = [min(v, MAX_LATERAL) if k == 1 else v for k, v in enumerate(offset)]
        want = [max(v, -MAX_LATERAL) if k == 1 else v for k, v in enumerate(want)]
        send(verb="reach", side="right", forward_m=round(want[0], 3),
             lateral_m=round(want[1], 3), up_m=round(want[2], 3),
             look_yaw_deg=args.yaw)
        time.sleep(args.settle)
        got = wrist()
        err = [target[k] - got[k] for k in range(3)]
        print(f"  cmd ({want[0]:+.3f},{want[1]:+.3f},{want[2]:+.3f}) -> "
              f"wrist ({got[0]:+.3f},{got[1]:+.3f},{got[2]:+.3f})  "
              f"err {dist(target, got):.3f} m  ({time.monotonic() - t0:.1f} s)")
        if i == 0:
            # First command is the probe: it measures this posture's gain per axis.
            moved = [got[k] - rest[k] for k in range(3)]
            # Capped at 1.6: a single probe with a small lateral command measured 2.73,
            # which is probably coupling rather than gain, and it shrinks each correction
            # to the point where the axis crawls. Convergence is monotonic either way; a
            # cap makes it fast without making it oscillate.
            gain = [max(0.1, min(1.6, (moved[k] / want[k]) if abs(want[k]) > 1e-6 else 1.0))
                    for k in range(3)]
            print(f"  measured gain {gain[0]:.2f}, {gain[1]:.2f}, {gain[2]:.2f}")
        if dist(target, got) <= args.tol:
            print(f"  within {args.tol} m after {i + 1} command(s)")
            return 0
        offset = [offset[k] + LAMBDA * err[k] / gain[k] for k in range(3)]
    print("  did not converge")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
