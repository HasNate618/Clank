"""How does the wrist actually respond to a step command?

Every servo loop in this project has used `time.sleep(1.6)` between commands, which was
never measured - it was chosen to be obviously longer than anything the body could take.
That is the wrong way round: if the arm settles in 0.4 s the demo can move four times
faster and stream a path instead of snapping, and if it takes 3 s then 1.6 s was too
short and every loop has been correcting a body that had not finished moving.

Measures rise time (to 63 %), settle time (inside 5 % of final), overshoot, and the
steady-state ratio of achieved to commanded displacement.
"""

from __future__ import annotations

import json
import math
import socket
import sys
import time
from pathlib import Path

SOCK = Path.home() / ".local/state/Clank/sessions/body/control.sock"
STATE = Path.home() / ".local/state/Clank/sessions/body/body_state.json"


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
    st = json.loads(STATE.read_text())
    return tuple(st["right_hand"]["p"])


def main() -> int:
    step = float(sys.argv[3]) if len(sys.argv) > 3 else 0.20
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    axis = sys.argv[1] if len(sys.argv) > 1 else "fwd"
    if axis not in ("fwd", "lat", "up"):
        raise SystemExit("axis must be fwd, lat or up")
    fwd, lat, up = ((step, 0.0, 0.0) if axis == "fwd" else
                    (0.0, step, 0.0) if axis == "lat" else (0.0, 0.0, step))

    send(verb="look", yaw_deg=0.0)
    time.sleep(2.5)
    send(verb="reach", side="right", forward_m=0.0, lateral_m=0.0, up_m=0.0,
         look_yaw_deg=0.0)
    time.sleep(3.0)
    start = wrist()
    print(f"rest wrist ({start[0]:+.3f},{start[1]:+.3f},{start[2]:+.3f}), "
          f"stepping {axis} {step:+.3f} m, sampling {seconds:.1f} s")

    send(verb="reach", side="right", forward_m=fwd, lateral_m=lat, up_m=up,
         look_yaw_deg=0.0)
    t0 = time.monotonic()
    samples: list[tuple[float, float]] = []
    while True:
        t = time.monotonic() - t0
        if t > seconds:
            break
        p = wrist()
        idx = {"fwd": 0, "lat": 1, "up": 2}[axis]
        samples.append((t, p[idx] - start[idx]))
        time.sleep(0.02)

    final = samples[-1][1]
    target = step
    print(f"  final displacement {final:+.3f} m of {target:+.3f} commanded "
          f"({100.0 * final / target:.0f} %)")

    def first(pred) -> float | None:
        for t, d in samples:
            if pred(d):
                return t
        return None

    rise = first(lambda d: d >= 0.63 * final)
    peak = max(d for _, d in samples)
    settle = None
    band = 0.05 * abs(final)
    for i, (t, d) in enumerate(samples):
        if all(abs(dd - final) <= band for _, dd in samples[i:]):
            settle = t
            break
    print(f"  rise to 63 % of final: {'—' if rise is None else f'{rise:.2f} s'}")
    print(f"  settle inside 5 %:     {'— never' if settle is None else f'{settle:.2f} s'}")
    print(f"  peak {peak:+.3f} m -> overshoot {100.0 * (peak - final) / final:+.1f} %")
    print("  trace (每 0.2 s): " .replace("每 ", "every ")
          + " ".join(f"{d:+.3f}" for _, d in samples[::10]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
