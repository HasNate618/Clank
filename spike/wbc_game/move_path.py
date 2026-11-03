"""Move the wrist along a straight line by streaming targets, then judge the motion.

Discrete steps settle in ~1 s each, so aiming a demo with them reads as a stutter. The
arm's own oscillation has a period near 0.4 s (`research/wrist-response.md`), so a target
stream well below that rate should be followed rather than fought.

Smoothness is measured, not eyeballed: a straight-line move should stay near the line
(cross-track error) and should not repeatedly change direction (reversals). A stuttering
move scores badly on both even when it ends in the right place.
"""

from __future__ import annotations

import argparse
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
    return tuple(json.loads(STATE.read_text())["right_hand"]["p"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.30)
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--dz", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=1.5, help="seconds of motion")
    ap.add_argument("--hz", type=float, default=10.0, help="target stream rate")
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--feedback", action="store_true",
                    help="close the loop on measurement instead of streaming open-loop")
    args = ap.parse_args()

    send(verb="look", yaw_deg=args.yaw)
    time.sleep(2.5)
    send(verb="reach", side="right", forward_m=0.0, lateral_m=0.0, up_m=0.0,
         look_yaw_deg=args.yaw)
    time.sleep(2.5)
    start = wrist()
    delta = (args.dx, args.dy, args.dz)
    print(f"start ({start[0]:+.3f},{start[1]:+.3f},{start[2]:+.3f}) "
          f"delta ({delta[0]:+.2f},{delta[1]:+.2f},{delta[2]:+.2f}) "
          f"over {args.duration:.1f} s at {args.hz:.0f} Hz")

    if args.feedback:
        # Open-loop streaming bowed 0.162 m off a 0.300 m line, because a forward offset
        # also drags the wrist sideways (measured Jacobian column [+0.72, +0.67, +0.11]).
        # Feedback measures that and removes it: no Jacobian, no inversion, no model of
        # the coupling - just an integrator that is damped enough to stay stable for any
        # positive gain (that is what `0.6` buys, see research/wrist-response.md).
        LAMBDA = 0.6
        gain = [1.0, 1.0, 1.0]
        offset = [0.0, 0.0, 0.0]
        n = max(2, int(args.duration * args.hz))
        dt = args.duration / n
        samples_fb: list[tuple[float, tuple[float, float, float]]] = []
        t0 = time.monotonic()
        for i in range(1, n + 1):
            frac = i / n
            want = tuple(start[k] + delta[k] * frac for k in range(3))
            got = wrist()
            err = [want[k] - got[k] for k in range(3)]
            if i > 1:
                pass
            offset = [offset[k] + LAMBDA * err[k] / gain[k] for k in range(3)]
            send(verb="reach", side="right",
                 forward_m=round(offset[0], 4), lateral_m=round(offset[1], 4),
                 up_m=round(offset[2], 4), look_yaw_deg=args.yaw)
            for _ in range(max(1, int(round(dt / 0.02)))):
                samples_fb.append((time.monotonic() - t0, wrist()))
                time.sleep(0.02)
        for _ in range(60):
            samples_fb.append((time.monotonic() - t0, wrist()))
            time.sleep(0.02)
        samples = samples_fb

    n = max(2, int(args.duration * args.hz))
    dt = args.duration / n
    samples: list[tuple[float, tuple[float, float, float]]] = []
    t0 = time.monotonic()
    for i in ([] if args.feedback else range(1, n + 1)):
        frac = i / n
        send(verb="reach", side="right",
             forward_m=round(delta[0] * frac, 4),
             lateral_m=round(delta[1] * frac, 4),
             up_m=round(delta[2] * frac, 4),
             look_yaw_deg=args.yaw)
        # Sample a few times between targets so the trace is finer than the stream.
        for _ in range(max(1, int(round(dt / 0.02)))):
            samples.append((time.monotonic() - t0, wrist()))
            time.sleep(0.02)
    # Let it settle, and keep sampling.
    for _ in range(60):
        samples.append((time.monotonic() - t0, wrist()))
        time.sleep(0.02)

    end = samples[-1][1]
    want = tuple(start[k] + delta[k] for k in range(3))
    final_err = math.dist(end, want)
    print(f"end   ({end[0]:+.3f},{end[1]:+.3f},{end[2]:+.3f})  "
          f"error {final_err:.3f} m")

    # Cross-track error: distance from each sample to the start->target line.
    axis = [want[k] - start[k] for k in range(3)]
    length = math.dist(start, want)
    unit = [v / length for v in axis] if length > 1e-9 else [0.0, 0.0, 0.0]
    worst = 0.0
    for _, p in samples:
        rel = [p[k] - start[k] for k in range(3)]
        along = sum(rel[k] * unit[k] for k in range(3))
        perp = [rel[k] - along * unit[k] for k in range(3)]
        worst = max(worst, math.sqrt(sum(v * v for v in perp)))
    print(f"worst cross-track deviation {worst:.3f} m over {length:.3f} m of travel")

    # Direction reversals along the line: a stuttering move reverses; a smooth one does not.
    along_track = []
    for _, p in samples:
        rel = [p[k] - start[k] for k in range(3)]
        along_track.append(sum(rel[k] * unit[k] for k in range(3)))
    reversals = 0
    step_sign = 0
    for a, b in zip(along_track, along_track[1:]):
        d = b - a
        if abs(d) < 0.002:
            continue
        sign = 1 if d > 0 else -1
        if step_sign and sign != step_sign:
            reversals += 1
        step_sign = sign
    print(f"direction reversals along the path: {reversals}")

    speeds = []
    for (t1, p1), (t2, p2) in zip(samples, samples[1:]):
        dt_s = t2 - t1
        if dt_s > 1e-6:
            speeds.append(math.dist(p1, p2) / dt_s)
    if speeds:
        mean = sum(speeds) / len(speeds)
        var = sum((s - mean) ** 2 for s in speeds) / len(speeds)
        print(f"speed mean {mean:.3f} m/s, sd {var ** 0.5:.3f}, peak {max(speeds):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
