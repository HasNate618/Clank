#!/usr/bin/env python3
"""One reach, then one measurement: how far is the hand from the target?

No correction passes, no head pin, and nothing that could move the body on purpose. The only
question is the gap, plus how much the head moved while the arm did.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

sys.path.insert(0, "/home/nate/Projects/Clank")
sys.path.insert(0, "/home/nate/Projects/Clank/spike/wbc_game")

import reach_vr as rv  # noqa: E402


def look(st: dict) -> tuple[float, float, float]:
    return tuple(st["head"]["p"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz", type=float, nargs=3, metavar=("X", "Y", "Z"))
    ap.add_argument("--side", default="right", choices=("left", "right"))
    ap.add_argument("--speed", type=float, default=0.26)
    ap.add_argument("--settle", type=float, default=1.2)
    ap.add_argument("--no-clamp", action="store_true")
    ap.add_argument("--no-rest", action="store_true")
    ap.add_argument(
        "--z-gain",
        type=float,
        default=1.0,
        help="scale the aim in height about the resting hand, to cancel the plant's vertical "
        "gain. 1.0 aims at the target; 1.58 divides the climb by the measured gain, so "
        "aim_z = rest_z + (goal_z - rest_z) / gain",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    target = tuple(args.xyz)
    if not args.no_rest:
        rv.send(verb="rest")
        time.sleep(4.0)

    st0 = rv.state()
    start = tuple(st0[f"{args.side}_hand"]["p"])
    goal = target
    if args.z_gain != 1.0:
        aim = (goal[0], goal[1], start[2] + (goal[2] - start[2]) / args.z_gain)
    else:
        aim = goal
    head0 = look(st0)
    root0 = tuple(st0["root_p"])
    dt = 0.05
    ref = list(start)
    t0 = time.monotonic()
    ticks = 0
    for _ in range(200):
        err = [aim[k] - ref[k] for k in range(3)]
        left = math.sqrt(sum(e * e for e in err))
        if left <= 0.03:
            break
        v = max(args.speed * min(1.0, left / 0.20), 0.04)
        ref = [ref[k] + err[k] / left * min(left, v * dt) for k in range(3)]
        cmd = tuple(ref) if args.no_clamp else rv.clamp_wrist(tuple(ref))
        rv.command(cmd, 0.0, args.side)          # no head, deliberately
        ticks += 1
        time.sleep(dt)
    for _ in range(int(args.settle / dt)):
        cmd = aim if args.no_clamp else rv.clamp_wrist(aim)
        rv.command(cmd, 0.0, args.side)
        ticks += 1
        time.sleep(dt)
    st1 = rv.state()
    hand = tuple(st1[f"{args.side}_hand"]["p"])
    head1 = look(st1)
    root1 = tuple(st1["root_p"])
    gap = math.dist(hand, target)
    res = {
        "target": [round(v, 3) for v in goal],
        "aim": [round(v, 3) for v in aim],
        "hand": [round(v, 3) for v in hand],
        "gap_mm": round(gap * 1000, 1),
        "dx": [round(target[k] - hand[k], 3) for k in range(3)],
        "head_moved_mm": round(math.dist(head0, head1) * 1000, 1),
        "root_moved_mm": round(math.dist(root0, root1) * 1000, 1),
        "hand_travel_mm": round(math.dist(start, hand) * 1000, 1),
        "seconds": round(time.monotonic() - t0, 2),
        "ticks": ticks,
    }
    pathlib.Path("/home/nate/.local/state/Clank/pick-pixel/last_reach.json").write_text(
        json.dumps(res)
    )
    if args.json:
        print(json.dumps(res))
    else:
        print(f"target {res['target']}  hand {res['hand']}")
        print(f"GAP {res['gap_mm']} mm   (dx dy dz = {res['dx']})")
        print(f"head moved {res['head_moved_mm']} mm, root {res['root_moved_mm']} mm, "
              f"hand travelled {res['hand_travel_mm']} mm in {res['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
