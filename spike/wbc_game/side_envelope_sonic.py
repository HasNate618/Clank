#!/usr/bin/env python3
"""Where can the wrist go *sideways* and *up*? The forward sweep is done.

Run: `MUJOCO_GL=disable spike/e3/.venv/bin/python spike/wbc_game/side_envelope_sonic.py`

`research/sonic-reach-envelope.md` swept `forward_m` only and said so: "No
measurement of how far *laterally* the wrist can go at a reachable depth." Then a
live servo run asked for `lateral 0.59` and the loop diverged
(`research/live-pixel-servo.md`), which is exactly the unmeasured direction.

A workspace clamp cannot be written against a limit nobody has measured, so measure
it first. Two sweeps at the reachable depth `forward_m = 0.45`:

1. `lateral_m` 0.00 .. 0.60, `up_m` fixed at 0.30
2. `up_m` -0.10 .. 0.60, `lateral_m` fixed at 0.20

`hand_target` takes offsets from the standing clip's wrist, which rests at
`(0.079, -0.308, -0.203)` m root-relative, so an offset is not an achieved position
and the achieved wrist is what gets reported.

Watch `FLAG_FALL_RISK` throughout: a lateral limit that topples the robot is a limit,
but it is not a workspace.
"""
import sys
import time

sys.path.insert(0, "/home/nate/Projects/Clank")
from clank.body import AUTH_R_HAND, FLAG_FALL_RISK, FLAG_SATURATED, BodyCommand, hold
from spike.wbc.sonic_executor import SonicExecutor

SETTLE, REACH = 60, 100
FWD, UP0, LAT0 = 0.45, 0.30, 0.20
LATERALS = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
UPS = [-0.10, 0.00, 0.20, 0.40, 0.60]

ex = SonicExecutor(verbose=False)


def run(cmd, n):
    st = None
    for _ in range(n):
        st = ex.step(cmd, now_ns=time.monotonic_ns())
    return st


def sweep(label, axis, values, base):
    print(f"\n{label}")
    print(f"{axis:>9} {'x':>7} {'y':>7} {'z':>7} {'res_m':>7} {'flags':>6}  notes")
    worst = 0
    for v in values:
        kw = dict(base)
        kw[axis] = v
        cmd = BodyCommand(
            authority=AUTH_R_HAND,
            right_hand=ex.hand_target("right", **kw),
            tolerance_m=0.02,
        )
        st = run(cmd, REACH)
        if st is None:
            continue
        p = st.right_hand.p
        notes = []
        if st.has_flag(FLAG_FALL_RISK):
            notes.append("FALL_RISK")
            worst = max(worst, 2)
        elif st.has_flag(FLAG_SATURATED):
            notes.append("saturated")
            worst = max(worst, 1)
        print(
            f"{v:>9.2f} {p[0]:>7.3f} {p[1]:>7.3f} {p[2]:>7.3f} "
            f"{st.residual_right_m:>7.3f} {st.flags:>6}  {' '.join(notes)}"
        )
        run(hold(), SETTLE // 2)
    return worst


run(hold(), SETTLE)
a = sweep(
    "lateral sweep, forward 0.45, up 0.30", "lateral_m", LATERALS,
    dict(forward_m=FWD, up_m=UP0),
)
run(hold(), SETTLE)
b = sweep(
    "vertical sweep, forward 0.45, lateral 0.20", "up_m", UPS,
    dict(forward_m=FWD, lateral_m=LAT0),
)
print(f"\nworst flags: lateral {a}, vertical {b} (2 = fall risk appeared)")
