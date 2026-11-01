#!/usr/bin/env python3
"""SONIC standing reach envelope, and whether FLAG_SATURATED means anything.

Run: `spike/e3/.venv/bin/python spike/wbc_game/reach_envelope_sonic.py`

Two questions, one run:

1. How far forward can the right wrist actually go from the standing clip?
   The "pick up JobBot's card" milestone needs a card that floats at roughly
   arm's length, and every previously measured target was inside +0.20 m.
2. Does `FLAG_SATURATED` distinguish "could not reach" from "ordinary tracking
   error"? It is raised whenever the residual exceeds the command's
   `tolerance_m`, whose contract default is 0.02 m - and measured aim error is
   0.035 m, so the answer may be "never".

Results are in `research/sonic-reach-envelope.md`.
"""
import sys, time
sys.path.insert(0, "/home/nate/Projects/Clank")
from clank.body import FLAG_SATURATED, AUTH_R_HAND, BodyCommand, hold
from spike.wbc.sonic_executor import SonicExecutor

SETTLE, REACH = 60, 100
TARGETS = [0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
TOLS = [0.02, 0.10, 0.20]

ex = SonicExecutor(verbose=False)
def run(cmd, n):
    for _ in range(n):
        st = ex.step(cmd, now_ns=time.monotonic_ns())
    return st
run(hold(), SETTLE)

print(f"{'fwd_m':>6} {'res_m':>7} " + " ".join(f"sat@{t:<5.2f}" for t in TOLS))
for fwd in TARGETS:
    cmd = BodyCommand(authority=AUTH_R_HAND,
                      right_hand=ex.hand_target("right", forward_m=fwd),
                      tolerance_m=0.02)   # flag is raised against cmd.tolerance_m
    st = run(cmd, REACH)
    sat = [bool(st.residual_right_m > t) for t in TOLS]
    print(f"{fwd:>6.2f} {st.residual_right_m:>7.3f} " +
          " ".join(f"{str(s):>8}" for s in sat))
    run(hold(), SETTLE // 2)
