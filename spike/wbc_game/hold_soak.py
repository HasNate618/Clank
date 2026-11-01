#!/usr/bin/env python3
"""P2's open soak: does freeze-hold actually hold?

Run: `spike/e3/.venv/bin/python spike/wbc_game/hold_soak.py`

Compares three ways to stand still for 20 s, through the real executor:

- `hold()`                  FREEZE_HOLD_AUTH, no poses -> `channel_pose` returns
                            the LAST ACHIEVED pose, re-read every tick.
- `hold(state=snapshot)`    the same command with one frozen snapshot, i.e. a
                            pose latched once at the start of the hold.
- `rest()`                  empty authority -> the clip's anchor frame.

Results are in `research/p2-soak.md`.
"""
import sys
import time

sys.path.insert(0, "/home/nate/Projects/Clank")

import numpy as np

from clank.body import hold, rest
from spike.wbc.sonic_executor import SonicExecutor

SETTLE, TICKS = 60, 1000          # 1.2 s settle, then 20 s measured


def measure(label, build, frame=0):
    ex = SonicExecutor(verbose=False)
    ex.hold_frame = frame
    ex.runner.reset(ex.motion, start=frame)
    for _ in range(SETTLE):
        ex.step(rest(), now_ns=time.monotonic_ns())
    snap = ex.last_state
    rw = []
    for i in range(TICKS):
        rw.append(np.array(ex.step(build(i, snap), now_ns=time.monotonic_ns()).right_hand.p))
    rw = np.array(rw)
    d = np.linalg.norm(np.diff(rw, axis=0), axis=1) / 0.02
    tail = d[len(d) // 2:]
    print(f"{label:38s} net {np.linalg.norm(rw[-1] - rw[0]):6.4f} m  "
          f"rms_last10 {np.sqrt(np.mean(tail ** 2)):.6f} m/s  "
          f"p95 {np.percentile(tail, 95):.6f}  fell {ex.fell}")


measure("hold()  trailing (last_state)", lambda i, s: hold())
measure("hold(state=snapshot)  latched", lambda i, s: hold(state=s))
measure("rest()  clip anchor", lambda i, s: rest())
