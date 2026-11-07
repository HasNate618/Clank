#!/usr/bin/env python3
"""Does the pixel servo actually drive the hand onto a pixel? Measured, not assumed.

Run: `spike/e3/.venv/bin/python spike/wbc_game/servo_sim_probe.py`

Unit tests prove `unproject` inverts `project_point`. Only the sim proves the body
follows it, so this closes the loop through the real MuJoCo + SONIC stack:

    project the achieved wrist -> pixel error -> servo_step -> command the wrist

Results are in `research/pixel-servo-sim.md`. Read that before trusting a pixel.

The FOV is a fixture (symmetric 90x90) because the compositor's live FOV is not
measured yet. Target and projection share one FOV, so the loop is self-consistent:
this measures *convergence*, not absolute pixel accuracy against the game.
"""
import sys
import time

sys.path.insert(0, "/home/nate/Projects/Clank")

import numpy as np

from clank.body import AUTH_R_HAND, BodyCommand, Pose, hold
from clank.projection import (
    Fov,
    _rotate_inverse,
    pixel_error,
    project_point,
    servo_step,
)
from spike.wbc.sonic_executor import R_WRIST, SonicExecutor, _pose_from

FOV = Fov.from_degrees(45.0, 45.0, 45.0, 45.0)
SETTLE, MAX_TICKS, DEADBAND = 60, 120, 0.005


def main() -> None:
    ex = SonicExecutor(verbose=False)
    pts, orns = ex._clip_targets()
    clip_r = _pose_from(
        pts[3 * R_WRIST:3 * R_WRIST + 3], orns[4 * R_WRIST:4 * R_WRIST + 4]
    )

    def run(cmd, n):
        for _ in range(n):
            st = ex.step(cmd, now_ns=time.monotonic_ns())
        return st

    run(hold(), SETTLE)
    st = ex.last_state
    if st is None:
        raise SystemExit("no state yet")
    rel = _rotate_inverse(st.head.q_xyzw, tuple(np.subtract(st.right_hand.p, st.head.p)))
    print(f"head  root-relative {tuple(round(v, 3) for v in st.head.p)}")
    print(f"wrist root-relative {tuple(round(v, 3) for v in st.right_hand.p)}")
    print(f"wrist in head frame fwd={rel[0]:.3f} left={rel[1]:.3f} up={rel[2]:.3f}")
    print(f"wrist pixel {project_point(st.head, st.right_hand.p, FOV)}")
    print(
        f"gaze toward wrist yaw={np.degrees(np.arctan2(rel[1], rel[0])):.1f} "
        f"pitch={np.degrees(np.arctan2(rel[2], rel[0])):.1f} deg\n"
    )

    print(f"{'target px':>13} {'depth':>6} {'start':>7} {'min':>7} {'final':>7} "
          f"{'|cmd-ach|':>10} {'ticks':>6} {'done':>5} {'fell':>5}")
    for target in ((0.5, 0.6), (0.5, 0.35)):
        for depth in (0.25, 0.45, 0.70):
            run(hold(), 20)
            start_err, best, final, ticks, done, point = None, 9.9, 9.9, 0, False, None
            for i in range(MAX_TICKS):
                st = ex.last_state
                if st is None:
                    raise SystemExit("no state yet")
                wp = project_point(st.head, st.right_hand.p, FOV)
                if wp is None:
                    break
                err = pixel_error(target, wp)
                start_err = err if start_err is None else start_err
                best = min(best, err)
                point, _ = servo_step(target, wp, st.head, FOV, depth, deadband=DEADBAND)
                if point is None:
                    done = True
                    break
                st = run(BodyCommand(
                    authority=AUTH_R_HAND,
                    right_hand=Pose(p=point, q_xyzw=clip_r.q_xyzw),
                ), 1)
                ticks = i + 1
                wp = project_point(st.head, st.right_hand.p, FOV)
                final = pixel_error(target, wp) if wp else 9.9
            gap = (float(np.linalg.norm(np.subtract(point, st.right_hand.p)))
           if point is not None else float("nan"))
            print(f"({target[0]:.2f},{target[1]:.2f}) {depth:>6.2f} {start_err:>7.3f} "
                  f"{best:>7.3f} {final:>7.3f} {gap:>10.3f} {ticks:>6} "
                  f"{str(done):>5} {str(ex.fell):>5}")


if __name__ == "__main__":
    main()
