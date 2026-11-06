#!/usr/bin/env python3
"""Solve the arm by IK and hand the policy 17 joints, instead of asking it for a wrist pose.

The decisive comparison for the arm: the pose path takes three commands and lands 65-84 mm out,
because the reported hand is the link plus a lever that swings with wrist rotation. This solves
the joints offline to sub-millimetre and puts them in the reference, where NVIDIA's own deploy
expects them (`upper_body_position[17]`).
"""
import math
import sys
import time

sys.path.insert(0, "/home/nate/Projects/Clank/spike/wbc")

import arm_ik  # noqa: E402  # pyright: ignore[reportMissingImports]
import numpy as np  # noqa: E402  # pyright: ignore[reportMissingImports]
import sonic_runner as sr  # noqa: E402  # pyright: ignore[reportMissingImports]
from sonic_executor import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    HOLD_CLIP,
    HOLD_FRAME,
)

def main() -> int:
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
    # mode 1 (VR) writes only 12 lower-body joints and the 3 VR poses - the 29-joint reference is
    # skipped entirely, so the upper-body override has nothing to read. mode 0 (motion tracking)
    # reads the whole reference, which is where the override lives.
    mode = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    r = sr.SonicRunner(mode=mode)
    print(f"mode {mode}  obs {r.enc.get_inputs()[0].shape}")
    motion = sr.load_motion(HOLD_CLIP)
    r.reset(motion, start=HOLD_FRAME)

    # Let it settle on the clip with no override, so the baseline is a state the body holds.
    r.run(motion, seconds=1.0, start=HOLD_FRAME, ramp=0.0)
    rest = arm_ik.vr_point(r, "right")
    print(f"rest right hand {[round(v, 3) for v in rest]}")

    target = rest + np.array([0.14, -0.10, 0.16])
    print(f"target            {[round(v, 3) for v in target]}  "
          f"({np.linalg.norm(target - rest) * 1000:.0f} mm from rest)")

    t0 = time.monotonic()
    vec, res = arm_ik.solve(r, "right", target)
    print(f"ik: residual {res * 1000:.1f} mm over 7 joints in {time.monotonic() - t0:.1f} ms")
    for side in ("left", "right"):
        print(f"  {side} payload slots {arm_ik.payload_slots(side)}")

    r.upper_body = vec
    r.run(motion, seconds=secs, start=HOLD_FRAME, ramp=0.0)
    got = arm_ik.vr_point(r, "right")
    # Did the policy actually adopt the joints we put in the reference? If it did but the hand is
    # still off, the payload mapping is wrong; if it did not, the reference is being ignored.
    want = np.array([vec[i] for i in arm_ik.payload_slots("right")])
    have = np.array([r.data.qpos[r.qadr[hw]] for hw in arm_ik.ARM_HW["right"]])
    print(f"joint tracking: max |have-want| {np.max(np.abs(have-want)):.3f} rad  "
          f"mean {np.mean(np.abs(have-want)):.3f}")
    err = float(np.linalg.norm(got - target))
    print(f"override on: hand {[round(v, 3) for v in got]}  error {err * 1000:.1f} mm")

    # Now remove the override but keep the body running: if the arm stays put, the reference is
    # what held it there and the clip alone does not.
    r.upper_body = None
    r.run(motion, seconds=0.6, start=HOLD_FRAME, ramp=0.0)
    back = arm_ik.vr_point(r, "right")
    print(f"override off: hand {[round(v, 3) for v in back]}  "
          f"drift {math.dist(back, got) * 1000:.1f} mm in 0.6 s")
    return 0 if err < 0.05 else 1

if __name__ == "__main__":
    raise SystemExit(main())
