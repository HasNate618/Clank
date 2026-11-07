"""servo_step commands the point itself: full step, no damping. With ~0.10 m of
tracking lag the arm overshoots, the error grows again, and the loop limit-cycles
(min 0.005 -> final 0.102). Test a blend toward the target instead."""
import sys, time
sys.path.insert(0, "/home/nate/Projects/Clank")
import numpy as np
from clank.body import AUTH_R_HAND, BodyCommand, Pose, rest
from clank.projection import Fov, _rotate_inverse, pixel_error, project_point, servo_step
from spike.wbc.sonic_executor import R_WRIST, SonicExecutor, _pose_from

FOV = Fov.from_degrees(45.0, 45.0, 45.0, 45.0)
SETTLE, MAX_TICKS, DEADBAND = 60, 150, 0.005
FRAME = 332

ex = SonicExecutor(verbose=False)
ex.hold_frame = FRAME
ex.runner.reset(ex.motion, start=FRAME)
pts, orns = ex._clip_targets()
CLIP_R = _pose_from(pts[3*R_WRIST:3*R_WRIST+3], orns[4*R_WRIST:4*R_WRIST+4])
for _ in range(SETTLE):
    ex.step(rest(), now_ns=time.monotonic_ns())
st = ex.last_state
assert st is not None
uv0 = project_point(st.head, st.right_hand.p, FOV)
if uv0 is None:
    raise SystemExit("the wrist is not in this view")
u0, v0 = uv0
depth = _rotate_inverse(st.head.q_xyzw, tuple(np.subtract(st.right_hand.p, st.head.p)))[0]
print(f"anchor {FRAME}: wrist pixel ({u0:.3f}, {v0:.3f}), depth {depth:.3f} m")
print(f"{'target':>14} {'gain':>5} {'start':>7} {'min':>7} {'final':>7} {'done':>5} {'ticks':>6}")

for target in ((0.5, 0.5), (u0 + 0.10, v0)):
    for gain in (1.0, 0.5, 0.3, 0.15):
        for _ in range(25):
            ex.step(rest(), now_ns=time.monotonic_ns())
        start_err, best, final, done, ticks = None, 9.9, 9.9, False, 0
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
            cur = np.array(st.right_hand.p)
            blended = cur + gain * (np.array(point) - cur)
            st = ex.step(BodyCommand(authority=AUTH_R_HAND,
                                     right_hand=Pose(p=tuple(blended), q_xyzw=CLIP_R.q_xyzw)),
                         now_ns=time.monotonic_ns())
            ticks = i + 1
            wp = project_point(st.head, st.right_hand.p, FOV)
            final = pixel_error(target, wp) if wp else 9.9
        print(f"({target[0]:.2f},{target[1]:.2f}) {gain:>5.2f} {start_err:>7.3f} {best:>7.3f} "
              f"{final:>7.3f} {str(done):>5} {ticks:>6}")
