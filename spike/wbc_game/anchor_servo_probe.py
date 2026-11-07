"""Anchor at a hands-in-view frame: does it hold, and does the servo converge?

Frame 0 hangs the wrist 0.64 m below the head and outside the frustum, so no
in-frame pixel was ever near the hand. Frames ~95/~332 put it 0.55 m forward at
(0.70, 0.52). Two questions: is such an anchor stable now that the latch is fixed,
and does the pixel servo converge from it?
"""
import sys, time
sys.path.insert(0, "/home/nate/Projects/Clank")
import numpy as np
from clank.body import AUTH_R_HAND, BodyCommand, Pose, hold, rest
from clank.bus import CommandBus
from clank.projection import Fov, _rotate_inverse, pixel_error, project_point, servo_step
from spike.wbc.sonic_executor import R_WRIST, SonicExecutor, _pose_from

FOV = Fov.from_degrees(45.0, 45.0, 45.0, 45.0)
SETTLE, MAX_TICKS, DEADBAND = 60, 150, 0.005

def make(frame):
    ex = SonicExecutor(verbose=False)
    ex.hold_frame = frame
    ex.runner.reset(ex.motion, start=frame)
    return ex

def speeds(rw):
    rw = np.array(rw)
    d = np.linalg.norm(np.diff(rw, axis=0), axis=1) / 0.02
    tail = d[len(d) // 2:]
    return float(np.linalg.norm(rw[-1] - rw[0])), float(np.sqrt(np.mean(tail ** 2)))

print("== hold stability through the bus (latch fix) ==")
for frame in (332, 95):
    ex = make(frame)
    bus = CommandBus()
    for _ in range(SETTLE):
        bus.publish_state(ex.step(rest(), now_ns=time.monotonic_ns()))
    rw = []
    for _ in range(1000):
        st = ex.step(bus.take(), now_ns=time.monotonic_ns())
        bus.publish_state(st)
        rw.append(np.array(st.right_hand.p))
    net, rms = speeds(rw)
    if ex.last_state is None:
        raise SystemExit("no state yet")
    px = project_point(ex.last_state.head, ex.last_state.right_hand.p, FOV)
    print(f"  frame {frame:3d}  net {net:.4f} m  rms_last10 {rms:.6f} m/s  fell {ex.fell}"
          f"  wrist pixel {None if px is None else (round(px[0],2), round(px[1],2))}")

print("\n== servo convergence from frame 332 ==")
pts, orns = make(332)._clip_targets()
CLIP_R = _pose_from(pts[3*R_WRIST:3*R_WRIST+3], orns[4*R_WRIST:4*R_WRIST+4])
ex = make(332)
for _ in range(SETTLE):
    ex.step(rest(), now_ns=time.monotonic_ns())
st = ex.last_state
assert st is not None
uv0 = project_point(st.head, st.right_hand.p, FOV)
if uv0 is None:
    raise SystemExit("the wrist is not in this view")
u0, v0 = uv0
depth = _rotate_inverse(st.head.q_xyzw, tuple(np.subtract(st.right_hand.p, st.head.p)))[0]
print(f"  wrist pixel ({u0:.3f}, {v0:.3f}) at depth {depth:.3f} m\n")
print(f"{'target':>16} {'start':>7} {'min':>7} {'final':>7} {'|cmd-ach|':>10} {'ticks':>6} {'done':>5}")

for target in ((u0 + 0.10, v0), (u0 - 0.10, v0), (u0, v0 + 0.15), (0.5, 0.5)):
    for _ in range(20):
        ex.step(rest(), now_ns=time.monotonic_ns())
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
        st = ex.step(BodyCommand(authority=AUTH_R_HAND,
                                 right_hand=Pose(p=point, q_xyzw=CLIP_R.q_xyzw)),
                     now_ns=time.monotonic_ns())
        ticks = i + 1
        wp = project_point(st.head, st.right_hand.p, FOV)
        final = pixel_error(target, wp) if wp else 9.9
    gap = (float(np.linalg.norm(np.subtract(point, st.right_hand.p)))
           if point is not None else float("nan"))
    print(f"({target[0]:.2f},{target[1]:.2f}) {start_err:>7.3f} {best:>7.3f} {final:>7.3f} "
          f"{gap:>10.3f} {ticks:>6} {str(done):>5}")
