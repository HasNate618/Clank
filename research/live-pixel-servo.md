# The pixel servo on the live body

`spike/wbc_game/servo_sim_probe.py` closes the loop inside MuJoCo. This closes it
against the running G1 driving Job Simulator, with the corrected rectilinear
projection (`research/camera-equals-hmd-pose.md` section 7). `[MEASURED 2026-11-01]`

The target pixel is a **constant** and the wrist's pixel is **computed** from
`body_state.json`, so no detector is in the loop: this measures the servo arithmetic
and the arm together, nothing else. Field of view 140 x 114.2 deg, depth fixed at
0.45 m, no workspace clamp (the function is stateless by design).

## Result: it converges when the target is reachable, and diverges when it is not

Target (0.50, 0.55):

| iteration | wrist pixel | error | commanded point |
| ---: | --- | ---: | --- |
| 0 | (0.457, 0.470) | 0.091 | (+0.301, +0.383, +0.257) |
| 1 | (0.401, 0.565) | 0.100 | (+0.470, +0.168, +0.226) |
| 2 | (0.425, 0.575) | 0.079 | (+0.503, +0.027, +0.212) |
| 3 | (0.486, 0.624) | 0.075 | (+0.504, +0.052, +0.199) |
| 4 | (0.478, 0.620) | **0.074** | (+0.503, +0.056, +0.203) |

It settles at ~0.074 after three iterations and the forward command saturates at
0.503 m, which is the arm's own forward limit. So the loop closes; the residual is
the arm, exactly as the sim reported.

Target (0.35, 0.60) - the same code, a different pixel:

| iteration | wrist pixel | error |
| ---: | --- | ---: |
| 0 | (0.488, 0.626) | 0.140 |
| 1 | (0.472, 0.701) | 0.158 |
| 2 | (0.482, 0.767) | 0.213 |
| 3 | (0.556, 0.943) | **0.400** |
| 4 | (0.531, 0.727) | 0.221 |

It **diverges**. The commands ask for `forward 0.023, lateral 0.593` and then
`forward -0.092, lateral 0.532` - points behind and to the side of the body that the
arm cannot occupy, so the achieved wrist goes wherever it likes and the error grows.

## What this establishes

The servo arithmetic is sound and closes on the real body: `project_point`,
`unproject` and `servo_step` agree with the live arm to within its own reach limit.

The thing that is missing is not arithmetic, it is the **workspace clamp**, and this
demonstrates on the live body why the adapter spec requires one. A servo that is
"deliberately stateless and knows nothing about joint limits" (by design, so it does
not become a second copy of those rules) will happily walk the arm out of its
workspace when the unprojected point lands there, and the error then grows instead of
shrinking.

Consequence for the adapter: clamping is not a safety nicety to add later, it is what
keeps the loop stable. `clank/validate.HAND_BOX` is +-0.80 m laterally, which does
**not** exclude `lateral 0.59`, so the existing box is not the clamp this needs - the
limit has to come from the arm's measured envelope (`research/sonic-reach-envelope.md`:
wrist translates to about 0.70 m forward and stops; 0.45-0.59 m tracks with 0.10-0.18 m
residual), not from the generic body box.

Probe kept privately; it needs the live session and the body's control socket.
