# The pixel servo on the live body

`spike/wbc_game/servo_sim_probe.py` closes the loop inside MuJoCo. This closes it
against the running G1 driving Job Simulator, with the corrected rectilinear
projection (`research/camera-equals-hmd-pose.md` section 7). `[MEASURED 2026-11-01]`

The target pixel is a **constant** and the wrist's pixel is **computed** from
`body_state.json`, so no detector is in the loop: this measures the servo arithmetic
and the arm together, nothing else. Field of view 140 x 114.2 deg, depth fixed at
0.45 m.

## 0. Correction: the first run was void, and why

Take one (recorded, then withdrawn) closed the loop and reported that it *diverged*
for a target at pixel (0.35, 0.60). That result was mine, not the arm's. Two errors,
both in the harness:

1. **Frame error.** `unproject` returns a root-relative **point**. The `reach` verb
   takes offsets **from the clip wrist**, whose standing rest is
   `(0.079, -0.308, -0.203)`. Take one passed absolute coordinates into an offset
   parameter, so every command was displaced by the rest position. The probe
   confirms the semantics: rest + `(0.45, 0, 0.30)` = `(0.529, -0.308, 0.097)`
   against an achieved `(0.504, -0.280, 0.165)`, a residual-consistent match.
2. **Stale rest.** The harness measured its rest reference after sending `hold`, and
   `hold` latches the pose the arm is *already in*. After a reach it latched the
   reached pose, not the standing clip. It has to send `rest`.

This is the same lesson as the rotation homography, twice over: a negative result from
an estimator is worth nothing until the estimator is checked against a case whose
answer is known. Here the known case was the rest wrist.

## 1. Corrected frames, no clamp

Target (0.50, 0.55):

| iteration | wrist pixel | error | commanded offset (fwd, lat, up) |
| ---: | --- | ---: | --- |
| 0 | (0.775, 2.153) | 1.626 | (+0.316, +0.244, +0.558) |
| 1 | (0.629, 0.508) | 0.136 | (+0.238, +0.514, +0.493) |
| 2 | (0.571, 0.508) | 0.083 | (+0.042, +0.667, +0.458) |
| 3 | (0.490, 0.447) | 0.104 | (+0.045, +0.666, +0.457) |
| 4 | (0.496, 0.455) | 0.095 | (+0.039, +0.665, +0.465) |

Converges and then oscillates in a 0.08-0.10 band.

Target (0.35, 0.60):

| iteration | wrist pixel | error |
| ---: | --- | ---: |
| 0 | (0.494, 0.439) | 0.216 |
| 1 | (0.525, 0.769) | 0.243 |
| 2 | (0.405, 0.246) | 0.359 |
| 3 | (0.494, 0.538) | 0.157 |
| 4 | (0.402, 0.334) | 0.271 |

Wanders. The commanded offsets say why: `forward -0.50, lateral +0.64` - a point
half a metre *behind* the torso at the far edge of the lateral sweep.

## 2. With the measured clamp in the loop

`clank.workspace.clamp_wrist` against `WRIST_BOX` (provenance in that module;
`research/sonic-reach-envelope.md` and `spike/wbc_game/side_envelope_sonic.py`).

Target (0.50, 0.55) still converges: 1.619 -> 0.190 -> **0.072** -> 0.093 -> 0.097,
with the lateral command pinned on the box face from iteration 2.

Target (0.35, 0.60), which diverged to 0.359 without the clamp:

| iteration | error | clamped? |
| ---: | ---: | :--- |
| 0 | 0.203 | yes |
| 1 | 0.123 | yes |
| 2 | 0.146 | yes |
| 3 | 0.194 | yes |
| 4 | 0.196 | yes |

Bounded oscillation in 0.12-0.20 instead of runaway growth. **Divergence becomes a
stall.**

## 3. What this establishes

The servo arithmetic is sound and closes on the real body, and the clamp is what
keeps it stable. Both halves of that matter:

* Where the target is inside the workspace the loop settles in ~3 iterations to about
  0.07-0.10 of a frame, and the residual is the arm - the same conclusion the sim
  reached.
* Where it is outside, the clamp stops the arm from being walked out of its workspace.

The clamp does **not** manufacture reach. A target pixel the arm cannot cover now
stalls at 0.12-0.20 forever, so `servo_step`'s deadband will never be met for it. The
adapter therefore needs a third outcome besides "converged" and "still trying":
**`unreachable`**, once the clamp is active and the error stops falling. Without it
the brain would keep aiming at a pixel it asymptotically cannot hit.

Residuals inside the box are 0.05-0.19 m, so "in the box" means "the arm goes the
right way", never "the hand arrived" - which keeps the glow cue
(`reach_outcome`) as the arrival test rather than pixel convergence.
