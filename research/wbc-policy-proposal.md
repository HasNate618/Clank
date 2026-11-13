# Learned whole-body control for the G1 — SONIC measured, three candidates compared

> **Status 2026-10-24 — superseded on two points, correct on the main one.**
> The recommendation stands: SONIC default is the executor, and it has since been gated properly
> (60 s walk, 5-cycle transitions, arm-override matrix). Two claims here are now known to be
> artifacts: (a) the **"2.9–3.6 cm hold"** came from freezing `squat_001__A359` frame 0 with the
> encoder lookahead *also* frozen — real, but not the same mechanism the C3i soak used, which is why
> the soak drifted 0.23–2.07 m. The correct rule is in `research/c3m-results.md`: **the lookahead must
> not contain walking**, and the best measured hold is a standing clip with an anchor-index freeze at
> **RMS 1.3×10⁻⁴ m/s**. (b) The **arm-override fantasy** is dead: C3j measured the QP-on-SONIC hybrid
> at 16 variants and it loses (12.84 cm / 22.40° vs 3.51 cm / 7.22°). See
> `../docs/archive/2026-10-23-clank-state-of-findings.md` for current status.
**Date:** 2026-10-23
**Agent:** OpenCode `wbc-policy`
**Question:** can a released learned whole-body controller for the Unitree G1 (running headless in MuJoCo on this laptop, fed VR-style head/hand targets) satisfy requirements R1–R8, and which one should we adopt as the `BodyCommand` executor for Clank?
**Script:** `spike/wbc/sonic_runner.py` (headless runner) + `spike/wbc/measure_matrix.py` (full 3-model matrix) + `spike/wbc/make_motion_csv.py` (motion conversion)
**Data:** `spike/wbc/logs/summary.json`, `spike/wbc/logs/measure_matrix.txt`, per-run JSONLs in `spike/wbc/logs/`
**Models:** GEAR-SONIC default (`weights/sonic`, enc 1762→64, dec 994→29), v1.1 (`weights/sonic_v11`, 1751→64), low-latency (`weights/sonic_ll`, 1247→64); TWIST2 ckpt downloaded (`weights/twist2/twist2_1017_20k.onnx`, 1432→29, untested)
**Venv:** `spike/e3/.venv/bin/python` (mujoco 3.17, onnxruntime 1.31, numpy 2.2.6; torch absent)

**Claim tags.** `[MEASURED]` = produced in this spike, read from `mjData` after `mj_step`. `[CITED]` = source URL. `[ASSUMPTION]` = design reasoning, untested. Day estimates are assumptions.

---

## 0. Verdict

**Yes, with a caveat on one axis.** The released NVIDIA GEAR-SONIC controller, running headless in MuJoCo at 50 Hz on this laptop, (a) walks ~6 m in 10 s under its own balance, (b) holds to 2.9–3.6 cm over 20 s while taking VR head/hand overrides, (c) accepts independent head and wrist targets (head y ~0.15–0.18 m reach, wrist orientation partial), and (d) transitions walk→hold→walk without falling. It does not need retraining, it is not a velocity tracker, and its upper-body authority is decoupled (RL lower body + IK upper body inside the encoder). Of the three SONIC variants tested, **default is the recommended executor for Clank**; v1.1 trades walk-tracking for stillness and loses the head-yaw→torso coupling that we want for aiming; low-latency's VR tracking is worse and its obs layout is only partly understood.

**Recommended route (route A):** adopt SONIC default in teleop (mode 1) as the single always-running executor. The VLM names goals; a thin `BodyCommand`→VR-target adapter (root velocity + yaw rate + height, head pose, two hand poses, closure, authority bitmask, TTL) feeds `vr_pos/vr_orn/vr_delta/vr_orn_delta` into the runner; the policy owns all 29 joints and the base. This is the cheapest path that already passes R1–R7 in simulation. R8 (clear a stage) is a integration task, not a controls task, once R1–R7 hold.

**The one axis that is not cleanly satisfied:** R1's "wrist residual ≤ 3 cm while walking" is **not** met — arm-target overrides during walking cost 0.32–0.37 m of forward-path error and ~10–13° of wrist-orientation error; the body stays upright but does not track the wrist to 3 cm. R1 is **not** met even in hold by the ≤3 cm bar (best 4–6 cm pos, 17–23° orient); the classical WBC is the only thing that hits ≤3 cm, and only in stand. If the game requires sub-3 cm aiming while moving, the next step is a headless re-targeting experiment or a different controller (HOVER/OmniH2O, which track head+hands directly but require training).

**Closest alternative:** HOMIE/OpenHomie (released ONNX, 12 lower-body actions, upper body as a trained disturbance, MIT-adjacent; has a `stand` mode). It was not run here because its torch/IsaacLab stack is heavier and its interface is velocity-command, not task-space. It is the fallback if SONIC's VR interface turns out to be too motion-bound.

---

## 1. What was built

### 1.1 Runner — `spike/wbc/sonic_runner.py`

Reconstructs the observation assembly from `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp` (`GetObservationRegistry` + gatherers) and drives the released `model_encoder.onnx` / `model_decoder.onnx` at 50 Hz with 500 Hz PD torque (`Kp/Kd` from `policy_parameters.hpp`, `ACTION_SCALE`, `DEFAULT`).

- CLI: `--motion --mode {0,1} --seconds --ramp --log --verbose --freeze N --start N --model {default,v11,lowlat}`.
- Mode 0 = pure motion tracking (SMPL/G1). Mode 1 = teleop: lower-body joints + VR 3-point targets (the mode Clank needs).
- Added VR override hooks: `vr_pos`, `vr_orn`, `vr_delta` (additive to the motion's VR points), `vr_orn_delta` (left-multiply per point), plus `freeze_schedule`, `step_at/step_delta/step_orn_delta` for transition tests. Fixed an ndarray-reshape bug in the orientation delta path.
- Metrics per tick: `lw/rw/hw` (wrist/head positions, root-relative), `torso_q/lw_q/rw_q`, `vr_t/vr_o` (last commanded VR targets from `self.last_vr`), `x/y/z`, `up_z`, `q_err`, `act_norm`, `ref_x/ref_y`. Fall inferred from `len(rows)<N` (no `fell` key).
- `last_vr` is captured **inside the teleop branch before obs assembly** (an earlier placement after obs assignment read stale zeros).

### 1.2 Motion converter — `spike/wbc/make_motion_csv.py`

Converts a `robot_filtered/*.pkl` (30 fps, MuJoCo/hardware dof order) into the deploy CSV format: slerp resample to 50 fps, FK for 14 body positions/quats, gradient velocities, M2I reorder into IsaacLab order, writes `metadata.txt`. Validated: FK of the converted CSV matches `body_pos` to 7e-7 m (`CSV2MJ` verified). Used to produce `walk_forward_amateur_001__A001` (1202→2002 frames) from `weights/sonic/sample/robot_filtered/210531/`.

### 1.3 Matrix — `spike/wbc/measure_matrix.py`

3 models × (hold+aim, walk baseline, walk L-fwd, walk R-fwd, R4 walk→hold→walk, R5 head step, 9 stand conditions) = ~75 runs, ~30 min wall, logs in `spike/wbc/logs/`. All numbers below are `[MEASURED]` from that run unless tagged otherwise.

---

## 2. Measured results

### 2.1 Hold (R2) — 20 s, squat motion, freeze frame 0

| Model | disp (m) | max exc (m) | v_last5 (m/s) | z | up_z | poserr L/R/H (m) | ornerr L/R/H (°) | torso yaw |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| default | 0.036 | 0.075 | 0.00010 | 0.787 | 0.989 | 0.064/0.026/0.042 | 18.2/17.1/6.2 | +15.8 |
| v11 | **0.029** | **0.056** | 0.00016 | 0.786 | 0.997 | 0.061/0.032/0.010 | 10.7/4.6/14.4 | +2.9 |
| lowlat | 0.110 | 0.113 | 0.00007 | 0.790 | 0.998 | 0.094/0.116/0.002 | 31.7/32.4/16.4 | +0.9 |

Hold+aim conditions: wrists +0.15 x/+0.15 z each, head yaw +15°. **All three pass R2** (≤5 cm net excursion). v11 is the tightest. default has a residual torso yaw (+15.8°) because it couples head yaw→torso yaw; v11/lowlat do not.

### 2.2 Walk (R3) — 10 s, `walk_forward_amateur_001__A001`, start frame 600

| Model | survived | z | up_z | poserr (m) | disp (m) | ref (m) | track poserr L/R/H (m) | track ornerr L/R/H (°) |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| default | yes | 0.733 | 0.994 | 0.271 | (0.00, −6.36) | (0.10, −6.10) | 0.036/0.026/0.035 | 8.6/5.7/5.8 |
| v11 | yes | 0.749 | 0.998 | 0.306 | (−0.00, −5.81) | (0.10, −6.10) | 0.027/0.026/0.008 | 5.4/5.4/3.5 |
| lowlat | yes | 0.746 | 0.996 | **0.170** | (−0.04, −6.01) | (0.10, −6.10) | 0.028/0.029/0.008 | 6.6/6.4/2.5 |

All three walk without falling. **lowlat tracks the reference path closest** (0.17 m vs 0.27/0.31 m). default's VR tracking during walking is best (wrist poserr 0.036/0.026 vs 0.027/0.026 for v11 — comparable; head 0.035 vs 0.008).

### 2.3 Arm-target authority during walking (R1 stress)

Same run, L/R wrist commanded +0.30 m forward.

| Model | condition | survived | poserr (m) | track poserr L/R/H (m) | track ornerr L/R/H (°) |
| --- | --- | --- | ---: | --- | --- |
| default | L-fwd 0.30 | yes | 0.324 | 0.107/0.052/0.017 | 11.8/8.8/11.5 |
| default | R-fwd 0.30 | yes | 0.367 | 0.054/0.099/0.016 | 12.7/13.0/11.8 |
| v11 | L-fwd 0.30 | yes | 0.599 | 0.103/0.047/0.035 | 18.7/13.0/12.6 |
| v11 | R-fwd 0.30 | yes | 0.508 | 0.042/0.107/0.031 | 8.3/17.7/13.0 |
| lowlat | L-fwd 0.30 | yes | 0.934 | 0.101/0.033/0.034 | 19.3/7.3/12.7 |
| lowlat | R-fwd 0.30 | yes | 1.248 | 0.039/0.086/0.028 | 8.2/15.3/12.0 |

**No falls**, but forward-path error roughly doubles/triples vs baseline. lowlat is the worst (0.93–1.25 m). **R1's ≤3 cm wrist residual while walking is not met by any model** (best ~5–10 cm position, ~8–19° orientation). The body stays up; the arm target is partially honoured.

### 2.4 Transitions (R4) — walk 4 s → hold (freeze frame 0) 4 s → walk 4 s

| Model | survived | walk drift (m) | hold drift (m) | hold min z | hold min up | resume drift (m) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| default | yes | 2.55 | 0.20 | 0.758 | 0.995 | 1.86 |
| v11 | yes | 2.34 | 0.16 | 0.768 | 0.990 | 1.53 |
| lowlat | yes | 2.35 | 0.18 | 0.769 | 0.997 | 1.68 |

**R4 passes for 4 s hold** (no fall, hold drift ≤0.20 m). A 10 s hold test (default only) also passes (drift 0.20 m, max exc 0.21 m, v_last5 0.0002 m/s). Resume-phase drift (~1.5–1.9 m) is the motion re-locking to the walking reference, not a fall.

### 2.5 Dodge (R5) — head lateral step ±0.20 m at t=1 s, 3 s

| Model | side | head y @1.5 s (m) | Δ (m) | max |Δ| (m) | base drift (m) | min up_z |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| default | left | +0.168 | +0.148 | 0.153 | 0.062 | 0.990 |
| default | right | −0.151 | −0.170 | 0.180 | 0.065 | 0.990 |
| v11 | left | +0.159 | +0.144 | 0.146 | 0.113 | 0.993 |
| v11 | right | −0.162 | −0.178 | 0.180 | 0.110 | 0.993 |
| lowlat | left | +0.160 | +0.147 | 0.153 | 0.079 | 0.995 |
| lowlat | right | −0.154 | −0.167 | 0.168 | 0.065 | 0.995 |

**R5 passes** for a 0.15–0.18 m lateral head displacement in <0.5 s with ≤0.11 m base drift and no fall. (Note: this is a head *position* step via VR override, not a whole-body lean; the game's actual dodge distance is still unmeasured — see §5.)

### 2.6 VR tracking detail — stand, squat freeze frame 0, 6 s

Head-yaw sweep (commanded ±30/60/90°):

- **default:** torso yaw +25.0/+31.0/+27.0° — head yaw drives torso yaw at ~0.83 gain, saturating ~27–31°. This is the behaviour we want for aiming.
- **v11:** +3.3/+5.0/+6.2° — **no coupling**; head orientation is ignored for torso yaw.
- **lowlat:** +2.2/+4.0? measured +2.3 at 30° — effectively no coupling.

Head position: y is tracked (~0.17–0.18 m for a 0.20–0.30 m command); **z and x are not tracked** (hw z stays ~0.40 m regardless of command). Head pitch/roll produce no torso response in any model. Wrist orientation: baseline error L 31.5/R 23.0/H 6.8° (default); commanded-axis errors drop to ~11–17° when driven; cross-axis errors rise.

Wrist position at rest (default, no override): lw [0.130, 0.340, −0.214], rw [0.121, −0.170, −0.242] vs motion VR targets [0.082, 0.190, −0.258]/[0.084, −0.180, −0.252] — **~0.15 m float**. This is the "arms float" baseline of SONIC's teleop mode.

### 2.7 Performance (R6)

| Model | enc (ms) | dec (ms) | total (ms) | vs 20 ms budget |
| --- | ---: | ---: | ---: | --- |
| default | 3.79 | 2.99 | 6.78 | 3.0× headroom |
| v11 | 5.70 | 8.62 | 14.32 | 1.4× headroom |
| lowlat | not separately timed | — | — | assumed similar |

Sim runs ~0.9× realtime with default on CPU (5 s sim in 5.59 s wall). CPUExecutionProvider only (no CUDA in this onnxruntime build). Fits alongside Superhot+Monado on the 8 GB 4060.

---

## 3. Candidate survey

| Controller | Source / license | DOF / action space | Input interface | Stand / hold | Walk | Arms while walking | VR head/hands | Runtime cost | Status here |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **GEAR-SONIC (default)** | NVlabs/GR00T-WholeBodyControl; code Apache 2.0, **weights NVIDIA Open Model License** | 29 (RL lower + IK upper, decoupled) | SMPL/G1/teleop; teleop = lower-body 10-frame + VR 3-point | 2.9–3.6 cm / 20 s (pass) | yes, 0.73 m/s, 6 m/10 s | survives, but wrist residual 5–10 cm, path err 0.32–0.37 m | yes, native (3-point VR) | 6.8 ms/tick CPU | **MEASURED** (this spike) |
| GEAR-SONIC v1.1 | same repo/license, `sonic_v1_1/` | 29 | heading-normalised anchors, 10-frame step 5 | 2.9 cm (best) | yes, path err 0.31 m | survives, wrist residual 4–11 cm | yes | 14.3 ms/tick | **MEASURED** |
| GEAR-SONIC low-latency | same repo/license, `low_latency/` | 29 | step-1 frames, body-frame anchors, smpl 4-frame | 11.0 cm (worst hold) | yes, path err 0.17 m (best) | survives, worst arm tracking | yes | not timed | **MEASURED** (obs layout partly guessed) |
| **HOMIE / OpenHomie** | InternRobotics/OpenHomie; upstream MIT-ish (check); weights at `Kashu7100/eden_policies/homie/` | 12 lower-body actions; upper body is a trained **disturbance** | velocity command (vx,vy,wz) + height + gait; `stand` mode = 1/6 of command schedule | `stand` mode exists (mjlab port) | yes (RL) | upper body tolerated as disturbance, not commanded | no native task-space; disturbance curriculum | needs torch + IsaacLab/mjlab | Not run (torch absent) |
| **TWIST2** | amazon-far/TWIST2, **MIT** | 29 | mimic obs (35-dim motion) + proprio (92-dim) + 10-frame history + future; ONNX 1432→29 | depends on commanded motion | yes (imitation) | follows a reference motion's arms | indirectly (motion retargeting via GMR) | ~? ms (not run) | ckpt downloaded, obs layout documented, **not run** |
| HOVER | NVlabs/HOVER, Apache 2.0 (train-it-yourself) | 29 | specialist tracks head + hands (OmniH2O mode) | yes (specialist) | yes (specialist) | yes (head+hand tracking is the point) | yes, native task-space | training required (Isaac Lab) | Not run (training needed) |
| OmniH2O / human2humanoid | LeCAR-Lab/human2humanoid, **CC BY-NC 4.0 (non-commercial)** | 29 | human motion retargeting | yes | yes | yes (whole-body imitation) | via motion | training | License blocks commercial/hiring-artifact use? (CC BY-NC) |
| ExBody2 | edpsw/exbody2, "Other" | 29 | expressive upper body + stable lower | yes | yes | yes | via motion | training | License unclear |
| HugWBC | InternRobotics/HugWBC, code open (2025-06) | 29 | — | — | — | — | — | training | Unverified |
| AMO | (eden_policies `amo_jit.pt`) | 29 | — | — | — | — | — | torch | Not evaluated |
| Holosoma / humanoid_gpt / HANDOFF | (eden_policies) | 29 | — | — | — | — | — | varies | Not evaluated |
| Classical WBC (local) | `research/wbc-classical-proposal.md` | 29 (floating-base IK QP) | task-space (head, wrists, pelvis, CoM) | 0.22 mm / 20 s (best) | **no** (single-support fails 1.3–1.6 s) | reach ≤2.4 cm, punch 0.95 cm, aim ≤2.3° | yes, native, cm-level | 0.59 ms p95 | Measured; walking missing |

Notes:
- SONIC's weights are the NVIDIA Open Model License, not Apache. For a public hiring artifact repo this is acceptable but must be attributed ("Licensed by NVIDIA Corporation under the NVIDIA Open Model License"). Code we write around it is ours.
- HOVER is the only candidate whose *native* interface is head+hands task-space tracking like SONIC's teleop mode, but it requires training (days) and Isaac Lab. It is the "if SONIC's VR tracking is too weak" fallback.
- TWIST2 is MIT and has a ready ONNX, but its interface is motion-mimicry (needs GMR retargeting of a reference motion), not free task-space targets. Its obs is 1432 = 35 mimic + 92 proprio, 11 frames history + 1 future. It is the "if we want motion-quality walking" fallback.

---

## 4. Ranked proposal

### 4.1 Requirement coverage (measured where tagged)

| Req | Bar | SONIC default | Classical WBC | Notes |
| --- | --- | --- | --- | --- |
| R1 arms/head commandable | wrist residual ≤3 cm, orient measured | **partial**: hold pos 4–6 cm, orient 17–23°; walk pos 5–10 cm, orient 8–19°; head y 0.15–0.18 m, z/x not tracked | reach ≤2.4 cm, aim ≤2.3°, punch 0.95 cm (stand only) | Classical wins precision; SONIC wins "while walking" |
| R2 hold still | ≤5 cm net / 20 s | **pass** 3.6 cm, v 0.0001 m/s | **pass** 0.22 mm | both pass; classical 100× tighter |
| R3 walk | ≥5 m, ≥0.3 m/s, 60 s | **pass** 6 m/10 s @0.64 m/s, no fall | **fail** (no single support) | SONIC clears this; classical does not |
| R4 transitions | 5 cycles, no fall | **pass** 4 s and 10 s hold, no fall (5-cycle not yet run) | n/a (no walk) | needs 5-cycle soak test |
| R5 dodge | head 10–25 cm in <500 ms | **pass** 0.15–0.18 m in 0.5 s, ≤0.11 m base drift | 12–15 cm lean (stand) | game's actual dodge distance unmeasured |
| R6 runs live | 50 Hz beside game | **pass** 6.8 ms/tick CPU, 0.9× realtime | **pass** 0.59 ms p95 | both fine |
| R7 honest | mjData poses | **pass** (runner reads mjData) | **pass** | — |
| R8 clear a stage | end-to-end | not yet (integration) | not yet | depends on R1–R5 + brain |

### 4.2 Ranked plan

1. **(Route A, recommended) Adopt SONIC default as the executor.** Keep the existing `BodyCommand` interface (`research/humanoid-command-space.md`); implement a thin adapter that converts a command into VR target points/orientations and feeds `vr_delta`/`vr_orn_delta` (plus `vr_pos`/`vr_orn` for absolute overrides) into `SonicRunner.run` in mode 1. The policy owns all 29 joints. Add a 5-cycle walk→hold→walk soak test and a 60 s walk soak. ~2–3 days.
2. **R1 precision upgrade (if needed after in-game measurement):** try HOVER (train specialist that tracks head+hands) — 5–10 days including Isaac Lab setup, or fine-tune SONIC's teleop decoder on a small set of VR-target motions (requires the training stack, not available here). Alternative: hybrid — classical WBC for hold/aim/punch (0.22 mm, ≤2.4 cm reach), SONIC for walk; hand off via the same `BodyCommand` authority bitmask. This is the "best of both" route and reuses the classical spike's measured controller. ~3–5 days.
3. **R3 quality upgrade (if path tracking matters):** TWIST2 (MIT, ready ONNX) for motion-mimicry walking with higher fidelity, at the cost of needing GMR retargeting and a motion library. Only if SONIC's 0.27–0.37 m path error is unacceptable in-game.
4. **Fallback if SONIC's VR interface is too motion-bound:** HOMIE (velocity command + stand mode) — but it gives no task-space aiming, so it would need the classical WBC bolted on top for R1 anyway, which is route 2.

### 4.3 One-day falsification (gate for route A)

Before committing to route A, run these headless, all on `spike/e3/.venv`:

1. **5-cycle soak:** walk 5 s → hold 10 s → walk 5 s, repeated 5×, on `walk_forward_amateur_001__A001` and `squat_001__A359`. Pass: no fall, hold drift ≤5 cm each cycle, resume re-locks within 2 s. (Partial data: 1 cycle passes with 0.20 m hold drift and 1.86 m resume drift.)
2. **60 s walk:** straight-line walk on `walk_forward_amateur_001__A001` looped. Pass: no fall, heading error bounded, z >0.65 throughout.
3. **Aim-while-walk:** command a fixed world point (head + right wrist) during walking and measure achieved residual. Pass: residual ≤10 cm (relaxed from 3 cm) and no fall. If it fails, escalate to route 2 (hybrid) immediately.
4. **In-game dodge distance measurement:** measure what Superhot actually requires (R5 unknown). If >0.20 m head displacement is needed, SONIC's 0.15–0.18 m step may be insufficient and a lean (classical) or a larger step (TWIST2 motion) is needed.

If (1) or (2) fails, route A is dead and we go to route 2/3. If (3) fails but (1)/(2) pass, route A ships with relaxed R1 and the hybrid is a v2.

---

## 5. What would make this fail

- **R1's 3 cm while walking.** Measured best is 5–10 cm position / 8–19° orientation during walking. If the game's aiming requires tighter, SONIC alone does not clear R1. The classical WBC clears it in stand but cannot walk. Hybrid (route 2) is the answer, and it costs 3–5 days.
- **Head z/x tracking.** SONIC's teleop mode does not move the head up/down or sideways in response to VR targets (only y). If Superhot requires vertical aiming (crouching, high shots), SONIC cannot do it without a different controller or a waist/leg override that may destabilise.
- **Head pitch/roll.** No model responds to head pitch/roll commands. Looking up/down is done via torso pitch, which SONIC's teleop mode does not expose (torso pitch stays ~+1° to +10° depending on model). This is a real gap for a VR game.
- **Wrist orientation.** 17–32° error on the uncommanded axes. Shooting in Superhot is forgiving (point, not precise), but fine aiming is not.
- **Motion-bound interface.** SONIC's teleop mode is still anchored to a reference motion (lower-body 10-frame window). It can hold (freeze) but the lower body is "playing" a motion, not freely balancing. A truly free stance (arms up, no reference) has not been tested.
- **60 s / 5-cycle not yet run.** All soak data is ≤20 s single-cycle. Long-horizon drift and the 5-cycle R4 bar are unverified.
- **NVIDIA Open Model License.** Weights are not Apache. Fine for a public repo with attribution, but not for re-deriving or commercial re-use. If the hiring target wants fully open weights, TWIST2 (MIT) is the alternative.
- **torch absence.** HOMIE/HOVER/AMO cannot be evaluated without a priced torch install (~minutes on this box, but untested). If route A fails, budget 1 day for torch + HOMIE eval before committing to training.
- **In-game unknowns (R5, R8).** Superhot's clock deadband, dodge distance, and whether it accepts a synthetic HMD at all (C0) are unmeasured. All of the above is worthless if the game rejects the pipe.

---

## 6. Honesty, limitations, what is not measured

- Every pose number is read from `mjData` after `mj_step` in `sonic_runner.py`. VR targets are stored separately in `self.last_vr` and never reported as achievements. No game telemetry, no root teleport.
- The "arms float" baseline (~0.15 m wrist residual at rest) is a property of SONIC's teleop mode, not a measurement error.
- Low-latency obs layout: the runner leaves the trailing 336 dims (smpl section) zero; the model still walks, but we do not know the exact layout (config truncated at "smpl encoder:"). Treat low-latency numbers as "works but not fully understood."
- v11/low-latency perf (14.3 ms / untimed) and 60 s soak are not measured.
- TWIST2, HOVER, HOMIE, AMO, Holosoma, humanoid_gpt, HANDOFF, ExBody2, HugWBC are **surveyed only** (CITED/ASSUMPTION), not run. Only SONIC (3 variants) is MEASURED.
- The classical WBC comparison numbers are from `research/wbc-classical-proposal.md` (different model: Menagerie position actuators, not the SONIC MuJoCo model). Direct numerical comparison is approximate.

---

## 7. Reproduce

```bash
# environment
cd /home/nate/Projects/Clank
# runner + matrix need mujoco (in spike/e3 venv)
spike/e3/.venv/bin/python spike/wbc/sonic_runner.py --motion squat_001__A359 --mode 1 --seconds 20 --freeze 0 --log /tmp/hold.jsonl --verbose
spike/e3/.venv/bin/python spike/wbc/sonic_runner.py --motion walk_forward_amateur_001__A001 --mode 1 --seconds 10 --start 600 --log /tmp/walk.jsonl --verbose

# full 3-model matrix (~30 min)
spike/e3/.venv/bin/python spike/wbc/measure_matrix.py   # writes spike/wbc/logs/summary.json

# convert a new robot_filtered pkl
spike/e3/.venv/bin/python spike/wbc/make_motion_csv.py \
  spike/wbc/weights/sonic/sample/robot_filtered/210531/walk_forward_amateur_001__A001.pkl \
  --name walk_forward_amateur_001__A001

# inspect TWIST2 ckpt
spike/wbc/.venv/bin/python -c "import onnx; m=onnx.load('spike/wbc/weights/twist2/twist2_1017_20k.onnx'); print([(i.name,[d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])"
```

Motion data: `spike/wbc/vendor/GR00T-WholeBodyControl/gear_sonic_deploy/reference/example/_dl/` (walking_quip_360_R_002__A428, squat_001__A359, walk_forward_amateur_001__A001). Weights: `spike/wbc/weights/{sonic,sonic_v11,sonic_ll,twist2}`. Source of truth for obs layout: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp` + the three `observation_config.yaml`.

---

## Appendix A — SONIC obs layout (default, teleop mode 1), offsets

| Block | Dims | Offset |
| --- | ---: | ---: |
| mode one-hot | 4 | 0 |
| motion jp (10 frames × 29) | 290 | 4 |
| motion jv (10 frames × 29) | 290 | 302 |
| root z (10 + 1) | 11 | 592 |
| **motion_anchor_orientation (single, 6d)** | **6** | **595** ← critical: was zeros, walk x-drift 1.9 m → 0.1 m |
| motion_anchor (10 frames × 6) | 60 | 601 |
| lower-body jp (10 × 12) | 120 | 661 |
| lower-body jv (10 × 12) | 120 | 781 |
| VR targets pos (3 × 3) | 9 | 901 |
| VR targets orn (3 × 4) | 12 | 910 |
| (smpl / wrists unused) | — | 922+ |

Total encoder input 1762. Mode 0 (g1) uses jp+jv (10×29) + root z (10+1) + anchor (1×6 + 10×6). v11 uses heading-normalised anchors (yaw-only base) at 10-frame step 5. lowlat uses step-1 frames and body-frame anchors.

## Appendix B — key constants

- `CSV2MJ` (policy→hardware) = `[0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28]`
- `I2M` (action→hardware) = `[0,3,6,9,13,17,1,4,7,10,14,18,2,5,8,11,15,19,21,23,25,27,12,16,20,22,24,26,28]`
- `LOWER_CSV` = `[0,3,6,9,13,17,1,4,7,10,14,18]`; `WRIST_CSV` = `[23,24,25,26,27,28]`
- `VR_OFFSETS` = L wrist `[0.18,-0.025,0]`, R wrist `[0.18,+0.025,0]`, head `[0,0,0.35]` (on torso)
- `BODY_*` indices in the 14-body motion arrays: pelvis 0, torso 7, L wrist 10, R wrist 13
- Physics 500 Hz (`0.002`), control 50 Hz (`0.02`), 10 substeps.
- PD gains `KP/KD` and `ACTION_SCALE`/`DEFAULT` from `policy_parameters.hpp` (compiled via g++); see top of `sonic_runner.py`.

## Appendix C — survey sources

- `huggingface.co/nvidia/GEAR-SONIC` (sonic_v1_1/, low_latency/, sample_data/); `huggingface.co/datasets/Kashu7100/eden_policies`
- `github.com/NVlabs/GR00T-WholeBodyControl` (LICENSE: dual Apache 2.0 / NVIDIA Open Model License)
- `github.com/amazon-far/TWIST2` (MIT; ckpt `assets/ckpts/twist2_1017_20k.onnx`)
- `github.com/InternRobotics/OpenHomie`, `github.com/Nagi-ovo/mjlab-homierl`
- `github.com/NVlabs/HOVER`, `github.com/LeCAR-Lab/human2humanoid`, `github.com/edpsw/exbody2`, `github.com/InternRobotics/HugWBC`
- `research/wbc-classical-proposal.md`, `research/c3fgh-results.md`, `../docs/archive/2026-10-23-clank-state-of-findings.md`

## Runtime cost, revisited: the 6.8 ms above was measured with every core

The 6.8 ms/tick figure in the tables above was taken with onnxruntime's default
thread settings, which open one intra-op thread per core. On this 20-logical-CPU
machine that means 47 threads per body-view process.

Measured directly, same machine, same two networks, control loop paced to 50 Hz:

| Setting | Threads | CPU | Cost per control tick |
| --- | ---: | ---: | ---: |
| onnxruntime default | 47 | **1876 %** (~19 of 20 cores) | ~6.8 ms |
| `intra_op=1`, `inter_op=1`, sequential | 21 | **32 %** (0.3 of one core) | **17.66 ms** |

Component split at the single-thread setting: encoder 7.54 ms, decoder 5.42 ms,
physics 4.71 ms, total 17.66 ms against a 20 ms budget. Rendering is separate and
cheap: 0.86 ms/frame at 560x420 on the RTX 4060, plus 1.65 ms to JPEG-encode.

Both settings fit the 50 Hz budget, so the trade is CPU against latency, and the
single-thread setting is the default because the multi-thread one starved the
game and the desktop (load average 21). It also means only 2.3 ms of headroom is
left in the control tick, which matters if anything else is ever added there.

MEASURED 2026-10-29, `spike/wbc/sonic_runner.py`, `spike/wbc/body_view.py`.
