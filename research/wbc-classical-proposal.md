# Classical whole-body control for the G1 — easy half measured, hard half scoped

> **Status 2026-10-24 — the verdict stands, and its open dependency is now resolved.**
> The QP is precise and cannot walk: unchanged. Its item *"C0 is still unrun"* is **answered** —
> C0 passed (`research/c0-results.md`): Superhot VR holds a focused session on `Monado: Remote HMD`
> and its view tracks our poses. Its second dependency also resolved against it: C3j showed the QP
> **cannot be composed onto SONIC** as a joint override (12.84 cm / 22.40° vs 3.51 cm / 7.22°, and
> integral correction tips the robot). Only a **wrist-orientation-only** override survives, improving
> pointing 7.22° → 5.08°.
**Date:** 2026-10-23
**Agent:** OpenCode `wbc-classical`
**Question:** can a classical whole-body controller (QP/HQP over the floating-base G1, plus a gait generator) satisfy the Clank requirements R1–R8?

**Script:** `spike/wbc_classical/wbc_spike.py`
**Data:** `spike/wbc_classical/logs/summary.json`, `logs/run_all.txt`, per-scenario CSVs
**Model:** Menagerie `spike/e3/vendor/mujoco_menagerie/unitree_g1/scene.xml` — 29 actuated joints, position actuators (`kp=500`, `dampratio=1`), `stand` keyframe, no neck. Same model and actuator interface as E3.

**Claim tags.** `[MEASURED]` = produced in this spike by `wbc_spike.py`, read from `mjData` after `mj_step`. `[CITED]` = prior local result or external source, file named. `[ASSUMPTION]` = design reasoning, untested. Day estimates are assumptions.

---

## Verdict

**Yes for the standing half; not yet for walking.** A 50 Hz floating-base task-space WBC — mink weighted QP with foot, pelvis, CoM, head, wrist, and posture tasks, joint/velocity limits, and rate-limited position targets — holds the G1 to **0.22 mm** of base excursion over 20 s, tracks a **±35°** waist look with **0.63°** yaw error and **3.3 mm** base excursion, reaches single-arm targets with **≤2.4 cm** residual and **≤2.3°** pointing error, punches with **0.95 cm** residual and **3.6°** pointing error (E3's DLS punch residual was **13.5 cm**), and produces a **12–15 cm** head dodge with weight shift without falling. It does **not** walk: a sustained **0.10 m** lateral weight shift is the measured limit (0.12 m tips), and every single-support foot-lift attempt tipped within **1.3–1.6 s** even after tuning the base/CoM costs. The recommendation is therefore to adopt this WBC as the standing/hold/look/reach/aim/punch/dodge executor now, and to treat walking as a separate **5–10 day** contact-QP + gait spike, gated by a one-day single-support falsification. Do not claim R3/R4/R8 from the results below.

---

## 1. What was built and measured

### 1.1 Controller

The measured controller is **weighted task-space IK over the full floating-base model**, not a contact-force QP. `mink.Configuration` contains all 36 `qpos` (7 floating base + 29 joints); `mink.solve_ik` chooses a `dq` over all 35 DoF. Only the 29 joint targets are sent to the position actuators; the physical base is never actuated or teleported. It follows contact physics after `mj_step`.

Task stack and costs `[MEASURED]`:

| Task | Cost | Notes |
| --- | ---: | --- |
| Left / right ankle-roll frame | 5000 pos / 5000 ori | planted feet |
| Pelvis frame | 5000 pos / 5000 ori | soft base task inside the IK |
| CoM (`subtree_com[1]`) | 100 | support/lean regulation |
| Head mesh (`head_link` geom) | 10 pos / 100 ori | 0 / 500 for look; 500 / 500 for dodge |
| Left / right wrist-yaw frame | 500 pos / 500 ori | controller pose |
| Posture | 0.1 | toward the `stand` keyframe |

Limits: `ConfigurationLimit(min_distance_from_limits=0.01)` and `VelocityLimit` at **2.0 rad/s**. The joint command is rate-limited to `q_cur + clip(q_des - q_cur, ±2.0*dt)`. Loop: **50 Hz** (`dt=0.02`), 10 MuJoCo substeps of 2 ms, solver `daqp`. Target changes are exponentially interpolated (`alpha = 1 - exp(-dt/tau)`, `tau=0.3 s`) in the continuous mixed scenario; the single-action scenarios interpolate position over 0.5–1.5 s and recompute the aim orientation per tick.

The distinction matters for the walking discussion: **this QP has no contact-force decision variables.** It is an IK QP whose floating base is a decision variable; contact forces are whatever MuJoCo produces after the joint targets are applied. The measured single-support failure is evidence that this formulation is insufficient for walking, **not** evidence that a contact-force QP will fail.

### 1.2 Honesty

Every reported pose is read from `mjData` after `mj_step`: base `qpos`, head `geom_xpos`/`geom_xmat`, wrist body `xpos`/`xmat`, foot body `xpos`/`xmat`, `subtree_com`, and the convex hull of the eight foot-sphere geoms. Targets are stored separately and are never reported as achievements. No game telemetry, no scripted camera, no root teleport. This satisfies R7 for the standing scenarios `[MEASURED]`.

### 1.3 Scenarios

`hold` (20 s neutral), `look` (20 s, ±35° waist yaw), `reach` (3 targets × left/right/both, 2 s each), `aim` (same 3×3 with the wrist z-axis pointed at a fixed world point), `punch` (left/right, 2 s), `dodge` (4 amplitudes × planted/weight-shift, 1.5 s), `shift` (sustained double-support lateral CoM shift), `lift` (single-support foot lift), `mixed` (a continuous stand → look → reach → punch → dodge sequence, both paused and unpaused).

---

## 2. The easy half: measured

### 2.1 Hold and look (R2)

| Metric | Hold, 20 s | Look, 20 s (±35°) |
| --- | ---: | ---: |
| Base max xy excursion | **0.00022 m** | **0.0033 m** |
| Base net xy at end | 0.000065 m | 0.0024 m |
| Head max xy excursion | 0.0014 m | 0.0116 m |
| Max base tilt | 0.075° | 0.330° |
| Max head tilt | 2.59° | 2.93° |
| Max head yaw error | 0.003° | **0.63°** |
| Min support margin | 0.0530 m | 0.0530 m |
| Max foot deviation | 0.0017 m | 0.0017 m |
| Max wrist residual | 0.0018 m | 0.0165 m |
| Max wrist orientation error | 0.28° | 1.02° |
| Solve mean / p95 / max | 0.51 / 0.95 / 4.82 ms | 0.59 / 1.01 / 7.25 ms |
| Fall | no | no |

All `[MEASURED]` from `logs/hold_neutral.csv` and `logs/hold_look.csv`.

The hold is **0.22 mm** of base excursion over 20 s. E3's comparable 20 s run — which included arm IK and the waist look — measured **≤3.08 cm x / 0.59 cm y** `[CITED: research/e3-spike-results.md]`. The scenarios are not identical (E3 moved the arms; this hold is neutral), so the honest comparison is the look row: **3.3 mm** base excursion while the waist moves ±35°, with **0.63°** yaw error. That is roughly an order of magnitude tighter than E3's whole-script excursion and is comfortably inside R2's ~5 cm bar.

Two caveats. The head is rigid to the torso — there is no neck `[CITED: research/e3-spike-results.md]` — so "look" is waist yaw plus base reaction, and the head translates **1.16 cm** during the look even though it tracks yaw to 0.63°. The head tilt is **2.6–2.9°** because the head task is soft relative to the base/posture stack; the HMD orientation is the torso orientation, not a stabilized camera. Both are body limitations, not controller bugs.

### 2.2 Reach, aim, punch (R1)

Nine reach configurations and nine aim configurations, all without a fall `[MEASURED]`.

**Reach (final wrist pose, 2 s after command):**

| Target | Config | Left residual | Right residual | Max base xy | Max ori err |
| --- | --- | ---: | ---: | ---: | ---: |
| forward (+0.20, ±0.05, +0.05 m from neutral) | left | 0.0094 m | 0.0129 m | 0.0147 m | 2.7° |
| forward | right | 0.0129 m | 0.0094 m | 0.0146 m | 2.7° |
| forward | both | 0.0220 m | 0.0220 m | 0.0288 m | 5.0° |
| up/out (+0.05, ±0.10, +0.18 m) | left | 0.0028 m | 0.0025 m | 0.0044 m | 1.2° |
| up/out | right | 0.0022 m | 0.0022 m | 0.0033 m | 1.2° |
| up/out | both | 0.0117 m | 0.0017 m | 0.0102 m | 1.6° |
| cross (+0.18, ∓0.10, +0.10 m) | left | 0.0075 m | 0.0056 m | 0.0080 m | 1.7° |
| cross | right | 0.0066 m | 0.0065 m | 0.0077 m | 1.7° |
| cross | both | 0.0188 m | 0.0175 m | 0.0174 m | 4.9° |

**Aim (wrist z-axis pointed at world point `[0.8, 0, 1.0]`; final pointing error):**

| Target | Config | Left aim err | Right aim err | Left residual | Right residual | Max base xy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| forward | left | **2.34°** | — | 0.0080 m | — | 0.0128 m |
| forward | right | — | **2.34°** | — | 0.0080 m | 0.0128 m |
| forward | both | 4.86° | 5.16° | 0.0208 m | 0.0205 m | 0.0260 m |
| up/out | left | **0.52°** | — | 0.0018 m | — | 0.0056 m |
| up/out | right | — | **0.18°** | — | 0.0031 m | 0.0063 m |
| up/out | both | 1.05° | 1.48° | 0.0149 m | 0.0087 m | 0.0287 m |
| cross | left | **2.09°** | — | 0.0071 m | — | 0.0081 m |
| cross | right | — | **2.03°** | — | 0.0089 m | 0.0092 m |
| cross | both | 6.26° | 5.67° | 0.0249 m | 0.0271 m | 0.0264 m |

**Punch** (wrist target +0.30 m forward, ±0.06 m lateral, +0.10 m up from neutral, aim orientation; 0.5 s ramp):

| Side | Final residual | Final ori err | Final aim err | Max base xy | Max base tilt |
| --- | ---: | ---: | ---: | ---: | ---: |
| left | **0.0095 m** | 3.47° | 3.64° | 0.0392 m | 4.25° |
| right | **0.0095 m** | 3.46° | 3.64° | 0.0392 m | 4.24° |

All `[MEASURED]` from `logs/reach_*.csv`, `logs/aim_*.csv`, `logs/punch_*.csv`.

Reading the numbers:

- Single-arm reach residuals are **2.2–12.9 mm**; both-arm residuals are **17.5–28.8 mm**. All final residuals are under R1's 3 cm bar, including both arms simultaneously.
- The aim controller points the **wrist z-axis** at the target within **0.18–2.34°** for one arm and **1.05–6.26°** for both. This is a pointing metric, not a full 6-DoF controller orientation: the roll about the pointing axis is free. A game that needs a specific controller roll must add a full orientation target; the residual will be larger than the pointing error.
- The punch residual is **9.5 mm**, vs E3's **13.5 cm** DLS residual `[CITED: research/e3-spike-results.md]`. This is a large improvement, but the scenarios differ: E3's arm-only DLS kept the legs and base fixed; this WBC lets the floating base/waist participate, and the punch is a 0.5 s quasi-static extension, not a fast strike. It is a reach that looks like a punch, not a measured punch speed.
- The transient orientation error during a fast aim change reaches **15.2°** (max over the trajectory) even though the final error is ≤2.34°. The wrist task at cost 500 cannot track a rotating pointing target instantaneously at 2 rad/s. For a game that aims while moving, the pointing error during the transition is the relevant number, not just the final value.

### 2.3 Dodge (R5)

The dodge commands a lateral head target and, in the `weight_shift` variant, moves the pelvis and CoM target by `min(0.5*amp, 0.06)` in the same direction. Duration 1.5 s; the profile ramps 0.4 s up, holds 0.2 s, returns over 0.4 s `[MEASURED]`.

| Requested head y | Variant | Peak achieved head y | Time to 90% peak | Time to peak | Max base xy | Max base tilt | Final head error | Fall |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.10 m | planted | 0.0144 m | 0.42 s | 0.52 s | 0.0014 m | 0.11° | 0.0024 m | no |
| 0.10 m | weight shift | **0.0840 m** | 0.50 s | 0.64 s | 0.0769 m | 0.76° | 0.0248 m | no |
| 0.15 m | planted | 0.0220 m | 0.42 s | 0.52 s | 0.0024 m | 0.19° | 0.0017 m | no |
| 0.15 m | weight shift | **0.1210 m** | 0.62 s | 0.82 s | 0.1046 m | 1.30° | 0.0187 m | no |
| 0.20 m | planted | 0.0296 m | 0.42 s | 0.52 s | 0.0032 m | 0.26° | 0.0036 m | no |
| 0.20 m | weight shift | **0.1344 m** | 0.64 s | 0.80 s | 0.1144 m | 1.37° | 0.0581 m | no |
| 0.25 m | planted | 0.0375 m | 0.42 s | 0.52 s | 0.0043 m | 0.32° | 0.0048 m | no |
| 0.25 m | weight shift | **0.1465 m** | 0.66 s | 0.86 s | 0.1246 m | 1.68° | 0.1129 m | no |

R5 asks for **10–25 cm** of head displacement in **<500 ms**, returning upright. The measured dodge:

- **Displacement: met.** The weight-shift variant reaches **8.4 / 12.1 / 13.4 / 14.7 cm** for requests of 0.10 / 0.15 / 0.20 / 0.25 m.
- **Timing: not met.** Time to 90% of peak is **0.50–0.66 s**; time to peak is **0.64–0.86 s**. The planted variant is fast (0.42 s to 90%) but only reaches **1.4–3.7 cm**.
- **Return: met at 0.15 m, not at 0.25 m.** Final head error is 1.9 cm at 0.15 m, 5.8 cm at 0.20 m, and 11.3 cm at 0.25 m within 1.5 s. The robot is upright (final tilt 2.4–3.5°), but the large dodge does not fully recover its lateral head position in the window.

The practical dodge point is **0.15 m request → 12.1 cm peak, 0.62 s to 90%, 1.9 cm return error, no fall**. That is a real lateral dodge, but it is a lean/weight shift, not a step, and it is slower than the R5 bar. Whether the game accepts 0.6–0.9 s is an unmeasured game question.

### 2.4 Continuous sequence (R4 proxy)

A 14.5 s unpaused sequence: stand → look ±35° → reach left/right → punch left/right → dodge 0.15 m → stand. A paused variant inserts 0.8 s neutral phases (20.1 s total). Both complete without falling `[MEASURED]`.

| Run | Ticks | Max base xy | Min support margin | Max wrist residual | Max wrist ori err | Max head yaw err | Max base tilt | Fall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| paused (20.1 s) | 1005 | 0.0937 m | 0.0315 m | 0.0734 m | 7.55° | 6.41° | 5.39° | no |
| unpaused (14.5 s) | 725 | 0.1008 m | 0.0179 m | 0.0767 m | 5.82° | **22.02°** | 3.57° | no |

The **0.10 m** max base excursion includes the intentional 0.15 m dodge; the reach/punch phases alone move the base **1.1–6.1 cm**. The **7.3–7.7 cm** wrist residuals occur during transitions, not at steady state; the single-action steady-state residuals are ≤2.9 cm. The unpaused **22°** yaw error is a transient: the look target jumps from +35° to −35° at the phase boundary, and the waist cannot traverse 70° instantly at the 2 rad/s limit. With 0.8 s neutral pauses, the max yaw error drops to **6.4°**. The game can pace look commands; the unpaused run shows what happens if it does not.

This is a proxy for R4 only for the upper-body transitions. It says nothing about walk→hold→walk, because there is no walking controller in this spike.

### 2.5 Solve time (R6)

`mink.solve_ik` with `daqp`, measured on CPU, headless `[MEASURED]`:

| Scenario | Mean | p95 | Max |
| --- | ---: | ---: | ---: |
| hold | 0.51 ms | 0.95 ms | 4.82 ms |
| look | 0.59 ms | 1.01 ms | 7.25 ms |
| reach (per config) | 0.58–1.08 ms | — | — |
| mixed paused | 0.57 ms | 0.94 ms | 1.71 ms |
| mixed unpaused | 0.81 ms | 2.44 ms | 6.90 ms |

The 50 Hz budget is 20 ms. The IK solve uses **3–13%** of it at the mean and **12%** at the p95 of the slowest run. R6's live co-residency with Superhot + Monado is **not** tested; this is a headless CPU number, and the live loop also pays for MuJoCo stepping, the bridge, and the game. The measured headroom is large, but the live claim remains an `[ASSUMPTION]` until run beside the game.

---

## 3. The hard half: walking is not a tuning problem

### 3.1 The measured boundary: double support works, single support does not

Two new scenarios probe the walking boundary without building a gait.

**Sustained lateral weight shift, both feet planted** (`shift`): the base and CoM targets move laterally by the requested amount over 3 s, then hold. `soft_base` lowers the base task to 500 and raises the CoM task to 500; `default` uses the standing weights `[MEASURED]`.

| Shift | Variant | Fall | Fall time | Max base xy | Final base y | Final CoM y | Max foot deviation | Min margin |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.06 m | soft_base | no | — | 0.0752 m | 0.0752 m | 0.0559 m | 0.0022 m | +0.0529 m |
| 0.08 m | soft_base | no | — | 0.1031 m | 0.1030 m | 0.0763 m | 0.0041 m | +0.0529 m |
| 0.10 m | soft_base | no | — | 0.1330 m | 0.1330 m | 0.0997 m | 0.0075 m | +0.0408 m |
| 0.12 m | soft_base | **yes** | 3.44 s | 0.2531 m | 0.2531 m | 0.1945 m | 0.0471 m | −0.0563 m |
| 0.06 m | default | no | — | 0.0773 m | 0.0770 m | 0.0587 m | 0.0239 m | +0.0415 m |
| 0.08 m | default | no | — | 0.1218 m | 0.1208 m | 0.0960 m | 0.0956 m | +0.0519 m |
| 0.10 m | default | no | — | 0.1654 m | 0.1622 m | 0.1288 m | 0.0848 m | +0.0264 m |
| 0.12 m | default | **yes** | 3.08 s | 0.2521 m | 0.2507 m | 0.2006 m | 0.0744 m | −0.0543 m |

The honest variant is `soft_base`: the default weights reach the same base position partly by **dragging the feet** (8.5–9.6 cm of foot deviation at 0.08–0.10 m shift), which is not a weight shift. With `soft_base`, the feet stay planted (2–7.5 mm deviation), and the limit is **0.10 m**: the CoM reaches 0.100 m, the stance foot center is at **0.1185 m**, and the margin is still +4.1 cm. At **0.12 m** the robot tips, with the base at 0.253 m and the margin at −5.6 cm.

**Single-support foot lift** (`lift`): shift 0.08–0.10 m over 3 s with the tuned `soft_base` weights, then lift the right foot 4–6 cm and move it forward 0–10 cm over 1 s `[MEASURED]`.

| Config | Fall | Fall time | Max lift before tip | Max base xy | Max tilt | Min margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shift 0.08, lift 0.04, forward 0.00 | **yes** | 4.68 s | 0.008 m | 0.2547 m | 13.2° | −0.022 m |
| shift 0.10, lift 0.04, forward 0.00 | **yes** | 4.54 s | 0.012 m | 0.2525 m | 16.8° | −0.020 m |
| shift 0.10, lift 0.04, forward 0.10 | **yes** | 4.68 s | 0.011 m | 0.2546 m | 16.8° | −0.064 m |
| shift 0.10, lift 0.06, forward 0.10 | **yes** | 4.84 s | 0.025 m | 0.2532 m | 17.2° | −0.049 m |

The lift starts at 3.2 s; every config tips **1.3–1.6 s later**, when the base excursion crosses 0.25 m. The commanded foot lift is 4–6 cm; the foot achieves only **8–25 mm** before the tip detector fires. A previous run with a looser 0.35 m fall threshold let the foot reach **18–58 mm** before the base crossed 0.35 m; the conclusion is the same. **The controller can shift 0.10 m laterally with both feet down; it cannot stand on one foot.** This is the walking bottleneck, and it appears before the first step, not at speed.

The root cause is the formulation, not the tuning. The IK QP has no contact-force or CoP variable; it cannot reason about the single-support wrench. When the swing foot leaves the ground, the physical robot rolls on the stance foot's sphere contacts, and the IK's joint targets do not regulate the resulting angular momentum. Raising the CoM task to 500–1000 and moving the base target further over the stance foot did not fix it in the tuning window; the tips persisted.

### 3.2 Option A — scripted gait + contact-force QP

**What it is.** Replace the IK-only QP with a whole-body QP over the floating-base dynamics: decision variables for joint accelerations/torques and contact forces; constraints for the floating-base equation of motion, friction cone / contact wrench cone, joint torque and velocity limits, and CoP inside the support polygon; tasks for swing-foot trajectory, stance-foot pose, CoM/ZMP, torso/head, and wrists. A gait state machine schedules double support (0.1–0.2 s), weight shift (0.3–0.5 s), single support/swing (0.4–0.6 s), and landing. Step length 0.10–0.20 m, swing height 0.04–0.06 m, cycle 0.8–1.2 s. Implementation options: Pinocchio + TSID, crocuddy/crocoddyl, OCS2, or a custom QP using MuJoCo's mass matrix and contact Jacobians with the installed `daqp`/`qpsolvers`.

**Why it might work.** The measured `shift` result says the task-space layer can move the CoM 0.10 m with both feet down; the missing piece is the single-support contact wrench. A contact QP can choose the stance-foot force/CoP to keep the CoM inside the foot, and can trade upper-body tracking against balance with explicit priorities. This is the standard classical humanoid walking stack, and the G1 has the ankle pitch/roll authority to modulate CoP.

**Why it might fail.** The G1's foot is a set of **sphere geoms**, not a flat polygon; the contact wrench cone is small and the foot can roll. The QP may find no feasible wrench for a 0.10 m CoM shift on one foot, or the ankle torque limits may bind. The single-support falsification below is designed to find out before spending the full estimate.

**Estimate.** **5–10 days** `[ASSUMPTION]`: 1–2 days QP formulation + MuJoCo dynamics, 2–3 days gait/contact schedule + swing trajectory, 2–3 days tuning, 1 day integration/logging. Risk: **high**. The estimate assumes the existing task-space upper-body stack can be reused as QP tasks; it does not assume the contact QP will work on the first formulation.

### 3.3 Option B — LIPM / capture-point footstep planner + inverse dynamics

**What it is.** Plan the CoM with a linear inverted pendulum model and a capture-point / divergent-component-of-motion controller; plan footsteps from the capture point; track the planned CoM and swing-foot trajectories with a whole-body inverse-dynamics QP. This is the more principled walking stack: it plans single support explicitly and gives a natural push-recovery mechanism.

**Why it might work.** The LIPM/capture-point layer handles the single-support instability analytically; the ID/QP realizes the plan with the full 29-DoF body. It is more robust to timing and pushes than a hand-scheduled gait, and it is the standard approach when the robot must recover rather than just repeat a cycle.

**Why it might fail.** The LIPM assumes a point mass and flat ground; the G1's head/arm motion shifts the CoM and adds angular momentum that the LIPM ignores. The measured reach/punch phases move the CoM by a few cm and the base by 1–6 cm; the walking controller must reject that as a disturbance while the upper-body tasks are active. The 8 GB RTX 4060 is not a constraint here (this is CPU control), but the implementation is larger than Option A.

**Estimate.** **7–14 days** `[ASSUMPTION]`: 2 days LIPM/footstep planner, 2–3 days capture-point/ZMP controller, 3–4 days ID/QP + swing, 2–3 days tuning, 1 day integration. Risk: **high**. Use this if Option A's single-support falsification fails but a contact QP is still plausible; if the contact QP itself cannot hold single support, Option B inherits the same foot-contact limitation.

### 3.4 Option C — hybrid learned legs + QP upper body

**What it is.** Use a **lower-body-only** locomotion policy (12 leg joints) for walking, and the measured task-space WBC for the 17 upper-body joints (3 waist + 14 arms). The policy receives root velocity/height/gait phase; the QP receives head/wrist/waist targets; a shared safety layer monitors CoM/support. The policy must be trained or selected to expect upper-body motion as a disturbance, and the QP must not write to the policy's leg joints.

**Why the prior learned route failed.** C3f/g/h bolted hard wrist IK onto the frozen 29-DoF Playground policy, which owns **all** joints including the arms. Hard IK fell in **3.39–4.84 s**; a 15% soft blend survived but never moved the wrist (0.119 m residual) `[CITED: research/c3fgh-results.md]`. The failure was an authority conflict: the policy used the arms for balance and the override removed its stabilizer.

**What is different here.** The proposed hybrid does **not** override a 29-DoF policy's arms. It requires a policy whose action space is lower-body-only, or one trained with an upper-body disturbance channel. HOMIE/OpenHomie is the obvious candidate: 12 lower-body actions, upper body treated as a trained disturbance `[CITED: research/humanoid-command-space.md]`. The QP's upper-body authority is then outside the policy's action space, not a fight over the same joints.

**Why it might fail.** The released HOMIE policy may not match the Menagerie G1 model, the 50 Hz interface, or the 8 GB GPU; the policy may still fall when the arms move beyond its disturbance envelope; the QP's head/arm targets may be larger than what the policy tolerates. The transfer is unverified. Do **not** attempt this by freezing the Playground policy's arms and handing them to the QP — that is the C3f failure with extra steps.

**Estimate.** **5–10 days** if a compatible lower-body policy exists and integrates `[ASSUMPTION]`; **15–30 days** if training/fine-tuning is required. Risk: **high**. The settling experiment is mandatory before the estimate is credible.

**Settling experiment (1 day).** Run the candidate lower-body policy at zero velocity for 20 s with (i) a 0.10 m lateral CoM shift, (ii) a 0.15 m head dodge, and (iii) a 0.20 m single-arm reach commanded through the QP. Pass: no fall, base excursion <10 cm, upper-body tracking residual ≤3 cm. Fail: any fall, or the wrist does not move. If it fails, the hybrid is not a shortcut.

### 3.5 Option comparison

| Option | What it adds | Days `[A]` | Risk | First experiment | Main failure mode |
| --- | --- | ---: | --- | --- | --- |
| **A. Scripted gait + contact QP** | Contact wrench/CoP control; gait state machine; swing trajectory | 5–10 | High | Single-support hold 2 s (below) | Sphere-foot contact cone too small; QP infeasible |
| **B. LIPM/capture-point + ID** | CoM planning; footstep planner; push recovery | 7–14 | High | 2-step walk in place, 3 gain sets | Upper-body CoM/angular-momentum disturbance; tuning time |
| **C. Hybrid learned legs + QP upper body** | Learned leg robustness; QP upper body outside policy action space | 5–10 (integration) / 15–30 (training) | High | Candidate lower-body policy + 0.10 m shift / 0.15 m dodge / 0.20 m reach | Policy/model mismatch; upper-body disturbance envelope too small |

---

## 4. Ranked proposal

### 4.1 Requirement coverage from this spike

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| R1 | Arms/head independently commandable; wrist residual ≤3 cm; orientation measured | **Standing: met.** Walking: not met. | Single-arm reach 2.2–12.9 mm; both-arm 17.5–28.8 mm; aim pointing 0.18–2.34° single, 1.05–6.26° both; punch 9.5 mm / 3.6° aim; orientation measured for every reach/punch. |
| R2 | Hold still ≤~5 cm net base over 20 s | **Met.** | Hold 0.22 mm; look 3.3 mm; mixed 9.4 cm only because it includes a 0.15 m dodge. |
| R3 | Walk ≥5 m, ≥0.3 m/s, 60 s, push recovery | **Not met.** | No walking controller. Single-support lift tips in 1.3–1.6 s; sustained weight shift limit 0.10 m. |
| R4 | 5× walk→hold→walk, no fall | **Not measured.** | The standing mixed sequence (look/reach/punch/dodge) completes without a fall, but there is no walk segment. |
| R5 | Head dodge 10–25 cm in <500 ms, return upright | **Partial.** | 12.1–14.7 cm achieved; 0.62–0.86 s to peak, not <0.5 s; return error 1.9 cm at 0.15 m, 11.3 cm at 0.25 m. |
| R6 | 50 Hz live on the 4060 laptop | **Solve time met headless; live unverified.** | 0.51–1.08 ms mean, 0.95–2.44 ms p95, ≤9.3 ms max; 20 ms budget. Superhot + Monado co-residency not tested. |
| R7 | Honest: achieved `mjData`, no fake poses | **Met.** | All reported poses read after `mj_step`; targets stored separately. |
| R8 | Complete a stage | **Not addressed.** | Depends on R3 + the game integration. |

### 4.2 Ranked plan

1. **Adopt the measured WBC as the standing/hold/look/reach/aim/punch/dodge executor (1–2 days, low risk).** It meets R1 for the standing case, R2, R5 on displacement, R6 on solve time, and R7. Wire it behind the existing `BodyCommand`/`BodyState` interface `[CITED: research/humanoid-command-space.md]`, keep the E3 rule that only achieved poses go to Monado, and report `AUTH_ROOT_VEL` as `UNSUPPORTED` until walking exists. The integration work is the bridge loop, the command interpolation, and the residual/saturation reporting — the controller itself is measured.
2. **Run the one-day single-support falsification for Option A.** This is the cheapest go/no-go for classical walking. If it passes, commit to the 5–10 day scripted gait + contact QP. If it fails, do not start the gait.
3. **If the falsification fails, run the one-day Option B test** (2-step walk in place with a capture-point controller). If B also fails, the classical walking path is not a one-week project; report that plainly and evaluate Option C or a different body/route.
4. **Option C only with a verified lower-body policy.** Do not bolt the QP onto the frozen 29-DoF Playground policy; C3f/g/h already showed that fails. The 1-day settling experiment decides whether the 5–10 day integration is credible.

The reason for this order: the standing half is already measured and useful to the game. Walking is the expensive unknown, and the single-support falsification tests the exact boundary where the measured controller failed. Spending 5–14 days on a gait before knowing whether a contact QP can hold one foot up would repeat the C3 mistake of tuning a controller that cannot satisfy the structural constraint.

### 4.3 One-day falsification (Option A gate)

**Goal.** Decide whether a contact-force QP can hold the G1 on one foot. This is the measured failure of the IK-only WBC, and the new ingredient in Option A.

**Setup.** Same Menagerie G1, flat ground, MuJoCo. Build a 50 Hz QP with the floating-base dynamics and the stance-foot contact wrench (or, as a cheaper first pass, a CoM/ZMP controller with a contact-force distribution). Sequence: 0–3 s shift the CoM 0.10 m over the left foot (the measured `soft_base` limit); 3–3.3 s lift the right foot to 4 cm; 3.3–5.3 s hold single support; 5.3–5.6 s set the right foot down. Log base excursion, base tilt, CoM, support margin, foot height, and contact forces after `mj_step`.

**Pass.** No fall for 6 s; base excursion <5 cm during the single-support hold; right foot lift ≥3 cm; base tilt <10°.

**Fail.** Any fall, base excursion >10 cm, or foot lift <2 cm. A fail means the contact QP cannot stabilize the G1's sphere-foot single support with a 0.10 m CoM shift; do not spend the 5–10 days on the gait. Go to Option B's 2-step test, then Option C.

**Why this test.** It is the smallest experiment that exercises the missing ingredient (contact wrench/CoP control) on the measured failure (single support). It is one day because the QP can start from the existing task-space stack and the same model; it does not need a footstep planner, a swing trajectory, or a gait state machine.

---

## 5. What would make this fail

1. **The sphere-foot contact model.** The G1 foot is four sphere geoms per foot, not a flat sole. The contact wrench cone may be too small for a 0.10 m CoM shift on one foot, or the foot may roll before the QP can modulate CoP. The measured IK-only tips are consistent with this; the one-day falsification will show whether a contact QP changes it.
2. **The QP solve budget.** The measured IK solve is 0.5–1.1 ms; a contact QP with 4–8 contact points and friction constraints could be 2–10 ms. That still fits 20 ms, but the live loop also runs MuJoCo, the bridge, Monado, and Superhot on one 8 GB laptop. If the contact QP plus the game misses 50 Hz, R6 fails even if the controller is correct.
3. **Upper-body vs balance authority.** The mixed standing sequence already shows wrist residuals up to **7.7 cm** during fast transitions and a **22°** yaw lag when look targets jump. Walking adds a much larger disturbance. The walking QP will sacrifice arm/head tracking to stay upright, and the game may see the hands lag or the HMD yaw lag. The measured steady-state residuals do not guarantee walking residuals.
4. **Dodge timing.** R5 asks for <500 ms; the measured dodge takes **0.62–0.86 s** to peak. If the game's bullets require a 500 ms dodge, this body cannot do it with a lean/weight shift. A faster dodge needs a ballistic step or a different controller, which is walking-adjacent work.
5. **No neck.** The HMD yaw is the torso yaw; a fast look turns the chest and perturbs balance. Standing look is measured and fine; walking + look is not. A game that needs independent head aim while walking will expose this body limitation.
6. **The game pipe.** R6/R7 are only meaningful if Superhot under Proton accepts the synthetic HMD at all. C0 is still unrun `[CITED: ../docs/archive/2026-10-23-clank-state-of-findings.md]`. If the game rejects the pipe, no body controller matters.
7. **The day estimates.** They are `[ASSUMPTION]`s. Contact-QP and gait tuning routinely take 2–3× the first estimate, especially on a body whose single-support behavior is already marginal. The measured 0.10 m weight-shift limit is a warning, not a guarantee that 5–10 days is enough.

---

## 6. Honesty, limitations, and what is not measured

- **No walking, no gait, no contact-force QP.** The measured controller is task-space IK over the floating base. The walking options are designs, not implementations.
- **No push recovery, no uneven ground, no external forces.** The game applies no forces in this spike; the hold is a stiff position-actuator stand on flat ground.
- **No live game/Monado integration.** The solve time is headless. The E3 bridge and the achieved-pose rule are the integration path, but the live co-residency is untested.
- **No hand closure.** The Menagerie G1 scene used here has no fingers/hands; the `BodyCommand` closure field is unsupported by this model. The `scene_with_hands.xml` option exists but was not used.
- **The aim metric is pointing, not full controller orientation.** `aim_R_min` rotates the wrist z-axis toward the point and leaves roll free. The final pointing error is 0.18–2.34° (single arm); a game-specified roll would add error.
- **The punch is quasi-static.** It is a 0.5 s wrist extension, not a measured strike speed or impact. The 9.5 mm residual is a reach result, not a punch-dynamics result.
- **The reach workspace is a 3-target sample**, all relative to the initial wrist pose. It is not a full workspace box.
- **The support margin is a 2D convex hull of the eight foot-sphere centers**, not the true contact patch. It is an approximation and is conservative in some directions.
- **The head tilt is 2.6–2.9°** in hold/look; the head task is soft. The HMD orientation is the torso orientation; the head mesh is rigid to the torso.
- **The dodge is a lean/weight shift, not a step.** It moves the base 7.7–12.5 cm and returns upright, but it does not step. The 0.25 m request does not fully return within 1.5 s (11.3 cm final head error).

---

## 7. Reproduce

```bash
cd /home/nate/Projects/Clank/spike/wbc_classical
/home/nate/Projects/Clank/spike/e3/.venv/bin/python wbc_spike.py --scenario all
# or: --scenario hold|look|reach|aim|punch|dodge|shift|lift|mixed
```

Outputs:

- `logs/summary.json` — every metric above, keyed by scenario.
- `logs/run_all.txt` — the printed run log.
- `logs/hold_neutral.csv`, `logs/hold_look.csv` — 50 Hz per-tick hold/look.
- `logs/reach_<target>_<config>.csv`, `logs/aim_<target>_<config>.csv` — per-tick reach/aim.
- `logs/punch_<side>.csv`, `logs/dodge_<amp>_<variant>.csv` — per-tick punch/dodge.
- `logs/shift_<shift>_<variant>.csv`, `logs/lift_<config>.csv` — walking-boundary probes.
- `logs/mixed_paused.csv`, `logs/mixed_fast.csv` — continuous sequence.

Environment: the E3 venv (`mujoco 3.17.0`, `mink 1.3.0`, `numpy 2.2.6`, `scipy 1.18.1`, `qpsolvers` with `daqp`). Model: Menagerie G1 `scene.xml`, same as E3. Nothing outside `spike/wbc_classical/` was modified; the design doc, other research files, `spike/e1`, `spike/e2`, and `spike/e3` were not edited.

---

## Appendix A — full reach/aim final values

### A.1 Reach (final, 2 s)

| Target | Config | L res (m) | L ori (°) | R res (m) | R ori (°) | Base max xy (m) | Margin (m) | Fall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| forward | left | 0.0094 | 2.54 | 0.0129 | 2.29 | 0.0147 | 0.0533 | no |
| forward | right | 0.0129 | 2.33 | 0.0094 | 2.53 | 0.0146 | 0.0533 | no |
| forward | both | 0.0220 | 4.61 | 0.0220 | 4.61 | 0.0288 | 0.0533 | no |
| up_out | left | 0.0028 | 0.76 | 0.0025 | 0.35 | 0.0044 | 0.0497 | no |
| up_out | right | 0.0022 | 0.34 | 0.0022 | 0.79 | 0.0033 | 0.0504 | no |
| up_out | both | 0.0117 | 1.60 | 0.0017 | 1.33 | 0.0102 | 0.0409 | no |
| cross | left | 0.0075 | 1.59 | 0.0056 | 1.22 | 0.0080 | 0.0533 | no |
| cross | right | 0.0066 | 1.08 | 0.0065 | 1.62 | 0.0077 | 0.0533 | no |
| cross | both | 0.0188 | 4.15 | 0.0175 | 3.49 | 0.0174 | 0.0533 | no |

### A.2 Aim (final, 2 s; pointing error at `[0.8, 0, 1.0]`)

| Target | Config | L aim (°) | L res (m) | R aim (°) | R res (m) | Base max xy (m) | Fall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| forward | left | 2.34 | 0.0080 | — | — | 0.0128 | no |
| forward | right | — | — | 2.34 | 0.0080 | 0.0128 | no |
| forward | both | 4.86 | 0.0208 | 5.16 | 0.0205 | 0.0260 | no |
| up_out | left | 0.52 | 0.0018 | — | — | 0.0056 | no |
| up_out | right | — | — | 0.18 | 0.0031 | 0.0063 | no |
| up_out | both | 1.05 | 0.0149 | 1.48 | 0.0087 | 0.0287 | no |
| cross | left | 2.09 | 0.0071 | — | — | 0.0081 | no |
| cross | right | — | — | 2.03 | 0.0089 | 0.0092 | no |
| cross | both | 6.26 | 0.0249 | 5.67 | 0.0271 | 0.0264 | no |

## Appendix B — controller parameters

```
dt                 0.02 s (50 Hz)
substeps           10 (MuJoCo dt = 0.002 s)
solver             mink.solve_ik, "daqp"
feet               5000 pos / 5000 ori
pelvis             5000 pos / 5000 ori
CoM                100
head               10 pos / 100 ori   (look: 0/500; dodge: 500/500)
wrists             500 pos / 500 ori
posture            0.1
configuration      min_distance_from_limits = 0.01
velocity limit     2.0 rad/s (all joints)
joint target limit 2.0 rad/s
target smoothing   alpha = 1 - exp(-dt/tau), tau = 0.3 s (mixed only)
model              Menagerie unitree_g1/scene.xml
                   nq=36, nv=35, nu=29, stand base z=0.79 m
actuators          position, kp=500, dampratio=1
aim point          world [0.8, 0.0, 1.0]
```

The weights are a tuned point, not a robust controller. The measured margins are in the tables above; the tuning window was not swept to failure for every weight. Treat the numbers as "this configuration works on this model," not as a robustness claim.
