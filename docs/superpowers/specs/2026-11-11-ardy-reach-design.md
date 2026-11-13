# ARDY reach — Design

- Date: 2026-11-11 (originally written MotionBricks-first as `2026-11-11-motionbricks-reach-design.md`; retitled and re-argued the same day after `research/motion-generators.md`)
- Status: accepted, being implemented. Plan: `docs/archive/plans/2026-11-11-ardy-reach.md`
- Parents: `research/humanoid-command-space.md` (BodyCommand contract), `research/upper-body-override.md`, `research/motion.md`, `research/wbc-classical-proposal.md`, `research/motion-generators.md` (the generator decision and its measurements)
- Implements against: `spike/wbc/sonic_executor.py`, `spike/wbc/sonic_runner.py`, `spike/wbc_game/reach_vr.py`, `clank/workspace.py`, `clank/player_scale.py`
- Does not implement: brain/planner changes, locate, glow, neck, HumanML3D, motion-model training

Claim tags: `[MEASURED]` local file or named run; `[CITED]` external source; `[OPERATOR]` the game's owner; `[ASSUMPTION]` untested. Builder calls are marked **(call)**.

## 1. Problem

Telling the body to grab something in front of it produces a dance: a long, wacky arm path, and a torso that folds so the hand can reach. The head does not look at the object.

Two diseases, already separated:

1. **The path.** `reach_vr` streams a Cartesian cruise on SONIC's VR 3-point channel, which is 9 positions + 12 quaternion floats with no velocity and no horizon `[CITED: NVIDIA observation_config.md:225-246, recorded in research/upper-body-override.md]`. Ours measures 0.77–1.05 s settle and +75 % vertical overshoot, 3.1–7.5 s and 13–27 direction changes per reach `[MEASURED: research/motion.md]`. A hand-written reference is why it looks hand-written.
2. **The posture.** The VR policy is a whole-body optimiser over a position setpoint: it moves the root and twists the waist to buy the arm millimetres. `WRIST_BOX` (`clank/workspace.py`) is a steering limit on that channel, not the arm's reach `[MEASURED: AGENTS.md; research/motion.md]`.

Two repairs are already falsified and stay falsified. Stuffing IK joints into SONIC: 1.14 → 14.40 cm on hold, and the 17-joint reference override delivers ~33 mm of a 235 mm command `[MEASURED: research/sonic-gate-results.md; research/upper-body-override.md]`. And authoring the pose with IK and asking a generator to interpolate to it: that was this document's first shape, and it exists only because MotionBricks cannot be told "hand here".

## 2. Claim

A grab is three jobs, and only the first one changed:

| Job | Who | What |
| --- | --- | --- |
| **Intent** | the adapter | right hand at the object; feet planted; or a root waypoint when the object is out of reach |
| **Motion** | ARDY-G1 | a motion prior produces the reach, the look and the steps from those constraints |
| **Physics** | SONIC mode 0 | tracks the generated reference; legs, balance and the 0.9 s look-ahead stay on the policy |

ARDY accepts end-effector constraints natively — `right-hand`, `left-foot`, `right-foot`, `root2d`, at arbitrary frames and over long horizons `[CITED: ardy/constraints.py; kimodo docs/source/user_guide/constraints.md]` — and its G1 skeleton names the joints we already report (`right_wrist_yaw_skel`, `right_hand_roll_skel`, `right_ankle_roll_skel`, `right_toe_base`) `[REPO: ardy/skeleton/definitions.py, G1Skeleton34]`. So the pose is *chosen by a motion prior* rather than solved around a target, which is the difference between a body that decides how to reach and one that is folded into place.

NVIDIA's G1 deployment is this pairing: a generator emits kinematic motion, SONIC tracks it `[CITED: arXiv:2604.24833 §7.4; nv-tlabs/kimodo README]`. We have the tracker; we were feeding it the wrong channel.

## 3. Goals, non-goals, anti-goals

### Goals

1. A reach or grab toward a 3D point looks like a person reaching: feet planted when it is in front, a turn or step when it is not, chest toward the object, hand on it.
2. The camera looks at the grab. G1 has no neck `[MEASURED: research/body-alternatives.md]`, so "look" is waist yaw when planted and root yaw when the body turns. **Measured limit, 2026-11-11: this goal is not reachable for off-axis objects.** The reach and the look compete for the same geometry — see §8.5. The goal stands for objects near the body's facing; elsewhere the honest outcome is a reach.
3. Generation is live: one ARDY call per window, 0.07–0.20 s for 2.08 s of motion `[MEASURED: §7]`.
4. The grab closes in the same process as the reach; a 2 s gap between reach and close drifted the hand 0.09 m `[MEASURED: AGENTS.md]`.
5. `hold`, `look`, `rest` keep working on the VR channel. This replaces the **reach/grab executor**, not the `BodyCommand` contract.
6. Honest outcomes: `done`, `saturated`, `timeout`, with the achieved hand residual in metres.

### Non-goals

- Training or fine-tuning ARDY, SONIC, or any text-to-motion model. (§12 explains why this is not the constraint it looks like.)
- A text encoder at runtime: the checkpoint conditions on a single pooled 4096-vector and v1 passes zeros `[MEASURED: §7]`.
- HumanML3D, AMASS, SMPL-H, or any new motion corpus.
- A neck, a second body, or replacing the G1.
- Replacing `BodyCommand` / `clank.intent`: the generator sits under the existing reach/grab skill.
- Continuous replanning during a grab. A grab generates once, and at most once more with a root goal.
- MotionBricks and Kimodo. They survive as the documented fallback (§14), not as this build.

### Anti-goals

- A reach that plants the hand by folding the torso while the head looks away.
- A reach that walks when the object is inside the planted envelope.
- Treating generated kinematic poses as achieved. The relay sends `mjData` after `mj_step`.
- Encoding a constraint channel that is silently inactive: a scalar `cfg_weight` zeroes constraint guidance (§7.3).
- Running ARDY inside the 50 Hz body process, or torch inside `clank/`.

## 4. Proven basis

- SONIC: encoder/decoder ONNX at 50 Hz, PD at 500 Hz, `spike/wbc/sonic_runner.py`. Mode 0 writes a 10×29 joint reference at stride 5 — a 0.9 s look-ahead. Mode 1 uses VR 3-point instead `[MEASURED: build_encoder_obs]`.
- Freeze-hold: standing clip `squat_001__A359` frame 0, anchor index frozen so the look-ahead cannot contain walking; hold RMS 1.3e-4 m/s `[MEASURED: research/c3m-results.md]`.
- ARDY-G1 on this machine, RTX 4060, GPU to itself `[MEASURED: research/motion-generators.md §9]`:

| Quantity | Measured |
| --- | --- |
| Weights resident / peak VRAM | 0.93 GB / **0.94 GB** |
| One 52-frame window (2.08 s), 10 steps | **0.07–0.20 s** |
| Hand-constraint residual, `cfg_weight=(1, 2)` | **45.5 mm** (33–80 mm across unseeded runs) |
| Root-waypoint adherence, 0.60 m requested | **0.566 m travelled**, 32.8 mm hand residual |
| Clip frame 0 vs live body | **1 mm / 0.8°** |
| Export | `(1, T, 36)`, z-up and +X forward — our pose36 exactly |

- `clank/` is stdlib-only; NumPy / MuJoCo / torch / ARDY live in a spike venv.
- In-box right-hand cruise target `(0.35, −0.28, 0.15)` m in the root frame landed at 16.4 mm `[MEASURED: research/motion.md]`.
- Reported hand point is `wrist_yaw_link` plus `VR_OFFSETS[side]` in the pelvis frame `[MEASURED: sonic_runner.VR_OFFSETS]`.

## 5. Architecture

```text
pixel / named point
        │
        v
PlayerScale.pixel_point        robot-space XYZ (game space ÷ declared scale)
        │
        v
constraints                    right hand at the point (+ facing)
                               feet planted, or a root waypoint out of reach
        │
        v
ARDY-G1 (25 Hz)                canonical stand placed at the body's root/heading,
   text_feat = zeros(1,1,4096)  cfg_weight = (text, constraint)
        │
        v
pose36 stream (T, 36)          MujocoQposConverter output, z-up, +X forward
        │
        v
resample 25 → 50 Hz            slerp the root; finite-difference the joints
        │
        v
SONIC mode 0                   tracks it; 10-frame look-ahead, flatten-hold at the end
        │
        v
mj_step 500 Hz                 achieved qpos → body_state.json → pose_relay → game
        │
        +-- fingers            14 motors, unchanged clasp, not in the 29
```

The adapter generates; the 50 Hz body process only writes a pose ring and plays deposited clips. Torch never enters the process that ticks SONIC (§9).

Deleting the adapter must leave `hold` / `look` / `rest` / manual mode intact. A missing checkpoint or GPU is `FLAG_UNSUPPORTED` on reach, never a session tear-down.

## 6. Constraints, not a pose

There is no still frame to author. The adapter states three things.

### 6.1 Inputs

- Latest achieved **36-float pose** from the executor's ring (§9.1). Not `BodyState.q`, which is policy-order and cannot be unpacked into pose36.
- Object point in **robot world metres**, via `PlayerScale.pixel_point`, so a declared scale cannot be applied to the camera and forgotten on the hand `[MEASURED: clank/player_scale.py — that omission is the 1.43× miss]`.
- Side: `left` | `right`.

### 6.2 Hand constraint

`RightHandConstraintSet` / `LeftHandConstraintSet` constrain the wrist joints plus hips, so the pose is lifted from the *current achieved pose* and only the hand is displaced. A single-frame constraint at the end of the window is the hardest case and still measured at 45.5 mm `[MEASURED: §4]`.

**(call)** v1 constrains the hand at the **last frame** of the window and leaves the rest free. If the residual stays above `ROOT_GOAL_RESIDUAL_M = 0.10 m`, escalate once (§6.4) rather than adding mid-window constraints.

**Open defect, verified in the code, fix before any live gate:** the constraint pins the wrist *link* (`right_wrist_yaw_skel`'s anchor), while the game and the clasp both look at the relayed hand, which is that link plus `VR_OFFSETS[:2] = [0.18, ∓0.025, 0]` (`spike/wbc/sonic_runner.py:114-116`). So the residual understates the game-visible error by 0.18 m, and `reach_ardy`'s clasp compares a levered hand against a point the wrist was pinned to, which means `--grab` cannot fire. Details and the fix: `research/reach-architecture-critique.md` §1.

### 6.3 Feet

Feet stay unconstrained in v1: the model's own foot-contact prior keeps a planted stance for a reach in front of the body, and constraining the feet of a *generated* pose we have not produced yet would constrain them to a pose the generator has not chosen. If live runs show skating, the next step is `left-foot` / `right-foot` at the first and last frames — a constraint, not an IK solve.

### 6.4 Escalation

When the object is out of reach, or the first generation lands more than `ROOT_GOAL_RESIDUAL_M` short, the adapter regenerates **once** with a `root2d` waypoint placing the root where the hand can reach (measured: 0.566 m of a 0.60 m request). One escalation, then stop with the best result. No mink, no constructed stepped pose: under ARDY the step *is* a constraint.

`end_pose.decide_next(...)` is the pure policy that decides generate / escalate / stop / saturated, unit-tested without a session.

`MAX_STEP_M = 1.5` still guards against a bad unprojection asking the body to walk across the room.

## 7. ARDY generation

### 7.1 Call shape

- **Model:** `ARDY-G1-RP-25FPS-Horizon52`, checkpoint at `~/.local/state/Clank/ardy/`. Text tower absent: `load_model(..., text_encoder=False)`.
- **Text:** `text_feat = zeros(1, 1, 4096)` — the checkpoint's `llm_shape` is `[1, 4096]`, one pooled vector per sample, and an empty prompt is already handled as zero (`ardy_model.py:325`) `[MEASURED: research/motion-generators.md §9]`.
- **Guidance:** `cfg_weight = (1.0, 2.0)`. Non-negotiable (§7.3).
- **Steps:** default `num_base_steps` (10); anything larger must be refused in Python.
- **Placement:** canonical stand translated to the body's current root and heading (`init_global_translation`, `init_first_heading_angle`). v1 does not pass `init_history_sequence`, because feature assembly lives in ARDY's data pipeline and our body state is MuJoCo `qpos`; SONIC starts from the live `mjData` regardless, so the small pose offset is absorbed by the tracker.
- **Output:** `model.motion_rep.inverse(...)` → `MujocoQposConverter.dict_to_qpos(...)` → `(1, T, 36)`.

### 7.2 What "reach" means as a constraint

The hand target is the object point in ARDY's model space (y up, z forward), derived by the same conversion the exporter uses in reverse. The wrist **orientation** is currently *constrained* rather than inherited: `hand_constraint` passes `global_joints_rots` through, so `RightHandConstraintSet` pins it as an equality. That is 3 DoF the prior would otherwise spend on looking human (palm-down approach, thumb lead), so either relax it or expect a systematic bias — see `research/reach-architecture-critique.md` §8 (R8).

### 7.3 Three traps

| Trap | Consequence | Rule |
| --- | --- | --- |
| scalar `cfg_weight` with `cfg_type="separated"` | `w_text, w_cstr = cfg_weight, 0.0`: constraints silently ignored, 870 mm instead of 60 mm `[MEASURED]` | always pass `(text, constraint)` |
| `num_denoising_steps > num_base_steps` | CUDA device-side assert, not a Python error | refuse in `check_steps` |
| constraint tensors split across devices | `create_pairs` stacks frame and joint indices; mixing raises | assemble constraints on the CPU |

Also easy to get wrong, and cheap to check: `num_frames` must be a multiple of `num_frames_per_token` (4).

## 8. SONIC tracking

### 8.1 Mode switch

`SonicExecutor` is hard-coded `mode=1`. During generated playback it runs **mode 0** so the 29-joint look-ahead is live.

1. Load the resampled clip (`jp`, `jv`, `bp`, `bq`).
2. `runner.mode = 0`. **Do not call `reset()`** — that teleports. Keep the current `mjData`.
3. Advance the clip index at 50 Hz so the look-ahead reads yet-unplayed frames.
4. At the last frame, flatten the look-ahead by repeating the last row across the remaining clip rows, which is the hold that cannot walk `[MEASURED: research/c3m-results.md, H1f]`.
5. Then `mode = 1` freeze-hold, claiming the achieved head and wrists so TTL expiry does not snap to the standing clip.

`sonic_dict` emits no `bl`/`ba`, and `SonicRunner.reset` reads both unconditionally `[MEASURED: sonic_runner.py:313-314]`, so a generated dict must not be handed to `reset()`.

### 8.2 Fingers

ARDY generates no Dex3 angles. Keep `GRASP_Q` / `HAND_SLEW_RAD = 0.25`. `--grab` starts the clasp when the **achieved** reported hand is within `CLASP_START_M = 0.08 m`, without waiting for the clip to end, through `AUTH_CLOSURE` on `control.sock`. Relay grab mode stays `--grab-mode both`.

### 8.3 Correction

At clip end: if the residual is within `TRACK_SUCCESS_M = 0.08 m`, done. If it is worse and the object is outside the planted envelope, `decide_next` returns `escalate` (one root goal). Never a Cartesian cruise, never `target += 0.6 * error` — that integrator walked the command to y = −0.92 m `[MEASURED: research/motion.md]`.

### 8.4 Head

No head pin during a generated reach. A constant 20 Hz pin twisted the torso and moved the root frame the target lived in `[MEASURED: research/motion.md]`. The generated clip contains whatever look it produces; after playback, freeze-hold claims the achieved head.

### 8.5 The look, measured (2026-11-11)

**The generated reach does not aim the torso at the object, and making it do so costs the reach.** This was investigated four ways and the answer is a property of the body, not a bug:

| attempt | torso→object | hand residual |
| --- | --- | --- |
| default (nothing asked) | 44° | **72 mm** |
| `root2d` facing goal on the first generation | 15° | 97 mm |
| turn-only clip, then reach | 35° | 93 mm |
| waist-rotation constraint (see below) | **8°** | **180 mm** |

- The turn is not the reach optimisation it was assumed to be. Signed measurement (`probe_ardy_track.py`, commit `ae1a8fd`) shows the torso turns only +0.2…+8.6° for an object 30° to the right and +10…+24° for one to the left — i.e. about as much as the *arm* needs, never toward the object. The offset is roughly the object's own bearing.
- Walking closer does not change it: the turn is the same at 0.30 m and 0.48 m. Distance changes whether the hand arrives (0.68 m is beyond the ~0.55 m arm and measures 154–213 mm), not where the torso points.
- Facing is expensive because it moves the object away from the working arm: the right hand's distance to it goes 0.35 → 0.53 m.
- A **waist-rotation constraint** does what it is asked (44.4° → 8.4°) and the tracker follows it, but the hand then stops 130–215 mm short and the torso leans 18–26°. Note for anyone retrying this: the **position** channel is geometrically dead here, because G1's waist chain is coaxial with yaw — pelvis and `waist_yaw` are the same point and `waist_roll`/`waist_pitch` sit 4.4 cm above it, so a 30° turn moves the pinned joints 0.0/2.1/2.1 mm. Only `global_joints_rots` expresses it.

The honest design sentence: on a body with no neck, looking at an off-axis object and reaching it with the far arm are the same geometric conflict, and the reach is the thing that must survive. A future fix is not a constraint but a **turn-then-step-closer** sequence, so the object is within easy arm reach from a facing stance; that is unmeasured.

## 9. Adapter / process split

Same topology as `reach_vr.py`: a client process generates, the body process plays. Torch and ARDY stay out of the 50 Hz process.

| Unit | Lives in | Notes |
| --- | --- | --- |
| Constraint authoring | adapter (`reach_ardy.py`) | object point → constraints |
| ARDY wrapper | adapter (`spike/wbc/ardy_gen.py`) | torch + ARDY; may block 0.07–0.20 s |
| Chain policy | `spike/wbc/end_pose.py` | pure, stdlib |
| 36-float pose ring | executor writes, adapter reads | §9.1 |
| Clip playback / mode switch | `spike/wbc/sonic_executor.py` | no torch |
| Clasp | executor `_clasp`, triggered by adapter `AUTH_CLOSURE` | in-process with the motors |
| `PlayerScale`, `WRIST_BOX`, intent, `BodyCommand` | `clank/` | stdlib; no generator imports |

### 9.1 Mailbox (session dir)

Under `~/.local/state/Clank/sessions/body/`:

| File | Writer | Meaning |
| --- | --- | --- |
| `pose36.json` | executor, every tick | ring of the last 32 achieved poses + `seq`/`t_mono_ns` |
| `mb_clip.npz` | adapter | resampled 50 Hz clip (`jp`, `jv`, `bp`, `bq`) |
| `mb_play.json` | adapter | `{"seq", "path"}` only — no world XYZ, no clip bytes |
| `mb_status.json` | executor | `playing` / `done` / `failed` + frame counter |
| `control.sock` | adapter | unchanged `BodyCommand`: freeze-hold, closure, trigger |

## 10. What success looks like

| | Today (cruise) | This design |
| --- | --- | --- |
| Path | 13–27 direction changes, 3–7 s | one generated window, 0.6–1.9 s of motion |
| In-front grab | torso folds, head off-axis | generated reach, residual ≤ 8 cm on the live body |
| Off-axis / far | crane, or `WRIST_BOX` clamp | the model turns or steps; one root-goal escalation |
| Look | not requested | in the generated motion |
| Hand constraint adherence | n/a | 45.5 mm measured (33–80 mm spread) |

The number that can still kill it: **SONIC mode 0 tracking a generated reference.** NVIDIA claims the pairing; we have not measured it on this deploy. >15 cm means stop, not fall back to the cruise.

## 11. Measurement gates

1. **Packing** — `mjData.qpos` → 36-float → `jp` → qpos round-trips under 1e-6 on the 29 hinges, hands untouched. ✅ *done*
2. **Envelope** — planted/step classification and root-goal arithmetic. ✅ *done*
3. **Generation** — 52 frames, no NaNs, starts at the placement pose, ends at the hand constraint within 0.10 m; VRAM < 2.0 GB. ✅ *done: 0.94 GB, 45.5 mm*
4. **Root goal** — 0.60 m request travelled to within 5 %. ✅ *done: 0.566 m*
5. **Tracked playback, game off** — SONIC follows a generated clip: residual < 8 cm **on the
   glove** (`VR_LEVER`; the earlier 66 mm was wrist-to-wrist and is not comparable), no fall,
   shape at least as good as the cruise's, look error < 15°. ✅ *measured 2026-11-11: median 66 mm over 6 runs, 4/6 inside the gate, 0 falls, head tracking within 6° of the clip's own look. The gate is met at its boundary, not comfortably — the 80 mm clasp threshold sits in the same place. Look relative to the **object** is a separate, unmet goal: 35–59° (see §8.5).*
6. **Live grab, game up** — one in-front and one out-of-reach grab, clasp on approach, `hold`/`look`/`rest` afterwards. *outstanding*
7. **Shape** — added 2026-11-11 because gates 1–6 can all pass with the stutter intact: path
   length ÷ net travel and reversals (`spike/wbc/path_shape.py`, the same definition
   `reach_smooth` counts), measured on the generated path and on three `reach_vr` runs of the
   same targets. The cruise on record is 2 reversals for a smooth path, 4 with two corrections,
   13–27 for the hand-written sweep, and nothing has ever counted a generated path.

## 12. Does text matter, and is a smaller model worth training

Short answer: not now, and probably never.

- Text is **not required** to run: the checkpoint conditions on one pooled 4096-vector, and zeros are the same path an empty prompt takes. v1 ships no encoder, which is where ~14 GB of VRAM would otherwise go `[MEASURED: research/motion-generators.md §3-4]`.
- If text is wanted later, the cheapest route is **distillation**: freeze the motion model, train a small encoder to reproduce LLM2Vec's 4096-d vector on our own caption set. The teacher supplies the labels, so no motion-dataset licence is involved.
- **Training a motion model is out.** Compute, licences, and the fact that a 4060 must run the result all say no. The only G1-native corpus at scale (BONES-SEED, 142,220 motions / 288 h) is gated to academics and sub-$1M startups and forbids "commercially substitutable" motion generators while permitting policies and trackers `[CITED: bones.studio/info/seed-license]`.

## 13. Builder calls (defaults, rejectable)

1. `cfg_weight = (1.0, 2.0)`; steps = `num_base_steps`; text zeros.
2. Hand constrained at the last frame only; wrist orientation inherited.
3. Feet unconstrained in v1; add foot constraints only if live runs skate.
4. One escalation to a root goal; `ROOT_GOAL_RESIDUAL_M = 0.10`, `TRACK_SUCCESS_M = 0.08`, `CLASP_START_M = 0.08`, `MAX_STEP_M = 1.5`.
5. No `init_history_sequence` in v1; placement via `init_global_translation` / `init_first_heading_angle`.
6. `hold` / `look` / `rest` stay on VR mode 1. Only reach/grab use this path.
7. `reach_vr.py` stays in the tree as the A/B until gate 6 passes.
8. Adapter generates; executor plays; torch out of the 50 Hz process.

## 14. Fallback

If ARDY cannot pass gate 5 or 6 on this machine, the fallback is **MotionBricks live with Kimodo-authored still poses**: Kimodo generates the still frames once from `right-hand` + foot constraints, so they are in-distribution, and MotionBricks in-betweens them at 2.67 ms with no torch in the loop. That is a plan revision, not a code switch — see `research/motion-generators.md` §6.

## 15. Why not the alternatives

- **MotionBricks alone** — fastest and smallest (2.67 ms, 195 M params, ONNX), but its targets are full 36-float poses and its hand/foot pins are SOMA-only, so the pose has to be invented before it can interpolate. That is the workaround this design removes.
- **Kimodo alone** — better motion quality per clip and the best authoring tool, but offline (100 denoise steps) and it cannot serve a live reach.
- **ARDY** — same constraint vocabulary as Kimodo, streaming, and measured here at 0.94 GB and 45.5 mm. Chosen.
- **IK or a QP as the executor** — measured and rejected: `research/sonic-gate-results.md`, `research/wbc-classical-proposal.md`.
- **HumanML3D** — a dataset, not a controller, and its licence and skeleton (SMPL-H, 20 fps) make it a training project rather than a runtime.

## 16. Open defects and the architecture critique

Three independent critiques (robotics/control, ML, product) were run against this design on 2026-11-11; the findings,
the alternatives they ranked, and the correcting measurements are in `research/reach-architecture-critique.md`. The
parts that change this document:

1. **The graded point is wrong by 0.18 m** — §6.2's warning. Gate 5's 66 mm is wrist-to-wrist; the game-visible error
   has never been measured, and the clasp gate cannot fire. Fix first.
2. **The 80 mm threshold is uncalibrated**, and at the declared 1.43× player scale 66 mm of body-space error is ~94 mm
   in the room. Measure the game's actual grab tolerance before treating 80 mm as a requirement.
3. **The baseline comparison is not like-for-like** (§3 above quotes the uncorrected cruise; its post-correction case
   is 34 mm / 2.62 s / 4 reversals), and **path shape — the property this design exists to fix — is not measured by
   any gate.**
4. **Proposed decomposition change: generate → track → servo → clasp.** The generator owns shape; the last 8 cm goes
   to the existing VR servo, which measures 16–37 mm against the generator's 45–80 mm. §6.4's escalation is not part
   of it.
5. **Authoring offline is a deployment choice, not a capability loss**, because v1's clip is a pure function of
   (root, heading, relative target). It also removes the sampling tail, and it is the only route left to §3's look
   goal — which is measured dead as a live constraint (§8.5).

None of this invalidates the constraint vocabulary or the tracked-playback work; it changes what the demo depends on
and what gets measured next.
