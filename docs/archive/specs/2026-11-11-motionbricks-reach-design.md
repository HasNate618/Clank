# MotionBricks reach — Design

> **Archived 2026-11-13.** Superseded by `docs/superpowers/specs/2026-11-11-ardy-reach-design.md`, which argues the same problem with a generated motion prior instead.
> Current behaviour: [`README.md`](../../../README.md) · agent rules: [`AGENTS.md`](../../../AGENTS.md).

- Date: 2026-11-11
- Status: **superseded** by `2026-11-11-ardy-reach-design.md`, which argues the same problem with a
  generator that accepts end-effector constraints. Kept for its measurements (the MotionBricks comparison,
  the fallback), not as a plan of record.
- Parent: `research/humanoid-command-space.md` (BodyCommand contract), `research/upper-body-override.md`, `research/motion.md`, `research/wbc-classical-proposal.md`
- Implements against: `spike/wbc/sonic_executor.py`, `spike/wbc/sonic_runner.py`, `spike/wbc_game/reach_vr.py`, `clank/workspace.py`, `clank/player_scale.py`
- Does not implement: brain/planner changes, locate, glow, neck, HumanML3D training, SONIC or MotionBricks retraining

Claim tags: `[MEASURED]` local file or named run; `[CITED]` external source; `[OPERATOR]` the person who owns the game; `[ASSUMPTION]` inference, untested. Builder calls are marked **(call)**.

## 1. Problem

Telling the body to grab something in front of it produces a dance: a long, wacky arm path, and a torso that folds so the hand can reach. The head does not look at the object. It looks unnatural, and the camera is often pointed somewhere other than the hand.

Two diseases, already separated:

1. **The path.** `reach_vr` streams a Cartesian cruise on SONIC's VR 3-point channel. That channel is 9 positions + 12 quaternion floats, no velocity, no horizon `[CITED: NVIDIA observation_config.md:225-246, recorded in research/upper-body-override.md]`. NVIDIA's own numbers for it are 121.9 ms mean latency and 6 cm mean wrist error. Ours are 0.77–1.05 s settle and +75 % vertical overshoot, 3.1–7.5 s and 13–27 direction changes per reach `[MEASURED: research/motion.md]`. A hand-written reference is why it looks hand-written.
2. **The posture.** The VR policy is a whole-body optimiser over a position setpoint. It will move the root and twist the waist to buy the arm millimetres. `WRIST_BOX` (`clank/workspace.py`, x 0.079–0.60, y −0.31–0.42, z −0.23–0.45 m) is a steering limit on that channel, not the arm's reach `[MEASURED: AGENTS.md; research/motion.md]`. Unclamped, the arm holds y = −0.389. The body still rearranges itself.

A third, already-falsified, repair: stuffing IK joints into SONIC. C3j hard-overrode ctrl and made hold 1.14 → 14.40 cm `[MEASURED: research/sonic-gate-results.md]`. The 17-joint reference override is unread in VR mode, and on the motion-tracking path the policy tracked the substituted joints at 0.125 rad mean error and delivered ~33 mm of a 235 mm command `[MEASURED: research/upper-body-override.md]`. **IK is not an executor for this body.**

## 2. Claim

A grab is three jobs, not one:

| Job | Who | What |
| --- | --- | --- |
| **Still frame** | classical IK / placement | A full G1 pose whose hand is on the object and whose chest faces it |
| **Path** | MotionBricks G1 | Kinematic inbetween from the last four achieved poses to that still frame |
| **Physics** | SONIC mode 0 | Track the generated clip; legs, balance, and the 0.9 s look-ahead stay on the policy |

NVIDIA's G1 deployment is this pairing: MotionBricks generates a kinematic reference, SONIC tracks it `[CITED: arXiv:2604.24833 §7.4, "we use the tracking controller from Luo et al. (2025)"]`. We already have the tracker. We have been using its other input (VR 3-point) for reaches.

HumanML3D is a dataset repo, not a controller. MotionBricks already trained on a much larger RP1 corpus retargeted to this skeleton. It is not used.

## 3. Goals, non-goals, anti-goals

### Goals

1. A `reach` / `grab` toward a 3D point produces motion that looks like a person reaching: feet plant when the object is in front, a step or pivot when it is not, chest toward the object, hand on the point.
2. The camera looks at the grab. G1 has no neck `[MEASURED: research/body-alternatives.md]`; "look" is waist yaw when planted, root yaw when stepping.
3. Live generation. One MotionBricks call per reach, 0.6–1.9 s of motion at 30 Hz, chained only if the approach is longer than one generation.
4. Grab still closes in the same process as the reach. A 2 s gap between reach and close drifted the hand 0.09 m `[MEASURED: AGENTS.md, the grab]`.
5. `hold`, `look`, `rest` keep working on the current VR path. This spec replaces the **reach/grab executor**, not the BodyCommand contract.
6. Honest outcomes: `done`, `saturated`, `timeout`. The adapter reports the achieved hand residual in metres.

### Non-goals (this spec)

- Training or fine-tuning MotionBricks, SONIC, or a text-to-motion model.
- HumanML3D, AMASS, SMPL-H, or any new motion corpus.
- A neck, a second body, or switching off the G1.
- Replacing `BodyCommand` / `clank.intent`. The generator sits under the existing reach/grab skill.
- Pixel locate, glow, or brain/planner changes.
- Using MotionBricks `IKPin` on G1. The runtime documents those pins as SOMA-only `[CITED: motionbricks/docs/runtime_api.md, TargetPoseInferenceParams.ik_pin_mask]`.
- Feeding mink/QP joint targets to SONIC as `upper_body` or ctrl. That experiment is closed.
- Continuous 10 Hz replanning during a grab. Locomotion-style replanning is out of v1; a grab generates once (twice if the first residual is large — §8.3).

### Anti-goals

- A reach that plants the hand by folding the torso while the head looks away. That is today's failure, not a fallback.
- A reach that walks when the object is inside the planted envelope. In-front grabs stay planted.
- Treating generated kinematic poses as achieved. The relay still sends `mjData` after `mj_step`.
- Running the TensorRT plan builder next to a live session. The 4060 has ~2.6 GB free while the game holds ~5.2 GB `[MEASURED: 2026-11-11, nvidia-smi during this design]`.

## 4. Proven basis

- SONIC encoder/decoder ONNX at 50 Hz, PD at 500 Hz, in `spike/wbc/sonic_runner.py`. Mode 0 writes 10×29 joint positions and velocities from the motion clip (`motion_frames`, step 5 → 0.9 s look-ahead at 50 Hz). Mode 1 skips that 29-joint reference and writes VR 3-point instead `[MEASURED: sonic_runner.build_encoder_obs]`.
- Freeze-hold: standing clip `squat_001__A359` frame 0, anchor index frozen, lookahead must not contain walking. Hold RMS 1.3×10⁻⁴ m/s `[MEASURED: research/c3m-results.md]`.
- Mink weighted IK on the floating-base G1, 50 Hz: planted feet cost 5000, pelvis 5000, CoM 100, head ori 100 (500 for look), wrists 500, posture 0.1 toward stand. Hold 0.22 mm base excursion / 20 s. Look ±35° waist at 0.63° yaw error, 3.3 mm base excursion. Single-arm reach 2.2–12.9 mm. Punch 9.5 mm. Cannot walk; single-support tipped in 1.3–1.6 s `[MEASURED: research/wbc-classical-proposal.md]`.
- 7-DoF DLS arm IK (`spike/wbc/arm_ik.py`) is sub-millimetre inside the pose, honest residual outside. Wrist orientation left free. Jacobian must run `mj_comPos` after `mj_kinematics` or it is identically zero `[MEASURED: arm_ik.py comments]`.
- MotionBricks G1 ONNX: 4 context + up to 4 target poses, each 36 floats, 30 Hz, 6–16 tokens, 18–58 valid frames after dropping 3 boundary frames each end. ~195 M params. FP32 2.67 ms / mixed FP16 1.63 ms at N=1 on a 4090 `[CITED: NVlabs/motionbricks docs/unitree_g1_model.md, docs/onnx_inference.md]`. Trained on 313,526 RP1 clips retargeted to a 34-joint G1 (29 hinges + free pelvis + 4 end-effector joints in the skeleton, 29 angles in the public interface).
- In-box cruise target `(0.35, −0.28, 0.15)` m in the root frame landed at 16.4 mm `[MEASURED: research/motion.md]`. That is the preferred right-hand offset for a stepped pose.
- Reported hand point is `wrist_yaw_link` plus `VR_OFFSETS` lever, then expressed in the pelvis frame `[MEASURED: sonic_runner.VR_OFFSETS, arm_ik.vr_point]`. Right lever `(0.18, 0.025, 0.0)` m.
- `clank/` is stdlib-only. NumPy / MuJoCo / ONNX / mink live in `spike/e3/.venv`.

## 5. Architecture

```text
pixel / named point
        │
        v
PlayerScale.pixel_point     robot-space XYZ   (game space ÷ declared scale)
        │
        v
end_pose()                  36-float G1 still frame
  planted mink  or  stepped placement + mink
        │
        v
MotionBricks G1             context: last 4 achieved poses @ 30 Hz
                            target:  1 still frame (num_poses_to_use=1)
                            output:  18–58 frames @ 30 Hz
        │
        v
resample 30 → 50 Hz         policy-order jp/jv + pelvis bp/bq
        │
        v
SONIC mode 0                50 Hz, 10-frame look-ahead into that clip
        │
        v
mj_step 500 Hz              achieved qpos → body_state.json → pose_relay
        │
        +-- fingers         14 motors, unchanged clasp, not in the 29
```

The 50 Hz body loop never waits on generation. Mink and MotionBricks run in the **adapter process** (`reach_mb.py`, same shape as today's `reach_vr.py`). The executor only plays a deposited clip. CUDA stays out of the process that ticks SONIC and writes `body_state.json`.

Deleting the generator must leave `hold` / `look` / `rest` / manual mode intact. A missing ONNX or a failed CUDA context is `FLAG_UNSUPPORTED` on reach, not a session tear-down.

Do **not** put the object XYZ or the clip on `BodyCommand`. `frame="world"` was removed because the executor accepted it and ignored it `[MEASURED: clank/body.py FRAMES]`. Clips are joint-space; the contract is task-space. §9 is the mailbox.

## 6. End pose

The G1 ONNX cannot take "hand here". Masks are per slot (4 context + 4 target), not per joint `[CITED: unitree_g1_n1.json; motionbricks/model/kernels.py write_motionbricks_conditions_kernel]`. The still frame must already have the hand and the facing.

Solve on **`scene_sonic_floor.xml`** (the SONIC model with hands), not the menagerie `scene.xml` mink was first measured on. The reported hand includes the VR lever; aiming the wrong body is a centimetre-class bias `[MEASURED: arm_ik.py]`.

### 6.1 Inputs

- Latest **36-float achieved pose** from the executor ring (§9), not `BodyState.q`. `BodyState.q` is policy-order minus `DEFAULT` `[MEASURED: sonic_executor.py writes r._q_policy()]` and cannot be unpacked into MotionBricks layout.
- The adapter seeds a **scratch** `MjModel` of `scene_sonic_floor.xml` from that 36-float (hands at current clasp or zero). It does not share `mjData` with the live sim.
- Object point in **robot world metres**. Obtained via `PlayerScale.pixel_point` (or the equivalent unproject) so a declared scale cannot be applied to the camera and forgotten on the hand `[MEASURED: clank/player_scale.py — that omission is the 1.43× miss]`. The object never crosses `control.sock`.
- Side: `left` | `right`.
- Stand keyframe of this XML (root z ≈ 0.79 m `[MEASURED: research/body-alternatives.md]`).

### 6.2 Envelope test

Let `p_root` be the object in the current pelvis frame (+X forward, +Y robot-left, +Z up). `yaw_err = atan2(p_root.y, p_root.x)`.

Treat as **planted** only if both:

1. `|yaw_err| ≤ 35°` — the measured waist-look range `[MEASURED: wbc-classical look, 0.63° error at ±35°]`. Wider than this, pivot; do not crane the torso to look.
2. A planted mink solve from the current qpos returns hand residual **< 5 cm**.

Otherwise **step / pivot**. **(call)** 5 cm is the planted residual gate, not the final tracking tolerance. SONIC will not hit 5 cm just because the keyframe did.

A grab 90° to the side pivots even if the arm could crane to it. That is the anti-dance rule, and it is why looking is a first-class task.

### 6.3 Planted solve

Mink (or the same weighted stack in `spike/wbc_classical/wbc_spike.py`) as a **one-shot** keyframe, not a 50 Hz loop.

| Task | Cost | Target |
| --- | ---: | --- |
| Left / right ankle-roll | 5000 pos / 5000 ori | current planted frames, not moved |
| Pelvis | 5000 pos / 5000 ori | current root; do not walk |
| CoM | 100 | current, small lean allowed |
| Head ori | 500 | +X of `head_link` / torso toward the object. Position cost stays low (10). |
| Active wrist pos | 500 | object point minus nothing — aim the **reported** hand (`wrist_yaw_link` + `VR_OFFSETS[side]`) |
| Active wrist ori | 100 | optional; v1 **(call)** leaves roll free, points the palm roughly at the object if cheap |
| Other wrist | 500 | hold achieved |
| Posture | 0.1 | stand keyframe, so the torso is not a crane |

Joint velocity limit 2.0 rad/s is irrelevant for a one-shot; still clamp each hinge to `jnt_range ± 0.02` as `arm_ik.solve` does.

If this residual is ≥ 5 cm, fall through to §6.4 even if yaw was small.

### 6.4 Stepped / pivot pose

Do not ask mink to walk. Place a standing pose, then IK the arms on it. MotionBricks generates the step.

Preferred hand in the **new** root frame **(call, from the in-box measurement)**:

```text
HAND_PREF_RIGHT = (0.35, -0.28, 0.15)   # m, +X forward, +Y left
HAND_PREF_LEFT  = (0.35, +0.28, 0.15)
```

Let `pref = HAND_PREF[side]`, `obj` the object in world, `root_z` the stand height.

1. Yaw so the preferred hand direction points at the object:

   `yaw = atan2(obj.y − root_y_guess, obj.x − root_x_guess) − atan2(pref.y, pref.x)`

   The current root xy is a sufficient first guess; one iteration is enough.

2. `root_xy = obj.xy − (Rz(yaw) @ pref).xy`. `root_z` = stand height.

3. Seed qpos from the stand keyframe: root xyz + yaw quaternion, legs at stand, waist 0, other arm at stand.

4. Mink on that seed: feet at the new ankles (they are in the air in the keyframe until SONIC tracks), head toward object, active hand on object, posture 0.1.

5. If hand residual is still ≥ 5 cm after placement, emit `saturated` and do not generate. The object is not a legal G1 grasp from a standing pose (too high, in the floor, inside the torso).

Cap the step. Job Simulator's room is small; a 3 m root translation is a bad unprojection, not a walk **(call: `MAX_STEP_M = 1.5`)**. Larger than that is `saturated` / `timeout`, not a generation.

### 6.5 Output

A 36-float pose in MotionBricks layout, world/runtime space (G1 is already Z-up, +X forward; `model_to_runtime` is identity `[CITED: motionbricks/docs/runtime_api.md]`):

```text
[root_x, root_y, root_z, qx, qy, qz, qw, 29 hinge angles]
```

Hinges in **hardware / `SonicRunner.qadr` order**, which is also `unitree_g1_n1.json joint_names`:

```text
left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll,
right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll,
waist_yaw, waist_roll, waist_pitch,
left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow,
left_wrist_roll, left_wrist_pitch, left_wrist_yaw,
right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow,
right_wrist_roll, right_wrist_pitch, right_wrist_yaw
```

Fingers are **not** in this vector. They stay on the clasp channel.

## 7. MotionBricks generation

### 7.1 Call shape

One generation per reach:

- **Context:** four poses at 30 Hz, oldest first, ending at the latest achieved pose. First call: repeat the current pose four times (`MotionRecorder.fill`). Later chain: the last four **played achieved** poses, not the kinematic preview. If those two diverge, the next generation starts from a pose the body never occupied `[CITED: runtime_api.md "Record what you show"]`.
- **Target:** the still frame in slot 0, `num_poses_to_use=1`. Extra target poses add velocity at the end `[CITED: runtime_api.md]`. A grab should stop.
- **Masks:** engine default. Context slots + first `num_target_poses` targets marked; local-root derivatives unmarked on the last context and last target `[CITED: write_motionbricks_conditions_kernel]`.
- **`allowed_token_counts`:** planted 6–10 (short reach); stepped 10–16 (walk needs duration). **(call)**
- **`correct_generated_diff`:** True. The engine spreads endpoint deltas so the clip starts at the last context pose and ends at the target. Root correction spans the whole clip; joint-angle correction fades over `JOINT_ANGLE_FADE_SECONDS` with a quadratic ease `[CITED: unitree_g1/engine.py _post_process_generated_poses]`. Do not skip this and ship raw `pred_qpos`.
- **Heading alignment:** default `(1, 0)` = face +X in G1 model space during inference; the engine restores world placement afterward.
- **TF32:** off. Enabling it moves G1 joint angles by several degrees for ~2 % speed at N=1, and the error grows across chained generations `[CITED: unitree_g1_model.md Inference]`.

Raw output is 24–64 frames; drop 3 boundary frames each end (`MODEL_BOUNDARY_FRAMES = 3`). Valid length `4 * pred_num_tokens − 6`, i.e. 18–58 frames, 0.6–1.9 s `[CITED: motionbricks/model/contract.py valid_frame_count]`.

### 7.2 Engine vs raw ONNX

**(call)** Use `MotionBricksUnitreeG1Engine` when CUDA + TensorRT are available. It owns alignment, decode, root smoothing, boundary trim, and endpoint correction.

The live session currently runs SONIC on **CPU** ONNX Runtime (`CPUExecutionProvider`, 1 thread) `[MEASURED: sonic_runner.py]`. MotionBricks is the new GPU tenant.

Fallback order:

1. TensorRT FP16, N=1, plan cached under `~/.local/state/Clank/motionbricks-trt/`. Build the plan with the **game down**. The 134-byte files in a shallow clone are LFS pointers; `git lfs pull` (or the Hugging Face G1 ONNX) is required.
2. ONNX Runtime CUDA, same tensors, workspace capped so it fits in the ~2.6 GB leftover. The caller must still apply heading alignment, trim, and endpoint correction — copy those kernels, do not invent a third version.
3. ONNX Runtime CPU. A 10 Hz burst of one generation can spend tens of milliseconds. A grab is not 10 Hz.

Warp / Newton / the Scene Preview apps are not runtime dependencies of Clank. The spike may vendor or submodule `NVlabs/motionbricks` for the engine and the ONNX. Sample BVH/CSV clips are NVIDIA Sample Data Evaluation License; we do not ship them into the session. Model weights: NVIDIA Open Model Agreement. Engine code: Apache 2.0.

### 7.3 Packing hazards

These are load-bearing. A silent swap looks like a new dance.

| Item | MotionBricks | MuJoCo / SONIC |
| --- | --- | --- |
| Root quaternion | **xyzw** | **wxyz** in `qpos[3:7]` |
| 29 hinges | hardware / `qadr` order | same names; SONIC **clip `jp` is policy/CSV order** (interleaved L/R). `CSV2MJ[csv_i]` is the hardware index `[MEASURED: sonic_runner.py:75-77]` |
| Hands | absent | interleaved in `qpos` after each arm (`g1_29dof_with_hand.xml`: left arm 7, left hand 7, right arm 7, right hand 7) |
| Rate | 30 Hz | SONIC 50 Hz, physics 500 Hz |
| Root units | metres | metres |
| Angles | radians | radians |

`jp` for SONIC from a MotionBricks frame:

```text
jp[csv_i] = mb_joints[CSV2MJ[csv_i]]
```

Inverse, hardware from a clip row: `mb_joints[mj_i] = jp[index of mj_i in CSV2MJ]`.

Pelvis for mode 0 `anchor_6d`: `bp[f, 0] = root xyz`, `bq[f, 0] = xyzw→wxyz(root quat)`. Other of the 14 bodies may stay zero; mode 0 does not read VR bodies. `jv` is a 50 Hz finite difference of `jp`, last frame copied.

Do not write MotionBricks angles into `qpos` by slicing `qpos[7:36]`. Hands live in the middle.

## 8. SONIC tracking

### 8.1 Mode switch

`SonicExecutor` today is hard-coded `SonicRunner(mode=1)` `[MEASURED: sonic_executor.py:122]`. During generated playback it must run **mode 0** so the 29-joint look-ahead is live. The encoder already branches on `obs[0:4] = [mode, 0, 0, 0]`.

On start of a generated clip:

1. Build a motion dict from the resampled clip (`jp`, `jv`, `bp`, `bq`, `n`).
2. Set `runner.mode = 0`. Do not call `reset()` (that teleports). Keep current `mjData`.
3. Play the clip by advancing the clip index at 50 Hz so `motion_frames(frame, num=10, step=5)` always looks forward into yet-unplayed frames, same as a walking clip.
4. When the clip index reaches `n − 1`, **flatten the look-ahead** (repeat the last frame for all 10 slots) and freeze the anchor index. This is C3m's H1f, the hold that does not walk `[MEASURED: research/c3m-results.md]`. Then switch back to mode 1 freeze-hold, claiming the achieved head and wrists so TTL expiry does not snap to the standing clip `[MEASURED: sonic_executor.py hold/rest split]`.

`AUTH_ROOT_VEL` stays unsupported as a BodyCommand field. Walking in this spec is a side effect of a stepped still frame, not a velocity command.

### 8.2 Fingers and grab

MotionBricks does not generate Dex3 angles. Keep `GRASP_Q` / `HAND_SLEW_RAD = 0.25` `[MEASURED: sonic_executor.py]`.

`--grab` (or skill `grab` that includes the reach) starts the clasp when the **achieved** reported hand is within **8 cm** of the object, without waiting for the clip to end. **(call)** 8 cm is inside the historical 9.5 mm–14 cm tracking band with slack for the first unmeasured SONIC-on-generated-clip run.

The adapter watches `BodyState.left_hand` / `right_hand` (those *are* the reported VR points) and sets `AUTH_CLOSURE` on `control.sock`. The executor already clasps in `_clasp` on that tick — same process as the motors, so this is not the 2 s two-process gap. Closing only after the clip ends, from a second Python, is the 0.09 m drift bug.

Relay grab mode stays `--grab-mode both` (squeeze + `hand_curl`). Squeeze alone leaves the glove open `[MEASURED: AGENTS.md, the fingers close]`.

### 8.3 Chaining and correction

Two different loops. Do not collapse them into "max two generations".

**Step chain.** After a stepped clip, re-run the envelope test on the latest 36-float pose. If the object is still outside the planted envelope and the root has not yet travelled `MAX_STEP_M`, build a **new** stepped still frame from here and generate again. Stop this loop on planted, on `MAX_STEP_GEN = 4` generations, or on **8 s** wall time. A 1.5 m approach at walk speed will not fit in one 0.6–1.9 s clip; treating that undershoot as `saturated` would strand the player mid-room. Context for each call is the last four **achieved** poses.

**Planted correction.** Once the object is inside the planted envelope (or the step chain gave up inside it), if achieved residual is **> 5 cm**, build one planted still frame and generate **once**. No further correction. No VR cruise, no `target += 0.6 * error` `[MEASURED: that integrator walked y to −0.92 m, research/motion.md]`.

If residual grows after that planted generation, stop and report `saturated` with best-so-far (the end of the previous clip, if closer).

### 8.4 Head pin

Do not re-send a head pose at 20 Hz during the generated reach. A constant pin hammered into the VR path twisted the torso and moved the root frame the target lived in `[MEASURED: research/motion.md, third pass]`. Mode 0 is tracking the clip, which already contains the look. After hold, freeze-hold claims the achieved head.

## 9. Adapter / process split

**(call)** Same topology as `reach_vr.py`: a client process generates, the body process plays. Mink + TensorRT do not share the 50 Hz SONIC process.

| Unit | Lives in | Notes |
| --- | --- | --- |
| End-pose IK | adapter (`spike/wbc/`, `reach_mb.py`) | scratch MuJoCo + mink; venv |
| MotionBricks wrapper | adapter | engine or ORT; venv; may block 3–50 ms |
| 36-float pose ring | executor writes, adapter reads | see mailbox below |
| Clip playback / mode switch | `spike/wbc/sonic_executor.py` | existing process; no CUDA |
| Clasp | executor `_clasp`, triggered by adapter `AUTH_CLOSURE` | already in-process with the motors |
| `PlayerScale`, `WRIST_BOX`, intent, `BodyCommand` | `clank/` | stdlib; no generator imports |
| Pose relay | `clank/pose_relay.py` | unchanged: achieved head/wrists. A stepped reach **moves the player in Job Simulator**. That is intended `[OPERATOR: 2026-11-11, out-of-reach → step/pivot]`. |

`WRIST_BOX` is **not** the planted envelope for this path. It remains the clamp for any leftover VR servo. The planted envelope is the mink residual in §6.2.

`STEP_GUARD_M = 0.15` in `reach_vr.py` treated a step as a fault. The new adapter must not inherit that guard for stepped reaches.

### 9.1 Mailbox (session dir)

All under `~/.local/state/Clank/sessions/body/` (same dir as `control.sock` / `body_state.json`):

| File | Writer | Meaning |
| --- | --- | --- |
| `pose36.npy` | executor, every tick | Ring of the last 32 achieved 36-float poses at 50 Hz, MotionBricks layout, plus `seq` / `t_mono_ns`. This is the context source. |
| `mb_clip.npz` | adapter | Resampled 50 Hz SONIC motion dict (`jp`, `jv`, `bp`, `bq`) ready to load. |
| `mb_play.json` | adapter latest-wins | `{ "seq": n, "path": "mb_clip.npz" }`. Executor loads on seq change, plays mode 0, does not `reset()`. |
| `mb_status.json` | executor | `{ "seq_ack": n, "status": "playing"\|"done"\|"failed", "frame": i, "n": N }`. |
| `control.sock` | adapter | Unchanged `BodyCommand`: freeze-hold, `AUTH_CLOSURE` / `AUTH_TRIGGER`. No clip, no world XYZ. |

Depth unproject stays in the adapter, outside the body loop.

## 10. What success looks like

Same object, same reset, compared to the cruise in `research/motion.md`.

| | Today (cruise) | This spec |
| --- | --- | --- |
| Path | 13–27 direction changes, 3–7 s | one planted generation, or a step chain of 0.6–1.9 s clips |
| In-front grab | torso folds, head off-axis | feet planted, chest toward object, residual ≤ 8 cm on the live body |
| Off-axis / far | crane or `WRIST_BOX` clamp | pivot/step, HMD turns toward the object |
| Look | not requested | waist or root yaw in the still frame |
| Grab close | `--grab` in-process | unchanged, gated on 8 cm |

The first number that can kill the design: SONIC mode 0 tracking error on a MotionBricks clip. NVIDIA claims the pairing; we have not measured it on this deploy. If the tracker delivers a fraction of the clip (the 33 mm / 235 mm joint-override failure mode), stop and do not paper over it with a VR cruise.

## 11. Measurement gates (in order)

A later plan can split these into tasks. The order is the design.

1. **Packing test, game off.** Round-trip `mjData.qpos` → 36-float → `jp` → `qpos` on the 29 hinges. Max hinge error < 1e-6. Quaternion xyzw/wxyz covered. Hands untouched.
2. **IK still frames, game off.** Planted: object 0.35 m forward of rest → residual < 15 mm, `|Δroot_xy| < 1 cm`, head yaw toward object within 2°. Stepped: object 1.0 m forward → root translated, hand residual < 15 mm on the keyframe, yaw faces it.
3. **Generation, game off.** Context = stand × 4, target = those still frames. Output starts at stand (endpoint correction), ends at the target, duration 0.6–1.9 s, no NaNs. Visual: planted is an arm reach, not a squat; stepped includes a step, not a moonwalk.
4. **SONIC track, game off.** Mode 0 on that clip, 50 Hz. Report achieved hand residual vs the still frame, root path vs clip root, falls. Gate: planted residual **< 8 cm**, no fall, look error **< 15°**. If planted residual > 15 cm, the pairing is not usable; do not proceed to the session.
5. **Live grab.** Reset body, operator pixel or locate, one in-front card grab, one off-axis or out-of-reach grab. Clasp in-process. Record `body_pose.csv` curl columns, residual, whether the HMD looked at the object.

VRAM: plan-build with `nvidia-smi` before/after, game down. Runtime with the session up; if OOM, drop to fallback 2/3 and record which.

## 12. Files (when built)

Suggested, not required to exist before the plan:

- `spike/wbc/end_pose.py` — envelope test, planted mink, stepped placement
- `spike/wbc/motionbricks_gen.py` — engine/ORT wrapper, 36-float I/O, trim/correct
- `spike/wbc/clip_from_gen.py` — 30 Hz 36-float → SONIC motion dict at 50 Hz
- `spike/wbc/sonic_executor.py` — mode 0 playback + flatten-hold (edit)
- `spike/wbc_game/reach_mb.py` — live adapter; `reach_vr.py` stays until §11.5
- `tests/test_end_pose.py`, `tests/test_mb_packing.py` — packing and envelope, no GPU

`clank/` gains no generator import.

## 13. Builder calls (defaults, rejectable)

Stated so they are not silent:

1. Planted yaw gate **35°**, planted residual gate **5 cm**, tracking success **8 cm**, clasp start **8 cm**, max step **1.5 m**.
2. Step chain **≤ 4** generations or **8 s**; then **one** planted correction. Not a global cap of two.
3. `HAND_PREF_RIGHT = (0.35, −0.28, 0.15)` m from the in-box cruise, not from `WRIST_BOX` centre.
4. Wrist orientation in the still frame is cheap pointing, not a 6-DoF controller pose. Roll free.
5. Engine first, ORT CUDA, ORT CPU. Plan built game-down. TF32 off. FP16 for live.
6. `hold` / `look` / `rest` stay on VR mode 1. Only reach/grab use this path.
7. Context is the executor's 36-float ring (achieved). Target `num_poses_to_use=1`. `BodyState.q` is not a pose source.
8. Adapter generates; executor plays. Clip mailbox in §9.1, not a `BodyCommand` field, not CUDA in the 50 Hz process.
9. HumanML3D out. MotionBricks `IKPin` out. Joint override out.
10. `reach_vr.py` cruise remains in tree as the A/B until §11.5; the live grab path switches only after that gate.

## 14. Left to the builder

- Mink vs a thin DLS-on-17 for the planted arm after the feet/look are applied by a cheaper waist yaw. Mink is the measured look+reach stack; a simpler solver is allowed if §11.2 still passes.
- Whether to import the MotionBricks Python package or copy the G1 encode/decode/post kernels. Behaviour must match §7.1.
- Exact cache path and download of the G1 ONNX (Hugging Face `nvidia/MotionBricks` or git LFS).
- Interpolation 30→50 Hz (linear on hinges, slerp on root quat).
- Whether chained generations reuse the same engine context buffers.
- File format of `pose36.npy` (npy vs jsonl vs mmap); it must be latest-wins and not block the 50 Hz writer.

## 15. Why not the alternatives

- **Polish the cruise.** Fixes some reversals `[MEASURED: research/motion.md]`; does not stop the torso folding, and cannot look.
- **Mink as the live executor.** Precise while standing `[MEASURED]`; cannot walk; cannot be composed onto SONIC `[MEASURED: C3j]`.
- **Bake clips offline.** Cheaper GPU story; Job Simulator objects are mostly fixed. Rejected: operator asked for live generation, and out-of-reach steps need a still frame computed from the actual object, not a library.
- **Let MotionBricks invent the still frame from a hand pin.** G1 has no per-joint mask and no IKPin. The generator would end at a full-body pose we did not choose, which is the current disease with a nicer interpolator.
