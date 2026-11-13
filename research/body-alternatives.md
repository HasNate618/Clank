# Simulated player body: alternatives to the Unitree G1

**Date:** 2026-10-23  
**Scope:** research only. No code, spikes, Monado, or training were run for this note. Measurements are from the MuJoCo models named below.  
**Question:** the project currently simulates a Unitree G1 as the player's body. Is there a better humanoid body — specifically RoboParty's RP1 — for an honest VR player body, and should Clank switch?

**Answer up front.** Keep the G1 29-DoF for v1. Do **not** switch to RP1. If a neck later becomes a measured blocker, the evidence-backed second body is Apptronik Apollo, not RP1, and it should sit behind a body-profile abstraction rather than fork the executor.

**Tag key:** `[C#]` cited web source (URL in §10), `[M#]` measured in this pass, `[P#]` project document, `[A#]` assumption or judgment.

---

## 1. What "better body" means for this project

The player body is not a locomotion demo. The game sees an HMD pose, two controller poses, and a grip value. Legs never appear in the in-game framing; the hard honesty problem is the **head, torso, and wrists**, plus not falling over while the arms move. `[P1 §1, §3]` The ranking criteria therefore are:

1. **A real head frame, and ideally a neck.** G1 has a rigid head: HMD orientation is waist/torso orientation, with the residual documented. A 2–3 DoF neck is a fidelity upgrade, not a v1 requirement. `[P1 §3, §7]`
2. **A 6-DoF wrist per arm.** A controller pose is 6-DoF. A 7-DoF arm (3 shoulder + elbow + 3 wrist) can realize it; a 4–5 DoF arm cannot without dropping orientation and calling the result honest. `[P3 §"Recommended architecture"; P1 §7]`
3. **Standing without a new training project.** Position actuators plus a stand keyframe let v1 stand on PD; a torque-only model with no keyframe is new controls work before C2 even starts. `[P2; P1 §7]`
4. **A qualified policy, or a cheap path to one.** G1's Playground ONNX exists and its zero-command drift is already measured (0.475 m in 5 s). A body without a policy is a stand-PD body until someone trains one. `[P1 §3; P5]`
5. **Adult scale only if the game actually needs it.** G1's HMD is 1.221 m, a child's eye height. That is a measured property, not a bug; C0 has not yet shown that Superhot is unplayable at that height. `[P1 §10; P2]`
6. **Asset and license quality.** A maintained MJCF with permissive terms, correct inertials, actuators, and keyframes is worth more than a nominally better kinematic sheet. `[C2, C8, C11, C16]`

Puppet mode — sending the requested HMD/controller poses without stepping MuJoCo — is excluded. Every candidate is judged on poses achieved in `mjData` and reported as achieved. `[P1 §3, §7]`

Judgment: criteria 1–4 are the discriminator. Criterion 5 is often raised as a reason to switch, but it does not favor RP1 (see §3).

---

## 2. The incumbent, measured

The G1 is a 1.32 m, ~35 kg, 23–43 DoF robot; the simulated model is the 29-actuated-joint variant with three waist joints, three wrist joints per arm, and no fingers or neck. `[C1, C2]`

E3 measured the actual simulated body at 500 Hz in Menagerie:

| Quantity | Measured |
|---|---|
| Actuated joints | 29 (`nq=36` with free base) `[M1]` |
| Mass | 33.34 kg in the model `[M1]` |
| Stand base height | 0.7905–0.7916 m `[P2]` |
| Horizontal drift, 20 s stand | ≤3.08 cm `[P2]` |
| Head frame | 1.221 m; head mesh is rigid on `torso_link`, no head/neck joint `[P1 §3; P2; M1]` |
| Left wrist frame | 0.721 m; shoulder at 1.082 m; shoulder→wrist 0.381 m `[M1]` |
| Playground ONNX at zero command | 0.475 m drift in 5 s `[P5]` |

What that buys: honest wrist orientation (3 wrist DoF), honest head pose **as torso pose** (the head really moves with the waist, so it is not puppet mode), and a measured stand that meets the 20 s think-time budget. What it does not buy: independent head look, a neck, or visible finger closure. `[P1 §3, §7; P2]`

The 1.221 m HMD is the head mesh's position, not the robot's 1.321 m visual top. Both are child-scale. `[P1 §10; M1]`

---

## 3. The named candidate: RoboParty RP1

### 3.1 It is a real robot, and the stack is unusually open

RP1 is a 1.35 m humanoid sold as RP1 (24 active DoF, 0 head DoF, 5-DoF arms, ~39 kg) and RP1 Pro (30 active DoF, 2 head DoF, 7-DoF arms, ~41 kg). Both have 2 waist DoF, 6 leg DoF, 550 mm single-arm length, and a claimed 3 h battery. `[C3, C4]`

RoboParty publishes hardware CAD/BOM/PCB, firmware, a Python/ROS 2 deployment stack, URDF/MJCF models, and Isaac Lab training code. The docs explicitly describe MuJoCo Sim2Sim checks and a 50 Hz policy / 250 Hz host loop. `[C4, C5]`

### 3.2 What the open simulation actually contains

I cloned `RP1_description` at `c2956c58` and loaded the MJCFs with MuJoCo 3.17.0. The numbers below are measured, not taken from the spec sheet.

| Variant | Active joints | Mass | Arms | Wrist | Head/neck | Actuators | Keyframe |
|---|---:|---:|---|---|---|---|---|
| `robot_24dof.xml` | 24 | 39.11 kg | 5 DoF | 1 (`left_wrist_joint`) | no neck; rigid head mesh on torso | torque (`<motor>`) only | none |
| `robot.xml` (28-axis) | 28 | 40.73 kg | 7 DoF | 3 (roll/yaw/pitch) | no neck; rigid head mesh on torso | torque (`<motor>`) only | none |

Both models top out at **1.326 m** to the visual head mesh, essentially the same as the G1's 1.321 m. The 28-axis model is the Pro body minus the two head joints; the 24-axis model is the standard body, which has no head joints to model. `[M2]`

The documentation is explicit about the omission: the two open models count **body motion joints only**, and "head and finger joints are excluded." The Pro's two head DoF are therefore a hardware feature that is **not actuated in the open model**. `[C4]`

The documentation is equally explicit about the policy gap: the 24-axis configuration has a complete training → Sim2Sim → deployment path and ships 24-axis velocity-tracking and dance policies. The **28-axis deployment configuration is ready, but its default policy files have not been released, and no dedicated training tasks are available yet.** `[C4]`

Licensing is not fully nailed down. The repository root at `c2956c58` contains no `LICENSE` file (checked 2026-10-23). The docs' license table assigns CERN-OHL-W-2.0 to hardware, CC BY-SA 4.0 to documentation, GPL-3.0 to firmware/deployment, and BSD-3-Clause to training code, but does not state the description repository's license. `[C4, C6]`

### 3.3 Direct answers

| Question | Answer | Evidence |
|---|---|---|
| Is RP1 real? | Yes; shipping product with open hardware/software. | `[C3, C4, C5]` |
| Is there a MuJoCo model? | Yes, 24- and 28-axis body models. | `[C4, M2]` |
| Does the model match the product? | Only the body. Head/finger joints are excluded; the Pro's 2-DoF head is not modeled. | `[C4, M2]` |
| Is there a neck in sim? | No. The head mesh is rigid with the torso; there is no head body or neck joint. | `[M2]` |
| Are the wrists honest? | 28-axis yes (3 DoF); 24-axis no (1 DoF). | `[C4, M2]` |
| Is there a policy? | 24-axis yes; 28-axis no default policy and no dedicated training task. | `[C4]` |
| Does it fix the 1.22 m height? | No. Body top 1.326 m vs G1's 1.321 m; both are child-scale. | `[M1, M2]` |
| Does it improve reach? | No. Shoulder→wrist ≈0.37 m vs G1's 0.38 m, despite the 550 mm arm spec. | `[M1, M2]` |
| Is it a drop-in? | No: different joint names/count, torque-only, no stand keyframe, no head frame. | `[M2]` |
| Worth switching for v1? | No. | Judgment, §7 |

The one configuration that fits Clank's wrist contract (28-axis) is the configuration with **no released policy**; the configuration with a policy (24-axis) has only a roll wrist and cannot realize controller orientation. That inversion is the core reason RP1 is not the right switch. `[C4, M2]`

---

## 4. Comparison: sim-ready bodies

### 4.1 Kinematics and frames

| Body (sim model) | Head/HMD frame | Mass | Active DoF | Neck | Wrist/arm | Shoulder→wrist | Source |
|---|---|---:|---:|---|---|---:|---|
| **G1 29** | head geom 1.221 m; visual top 1.321; rigid on torso | 33.3 kg model / 35 kg spec | 29 | 0 (3 waist) | 3 | 0.381 m | `[M1, C1, C2]` |
| **G1 + hands** | same (head geom 1.221; top 1.321) | 34.4 kg model | 43 | 0 | 3 | 0.381 m (wrist) | `[M1, C2]` |
| **RP1 24** | head mesh 1.326 m; rigid; no head joint | 39.1 kg | 24 | 0 | 1 | ≈0.37 m | `[M2, C3, C4]` |
| **RP1 28** | head mesh 1.326 m; rigid; no head joint | 40.7 kg | 28 | 0 | 3 | ≈0.37 m | `[M2, C3, C4]` |
| **Apollo 32** | neck base 1.575 m; visual top 1.725 | 80.9 kg model / 72.6 kg spec | 32 | 3 | 3 | 0.581 m | `[M1, C7, C8]` |
| **T1 23** | head links; visual top 1.158 | 31.6 kg model / ~30 kg spec | 23 | 2 | 0 (4-DoF arm) | 0.221 m | `[M1, C10, C11]` |
| **H1-2 27** | head fixed; 1.78 m | ~70 kg | 27 | 0 | 3 (incl. elbow roll) | 685 mm total arm | `[C14, C15]` |
| **N1 23** | camera link 1.165; visual top 1.242 | 39.7 kg | 23 | 0 | 1 | 0.232 m | `[M1, C16]` |

### 4.2 Integration and policy

| Body | MJCF / license | Actuators / stand | Policy status | Call |
|---|---|---|---|---|
| **G1 29** | Menagerie, BSD-3 | position; stand keyframe | Playground ONNX; 0.475 m/5 s drift measured | **keep** |
| **G1 + hands** | Menagerie, BSD-3 | position; stand keyframe (thumb keyframe limit bug) | same G1 policy; fingers not in policy | closure upgrade |
| **RP1 24** | MJCF; description repo has no LICENSE file | torque only; no keyframe | 24-axis ONNX + dance; full deploy path | **no** (rigid head, 1-DoF wrist) |
| **RP1 28** | same | torque only; no keyframe | **no default policy; no training task released** | **no** |
| **Apollo 32** | Menagerie, Apache-2.0 | position; stand keyframe | Playground joystick env; no checkpoint verified | phase-2 candidate |
| **T1 23** | Menagerie, Apache-2.0 | position; stand keyframe | Playground T1; Booster Gym; Holosoma 29-DoF ONNX (model not verified) | no (no wrist) |
| **H1-2 27** | URDF only; no Menagerie model | unknown; no keyframe | none found | no (no sim/policy) |
| **N1 23** | Menagerie, Apache-2.0 | motors; stand keyframe | none found | no (no neck, 1-DoF wrist) |

Sources for §4: `[M1]`, `[M2]`, `[C1–C16]`, `[C23]`.

---

## 5. Why the other bodies are not the answer

- **Booster T1 (23 DoF).** It is the closest cheap body with a real 2-DoF neck in sim, position actuators, and a stand keyframe. But the Menagerie model's arms are 4 DoF (shoulder pitch/roll, elbow pitch/yaw) with **no wrist**, so a controller pose cannot be tracked honestly. A 29-DoF T1 with 7-DoF arms exists in the Booster/Holosoma ecosystem, but the 29-DoF MJCF and ONNX compatibility were not verified here; that is a separate integration project, not a drop-in. `[M1, C10–C13]`
- **Apptronik Apollo (32 DoF).** The only adult-scale, sim-ready body with **both** a 3-DoF neck and 3-DoF wrists, plus a Menagerie model, position actuators, and a stand keyframe. It is the right *second* body if neck + adult height become hard requirements. It is 80.9 kg and has no verified Playground checkpoint; for v1's stand-and-arms scope that is acceptable, for C3 it is not. `[M1, C7–C9]`
- **Unitree H1-2 (27 DoF).** Adult (1.78 m, 70 kg), 7-DoF arms with three orientation DoF after the elbow (elbow roll + wrist pitch/yaw), but **no neck** and no Menagerie model or qualified policy; only a URDF. Menagerie carries the older H1, not the H1-2. It would be a larger integration than Apollo with no neck benefit. `[C14, C15, C30]`
- **Fourier N1 (23 DoF).** Has a camera link but **no neck**, a 5-DoF arm with a single wrist DoF, torque motors, and no policy found. It cannot track controller orientation. `[M1, C16]`
- **Fourier GR-1/GR-2.** URDF repositories exist (GPL-3.0), but no Menagerie model or learned policy was found in this pass. Not evaluated. `[C17]`
- **Agility Digit.** Community SDF repackaged from Agility's MJCF; no Menagerie model and no policy found. Not evaluated. `[C18]`
- **Berkeley Humanoid / Lite.** The Menagerie Berkeley Humanoid is 12 DoF with no arms. Wrong body for a player. The Lite is a cheap research platform, not a teleoperation body. `[C19, C29]`
- **ToddlerBot 2XM.** 0.56 m, 3.4 kg, 30 active DoF, 2-DoF neck, 7-DoF arms, $6k, MIT code / CC BY-NC-SA design. It is the cheapest necked body and a legitimate desk-scale teleop platform, but it is toy-scale and low-torque. Wrong player body for Superhot. `[C20–C22]`
- **Figure 02 / Tesla Optimus.** No public MJCF/URDF found in Menagerie or vendor repositories in this pass. Not testable. Negative finding, not proof of absence. `[C23]`

---

## 6. What would change in Clank if we switched

This is the part that makes the decision concrete. The current design is not body-agnostic at the joint-vector level, even though the task-space command is.

| Concern | G1 today | RP1 28 | Apollo 32 |
|---|---|---|---|
| `BodyState.q` | 29 | 28 | 32 |
| Head authority | `AUTH_HEAD` → waist yaw + base yaw; ±0.6 rad cap | waist yaw/roll; head rigid | `AUTH_HEAD` → 3-DoF neck; `AUTH_WAIST` → torso |
| Head limits | ±0.6 rad heuristic | waist yaw ±2.91; no neck | neck yaw ±1.658 (±95°), roll ±0.785 (±45°), pitch −0.262…+0.524 (−15…+30°) `[M1]` |
| HMD height | 1.221 m | no head frame; head mesh top 1.326 m | 1.575 m neck base; 1.725 m top |
| Shoulder / wrist | 1.082 / 0.721 m | 1.099 / 0.878 m | 1.367 / 0.813 m |
| Arm reach | 0.38 m | 0.37 m | 0.58 m |
| Stand | position actuators + `stand` | torque only, no keyframe | position actuators + `stand` |
| Policy | Playground ONNX, measured | none released for 28-axis | Playground env, checkpoint unverified |
| Closure | none | optional hands; not in body model | none in model |
| License | BSD-3 | unclear (no LICENSE in description repo) | Apache-2.0 |

### 6.1 BodyCommand / BodyState

`BodyCommand` is already writer-agnostic and task-space. The body-specific parts are the joint vector (`q[29]`), the frame names (head, left/right wrist), the authority mapping (which joints implement `AUTH_HEAD` and `AUTH_WAIST`), the supported mask, and the policy manifest. `[P1 §6–§7]`

If a second body is added, do it as a **`BodyProfile`** consumed by the executor, not a second executor:

- joint names and `q_dim`; frame names for head and wrists;
- neck/waist joint sets and measured limits;
- actuator mode (position/torque), gains, default pose, and whether a stand keyframe exists;
- policy path, observation layout, action scale, and control rate;
- HMD height, workspace box, and per-body calibration offsets.

Adapters and the authority bits stay unchanged. The mapping from `AUTH_HEAD` to neck vs waist is a profile field, not a branch in the adapter. Judgment: this is the cheap part of dual-support; the expensive part is re-qualifying stand, IK, and the policy. `[P1 §6–§7]`

### 6.2 Neck and look

With a real neck, `AUTH_HEAD` can drive head orientation directly and `AUTH_WAIST` stays waist/torso. The current ±0.6 rad waist cap and the "drop the remainder if `AUTH_ROOT_VEL` is unsupported" rule become body-specific: a 3-DoF neck can look without turning the base, but the balance envelope still limits how far the head can go while the legs are on a frozen policy. `[P1 §7; P3 §"Head-look with no neck"]`

The literature supports "neck is a plus, not a blocker": OmniH2O tracks a head keypoint on the neckless H1 from 3-point VR input, and its ablation shows 3-point ≈ 22-point tracking; TWIST2 adds a 2-DoF neck specifically to improve head fidelity. So the current waist-substitution design is defensible for v1, and a neck is a genuine upgrade, not a repair. `[C31, C32, P3]`

### 6.3 Height and workspace

Switching to Apollo changes the HMD from 1.221 m to roughly adult eye height and increases shoulder→wrist reach from 0.38 m to 0.58 m. That invalidates the E3 offsets, the C4 strike offsets, the workspace box, and the Superhot spatial calibration; all need re-measurement. It may also fix the "child in an adult kitchen" scale problem, but that problem has not been measured yet. `[P1 §10; P2]`

Switching to RP1 does not change the body height (1.326 m visual top vs G1's 1.321 m) and offers no independent head frame; the HMD would still be a fixed offset on the torso. It does **not** solve the adult-scale issue. `[M1, M2]`

### 6.4 Policy and balance

G1 has a measured stand and a known policy drift. Apollo has the assets for stand PD and a Playground environment, but no verified checkpoint. RP1 28-axis has neither a released policy nor a training task; RP1 24-axis has a policy but the wrong wrist. Retraining a humanoid policy is not a v1 activity here; I did not attempt it, and the 8 GB RTX 4060 is not assumed sufficient. `[A1; C4; C9]`

For v1's `hold`, stand PD is enough on G1 and should be enough on Apollo or T1 (position actuators + keyframe), though neither was stood up in this pass. On RP1 you would first have to write and tune a PD stand for a torque-only model with no keyframe. That is the hidden cost of "just switch the body." `[M2; P1 §7; A5]`

### 6.5 IK

G1, Apollo, and RP1-28 have 7-DoF arms, so the existing mink `FrameTask` on the wrist plus a `PostureTask` transfers. T1's 4-DoF arm and N1's 5-DoF arm cannot realize a full 6-DoF wrist pose; using them means dropping controller orientation and documenting a residual that the honesty rule currently forbids. G1+hands adds 14 finger joints; drive closure separately or put it in a low-priority task, not in the balance policy. `[P3; P1 §7; M1]`

### 6.6 Honesty

A body with no head frame forces you to invent one. On RP1, the head mesh is real and reaches 1.326 m, but it is rigid to the torso and has no joint; the HMD pose is therefore torso pose plus a fixed offset, exactly the G1 situation. The Pro's 2-DoF head would be a real improvement, but it is absent from the open model. A body with a 1-DoF wrist cannot claim honest controller orientation. A body with a 3-DoF neck and 3-DoF wrists (Apollo) can. `[M1, M2, C4]`

---

## 7. Recommendation

**v1: keep the G1 29-DoF. Freeze the body.** Finish C0 (game path) and C1–C2 (command bus, stand/hold) on the measured G1. Do not spend the v1 budget on a body swap that does not improve the head or wrist contract. `[P1 §14]`

**Do not switch to RP1.** It is a real, impressively open robot, but for this project the open model is a neckless, torque-only body whose 24-axis configuration has only a roll wrist and whose 28-axis configuration has no released policy. It is the same height and about the same reach as the G1. The one thing that would make it interesting — the Pro's 2-DoF head — is not in the simulation. `[C4, M2]`

**If a neck becomes a measured blocker after C2, add Apollo as a second body behind `BodyProfile`.** Apollo is the only adult-scale, sim-ready candidate in this pass with a 3-DoF neck, 3-DoF wrists, a Menagerie model, position actuators, and a stand keyframe. Its weaknesses are mass (80.9 kg) and the absence of a verified policy; both are acceptable for stand-and-arms and both must be resolved before claiming locomotion. `[M1, C7–C9]`

**If closure becomes the blocker, use G1+hands, not a new body.** The G1 with hands adds 14 finger joints and keeps the stand keyframe and base height, the same HMD height, and the same wrist frames. `[M1, C2]`

**If RP1 is wanted for portfolio novelty, make it a separate post-v1 demo, not the critical path.** The honest version is either (a) the 24-axis policy with a documented 1-DoF wrist limitation, or (b) a trained 28-axis policy, which is a training project with no released checkpoint or task. Neither belongs in v1. `[C4; A2]`

### Decision triggers

| Trigger | Then |
|---|---|
| C0 shows 1.221 m makes Superhot unplayable (scale, reach, doorways) | Evaluate Apollo; RP1 does not fix height |
| C2/C4 shows the waist-only look residual breaks a required skill | Evaluate Apollo's neck, or a G1 neck mod |
| A required skill depends on visible finger closure | G1+hands |
| RP1 is wanted purely for novelty | Separate demo after v1; not the player body |

A G1 with an added 2-DoF neck is also technically possible (TWIST2-style), and would be cheaper than a full body swap. It changes the body and the mass distribution, so it is not v1 work either; it is a fallback if Apollo proves too heavy or too poorly instrumented. `[C32; A3]`

---

## 8. What the literature uses (and why it matters for the portfolio)

The recent whole-body teleoperation work that Clank's research file leans on is overwhelmingly on the G1: HOVER, HOMIE, ExBody2, and TeleGate all use G1 variants, and OmniH2O's neckless-H1 result is the standard evidence that 3-point head+hands tracking can absorb head pose through the waist and base. `[C24–C27, C31, P3]`

That matters for the hiring signal. A reviewer can map G1 work onto known papers and known failure modes (arm-control-vs-balance conflict, action-scale mismatches, waist ownership). RP1's public history is days old (October 2026), and I found no VR/teleoperation paper using it; its published research is locomotion and foundation-model work. Switching would make the demo more novel but less legible: the story becomes "integrate a new body" rather than "solve whole-body VR control on a known humanoid." `[C5; A4]`

This is a judgment, not a measurement. If the goal is specifically to be early on RP1, that is a different project with a different risk profile.

---

## 9. Uncertainty / what was not verified

- **Apollo Playground checkpoint.** The joystick environment exists, but I did not verify that a trained checkpoint is published or that it runs with this Menagerie model. Apollo's stand-PD path is available without one; C3 is not. `[C9, M1]`
- **RP1 28-axis policy.** The docs say the default policy files are not released and no dedicated training task exists. That could change; if it does, the trade-off changes. `[C4]`
- **RP1 24-axis policy.** Its existence and the Sim2Sim path are documented; I did not run the policy or measure stand drift. `[C4]`
- **RP1 description license.** No `LICENSE` file at the repository root; the docs' license table does not explicitly cover the description repo. Treat reuse terms as unclear until confirmed. `[C4, C6]`
- **RP1 head.** The visual head mesh is present, but there is no head body, joint, or separate HMD frame; the Pro's 2-DoF head is hardware-only in this pass. `[M2]`
- **T1 29-DoF.** Holosoma ships a `ppo_t1_29dof.onnx`, but I did not verify that its MJCF matches the 29-DoF T1 or that the Menagerie 23-DoF model can be upgraded cleanly. `[C13, M1]`
- **H1-2 and N1.** URDF/MJCF assets exist, but no qualified policy was found and neither was run. `[C14, C16]`
- **Apollo/T1 stand PD.** Position actuators and a stand keyframe exist, but neither body was stood up in this pass; the "should be enough" claim is an inference, not a measurement. `[A5]`
- **Figure 02 / Optimus.** The absence of a public model is a negative finding from this pass, not proof that none exists. `[C23]`
- **No RP1 VR/teleop paper.** The absence is a search result, not proof of absence. `[C5]`
- **Heights.** Product height, visual mesh top, and HMD frame are different quantities. The table reports the measured mesh top where available; a real HMD mount must be measured per body. `[M1, M2]`
- **Superhot at 1.221 m.** Unmeasured until C0. Do not switch bodies on the assumption that it is unplayable. `[P1 §10]`
- **G1 + hands.** Joint count and model exist (34.4 kg, 43 actuated); hand collision behavior and policy interaction were not measured. `[M1, C2]`
- **Local training.** Not attempted on the RTX 4060 8 GB; treat retraining as out of scope for v1. `[A1]`

---

## 10. Sources

### Project documents

- `[P1]` `../docs/2026-10-23-clank-design.md` — body constraints, `BodyCommand`, authority bits, executor, C0–C5, HMD height.
- `[P2]` `research/e3-spike-results.md` — G1 stand, 500 Hz, packet match, head/wrist measurements.
- `[P3]` `research/g1-integration-control.md` — WBC architecture, neckless head-look, IK, policy caveats.
- `[P4]` `RESEARCH.md` §5 — body mapping, G1 29/43 DoF, no neck.
- `[P5]` `stubborn-gamer/SEED.md` — Playground policy drift, why G1 was chosen.

### Measurements

- `[M1]` MuJoCo Menagerie at commit `cda5236f98383f6fbaa4820c6b18dba892cd087f` (2026-10-19), models `unitree_g1`, `apptronik_apollo`, `booster_t1`, `fourier_n1`, `berkeley_humanoid`, `unitree_h1`; MuJoCo 3.17.0 in `spike/e3/.venv`. Measured `nq`, `nu`, mass, keyframes, actuator bias/gain, joint ranges, body `xpos`, and mesh world bounds. https://github.com/google-deepmind/mujoco_menagerie/tree/cda5236f98383f6fbaa4820c6b18dba892cd087f
- `[M2]` RoboParty `RP1_description` at commit `c2956c58b7381c0e7433e200ef59930468fe534b` (2026-10-22), `mjcf/robot.xml`, `robot_24dof.xml`, `robot_hand.xml`; same MuJoCo. Measured 24/28 joints, mass, no keyframes, torque actuators, joint ranges, body `xpos`, mesh world bounds. https://github.com/Roboparty/RP1_description/tree/c2956c58b7381c0e7433e200ef59930468fe534b

### Web sources

- `[C1]` Unitree G1 product page (1.32 m, ~35 kg, 23–43 DoF, from $13.5K, arm span 0.45 m). https://www.unitree.com/g1/
- `[C2]` MuJoCo Menagerie `unitree_g1` (BSD-3-Clause; 29 DoF; position actuators; `stand`/`home`/`knees_bent`; `g1_with_hands.xml`). https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1
- `[C3]` RoboParty RP1 product page (1.35 m; 39/41 kg; 24/30 DoF; head 0/2; arms 5/7; 2 waist; 550 mm arm; four head configurations). https://www.roboparty.com/products/RP1
- `[C4]` RoboParty RP1 documentation (24/28 body models; head and finger joints excluded; 28-axis policy not released; 24-axis policy path; MuJoCo/Isaac Lab; licenses). https://roboparty.com/documents/roboto-01
- `[C5]` RoboParty RP1 repository (open-stack overview, repository guide, 2026 history, 731 stars at access). https://github.com/Roboparty/RP1
- `[C6]` RoboParty RP1_description repository (URDF/MJCF/meshes; no LICENSE file at repository root as of 2026-10-23). https://github.com/Roboparty/RP1_description
- `[C7]` Apptronik Apollo announcement (5 ft 8 in / 1.73 m, 160 lb / 72.6 kg). https://apptronik.com/news-collection/apptronik-unveils-apollo
- `[C8]` MuJoCo Menagerie `apptronik_apollo` (Apache-2.0; 32 DoF; position actuators; `stand` keyframe). https://github.com/google-deepmind/mujoco_menagerie/tree/main/apptronik_apollo
- `[C9]` MuJoCo Playground (G1, T1, Apollo, Berkeley Humanoid joystick environments; ONNX/IPC deployment). https://github.com/google-deepmind/mujoco_playground
- `[C10]` Booster T1 product page (1.18 m; ~30 kg; 23 DoF; 2-DoF head; 4-DoF arms, 7-DoF with gripper). https://www.booster.tech/booster-t1/
- `[C11]` MuJoCo Menagerie `booster_t1` (Apache-2.0; 23 DoF; position actuators; stand keyframe). https://github.com/google-deepmind/mujoco_menagerie/tree/main/booster_t1
- `[C12]` Booster Gym (T1 RL training and MuJoCo sim2sim). https://github.com/BoosterRobotics/booster_gym
- `[C13]` Holosoma (Apache-2.0; `ppo_t1_29dof.onnx`, `fastsac_g1_29dof.onnx`; MuJoCo sim2sim). https://github.com/amazon-far/holosoma
- `[C14]` Unitree H1-2 developer page (27 DoF: 6/leg, 1 torso, 7/arm; 1.78 m; ~70 kg; arm 685 mm; no neck). https://support.unitree.com/home/en/H1_developer/About_H1-2
- `[C15]` Unitree H1-2 product page. https://www.unitree.com/h1/
- `[C16]` MuJoCo Menagerie `fourier_n1` (Apache-2.0; 23 DoF; camera link). https://github.com/google-deepmind/mujoco_menagerie/tree/main/fourier_n1
- `[C17]` Fourier GRx models repository (GPL-3.0; URDF/GR-1/GR-2). https://github.com/FFTAI/Wiki-GRx-Models
- `[C18]` Digit V3 SDF model (community repackaging derived from Agility's MJCF). https://github.com/yu-fz/Digit-V3_SDF_Model
- `[C19]` MuJoCo Menagerie `berkeley_humanoid` (BSD-3-Clause; 12 DoF). https://github.com/google-deepmind/mujoco_menagerie/tree/main/berkeley_humanoid
- `[C20]` ToddlerBot paper (0.56 m, 3.4 kg, 30 active DoF, 2-DoF neck, 7-DoF arms, <$6k). https://arxiv.org/abs/2502.00893
- `[C21]` ToddlerBot project site (30 active DoF; 2-DoF neck; 2-DoF waist; MIT code, CC BY-NC-SA design). https://toddlerbot.github.io/
- `[C22]` MuJoCo Menagerie ToddlerBot 2XM (MIT; 44 joints including fingers). https://github.com/google-deepmind/mujoco_menagerie/tree/main/toddlerbot_2xm
- `[C23]` MuJoCo Menagerie index (humanoid inventory; no Figure/Optimus). https://github.com/google-deepmind/mujoco_menagerie
- `[C24]` HOVER (G1; versatile humanoid controller with upper/lower-body mode masks). https://arxiv.org/abs/2410.21229
- `[C25]` HOMIE (G1; upper-body pose curriculum under balance). https://arxiv.org/html/2502.13013
- `[C26]` ExBody2 (G1; decoupled upper-body keypoints and root velocity). https://arxiv.org/html/2412.13196
- `[C27]` TeleGate (29-DoF G1; 50 Hz policy / 500 Hz PD in MuJoCo). https://arxiv.org/html/2602.09628v1
- `[C28]` MuJoCo Playground paper. https://arxiv.org/html/2502.08844v1
- `[C29]` Berkeley Humanoid Lite. https://arxiv.org/abs/2504.17249
- `[C30]` MuJoCo Menagerie `unitree_h1` (H1, not H1-2; 19 DoF). https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_h1
- `[C31]` OmniH2O (H1; 3-point head+hands tracking; ablation vs 22-point). https://arxiv.org/html/2406.08858
- `[C32]` TWIST2 (adds a 2-DoF neck; closed-form yaw/pitch from rotation). https://arxiv.org/html/2511.02832

### Assumptions / judgments

- `[A1]` Local retraining of a humanoid policy on the RTX 4060 8 GB is out of scope for v1; I did not attempt it.
- `[A2]` RP1's open stack is genuinely interesting for a portfolio, but only as a separate, non-critical demo; the v1 claim is whole-body VR control, not body porting.
- `[A3]` A G1 neck mod is technically feasible but changes the body and the mass distribution; not evaluated.
- `[A4]` Hiring reviewers are more likely to recognize G1-based whole-body control than RP1 work. This is a judgment about signal, not a measured fact.
- `[A5]` Stand PD on Apollo and T1 is assumed to work from position actuators plus a stand keyframe; neither was tested here.
