# Clank — state of findings

**Date:** 2026-10-23, revised **2026-10-24** after C3m and C0.
**Purpose:** one shared ground-truth file so no researcher, builder or reader re-derives E1–E3, C3, C3m or C0. It states the goal *with no caveats*, what is proven, what is falsified, what is still open, and the pass criteria any proposal must meet.

**Claim tags.** `MEASURED` = a local spike on this machine, file named. `CITED` = source URL. `ASSUMPTION` = reasoning, untested.

**What changed in this revision.** The 10-23 version ended with *"what is missing is an executor that can honour [the command struct]"*. Since then: an executor was found and measured (**GEAR-SONIC**), a second was measured and is precise but cannot walk (**mink QP**), the hybrid of the two was measured and is **dead**, the earlier "hold" failure was traced to **our own harness**, and **C0** proved a commercial game accepts the pipe. §4's R1–R8 now carry measured status instead of being hypothetical, and §7 lists what is genuinely still open.

---

## 0. The goal, stated without softening

A simulated **Unitree G1** in MuJoCo plays **Superhot VR**, rendered by the real game, with no headset and no controllers. It must be capable of:

1. **Walking** — move between positions in the level under its own balance.
2. **Dodging** — avoid incoming fire while staying upright.
3. **Aiming** — independently position its head and hands at a target.
4. **Completing a stage** — clear at least one stage with its own body.

Everything the game sees comes from the simulation: the emulated HMD pose is the robot's *achieved* torso pose, the emulated controller poses are its *achieved* wrist poses, and the robot's only camera is a capture of the game window. No faked poses, no game telemetry as control, no floating hands.

The project exists as a hiring artifact for **real-time controls and systems integration on a humanoid** (Apptronik / Figure / Agility / NVIDIA-shaped internships). The claim that must be true is **balance-constrained whole-body control with an OpenXR runtime as the task interface**. The demo may look clumsy. It may not be dishonest.

---

## 1. What is proven working

### 1.1 The OpenXR injection pipe — E1, E2 `MEASURED`

- Monado's `remote` driver accepts a client over **TCP 4242**. Magic `mndrmt3\0`, **376-byte** packets. One client, hub binds `0.0.0.0`, **last pose freezes on disconnect**.
- `P_OVERRIDE_ACTIVE_CONFIG=remote` is sufficient; **no `config_v0.json`**. `IPC_EXIT_ON_DISCONNECT=off`, `XRT_NO_STDIN=1`.
- Compositor must run with `XRT_COMPOSITOR_COMPUTE=false` (GRAPHICS queue). The compute compositor and `XRT_DEBUG_GUI=1` both **SIGSEGV** in the Wayland presentation queue-family check. `VK_DRIVER_FILES=/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json`.
- Runtime: `monado-psvr2-psvr2-work-steamvr-2026-06-04` (Beyley PSVR2 fork).

### 1.2 Frame capture — E2 `MEASURED`

- Working capture is **grim on the Monado window** at **36.9 fps**, located via `hyprctl clients -j`.
- **Pose → pixel latency 30–59 ms (mid 44 ms)**, ~27 ms of which is grim's own overhead.

### 1.3 The body → HMD bridge — E3 `MEASURED`

- G1 stepped at 500 Hz PD with a 50 Hz publisher: **1001 packets in 20.02 s**; positions/quats matched `mjData` after frame conversion with **max error 0.0**.
- HMD yaw tracked torso to **0.0108°**. `hello_xr` followed.
- Frame conversion (MuJoCo Z-up `wxyz` → OpenXR Y-up `xyzw`): `xr_x = -mj_y`, `xr_y = mj_z`, `xr_z = -mj_x`.
- HMD height **1.221 m** (`stand`), **1.170 m** (`knees_bent`).

### 1.4 The executor: GEAR-SONIC — C3i, C3j, C3m `MEASURED`

`nvidia/GEAR-SONIC` (weights: **NVIDIA Open Model License**, not Apache), encoder 1762→64, decoder 994→29, **50 Hz policy / 500 Hz PD**, headless in `spike/wbc`. **6.8 ms/tick CPU, ~0.9× realtime.**

| Capability | Measurement |
| --- | --- |
| **Walk** | **39.79 m path / 37.82 m net in 60 s**, 0.663 / 0.631 m/s, yaw drift **−2.10°**, path error mean 0.230 / max 0.554 m, tilt 9.48°, no fall |
| **Hold** | **RMS velocity 1.3×10⁻⁴ m/s settled — 0.02% of walking speed; p95 2×10⁻⁴.** (Under Superhot this pinned its 0.005 clock; the game changed, the number survives) |
| **Walk → hold → walk** | **5/5 cycles, 100 s, no fall**, worst tilt 9.5°, cumulative hold 0.478 m, **walk resumes within ~0.2 m** of reference, re-lock 4.14–4.94 s |
| **Dodge** | **0.15–0.18 m in < 0.5 s** |
| **Aim while walking, ahead** | **3.51 cm / 7.22°** (p95 5.63 cm) |
| **Aim while walking, 30° off-axis** | **8.63 cm / 28.33°** — the weak axis |
| **Head yaw → torso yaw** | gain **~0.83**, saturating at **27–31°** |

**The hold recipe matters and the obvious answer is wrong.** The ingredient is **whether the encoder's 0.9 s lookahead still contains walking**, not which freeze mode you pick. Anchor-only freeze of a walking clip: **3.93 m / 20 s at 0.20 m/s RMS**. Same pose with the horizon flattened: **0.11 m / 0.0085**. But a **standing clip with an anchor-index freeze** is the best configuration measured (**1.9×10⁻⁴**), and flattening its horizon *degrades* it (**0.0233**). Ship: **switch to `squat_001__A359` frame 0, anchor index held.**

**Known limits of this executor:** wrists hold to **4–6 cm / 17–23°** and walk to **5–10 cm / 8–19°**; head tracks **y only** (0.15–0.18 m for a 0.20–0.30 m command) with **no head z/x and no pitch/roll**; arms float ~0.15 m off the body. Holds were measured to 20 s and cycling to 10 s per hold.

### 1.5 The game accepts the pipe — C0 `MEASURED`

`research/c0-results.md`.

- Superhot VR reaches **`XR_SESSION_STATE_FOCUSED`** on a headset that does not exist (`System Properties: Name="Monado: Remote HMD"`, 58 runtime extensions), and its **view tracks our poses**: a ±35° sweep gives **41.97 mean pixel change / 59.3% of pixels vs 4.74 / 4.0% static — 9× over scene animation**. Operator saw the camera sweep.
- Captures are real game content (**0.00% near-black, 3175 distinct colours**), and were **99.89% black** before the runtime pointer was fixed.
- The blocking chain, for anyone repeating it: `/etc/xdg/openxr/1/active_runtime.json` is invisible inside the Proton container; the user-level path is `~/.config/openxr/1/active_runtime.json` (**not** `~/.local/share`); the loader caches its runtime list per process so **a restart is mandatory**; under Wine the path is prefix registry → `C:\openxr\wineopenxr64.json` → `wineopenxr.dll` → host loader.

### 1.6 The robot

- **29 actuated joints**: 12 leg, 3 waist, 14 arm. **No neck** — `head_link` is a rigid mesh inside `torso_link`. No camera. `left_shoulder_pitch` moves the EE 0.151 m.
- Waist: yaw **±2.618 rad**, roll/pitch **±0.52 rad**. The waist is the only "neck" available.
- Menagerie at `spike/e3/vendor/mujoco_menagerie/unitree_g1/`.

---

## 2. What is falsified, and why

Negative results are deliverables. Each of these was measured, not argued away.

### 2.1 A velocity-tracking locomotion policy is not a whole-body controller — C3a–C3h `MEASURED`

All against the MuJoCo Playground G1 policy (ONNX, 903,543 B) on the **correct** collision model, `spike/c3/playground_g1/xmls/scene_mjx_feetonly_flat_terrain.xml`. **Root cause, stated once: it is a 29-DoF joint-position velocity tracker. Its action space is the whole body, it owns every joint, and it was trained with no notion of an external upper-body authority.**

| Finding | Number |
| --- | --- |
| Walks straight, briefly | +0.23° / 10 s, 7.45 m, 0.743 m/s |
| Does not stay straight | **−17.3° / 30 s** |
| Hard wrist IK while walking | **FALL 4.84 s** (neutral), **3.39 s** (reach) |
| Hard wrist IK at **zero** command | **FALL 5.27 s** after 1.72 m drift — not a speed problem |
| Soft 15% blend | survives, but **wrist never moves** (0.119 m residual) |
| Waist look while walking | +34.6° arc, but base does a **−185° U-turn** |
| Look + arms while walking | **FALL 2.95 s** |
| Best zero-command PI loop | 0.274 / 0.326 / 0.361 m at 20 / 60 / 120 s, **still stepping at 0.12 m/s RMS** |
| Walk → hold, gain-scheduled | **FALL 3.11–4.24 s** |
| Walk → static stand, any ramp 0.05–2.0 s | **FALL 0.64–1.49 s, all variants** |

### 2.2 The QP-on-SONIC hybrid is dead — C3j `MEASURED`

A mink QP that overrides SONIC's joints was tried in 16 variants. It fights the policy:

| Variant | Hold | Walking aim |
| --- | --- | --- |
| Baseline (no override) | 1.14 cm / 7.33° | 3.51 cm / 7.22° |
| Hard arm override | **14.40 cm / 24.00°** | **12.84 cm / 22.40°** |
| `rate:2.0` / `rate:6.0` | 12.43 / 13.08 cm — no help | — |
| `blend:0.5` / `blend:0.75` | 4.88 / 7.49 cm | — |
| **`orient` only (wrist rotation)** | 3.27 cm / **5.08°** | aim 4.42° — **the one variant that helps** |
| `int:0.5`, `int:1.0` | **FALL** (tilt 64.7° / 74.8°) | **FALL** |

`arm_track_err_final ≈ 0.23`: the IK never converged. **Only wrist-orientation override survives** (7.22° → 5.08° pointing), and is worth at most a polish, not a redesign.

### 2.3 The classical QP is precise but cannot walk — `research/wbc-classical-proposal.md` `MEASURED`

mink floating-base QP at 50 Hz, `daqp`, 10×2 ms substeps.

- **Very good at precision:** hold **0.22 mm / 20 s** (head 1.4 mm); look ±35° at **0.63°** error; reach **2.2–12.9 mm** single-arm; aim **0.18–2.34°**; punch **9.5 mm / 3.6°** (E3's DLS was 13.5 cm); solve **0.51–1.08 ms** mean.
- **Structurally cannot walk:** weight-shift limit **0.10 m** (0.12 tips), single support tips in **1.3–1.6 s**, because it has no contact-force/CoP decision variable. Dodge 12.1–14.7 cm but **0.62–0.86 s** to peak.
- **Reach envelope ~0.20 m.** A probe at **0.70 m** forward, eye level, came up **19.7–27.3 cm short**.

**Verdict: the QP is a fine upper-body controller and not a body.** It cannot be composed onto SONIC (§2.2), so it is not currently in the executor.

### 2.4 Retracted claims — do not re-inherit

- **"0.516 m/s verified"** was the x-component of an **arc**; the arc was our **Menagerie** XML, not the policy. Same ONNX: **+0.23°/10 s** on Playground vs **+174.5°/10 s** on Menagerie.
- **"SONIC's hold is motion-dependent — quote 3.6 cm, or 0.23–2.07 m per hold."** Both were artifacts of freezing only the anchor index of a walking clip (§1.4).
- **"Slow the walk until the arms survive."** Falsified by the zero-command fall at 5.27 s.
- **"The Playground policy can be fixed."** It cannot; it has no stop and no upper-body authority.

---

## 3. Why the gap was structural

Published whole-body humanoids do not bolt upper-body authority onto a locomotion policy. They train **one policy whose command space already includes the upper body**: HOVER/OmniH2O track head + hands directly; HOMIE keeps **12 lower-body actions** and treats the upper body as a disturbance it was trained to tolerate; ExBody separates expressive upper body from stable lower body. Unitree's own `rt/arm_sdk` records the same failure in the wild.

`research/humanoid-command-space.md` argues the native interface is a **masked, root-relative task-space command** (root velocity + yaw rate + height, head pose, two hand poses, closure, authority bitmask, TTL, termination) — not a verb enum. **That contract survives unchanged.** What changed is that an executor now exists that honours most of it.

---

## 4. Requirements, with measured status

The pass bars. `research/wbc-*.md` proposals are still judged against these.

| # | Requirement | Pass bar | **Status 2026-10-24** |
| --- | --- | --- | --- |
| R1 | **Arms and head independently commandable** | wrist residual **≤ 3 cm** while walking and holding; orientation error measured | **PARTIAL** — 3.51 cm / 7.22° **ahead** (just over), **8.63 cm / 28.33° at 30° off-axis** (fails). Orientation now measured: 7.22° → 5.08° with wrist-orient override. Head y only, no pitch/roll |
| R2 | **Holds still enough that a slow think does not disturb what it touches** | ≤ ~5 cm net over a 20 s think | **MET with a stated caveat** — RMS **1.3×10⁻⁴ m/s**, but a one-time **8–12 cm settle** and a **0.13–0.18 m/s transient for 3–5 s** on entry. A think costs **20 s of wall-clock**; there is no game clock to pay, because Job Simulator does not scale time |
| R3 | **Walks** | ≥ 5 m, ≥ 0.3 m/s, 60 s no fall | **MET** — 39.79 m / 60 s at 0.663 m/s, yaw −2.10°, no fall. Push recovery untested |
| R4 | **Transitions do not fall** | 5 consecutive walk→hold→walk | **MET** — 5/5, no fall, resumes within 0.2 m |
| R5 | **Dodges** | head displacement 10–25 cm in < 500 ms, upright | **MET on displacement** — 0.15–0.18 m in < 0.5 s. What the *game* requires is unmeasured |
| R6 | **Runs live on this machine** | 50 Hz beside Superhot + Monado + MuJoCo on an 8 GB RTX 4060 | **OPEN** — SONIC is 6.8 ms/tick (~0.9× realtime) headless; never run beside the game |
| R7 | **Honest** | every pose is `mjData` after `mj_step` | **MET by construction** in E3/C3i–C3m. **Caveat for C0**: its poses were scripted constants built from measured geometry, so the game test was honest but not sim-driven |
| R8 | **Completes a stage** | one early stage cleared end-to-end, recorded | **OPEN** — gated on C0-3/5/6 and C1–C5 |

**Also open, and cheap:** the **E5 brain bake-off** (design doc §14) — four candidates on 100 labelled digest frames, scored against majority-class and the scripted commander. Non-blocking for the milestone. `research/d1-capabilities.md` §6b already measured the decisive cost: d1 over a text digest is **5.5 s**, over an image **7 s at 116 tokens / 36–39 s at 591** — which is why the brain is a digest classifier and not a VLM.

**The binding constraint is now R1's off-axis case.** At 30° off-axis the body cannot aim well, and it cannot be fixed by arm authority (§2.2). The design consequence is a rule, not a bug: **the body must turn to face its target rather than aim sideways.** Everything about "walk to a position, then aim" must be planned around that.

---

## 5. The game side — what is known

**Job Simulator: nothing is measured yet.** The target changed on 2026-10-31. Everything below is `[OPERATOR]` description plus one wiki citation, not measurement:

- JobBot speaks instructions in the museum hub and inside each job `[CITED: https://jobsimulator.fandom.com/wiki/JobBot]`.
- Level select is a museum hub: take a business card, take a cartridge from a tray, insert it into the machine, pull the lever. In-range cues `[OPERATOR 2026-11-01]`: grabbables light **blue** when the hand is close; the cartridge slot is **painted blue**; a **yellow ghost cartridge** at the slot means release will insert. Superhot's green slot oracle does not apply. Playbook: `docs/superpowers/specs/2026-10-31-brain.md` §6.2.
- Store Clerk opens with spoken chores: take a ticket, clear the counter, turn on the security camera.
- There is **no time dilation and no combat**, so a think costs wall-clock.
- **No state API exists.** That negative search result now matters more, not less (`RESEARCH.md` §12.5): occupancy can only come from the latched grip and pixels, never from telemetry.

**Superhot, kept as provenance and retired as a premise.** These values were read from that game's serialized assets; they are not premises of this design: `TimeControl.minTimescale = 0.005` (0.5% speed, not a pause), `maxTimescale = 1.0`, `tickThreshold = 0.1`, `movementTick = (0.1, 0.1)`, `rotation = 0.05`; `Bullet.COLLISION_RADIUS` 0.03 m, `distanceCheckTimeTreshold` 2.5 m (sic — the game's own spelling), `maxHandsDistance` 2.0 m; The Fall's 1.41 m enemy and 0.70 m pistol/exit pyramid; scene order The Fall → Cage Fight → Corridor → Dojo → Airport. Detail in `research/wbc-game-proposal.md`. **Do not re-inherit any of it as a requirement.**

**The open question that decides the demo is now a measurement, not a gamble:** how far is Job Simulator's counter from the body's spawn, and where does each chore object sit relative to the reachable set? Nobody has measured this. The 0.70 m pedestal that made walking load-bearing under Superhot does not exist here.

---

## 6. Decisions locked

- **Monado `remote` driver** is the injection path. No custom OpenVR driver; SteamVR avoided.
- **Job Simulator** is the first game and the first adapter (changed 2026-10-31; Superhot VR is a possible second). The slow brain is licensed by the absence of combat and by the world not threatening the body, **not** by a frozen clock — an assumption, not a measurement (`docs/superpowers/specs/2026-10-31-brain.md` §3.3).
- **Native interface = `BodyCommand` / `BodyState`**, masked task-space with authority bitmask, TTL, termination. Game verbs are adapters, not the interface.
- **Three loops, one socket**: body 50 Hz packets / 500 Hz PD; skills 10–30 Hz; brain 5–20 s. The VLM never writes joints or poses.
- **`hold` is load-bearing and active**, never "send zero".
- **Executor = GEAR-SONIC.** The Playground ONNX is off the critical path. The standalone QP is not a body. The QP-on-SONIC joint override is dead; **wrist-orientation-only override** is the one permitted refinement.
- **Keep the G1.** No RP1 swap.
- **Brain: unresolved, and the earlier default was not evidence-based.** CPU-only, no cloud. The 10-23 version named MiniCPM5-2B; nothing ever compared candidates on *this* layer. `LFM2.5-VL-3B` (**0.21 s p50 / 0.24 s p95**, 4,658 MiB) was the stubborn-gamer bench's *"best compact fast candidate, particularly for routine decisions"* — which is our layer — and was struck out here for a **planner** failure (73% `navigate_to` on hard home-assistant cases) that does not apply. Current candidates: `LFM2.5-VL-3B` (0.21 s), `LFM3-1.7B` (text, 256K ctx, native tool calling, ~1 GB at 4-bit), `d1-2B` (**1.7 s p50 CPU text-only**, third party; no generated tokens), MiniCPM5-2B (20 s p50 / 31 s p95, 3,932 MiB). MiMo-9B (5.07 s p50, 7,346 MiB) does not fit beside the rest. **Decided by E5, not by argument** — §18.8 of the design doc.

---

## 7. What is still open

Ordered by what blocks what. Each has a concrete next action.

| # | Open item | Next action | Notes |
| --- | --- | --- | --- |
| 1 | **C0-3: trigger reaches the game** | `c0_client.py --mode trigger` with a pistol in hand | needs the tutorial cleared |
| 2 | ~~**C0-5: clock floor in practice**~~ **RETIRED 2026-10-31** — it measured Superhot time dilation; Job Simulator has none. The 0.005 floor stays as provenance only | hold vs walk, pixel-only clock-ratio protocol (`spike/c0/README.md`) | only if Superhot returns as a second adapter |
| 3 | **C0-6: The Fall's 0.70 m pistol/exit** | stand still in level 1 and ask what is visible and reachable | **biggest design risk** — 0.70 m vs 0.20 m envelope |
| 4 | **R6: live co-residency** | SONIC + Monado + Superhot + grim resident, measure dropped ticks | 6.8 ms/tick and 0.9× realtime headless |
| 5 | **R1 off-axis aiming** | plan movement so the body faces its target; measure at 0°, 15°, 30° after turning | do not attempt arm-authority fixes |
| 6 | **Wrist orientation in the E3 body** | never measured on the 500 Hz bridge | E3 measured position only |
| 7 | **Grab confirmation** | `maxHandsDistance` is 2.0 m — what closes a grab? | unknown |
| 8 | **Scene memory** | enemy bearing/range plus staleness over a 5–20 s think | nothing designed yet |
| 9 | **Hold under an arm or head command** | hold with a commanded wrist target, 20 s | holds were measured with arms idle |
| 10 | **Holds longer than 20 s** | 60 s hold, watch for creep | untested |
| 11 | **Does the 8–12 cm settle matter in game?** | observe in C0-5 | only visible with pixels |
| 12 | **C3k: unify SONIC's model with Menagerie** | **deferred, probably unnecessary** — the two controllers are no longer composed | would only matter if the QP returns |

---

## 8. Current environment state

As left by C0 on 2026-10-24:

- `monado-service` running detached (`/tmp/c0-monado.log`), listening on 4242, holding `/run/user/1000/monado_comp_ipc`.
- **`monado.socket` is stopped** and must stay stopped while our service owns the IPC path. A reboot restores it; to restore by hand, kill our service first.
- `~/.config/openxr/1/active_runtime.json` exists — **this is the fix that makes Monado visible to Proton games.** Keep it.
- `~/.local/share/openxr/1/active_runtime.json` is **inert** (not a searched path); safe to delete.
- Superhot runs on the laptop panel (`eDP-1`); the working monitor (`DP-1`) is untouched.
- Hyprland raises **"Application Not Responding"** for the Monado window because Monado blocks in `renderer_wait_for_present`. Cosmetic; kill the dialog.

---

## 9. Rules for anyone using this file

- Tag every claim `MEASURED` / `CITED` / `ASSUMPTION`. A number without a tag is not a finding.
- Prefer measuring to arguing. A one-hour MuJoCo experiment beats a paragraph of reasoning.
- **A negative result is a deliverable.** Never sweep until something survives and report only that; report the tuning window and its fragility.
- Do not assume a paper's number transfers. This project has been bitten by inherited numbers twice, by evaluating a policy on the wrong collision model once, and by trusting a harness artifact once (§2.4).
- The goal is not negotiable down to "a turret". A proposal that cannot reach R1–R5 should say so plainly and price what it would take.
