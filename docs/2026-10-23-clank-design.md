# Clank design

**Date:** 2026-10-23
**Status:** written from locked conversation + E1–E3 measurements; builder-check settlements and the C3 outcomes in §18. Not an implementation plan. Nothing below authorizes product code until this document is approved. **C3 ran 2026-10-23 (twice) and failed on the Playground policy; the same day's WBC research found the answer and §7 was rewritten from it.** The walking policy reproduces the archived numbers exactly, the heading curve was our collision model, and the two-mode body died. But the Playground ONNX is superseded as the executor: **GEAR-SONIC walks, holds and transitions**, and a **classical mink QP owns precision** to 0.22 mm. The game's clock floor is measured at **0.005**. **C0 ran 2026-10-24 and passed its gate: Superhot VR runs a focused session on a headset that does not exist, and its view tracks our poses** (`research/c0-results.md`). See §18.3–§18.7.
**Question this answers:** what to build so a simulated Unitree G1 can play Superhot VR through an honest body, with a slow local VLM naming goals rather than poses.

Related research (not this document): `RESEARCH.md`, `research/e1-spike-results.md`, `research/e2-spike-results.md`, `research/e3-spike-results.md`, `research/system1-game-agents.md`, `research/humanoid-command-space.md`, `research/g1-integration-control.md`.

---

## 1. Intent

A hiring artifact for real-time controls and systems integration on a humanoid: Apptronik / Figure / Agility / NVIDIA-shaped internships. The claim that has to be true on camera is **balance-constrained whole-body control with an OpenXR runtime as the task interface**. The demo can look clumsy. Fake poses cannot.

The project is not “a robot plays a video game.” **Job Simulator is the first *adapter*** over a general body command, and the body must still make sense for another VR title or a non-game reach. Superhot VR was the first title chosen; since 2026-10-31 it is at most a *second* adapter, because the brain it was chosen to service was replaced (`docs/superpowers/specs/2026-10-31-brain.md`).

---

## 2. Goals, non-goals, anti-goals

### Goals

1. Run a commercial VR game with no headset and no controllers. The game’s HMD and controllers are the simulated G1’s *achieved* torso/head and wrists.
2. The game’s composited view is the robot’s only camera (desktop capture of the Monado/hello_xr/game window).
3. A slow local VLM selects goals. A 50 Hz executor turns those into joint targets. MuJoCo physics is the only source of OpenXR poses.
4. Ship: a public repo, a README a hiring manager can read in two minutes, and a 60–90 s recording of the loop. Failure in-game is allowed. The video must show the body in the view and the honesty numbers (latency, residual, whether the pose came from `mjData`).

### Non-goals (v1)

- Retraining the locomotion policy or any VLA (HOVER, HOMIE, Helix S1, π0, ACT).
- ROS2 (covered in club work).
- Game telemetry as a control input. Scoreboard-only is allowed later; it is not in the motor path.
- A force loop from `xrApplyHapticFeedback` into MuJoCo.
- VRChat OSC / Resonite as a substitute for the emulated-device path.
- Beat Saber, Gorilla Tag, industrial PLC work.
- Cross-game generality as a *deliverable*. The *interface* must be general; only Superhot is required to run.
- GPU VLM next to Superhot + Monado + MuJoCo on the 8 GB 4060.

### Anti-goals (a failure even if the demo “works”)

- Sending desired IK targets, scripted oscillators, or VLM-emitted poses to Monado.
- Looking like a game hack: injecting into the title, reading game memory, or driving Superhot through an unofficial API.
- A 20-second think that walks the G1 around because `hold` was implemented as zero command.
- Claiming a punch *hit* when E3’s residual is 13.5 cm.
- Framing the README as “robot plays Beat Saber / Superhot” instead of whole-body control on an OpenXR task.

---

## 3. Constraints

**Machine (measured).** NixOS laptop, RTX 4060 8 GB (driver 610.57.04), Hyprland/Wayland (`DISPLAY=:0`, `WAYLAND_DISPLAY=wayland-1`). Steam + Proton + gamescope already configured. Monado is Beyley’s PSVR2 fork, store path `/nix/store/89h2h18iz10aad69jpa2wvxmlhlg6k5i-monado-psvr2-psvr2-work-steamvr-2026-06-04`, runtime JSON `/run/current-system/sw/share/openxr/1/openxr_monado.json`. xrizer is the OpenVR translation layer.

**Do not mutate.** Do not write `~/.config/monado/config_v0.json`. Do not change the installed `monado.service` `Environment=`. Use a transient `systemd-run --user` unit (`P_OVERRIDE_ACTIVE_CONFIG=remote`). Restore `monado.socket` when a spike ends.

**Installed unit is the wrong unit for this project.** It sets `XRT_COMPOSITOR_COMPUTE=true` (SIGSEGV on this Wayland/NVIDIA presentation path) and `VK_DRIVER_FILES=…/nvidia_icd.x86_64.json` (file does not exist). Real ICD: `/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json`. Working compositor: `XRT_COMPOSITOR_COMPUTE=false` (GRAPHICS queue). Debug-GUI mirror is dead.

**Honest body.** After every `mj_step`, packet poses come from `mjData` body/geom poses, converted MuJoCo Z-up/`wxyz` → OpenXR Y-up/`xyzw`. Conversion is the E3 one (see `spike/e3/body_bridge.py`). Packet vs `mjData` position error was **0.0 m**; yaw error **≤ 0.0108°**.

**G1 (Menagerie `unitree_g1/scene.xml`, commit `cda5236f`).** 29 actuated joints, **no neck**. HMD orientation = torso orientation. Head position = `head_link` mesh geom. Playground `g1_policy.onnx` is **not yet in the loop**. E3 stood on Menagerie position actuators: base height 0.7905–0.7916 m, ≤ 3.08 cm horizontal in 20 s. The Playground policy at zero command drifts **0.475 m in 5 s** (`stubborn-gamer/SEED.md`). Those are different regimes. Do not treat E3 stand as `hold` once the ONNX is loaded.

**Brain latency (measured, stubborn-gamer).** MiniCPM5-2B ~20 s p50; MiMo-9B ~5 s p50; LFM ~0.21 s but degenerate (73% `navigate_to`). Default: **CPU llama.cpp**, MiniCPM5-2B first. LFM is disqualified as a commander.

**Claim tags.** Product docs and README inherit stubborn-gamer discipline: **MEASURED** / **CITED** / **ASSUMPTION**. Do not present E3’s stand drift as the policy’s drift.

**Language.** Python for v1. No C++ until a measured reason (packet rate, compositor, etc.).

---

## 4. What already exists (throwaway, keep as seed)

| Spike | Verdict | Keep |
| --- | --- | --- |
| E1 | OpenXR `xrLocateViews` follows TCP pose. Wire: 376-byte `mndrmt3`, port 4242, one client, last pose freezes on disconnect. | `spike/e1/remote_pose_client.py` structs |
| E2 | Pixels on RTX 4060. Capture: grim on the Monado window, **36.9 fps**. Pose→pixel **30–59 ms** (mid ~44 ms), including ~27 ms grim overhead. | Transient unit env, ICD path, grim geometry |
| E3 | Simulated G1 drives the same client. 1001 packets / 20.02 s at 50 Hz. Reach residual **4.3 cm**, punch **13.5 cm**. DLS on 7 arm DoFs/side, 2 rad/s rate limit. mink installed, unused. Trigger bit follows punch *phase*. | `spike/e3/body_bridge.py`, vendor Menagerie, `.venv` recipe, frame conversion, honesty CSV |

Spikes stay labeled throwaway until a later request promotes them. The production loop should reuse the protocol, conversion, and compositor recipe rather than rewrite them from memory.

---

## 5. Architecture

Four layers. Game verbs never enter the executor. OpenXR never sees a desired pose.

```
brain (local VLM, 5–20 s, async, CPU)
   └─ skill adapter (10–30 Hz)          # Superhot is adapter #1, not the schema
        └─ BodyCommand (task space)     # general robotics interface
             └─ executor (50 Hz)        # IK / stand / later locomotion ONNX
                  └─ JointTarget (29 floats)
                       └─ PD (500 Hz) → mj_step → achieved pose
                            └─ OpenXR packet (mndrmt3 :4242)
                                 └─ Monado remote + xrizer + game
                                      └─ grim window capture → still + digest → brain
```

Rates:

| Loop | Rate | Owns | Must not own |
| --- | --- | --- | --- |
| PD / physics | 500 Hz (`sim_dt` 0.002) | `mj_step`, joint tracking | goals, OpenXR |
| Executor + packets | 50 Hz | IK/policy → `ctrl`; packets from **achieved** poses | Superhot verbs, VLM tokens |
| Adapter | 10–30 Hz (interpolated at 50 Hz) | mask + targets + duration | joint angles, OpenXR poses |
| Brain | 5–20 s | one command onto the bus | motion, mid-swing steering |

The VLM never blocks the 50 Hz loop. Invalid or stale commands become `hold`. Ballistic skills (`duration_ms > 0`, `preempt=False`) run to termination; new commands queue.

---

## 6. Native command contract

This is the robotics interface. Schema name: `clank.body/1`. JSON on a Unix socket or a thread-safe latest-wins queue (builder chooses the transport; the fields are fixed).

Authority is a bitmask. Unset fields are **ignored, not zeroed**. That is how “mode” is expressed. There is no verb enum below the adapter.

```
AUTH_ROOT_VEL    = 1 << 0   # v_xy, yaw_rate
AUTH_ROOT_HEIGHT = 1 << 1
AUTH_HEAD        = 1 << 2
AUTH_L_HAND      = 1 << 3
AUTH_R_HAND      = 1 << 4
AUTH_CLOSURE     = 1 << 5   # 0 open .. 1 closed; grip/squeeze, latched per hand
AUTH_WAIST       = 1 << 6   # yaw, pitch, roll (bounded)
AUTH_GAIT        = 1 << 7   # unused in v1; executor reports UNSUPPORTED
AUTH_POSTURE     = 1 << 8   # unused in v1 except stand default
AUTH_TRIGGER     = 1 << 9   # 0 released .. 1 pressed; momentary, its own channel
```

Grip and trigger are separate channels because they have different shapes: a grip
is held while a trigger is pressed, and Superhot needs both at once (hold a gun,
pull its trigger). `KNOWN_AUTH = (1 << 10) - 1`.

Semantics, measured: the grip **latches** - a command that does not mention it
leaves it where it was, like a physical button - and `release` is how it opens. The
trigger stays pressed for `FIRE_HOLD_TICKS` and then releases by itself, because a
latched trigger fires once per session.

**`BodyCommand`** (writer-agnostic):

- `schema`, `seq`, `t_mono_ns`, `ttl_ms` (default 200 for continuous; ballistic uses `duration_ms`), `source` ∈ {human, vlm, planner, script}
- `authority`, `frame` ∈ {root, world} — default **root**
- Base: `v_xy` (m/s), `yaw_rate` (rad/s), `height` (m or null = executor default)
- Upper: `head`, `left_hand`, `right_hand` as `{p[3], q_xyzw[4]}` root-relative metres; `closure` (L,R); `trigger` (L,R); `waist` (yaw, pitch, roll) or null
- Optional: `gait`, `posture` — v1 executor may ignore and set `UNSUPPORTED`
- Execution: `priority`, `preempt` (false = queue behind ballistic), `duration_ms` (0 = until preempted), `done_when` ∈ {duration, target, external}, `tolerance_m` (default 0.02), `max_joint_speed` (default **2.0 rad/s**, E3), `max_hand_speed` (default 1.0 m/s)

**`BodyState`** (feedback; achieved only):

- `seq_ack`, `t_mono_ns`
- `root_p`, `root_q`, `v_xy`, `yaw_rate`
- `q[29]`, `qd[29]`
- achieved `head`, `left_hand`, `right_hand` (same Pose type, **after** `mj_step` + conversion as used internally; document whether this struct is MuJoCo-frame or OpenXR-frame — **call: MuJoCo world for state, convert only at the packet boundary**)
- `residual_head_m`, `residual_left_m`, `residual_right_m`
- `flags`: SATURATED, TTL_EXPIRED, FALL_RISK, UNSUPPORTED
- `supported`: authority bits this executor can honour *now*

**Validation (strict, cheap).** Wrong schema, non-finite numbers, unknown bits, workspace box exceeded, joint/velocity limits, TTL expired on arrival → reject, execute `hold`, increment an invalid counter into the next brain digest. The executor owns feasibility: clamp, rate-limit, solve, report SATURATED / FALL_RISK. The brain must not assume the request was achieved.

**TTL vs ballistic.** TTL gates *acceptance*. Once a non-preemptible skill is accepted, it runs to `done_when`. Continuous commands (`duration_ms=0`) are latest-wins sample-and-hold.

**World reconciliation (when locomotion exists).** Thumbstick in the OpenXR packet is filled from *achieved* base velocity (Playground gain ~1.03), not from the desired walk command. Until the ONNX is in the loop, thumbstick is zero and `AUTH_ROOT_VEL` is `UNSUPPORTED`.

---

## 7. Executor (System 1, not a new model)

v1 System 1 is **two measured controllers and no training**: a learned whole-body controller for locomotion and balance, and a classical QP for upper-body precision. Do **not** train Helix/π0/ACT/HOVER. Rewritten 2026-10-23 from `research/wbc-policy-proposal.md` and `research/wbc-classical-proposal.md`.

| Subsystem | v1 | Status |
| --- | --- | --- |
| **Locomotion + balance** | **GEAR-SONIC default** (`nvidia/GEAR-SONIC`; encoder 1762→64, decoder 994→29; 50 Hz policy, 500 Hz PD). **60 s walk: 39.79 m at 0.663 m/s, yaw drift −2.1°, tilt 9.5°, no fall, path error 0.23 m mean over 40 m.** 10 s walk: 6.36 m at 0.64 m/s. Dodge **0.15–0.18 m in < 0.5 s**. **6.8 ms per tick CPU**, ~0.9× realtime. **Hold is motion-dependent — see the next row.** | **MEASURED** (C3i + 3-variant sweep; default recommended) |
| **Hold** | **RMS velocity 1.3×10⁻⁴ m/s once settled — 0.02% of walking speed.** Recipe: **switch the reference to a standing clip (`squat_001__A359` frame 0) and hold the anchor index there** — an *anchor-only* freeze on a standing clip (1.9×10⁻⁴), **not** a flat horizon (2.3×10⁻²). A one-time settle of 8–12 cm that RMS does not capture, and entering from a walk costs a **~3–5 s transient at 0.13–0.18 m/s**. | **MEASURED** (C3m) |
| **Upper-body precision** | **mink floating-base QP** (`spike/wbc_classical/wbc_spike.py`; 50 Hz, 10×2 ms substeps, `daqp`). Hold **0.22 mm base / 20 s** (head 1.4 mm). Look **±35°** with **0.63°** yaw error and **3.3 mm** base. Reach single-arm **2.2–12.9 mm**, both arms **17.5–28.8 mm**. Aim pointing **0.18–2.34°** single, 1.05–6.26° both. Punch **9.5 mm / 3.6°** (E3's DLS punch was 13.5 cm). Dodge **12.1–14.7 cm** but **0.62–0.86 s** to peak. Solve **0.51–1.08 ms** mean, 2.44 ms p95 | **MEASURED** |
| Waist / look | SONIC couples head yaw → torso yaw at **~0.83 gain, saturating ~27–31°** — the virtual neck works natively. The QP instead commands waist yaw directly (±2.618 rad available; roll/pitch ±0.52 rad). E3 demonstrated ±35° | MEASURED |
| **Arms while walking** | **SONIC's own arms, and they are the answer — the QP override failed.** Walking at a fixed aim point: **3.51 cm / 7.22°** straight ahead (max 7.19 cm), degrading to **8.63 cm / 28.3°** at 30° off-axis. Holding: 1.14 cm / 7.33°. A hard QP override made it **4× worse** (14.40 cm holding, 12.84 cm walking) and integral correction **tips the robot** (65–75° tilt). Only a **wrist-orientation-only override** helps: 7.22° → **5.08°** pointing, position unchanged. | **MEASURED** (C3j, 16 variants) |
| Hands | No finger DoF. `closure` maps to controller `trigger_value` / `squeeze_value` in the packet, only while the *achieved* pose is in the strike/aim phase | `scene_with_hands.xml` only if grab becomes a goal |
| IK rate | QP at 50 Hz, PD at sim rate. Never IK inside the 500 Hz loop | same |

**The QP-on-SONIC hybrid is dead — measured, not argued.** C3j tested exactly the override that row used to propose: hard, rate-limited (2 and 6 rad/s), blended (50%, 75%) and orientation-only, on both hold and walking targets. **Every full override made the arms worse** — 1.14 → 14.40 cm holding, 3.51 → 12.84 cm walking — and integral correction fell at 65–75° tilt in every case. `arm_track_err_final ≈ 0.23` says the IK never converged toward its own target: the policy produces its own arm motion and a joint-space override fights it rather than commanding it. This is the **same mechanism as C3f**, just non-fatal to balance. **Do not reintroduce a QP arm override.** A wrist-orientation-only override is safe and buys 7.2° → 5.1° pointing; keep it as a refinement, not as the aiming solution.

**Known gaps in SONIC's interface.** Head position tracks **y only** (0.15–0.18 m for a 0.20–0.30 m command); **z and x are not tracked**, and **head pitch/roll produce no response**. The G1 has no neck, so looking up or down must come from the waist, which SONIC does not expose. Eye-level targets are fine; anything above or below is not. Wrist orientation error is **17–32°** on uncommanded axes, and the arms float ~**0.15 m** from the motion's own VR points at rest.

**Reach envelope is the other hard limit.** A probe at the game's actual pedestal distance (0.70 m forward, eye level) leaves the wrist **19.7–27.3 cm short** with the same QP and default weights. The measured reliable envelope is ~0.20 m. Anything the game wants touched at arm's length needs a **step**, not a longer reach — see §12 and `research/wbc-game-proposal.md`.

**Root frame.** Command poses are root-relative in a **gravity-aligned yaw frame** (heading from the pelvis yaw, z up). `BodyState` head/hands/`root_p` are **MuJoCo world**. Convert to root-relative at the adapter/executor boundary; convert to OpenXR only at the packet.

**Unsupported authority bits.** The executor **ignores** them (does not reject the whole command). It reports them in `supported` as unset and may set `UNSUPPORTED` on `flags`. Adapters should mask using the last `supported` value. `AUTH_ROOT_VEL` on a stand-PD executor is therefore a no-op, not a validator error.

**No mode switch is needed.** C3d showed the *Playground* policy could not be handed from walking to a static pose. SONIC does not have that problem: walk → hold → walk passes at both 4 s and 10 s holds. So the body is **one always-running controller**, and `hold` is a behaviour that controller already has, not a second controller or a frozen open-loop pose. Do not reintroduce a static `stand` ctrl in the app. (The 5-cycle soak and a 60 s walk are still unrun — that is C3i, and it is what makes this sentence safe to repeat.)

**`hold` (load-bearing, v1) = policy at zero velocity with station-keeping.** Not a frozen pose. C3d proved a static pose cannot be entered from walking; C3b/C3e proved zero command still walks. With the P-only heading loop engaged at zero command: **1.344 m creep / 20 s (0.067 m/s), heading held to 2.49°** on the Playground model. That is **9% of walking speed** (0.743 m/s).

Consequence for the freeze: **a think costs wall-clock, and only wall-clock.** Job Simulator does not scale time with player motion, so the arithmetic that used to sit here (a 20 s think at a 0.005 floor costs ≈0.1 s of game time) is deleted with the game. The body bar is unchanged: creep must be low enough that a 20 s think does not move a held object out of place or drift the wrists out of the reach cone. Keep the < 2 cm figure as the stretch target for C3g's integral action. Whether a slow turn is *safe* in Job Simulator is `[ASSUMPTION: genre; unmeasured]`; the cheap test is two stills 20 s apart in freeze-hold, diffed.

**`look`.** Clamp `waist_yaw` to **±0.6 rad**. Request the remainder as `yaw_rate`. If `AUTH_ROOT_VEL` is unsupported, drop the remainder, set `SATURATED`, and accept a **±35° v1 look arc**. That is allowed for Superhot v1. Do not pretend a 180° turn.

**Effect check after every ballistic skill.** Wrist travel vs 0.151 m shoulder-pitch authority; IK residual; whether Superhot time advanced during an intended hold (frame-diff); upright flag. Compact result goes to the next brain prompt. Same skill retry **once**, then something else or hold. No punch loop.

---

## 8. Adapters

Adapters compile a named intent into `BodyCommand`. They are the only place Superhot verbs exist.

| Adapter | BodyCommand | v1 status |
| --- | --- | --- |
| `hold` | `ROOT_VEL + ROOT_HEIGHT + HEAD + L_HAND + R_HAND + WAIST`; v=0; poses = last achieved | **required** |
| `look(yaw)` | `HEAD + WAIST + ROOT_VEL`; waist-yaw split | **required** |
| `reach(side, p)` | one hand + hold the other; `done_when=target` or duration ~2–3 s | **required** |
| `strike(side, p)` | one hand + waist/head; duration ~2 s (E3 punch phase); `preempt=False`; trigger/closure follows achieved phase | **required** after reach; do not name it “hit” |
| `move(v_xy)` | `ROOT_VEL`; policy stays running, outer loop off | **enabled.** C3e: walks straight (+0.23° / 10 s) with **no** outer loop on the Playground model. Arm authority while moving is pending C3f. |
| `dodge` | — | **disabled**; `UNSUPPORTED` |
| `trigger` | not a body skill; a momentary press on its own channel (`AUTH_TRIGGER`) | mapped from aim/strike |

A cheap 2D tracker (KCF-class on the grim crop) **may** update the hand target during `reach`/`strike` at 10–30 Hz. The VLM does not. If the tracker is not written yet, scripted/root-relative offsets (E3 `REACH` / `PUNCH` local vectors) are acceptable.

---

## 9. Brain

- **Input:** one grim still + a ≤10-line text digest (`skill`, `supported` bits, residuals, drift estimate, last outcome, optional bbox). Pixels-only planning measured 2.1× worse in stubborn-gamer; do not omit the digest.
- **Output:** skill + side only (`hold` | `look` | `reach` | `strike` + `left`/`right` where needed) plus optional `ttl_ms`. No 3D point. The adapter supplies E3 local offsets (`REACH` / `PUNCH`) as `p`. README claim: “named a skill,” not “chose a 3D target.” 2D→3D is out of v1. The VLM does not emit joint angles or OpenXR poses.
- **Where it runs:** CPU only, no cloud API. **The model is unresolved — E5 in §14 decides it.** MiniCPM5-2B was the default, but nothing ever compared candidates *on this layer*: it was chosen after `LFM2.5-VL-3B` (0.21 s p50) was struck out for a **planner** failure that does not apply to skill selection. See §18.8.
- **What the job actually is:** classification, not planning — a five-way `choice` plus a side. So the natural executor is a **closed-vocabulary decision head**, not a generative VLM. `[MEASURED 2026-10-28]` d1-2B over a **text digest** costs **5.5 s** (Q8_0/2.2B, 12 threads; a third party got **1.7 s** on Q4_K_M/4 threads), while d1 over an **image** costs **7 s at 116 image tokens and 36–39 s at 591** — the pixel path is 5–7× too slow on this CPU, which is an independent confirmation that the digest is the right interface (§9 input, not an optimisation).
- **When it runs:** only in `hold` (or after a ballistic terminates). Superhot’s clock makes seconds legal *if hold holds*. The architecture must still be correct if the world does not pause.
- **Not in v1:** streaming tokens into motors; visual-servoing off VLM tokens; GPU residency during a captured Superhot run.

A scripted commander (E3 PHASES, then a JSON file / socket writer) is the first writer. The VLM is a drop-in writer on the same bus. **The pipe must work with the scripted writer before the VLM is plugged in.**

---

## 10. OpenXR pipe

- Protocol: Monado `remote`, magic `mndrmt3\0`, 376-byte `r_remote_data`, TCP 4242, handshake = two packets then client streams. Hub binds `0.0.0.0`, **one client**. Disconnect freezes last pose.
- Transient unit env (copy from E2/E3): `IPC_EXIT_ON_DISCONNECT=off`, `XRT_NO_STDIN=1`, `P_OVERRIDE_ACTIVE_CONFIG=remote`, `XRT_COMPOSITOR_COMPUTE=false`, `VK_DRIVER_FILES=/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json`, binary `/run/wrappers/bin/monado-service`.
- `hello_xr`: `/nix/store/4z8ibn91k9r6yismm2nrcnl19b1v9rp7-openxr-loader-1.1.54/bin/hello_xr -g Vulkan2` with stdin held open. `XR_RUNTIME_JSON` = the monado json above.
- **Measured 2026-10-28 — the xrizer assumption was wrong.** Superhot ships `UnityOpenXR.dll` and reaches Monado **directly**; there is no xrizer in the proven path, and no `XR_RUNTIME_JSON` / `PRESSURE_VESSEL_IMPORT_OPENXR_1_RUNTIMES` is needed. Under Wine the chain is: prefix registry `ActiveRuntime = C:\openxr\wineopenxr64.json` → `wineopenxr.dll` → host OpenXR loader → Monado. The host loader finds the runtime through **`~/.config/openxr/1/active_runtime.json`** — `/etc/xdg` is invisible inside pressure-vessel and `~/.local/share/openxr/1/` is not a searched path — and it **caches the runtime list per process**, so a game restart is mandatory after creating that file. Full detail: `research/c0-results.md`.
- HMD pose is the achieved `head_link` (**~1.221 m**, not a hidden 1.6 m eye). Document that height. Prefer OpenXR **STAGE** with floor = MuJoCo ground; measure yaw/height bias against Monado `initial_offset` in C0. If Superhot is unplayable at child height, add a **named** `hmd_z_offset` (default 0). Never smuggle scale into the honesty claim.
- Do not start SteamVR, do not write a custom OpenVR driver, unless Superhot refuses the synthetic HMD. Then stop and re-plan; that is the only allowed game switch.

---

## 11. Capture (the robot eye)

Working path: grim the Monado / game window (`class: openxr` for hello_xr). ≥10 Hz is the bar; E2 measured 36.9 fps. Record pose→pixel when changing the capture path.

Dead paths (do not retry as the primary): `XRT_DEBUG_GUI=1`, compute compositor.

The brain gets **stills during hold**, not a 30 Hz video into the VLM. The tracker, if present, may see the stream.

---

## 12. Job Simulator milestone (E4)

Rewritten 2026-10-31. What survives from the Superhot version of this section is its measurement discipline, which is game-agnostic: packets == achieved `mjData`, grip logged as an input, no fake poses.

- Title: Job Simulator (Steam 448280), Proton, offline. **Nothing in this section is measured yet.** Every Job Simulator fact below is `[OPERATOR]` or `[CITED]`, and the milestone re-earns them with a C0-JS run.
- **First target: the museum hub.** JobBot speaks; take the business card; take a cartridge from the tray; take the **store clerk** cartridge; insert it; pull the lever labelled *pull to job* `[CITED: https://jobsimulator.fandom.com/wiki/JobBot]`. The brain spec treats this as the first proof (`docs/superpowers/specs/2026-10-31-brain.md` §6.2, §16). In-range cues `[OPERATOR 2026-11-01]`: grabbables light **blue** when the hand is close enough to grab; the cartridge **slot is painted blue**; a **yellow ghost cartridge** at the slot means the game will accept a release. Superhot's green slot is not this game.
- **Second target: Store Clerk, opening chores only.** Take a ticket; clear the counter (two cans); turn on the security camera by moving a hand near its power button `[OPERATOR]`.
- **There is no clock to be an ally.** Job Simulator does not scale time with player motion, so a 20 s think costs 20 s of wall-clock and nothing else. Superhot's `minTimescale = 0.005` is kept as **provenance only** (`research/wbc-game-proposal.md`), not as a licence for this design. Run the brain only in freeze-hold — because a slow call must not let the wrists drift, not because time is stopped.
- **Walking is not on the v1 critical path.** Demoted to v2 on 2026-10-31; Job Simulator needs a turn to face a counter and a reach across it, not a dodge (§18.2). `move` stays implemented and measured; it is simply not required here.
- **Pass:** any spoken chore completed on video, the game view and the sim body both visible, the CSV proves packets == achieved `mjData`, and grip is logged as an input.
- **Allowed fail:** robot in the level, flailing. That is still the video if honesty holds. Also allowed: reaching the wrong object; dropping the cartridge; never finding the camera button; hearing the instruction and doing nothing.
- **Hard fail:** fake poses; using a game API or game memory in the motor path; a mouse/keyboard hack; claiming a chore completed when the transcript and pixels do not show it; substituting a scripted chore sequence for a brain decision.
- **Open game-side unknowns, all unmeasured:** does JobBot's speech reach a capturable audio stream; is the instruction board legible at 1.19 m eye height; does the store require locomotion between stations; how far is the counter from spawn, and is the cartridge slot inside the reachable set; which controller channel is *grab* on this build. Protocol: a C0-JS run before E4, in the shape of `spike/c0/`.

---

## 13. What you ship

1. Repo with this design, RESEARCH, spike results, and the production (or promoted-spike) loop.
2. README: two-minute version of §1–§2 plus one architecture diagram matching §5 (not the stale debug-GUI diagram in `RESEARCH.md`).
3. 60–90 s recording: sim view and/or Monado view, plus on-screen or burned-in numbers (packet rate, residual, hold drift, pose source = `mjData`).
4. Honesty CSV in the same shape as E3 (`mj_*` vs `packet_*`).

No git init/commit until asked. No NixOS module edits unless asked (the compositor/ICD bug in `psvr2-vr.nix` is out of scope; work around it with the transient unit).

---

## 14. Work order

Promote or rewrite against this contract, in order. Each step has a pass and an allowed fail. Do not skip honesty logging.

| Step | What | Pass | Allowed fail |
| --- | --- | --- | --- |
| C0 | Superhot under the existing Proton/Monado remote stack, scripted HMD yaw + trigger | View follows TCP pose; trigger/grip reaches the game; grim captures the *game* window | **PASS on the gate** (2026-10-24, `research/c0-results.md`). Session reached `FOCUSED` on `Monado: Remote HMD`, 58 runtime extensions, view tracked a ±35° sweep at **41.97 mean pixel change / 59.3% of pixels vs 4.74 / 4.0% static — 9× over scene animation**, tutorial reached, captures are real game content. Trigger, clock floor and The Fall geometry not yet measured. |
| C1 | `BodyCommand` / `BodyState` + validator + bus + scripted writer driving E3’s G1 | CSV proves packets == achieved poses; hold / look / reach work on hello_xr | IK still ugly |
| C2 | Active `hold` on the chosen executor, 20 s | packet translation < 2 cm and rotation < 2° over 20 s | **exceeded in sim**: the QP holds **0.22 mm**, SONIC holds **3.6 cm** — SONIC misses the 2 cm bar, the QP passes. Use the QP for hold when precision matters |
| C3i | **SONIC gate**: 5-cycle walk→hold→walk soak, 60 s walk, aim-while-walk residual ≤10 cm | no fall across 5 cycles; 60 s walk; residual ≤10 cm | **PASS on the walk (39.79 m / 60 s, 0.663 m/s, −2.1° yaw, no fall) and on aim (3.51 cm / 7.22° ahead). PASS on soak survival (5/5, 100 s, tilt 16.7°) but FAIL on the hold bar — the 10 s holds drift 0.23–2.07 m.** Live co-residency still unrun. |
| C3j | **Arm override on SONIC**: keep its 12 leg joints and waist, replace the 14 arm joints with QP IK toward the same VR target, while walking | wrist residual **< 3 cm while walking**, no fall, orientation reported | **FAIL (measured), hypothesis falsified.** Hard/rate/blend overrides all 4× worse in position; integral variants fall at 65–75° tilt. Orientation-only override helps (7.22° → 5.08°). The two-controller joint-override architecture is not achievable. |
| C3m | **Hold as a reference-motion switch, not a freeze**: walk → switch to a standing clip → switch back, 5 cycles | hold drift < 5 cm per cycle; resume re-locks; no fall | **PASS (measured).** Settled hold RMS **1.3×10⁻⁴ m/s** (0.02% of walking); 5/5 cycles, no fall, tilt ≤9.5°, cum net 0.478 m, walk resumes within 0.2 m. Misses the <5 cm bar only through an 8–12 cm one-time settle. **The earlier 0.23–2.07 m holds were an anchor-only freeze of a *walking* clip** — its 0.9 s lookahead still instructed the policy to walk. |
| C3k | **Unify the model**: SONIC runs its own MuJoCo model, the QP ran Menagerie `scene.xml`. Re-measure the QP on SONIC's model (or the reverse) | the QP's hold/reach/aim reproduce within the reported margins on SONIC's model | numbers do not transfer → the hybrid is not measurable; pick one model for the entire app |
| C3l | Classical walking, only if C3i fails: single-support falsification for a contact-force QP | 6 s one-foot hold, base excursion < 5 cm, foot lift ≥ 3 cm | tips as the IK-only QP did (1.3–1.6 s) → classical walking is not a one-week project; report it |
| C3a | Policy drives legs; arms were meant to stay on IK | Port reproduces 0.515 m/s and 0.471 m / 5 s | **FAIL (measured)** — policy curves +73.6°/5 s; arms-on-IK falls at 25.6 s |
| C3b | Policy at zero command, 20 s, vs the 2 cm freeze budget | drift of the same order as C2 | **FAIL (measured)** — 1.736 m base / 1.727 m head in 20 s. The policy is not a hold mode. |
| C3c | Waist ownership | one option measured and documented | **A kept.** B gives +34.3° look but 0.093 m/s and falls at 2.5 s with arm IK |
| C3d | Hand the body from the policy to a static stand pose and back | 5 clean transitions, no fall, position recovered | **FAIL (measured)** — every walk→park falls in 0.64–1.49 s across ramps 0.05–2.0 s, a 2–3 s brake, and a `knees_bent` pose. The policy has no stop; the static pose has no balance feedback. Two-mode body abandoned. |
| C3e | Cause of the curve; whether closed-loop cancels it | Playground XML walks straight, **or** feedback holds heading and drift | **PASS on the model fix** (Menagerie +174.5°/10 s vs Playground **+0.23°/10 s**, same ONNX). **ALLOWED-FAIL on the loop**: P-only bends Menagerie heading to 7.3° RMS at ky=2/kv=1 but 11/20 configs fall; on Playground the loop is unnecessary and adds 0.44 m lateral |
| C3f | **Arms while walking on the Playground model** — the C3a arm result was taken on the wrong XML and is void until re-run | neutral wrist hold and a moving reach both survive 30 s with no fall; report walk quality delta | falls again → the policy owns all 29 while moving and the body cannot aim while travelling |
| C3g | Station-keeping by closed loop with **integral** action at zero command (C3e-2 was P-only) | creep falls well below 1.344 m / 20 s; < 2 cm / 20 s is the stretch target | creep stays ~0.067 m/s → the freeze becomes the C0 question "does Superhot tolerate 9% speed" rather than a controller problem |
| C3h | Gain scheduling instead of mode switching: loop **off** while moving, **on** at zero command | straight walk *and* held heading in one schedule | scheduling itself destabilises → keep loop off, accept creep |
| C4 | Digest; scripted strike using E3 offsets (tracker optional) | residual logged; trigger from achieved phase | miss by 13 cm |
| C5 | VLM writer on CPU, same bus; skill+side only | one valid command per still; invalid → hold | bad choices |
| E5 | **Brain bake-off on the digest** — `LFM2.5-VL-3B`, `LFM3-1.7B`, `d1-2B` (text), MiniCPM5-2B, all on the same 100 labelled frames | beats the **majority-class** baseline *and* the scripted commander's labels; reports p50/p95 and a 5-bucket reliability table | the only measurement that can settle §9. Cost: digest-only calls, minutes of compute. Non-blocking for C0-3/5/6 |
| E4 | Superhot with the body | body in the level on video | no objective |

C0 may run in parallel with C1 and with C3i/C3j/C3k. Do not start C5 before C0 and C1. Do not start E4 before C0, C1, C2, C3i and C3j. **The Playground `g1_policy.onnx` is no longer on the critical path**; C3a–C3h stay in this table as the record of why (see §18.3–18.4).

---

## 15. Calls marked as the author’s (builder: do not relitigate unless measured wrong)

1. Python v1; uv + the E3 `.venv` recipe is an acceptable env (`mujoco==3.17.0`, mink available).
2. MiniCPM5-2B on CPU llama.cpp as first VLM; LFM out.
3. VLM emits skill+side only; adapter supplies E3 offsets as `p`.
4. `BodyState` poses are MuJoCo world; commands are root-relative in a gravity-aligned yaw frame; OpenXR conversion only at the packet.
5. Waist yaw cap ±0.6 rad until MJCF ranges are read. Look remainder dropped if `AUTH_ROOT_VEL` unsupported (`SATURATED`, ±35° v1 arc).
6. DLS first; mink only if we decide the 13.5 cm punch residual is the thing to spend time on.
7. **The simulation/collision model is Playground's `scene_mjx_feetonly_flat_terrain.xml`**, not Menagerie's `scene_mjx.xml`. Menagerie stays for visualisation/URDF parity. `move` is a velocity pass-through into a policy that is always running, not a mode switch. `dodge` and gait bits stay `UNSUPPORTED`.
8. Superhot is the only required game; grim is the eye; transient Monado unit as above.
9. No ROS2, no policy retrain, no force loop, no game API.
10. Spike code is a seed, not a package name. Production modules can live under `clank/` (or similar); builder picks names.
11. Public GitHub is the ship *target*; do not `git init` until asked.
12. C0 is the game-side go/no-go (synthetic HMD + trigger + grim on the game window), not E4.
13. HMD height: **≈1.170 m** in the policy pose (`knees_bent`), 1.221 m in the straight-legged `stand` — both measured. The policy pose is the one that matters, since the body never stands. STAGE, floor = sim ground. Named `hmd_z_offset` default 0; if C0 shows child height is unplayable, change it explicitly, never silently.
14. Unsupported authority bits are ignored, not a full reject.
15. **One controller, always running.** No park mode, no static stand pose entered from walking, no frozen open-loop pose in the app. "Hold" = zero velocity command plus station-keeping, with creep measured rather than assumed to be zero. Arm authority while moving is pending C3f.

---

## 16. Left to the builder

- JSON vs dataclass internally; socket vs in-process queue.
- Exact module layout, logging library, tests.
- Whether to promote `spike/e3/body_bridge.py` in place or wrap it.
- mink vs DLS switch criterion beyond “if we choose to spend time on residual.”
- Tracker implementation (or none, using E3 local offsets).
- llama.cpp invocation details (binary path, context, stop tokens).
- README prose and video editing.
- How to launch Superhot under Proton once C1–C2 pass (use the existing Steam/Proton stack; do not invent a new runtime).

---

## 17. Success bar for the portfolio

Enough: a recording where a G1 in MuJoCo moves, hello_xr or Superhot’s view moves *because the sim moved*, a CSV that proves packet == `mjData`, and a README that says the VLM named a skill while IK missed by 13 cm.

Not enough: a VLM demo on a game screenshot with a mouse; a Superhot run driven by a sine-wave HMD; a claim of whole-body tracking after bolting unbounded IK onto the walk policy.

The phrase “balance-constrained locomotion” requires **both** legs in the loop (C3a) **and** station-keeping that actually holds (C3b). **Both have now passed (2026-10-24).** Legs in the loop: **39.79 m in 60 s at 0.663 m/s, yaw −2.10°, no fall** (C3i-2). Station-keeping that holds: **RMS 1.3×10⁻⁴ m/s settled, 5/5 clean walk→hold→walk cycles** (C3m). The project may therefore claim **balance-constrained locomotion**. What it may **not** claim yet: independent off-axis aiming (8.63 cm at 30°, §7), live co-residency beside the game (R6), or a cleared stage (R8).

---

## 18. Builder-check settlements (2026-10-23)

From `docs/2026-10-23-Clank-design-builder-check.md`.

Builder-check items 1, 3, 4, 5 (item 2 became §18.2):

- **Q1 — game path is untested.** Added **C0**: Proton/xrizer/Monado remote + scripted pose + trigger + grim on the game window, before C5/E4. Body work (C1–C2) may proceed in parallel. If C0 fails, switch game or driver; do not discover that at E4.
- **Q3 — hold / look / unsupported bits.** Settled in §7: gravity-aligned yaw root frame; ignore unsupported bits; v1 hold = stand PD + frozen last `q`; look clamps at ±0.6 rad; remainder dropped until the policy is in.
- **Q4 — height / space.** Raw `head_link` ~1.221 m in `stand`, STAGE, floor = sim ground, documented `hmd_z_offset` default 0. Re-measured after C3a because the policy's rest pose is `knees_bent`.
- **Q5 — VLM is a skill classifier.** Skill+side only; adapter owns `p` via E3 offsets.

**18.2 Reopened and re-closed 2026-10-31 — legs are off the v1 critical path.** The 2026-10-23 decision was: keep Superhot and add the walking policy rather than switching to a stationary game. Its entire rationale was “Superhot forces lateral movement — dodge bullets, reach cover, close distance.” **The game changed, so the rationale is gone.** Job Simulator needs a turn to face a counter and a reach across it, not a dodge.

Decision 2026-10-31: **walking is demoted to v2.** It is not needed for the Job Simulator milestone and it is not on the critical path. This is a scope reduction, not a capability retraction — the walk stays measured (§18.5; `research/sonic-gate-results.md`: 39.79 m in 60 s, 0.663 m/s, yaw −2.10°, no fall). `move` remains implemented and reachable; it is simply not required to complete the milestone. A fixed turret is still the thing to avoid, so the body turns and leans; it does not stand rigid.

Three consequences accepted, all measured-or-measured-later rather than assumed:

1. The policy's rest pose is `knees_bent`, not `stand`. The 1.221 m HMD height was taken in the straight-legged stand and must be re-measured; Superhot's scale question depends on it.
2. The policy outputs all 29 joints, including the three waist joints. So ±35° independent head-look may be lost. Overriding the waist for look while the policy balances is the documented failure mode on this robot (forward lean, endless stepping). Resolved by C3c, not by assumption.
3. Zero-command drift may destroy the Superhot freeze that makes a 5–20 s brain legal. Measured by C3b. If it persists, C3 has made the game harder, not easier, and the game choice reopens.

Re-fetch note: the policy (`g1_policy.onnx`, 903,543 B) and its reference script lived in `/tmp` and are gone. The recipe survives in `stubborn-gamer/bench/g1_policy_test.py`; `scene_mjx.xml` is vendored at `spike/e3/vendor/mujoco_menagerie/unitree_g1/`; `onnxruntime` is not yet in the E3 venv.

### 18.3 C3 outcome (2026-10-23) — legs measured, body is two-mode

Full detail in `research/c3-legs-results.md`. Policy source: official `mujoco_playground` ONNX, sha256 `db2eb258…`, 903,543 B; `onnxruntime 1.31.0` added to `spike/e3/.venv`.

- **The port is correct.** 0.515 vs 0.516 m/s; 0.471 vs 0.475 m / 5 s. The ONNX and the environment are not the problem.
- **The archived “walks at 0.516 m/s” was the x-component of an arc.** Path over 5 s is 3.558 m, heading rotates +73.6°, and a 30 s run circles a full 360°. Inherited confidence in this policy was overstated.
- **Arms cannot ride along.** Fall at 25.6 s walking with neutral arm IK; 3× creep at zero command; fall at 2.5 s with the waist override plus arms.
- **The policy is not a hold mode.** 1.736 m base / 1.727 m head drift in 20 s at zero command.
- **Waist: A.** The policy owns it. Option B’s +34.3° look costs 82% of forward speed and falls with arm IK.
- **HMD height 1.170 m** in the policy pose (1.221 m straight-legged).

Consequence: the freeze is **not** dead, but it is not what §18.2 assumed. Stand PD was supposed to be the freeze; C3d then proved a static stand pose cannot be entered from walking. The freeze therefore has to be **policy-level creep control**, not a frozen pose — see §18.4.

### 18.4 C3d / C3e outcome (2026-10-23) — model fixed, two-mode body dead

Full detail in `research/c3de-results.md`; harness `spike/c3/measure_c3de.py`; data `spike/c3/c3de_measurements.json`.

**C3e-1 PASS — the curve was our model.** Identical ONNX, identical kinematics/mass/timestep/joint order, two collision models:

| Model | heading / 10 s | forward | path speed | max tilt |
| --- | --- | --- | --- | --- |
| Playground `scene_mjx_feetonly_flat_terrain.xml` | **+0.23°** | 7.426 m (0.743 m/s) | 0.745 m/s | 4.8° |
| Menagerie `scene_mjx.xml` | **+174.5°** | −0.346 m | 0.518 m/s | 9.9° |

Differences that matter: 6 foot/leg colliders vs 25 box colliders; actuator damping term 0 vs −2; **ankle-roll gain 2 vs 20**; wrist gain 2 vs 20; solver iters 3/5 vs 5/8. Attribution is an ASSUMPTION (not decomposed), the A/B is not. **Adopt the Playground XML for policy evaluation.** This supersedes C3a’s “does not walk straight” — on the right model it does, and the two inherited errors (0.516 m/s was an arc; the arc was our XML) should both be stated in the README.

**C3e-2 ALLOWED-FAIL.** P-only loop: Menagerie heading bends to **7.3° RMS** at `ky=2, kv=1` but with **0.511 m lateral RMS** and **11/20 sweep configs falling**. On the Playground model the loop is unnecessary and harmful (adds 0.44 m lateral, drops 0.743 → 0.610 m/s). At zero command on Playground: **1.344 m creep / 20 s** (0.067 m/s), heading held to 2.49°. **Do not ship the loop as a straightness fix.**

**C3d FAIL — the two-mode body is abandoned.** `park → walk` is safe (initial park settles in 0.04 s at 0.1° tilt; a 2 s walk segment holds 5–6°). **Every `walk → park` falls** in 0.64–1.49 s: ramps 0.05–2.0 s, a 2 s brake (still moved 1.72 m), a 3 s brake (2.15 m), a slow 0.05 m/s approach (0.98 m), and a `knees_bent` static pose (0.64 s). Root cause: **the policy has no stop**, and a static `stand` ctrl has **no balance feedback** — it is only stable from its exact initial condition. Not a ramp bug.

**What replaces it.** The policy is always running; the outer loop is gain-scheduled (off while moving, on at zero command). Because a static pose is gone, “hold” means **zero velocity command plus station-keeping with measured creep**, not zero motion. Creep is 9% of walking speed; **what that costs is a wall-clock cost, not a game-clock cost** — Job Simulator does not scale time with player motion, so a 20 s think takes 20 s, and the open question is whether a held object visibly drifts.

**Open after C3d/C3e:** C3f (arms while walking, re-tested on the right XML — the old failure is void), C3g (integral action on station-keeping; the C3e-2 loop was P-only), C3h (gain scheduling). Also unresolved: C3c’s waist numbers were taken on the wrong model and need a re-test before "no independent look while walking" is treated as final. Honesty caveat carried forward: C3d’s reach-after-transition was never exercised, because the transition fell first — moot for now, since transitions are gone.

### 18.5 WBC routes (2026-10-23) — three parallel research routes, and §7 was rewritten from them

Three agents ran against `docs/2026-10-23-Clank-state-of-findings.md` with deliberately non-overlapping mandates: acquire a learned controller, build a classical one, and pin down the game-side contract. Deliverables: `research/wbc-policy-proposal.md`, `research/wbc-classical-proposal.md`, `research/wbc-game-proposal.md`.

**Route A — a released whole-body controller exists and it works.** GEAR-SONIC default, headless in MuJoCo on this laptop: **6.36 m in 10 s** at 0.64 m/s, **hold 3.6 cm / 20 s**, **walk → hold → walk at 4 s and 10 s without falling**, dodge **0.15–0.18 m < 0.5 s**, **6.8 ms/tick CPU**. Its native teleop interface is head + two hands, and head yaw drives torso yaw at **~0.83 gain**. This is the first controller in the whole project that walks, holds and transitions at the same time. C3 was not wrong about the Playground ONNX; it was wrong to generalise from it. Limits: wrist **4–6 cm / 17–23°** in hold and **5–10 cm / 8–19°** walking; head **y**, only; no head z/x, no pitch/roll. Weights are the **NVIDIA Open Model License**, not Apache — attribute it.

**Route B — the precision half is classical, and it is excellent.** A mink floating-base QP: **hold 0.22 mm / 20 s**, look **±35°** at **0.63°** yaw error and **3.3 mm** base, reach single-arm **2.2–12.9 mm**, both arms **17.5–28.8 mm**, aim **0.18–2.34°** single, punch **9.5 mm / 3.6°** against E3's **13.5 cm**, solve **0.51–1.08 ms** mean. It cannot walk and the reason is structural, not tuning: sustained lateral weight shift is limited to **0.10 m** (0.12 m tips) and single support tips in **1.3–1.6 s**, because the formulation has no contact-force or CoP variable. A new probe also establishes the **reach envelope: 0.70 m forward at eye level leaves the wrist 19.7–27.3 cm short.**

**Route C — the game-side contract was measured, and it is now provenance.** These are Superhot's serialized-asset values, retained because the *method* was the cheapest good measurement in the project (read the game's own constants where they are readable), not because they constrain this design: `TimeControl.minTimescale = **0.005**` in levels 0–5, so that game's floor was **0.5% speed, not a pause**; The Fall's first enemy **1.41 m**, pistol and exit pyramid **0.70 m** at eye level; `Bullet.COLLISION_RADIUS` **0.03 m**, `distanceCheckTimeTreshold` **2.5 m**, `maxHandsDistance` **2.0 m**; Corridor needs **12.31 m** of traversal. Job Simulator has no time scaling and none of these are re-inherited as requirements.

**What changed in this document.** §7 was rewritten: the executor is **SONIC for locomotion/balance + the QP for upper-body precision**, with no training on the critical path. §12 now carries The Fall's geometry and the clock floor. The Playground `g1_policy.onnx` is off the critical path entirely.

**What is NOT settled, stated plainly.** (1) **Arms while walking is still unowned** — SONIC's arms cannot aim, the QP cannot walk, and the override has not been tried (C3j). (2) The two controllers run on **different models** (SONIC's own vs Menagerie) — every hybrid number is approximate until C3k unifies them. (3) Nothing has been measured **live**; both controllers are headless. (4) **C0 is still unrun**, and without it none of this matters. (5) SONIC's 5-cycle soak and 60 s walk are unrun.

**Resolved 2026-10-24 — every item above is now measured or answered.** (1) Arms while walking: the QP-on-SONIC override was **tried and failed** (C3j, 16 variants — 12.84 cm / 22.40° against a 3.51 cm / 7.22° baseline); only a **wrist-orientation-only** override survives, 7.22° → 5.08°. (2) Model unification (C3k) is **moot** — the two controllers are no longer composed. (3) Live co-residency is **still open** (R6). (4) **C0 passed**: Superhot holds a focused session on the fabricated headset and its view tracks our poses (`research/c0-results.md`). (5) The 5-cycle soak and 60 s walk **ran**: 5/5 cycles with no fall, **39.79 m in 60 s**. Hold was re-measured (C3m) at **RMS 1.3×10⁻⁴ m/s** with a standing-clip anchor-index freeze — see §18.7.

**Anti-lesson worth keeping.** Every time this project trusted an inherited number it was wrong, and every time it measured the cheap thing the picture changed: the archived speed was an arc, the arc was our XML, and the "freeze is impossible" claim dissolved into a 0.005 constant sitting in a serialized asset. Measure the constant before designing around it.

### 18.6 SONIC gate and arm-override (2026-10-23) — walking is real, the hybrid is dead

Full detail in `research/sonic-gate-results.md`; scripts `spike/wbc/gate_c3i.py`, `spike/wbc/arm_override_c3j.py`; data `spike/wbc/logs/{c3i,c3j}_summary.json`. C3k (model unification) was not attempted.

**C3i — the walking half of the goal is real.** 60 s continuous walk: **39.79 m at 0.663 m/s**, yaw drift **−2.1°**, max tilt **9.5°**, path error 0.23 m mean over 40 m, no fall. That is the strongest locomotion result this project has. Five walk→hold→walk cycles survive 100 s with no fall, and **the walk resumes correctly every time** (resume distance 2.83–3.06 m against a 3.10–3.15 m reference) — so C3h's transition problem is solved, by a controller that simply has a stand.

**C3i — but the hold bar fails, and the reason matters.** The 10 s holds drifted **0.23, 0.50, 0.28, 1.82, 2.07 m**, cumulative 4.89 m, tilt to 16.7°. Cause: this soak froze a **walking** clip mid-stride, whereas the 3.6 cm figure in §7 came from freezing a **standing** clip (`squat_001__A359` frame 0). Both are correct and they measure different things. The design doc had been quoting the flattering one. **Consequence: `hold` is a reference-motion switch, not a freeze** — that is C3m, and it is the cheapest remaining path to the freeze Superhot needs.

**C3i — aiming passes at the gate bar and is narrower than it sounds.** Fixed aim point, walking: **3.51 cm / 7.22°** straight ahead (max 7.19 cm), **8.63 cm / 28.3°** at 30° off-axis. At the game's 1.41 m enemy distance, 7.2° is ≈17 cm of pointing error. The body aims in the direction it walks and poorly to the side. Holding is *worse* than walking (1.14 cm but 7.33°), which is backwards from intuition and consistent with a motion-tracking controller.

**C3j — the QP-on-SONIC hybrid does not work, and this closes route 2 from §18.5.** Hard override, 2 and 6 rad/s rate limits, 50% and 75% blends, orientation-only, and integral variants were all tested on hold and walking targets. Every full override made the arms **4× worse in position** (1.14 → 14.40 cm holding; 3.51 → 12.84 cm walking) and ~3× worse in orientation. Integral correction **falls the robot** at 65–75° tilt in every configuration. `arm_track_err_final ≈ 0.23` — the IK never converged, because the policy emits its own arm motion and a joint-space override fights it rather than commanding it. **Same mechanism as C3f, non-fatal to balance.** The one useful variant is wrist-orientation-only: **7.22° → 5.08°** pointing with position unchanged. Keep that as a refinement; it is not the aiming solution.

**What this settles, and what it does not.** Settled: SONIC walks and transitions; the body aims straight ahead to ~3.5 cm / 7°; the two-controller joint-override architecture is dead; aiming off-axis is the body's weak axis. Not settled: whether a standing-clip hold gives a usable freeze (C3m); live 50 Hz co-residency; and **C0, still unrun**, without which none of it reaches the artifact.

### 18.7 C3m — the freeze exists, and the bad holds were our harness

Detail in `research/c3m-results.md`; script `spike/wbc/hold_c3m.py`; data `spike/wbc/logs/c3m_summary.json`. 4 m 04 s of compute, nothing re-run.

**The C3i soak's 0.23–2.07 m holds were a measurement bug.** SONIC's encoder consumes a **10-frame horizon at 20 ms spacing ≈ 0.9 s of lookahead**. The soak froze only the *anchor* frame index, so the next nine frames still said "here is the rest of your step", and a motion tracker tracked them. Same frozen pose, same frame, same clip:

| Freeze mode on the *same* walk-mid-stride pose | net / 20 s | RMS v (last 10 s) |
| --- | ---: | ---: |
| anchor index only | 3.928 m | 0.2005 m/s |
| whole 10-frame horizon frozen | 0.110 m | 0.0085 m/s |

36× less drift, 24× less motion, one mode change. Nothing about clip choice was involved, so §18.6's "the hold is motion-dependent, we were quoting the flattering clip" was the wrong diagnosis and is superseded.

**The freeze is real, and the correct recipe is the opposite of the obvious one.** Best measured hold: **RMS 1.3×10⁻⁴ m/s settled — 0.02% of walking speed**, p95 2×10⁻⁴. The Superhot inference that this “pins the clock at its 0.005 floor” is retired with the game; the number is a body fact and survives. The shipping form is **a switch to a standing clip (`squat_001__A359` frame 0) with an anchor-index freeze** (1.9×10⁻⁴), **not** a flat horizon (2.3×10⁻²). A flat horizon *rescues* a walking clip (0.2005 → 0.0085) and *degrades* a standing one (0.00019 → 0.0233), apparently by stripping the velocity/phase information the policy expects. The deciding factor is **whether the lookahead contains walking motion** — not which freeze mode you pick. Use a flat horizon only as a repair when you must freeze a walking clip in place.

**Entered from a walk it survives and resumes.** Five walk→hold→walk cycles, 100 s: no fall, worst tilt **9.5°**, cumulative hold **0.478 m**, walk resumes within **~0.2 m** of reference every cycle, largest switch-edge tilt **5.5°**. That is the transition C3d could not do (fell 0.64–1.49 s) and C3h could not do (fell 3.11–4.24 s).

**What a think costs.** Immediately after the switch the body is still moving at **0.13–0.18 m/s** for ~3–5 s; the per-cycle hold RMS of 0.050 is that transient averaged over a 10 s hold, not drift. So a 20 s think **begins with a 3–5 s settle** and only then is the body still. Think after the settle, or budget for it. (Superhot priced this in game time because the clock scaled with motion; Job Simulator makes it wall-clock, which downgrades the settle from a safety problem to a spectator one.)

**R2 status: met, with a stated caveat.** The <5 cm bar is missed only by a **one-time settle of 8–12 cm** while RMS falls to 10⁻⁴; it is not creep. Whether an 8 cm shift on stopping matters in game is a C0 observation.

**Known defect, recorded rather than hidden.** The 5-cycle test attempted 4 configs; 2 completed (`survived=True`, full per-cycle data) and one crashed in `place_region` (`quat_mul` on a `(T,14,4)` array), so **`hold_c3m.py` carries an unpatched bug** in that path. The two completed configs are enough for the conclusion; the missing ones are not load-bearing.

**Net effect on the goal.** Walk, dodge, hold and transition are all now measured on one controller: **39.79 m / 60 s walking, 0.15–0.18 m dodge, RMS 1.3×10⁻⁴ hold, 5/5 clean cycles.** Aim is the single weak axis (3.5 cm / 7.2° ahead, 28° at 30° off-axis). **C0 is the only remaining gate that can invalidate the whole thing**, and it is still unrun.


### 18.8 The brain choice was never earned (2026-10-28)

`MiniCPM5-2B` sat in this document as the default brain with no measurement behind it. The reasoning trail does not support it, and three separate pieces of our own work point elsewhere.

**The disqualification was borrowed from the wrong layer.** `stubborn-gamer/research/model-test-findings.md` concludes: *"Best compact fast candidate: LFM2.5-VL-3B, particularly for routine decisions."* Routine decision — pick one of a small closed set from a compact typed state — **is exactly this layer**. LFM was struck out in Clank's docs for emitting `navigate_to` on 73% of **hard home-assistant planning cases**, which is the layer above us. The bench's own caveat for LFM was *"not a proven general planner"*: true, and irrelevant to a five-way skill choice. What survived into these docs was "LFM disqualified", without the qualifier that mattered.

**The bench also says vision is the wrong axis:** *"Vision was not the main reasoner... a text/JSON belief state produced by perception and tracking is likely more valuable than spending most of the VRAM budget on a large vision model."* A **VL** model is therefore the wrong shape for the brain, and `LFM3-1.7B` (text, 256K context, native tool calling, ~1 GB at 4-bit) fits the contract better than either VL candidate.

**And the category is cheap now.** `research/d1-capabilities.md` §6b measured on this machine: d1-2B over a **text digest** at **5.5 s** (Q8_0/2.2B/12 threads; a third party measured **1.7 s** on Q4_K_M/d1-2B/4 threads), versus **7 s at 116 image tokens and 36–39 s at 591** for the same model over a picture. So the pixel path is 5–7× too slow, and the digest path is 7–13× faster than MiniCPM5-2B's 20 s p50. Answers were byte-identical across repeats, as a zero-token readout must be.

**Consequence.** §9 keeps the contract — digest in, skill+side out, CPU, no cloud — and loses the model name. **E5** decides it. Two cautions carry over and must survive any bake-off: a **fine-tuned 421M Laya beat zero-shot 2B d1** (99.9% vs 55.4%) on a narrow closed-vocabulary task, so the axis that decides is task-specific training rather than parameters; and **no calibration metric exists for d1 anywhere** (no ECE, no Brier), while `confidence`-gated `hold` is precisely a calibration claim. A majority-class baseline and a reliability table are part of E5's pass bar for that reason.
