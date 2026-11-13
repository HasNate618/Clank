# The Fall first: a standing-G1 demo against Superhot's measured opening

> **Status 2026-10-24 — two premises in this document are false, the game-side measurements are good.**
> Corrected: (1) **C0 has run and passed** (`research/c0-results.md`) — the game is installed and
> accepts synthetic HMD poses. (2) **xrizer is not in the proven path.** Superhot ships
> `UnityOpenXR.dll` and reaches a focused session directly through the prefix's
> `wineopenxr.dll` shim → host loader → Monado. The runtime pointer that made it work is
> **`~/.config/openxr/1/active_runtime.json`**, not an `XR_RUNTIME_JSON` launch option, and the
> launch **must** come from the Steam client UI (CLI `-applaunch` fails `AppError_9`). Proton was
> `cachyos-11.0-20260703-slr`, not GE-Proton. (3) The body can now **walk** (39.79 m / 60 s) and
> **hold** (RMS 1.3×10⁻⁴ m/s), so the "standing G1, no walking" framing is narrower than what is
> available. The level geometry below — The Fall's 1.41 m enemy, 0.70 m pistol and exit, the 0.005
> clock floor — is still the best information we have and is `MEASURED`.
**Date:** 2026-10-23
**Agent:** OpenCode `wbc-game`
**Question:** given the measured game-side facts and the measured classical WBC, which whole-body demo should be built first, and what would it prove?

**Inputs:** `../docs/archive/2026-10-23-clank-state-of-findings.md`, `../docs/2026-10-23-clank-design.md`, `research/wbc-classical-proposal.md`, `research/wbc-policy-proposal.md`, `/tmp/opencode/spawn_contract.txt`, `/tmp/opencode/level1_scan.txt`, `/tmp/opencode/level3_scan.txt`, `/tmp/opencode/level4_scan.txt`, `/tmp/opencode/gamedata.xml`, `/tmp/opencode/il2cpp/dump.cs`.

**Claim tags.** `[MEASURED]` = produced locally by a named script/run. `[CITED]` = prior local result or external source, named. `[ASSUMPTION]` = reasoning, untested. Day estimates are assumptions.

**Scope.** Research proposal. No game, Proton, Monado, NixOS, or existing spike was modified. The only new executable is `spike/wbc_game/reach_far_test.py`, a read-only probe against the existing WBC model.

---

## Verdict

**The first honest stage target is The Fall (story level 1, scene `CezaryVR_ReverseFall`), played standing.** It is the first early stage whose combat is authored at conversational distance: the first enemy is **1.41 m** in front of the player, and the pistol and exit pyramid are **0.70 m** forward at eye level `[MEASURED: level1 transforms]`. The measured classical WBC already supplies the standing half of the skill set — hold **0.22 mm** over 20 s, look **±35°** with **0.63°** yaw error, aim **0.18–2.34°** single-arm, punch **9.5 mm** `[MEASURED: research/wbc-classical-proposal.md]`. Superhot's serialized clock floor is **0.005** `[MEASURED: TimeControl in levels 0–5]`, so a 20 s VLM think at the floor costs **0.1 s** of game time `[ASSUMPTION: floor and movement mapping]`.

**The demo is conditional, not guaranteed.** A new probe shows the standing WBC cannot touch the 0.70 m pedestal: the final wrist residual is **27.3 cm** for the pistol and **19.7 cm** for the pyramid, with the base moving 3.4–6.7 cm `[MEASURED: spike/wbc_game/reach_far_test.py]`. The WBC's measured reach envelope is 0.20 m; the G1's arm is about 0.5 m. Therefore:

1. If the game **hands the pistol to the player** (the guide's "the gun you are given" `[CITED: readyvrone.com]`) and ends the level without requiring a pedestal touch, Demo 1 is **aim + trigger + hold** — exactly the measured WBC strengths.
2. If the pistol must be picked up or the exit pyramid must be touched, The Fall needs a **step** (walking), and the first standing-only target becomes **Cage Fight** (story level 2), where the opening enemy is **0.90 m** away and closes into punch range `[MEASURED: level2 transforms]`. Cage Fight's exit must be checked for the same touch requirement.

**Walking is no longer unscoped, but it is still not on Demo 1's critical path.** Corridor (story level 3) puts the first enemy factory **12.31 m** away and the first item **1.62 m** away `[MEASURED: level3 scan]`; it requires traversal and turning. Do not make Corridor the first target. The parallel `research/wbc-policy-proposal.md` measures **GEAR-SONIC default**: **6.36 m/10 s**, hold **3.6 cm/20 s**, walk→hold→walk without falling (4 s and 10 s holds), and dodge **0.15–0.18 m in <0.5 s** `[MEASURED: research/wbc-policy-proposal.md]`. That is the recommended walking executor (route A), but its R1 tracking is partial (hold 4–6 cm / 17–23°, walking 5–10 cm / 8–19°; head z/x and pitch/roll not tracked), and its 5-cycle/60 s/live tests are unrun. The first walking gate is SONIC's one-day falsification, not the classical single-support test; the classical contact-QP remains a fallback, and the hybrid WBC-stand + SONIC-walk is the precision route.

**C0 has never been run and is the first go/no-go.** The game is not installed; the synthetic-HMD path under Proton/xrizer/Monado is still untested `[CITED: ../docs/archive/2026-10-23-clank-state-of-findings.md §4]`. Demo 1 is meaningless if C0 fails.

---

## 1. What the game asks for — measured

### 1.1 Build and story order

The build is Unity **2022.3.62f3**, IL2CPP metadata v31, `xrsdk-pre-init-library=UnityOpenXR`, AppId **617830** `[MEASURED: boot.config / steam_emu.ini]`. `globalgamemanagers.assets` fixes the scene order `[MEASURED]`. The first story levels are:

| Build index | Scene | Story name | Role in this proposal |
| ---: | --- | --- | --- |
| 0 | `VRLaunch` | boot/hub | not a level |
| 1 | `CezaryVR_ReverseFall` | **The Fall** | **Demo 1** |
| 2 | `CageFight_VR` | **Cage Fight** | standing-only fallback |
| 3 | `CezaryVR_Corridor` | **Corridor** | Demo 2, walking gate |
| 4 | `CezaryVR_Dojo` | **Dojo** | later |
| 5 | `CezaryVR_AirportControl` | **Airport** | later |

The display names and the order for levels 1–3 are `[CITED: https://readyvrone.com/vr-gaming/superhot-vr-game-guide/]` and `[CITED: https://www.ludo.guide/guide/superhot-vr]`; the scene names and indices are `[MEASURED]`. The `<Story>` block in the extracted `GameData.xml` lists AirportControl, Helipad, BreakIn, LotusRoad, SHMUP, and AdrianVR_Test — it is **not** the opening story order; do not use it for ordering `[MEASURED: /tmp/opencode/gamedata.xml]`.

### 1.2 The Fall's geometry

The player's XR origin is `_CameraRig` at **(10.502, 3.623, -7.900)** with a measured yaw of **−90.2°** (forward is −x). A second camera rig, `VRCameraRig`, sits at (10.405, 3.587, -7.936) with `VRCamera (eye)` at local zero; the XR rig is the one carrying `[CameraRig OCULUSVR]` and `XRRig` `[MEASURED: level1 hierarchy]`. The runtime eye is the XR origin plus the HMD pose. At the E3 standing height (**1.221 m**) the eye is at **4.844 m**; the level's catwalk origin is at **3.400 m**, and the combat objects sit at **4.8–4.9 m** — i.e. at standing eye level. The exact in-game eye height is a C0 measurement, not an assumption carried into the honesty claim.

| Object | World position | Horizontal Δ from XR origin | Vertical Δ from XR origin | Role |
| --- | ---: | ---: | ---: | --- |
| `EnemyFactory (12)` | (9.16, 3.50, −7.90) | **1.35 m** | −0.12 m | spawn point under `EnemiesToSpawn` |
| `EnemyBody` / first enemy | (9.09, 3.60, −7.90) | **1.41 m** | −0.02 m | first enemy |
| enemy head mimic (`Chest`) | (~9.16, 5.37, −7.90) | ~1.4 m | +1.75 m | head/aim reference |
| `PistolGunPickup` | (9.80, 4.90, −7.90) | **0.70 m** | +1.28 m | pistol; at eye level |
| `EndingPyramid` | (9.80, 4.80, −7.90) | **0.70 m** | +1.18 m | exit, **inactive** at start |
| `BlackCube` | (10.40, 3.70, −7.90) | 0.13 m | +0.08 m | intro/blackout object |
| `IntroPiOSMonitor` | (8.32, 4.85, −8.13) | 2.51 m | +1.23 m | intro screen |
| `IntroOuterMonitors` | (7.30, 5.18, −7.92) | 3.56 m | +1.56 m | intro screens |
| `SuperText` | (7.30, 5.50, −7.98) | 3.71 m | +1.88 m | text |
| catwalk prefabs | (10.50, 3.40, −6.70) etc. | 1.22–2.34 m | −0.22 m | floor/staging |
| `EnemiesToSpawn` | (−8.28, 8.13, 17.66) | 32.04 m | +4.51 m | staging parent; factory is a child |
| `Zwischenplatz` | (−57.29, −12.84, −15.61) | 70.18 m | −16.46 m | staging parent of the pistol |

`[MEASURED: /tmp/opencode/spawn_contract.txt, level1 section; /tmp/opencode/level1_scan.txt]`

**Do not read the 1.46 m `PistolGunPickup` distance in the spawn-contract table as the eye-to-gun distance.** That is the 3D distance from the XR origin (0.70 m horizontal + 1.28 m vertical). Eye-relative, the pistol and pyramid are **0.70 m forward and within 0.06 m of eye height**. The enemy's chest is **1.41 m forward at eye height**.

The first enemy is authored directly ahead, on the –x side: the player faces –x in this level. The catwalk staging is around the player. No yaw is needed to see the first enemy; a small pitch may be needed for the head.

### 1.3 Flow control — what the level owns

`CezaryVR_ReverseFallFlowControl` is a 404-byte MonoBehaviour on `LevelFlowControl`; its serialized references map as follows `[MEASURED: raw PPtr parse, level1 object 25612]`:

| Field | Referenced object | What it is |
| --- | --- | --- |
| `RenderersAfterMonitors` | 18 catwalk/lamp/pipe objects | intro staging that appears after the monitors |
| `FirstEnemies` | `EnemiesToSpawn` (597) | first wave container |
| `Item` | `PistolGunPickup` (2026) | the pistol, parented under `Zwischenplatz` |
| `ZwischenObjects` | `Zwischenplatz` | staging parent ("in-between place") |
| `SuperText1` | `SuperText` | text |
| `FinalPyramid` | `EndingPyramid` (3851), **inactive** | exit |
| `IntroVideoMonitor` | `IntroPiOSMonitor` | intro video |
| `IntroOuterMonitors` | `IntroOuterMonitors` | intro screens |
| `IntroInnerMonitor` | `IntroInnerMonitor` | intro screen |
| `BlackCube` | `BlackCube` (1973) | blackout/intro |
| `FirstEnemyRenderer` | SkinnedMeshRenderer (23218) | first enemy renderer |
| `FirstEnemyLayers` | `rig` (5334) | first enemy rig |

The `PistolGunPickup` is a child of `Zwischenplatz` at local (67.086, 17.737, 7.707); with the parent at (−57.29, −12.84, −15.61) it lands at the pedestal `[MEASURED]`. This is a staging pattern, not evidence that the gun is parked 70 m away. The guide says "use the gun you are given" `[CITED]`; the serialized `Item` being a `GunPickup` and the `ZwischenObjects` being the staging parent are consistent with a handoff, but **the handoff is not proven by the asset data**. C0 logs the pistol's parent and pose against the achieved hand.

The first enemy's factory is not active at start: `EnemyFactory (12)` is a child of `EnemiesToSpawn` and is **inactive**; the flow control activates it. The fields imply a scripted intro → first enemy → item/exit state machine; C0 logs the actual order. A scripted commander can use the serialized references; a VLM is not needed for the first clear.

### 1.4 The clock

`TimeControl` is a 48-byte MonoBehaviour, identical in levels 0–5 `[MEASURED: raw MB floats]`:

| Field | Value |
| --- | ---: |
| `minTimescale` | **0.005** |
| `maxTimescale` | **1.0** |
| `inDash` | 0.0 |
| `timescalevalue` (serialized) | 0.0 |
| `referenceFixedDeltaTime` (class constant) | 0.011111111 (~90 Hz) `[MEASURED: dump.cs]` |

`TimeshiftingTransform` has an `isHead` flag and separate movement/rotation multipliers; the player/HMD/hand transforms use it, but the exact runtime attachment is an assumption. Its serialized values on the head are `tickThreshold = 0.1`, `movementTick = (0.1, 0.1)`, a movement time-shift multiplier of **10.0**, a minimum amplitude threshold of **0.0007**, and a rotation threshold of **0.05** `[MEASURED: serialized floats; exact semantics ASSUMPTION]`. The class has `isHead`, `workInTicks`, and separate movement/rotation multiplers `[MEASURED: dump.cs]`.

What this buys the demo:

- The serialized floor is **0.5% speed**, not a pause. A 20 s think at the floor advances the world **0.1 s** `[ASSUMPTION: floor applies; C0 measures]`.
- The WBC hold's maximum base excursion is **0.22 mm** and its maximum head excursion is **1.4 mm** over 20 s `[MEASURED: research/wbc-classical-proposal.md]`. Both are far below the **0.1** movement tick; the clock should sit at or near the floor during a hold.
- The WBC look moves the head **1.16 cm** and the base **3.3 mm** over 20 s `[MEASURED]`. Run the VLM only in `hold`, not during `look`/`reach`, so the larger head motion does not advance the clock.
- The mod `MINIMAL_TIMESCALE_HIGHER` ("time flows faster even when you…") confirms the floor is a real game concept `[MEASURED: GameData.xml]`. Do not enable it; the measurement uses the default story.

The time-scale mapping is not decompiled. The C0 protocol in §5 measures the ratio with a moving object rather than trusting the constant alone.

### 1.5 Combat constants

| System | Serialized value | Implication |
| --- | ---: | --- |
| `MindDeathWaveSystem.activationTimeShift` | 0.45 s | wave activation delay |
| `deathWaveInterval` | 1.25 s | wave cadence |
| `distanceCheckTimeTreshold` | 2.5 m | the first enemy at 1.41 m is inside |
| `maxHandsDistance` | 2.0 m | hand-distance gate |
| `maxSingleModeDeathWaveUses` / `maxDualModeDeathWaveUses` | 1 / 1 | one use per mode |
| `Bullet.COLLISION_RADIUS` | 0.03 m | dodge envelope is small |
| `Bullet.BROADPHASE_RADIUS_SQUARED` | 1.0 | 1 m broadphase |

`[MEASURED: dump.cs + serialized MindDeathWaveSystem in levels 1/3/4/5; Bullet class constants]`

The mind-death-wave is a potential second kill path at 1.41 m, but its trigger is not decompiled and it is not the planned clear. A 12 cm lean is larger than the 3 cm bullet radius; whether it clears a shot depends on the aim point and the body/head width. The WBC's 12.1 cm dodge is 4× the bullet radius, but it takes 0.62 s to peak — the game's time scaling is what makes that potentially acceptable. Do not rely on the dodge for The Fall; shoot first.

### 1.6 The later stages

| Stage | First enemy | First item | What it demands |
| --- | ---: | ---: | --- |
| Cage Fight (`CageFight_VR`) | `EnemyBody` **0.90 m** | throwable dog **1.19 m** | melee; enemies close; exit pyramid 1.72 m |
| Corridor (`CezaryVR_Corridor`) | factory **12.31 m** | Motorola throwable **1.62 m** | traversal, turning, both ends |
| Dojo (`CezaryVR_Dojo`) | factory **10.26 m** | — | traversal; camping point 12.98 m |
| Airport (`CezaryVR_AirportControl`) | six named enemy waves (First–Sixth), 5 teleport pairs | activator item | long scripted sequence |

`[MEASURED: level2/3/4 spawn tables; AirportControlFlowControl fields]`

The Corridor's `Item` is a `ThrowableItem` (Motorola), not a gun; its first enemy factory is 12.31 m away, and `EnemiesToSpawn` is 21.14 m away `[MEASURED]`. The Airport flow control has six named enemy-wave fields (`FirstEnemies`…`SixthEnemies`), `TeleportTo1–5`/`LookAtTeleport1–5`, `TimeMovesOnly`, `SuperText1–6`, `HideForPhase4`, and `HideOnFinalPhase` `[MEASURED: dump.cs]`. Both are post-walking targets.

---

## 2. What the body can do today — measured

The classical WBC in `research/wbc-classical-proposal.md` is **weighted task-space IK over the floating-base G1**, 50 Hz, `mink` QP, position actuators, Menagerie `scene.xml` (29 joints, no neck, no fingers). It is not a contact-force QP and it does not walk. Its measured standing results:

| Skill | Result | Caveat |
| --- | ---: | --- |
| Hold, 20 s | base **0.22 mm**, head 1.4 mm | neutral stand; Menagerie model |
| Look, ±35° waist | base **3.3 mm**, yaw error **0.63°**, head tilt 2.9° | head is rigid to the torso; no neck |
| Reach, single arm | **2.2–12.9 mm** | targets +0.20 m from neutral wrist |
| Reach, both arms | **17.5–28.8 mm** | targets +0.20 m |
| Aim, single arm | **0.18–2.34°** pointing | pointing, not full 6-DoF orientation |
| Aim, both arms | **1.05–6.26°** | pointing |
| Punch | **9.5 mm**, 3.6° aim | 0.5 s quasi-static extension, not impact |
| Dodge | **12.1–14.7 cm** peak | 0.62–0.86 s to peak; lean, not step |
| Solve time | **0.51–1.08 ms** mean, ≤2.44 ms p95 | headless CPU; live co-residency untested |

`[MEASURED: spike/wbc_classical/logs/summary.json, via research/wbc-classical-proposal.md]`

The **C3 lineage's** walking side is the learned Playground policy's regime: **0.743 m/s**, +0.23°/10 s on the Playground XML, but arms-on-IK falls in **3.39–4.84 s**, zero-command creep is **1.344 m/20 s**, and walk→hold falls in **3.11–4.24 s** `[MEASURED: ../docs/2026-10-23-clank-design.md §18]`. That lineage is superseded as the walking candidate by SONIC (§2.3); it remains the record of why the frozen 29-DoF policy is not bolted to the WBC. The WBC's 0.10 m weight-shift limit and 1.3–1.6 s single-support tip are the boundary `[MEASURED: research/wbc-classical-proposal.md §3.1]`.

### 2.1 The new far-reach probe

The Fall's pistol and exit are 0.70 m from the eye. The WBC's published reach targets stop at 0.20 m. A probe with the same controller and default weights, targeting the head position + 0.70 m forward at eye level, gives `[MEASURED: spike/wbc_game/reach_far_test.py, logs/reach_far_test.txt]`:

| Target | Side | Final wrist residual | Base max | Min margin | Fall |
| --- | --- | ---: | ---: | ---: | --- |
| pistol at 0.70 m | right | **27.3 cm** | 3.4 cm | 3.1 cm | no |
| pistol at 0.70 m | left | **27.3 cm** | 3.4 cm | 3.2 cm | no |
| pyramid at 0.70 m | right | **19.7 cm** | 6.7 cm | 0.2 cm | no |
| pyramid at 0.70 m | left | **20.1 cm** | 6.0 cm | 0.8 cm | no |
| 0.80 m | right | 34.0 cm | 4.3 cm | 2.0 cm | no |

The controller does not fall, but the target is out of reach at the measured posture: the final wrist is still 20–34 cm short. A 0.10 m base shift (the measured limit) would recover at most 10 cm; a **step** is needed for the pedestal/exit. This is the decisive limitation for Demo 1 and the reason the gun-handoff/level-end assumptions are stated as conditions, not facts.

The aim path is different: aiming does not require touching the 0.70 m object; it points the wrist at the enemy 1.41 m away. The WBC's single-arm aim error of **2.34°** is **5.7 cm** at 1.41 m `[ASSUMPTION: pointing error → lateral miss; the enemy body is much wider]`. The pistol ray is along the controller's forward axis; calibrate the wrist-z → controller-forward transform in C0. The aim scenario is a slow quasi-static extension (2 s), not a fast strike; the 15.2° transient means fire only after the aim settles.

### 2.2 One model or three? The walking choice

The WBC was measured on Menagerie `scene.xml`; SONIC uses its own MuJoCo model; the old Playground policy is the C3 lineage. These are different bodies. A standing-only Demo 1 stays on Menagerie. The walking route is now a measured choice, not a blank:

- **SONIC default (route A, recommended by `research/wbc-policy-proposal.md`):** walks **6.36 m/10 s**, holds **3.6 cm/20 s**, transitions 4 s/10 s without falling, dodges **0.15–0.18 m <0.5 s**; R1 partial; **6.8 ms/tick** CPU; 5-cycle/60 s/live unrun.
- **Classical WBC + Option A/B walking:** one Menagerie model; WBC precision; walking unproven (single-support tips 1.3–1.6 s); 5–14 days.
- **Hybrid WBC stand + SONIC walk:** best precision plus measured walking; handoff/authority untested; 3–5 days (policy proposal route 2).

C3f's failure was the frozen 29-DoF **Playground** policy, not SONIC; do not use it as evidence against SONIC. Do not compare Menagerie WBC residuals to SONIC's model; state the model in every caption.

### 2.3 The learned walking candidate — SONIC default

| Skill | SONIC default `[MEASURED: research/wbc-policy-proposal.md]` | Caveat |
| --- | ---: | --- |
| Hold, 20 s | **3.6 cm net**, 7.5 cm max, 0.0001 m/s | wrist pos 4–6 cm, orientation 17–23°; arms float ~0.15 m |
| Walk, 10 s | **6.36 m**, 0.64 m/s, no fall | path error 0.271 m; 60 s unrun |
| Arms while walking | survives | path error 0.32–0.37 m; wrist 5–10 cm / 8–19° |
| Transitions | 4 s and 10 s hold, no fall | 5-cycle unrun; hold drift 0.20 m; resume drift 1.86 m |
| Dodge | **0.15–0.18 m** in <0.5 s | head-position step; game's required distance unmeasured |
| Head | y 0.15–0.18 m | z/x and pitch/roll not tracked; torso yaw +25–31° at head yaw ±30–90° |
| Compute | **6.8 ms/tick** CPU, 0.9× realtime | live co-residency with the game unverified |

SONIC is the walking route, not the Demo 1 body. Its hold is ~160× looser than the WBC's 0.22 mm, and its wrist/head tracking is the weak axis; the WBC wins R1 in stand. The hybrid is the only route with both measured precision and measured walking, and its transition is the open question.

---

## 3. Requirement coverage for Demo 1

The state-of-findings R1–R8 are project bars, not demo bars. Demo 1 is allowed to be a strict subset; it must not claim the rest.

| # | Requirement | Demo 1 status | Evidence / gap |
| --- | --- | --- | --- |
| R1 | Arms/head commandable; wrist residual ≤3 cm; orientation measured | **WBC stand: met. SONIC: partial. 0.70 m pedestal: not met.** | WBC reach 2.2–12.9 mm, aim 2.34°; SONIC hold 4–6 cm / 17–23°, walking 5–10 cm / 8–19°; far-reach probe 19.7–34.0 cm at 0.70–0.80 m. The game shot uses the WBC; the pickup/exit needs a step. |
| R2 | Hold ≤~5 cm base over 20 s | **WBC: met. SONIC: met.** | WBC 0.22 mm; SONIC 3.6 cm; both pass, WBC ~160× tighter. The WBC is the Demo 1 body. |
| R3 | Walk ≥5 m, ≥0.3 m/s, 60 s, push recovery | **WBC: not met. SONIC: sim pass.** | SONIC 6.36 m/10 s, 0.64 m/s, no fall; 60 s unrun; live unverified. Project route, not Demo 1. |
| R4 | 5× walk→hold→walk, no fall | **WBC: n/a. SONIC: partial.** | SONIC passes 4 s and 10 s holds; 5-cycle soak unrun. Demo 1 has no walk→hold. |
| R5 | Dodge 10–25 cm in <500 ms, return upright | **WBC: partial. SONIC: sim pass.** | WBC 12.1–14.7 cm, 0.62–0.86 s; SONIC 0.15–0.18 m <0.5 s; the game's required dodge is unmeasured. The Fall should shoot first. |
| R6 | 50 Hz live on the 4060 laptop | **Both headless pass; live unverified.** | WBC 0.51–1.08 ms mean; SONIC 6.8 ms/tick CPU, 0.9× realtime; Superhot + Monado + MuJoCo co-residency is the live measurement. |
| R7 | Honest: achieved `mjData`, no fake poses | **Both pass in simulation; integration must preserve.** | Achieved poses only; trigger is a controller button, not a pose; no game telemetry. |
| R8 | Complete a stage | **Demo 1 target; conditional on the level-end path.** | The combat is within the WBC's measured skill set; the exit/pickup may need a step. |

**Project-route update:** SONIC default passes R3, R4 (partial), and R5 in simulation; it does **not** change Demo 1's WBC choice. Demo 1 is the first honest **stage clear**; the walking/whole-body claim belongs to the SONIC or hybrid route, after its own gates.

---

## 4. The demo we should build

### 4.1 Demo 1 — "The Fall, standing"

**Scope.** MuJoCo Menagerie G1 + measured WBC; `BodyCommand`/`BodyState`; achieved head/wrists to Monado `remote`; xrizer; Superhot under GE-Proton; grim capture; scripted commander first, VLM second. No walking, no root velocity, no fake poses.

**Sequence.**

1. **C0** (below): confirm the synthetic HMD, trigger, grim capture, the pistol handoff, the level-end trigger, and the clock ratio.
2. **Hold** at the level start. Let the intro/monitors finish; the game clock should be at/near 0.005. If the intro is gated on game time, a pure hold stalls it: use a small scripted head motion to advance the intro, then return to hold. C0 measures which gate applies. Log the enemy/pistol poses.
3. **Look** at the enemy (the enemy is already ahead; small correction only). Keep the head motion inside the ±35° arc; stop moving before the shot.
4. **Aim** the right wrist at the enemy chest 1.41 m ahead, using the WBC aim task; hold the aim steady for ≥0.5 s. The transient aim error is 15.2°, so do not fire during the swing.
5. **Trigger.** Set the controller trigger (a button, not a pose) when the aim is settled. Verify the hit.
6. **Exit.** If the game hands the pistol and ends the level after the kill, log the level transition and stop. If the exit pyramid requires a touch at 0.70 m, the standing WBC cannot reach it; do **not** teleport or fake the hand. Add the walking gate or switch to Cage Fight.

**Pass:** the game shows the stage clear; the video shows the sim body and the game view; the CSV proves packets == achieved `mjData`; the trigger is logged as an input; the clock stayed near the floor during the hold.

**Allowed fail:** the shot misses because the aim transform is miscalibrated; the enemy kills the body; the body falls. These are honest failures and still produce evidence. **Hard fail:** fake poses, game memory/telemetry in the motor path, a mouse/keyboard game hack, or claiming a stage clear when the level did not transition.

**Why this is the right first demo.** It uses the WBC's strongest measured skills (hold, look, aim, slow extension), it avoids walking/transition, it exercises the full honest OpenXR pipe, and it makes the game's clock the ally of the slow VLM. It is a real stage clear, not a turret in a custom scene.

**Staging, not architecture.** Demo 1 is a deliberate stand-only milestone, not the final body. The final whole-body executor is SONIC route A or the hybrid WBC-stand + SONIC-walk; the WBC is used here because it is the only measured ≤3 cm hold/aim and because The Fall needs no walking. Do not present a two-mode WBC/SONIC switch as the production design without a measured handoff; the Menagerie WBC and SONIC are different models.

### 4.2 Fallback — "Cage Fight, standing"

If The Fall requires a 0.70 m pickup or pedestal touch, Cage Fight is the next candidate: the opening `EnemyBody` is **0.90 m** away, the throwable dog is **1.19 m**, and the guide says the stage starts by punching an enemy `[MEASURED: level2 transforms; CITED: readyvrone.com]`. The WBC's punch is 9.5 mm at +0.30 m; the enemy closes into range. The same conditions apply: the level's `TeleportPyramid` is **1.72 m** from the origin — if the level end requires touching it, Cage Fight also needs a step. Measure both level-end triggers in C0 before committing.

The melee path also demonstrates the WBC's punch and does not depend on the gun handoff. It is a better fallback than hacking the pickup.

### 4.3 Demo 2 — "Corridor, walking"

Corridor is the right **second** target, not the first. The first enemy factory is 12.31 m away, the item is a throwable at 1.62 m, and the level demands traversal, turning, and enemies from both ends `[MEASURED: level3; CITED: ludo.guide]`. The body must walk and then hold/aim. The measured walking candidate is SONIC default (route A), not the classical WBC:

- **Primary (route A):** integrate SONIC default. Its one-day gate is 5-cycle walk→hold→walk, a 60 s walk, aim-while-walk residual ≤10 cm, and the in-game dodge measurement (`research/wbc-policy-proposal.md` §4.3). If the soak and 60 s walk pass, the body can walk/hold/transition; if the aim-while-walk test fails, use the hybrid; if the soak/60 s fails, the classical contact-QP is the fallback.
- **Classical fallback:** the one-day single-support falsification from the classical proposal; if it passes, 5–10 days of contact-QP + scripted gait on Menagerie; one model, WBC upper body.
- **Hybrid (precision route):** WBC for stand/hold/aim/punch + SONIC for walking; the handoff/authority is untested; 3–5 days. This is the only route with both measured precision and measured walking.

Do not start by bolting the WBC onto the frozen 29-DoF **Playground** policy; C3f measured that authority conflict (falls in 3.39–4.84 s). SONIC is a different controller with a decoupled upper body; C3f does not condemn it.

The game's clock during a SONIC hold is a new measurement: SONIC's 3.6 cm base is below the 0.1 tick, but its 4–6 cm wrist and 17–23° orientation may cross the rotation deadband. If the clock advances, use the hybrid (WBC hold) or keep the VLM off during the hold.

### 4.4 The VLM's role

The VLM is **not** in Demo 1's critical path. The first clear is scripted: the game-side state machine knows the enemy/pistol/exit references. After the scripted clear, add the VLM on the same bus:

- Input: one grim still + a ≤10-line digest (skill, supported bits, residuals, hold drift, last outcome). Pixels-only planning was 2.1× worse in stubborn-gamer `[CITED: design doc §9]`.
- Output: skill + side only (`hold` | `look` | `reach` | `strike`), no 3D point, no joint angles, no OpenXR pose.
- Run only in `hold`; 20 s p50 is acceptable at the clock floor; the WBC's 0.22 mm hold keeps the clock there.
- The adapter supplies the target (the known enemy chest) from the game-side state machine or a 2D tracker; the VLM does not emit poses.

The VLM's first useful decision on The Fall is trivial ("enemy ahead, gun in hand, fire"); its value is proving the slow-brain/50 Hz-body loop, not the stage.

### 4.5 Work order and estimates

| Step | What | Estimate `[ASSUMPTION]` | Gate |
| --- | --- | --- | --- |
| C0 | Proton/xrizer/Monado remote + scripted pose + trigger + grim; The Fall's handoff/exit/clock | 1–2 days | C0 pass; otherwise stop |
| W1 | WBC + `BodyCommand` + 50 Hz bridge + achieved-pose logging | 1–2 days | packets == `mjData`; hold/look/aim |
| W2 | The Fall scripted clear (handoff, aim, trigger, level end) | 2–3 days | stage clear on video |
| W3 | VLM writer on the same bus | 1–2 days | one valid command per still; invalid → hold |
| W4 | SONIC 1-day gate: 5-cycle, 60 s walk, aim-while-walk ≤10 cm, in-game dodge | 1 day | pass/fail |
| W5 | Walking executor: SONIC route A (2–3 d); classical Option A (5–10 d) or hybrid (3–5 d) if gates fail | 2–10 days | walk 5 m, hold, no fall; aim residual |
| W6 | Corridor | after W5 | traversal + stage clear |

Demo 1 is roughly one week after C0 passes. Demo 2 is one to two weeks after the SONIC gate passes. The estimates are not commitments; the walking/aim integration is the high-risk part.

---

## 5. C0 protocol — the unrun game-side measurement

### 5.1 Setup

- Extract `Superhot-VR.rar` (actually a ZIP) to a Steam-library-style directory or run it as a non-Steam game. Do not modify the game files. `[MEASURED: file is a ZIP; no installed Steam copy]`
- GE-Proton10-34, xrizer 0.5, Monado PSVR2 fork, transient unit with `P_OVERRIDE_ACTIVE_CONFIG=remote`, `IPC_EXIT_ON_DISCONNECT=off`, `XRT_NO_STDIN=1`, `XRT_COMPOSITOR_COMPUTE=false`, `VK_DRIVER_FILES=/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json`; `PRESSURE_VESSEL_IMPORT_OPENXR_1_RUNTIMES=1`; `XR_RUNTIME_JSON=/run/current-system/sw/share/openxr/1/openxr_monado.json` `[CITED: design doc §10]`.
- Run `hello_xr -g Vulkan2` first; confirm the pose pipe and grim capture. Then launch the game.

### 5.2 What to log

| Measurement | Method | Pass |
| --- | --- | --- |
| View follows TCP pose | scripted HMD yaw ±35°; record the game window | the view follows; no freeze |
| Trigger reaches the game | scripted trigger on/off; observe the pistol/UI | trigger registers |
| Grim captures the game window | `hyprctl clients -j`, `grim -g` | game pixels, not the desktop |
| HMD height / scale | log game eye vs robot head; enemy/pistol apparent size | documented height; no silent scale |
| Pistol handoff | log `PistolGunPickup` parent/pose vs achieved right wrist; observe | gun is in hand (or pickup distance known) |
| Exit trigger | observe `EndingPyramid` activation; try a ray/hand at 0.70 m | level ends without a step (or a step is required) |
| Clock ratio | hold 20 s; track a moving object (enemy, bullet, dropped gun); repeat with a known HMD motion | floor ≈0.005; hold advances ≤~0.1 s |
| Pose→pixel latency | E2 method; game window | 30–59 ms + game render; documented |

**Clock measurement detail.** The game's time-scale is not exposed. Use a moving object with a constant game-space speed (the enemy's approach, a bullet, or a dropped pistol under gravity) and measure its screen/world displacement per real second at (a) hold and (b) a known HMD motion. The ratio of the two rates is the time-scale ratio. If no object is available, use the level's own timer/scoreboard at the end of the stage; do not use game memory.

**Height/scale detail.** The robot's standing HMD is **1.221 m** (straight-legged) and **1.170 m** (knees-bent) `[MEASURED: E3/design]`. The level's XR origin is at 3.623 m; the objects at 4.8–4.9 m are authored at standing eye level. Use STAGE with floor = sim ground; `hmd_z_offset` default 0; if the game is unplayable at the robot's height, change the offset explicitly and document it. Do not smuggle scale into the honesty claim.

### 5.3 C0 fail handling

- **Game rejects the synthetic HMD** after the Monado/xrizer recipe: stop and re-plan. The design allows a game switch only here; do not discover it at E4.
- **Trigger/ray does not reach**: calibrate the controller's forward axis; if the game requires a physical proximity the body cannot produce, stop.
- **Pistol not handed over** and pickup >0.5 m: The Fall needs a step; use Cage Fight or the walking gate.
- **Exit requires a 0.70 m touch**: same.
- **Clock floor is not ~0.005**: the VLM think budget is smaller; scripted first; re-plan the brain rate.

---

## 6. Kill criteria

1. **C0 fails.** No synthetic HMD, no trigger, no game capture → the artifact has no game-side leg. Stop.
2. **The gun is not handed over and the exit requires a touch.** The standing WBC's 0.70 m reach is 20–27 cm short; a step is required. Do not teleport, do not fake the hand. Switch to Cage Fight if its end is automatic, or make walking the next milestone.
3. **The walking candidate fails its gate.** SONIC's 5-cycle or 60 s walk fails → classical Option A; SONIC's aim-while-walk ≤10 cm fails → hybrid WBC stand + SONIC walk; both fail → the body cannot walk+aim at the game's bar. Report it; do not claim R3/R4/R5.
4. **SONIC's hold advances the clock.** Its 3.6 cm base is below 0.1, but its 4–6 cm wrist / 17–23° orientation may cross the rotation deadband. Use the WBC/hybrid for hold or keep the VLM off; measure.
5. **The level requires a 180° turn or traversal.** The WBC's look is ±35°; there is no walking. The Fall's first enemy is ahead, but the intro/exit may turn the player. Measure; if it does, The Fall is not a standing-only stage.
6. **The clock advances during hold.** If the game's floor is high or the WBC's head motion crosses the deadband, the 20 s VLM think is unsafe. Reduce the think time or script the first clear; do not claim the clock freeze.
7. **The aim ray cannot hit at 1.4 m.** The WBC's aim is 2.34° / 5.7 cm; if the game's controller ray is offset or the enemy hitbox is small, fix the wrist→controller transform; if it still misses, the aim skill is not sufficient for this game.
8. **Live 50 Hz co-residency fails.** R6 fails even if the controller is correct; reduce the VLM/vision load, not the control rate. Do not hide it in the video.

Demo 1 criteria are 1–2 and 5–8; the walking-route criteria are 3–4.

---

## 7. What is not claimed

- **No walking by the WBC, no R3/R4 for Demo 1.** The WBC is a standing controller. Demo 1 is a standing stage clear; SONIC's walking is the project route, measured but with partial R1 and unrun 5-cycle/60 s/live tests.
- **No claim SONIC meets R1.** Hold 4–6 cm / 17–23°, walking 5–10 cm / 8–19°; head z/x and pitch/roll are not tracked. The game's shot uses the WBC.
- **No claim the hybrid handoff is solved.** WBC stand + SONIC walk is the precision route, not a measured transition.
- **No fast dodge from the WBC.** Its peak is 0.62–0.86 s, not <0.5 s. SONIC reaches 0.15–0.18 m in <0.5 s in simulation; the game's required dodge is still unmeasured.
- **No full 6-DoF wrist orientation.** Aim is a pointing metric; the roll about the pointing axis is free. Calibrate the gun ray.
- **No fingers.** The Menagerie G1 has no hand DoF; the trigger/grip are controller buttons, not poses.
- **No live R6.** The solve times are headless; C0/Demo 1 is the live test.
- **No game API, telemetry, memory injection, or fake poses.** The game sees only the OpenXR poses/buttons; the eye is grim on the game window.
- **No claim that the 0.005 floor is a full pause.** It is a serialized 0.5% floor; the 0.1 s/20 s figure is an assumption until C0 measures it.
- **No claim that the models are interchangeable.** Menagerie WBC, SONIC, and Playground numbers are different bodies until unified.

---

## 8. Reproduce

### Game-side data

The asset extraction and scan scripts are ephemeral in `/tmp/opencode`; the tables in this document are the durable record. To regenerate:

```bash
# extract read-only assets
unzip -p "/home/nate/Downloads/Superhot-VR.rar" \
  "SUPERHOT VR/SUPERHOT_VR_Data/il2cpp_data/Metadata/global-metadata.dat" \
  > /tmp/opencode/il2cpp/in/global-metadata.dat

# scans (uses /tmp/opencode/uvenv with UnityPy 1.25.4)
/tmp/opencode/uvenv/bin/python /tmp/opencode/scan_level.py level1
/tmp/opencode/uvenv/bin/python /tmp/opencode/scan_level.py level3
/tmp/opencode/uvenv/bin/python /tmp/opencode/scan_level.py level4
```

Raw data: `/tmp/opencode/spawn_contract.txt`, `level1_scan.txt`, `level3_scan.txt`, `level4_scan.txt`, `level5_scan.txt`, `gamedata.xml`, `il2cpp/dump.cs`, `il2cpp/stringliteral.json`.

### WBC and far-reach probe

```bash
# the published WBC scenarios
cd /home/nate/Projects/Clank/spike/wbc_classical
/home/nate/Projects/Clank/spike/e3/.venv/bin/python wbc_spike.py --scenario all
# -> logs/summary.json, logs/run_all.txt

# the new 0.70 m standing-reach probe
cd /home/nate/Projects/Clank/spike/wbc_game
/home/nate/Projects/Clank/spike/e3/.venv/bin/python reach_far_test.py | tee logs/reach_far_test.txt
```

Model: `spike/e3/vendor/mujoco_menagerie/unitree_g1/scene.xml` (Menagerie, 29 actuated joints, position actuators, `kp=500`, no neck). The probe uses the WBC's default weights and a 2 s hold with a 1.5 s ramp; it is not a full workspace survey.

### Learned walking candidate

The walking route's full numbers, caveats, and obs layout are in `research/wbc-policy-proposal.md`. Its matrix:

```bash
cd /home/nate/Projects/Clank
spike/e3/.venv/bin/python spike/wbc/measure_matrix.py
# -> spike/wbc/logs/summary.json (~30 min)
```

### External sources

- Level 1 The Fall, level 2 Cage Fight, "the gun you are given": https://readyvrone.com/vr-gaming/superhot-vr-game-guide/
- Level 3 Corridor: https://www.ludo.guide/guide/superhot-vr

---

## Appendix A — The Fall raw references

`CezaryVR_ReverseFallFlowControl` (level1, object 25612, 404 bytes) `[MEASURED]`:

```
RenderersAfterMonitors: 18 GameObjects (catwalks, lamps, pipes)
FirstEnemies:           597   EnemiesToSpawn
Item:                   2026  PistolGunPickup  (GunPickup MB 26563)
ZwischenObjects:        2012  Zwischenplatz
SuperText1:             3012  SuperText
FinalPyramid:           3851  EndingPyramid (inactive)
IntroVideoMonitor:      2876  IntroPiOSMonitor
IntroOuterMonitors:     2159  IntroOuterMonitors
IntroInnerMonitor:      2454  IntroInnerMonitor
BlackCube:              1973  BlackCube
FirstEnemyRenderer:     23218 SkinnedMeshRenderer
FirstEnemyLayers:       5334  rig
```

`PistolGunPickup` hierarchy: `PistolGunPickup (2026) <- Zwischenplatz (2012) <- Gameplay`; local position (67.086, 17.737, 7.707); world (9.80, 4.90, −7.90); active `[MEASURED]`. `EndingPyramid` world (9.80, 4.80, −7.90); active **false** `[MEASURED]`.

## Appendix B — Clock and combat constants

```
TimeControl (48 B, levels 0–5 identical):
  minTimescale  0.005
  maxTimescale  1.0
  inDash        0.0
  timescalevalue 0.0
  referenceFixedDeltaTime 0.011111111

TimeshiftingTransform (serialized head values):
  tickThreshold  0.1
  movementTick   (0.1, 0.1)
  movementTimeShiftMultiplayer 10.0
  _minAmplitudeThreshold 0.0007
  rotation threshold 0.05

MindDeathWaveSystem (identical levels 1/3/4/5):
  activationTimeShift 0.45
  deathWaveInterval 1.25
  distanceCheckTimeTreshold 2.5
  maxHandsDistance 2.0
  maxSingleModeDeathWaveUses 1
  maxDualModeDeathWaveUses 1
  middleSoundFadeTime 0.75
  middleSoundIncreaseTime 0.25
  middleSoundMaxVolume 1.0

Bullet:
  COLLISION_RADIUS 0.03
  BROADPHASE_RADIUS_SQUARED 1.0
```

## Appendix C — spawn distances, levels 1–5

From `/tmp/opencode/spawn_contract.txt`; distances are from the `_CameraRig` XR origin, not the eye `[MEASURED]`:

| Level | XR origin | Nearest combat objects |
| --- | --- | --- |
| 1 The Fall | (10.50, 3.62, −7.90) | enemy factory 1.35 m; enemy body 1.41 m; pistol 1.46 m 3D (0.70 m horizontal at eye level); pyramid 1.37 m 3D (0.70 m horizontal) |
| 2 Cage Fight | (19.30, 1.87, 1.09) | enemy body **0.90 m**; throwable dog 1.19 m; teleport pyramid 1.72 m |
| 3 Corridor | (11.54, 3.40, −8.78) | Motorola throwable 1.62 m; enemy factory 12.31 m; EnemiesToSpawn 21.14 m |
| 4 Dojo | (player at origin in the scan) | enemy factory 10.26 m; camping point 12.98 m |
| 5 Airport | — | not scanned; flow control has six enemy-wave fields / 5 teleports |

## Appendix D — far-reach probe output

`spike/wbc_game/logs/reach_far_test.txt` `[MEASURED]`:

```
head0 [0.00369 0.00002 1.21913]
wrist0 L [-0.01696  0.22180  0.72090] R [-0.01696 -0.22179  0.72090]
base0 [0. 0. 0.79] com0 [0.00328 0.00008 0.69185]
gun_eye_0.70     right res=27.29cm ori=9.58deg base_max=3.37cm tilt=4.66deg margin=3.11cm fall=False
gun_eye_0.70     left  res=27.34cm ori=10.02deg base_max=3.40cm tilt=4.56deg margin=3.18cm fall=False
pyramid_eye_0.70 right res=19.73cm ori=9.27deg base_max=6.68cm tilt=6.23deg margin=0.21cm fall=False
pyramid_eye_0.70 left  res=20.08cm ori=9.06deg base_max=5.95cm tilt=6.10deg margin=0.83cm fall=False
eye_0.80         right res=33.97cm ori=9.20deg base_max=4.29cm tilt=5.56deg margin=2.02cm fall=False
eye_0.80         left  res=34.11cm ori=8.82deg base_max=4.08cm tilt=5.45deg margin=2.18cm fall=False
```

The probe's 0.70 m target is at the head + 0.06 m (pistol) or −0.04 m (pyramid); the residual is the final wrist error after 2 s. The body remains upright, but the support margin at the pyramid target is only 0.2 cm — another reason not to force the reach.
