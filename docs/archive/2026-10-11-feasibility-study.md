# RESEARCH — Can a simulated body play a VR game with no headset or controllers?

> **Archived 2026-11-13.** The 2026-10-11 feasibility study, written when no code existed here. Its questions have since been measured or replaced; it is the record of the prior art, not a plan.
> Current behaviour: [`README.md`](../../README.md) · agent rules: [`AGENTS.md`](../../AGENTS.md).

> **Superseded in places — read `2026-10-23-clank-state-of-findings.md` first.**
> This is the 2026-10-11 feasibility study, written when no code existed here. Most of it has since
> been measured or replaced: Monado's `remote` driver is proven end-to-end (E1–E3), the executor is
> GEAR-SONIC (C3i–C3m), and Superhot VR accepts synthetic HMD poses (C0).
> No longer authoritative: **§2.4** (a custom OpenVR driver is not needed), **§4** (capture is grim,
> not the debug GUI), **§9** (the first path is decided), **§10** (the open questions are answered or
> tracked in the state doc's §7). Sections §1, §5–§8 and §11 remain valid analysis.

**Status:** web research, 2026-10-11. **No code exists in this repo.** Nothing here has been
measured on this machine. This document is a source-verified feasibility study, not a plan.

**Question.** Can a VR game be run with no headset and no controllers, with a *simulated body* as
the player? End goal: a Unitree G1 humanoid in MuJoCo playing a VR game.

**Answer.** Yes. Every required interface exists and is documented. The specific combination
(simulated humanoid body ↔ off-the-shelf commercial VR game) has **not** been published by anyone;
the closest prior art stops one step short (§8).

**Claim tags.** Every factual statement is one of:
- **MEASURED** — observed on this machine or in a local file, with the how.
- **CITED** — a source URL is given.
- **ASSUMPTION** — reasoning, untested.

---

## 1. What "playing a VR game" reduces to

A VR game asks the runtime for exactly three things and nothing else:

1. an **HMD device with a tracked pose** (both eyes render from it),
2. **two tracked controllers** with poses and button/axis state,
3. a **compositor** that presents the frames somewhere.

So "emulate VR" reduces to two problems:

- **Injection:** supply HMD + controller poses and input from code. (§2)
- **Capture:** get the rendered frames back out as the robot's eye. (§4)

Locomotion, physics and the player's legs are *not* part of this interface — VR games abstract
them. That has a large consequence for the body mapping (§5).

---

## 2. Injection mechanisms (supply the body)

| Mechanism | Devices provided | Transport for poses/input | Platform | Verdict |
| --- | --- | --- | --- | --- |
| **Monado `remote` driver** | HMD + controllers | **TCP, documented structs** | Linux/Win | **Primary candidate** |
| SteamVR **null driver** | HMD only | none (no tracking, no controllers) | Win/Linux | Insufficient alone |
| **Custom OpenVR driver** | HMD + controllers | you choose | Win/Linux | Fallback / max control |
| **ALVR** server driver | HMD + 2 controllers | TCP/UDP to a client | Win/Linux | Viable, heavier |
| **VRChat OSC** | n/a | OSC `/input/*` | any | No emulation needed |
| Monado `simulated` | HMD + 2 controllers | **none — not drivable** | Linux/Win | Eliminated |
| Monado `qwerty` | HMD + controllers | keyboard/mouse (SDL events) | Linux/Win | Needs synthetic input |
| OpenVR-InputEmulator | virtual controllers | shared memory client lib | Windows-centric | Eliminated for Linux |

### 2.1 Monado `remote` driver — the primary candidate

Monado is an open-source, conformant OpenXR runtime ([monado.freedesktop.org](https://monado.freedesktop.org/)).

- **CITED:** the remote driver is a **TCP server**; `r_hub.c` performs `bind`/`listen`/`accept`,
  spawns a per-connection thread, and validates a magic header (`R_HEADER_VALUE = "mndrmt3\0"`)
  before accepting packets. Source: [`r_hub.c`](https://gitlab.freedesktop.org/monado/monado/-/raw/main/src/xrt/drivers/remote/r_hub.c).
- **CITED:** the wire format is defined in [`r_interface.h`](https://gitlab.freedesktop.org/monado/monado/-/raw/main/src/xrt/drivers/remote/r_interface.h):
  - `r_head_data` — per-view `fov` + `pose` (so the HMD pose **and** the eye projection are yours).
  - `r_remote_controller_data` — `pose`, `linear_velocity`, `angular_velocity`, `hand_curl[5]`,
    `trigger_value`, `squeeze_value`, `squeeze_force`, `thumbstick`, `trackpad`, plus every
    boolean: `system_click/touch`, `a_click/touch`, `b_click/touch`, `trigger_click/touch`,
    `thumbstick_click/touch`, `trackpad_touch`, `hand_tracking_active`, `active`.
- **CITED:** activation is a config file — `~/.config/monado/config_v0.json` with
  `{"active":"remote","remote":{"version":0,"port":4242}}`, or `P_OVERRIDE_ACTIVE_CONFIG=remote`
  (port then defaults to 4242). Source: [redstrate, "Using OpenXR without real hardware"](https://redstrate.com/blog/2023/11/using-openxr-without-real-hardware/).
- **CITED:** this is the driver WiVRn is built on (WiVRn's server is Monado-based and its README
  lists Monado as its runtime: [WiVRn README](https://raw.githubusercontent.com/WiVRn/WiVRn/master/README.md)).
- **ASSUMPTION:** `mndrmt3` is a version tag; the struct layout must match the exact Monado build
  you run. A mismatch should fail loudly on the header, but field-level drift is possible.

### 2.2 Monado drivers that are *not* usable

- **CITED:** `simulated` is documented in-source as *"Simple do-nothing simulated driver"* with only
  `SIMULATED_MOVEMENT_WOBBLE | ROTATE | STATIONARY`. Not configurable at runtime.
  Source: [`simulated_interface.h`](https://gitlab.freedesktop.org/monado/monado/-/raw/main/src/xrt/drivers/simulated/simulated_interface.h).
- **CITED:** `qwerty` *"enables basic control of a virtual HMD and controllers with keyboard and
  mouse"* via `QWERTY_ENABLE=1`. Source: [Developing with Monado](https://monado.freedesktop.org/developing-with-monado.html).
  Usable only by injecting synthetic input events — strictly worse than `remote`.
- **CITED:** other Monado drivers exist that may matter later: `solarxr` (SlimeVR's SolarXR
  protocol), `ht_ctrl_emu` (controller emulation), `sample` (driver template), `multi_wrapper`,
  `twrap`. Source: Monado repo tree `src/xrt/drivers/`.

### 2.3 SteamVR null driver

- **CITED:** provides an HMD with **no tracking and no controllers**, rendering into a desktop
  window. Configuration: `drivers/null/resources/settings/default.vrsettings` → `"enable": true`;
  `resources/settings/default.vrsettings` → `"requireHmd": false`, `"forcedDriver": "null"`,
  `"activateMultipleDrivers": true`. Source: [username223/SteamVRNoHeadset](https://github.com/username223/SteamVRNoHeadset).
- **CITED:** on Linux the same recipe applies, plus `EnableNullDriver = true`; register/remove
  third-party drivers with `SteamVR/bin/linux64/vrpathreg`. Sources:
  [ChristophHaag gist](https://gist.github.com/ChristophHaag/79587495848f34771a0a2589eb42f198),
  [Brown VR wiki](https://www.vrwiki.cs.brown.edu/hardware/vr-hardware/hardware-emulators/null-driver-tutorial).
- **CITED:** `ar-zadeh/VR-Emulator-Driver` replaces the null driver with one that emulates a
  **headset and two controllers**, driven by editing text files and pressing keyboard keys. 1 star,
  last push 2024-10; useful as a code reference, not as a dependency.
  Source: [repo](https://github.com/ar-zadeh/VR-Emulator-Driver).

### 2.4 Custom OpenVR driver

- **CITED:** Valve ships a driver tutorial whose end result is *"a virtual controller that takes the
  HMD position as its pose, and has some basic inputs"*. Source:
  [`samples/drivers/drivers/tutorial/README.md`](https://raw.githubusercontent.com/ValveSoftware/openvr/master/samples/drivers/drivers/tutorial/README.md).
- **CITED:** the input side is the `IVRDriverInput` interface. Source:
  [IVRDriverInput Overview](https://github.com/ValveSoftware/openvr/wiki/IVRDriverInput-Overview).
- **CITED:** Linux is proven: `SteamVR-OpenHMD` builds with CMake, registers with
  `vrpathreg adddriver <builddir>` where the directory contains `driver.vrdrivermanifest`.
  Source: [SteamVR-OpenHMD Readme](https://raw.githubusercontent.com/ChristophHaag/SteamVR-OpenHMD/master/Readme.md).
- **CITED:** pose-over-socket precedent: [VirtualMotionTracker](https://github.com/gpsnmeajp/VirtualMotionTracker)
  is a C++ OpenVR driver that *"receives the virtual tracker attitude on OSC and transfers the
  attitude to SteamVR"* ([how it works](https://gpsnmeajp.github.io/VirtualMotionTrackerDocument/howitworks/)).
  It creates trackers, not controllers.
- **CITED (negative):** the SlimeVR OpenVR driver is **currently broken on Linux**
  (`bridge error: No such file or directory`). Source: [SlimeVR-OpenVR-Driver #43](https://github.com/SlimeVR/SlimeVR-OpenVR-Driver/issues/43).
  Treat third-party OpenVR drivers on Linux as workable-but-finicky.

### 2.5 OpenVR-InputEmulator — eliminated

- **CITED:** it *"allows to create virtual controllers, emulate controller input, manipulate poses
  of existing controllers and remap buttons"*, with a shared-memory client library. But it *"hooks
  into the HTC Vive lighthouse driver"* and its last substantive commits are 2018. Source:
  [matzman666/OpenVR-InputEmulator](https://github.com/matzman666/OpenVR-InputEmulator).
  Not a Linux path.

### 2.6 ALVR — a virtual-device layer you could reuse

- **CITED:** the streamer runs on **Windows and Linux**; its SteamVR driver creates an HMD and left/right
  controllers (`DriverProvider` holds `hmd`, `left_controller`, `right_controller`, hand trackers);
  the client talks to the driver over **TCP or UDP sockets**, and video is streamed back to the client.
  Sources: [How ALVR works](https://github.com/alvr-org/ALVR/wiki/How-ALVR-works),
  [`alvr_server.cpp`](https://github.com/alvr-org/ALVR/blob/dc3abf3b/alvr/server_openvr/cpp/alvr_server/alvr_server.cpp).
- **ASSUMPTION:** writing a "fake ALVR client" fed by MuJoCo would reuse both the virtual devices and
  the video return path — attractive, but the client protocol and codec path are heavy compared to §2.1.

### 2.7 VRChat OSC — no emulation at all

- **CITED:** VRChat exposes an official **OSC input controller**: *"you could create a program to
  control VRChat with your head"* — `/input/*` messages drive movement, look, jump and more, plus
  avatar parameters. Source: [OSC as Input Controller](https://docs.vrchat.com/docs/osc-as-input-controller),
  [OSC DIY](https://docs.vrchat.com/docs/osc-diy).
- **CITED:** VRChat's Easy Anti-Cheat blocks **modified clients**, not OSC. Source:
  [VRChat Security Update](https://hello.vrchat.com/blog/vrchat-security-update).
- This is the cheapest route to "a real VR game being played", at the cost of VRChat being the game.

---

## 3. This machine already runs most of the stack

**MEASURED** (read from `/home/nate/Projects/nixos-config-public/modules/nixos/`):

- `modules/nixos/psvr2-vr.nix`: **Monado is enabled as the default OpenXR runtime**
  (`services.monado.enable = true; defaultRuntime = true`), built from Beyley's PSVR2 fork pinned at
  commit `47f9d39e`. NVIDIA env: `XRT_COMPOSITOR_COMPUTE = "true"`,
  `U_PACING_COMP_MIN_TIME_MS = "5"`, `VK_DRIVER_FILES = …nvidia_icd…`.
- Same module installs **`xrizer`** and **`opencomposite`**, and writes
  `~/.config/openvr/openvrpaths.vrpath` with `runtime = ["${pkgs.xrizer}/lib/xrizer"]`.
- `modules/nixos/steam.nix`: Steam + Proton + `gamescope`, 32-bit graphics, `protonup-ng`.

**CITED:** `xrizer` is *"a reimplementation of OpenVR on top of OpenXR. This enables you to run
OpenVR games through any OpenXR runtime without running SteamVR."* Source:
[Linux VR Adventures wiki](https://vronlinux.org/docs/fossvr/xrizer/). OpenComposite is the older
equivalent ([wiki](https://wiki.vronlinux.org/docs/fossvr/opencomposite/)).

**Consequence (ASSUMPTION):** OpenVR-only titles (Beat Saber included) can already reach Monado on
this machine with no SteamVR. Replacing the PSVR2 with a scripted device source is the entire delta.

---

## 4. Capture — the robot's eye

> **Measured 2026-10-23 (E2), and it contradicts this section's first row:** `XRT_DEBUG_GUI=1` and the
> compute compositor both **SIGSEGV** in the Wayland presentation queue-family check on this box. The
> working capture is **grim on the Monado window at 36.9 fps**, pose→pixel **30–59 ms (mid 44 ms)**.
> See `research/e2-spike-results.md`. The survey below is the 10-11 reasoning; grim is the answer.

| Path | Mechanism | Platform |
| --- | --- | --- |
| Monado debug-GUI mirror | `comp_mirror_to_debug_gui` blits composited frames into an SDL window; `XRT_DEBUG_GUI=1` | Linux/Win |
| SteamVR display mirror | ordinary desktop window | Linux/Win |
| OpenXR API layer | readback of submitted eye textures | Win (mature), Linux (DIY) |

- **CITED:** Monado's `comp_mirror_to_debug_gui` is a *"Helper struct for mirroring the compositors
  rendering to the debug ui, which also enables recording."* Source:
  [Monado docs](https://monado.pages.freedesktop.org/monado/structcomp__mirror__to__debug__gui.html).
- **CITED:** `OpenXR-Layer-OBSMirror` captures native OpenXR apps or the SteamVR/OpenVR compositor
  directly in OBS, *"can also use a wider, steadier recording camera while the headset continues to
  look and track normally"* — but the plugin and layer are Windows-oriented.
  Source: [repo](https://github.com/elliotttate/OpenXR-Layer-OBSMirror).
- **CITED:** `xr-sim` is *"a deterministic, headset-free OpenXR runtime… simulated head and
  controller poses, actions, session-state transitions, frame pacing, fault injection, scripted
  input, layer telemetry, and capture support"* — an existence proof for the whole idea, but
  **Windows x86/x64 only**. Source: [repo](https://github.com/phunkaeg/xr-sim).
- **CITED (negative):** SteamVR screenshot support on Linux has a long-standing bug.
  Source: [SteamVR-for-Linux #185](https://github.com/ValveSoftware/SteamVR-for-Linux/issues/185).

**ASSUMPTION:** window capture costs ≈1 frame plus pipeline latency, acceptable for a 10–30 Hz robot
camera. The end-to-end latency chain is the first thing to measure (§9).

---

## 5. The body mapping — the genuinely hard part

**MEASURED** (inspection of `/tmp/menagerie/unitree_g1/g1.xml`; consistent with `stubborn-gamer/SEED.md`):

- **29 actuated joints**: 12 leg, **3 waist** (`waist_yaw/roll/pitch`), 14 arm (7 per side:
  shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw).
- **No neck joint.** `head_link.STL` is a rigid mesh inside `torso_link`. The HMD orientation is
  therefore the **torso** orientation; the robot cannot glance without moving waist or whole body.
- Sites `imu_in_pelvis`, `imu_in_torso`, `left_foot`, `right_foot` (4 IMUs). **No camera.**
- `scene_with_hands.xml` = 43 DoF (adds 14 hand DoF).
- Verified policy (from `SEED.md`): Playground `g1_policy.onnx` runs headless on CPU, 0.5 m/s
  commanded → 0.516 m/s achieved, but **0.475 m of drift in 5 s at zero command**.

**The framing that matters:** VR games model the player as a *floating head plus two hands*.
Locomotion is abstracted (joystick/teleport), so **the legs never appear in-game**. "The G1 plays a
VR game" is therefore really *whole-body balance + waist/arm control realizing the poses the game
demands* — a whole-body-control problem, not a locomotion problem.

Two fidelity tiers, and only one is honest under `stubborn-gamer`'s observation rule:

- **Puppet mode (dishonest):** send desired hand/head poses directly to the game. The game accepts
  anything; the MuJoCo body is bypassed. Fine as scaffolding, but the robot is doing nothing.
- **Honest mode:** MuJoCo integrates torques; the wrist and torso poses *emerging from the
  simulation* are what gets sent to the game. This is the only version where the game is playing
  the robot's body rather than a marionette.

**ASSUMPTION:** the game applies **no forces** to the robot — a saber hitting a note does not push
the arm. Closing that loop (game contact events → external forces in MuJoCo) is an available
extension, not a given.

---

## 6. Game choice is bounded by the brain's latency

**MEASURED** (`stubborn-gamer/SEED.md`): the strategist models run **p50 5–20 s, p95 up to 41 s**;
the one fast model (LFM2.5-3B, 0.21 s) is degenerate (73% of answers `navigate_to`).

- **Rhythm games (Beat Saber): sub-100 ms reaction required — ASSUMPTION.** Impossible for a
  VLM-in-the-loop brain; viable only with a learned reactive policy. Beat Saber scoring is
  timing- and swing-angle-dependent ([ACC wiki: Time Dependence](https://wiki.accsaber.com/Beat_Saber/TD/),
  [precise scoring system](https://store.steampowered.com/news/app/620980/view/4036870438631348235)).
- **Sandbox / slow games** (Job Simulator, I Expect You To Die, Moss, VRChat) tolerate multi-second
  decisions. **This is where a VLM brain fits.**
- **CITED:** Beat Saber does run on Linux: [ProtonDB](https://www.protondb.com/app/620980/),
  [vronlinux DB](https://db.vronlinux.org/games/620980.html) — with caveats: SteamVR 2.10.4 crashes
  newer builds ([SteamVR-for-Linux #786](https://github.com/ValveSoftware/SteamVR-for-Linux/issues/786)),
  and Monado needs `OXR_PARALLEL_VIEWS=0`.
- **ASSUMPTION:** "headless" here means *no HMD*, not *no display server*. The game and compositor
  still need a GPU-backed display (Wayland/Xvfb/gamescope).

---

## 7. Prior art

**CITED — Robo-Saber** (Eurographics 2026) is the closest work that exists: *"the first motion
generation system for playtesting virtual reality (VR) games. Our player model generates VR headset
and handheld controller movements from in-game object arrangements, guided by style exemplars and
aligned to maximize simulated gameplay score."* Sources:
[project page](https://robo-saber.github.io/), [arXiv 2602.18319](https://arxiv.org/abs/2602.18319),
[code](https://github.com/namheegordonkim/robo-saber) (Linux x86-64, CUDA, includes
"physics-based tracking code").

Two facts about it matter:

1. It **simulates** Beat Saber gameplay; it does not run the game binary.
2. Its own stated limitation: *"degraded gameplay performance due to physics-based tracking"* and
   motions that *"remain far from human-like or realistic."*

**CITED (inverse direction):** VR→robot teleoperation is well-trodden — e.g.
[XRoboToolkit-Teleop-Sample-Python](https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python)
(Quest controllers → MuJoCo robot arms) and
[vr-teleop-kit](https://github.com/Dream-Machines-Robotics/vr-teleop-kit). The IK machinery is the
same, run backwards.

**No published instance of a physics-simulated humanoid body playing a real commercial VR game.**

---

## 8. Recommended architecture

```text
MuJoCo G1 (29 DoF, no neck)
  ├─ torso_link pose  ─────────────► HMD pose + per-view FOV ┐
  ├─ wrist_link poses ─────────────► 2 controller poses      │  mndrmt3 packets over TCP :4242
  └─ task policy      ─────────────► trigger/grip/thumbstick ┘
                                                             ▼
                                                Monado + `remote` driver
                                                             ▼
                              game (native OpenXR, or OpenVR title via xrizer)
                                                             ▼
                        Monado debug-GUI mirror → window capture → robot camera
```

Rationale: native Linux, no SteamVR, protocol verified in source, and §3 shows the machine already
runs Monado + xrizer. SteamVR + a custom OpenVR driver remains the fallback if a title demands
SteamVR specifically.

---

## 9. First path — undecided, with a path-agnostic first experiment

The choice was left open. Here is the honest comparison.

| Option | Cost | Proves | Main risk |
| --- | --- | --- | --- |
| **Monado `remote` + fake client** | medium | the core mechanism, on Linux, for any OpenXR/OpenVR title | packet ABI vs pinned fork; debug-GUI capture with the compute compositor |
| **VRChat OSC** | low | "a real VR game is being played" end-to-end soonest | VRChat only; bypasses the emulation problem entirely |
| **Custom OpenVR driver** | high (C++) | max portability + HMD/controller control | Linux driver friction (SlimeVR #43) |

**Decided 2026-10-23:** Monado `remote` + a fake client. E1/E2/E3 measured it end to end (376-byte
packets over TCP 4242, poses identical to `mjData`, a real game bound to it in C0). The table's stated
risk — debug-GUI capture with the compute compositor — was real: that path SIGSEGVs, and grim is the
substitute. Its other risk, packet ABI drift against the pinned fork, did not materialise.
| **Reimplement the game in sim** | medium | full control, matches Robo-Saber | arguably no longer "playing a VR game" |

**Do this first regardless of the choice** (it is the load-bearing unknown for the primary path and
costs hours, not days):

> **Experiment E1 — Monado remote pose round-trip.**
> Activate the `remote` driver, connect a throwaway client over TCP, and stream one HMD pose.
> **Pass:** the game's rendered view moves as the scripted pose moves. **Fail:** header/ABI
> rejection, or no effect — which would redirect the project to the custom-driver path before any
> robot work is spent.

Then, in order:

- **E2 — capture:** mirror Monado's composited frame into a window and read it back at ≥10 Hz; record
  the measured latency.
- **E3 — honest body:** feed the *simulated* G1 wrist/torso poses (not desired poses) into E1's client
  and confirm the game follows the physics body.
- **E4 — task:** pick the game class per §6 and attempt the first in-game objective. See §11 for the
  game/direction analysis that should inform this choice.

---

## 10. Risks and open questions (test, don't argue)

1. **ABI drift:** does the `mndrmt3` layout match the pinned Beyley fork commit? (E1 answers this.)
2. **Build flags:** are `remote` and the debug GUI compiled into that Monado, and do they coexist
   with `XRT_COMPOSITOR_COMPUTE = "true"`?
3. **Display:** confirm the game + compositor run on a GPU-backed display without an HMD
   (Wayland/Xvfb/gamescope), on NVIDIA, on NixOS.
4. **Latency:** measure the full chain (game frame → capture → perception → action → pose → game frame).
   This bounds the playable game class (§6).
5. **ToS:** automating online leaderboards (e.g. ScoreSaber) is a rules question; local/offline play
   is not. Not legal advice.
6. **Licensing:** Monado is BSL-1.0; ALVR MIT; OpenVR-InputEmulator GPL-3.0; xrizer — check before
   vendoring.

---

## 11. Games and directions

### 11.1 The interface insight that unlocks the legs

Most VR games reduce locomotion to a **2D velocity command on the thumbstick**. The G1's pretrained
Playground policy takes **exactly that** — its observation vector is
`linvel(3) gyro(3) gravity(3) command(3) joint_angles(29) …`. So `game thumbstick → locomotion
policy command` is a near-1:1 mapping. **ASSUMPTION:** the legs are therefore not a mismatch but a
solved channel.

Buttons are free (emulated, not physical). The only channel the G1 cannot natively produce is
**hand curl** (`hand_curl[5]` in Monado's remote struct) — that needs `scene_with_hands.xml` or gets
faked. Net: **the game supplies the velocity command, the arms supply the skill, and balance is the
constraint nobody else is solving.**

### 11.2 Candidate games

Grouped by demand profile. Linux compatibility is **settled for all 13 candidates — see §12.5**;
the unresolved question is whether any can be driven with a *synthetic* HMD.

| Game | Demands | Slow-brain fit | Notes |
| --- | --- | --- | --- |
| **I Expect You To Die** | Seated puzzles, hands only, no locomotion | high | Tolerant of 20 s decisions. Best pure "brain" fit. |
| **Job Simulator** | Manipulation, tolerant | high | Grab = grip button; no fingers needed. |
| **The Room VR** / **A Fisherman's Tale** | Puzzle, inspection | high | Very tolerant. |
| **Walkabout Mini Golf** | Precision arm, real physics, turn-based | high | Objective score, no locomotion; open right-eye distortion bug on SteamVR/Linux (§12.6). |
| **Moss** | Third-person, controller-driven | high | Robot body barely matters. |
| **VRChat** | Social, avatar params | high | **Official OSC `/input/*` API** — zero emulation (§2.7). |
| **Superhot VR** | Punch, aim, dodge, **time = motion** | medium | See §11.4. |
| **Blade & Sorcery** | Melee swing physics, moddable | medium | Swing velocity matters. |
| **Bonelab / Boneworks** | Physics sandbox, mods | medium | Permissive modding; good authoring surface. |
| **Pistol Whip** | Aim + trigger + **whole-body dodge** | low | Dodge is a balance task; rhythm timing. |
| **Until You Fall** / **Synth Riders** | Rhythm melee | low | |
| **Eleven Table Tennis** | Real-time physics skill | none | Human reaction times. Brutal. |
| **Beat Saber** | Saber pose + timing, sub-100 ms | none | Legible score; prior art (Robo-Saber, §7). Needs a learned policy. |
| **Half-Life: Alyx** | Long-horizon, shooting, physics puzzles | medium | **Native Linux.** Thumbstick locomotion. Best fit for the long-horizon thesis. |
| **Gorilla Tag** | **Arm locomotion** (no legs), social | medium | Locomotion *is* the arm DoF. Multiplayer jank. |
| **Stride / Sprint Vector** | Arm-swing parkour | low | |
| **To the Top / Climbey** | Climbing, reach | low | Tests arm reach limits. |
| **Resonite / VRChat Udon / Rec Room** | In-game scripting (LogiX, Udon, Circuits) | high | Author the task *inside* VR, with a real API. |

### 11.3 Directions

1. **VR-Bench** — commercial VR games as an embodied-agent benchmark. VR is nearly absent from agent
   benchmarks; a harness + N games + pre-registered metrics is a real artifact.
2. **Reactive player policies** — IL from BOXRR-23 + RL with the game score as reward. Extends
   Robo-Saber to a *physics* body, which is precisely the gap its limitations section names (§7).
3. **Games as reward functions for balance-constrained whole-body control** — the game supplies
   desired end-effector trajectories; balance is the constraint; the score is the reward. A
   human-designed reward with no reward engineering.
4. **Closed force loop (haptic grounding)** — game contact events → external forces in MuJoCo. Right
   now a saber passing through a note pushes nothing; closing it makes "honest mode" fully honest (§5).
5. **Long-horizon autonomy in a human-designed world** — Alyx or a puzzle game with a VLM brain:
   memory, re-planning, bounded retry. The `stubborn-gamer` thesis, tested in an environment designed
   by humans to mess with humans.
6. **Synthetic playtesting data** — a physics-based player model generates more realistic playtest
   data than a kinematic one.
7. **Social embodiment** — a robot as a player in VRChat. Legible and fun; hard to measure.

### 11.4 Assessment

**Most interesting flagship: Superhot VR**, as the vehicle for direction #3 with the force loop (#4)
closed.

Superhot's mechanic is *time moves only when you move*. That turns the robot's hardest and most
under-appreciated problem — **holding a stance while thinking, on a base that drifts 0.475 m in 5 s
at zero command** (MEASURED, §5) — into the literal win condition. `stubborn-gamer` §6 defect #7
requires that any design gating on distance "own a hold/stance mechanism"; Superhot *is* that
mechanism, gamified. It is also the rare commercial action game that tolerates the measured 5–20 s
p50 decision latency, because the world only advances when the body does.

It exercises the G1's whole DoF spread rather than one subsystem: punches (arms), aim (arm + wrist),
dodge (waist + legs + balance), locomotion (thumbstick → velocity command → Playground policy). And
it is objective — level completion, kills, time. **No prior art:** Robo-Saber chose Beat Saber; nobody
has chosen a game whose *clock is driven by the body*.

**Caveats (ASSUMPTION):** once motion is committed, bullets move at normal speed, so dodge speed and
arm reach become binding — the G1's arm authority is limited (MEASURED: `left_shoulder_pitch` moves
the end effector 0.151 m) and Superhot levels assume human reach. Its Linux status is now **verified**
(§12.6).

**First game milestone: Superhot VR too — revised from this section's earlier draft.** The earlier plan
was to prove the pipe on a forgiving turn-based game (Walkabout, I Expect You To Die) first. The §12.5
per-game check kills that plan: Walkabout has an open right-eye distortion bug on SteamVR/Linux, and
IEYTD has documented OpenXR-init and controller-manifest quirks. Superhot is the **best-verified**
target on this exact stack (ProtonDB: "works with Envision/Monado/xrizer"), needs no game API because
it is driven through the emulated devices, and its clock already tolerates slow decisions. The
forgiving-game detour would cost a milestone and buy nothing.

Keep **E1** on a trivial OpenXR sample (Monado's own test app, or a `hello_xr` build), **not** on a
game — that isolates the packet/ABI question from game-specific behaviour.

**Parallel cheap track — with a caveat.** VRChat OSC (`/input/*` + `/tracking/trackers/*`) or Resonite
(`ResoniteLink`) can demonstrate "a real VR game is being played" end-to-end with no driver emulation.
But this proves the *task* loop, **not the emulation path**: it exercises a sanctioned API instead of
E1, so it cannot substitute for it. (An earlier revision of this document proposed Resonite as the
plumbing test. That was wrong — it bypasses the emulated-device path entirely.)

---

## 12. Integration: control, world reconciliation, per-game reality

Three parallel research briefs (2026-10-11) covered how the G1 body, the two worlds, and the per-game
surface actually connect. Distilled below; every claim is CITED.

### 12.1 Control — how the body tracks game targets

**The central finding: do not bolt an arm controller onto the locomotion policy.** Every credible
system that tracks external head/hand poses on a humanoid **retrains or fine-tunes a single
whole-body policy whose command space includes upper-body pose**. The failure mode is documented on
this exact robot: Unitree's `rt/arm_sdk` runs arm control on top of a balance controller that is
unaware of the arms, and integrators report forward lean, torso oscillation and an infinite stepping
loop when arms are held out ([Robotics Knowledgebase](https://roboticsknowledgebase.com/wiki/common-platforms/unitree-g1/),
[unitree_sdk2_python #173](https://github.com/unitreerobotics/unitree_sdk2_python/issues/173)).

| Pattern | How it works | Evidence |
| --- | --- | --- |
| Intervention / action override | Sample alternative upper-body actions during training (p=0.005) so the policy is robust to *any* external upper-body signal; also improves locomotion tracking | HUGWBC |
| Retrain with an upper-body command space | command = `[v_x, ω_yaw, h_torso]` + arbitrary upper-body joint poses, with an upper-body pose curriculum | HOMIE, ExBody/ExBody2 |
| Decoupled, with a lever arm | Hand targets go straight to PD, bypassing the policy | HumanPlus; **Unitree `rt/arm_sdk` is this pattern and is the documented failure** |
| Residual RL on a frozen base | `a_t = a_t^b + a_t^r` | ASAP, FAST; RobotDancing warns residual authority belongs on hip/knee-pitch only — do not ask residuals to fix 14 arm joints |

Mode-masking is the cheapest version: HOVER distills one policy with independent upper/lower sparsity
masks, giving >15 modes including exactly *"upper-body kinematic tracking + lower-body velocity
tracking"* ([HOVER](https://arxiv.org/abs/2410.21229)).

**Recommended staging:** layered now (rate-limited joint-space IK for the arms + conservative waist
yaw + untouched leg actions + velocity-command bias), retrain later. OmniH2O's ablation (3-point ≈
22-point tracking) is the evidence that head+hands alone is a sufficient interface
([OmniH2O](https://arxiv.org/html/2406.08858)).

**Head-look with no neck.** The H1 has the same limitation, and OmniH2O still tracks a head keypoint
from 3-point VR input. Standard practice is **yaw substitution**: waist yaw takes a bounded share and
the remainder becomes a base-yaw steering term. HRP-2 teleoperation found head yaw beyond ~40°
*relative to the waist* triggered stepping ([Steering a Humanoid Robot by Its
Head](https://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/IEEE_IROS_2009/papers/0305.pdf)).
**The fidelity ceiling is structural:** yaw substitution cannot realize HMD roll/pitch without moving
the torso off the balance policy's happy path. Expect a documented, app-visible head-pose residual.

**Tooling.** `mink` (differential IK on MuJoCo: `FrameTask`/`ComTask`, joint limits, collision,
`qpsolvers`) or `pink` (Pinocchio + `qpsolvers`; ships a free-flyer humanoid example with CoM, torso
and fixed-foot tasks at 200 Hz; Linux x86_64 only). Raw MuJoCo has **no IK solver**: `mj_jacSite`/
`mj_jacBody` + damped least squares + `mj_integratePos`.

**Rates.** Dominant pattern is policy 50 Hz / PD 500–1000 Hz. TeleGate runs exactly that on a 29-DoF
G1 in MuJoCo with a 37.5 Hz low-pass between them. **Never run IK in the 500 Hz loop** — solve on the
target tick, hold the joint target, let PD interpolate, add velocity feedforward (η 0.8–0.9; η=1
overshoots) and rate-limit.

**Integration gotchas (severity).**

- **HIGH — action scale is not portable:** `unitree_rl_gym` uses `action_scale = 0.25` for the 12-DoF
  G1 while Playground metadata gives 0.5. A wrong scale halves or doubles every leg action. Read it
  from the exported manifest.
- **HIGH — waist ownership is ambiguous:** Unitree's `g1_arm5_sdk_dds_example.py` lists
  `WaistYaw/Roll/Pitch` in the arm_sdk joint set while `xr_teleoperate` defines the arm index as arms
  only. Decide and gate explicitly.
- **MEDIUM —** the G1 has PR vs AB ankle modes; ASAP trained a 4-DoF *ankle-only* delta model for
  exactly this sim-to-real gap.
- **MEDIUM —** Menagerie `unitree_g1/g1.xml` ships position actuators marked "needs tuning"; use
  `scene_mjx.xml` gains for Playground-consistent behaviour.

### 12.2 Reconciling the two worlds

No published system does exactly this. The three adjacent solved problems are avatar telepresence
between dissimilar rooms, humanoid VR teleoperation (HOMIE), and bilateral telelocomotion.

**Locomotion divergence — and the fix.** Walk-in-place plus game-driven translation is the established
reconciler; the clearest precedent freezes the avatar's position and teleports it when the user stops,
so physical displacement is never mapped to virtual displacement ([arXiv
2103.04380](https://ar5iv.labs.arxiv.org/html/2103.04380)). A foot-tapping WIP humanoid interface
achieved **0.47 m average tracking error** — the same order as the G1's measured 0.475 m/5 s drift, a
useful sanity scale.

> **The cheapest reconciler inverts the dependency: close the velocity loop on the robot side.**
> Command the game a thumbstick value derived from *achieved* base velocity, so the game's camera
> translation is driven by the robot's real motion rather than the reverse. The Playground policy's
> measured gain is 0.516/0.5 = **1.03**, so the mapping is nearly unity. No base teleporting required.

Base resync is mechanically easy (write the free-joint `qpos`, or make the root a mocap body and set
`mjData.mocap_pos`/`mocap_quat`) but **teleporting injects a step change in `qvel`, which lands
directly in the policy's `linvel(3)`/`gyro(3)` observations and can destabilise it.** Prefer
rate-limited or footstep-aligned correction.

Redirected-walking perceptual thresholds are irrelevant to a robot (no cybersickness) —
**[ASSUMPTION]**, the researcher's inference, not a literature finding.

**Frames.** OpenXR: right-handed, metres, quaternion `(x,y,z,w)`, pose = orientation then position;
`LOCAL` is gravity-aligned (pitch/roll excluded, yaw-locked), `STAGE` adds a floor origin, `VIEW` is
not gravity-aligned. MuJoCo: right-handed, Z-up, X forward, Y left, quaternion `wxyz`. Pitfalls:
**quaternion order differs**; MuJoCo 6D velocities are **mixed-frame** (linear in world, angular in
body); MJCF `eulerseq` defaults to `"xyz"` while URDF `rpy` is `"XYZ"`. Monado's `r_head_data.center`
is *view space*, with per-view FOV and per-view pose relative to it, and the hub applies
`xrt_tracking_origin::initial_offset` — **you must invert whichever reference space the game requests
(LOCAL vs STAGE differ by a floor offset and a yaw) or you get persistent yaw/height bias.** Tracking
loss keeps a last-known pose while clearing the `TRACKED` bit, so the client must read the valid bits:
a frozen pose is otherwise indistinguishable from a stationary robot.

**Infeasible targets.** The published answer is *retarget to the nearest feasible configuration* —
not clamp and not refuse (SEIKO: a QP rewriting Cartesian commands into a feasible posture plus
contact wrenches, with anti-windup clamping around the **achievable** pose, bang-bang velocity/accel
bounds, and admittance with a deadband). SEIKO is quasi-static and will fight a dynamic locomotion
policy. **The key interface lesson:** CBF-WBC silently overrides infeasible references and flags that
*"the planner should be informed accordingly"* — **your bridge must be told when a target was
infeasible, or the game's HMD pose silently diverges from the robot's torso with no feedback path.**

The strong version: do not filter at runtime — train the policy with an upper-body pose curriculum so
the feasible set is large (HOMIE).

### 12.3 Force loop

The only *standard* game→XR channel is haptic output, and it is game→runtime, not runtime→game:
`xrApplyHapticFeedback` gives at most a scalar amplitude + duration per hand — an extremely lossy
contact proxy — and only if the runtime is instrumented, since an OpenXR API layer sits between game
and runtime. [XR-Telemetry](https://github.com/mledour/XR-Telemetry) is a working layer template that
also yields per-frame latency telemetry.

Game mods are the high-bandwidth option but per-game and fragile (UEVR reflection, `uevr-mcp`). The
clean event channel exists only in open-source engines — ViZDoom exposes health/position/reward/actor
lists and runs headless — and a commercial VR title gives you none of it. Pixel inference is the
game-agnostic fallback (P2P0.1 runs at 20 Hz on one consumer GPU), but a black-box replication
benchmark shows agents recover visual appearance far more reliably than *mechanics*, so expect
contact-event inference to be the weak link.

**No prior work found** on feeding forces derived from a commercial VR game back into a physics
simulator — a negative search result, not proof of absence. If built, the classical recipe is
penalty/spring `F = K·d·n` with a virtual proxy, mapped to MuJoCo `xfrc_applied`/`qfrc_applied` on the
hand link.

### 12.4 Latency

Monado's frame pacing is predictive (`u_pa_predict`), and `u_pa_latched` **may be called multiple
times for the same frame when the app's cadence is below the compositor's** — so **a slow or stalled
pose client manifests as the game holding a stale pose, not as a dropped frame.** The bridge must
*detect staleness*; it will get no back-pressure. Read the game's submitted `predictedDisplayTime` and
`predictedDisplayPeriod` to align injection with the frame it will actually be used for.

Measured motion-to-photon baselines: **21–42 ms** at sudden movement onset, functionally **2–13 ms**
once prediction engages; Vive Pro 2 @90 Hz 41.7 ms; Pico 4 49.2 ms; a minimal low-level OpenXR app
~31–32 ms. In XR teleoperation, network delay dominates and error grows with RTT (100 ms → 1000 ms),
with human tolerance degrading around **+200 ms**. Measurement harness: XR-Telemetry's per-frame CSV
(`app_cpu_ns`, `render_ns`, `end_frame_ns`, `frame_total_ns`, `gpu_time_ns`, `headroom_pct`, and the
`post_end_ns` residual = time outside OpenXR calls) — with its own caveat that a passive layer sees
only the OpenXR-calling thread.

### 12.5 Per-game reality check

**All 13 candidate titles run acceptably on Linux** (12 via Proton; Alyx has a native Vulkan build) —
the Linux question is largely settled. But **every** community report assumes a real streaming
headset. The actual blocker is the HMD: OpenXR runtimes such as SteamVR **refuse to initialise when
no HMD is detected** (`xrCreateInstance` fails; Unity's OpenXR plugin loops restart errors). The
synthetic HMD must therefore be supplied *below* the game.

- **The correct injection point is SteamVR `TrackingOverrides`** — a custom driver overriding
  `/user/head`, `/user/hand/left`, `/user/hand/right` "at the system's raw pose level". Valve states
  explicitly that overriding poses inside the app (`WaitGetPoses`) is **not** equivalent, because only
  the driver-level route feeds SteamVR's forward-prediction engine.
- SteamVR null driver: current status **unverified** (a 2020 report describes a "no headset detected"
  gate blocking controller pairing in that configuration).
- OpenVR-InputEmulator: Windows-era, unmaintained; Valve engineers warn hook drivers are "a major
  source of instability".
- `callum1h3/SteamVR-Unity-Integration` replaces `driver_null` so an external process writes
  HMD/left/right poses, and claims anti-cheat bypass. **Treat that claim as the author's assertion —
  unverified, and the Gorilla Tag record contradicts it.**
- `xr-sim` and fholger's OpenXR-Simulator are **Windows-only**.
- Proton↔host runtime needs `PRESSURE_VESSEL_IMPORT_OPENXR_1_RUNTIMES=1` (already set in
  `psvr2-vr.nix`); OpenVR games under Proton need xrizer/OpenComposite.

**Telemetry — for evaluation, never for control.** Beat Saber has by far the richest ecosystem:
`beatsaber-http-status` (ws://localhost:6557, with a `noteSpawned` event), BSDataPuller
(ws://localhost:2946, `MapData`/`LiveData` including per-note cut results), SocketSaber and
BS_AnotherStatusSocket. Its replay formats (BSOR / legacy ScoreSaber) encode exactly the per-frame
(head, left hand, right hand) pose stream this project needs. VRChat OSC exposes `/input/*`, avatar
parameters, and **`/tracking/trackers/head|1-8`** position + rotation (translation frame-exact, head
yaw slowly lerped; 300 ms single-message / 10 s timeout). Resonite exposes a `ResoniteLink` WebSocket
JSON API with realtime read/write and an official Linux headless client. No state API was found for
Superhot, Walkabout, IEYTD, Job Simulator, Pistol Whip or Eleven — a **negative search result**.

**Fidelity nuance:** reading game telemetry is legitimate for *evaluation* (it is the scoreboard) but
must never be a control input, or the robot is reading privileged state.

**Anti-cheat.** Only VRChat ships kernel-level EAC — it blocks client modification and injection, but
**OSC is unaffected**. Gorilla Tag runs server-side anti-cheat plus behavioural enforcement (bans for
mods in public *and* private lobbies; long-arms blocked Oct 2024) → **ban risk, avoid**.

### 12.6 Corrections to §11

- **Superhot VR's Linux status is verified**, not unknown: ProtonDB reports "works with
  Envision/Monado/xrizer" — the exact stack on this machine.
- **Walkabout Mini Golf has an open right-eye distortion bug above 100% render scale** on
  SteamVR/Linux ([SteamVR-for-Linux #872](https://github.com/ValveSoftware/SteamVR-for-Linux/issues/872));
  WiVRn reporters are unaffected. This is why §11.4 no longer proposes it as the first milestone.
- **I Expect You To Die** has documented OpenXR-init and controller action-manifest problems, which
  removes it as the "forgiving first game" for the same reason.
- **Superhot VR is the best-verified target on this stack** (ProtonDB: "works with
  Envision/Monado/xrizer"), which is why §11.4 promotes it from flagship-only to first game milestone.
- §11.2's "verify Linux" caveat is now largely resolved: all 13 run on Linux. The unresolved question
  is **whether any can be driven with a synthetic HMD** — no confirmed report exists for any of the 13,
  which is exactly what E1 tests.
- **Uncertainty register:** "HRL / Hybrid Robotic Transformer" could not be verified — do not cite it
  (nearest real systems: Figure Helix, NVIDIA HOVER/GR00T). G1 waist joint limits are unverified (the
  ±40° figure is HRP-2's). Monado's `r_interface.h` now carries a 2026 NVIDIA copyright, so diff your
  pinned revision against the published docs. The Unity+G1 VR paper (arXiv 2604.16903) has an anomalous
  identifier — verify before citing.

---

## Sources

- Monado: [site](https://monado.freedesktop.org/) · [developing](https://monado.freedesktop.org/developing-with-monado.html) · [`r_interface.h`](https://gitlab.freedesktop.org/monado/monado/-/raw/main/src/xrt/drivers/remote/r_interface.h) · [`r_hub.c`](https://gitlab.freedesktop.org/monado/monado/-/raw/main/src/xrt/drivers/remote/r_hub.c) · [`simulated_interface.h`](https://gitlab.freedesktop.org/monado/monado/-/raw/main/src/xrt/drivers/simulated/simulated_interface.h) · [debug GUI mirror](https://monado.pages.freedesktop.org/monado/structcomp__mirror__to__debug__gui.html)
- No-hardware guide: [redstrate](https://redstrate.com/blog/2023/11/using-openxr-without-real-hardware/)
- SteamVR null driver: [SteamVRNoHeadset](https://github.com/username223/SteamVRNoHeadset) · [Brown VR wiki](https://www.vrwiki.cs.brown.edu/hardware/vr-hardware/hardware-emulators/null-driver-tutorial) · [Linux gist](https://gist.github.com/ChristophHaag/79587495848f34771a0a2589eb42f198) · [VR-Emulator-Driver](https://github.com/ar-zadeh/VR-Emulator-Driver)
- OpenVR drivers: [tutorial](https://raw.githubusercontent.com/ValveSoftware/openvr/master/samples/drivers/drivers/tutorial/README.md) · [IVRDriverInput](https://github.com/ValveSoftware/openvr/wiki/IVRDriverInput-Overview) · [SteamVR-OpenHMD](https://raw.githubusercontent.com/ChristophHaag/SteamVR-OpenHMD/master/Readme.md) · [VirtualMotionTracker](https://gpsnmeajp.github.io/VirtualMotionTrackerDocument/howitworks/) · [OpenVR-InputEmulator](https://github.com/matzman666/OpenVR-InputEmulator) · [SlimeVR Linux issue](https://github.com/SlimeVR/SlimeVR-OpenVR-Driver/issues/43)
- ALVR: [How it works](https://github.com/alvr-org/ALVR/wiki/How-ALVR-works) · [`alvr_server.cpp`](https://github.com/alvr-org/ALVR/blob/dc3abf3b/alvr/server_openvr/cpp/alvr_server/alvr_server.cpp)
- OpenXR→OpenVR bridges: [xrizer](https://vronlinux.org/docs/fossvr/xrizer/) · [OpenComposite](https://wiki.vronlinux.org/docs/fossvr/opencomposite/)
- Capture: [OpenXR-Layer-OBSMirror](https://github.com/elliotttate/OpenXR-Layer-OBSMirror) · [xr-sim](https://github.com/phunkaeg/xr-sim) · [SteamVR-for-Linux #185](https://github.com/ValveSoftware/SteamVR-for-Linux/issues/185)
- VRChat: [OSC input controller](https://docs.vrchat.com/docs/osc-as-input-controller) · [OSC DIY](https://docs.vrchat.com/docs/osc-diy) · [security/EAC](https://hello.vrchat.com/blog/vrchat-security-update)
- Beat Saber on Linux: [ProtonDB](https://www.protondb.com/app/620980/) · [vronlinux DB](https://db.vronlinux.org/games/620980.html) · [SteamVR-for-Linux #786](https://github.com/ValveSoftware/SteamVR-for-Linux/issues/786) · [scoring](https://store.steampowered.com/news/app/620980/view/4036870438631348235) · [time dependence](https://wiki.accsaber.com/Beat_Saber/TD/)
- Prior art: [Robo-Saber](https://robo-saber.github.io/) · [arXiv 2602.18319](https://arxiv.org/abs/2602.18319) · [code](https://github.com/namheegordonkim/robo-saber) · [XRoboToolkit teleop](https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python) · [vr-teleop-kit](https://github.com/Dream-Machines-Robotics/vr-teleop-kit)
- Local: `stubborn-gamer/SEED.md`, `stubborn-gamer/bench/`, `nixos-config-public/modules/nixos/{psvr2-vr,steam}.nix`, `/tmp/menagerie/unitree_g1/g1.xml`
- Full integration briefs (verbatim, source of §12): `research/g1-integration-control.md`,
  `research/g1-integration-world.md`, `research/g1-integration-games.md`
