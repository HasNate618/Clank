<p align="center">
  <img src="docs/media/title-card.svg" width="608" alt="Clank - probationary unit">
</p>

# Clank

**A Unitree G1 humanoid simulated in MuJoCo, a released NVIDIA whole-body controller, and a commercial VR game that renders from head and hand poses we send through Monado. No headset, no controllers, no game files touched.**

Clank is a real-time controls and systems-integration project that uses a commercial VR game as an adversarial task interface. The game is the framing device. The work is a 29-DoF humanoid at 50 Hz behind an undocumented socket, where every claim is measured and every failure is written down.

> **The scoreboard, up front.** A commercial VR game accepts a headset that does not exist, and one simulated humanoid now drives both the headset and the two controllers. A G1 in MuJoCo walks 39.79 m in 60 s without falling and holds to RMS 1.3×10⁻⁴ m/s; the same body's achieved poses are relayed into Superhot at 50 Hz, where commanding `look_left` rotates the game's view by **28.6°**.
>
> **The two halves are joined.** `body_view → body_state.json → pose_relay → Monado :4242 → game`, one simulation and no retargeting, so the body tile and the game show the same robot rather than two lookalikes. Co-residency is measured: beside the running game the body holds **44.2 Hz** against **44.5 Hz** alone, for **+177 MiB** of GPU memory.
>
> **The hand closes on a real object now.** On 2026-11-12 the body reached a card in Job Simulator along a
> **generated** path — an end-effector constraint into ARDY-G1, 52 frames, tracked by SONIC — closed its hand
> 78 mm out, and still held it 300 ticks later (glove residual **24.7 mm**; `research/evidence/ardy-live-grab/`).
> That is a scripted reach, not play.
>
> **Still not true: nobody has played it.** No level has been cleared, the body does not walk in-game (it is
> clip-anchored: one generated window reaches, it does not step — the escalation that would walk it has never
> converted), and the torso does not turn to look at what it reaches for (60–64° off, measured four ways and
> recorded as a limitation, not a bug). With the game running the simulation advances at **0.89× real time**.
> Everything below is labelled with what is proven, what is not, and where the number came from.

<!-- ![clank dashboard](docs/dashboard.png) -->

---

## What this demonstrates

- **Real-time humanoid control.** 29 DoF (12 legs, 3 waist, 14 arms; no fingers, no neck), MuJoCo 3.17 MJCF physics, a 50 Hz policy with 500 Hz PD, and a measured **17.66 ms** of work per 20 ms control tick (encoder 7.54, decoder 5.42, physics 4.71).
- **Integrating a released learned controller.** NVIDIA GR00T-WholeBodyControl / GEAR-SONIC, the released ONNX weights, encoder 1762→64 and decoder 994→29 through ONNX Runtime — plus a real performance bug found by measurement: onnxruntime opened one intra-op thread per core and consumed 1876 % CPU across 47 threads, load average 21, for two networks that small.
- **Systems triage across a hostile five-layer stack.** Wine/Proton → OpenXR loader → Monado → hidden compositor → game. The working path was found by diagnosing each layer separately: a runtime pointer invisible inside the container, a loader that caches its manifest list per process, and a Wine shim that forwards Windows OpenXR calls to the host loader.
- **A designed command contract, not an ad-hoc pipe.** A masked task-space `BodyCommand` / `BodyState` protocol over a Unix socket: authority bitmask, TTL on acceptance, monotonic sequencing, latest-wins sample-and-hold, non-preemptible ballistics, and strict validation that never raises into the control loop. Standard-library Python only.
- **Joining a simulator to a commercial binary.** One simulation feeds both the operator's view and the game: achieved `mjData` poses are lifted out, root-composed, frame-changed from Z-up to Y-up, and relayed as 376-byte packets. The subtlety is that the body's poses are root-relative and the root carries its own rotation, so skipping the composition leaves the heading wrong while every position still looks plausible — there is a quaternion test that fails if it is skipped.
- **Process supervision and a reproducible environment.** One command brings up Monado, a hidden Weston VNC compositor, websockify, Steam, the game and the pose client as supervised process groups, in a session you can see in a browser and kill cleanly. NixOS flake devShell; 323 unit tests, no network, no GPU.
- **Measuring instead of assuming.** Latency budgets, an A/B pixel proof that the view really follows our poses, and documented negative results — two of this project's own hypotheses falsified, one harness bug caught and corrected, and two UI verbs (`look left`, `dodge`) that had never been measured and were wrong: the first turned the robot right, the second moved 1 cm on the wrong axis.

---

## Architecture

Two pipes. The game pipe runs today. The body pipe is measured today. The edge between them is the missing piece, and it is drawn as missing.

```text
ONE SIMULATION, TWO CONSUMERS

  BodyCommand (JSON lines, Unix socket) → CommandBus → SonicExecutor
    → MuJoCo G1, 29 DoF, 50 Hz policy / 500 Hz PD
    ├─ achieved mjData poses → body_state.json → dashboard tiles (operator)
    └─ achieved mjData poses → pose_relay → 376-byte mndrmt3 @ 50 Hz
         → Monado [remote driver] :4242 → the game

GAME PIPE — live
  weston --backend=vnc --renderer=gl     private display, no desktop window
    ├─ Steam client UI → Proton/Wine → SUPERHOT_VR.exe
    │    UnityOpenXR.dll → wineopenxr.dll → host OpenXR loader
    └─ VNC → websockify → noVNC dashboard view, operator-only
```

The relay (`clank/pose_relay.py`) speaks Monado's `remote` driver wire format: a 376-byte `r_remote_data` packet, header `mndrmt3`, at 50 Hz over TCP 4242. It reads the state file the body view already writes, so the game and the operator's tile are driven by the same `mjData` — there is no second simulator to drift.

The frame change is measured, not chosen: MuJoCo is Z-up with +X forward and the driver wants Y-up with −Z forward, so `xr_x = -mj_y`, `xr_y = mj_z`, `xr_z = -mj_x`. Positions are root-composed first, because `BodyState` poses are root-relative and the hold clip stands at about −90° about Z.

`spike/c0/c0_client.py` still exists as the scripted client: it sends constants derived from measured G1 geometry (eye height 1.221 m, wrists at (∓0.222, 0.721, 0.017) m) and remains the tool for isolating whether a fault is in the body or in the pipe.

**The command contract.** The brain is supposed to name a skill and a side; it never writes joints or poses. Everything below the adapter speaks one masked task-space command:

| Field | Meaning |
| --- | --- |
| `authority` | 9-bit mask. Unset bits are *ignored, not zeroed*, so a mode is a mask. `hold` is the empty mask: nothing requested, the executor's standing behaviour applies. |
| `seq` | Monotonic. A stale sequence number never rewinds the latest command. |
| `ttl_ms` | 200 ms default. Expiry gates **acceptance**, not execution: an expired command is dropped and the body holds. |
| `duration_ms`, `preempt` | A non-preemptible ballistic runs to completion; newer commands queue behind it instead of blending into it mid-skill. |
| `head`, `left_hand`, `right_hand` | Root-relative pose, metres and `xyzw` quaternion. |
| `closure` | 0 open → 1 closed; the game adapter maps it to trigger/grip. |
| `BodyState` | Achieved `mjData` poses, per-effector residuals, and flags (`SATURATED`, `TTL_EXPIRED`, `FALL_RISK`, `UNSUPPORTED`). Never what was asked for. |

Validation is deliberately strict and never raises into the 50 Hz loop: schema, unknown authority bits, non-finite numbers, quaternion norm, workspace box, joint and hand speed limits, TTL on arrival. A rejection means execute `hold` and increment an invalid counter that reaches the next brain observation.

---

## Status: what is proven, what is not

This is the table a technical interviewer should check first. Nothing in the "not proven" column is implied by the "proven" column.

| Capability | State | Evidence |
| --- | --- | --- |
| Game binds to a fabricated HMD | **proven** — session reached `XR_SESSION_STATE_FOCUSED`, 58 runtime extensions | `research/c0-results.md` |
| Game's view follows our poses | **proven** — 9× mean pixel change, 15× changed area, pose-only A/B | `research/c0-results.md` |
| Real game pixels are capturable | **proven** — tutorial frame with real content: 41 % bright, 3175 distinct colours | `research/c0-results.md` |
| 29-DoF body walks 60 s without falling | **proven** — headless CPU | `research/sonic-gate-results.md` |
| Body holds still | **proven when settled**; entering `hold` from a walk costs a settle transient | `research/c3m-results.md` |
| Body aims straight ahead | **proven** — 3.51 cm / 7.22° | `research/sonic-gate-results.md` |
| Body aims off-axis | **fails** — 8.63 cm / 28.33° at 30°, so the body must turn to face a target | `research/sonic-gate-results.md` |
| Sim pose → wire packet is exact | **proven** — 1001 packets / 20.02 s, max position and quaternion error 0.0 | `research/e3-spike-results.md` |
| `BodyCommand` bus (TTL, sequencing, validation) | **implemented and tested** | `clank/body.py`, `clank/bus.py`, `clank/validate.py`, `tests/` |
| Executor honours `BodyCommand` | **implemented in the spike**; runs the third-person body view | `spike/wbc/sonic_executor.py` |
| **Body drives the game** | **proven** — the game's HMD yaw follows the body's head: `look_left` −28.6°, `look_right` +54.3°, `hold` returns to baseline | `research/r6-co-residency.md` |
| Body and operator tile show the same robot | **proven by construction** — one `mjData`, one relay, no retargeting | `clank/pose_relay.py` |
| Live co-residency on the 8 GB card | **proven** — 44.2 Hz beside the game vs 44.5 Hz alone, +177 MiB, GPU util 31.5→31.7% | `research/r6-co-residency.md` |
| Dodge moves the body sideways | **a lean, not a step** — 0.106–0.134 m head shift for a 0.14 m command, base moves ≤0.008 m | `spike/wbc/sonic_executor.py` |
| Controller *orientation* convention | **unverified** — positions are exact, but the wrist frame is not an OpenXR grip, so in-game aim may be rotated by a fixed offset | — |
| **Trigger reaches the game** | **not measured** — the relay sends `trigger=0`; needs a pistol in hand | `research/c0-results.md` |
| Clock floor behaviour in play | **not measured** — the 0.005 value is read from serialized assets | `research/wbc-game-proposal.md` |
| The Fall's 0.70 m pistol and exit | **not reachable standing** — 27.3 cm wrist residual on the pistol | `spike/wbc_game/logs/reach_far_test.txt` |
| Walking under command | **not built** — the body is clip-anchored, so `AUTH_ROOT_VEL` is not honoured in a hold session | `spike/wbc/sonic_executor.py` |
| A brain / VLM in the loop | **not built** — the dashboard buttons are the only live `BodyCommand` writer | `clank/scripted.py` |
| **Level cleared** | **no** | — |

---

## Measured results

All values below are read from this repository. `research/` holds the per-spike result documents; every row names its source.

| Metric | Value | Conditions | Source |
| --- | ---: | --- | --- |
| Walk, 60 s | **39.79 m** path / 37.82 m net; 0.663 / 0.631 m/s; yaw drift −2.10°; tilt 9.48°; no fall | GEAR-SONIC default, headless CPU | `research/sonic-gate-results.md` |
| Hold, settled | **RMS 1.3×10⁻⁴ m/s** (0.02 % of walking speed) | 20 s hold, standing clip, anchor-index freeze | `research/c3m-results.md` |
| Hold entered from a walk | RMS 0.050 m/s mean (worst 0.067), 5/5 cycles, no fall, walk resumes within ~0.2 m | settle transient dominates | `research/c3m-results.md` |
| Aim, dead ahead | **3.51 cm** position / **7.22°** orientation | walking, fixed aim point | `research/sonic-gate-results.md` |
| Aim, 30° off-axis | **8.63 cm** / **28.33°** — the weak axis | same | `research/sonic-gate-results.md` |
| Dodge, as shipped | **0.106–0.134 m** head shift for a 0.14 m command, base moves **≤0.008 m** — a lean, not a step | two trials each way, live body | this session |
| Head yaw → torso yaw | **~0.83 gain**, saturating at **27–31°** | SONIC's native coupling; aim is not turn | `research/wbc-policy-proposal.md` |
| Forward reach: travel vs precision | wrist travels **0.41 m** at `forward_m=0.35` (residual **0.20 m**); **0.51 m** when raised to `up_m=+0.30` (residual **0.096 m**) | SONIC, offline, correct forward axis | `research/strike-results.md` §Addendum |
| Control cost, 1 thread | **17.66 ms** per 20 ms tick: encoder 7.54, decoder 5.42, physics 4.71 — 32 % of one core | onnxruntime pinned to 1 thread (the default) | `research/wbc-policy-proposal.md` |
| Control cost, all cores | **6.8 ms** per tick at **1876 %** CPU, 47 threads, load average 21 | onnxruntime defaults — this is the cost the pinning removes | `research/wbc-policy-proposal.md` |
| Render + encode | **0.86 ms** per 560×420 frame + 1.65 ms JPEG on the RTX 4060 | EGL headless, 10 fps tile | `research/wbc-policy-proposal.md` |
| Sim → wire fidelity | **1001 packets / 20.02 s** at 50 Hz, max position error **0.0** | E3 body bridge, live | `research/e3-spike-results.md` |
| Game session | **`FOCUSED`**, 58 runtime extensions, system `Monado: Remote HMD` | Superhot VR, Proton, RTX 4060 | `research/c0-results.md` |
| View follows pose | static **4.74** mean Δ / **4.0 %** pixels vs sweeping **41.97** / **59.3 %** | same session, same scene, 3.4 s apart | `research/c0-results.md` |
| Capture and latency | **36.9 fps**; pose → pixel **30–59 ms** (mid 44 ms, incl. ~27 ms capture overhead) | `grim` on the laptop panel, `hello_xr` | `research/e2-spike-results.md` |
| **Game follows the body** | `look_left` **−28.6°**, `look_right` **+54.3°**, `hold` back to baseline **±0.8°**; head height steady 1.189 m | live session, mode `body`, 50 Hz relay | `research/r6-co-residency.md` |
| Co-residency (R6) | body **44.2 Hz** with the game vs **44.5 Hz** alone; policy-only 45.8 Hz in both | two 45 s windows, RTX 4060 8 GB | `research/r6-co-residency.md` |
| Co-residency cost | **+177 MiB** GPU (1144→1321), util 31.5→31.7%, load1 1.93→2.20, game 0.42→0.51 cores | same | `research/r6-co-residency.md` |
| Policy rate | **49.6–49.9 Hz idle** (0.99–1.00× real time) at `--fps` 10/15/30/50; **44.2–44.5 Hz with the game** (0.89×) | 10 substeps × 2 ms per tick | this session |
| Second camera angle | **~0.6 ms** per tick; body viewer RSS **948 MB** with SONIC loaded | EGL on the 4060, 640×480 | `research/gl-render-cost.md` |
| Pose feed decoupled from rendering | `body_state.json` **45.8/s** while frames render at **4.5/s** | `--fps 5` | this session |
| Wrist residual at the clip's own target | **0.1557 m** left / **0.1208 m** right (settled hold: 0.0609 / 0.0324) | median, 1000 frames, achieved `lw`/`rw` minus commanded `vr_t[0:3]`/`vr_t[3:6]` | `spike/wbc/logs/c3m_H3_squat_f0_from_squat.jsonl` |
| Repo test suite | **323 tests**, standard library only | no network, no GPU, no game | `python3 -m unittest discover -s tests -t .` |

---

## Key engineering decisions, and the negative results that forced them

**1. Use the released whole-body controller, not a from-scratch one.**
The from-scratch classical QP is *excellent* and cannot walk: 0.22 mm of base excursion over a 20 s hold, ±35° look to 0.63°, single-arm reach residuals of 2.2–12.9 mm. But every single-support foot lift tipped the robot within 1.3–1.6 s, and 0.10 m of lateral weight shift is the measured limit (0.12 m tips). That is the walking bottleneck appearing *before* the first step. So the QP is a precision executor for standing skills, and walking came from the released policy instead — which is why this repository contains two harnesses and recommends only one.

**2. Do not override the policy's arms with inverse kinematics.**
Hypothesis: SONIC's arms are already driven by VR targets in its training distribution, so an IK override toward the same target should be tolerated. **Falsified.** A hard override makes the arms roughly 4× worse (3.51 → 12.84 cm while walking; 1.14 → 14.40 cm while holding), rate-limiting does not help, and integral correction tips the robot at 65–75° tilt. The only surviving refinement is wrist-orientation-only: 7.22° → 5.08°. The controller is a motion tracker; it produces its own arms, and a joint-space override fights it.

**3. `hold` is a reference-motion switch — not "send zero", and not a freeze of whatever clip is loaded.**
This one caught our own harness. Freezing only the *anchor index* of a walking clip left the encoder's 0.9 s lookahead still walking, so the "hold" travelled 3.93 m in 20 s at 0.21 m/s without falling — exactly what a walking policy told to keep walking should do. Freezing the whole horizon is 36× better (0.2005 → 0.0085 m/s), and switching the reference to a standing clip is better again (1.3×10⁻⁴ m/s). The lesson is not "SONIC cannot hold"; it is that a hold is only meaningful when you state what the lookahead was doing.

**4. Inject at the OpenXR runtime, not with a custom OpenVR driver.**
Superhot ships `UnityOpenXR.dll` and binds Monado's `remote` driver directly, so neither a custom driver nor SteamVR nor a translation layer was needed. The price was a three-fault hunt, all in `research/c0-results.md`: `/etc/xdg` is the *container's* inside pressure-vessel so the active-runtime pointer must live in `~/.config/openxr/1/`; `~/.local/share/openxr/1/` is not a searched path at all; and the loader enumerates manifests once per process, so fixing the pointer requires a game restart. The launch must also come from the Steam client UI — CLI `-applaunch` resolves a non-Steam shortcut to a store AppID and fails `AppError_9`, `rungameid` reports `Unknown GameID type`, and a hand-run Proton entry point dies in `bwrap`.

**5. Keep the commercial game.**
Sanctioned pose APIs (VRChat's OSC input, Resonite) are explicit non-goals in the design, because they hand you the interface and delete the systems problem. Equally out of bounds: reading game memory, injecting into the title, or driving it through an unofficial API. The game is unmodified commercial software you must own, and the interface is the deliverable.

**6. Hide the game in a compositor we control.**
`gamescope --backend headless` with PipeWire capture produced black frames; a Hyprland headless output produced black frames in `grim`. A nested `weston --backend vnc --renderer gl --xwayland` session does work: Steam and the game run inside it, never touch the host desktop, and the dashboard views it through websockify/noVNC. `weston-screenshooter` remains only for the headless `--preview-smoke` check.

**7. Pin ONNX Runtime to one thread.**
With default settings onnxruntime opened one intra-op thread per core and the body view held about 19 of 20 logical CPUs: 1876 % CPU, 47 threads, load average 21. The networks are 1762→64 and 994→29; a thread pool costs more in spin-wait than it saves in FLOPs. One intra-op and one inter-op thread, sequential execution: 21 threads, 32 % of one core. That costs latency — 6.8 ms/tick becomes 17.66 ms — and buys back 19 cores, which is the right trade for a body that has to share the machine with the game. It also leaves only 2.3 ms of slack in the control tick, which is worth knowing before anything else is added there.

**8. Join by relaying the file the viewer already writes, not by adding a second client.**
The obvious shortcut was a standalone pose client that runs its own `SonicExecutor`, like `c0_client.py` with body poses. That would have been two simulations: the tile would show one body while the game saw a slightly different one, and every screenshot would be a quiet lie. Instead `pose_relay.py` reads the `body_state.json` the viewer already publishes, which means one `mjData` drives both consumers and the relay stays standard-library (no MuJoCo, no NumPy), so it runs under the dashboard's interpreter and the test suite covers it. The relay also reports staleness explicitly — it holds the last pose and marks the row `stale` rather than freezing silently, and sends the measured standing pose before any state arrives so the game sees a humanoid, not the origin.

**9. Write the join's pose feed every control tick, not every rendered frame.**
`body_state.json` is what the relay reads, so gating it on the render cadence made the game's pose rate a side effect of the tile's frame rate: at `--fps 10` the body still stepped at 45 Hz but the game received 100 ms pose steps. Someone dropping the frame rate for CPU headroom would have degraded the join without touching it. Measured after the fix: state updates 45.8/s while frames render at 4.5/s.

**10. Trust the measured reach, and read travel and precision as separate numbers.**
The wrist travels **0.41 m** forward at `forward_m=0.35`, and **0.51 m** when the target is raised to `up_m=+0.30`. What it does *not* do is arrive precisely: the residual at those two points is **0.20 m** and **0.096 m**, and it does not improve as the command gets smaller. Every earlier “~0.20 m envelope” figure came from sweeping `forward_m` while the harness moved the wrist **across the body** — the axis the policy is worst on — and is retired (`research/strike-results.md` §Addendum). The consequence is still a design rule, not a rounding error: the body *turns to face* a target rather than aiming sideways, because at 30° off-axis the arm error grows from 3.51 cm to 8.63 cm and from 7.22° to 28.33°. Under Job Simulator nothing sits at 0.70 m, so whether a step is needed is unmeasured.

---

## How to run

**Prerequisites.**
- **NixOS** with Steam. Tested on a Hyprland/Wayland host; nothing else is claimed.
- **Your own copy of Superhot VR** (not included, never will be), added to Steam as a non-Steam game and running under Proton.
- **Monado** with the `remote` driver built in.
- **weston** (VNC backend), **weston-debug** for Steam window detection, and **weston-screenshooter** only for the headless `--preview-smoke` check — supplied by `flake.nix`; `run.sh` re-enters `nix develop` when they are not on `PATH`.
- **novnc** and **websockify** — the browser client assets and the WebSocket-to-VNC proxy. Supplied by the devshell.
- **The NixOS PAM service `weston-remote-access`** — authenticates the VNC viewer. This is host configuration, not something the repo installs.
- **ImageMagick** (`magick`) and **mpv** — only for `--preview-smoke`; also in the devshell.
- **Python 3.12+**, standard library only. No headset, no controllers.

## Setup

```bash
git clone <this repo> && cd Clank
cp config.example.toml config.toml        # edit paths, Steam shortcut id, Proton app id
./run.sh                                  # serves http://127.0.0.1:8765
# run.sh reuses a dashboard already on that port; run.sh --replace stops it first.
```

In `config.toml`, the values you almost certainly must change:

| Key | What to put there |
| --- | --- |
| `paths.steam_bin` | your Steam launcher |
| `paths.game_exe` | path to `SUPERHOT_VR.exe` |
| `paths.openxr_active_runtime` | your Monado runtime pointer (below) |
| `paths.session_root` | where session logs go |
| `launch.shortcut_id`, `launch.proton_app_id` | your Steam shortcut's AppId |

**The OpenXR runtime pointer is required, and clank only reads it.** Under Wine the game resolves its runtime through the Proton prefix registry → `wineopenxr.dll` → the host OpenXR loader, which looks for `~/.config/openxr/1/active_runtime.json`. Note that `/etc/xdg` is invisible inside the Proton container and `~/.local/share/openxr/1/` is *not* a searched path. Create it from your Monado install:

```bash
mkdir -p ~/.config/openxr/1
cp /etc/xdg/openxr/1/active_runtime.json ~/.config/openxr/1/active_runtime.json
```

If a game is already running, restart it — the loader caches its runtime list per process.

**Development.**

```bash
nix develop                                   # python3 + weston + novnc + websockify + mpv + magick, pinned by flake.lock
python3 -m unittest discover -s tests -t .    # 323 tests, no network, no GPU
nix flake check                               # same suite, sandboxed
./run.sh --preview-smoke                      # headless Weston screenshot-tool check, without launching Steam
```

**The controller, headless.** The simulator spikes use their own venv because they need MuJoCo, NumPy and ONNX Runtime while `clank/` stays standard-library only:

```bash
cd spike/e3
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python 'mujoco==3.17.0' 'numpy<2.3' mink imageio
uv pip install --python .venv/bin/python onnxruntime
cd ../..

# the three SONIC gates (write the summary JSONs the research docs cite)
# hold_c3m.py has one known crash in an extra 5-cycle variant; the two configs
# that completed are the cited evidence, in spike/wbc/logs/c3m_summary.json
spike/e3/.venv/bin/python spike/wbc/gate_c3i.py
spike/e3/.venv/bin/python spike/wbc/arm_override_c3j.py
spike/e3/.venv/bin/python spike/wbc/hold_c3m.py

# one gate re-run by hand
spike/e3/.venv/bin/python spike/wbc/sonic_runner.py \
  --motion squat_001__A359 --mode 1 --seconds 20 --freeze 0 --log /tmp/hold.jsonl --verbose

# third-person MuJoCo view, driving BodyCommand through the SONIC executor
spike/e3/.venv/bin/python spike/wbc/body_view.py --out /tmp/body --reel --seconds 20

# the join, by hand: body view publishing state, relay feeding Monado
spike/e3/.venv/bin/python spike/wbc/body_view.py --out /tmp/body \
  --fps 50 --views three_quarter,side --control /tmp/body/control.sock &
python3 -m clank.pose_relay --state /tmp/body/body_state.json \
  --log /tmp/pose.csv --duration 60
```

Offscreen rendering needs no display: `spike/wbc/gl_boot.py` re-execs with the NixOS libglvnd path, because NixOS ships `libEGL_nvidia.so` but no plain `libEGL.so`, and `LD_LIBRARY_PATH` is only read at process start.

Weights are `nvidia/GEAR-SONIC` (encoder 1762→64, decoder 994→29) under the **NVIDIA Open Model License**, not Apache — code written around them here is MIT. The controller code wraps them unmodified.

---

## Controls

The dashboard is a single local page with a state badge, session id, run phase, uptime, log tails and the preview. A wait shows a spinner and what it is for; a state that needs you does not.

| Control | Effect |
| --- | --- |
| **Start** | Boots the stack: Monado → Weston (VNC) → websockify → pose client → Steam → game. `409` if a session is already running. Once per stack, not once per attempt. |
| **Start run** | One repeatable attempt, leaving the stack up: a fresh `sessions/<id>/runs/<run>/` (diary, log, frames), the overlay cleared, the game's saved job wiped and the game relaunched, the body reset, then the brain loop. Refused before anything is spent if the model server is not up serving the expected model. |
| **Stop run** | Ends the attempt (the brain loop and its process group) and keeps the stack, the game and the body running. |
| **Shut down** | Tears the session down; the run is stopped first, then idempotent cleanup — surviving processes are logged and surfaced in `/api/state`. |
| **mode** | Switches the pose client: `hold`, `yaw`, `trigger`, `dodge`, `walk`, `clock`, `manual`, `body`. `body` is the join — the relay sends the simulated humanoid's achieved poses instead of scripted constants. |
| **kill** | Force-stops the game or Steam specifically. These are force kills and they hit *your* Steam. |
| **preview** | VNC stream with PAM login; view-only. The stream is 1280x720 at a target 30 fps; the dashboard crops the top 32 px status panel, so the display area is 1280x688. Manual mode still uses pointer lock. |
| **body** | Optional third-person MuJoCo tile (`/api/body.jpg`), run as a sidecar that cannot delay or break a session. It needs the spike venv and weights, and is a *viewer* — it is not the pose source. |
| **Recordings** | `GET /recordings` lists past runs and plays one: the composite video (game view left, robot view right, the game's audio) beside that run's diary. Clicking a diary row seeks the video to that moment. One file per run, written while the run happens and composed after it; `[recording]` in `config.toml` turns it off. |
| **mute** | `POST /api/audio/mute` toggles the game’s PipeWire output mute from the dashboard header button. |

In `manual` mode, click the preview to pointer-lock the browser and drive the hidden game directly: mouse look, WASD move, arrows move the active hand, wheel reaches, F throws a one-shot punch with the active hand, LMB/T fires, RMB/G grips, Tab switches hands. Esc releases the pointer immediately; the last pose keeps streaming. Browser batches go to `POST /api/manual/input` and are forwarded to the manual pose client over a per-session Unix datagram socket.

## Safety notes

- **This launches Steam and Superhot on your machine**, in a nested Weston session with a NULL-compositor Monado, so it does not draw on your desktop. Your mouse and keyboard focus stay yours until you click the preview in `manual` mode to capture the pointer; Esc returns them.
- **The preview stream is loopback-only.** Weston's VNC backend listens on `127.0.0.1:5901` and websockify on `127.0.0.1:6080`; noVNC assets are served by the dashboard, and credentials are not stored.
- **The kill buttons are force kills.** They terminate *your* Steam and the game process. Anything you had unsaved in other Steam games does not survive a `kill steam`.
- **One session at a time** by design; concurrent starts are guarded and rejected.
- **It does not modify your OpenXR configuration.** It validates that the runtime pointer exists and is readable, and fails preflight with a message if it does not.
- **No game files are included.** Superhot VR is commercial software you must own.
- Cleanup is best-effort and verified rather than assumed: after a stop, any surviving process is logged and surfaced in `/api/state`.

### VNC and PAM

The VNC viewer authenticates through this NixOS PAM service:

```nix
security.pam.services.weston-remote-access = {
  text = ''
    auth sufficient pam_unix.so likeauth try_first_pass
    auth required pam_deny.so
    account required pam_unix.so
  '';
};
```

VNC TLS is disabled: the stream has no confidentiality or server authentication, so keep it on loopback. The dashboard control API has no authentication either; keep `dashboard.host` on loopback and do not expose it.

## Repo map

| Path | What it is |
| --- | --- |
| `clank/` | **The app.** Config validation, process-group supervision, session orchestrator, VNC preview transport, dashboard HTTP API and UI, the `BodyCommand`/`BodyState` bus and its validator, manual input bridge. Standard library only. |
| `tests/` | Unit tests: config, process groups, orchestrator concurrency guards, dashboard API, VNC preview lifecycle, command bus, validation. |
| `spike/c0/` | **Pose client and launchers.** The `remote`-driver client (`hold`, `yaw`, `trigger`, `dodge`, `walk`, `clock`), the manual client, the runtime launcher. |
| `spike/e1/`–`spike/e3/` | **Injection and evidence.** OpenXR pose injection, compositor capture and latency measurement, and the MuJoCo body → OpenXR bridge with its honesty CSV. |
| `spike/c3/`, `spike/wbc/`, `spike/wbc_classical/`, `spike/wbc_game/` | **Controller harnesses and logs.** SONIC gates and arm-override matrix, hold/look investigation, classical QP, the 0.70 m reach probe. |
| `research/` | **Findings.** Per-spike results with measured numbers, plus controller and game-side surveys — the source for nearly every figure in this README. |
| `docs/` | **Design.** The design document, its builder-check, the state of findings, and the dashboard spec (historical for the PNG preview; the current preview is VNC). |
| `RESEARCH.md` | The original feasibility study (partly superseded; see its banner). |
| `flake.nix`, `run.sh` | Pinned devShell and the one-command entry point. |

---

## Scope and limitations

Deliberately narrow: **NixOS, Superhot VR, a local Steam install, a local Monado runtime, one session at a time, no headset, no controllers.** This is not a general VR-game bridge, not a headset emulator for arbitrary titles, and not a robot-learning research platform. GPU-side VLM inference next to Superhot, Monado and MuJoCo on the same 8 GB card is out of scope.

The sharpest limitations, stated as facts rather than caveats:

- **Nobody has played it.** The body drives the view and it is visibly alive, but the trigger has never reached the game and no level has been cleared. Looking is not playing.
- **The body cannot walk under command.** It is clip-anchored: `AUTH_ROOT_VEL` is not honoured in a hold session, so it leans and reaches on the spot. "Dodge" is a 12 cm lean, measured, not a sidestep.
- **It runs 11% slow beside the game.** 44.2–44.5 Hz against a 50 Hz design point, and each tick advances exactly 20 ms of simulated time, so the game sees 0.89× real time. Idle it hits 49.6–49.9 Hz. Superhot scales its own clock with player motion, which absorbs most of this, but it is a real number and it varies with load.
- **Off-axis aiming fails.** At 30° the body points 28.3° off, so it must turn rather than aim sideways.
- **Reach travels 0.41–0.51 m but lands 0.10–0.20 m off.** Travel rises monotonically with the command; precision does not improve when the command is shorter. Job Simulator's object distances are unmeasured.
- **Controller orientation is unverified.** Positions are exact, but the robot's wrist frame is not an OpenXR controller grip, so the in-game aim ray may be rotated by a fixed offset. Untested.
- **`hold` has a settle transient.** The 1.3×10⁻⁴ m/s figure is the settled hold; entering it from a walk costs about 0.05 m/s RMS for the first seconds.
- **A static ~0.156 m offset between the model and the clip reference is unexplained.** Commanding the clip's exact f0 wrist target still leaves a **0.1557 m** left-wrist residual (0.1208 m right, median over 1000 frames of `spike/wbc/logs/c3m_H3_squat_f0_from_squat.jsonl`). In a settled hold the same comparison reads 0.0609 / 0.0324 m, so the offset is specific to the clip-anchored case and inflates every residual measured there.

The design's own anti-goals are worth repeating because they are the reason the numbers can be trusted: no desired IK targets or VLM-emitted poses in the pose path, no reading game memory, no claiming a punch hit when the residual is 13.5 cm, and no framing this as "robot plays Superhot" instead of whole-body control on an OpenXR task.

---

## License

MIT — see [`LICENSE`](LICENSE). Superhot VR is proprietary software owned by its developers and is not distributed here; the GEAR-SONIC weights are licensed by NVIDIA under the NVIDIA Open Model License.
