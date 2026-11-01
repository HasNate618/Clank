# Targeting Adapter — Design

- Date: 2026-10-30
- Status: draft for review; no implementation until this document is accepted
- Scope: close the loop so the simulated G1 can put a hand on a Superhot object and grab it. The hub floppy sequence is the first proof. The brain is a drop-in writer on the same intent socket, not part of this build.
- Parent: `docs/2026-10-23-clank-design.md` §§5–9, §11. Play rules: `research/superhot-play.md`.
- Target machine: the existing NixOS/Hyprland host, live session path (Monado remote + Weston VNC + Superhot). Do not take over the host desktop.

Claim tags in this document: `[MEASURED]` local file or named run; `[OPERATOR]` told by the person who owns the game; `[ASSUMPTION]` inference, untested.

## 1. Problem

The join works. Manual mode is fully playable `[OPERATOR]`: grip, throw, aim, shoot all go through the 376-byte `mndrmt3` packet. In `body` mode the simulated humanoid's achieved torso and wrists drive the HMD and controllers `[MEASURED: look_left −28.6°, look_right +54.3°, height 1.189 m]`. Grip latches; trigger is a 3-tick press `[MEASURED: spike/wbc/sonic_executor.py, commit 4fbfa56]`.

Nothing can aim at an object. Every dashboard verb is a fixed open-loop pose (`look_left` is ±35°, `aim` is 0.22 m forward of the clip wrist, saturating at ~0.20 m `[MEASURED]`). There is no way to say "the disk is there." A brain that named `grab right` would hit that same constant.

The missing piece is an **adapter that consumes a 2D target and produces `BodyCommand`s**, using two sensors we already own or can read:

1. **Projection.** We emit the HMD pose and we know the wrist in `mjData`. The pixel error between the named target and the projected wrist is a 2D signal. Drive it to zero.
2. **In-range signals the game already emits**, and they are not all the same thing
   `[OPERATOR 2026-10-30]`: the hub floppy and the headset have **no** pre-grab glow
   (proximity then one press; the confirmation is the object appearing in the hand),
   while the **insertion slot glows green** - large and saturated - and only the
   other, in-level grabbables have the faint proximity highlight. None of it is a
   model output.

d1 is not this adapter. On GPU it is a hold-time chooser (digest 62–176 ms `[MEASURED 2026-10-30, llama-server :11434, d1-Omni-2B-Q8_0]`). `box` is rejected on that build. Grid-choice grounding was near chance on one unverified frame. Even a perfect box is useless until something can servo to a pixel.

## 2. Proven basis

- Packet path and buttons work in manual mode `[OPERATOR]`. Default relay grab is squeeze (`--grab-mode squeeze`).
- One sim, one consumer: `body_state.json` at ~46 Hz, frames independently `[MEASURED: commit 0753032]`.
- `BodyCommand` / `BodyState` schema `clank.body/1` already carries per-hand `closure` and `trigger`, `AUTH_CLOSURE` / `AUTH_TRIGGER`, and `FLAG_SATURATED`.
- Grip latches across `hold()` and other verbs; `release_*` opens it. Trigger does not latch.
- Reach envelope ~0.20 m. Aim at 0.22 m reports `SATURATED`. Wrist residual vs clip VR points: 0.1557 m left / 0.1208 m right median `[MEASURED: research/c3m-results.md]`.
- G1 has no neck. Head pose is derived. "Over the head" is unmeasured and may sit outside the envelope.
- Superhot is authored around spawn. Walking more than about 1 m breaks the next teleport `[OPERATOR]`.
- Game framebuffer today is Weston VNC on `127.0.0.1:5901`, viewed through noVNC. `weston-screenshooter` **blocks** in VNC mode `[MEASURED: research/r6-co-residency.md]`. `body_latest.jpg` is the MuJoCo tile, not the game.
- `wire.fill` sets `per_view_data_valid = False` and does not write FOV `[MEASURED: clank/wire.py]`. Projection cannot assume we emit FOV until we start, or until the remote driver's default is measured.

## 3. Goals and non-goals

### Goals

1. An **intent** writer (script first, operator pixel second, d1 later) names `skill + side + optional pixel`. No 3D point. No joints. No OpenXR poses.
2. An **adapter** compiles that intent into a stream of `BodyCommand` on the existing control socket. The body process does not learn Superhot verbs.
3. **Parametric** `look` and `reach` — a yaw or a wrist offset, not a dashboard constant.
4. **Wrist → HMD projection**, unit-testable with the game off, using poses from `BodyState` and a **fixture FOV**. Live FOV is a measured input, not a guess.
5. **Glow** as a patch brightness delta against a hand-away baseline. Boolean `yes|no|unknown`. The adapter **reports** it and does not grab on its own.
6. A **hub script** on that adapter: look → reach until glow → grab → carry → release. Floppy lifts, or the residual is the blocker, or "over the head" saturates. All three are deliverables.
7. Honest outcomes: `done`, `saturated`, `no_glow`, `timeout`, `no_frame`. The next writer (script or brain) sees them.

### Non-goals (this spec)

- Wiring d1, E5, or any model into the loop.
- Object detection / `box` / grounding. A pixel arrives already named.
- Walking the level. A later bounded 0.3–0.7 m step, inside ~1 m of spawn, is out of this spec.
- Headset-over-head as a known pose. Measure; if `SATURATED`, stop and record.
- Fine-tuning, policy retraining, game telemetry as control, ROS2.
- Changing `clank/` away from stdlib-only. Image decode and numeric projection that need NumPy/PIL live next to the executor in `spike/wbc/`.
- Auto-grab when glow becomes true. Grab is its own intent.
- Consecutive-frame differencing as the glow oracle.

## 4. Architecture

Three loops, one body socket. Same rates as the parent design; the new piece is the adapter actually existing.

```text
Intent writer (script | operator click | later d1)
    skill + side + optional pixel[u,v]
    latest-wins, only accepted in hold / after a ballistic ends
        |
        v
Adapter  10–30 Hz     spike/wbc (venv: numpy/PIL as needed)
    look sweep, project wrist, read glow, emit BodyCommand
    never blocks the 50 Hz body
        |
        v
Body     50 Hz        existing SonicExecutor + body_view
    BodyCommand → mj_step → body_state.json
    grip latches; trigger is FIRE_HOLD_TICKS
        |
        v
Relay    ~50 Hz       clank/pose_relay.py (stdlib)
    achieved poses → mndrmt3 :4242
```

| Loop | Rate | Owns | Must not own |
| --- | --- | --- | --- |
| Body + packets | 50 Hz | `mj_step`, achieved poses, latched grip | Superhot verbs, pixels, glow |
| Adapter | 10–30 Hz | intent → `BodyCommand`, projection, glow sample | joint angles, which object |
| Writer | script: evented; d1: only in hold | skill + side + pixel | motion, "is it grabbable", 3D |

Invalid or stale intents become `hold`. The 50 Hz loop never waits on a frame or a model.

### 4.1 Process split

| Unit | Lives in | Why |
| --- | --- | --- |
| `clank.intent` schema + parse/validate | `clank/` (stdlib) | same bar as `body.py` |
| Adapter process (skill loop) | `spike/wbc/` | needs the body venv; may decode JPEG / do pinhole math |
| Glow sampler | same adapter process | needs game pixels |
| Frame source | adapter, via a VNC/RFB client or `DISPLAY=:1` grab | `weston-screenshooter` is dead in VNC; dashboard canvas is the operator, not a sensor |
| Hub script | stdlib writer against the intent socket | first brain stand-in |
| Dashboard verbs | keep as shortcuts that emit intents or `BodyCommand`; do not grow a second control path | |

Deleting the adapter must leave manual mode, the join, and open-loop verbs intact.

## 5. Intent contract

New schema name: `clank.intent/1`. JSON, latest-wins, same transport style as `BodyCommand` (Unix socket or queue). This is **not** a field on `BodyCommand`. Pixels are not robotics state.

```text
{
  "schema": "clank.intent/1",
  "seq": 0,
  "t_mono_ns": 0,
  "ttl_ms": 2000,
  "source": "script" | "human" | "vlm",
  "skill": "hold" | "look" | "reach" | "grab" | "release" | "fire" | "strike" | "dodge" | "twitch",
  "side": "none" | "left" | "right",
  "pixel": null | [u, v],
  "yaw_rad": null | <float>
}
```

- `pixel` is **normalized in the HMD view**, origin top-left, `u` right, `v` down, both in `[0, 1]`. It is not a VNC pixel and not a 3D point. Mapping into the capture crop is the adapter's job.
- `yaw_rad` is only for `look` without a pixel (scripted sweep). Sign: same as `look_target` today — positive yaw is the robot's left `[MEASURED: spike/wbc/body_view.py]`.
- `side` is `none` for `hold`, `look`, `twitch`. Required for `reach` / `grab` / `release` / `fire` / `strike` / `dodge`.
- Unknown skill, missing side, pixel out of range, wrong schema → reject, adapter holds, increment `invalid_count` into the next digest.

The brain the parent design described is this writer with `pixel` optional. Adding a pixel does not violate "no 3D point." 2D→3D stays out: the adapter never reconstructs floppy XYZ.

## 6. Adapter skills

Each skill is a finite behaviour with a recorded outcome. The adapter is the only place Superhot verbs exist (parent §8).

| Skill | BodyCommand | Runs until | Outcome |
| --- | --- | --- | --- |
| `hold` | `FREEZE_HOLD_AUTH` at the last achieved head and both wrists — head + both hands claimed, grip untouched | replaced | `done` |
| `rest` | empty authority — the executor's standing clip decides the VR points | replaced | `done` |
| `look` | `AUTH_HEAD` (+ waist if we start using it) toward `yaw_rad`, or toward centering `pixel` | yaw error small, or `SATURATED` at the ±35° v1 arc | `done` / `saturated` |
| `reach` | `AUTH_{L,R}_HAND` toward a wrist target the servo updates | see §7 | `done` / `saturated` / `no_glow` / `timeout` / `no_frame` |
| `grab` / `release` | `AUTH_CLOSURE` 1.0 or 0.0 on that side | one command; grip latches | `done` |
| `fire` | `AUTH_TRIGGER` 1.0 on that side | executor's `FIRE_HOLD_TICKS` | `done` |
| `strike` | existing punch target, ballistic, `preempt=False` | duration | `done` / `saturated` |
| `dodge` | existing lean | duration | `done` |
| `twitch` | small head or wrist motion, then `hold` | one increment | `done` |

`reach` **does not** close the hand. Glow true means "in range now"; the writer sends `grab`. If the adapter grabbed on glow, a false highlight would steal every disk.

**`hold` is freeze-hold, and `rest` is the clip return.** The split exists because the executor's default for an unmentioned channel used to be the standing clip's own VR points, so an empty mask snapped a reached hand back to rest — on every TTL expiry, and on every command that named only one channel. A `look` issued while holding the card dropped it. Now `channel_pose` in `clank/body.py` decides each channel: empty authority is `rest` (clip); a commanded pose wins; otherwise the channel holds what the body achieved. `rest` is not a planner output — a brain that says `hold` means freeze, never reset. `FREEZE_HOLD_AUTH` excludes `AUTH_CLOSURE` (grip already latches, and freezing a hand must not reopen it), `AUTH_ROOT_VEL` and `AUTH_WAIST` (neither can be honoured in a hold session).

Open-loop verbs stay for the dashboard. They are the same skills with no pixel and a constant offset (`aim` = `reach` at 0.22 m, no glow gate).

## 7. Projection (image-space servo)

### 7.1 Inputs

- Wrist and head from `BodyState`, composed through the root exactly as `wire.body_state_to_xr` already does (root_q is `(w,x,y,z)`).
- FOV: the four OpenXR angles the compositor is actually using. **Until measured or written into the packet, tests use a fixture.** Do not invent the remote driver's default in production code.
- Target `pixel` `(u*, v*)` in the HMD view.

### 7.2 Output

Project the wrist through that camera to `(u, v)` in the same normalized view.

Pinhole with asymmetric FOV, OpenXR camera (−Z forward, Y up), then convert to top-left normalized:

- Camera-frame point: `p_cam = R_head^T (p_wrist − p_head)` in the XR frame.
- If `p_cam.z >= 0` the wrist is behind the view → no projection, outcome stays `no_frame` for glow and the servo must not wind up.
- `u = (atan2(p_cam.x, −p_cam.z) − angle_left) / (angle_right − angle_left)`
- `v` analogously from `angle_up` / `angle_down`, flipped to top-left.

Unit tests (no game, no MuJoCo): a wrist on the head's −Z axis projects to ~center; a wrist toward `angle_right` projects to `u ≈ 1`; behind-camera is rejected. Exact numbers come from the fixture FOV in the test.

### 7.3 Servo

Error `e = (u*, v*) − (u, v)`. While `|e|` is above a pixel tolerance (spec default: **0.02** of the view, ~2% of the image), step the wrist in the plane of the HMD, root-relative, clamped to the measured ~0.20 m envelope. Rate-limit with `max_hand_speed` (contract default 1.0 m/s).

A pixel error of zero means **on the ray**, not in grab range. Depth is §8.

If `FLAG_SATURATED` is set and error is still large, stop. Do not grow the target past the envelope. The 0.156 m / 0.121 m residual is in the budget; if the servo saturates short of the pixel, that is the deliverable, not a retry loop.

## 8. In-range signals

Superhot gives three different "you may act now" signals, and conflating them
produces a script that waits forever or grabs at nothing. `[OPERATOR 2026-10-30]`

| Signal | Applies to | Looks like | Confirmation |
| --- | --- | --- | --- |
| **Proximity, no glow** | hub floppy, headset | nothing changes | the object appears in the hand |
| **Green slot** | disk insertion point | large saturated green region | the game accepts the release |
| **Subtle highlight** | other in-level grabbables | faint local brightening | the object lifts on grab |

The first is not detectable at all, and does not need to be: presses are cheap and
the pickup itself is the read-back. The second is the easiest problem in the whole
game - a hue gate, not perception. The third is what §8.4's tail statistic is for,
and it is **not** used by the hub.

### 8.6 Job Simulator (the current writer's game)

The table above is Superhot. Job Simulator's cues are different
`[OPERATOR 2026-11-01]` and are specified in
`docs/superpowers/specs/2026-10-31-brain.md` §6.2:

| Signal | Looks like | Confirmation |
| --- | --- | --- |
| **Blue proximity** | a grabbable/interactable near the hand lights blue | `grab` is now in range |
| **Blue slot** | the cartridge slot itself is painted blue | destination for a held cartridge (GROUND / hue-gate centroid) |
| **Yellow ghost** | a yellow ghost of the cartridge appears at the slot | the game will accept `release` (insert) |

`clank/glow.py`'s green hue gate (`is_green` / `green_centroid`) is not this
game. Do not implement a Job Simulator hub path against §8.1–§8.2. Conflating
the slot's persistent blue with the proximity blue produces a grab at the slot
from across the room; waiting for blue instead of yellow produces a never-release.
Live fields are `glow` (wrist proximity) and `ghost` (insert cue), both reported,
neither an auto-grab nor an auto-release.

### 8.1 Hub pickup: press, then look at the hand

`reach` toward the disk until projection error is inside tolerance, then `grab`.
There is no glow to wait for. Termination is the **appearance change at the
projected wrist**: the flat rectangle that replaces the hand. Score it as a
fraction of changed pixels in a patch there (§8.4), not as a maximum - an object
arriving covers a real area, and a single bright pixel is a specular edge.

If nothing appears and the arm reports `SATURATED`, the disk is out of reach. That
is the measured-residual finding, not a retry.

### 8.2 Insertion slot: a hue gate

Detect the green region directly: a pixel counts when its green channel exceeds
both others by a margin and clears a brightness floor; a patch counts when such
pixels cover a minimum fraction; the target is their centroid, normalised to the
HMD view. `reach` servos to that centroid and `release` inserts.

A centroid is a 2D target, so this stays inside the design's no-3D rule: the game
paints where the slot appears, and we point at the paint.

### 8.3 Headset: same as the disk

Proximity and a press, then carry over the head. No glow, so it shares §8.1's
termination. The reach itself is unmeasured (the G1 has no neck), and `SATURATED`
is the recorded outcome if "above the head" is outside the envelope.

### 8.4 Subtle highlight (in-level only)

Kept for the non-hub case, and only that:

1. Baseline frame `B` with the hand retracted; current frame `F`.
2. Patch around the named pixel (spec default **5%** of the crop's min side, at
   least 16 px).
3. `D = clip(F - B, 0)` per channel. Score = **95th percentile of `max(R,G,B)`**
   over **every** pixel in the patch, not only the changed ones - collecting only
   changed pixels collapses the statistic to "brightest changed pixel", which a
   single stray pixel satisfies.
4. `glow = yes` iff score passes a threshold measured from a real on/off pair.

Forbidden as an oracle: `F[t] - F[t-1]` (the approaching hand is a huge diff);
full-frame means (timing flashes, HUD, controller mesh); any model answering "is
it grabbable".

### 8.5 How `reach` terminates

| Condition | Outcome |
| --- | --- |
| projection inside tolerance and pickup confirmed | `done` |
| projection inside tolerance, nothing appeared, arm saturated | `saturated` |
| frames unavailable or older than 0.3 s | `no_frame` |
| no termination before the deadline | `timeout` |

`reach` never closes the hand. `grab` and `release` are separate intents, so a
false appearance signal cannot steal a disk.

## 9. Frame capture

The glow and the operator preview are the same compositor view, not the MuJoCo tile.

| Path | Status |
| --- | --- |
| noVNC in the dashboard | operator only; not a sensor |
| `weston-screenshooter` | **dead** in VNC mode |
| `body_latest.jpg` | wrong scene |
| Second RFB client on `127.0.0.1:5901` | **primary candidate** |
| `DISPLAY=:1` grab (Xwayland) | fallback if RFB is hostile |

VNC is 1920×1048 of Weston `[MEASURED: /api/state]`, not a raw HMD eye. The adapter must map HMD-normalized `(u,v)` into that crop **once**, from a calibration: known look, known object, recorded correspondence. Parent §11 already required pose→pixel when the capture path changes. Do that measurement before claiming the servo can see.

Frame age > 0.3 s (same order as `STALE_AFTER_SEC` on the relay) → treat as `no_frame`, do not servo on a stale picture.

`clank/` does not grow a JPEG dependency. The adapter process may.

## 10. Hub sequence

This is the first writer. It is the sequence in `research/superhot-play.md` §1, as intents, not a planner.

Assume the session is already at the hub. The script does not launch Superhot.

| Step | Intent | Adapter stops when | Writer next |
| --- | --- | --- | --- |
| 1. Acquire | `look` with `yaw_rad` steps, or one operator `pixel` | disk pixel known | if look saturates, fail honestly |
| 2. Put the hand on it | `reach` + that pixel + `side=right` (or closer hand) | projection inside tolerance | no glow to wait for: press is cheap |
| 3. Grab | `grab` right | **floppy appears at the projected wrist** | nothing appeared + `saturated` = out of reach |
| 4. Carry | `reach` toward the **green** slot centroid | projection inside tolerance | centroid from the hue gate, §8.2 |
| 5. Insert | `release` | closure 0 | |
| 6. Headset | `reach` up, `grab`, carry over head, `release` | **unmeasured** | `saturated` is the allowed fail; do not invent a stretch |

Stay in spawn: this script sends no `AUTH_ROOT_VEL`. If a later spec adds a step, cap net translation from the session's start root at **1.0 m**.

Two hands are allowed by the game and not required here. Use one hand.

## 11. In-level vocabulary (closed, not a plan)

When a level exists, the writer (script or d1) picks from the same intent set. Policy the operator stated, as choices, not search:

- grab whatever is closest (throwable or gun)
- fight the closest threat, or the one that can shoot from here
- empty gun still `fire` (click is telemetry)
- after teleport: `hold`, re-acquire, do not walk back
- time advance is `twitch`, then `hold` — not a step

This spec does not implement in-level play. It only forbids inventing skills outside this list.

## 12. Writers

All writers speak `clank.intent/1`. If a path needs the operator to hold a mouse for the **servo**, the scaffolding leaked. An operator click that **names a pixel once** is a legal stand-in for d1 `box`.

| Writer | When | What it proves |
| --- | --- | --- |
| Hub script | first demo | adapter + body can pick up a disk whose pixel was scripted or clicked |
| Operator pixel | calibration and unblocking acquire | the servo, independent of detection |
| d1 | **after E5**, and only in `hold` | skill + side (+ pixel only on a runtime that has `box`) |

d1 on the measured GPU endpoint:

- Digest `choice`×2 + `noul` is 62–176 ms `[MEASURED 2026-10-30]`. That is a ~5 Hz hold-time chooser. It is not a servo.
- Image calls are 0.5–1.1 s and spike GPU util to 100% with ~1.2 GB VRAM headroom next to Superhot `[MEASURED 2026-10-30]`. Optional; not required for this spec.
- `box` is unavailable on that llama.cpp build. Do not pretend grid-choice is grounding.
- Parent §9 "CPU only" / "no GPU residency in v1" is amended: GPU d1 **may** sit beside a captured run if VRAM is re-measured, but the adapter must run correctly with d1 **off**. The servo never waits on it.

E5 (majority-class vs digest on ~100 labelled frames) stays the gate to plug d1 in. Frames that grab never changed will make majority class `hold`; do not run E5 as a substitute for the hub script.

## 13. Status the writer reads

Publish beside `body_state.json` (adapter-owned file, not a `BodyState` field — glow is not achieved physics):

```text
{
  "seq_ack": 0,
  "skill": "reach",
  "side": "right",
  "pixel": [0.42, 0.51],
  "projected": [0.40, 0.48],
  "error": 0.023,
  "glow": "yes" | "no" | "unknown",
  "outcome": "running" | "done" | "saturated" | "no_glow" | "timeout" | "no_frame",
  "flags": 0
}
```

`flags` copies the last `BodyState.flags` so `SATURATED` is visible without parsing two schemas for the common case.

## 14. Amendments to the parent design

These are intentional and local. Do not silently rewrite §9 in the parent file until this spec is accepted.

| Parent | Change |
| --- | --- |
| §8 `reach(side, p)` | `p` may be updated each adapter tick from projection error. Constant E3 offsets remain the no-pixel path. |
| §8 "cheap 2D tracker may update the hand" | Glow + projection **are** that tracker. KCF is not required. |
| §9 output "skill + side only" | Optional `pixel`. Still no 3D, no joints, no OpenXR. |
| §9 "2D→3D is out of v1" | Unchanged. A ray + a glow is not a reconstructed point. |
| §9 "CPU only" / no GPU residency | Adapter is CPU. d1 GPU is optional and off the critical path. |
| §11 grim as the eye | Grim/headless screenshooter is dead. VNC framebuffer is the glow eye. Brain stills remain hold-only. |
| §12 first target The Fall | **Hub first** (`research/superhot-play.md`). The Fall is still the in-level milestone after a floppy lifts. |

## 15. Work order

Pass/fail for each step. Do not start a later step to debug an earlier one.

| Step | Pass | Allowed fail |
| --- | --- | --- |
| S0 Intent schema + tests | round-trip, reject bad pixel/side | — |
| S1 Parametric `look` / `reach` | dashboard or socket can request a yaw or wrist offset; body moves | look saturates at ±35° |
| S2 Projection unit tests | fixture FOV, on-axis / edge / behind | — |
| S3 Measure live FOV and VNC crop map | pose→pixel table in `research/` | map unstable → stop, do not servo |
| S4 Glow pair | two manual stills separate on the 95th-pct score | they don't → glow abandoned, operator calls range |
| S5 Glow in the adapter | `glow` flips on a known on/off pair replayed from disk | — |
| S6 Hub script, disk lift | floppy leaves its slot on video + CSV | `no_glow` or `saturated` recorded with screenshots; residual named |
| S7 Headset-over-head | measured reach vs envelope | `SATURATED` documented |
| E5 | only after S6 produces frames that are not all `hold` | majority-class wins → d1 stays unplugged |

## 16. Review focus

Failure modes this spec implies and that a later plan must pin with tests:

1. **Wrong crop.** Servo zeros error on the Weston panel or a letterbox, never on the disk.
2. **FOV assumed, not measured.** Projection is then a consistent lie.
3. **Glow on `F[t]−F[t−1]`.** Hand approach looks like success.
4. **Adapter auto-grabs on glow.** A time-slow flash picks up the wrong object.
5. **Pixel stuffed into `BodyCommand`.** The robotics contract grows a 2D field every consumer must ignore.
6. **d1 in the 10–30 Hz loop.** A 62 ms chooser is still the wrong layer; a 1 s image call is a stuck reach.
7. **`reach` past 0.20 m.** `SATURATED` must terminate; do not wind up into a fall.
8. **Ctrl-C with grip latched.** Pre-existing; hub script must `release` on a clean stop. Dirty stop may leave the in-game hand closed.

## 17. Out of this spec, still true

- `AUTH_ROOT_VEL` is not honoured in a hold session. Do not build touring to make S6 pass.
- Controller orientation (wrist frame → OpenXR grip) is unmeasured; the aim ray may carry a fixed rotation. If S2 passes in sim and S6 misses in game with glow never appearing, measure this before blaming glow.
- Task 6 of `docs/superpowers/plans/2026-10-29-grab-buttons-through-the-join.md` (hub A/B on squeeze/curl) is **retired**: manual play already proved the game accepts our buttons. Curl remains documented as inert for controller-bound titles.
