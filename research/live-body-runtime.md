# Live body runtime — the latch holds, the startup transient falls

- Date: 2026-11-01
- What: drove a freshly-restarted `spike/wbc/body_view.py` over its real
  `control.sock`, reading the real `body_state.json` at 10 fps. This is the
  runtime the adapter will use — `body_view` owns a `CommandBus`, so the latch fix
  is exercised through `bus.take()` exactly as in production, not through a probe.
- Setup: `body_view.py --fps 10 --width 560 --height 420 --views three_quarter`.
  Env needed: the previous process's `LD_LIBRARY_PATH` (1357 chars, includes
  `/run/opengl-driver/lib` and two libglvnd stores) plus `MUJOCO_GL=egl`. Without
  it every GL backend fails from a plain shell: `egl` and `osmesa` give
  `NoneType.eglQueryString`, `glfw` gives "no OpenGL platform library".
- Status: `[MEASURED]`, 2026-11-01, one session.

## 1. The latch works in the live loop

Reach right, forward offsets, then 5–7 s of polling. `res_r` is the residual
against the commanded wrist; the wrist stops changing once the command expires.

| `forward_m` | settled `res_r` | settled wrist (root frame) | fall |
| ---: | ---: | --- | :---: |
| 0.20 | 0.024 | (0.284, −0.123, −0.212) | no |
| 0.35 | 0.069 | (0.467, −0.079, −0.160) | no |
| 0.45 | 0.017 | (0.470, −0.182, −0.040) | no |
| 0.55 | 0.25–0.47 | (0.573, +0.094, +0.255) | no |
| 0.65 | 0.187 | (0.664, +0.064, +0.156) | no |

Up to 0.45 m the body tracks and then **freezes exactly** — five consecutive
samples identical to the millimetre. That is the freeze-hold working through the
real bus, and it is the end-to-end confirmation the unit tests could not give.

Past 0.45 m it tracks badly and swings the arm across the body (y goes positive,
i.e. to the robot's left, from a right hand). Consistent with the envelope probe:
the well-tracked band ends around 0.45–0.59 m.

## 2. A latency-to-first-command fall

The **first** attempt did not go this way. A 0.55 m reach sent 20 s after process
start produced `flags = 5` (`FLAG_SATURATED | FLAG_FALL_RISK`) at +4.5 s with the
wrist at (0.278, +0.663, +0.138), and `body_view` exited:

```
body_view: FALL detected, stopping
```

The idle baseline before that reach was `flags = 1` with `res_r = 0.132`, i.e. the
bus had already latched a pose the body was not tracking. In that one session the
**freeze-hold latched a state the body was still settling into**, and the reach
was issued from there. Re-running the same 0.55 m reach against a settled body
(row 1) does not fall.

So: the fall is attributable to **latching before the body settles**, not to the
reach distance. A latch taken at process start freezes whatever transient exists,
and every later command starts from it. That wants a settle-on-clip before the
first latch is allowed — `rest()` until the residual drops, then accept holds.
Not implemented; recorded as a defect.

## 3. `FLAG_SATURATED` is useless, confirmed live

At the 0.02 m default tolerance:

- idle, nothing commanded that anyone asked for: `flags = 1`, `res_r = 0.132`.
- a *good* 0.45 m reach: `res_r` bottoms out at 0.017, and the body flag does not
  become informative because **the residual is OR'd across head, left and right**
  — one badly-tracked channel keeps the flag set, and the reader cannot tell which.

This is the same conclusion as `research/sonic-reach-envelope.md`, now from the
real runtime. `brain.reach_outcome` routing through the glow cue is the right
answer; the flag cannot carry the claim.

## 4. The probe is not a stand-in for the live loop

Two probe replays failed to reproduce the fall:

| replay | fall |
| --- | :---: |
| one command held 7 s, 0.20/0.30/0.45/0.55/0.65 m | none |
| exact verb command through a `CommandBus`, ttl 2 s | none |
| same, ttl 60 s | none |

In the probe the wrist's y stayed in [−0.301, −0.053]; in the live fall it reached
+0.663. So the probe's "no fall" results are **not evidence about the live loop**.
The remaining difference is the startup latch in §2, which the probe never
reproduces because it settles on the clip first.

Consequence: every `[MEASURED] no fall` in `research/sonic-reach-envelope.md` and
`research/pixel-servo-sim.md` is a statement about the probe harness. Do not quote
them as live safety.

## 5. Operational notes

- Rendering needs the inherited `LD_LIBRARY_PATH`; `nix develop` alone does not
  provide a working EGL for the e3 venv (the venv's numpy also breaks under the
  devshell unless `PYTHONPATH`/`PYTHONHOME` are unset).
- Two `body_view` instances had been running for six hours, writing to
  `state/Clank/` and `state/SuperBot/` simultaneously, and the `Clank` log carried
  173 lines of "another body view is already listening". One was stopped and
  replaced; the `SuperBot` one was left alone.
- `:8765` is held by `python3 -m superbot` — the pre-rename package, which no
  longer exists on disk. It will collide with `run.sh`'s noVNC path.

## 6. Unmeasured

- One session, one clip, right arm.
- No repeated startup-transient trials, so §2's attribution rests on a single
  observation against a single non-falling control.
- Nothing with Monado or the game attached.
- The settle-before-latch fix in §2 is proposed, not implemented or tested.

## 7. Fix: settle on the clip before the bus may latch

`spike/wbc/body_view.py` now takes `--settle-s` (default 2.0) and, before the main
loop, steps `rest()` and publishes, so the first pose the bus can latch is one the
body can stand in. Commands that arrive during the settle queue in the bus.

Same session parameters, same reach that previously fell:

| | before | after |
| --- | ---: | ---: |
| idle `res_r` at the latched pose | 0.132 | **0.0183** |
| reach 0.55 m as the first command | FALL_RISK at +4.5 s, process exited | no fall, settles at 0.13 |

The residual against the latched pose drops 7×, and the reach completes. The
proposed mechanism held: the latch was freezing a startup transient, and a
transient is not a pose to hold.

One session, one trial. The retest is the same reach that failed, which is the
right control, but the "before" column is still a single observation.

## 8. Full stack launched: the robot is driving Job Simulator

Started the whole thing programmatically — `./run.sh` (which takes the `nix develop`
branch, since `weston`/`novnc` are not on the bare PATH) then
`POST /api/session/start` on the dashboard. Session `20261101T223603Z-27abb1`.

| component | state |
| --- | --- |
| Monado (remote driver, 4242) | alive, client connected |
| Weston (VNC backend, 1920x1080, socket `clank-27abb1`) | alive |
| noVNC proxy (6080) / VNC (5901) | alive / alive |
| pose relay | alive — `handshake ok`, 50 Hz, heading +90 declared |
| Job Simulator under Proton + OpenComposite | alive |
| body_view | alive, autostarted by the orchestrator |

Pose stream health from the session's `body_pose.csv`, last 500 rows:

- **500/500 `live`**, no stale rows.
- mean Δt **20.00 ms**, max gap 20.9 ms — a clean 50 Hz with no drops.

So the loop works end to end: MuJoCo → `body_state.json` → relay → Monado →
OpenComposite → the game. This is the first time that has been recorded as a
single verified run rather than inferred from parts.

Two operational notes worth keeping:

1. The orchestrator scrubs `PYTHONPATH` for the body subprocess (the running body
   has `PYTHONNOUSERSITE`/`PYTHONHASHSEED` and nothing else), which is why the e3
   venv works under `nix develop` even though the devshell's `PYTHONPATH` breaks
   that venv's numpy if you invoke it directly.
2. A manual `body_view` must be stopped before a launch, or two of them write to
   the same `sessions/body` directory.

## 9. Headless capture exists; the earlier table was wrong

**Corrected 2026-11-01.** The old version of this section said every capture path
needed the operator's VNC password. That is false: two paths were measured, neither
uses a password, and both produced real 1920x1080 game frames. Full commands, pixel
statistics and the failed attempts are in `research/capture.md`.

| path | outcome |
| --- | --- |
| `weston-screenshooter` | **works.** It ignores its output argument and writes `wayland-screenshot-<ts>.png` into the **cwd**; the old "exits 0, writes nothing" was a wrong-place lookup. Vertically mirrored under this GL/VNC compositor. `clank.capture.capture_frame` wraps it. |
| X11 grab of the `Job Simulator` window on Xwayland `:1` | **works, upright, no VNC.** `spike/wbc_game/x11_grab.py` uses `XGetImage`; window `0x4a00001`, 1920x1080. |
| `spike/vnc_grab.py` on 5901 | still fails: RSA-AES256 against a **local account** (`VNC: wrong user ''`). Not pursued, by constraint. |
| `ffmpeg -f x11grab` | this build has no `x11grab`. |
| `weston-debug -l` | no pixel/screenshot scope (log, scene-graph, timeline, proto, vnc-backend, renderer details only). |
| `spike/c0/capture_game.sh` | Superhot-era dead code, wrong stack. |
| Weston `--password` | does not exist in Weston 16.0.0; auth is neatvnc + PAM. |

One real caveat, also measured: on the VNC backend `weston-screenshooter` only
completes while an authenticated VNC client is attached. The live session has the
operator's browser connected, which is why capture works there. A fresh unattended
session with no browser should use the X11 path (`:1` is already recorded by the
orchestrator as `processes.weston.display`), or an authenticated client must be
attached first - and that needs the password.

Orientation: the compositor logs `screen capture uses y-flip: yes`; its PNGs are
mirrored across the horizontal axis. The VNC canvas and the X11 grab are upright.
Do not compare the two orientations without flipping one.
