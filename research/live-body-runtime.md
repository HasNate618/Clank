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
