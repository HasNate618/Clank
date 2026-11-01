# R6 — does the body policy fit beside the game?

Measured 2026-10-29 against a live C0 session: Monado `remote` driver, private
Weston (VNC backend, socket `clank-a36287`), Superhot VR under Proton bound as
`Unity Application`. Nothing ran on the host desktop.

R6 was the last open gate standing between the body spike and the join: the
design wants a 50 Hz body loop *and* a running game on an 8 GB laptop card.

## Verdict

**PASS, with the standing 50 Hz shortfall unchanged.** The game costs the body
nothing measurable, and the body costs the game nothing measurable.

## Method

Body viewer run as a separate process beside the session — the deployment shape,
not a simulation of it. Ticks counted from `body_state.json` mtime changes;
GPU from `nvidia-smi`; game throughput from the exe's own CPU time in
`/proc/[pid]/stat`; system contention from `/proc/loadavg`. Sampling at 1 Hz.

Two windows, 45 s each: **A** game alone, **B** game + body at `--fps 50` with
two views.

| metric | A: game alone | B: game + body |
| --- | --- | --- |
| GPU memory peak | 1144 MiB | 1321 MiB |
| GPU utilization mean | 31.5% | 31.7% |
| load1 mean / peak | 1.93 / 2.08 | 2.20 / 2.41 |
| game cores used | 0.42 | 0.51 |
| body viewer, 2 views | — | 44.2 Hz |
| viewer errors | — | 0 |

Cross-check with the no-game baseline from `research/gl-render-cost.md`:

| configuration | no game | game running |
| --- | --- | --- |
| policy only (`--no-render`) | 45.8 Hz | 45.8 Hz |
| two views | 44.5 Hz | 44.2 Hz |

The body adds **177 MiB** of GPU memory and does not move utilization. The game
loses nothing measurable; it even reads slightly higher because its own CPU time
rises while the loop is up (0.42 → 0.51 cores), which is contention it absorbs
without losing throughput.

## The 50 Hz question, stated honestly

The body does **not** reach 50 Hz — it reaches ~44 with two views. That is the
pre-existing policy ceiling measured on an idle machine (encoder 7.54 ms +
decoder 5.42 ms + physics 4.71 ms pinned to one thread each), not a co-residency
penalty. R6 answers only "does the game cost the body anything"; the answer is
no. Whether the submit stays under 50 Hz or the design target moves to 45 Hz is
a separate decision, and it is open.

## Three instrumentation errors, recorded

Each of these produced a wrong number that looked like a finding:

1. **13.4 Hz render cost.** The sampler spawned `nvidia-smi` and `ps` every
   50 ms inside its own window. The instrument starved the subject. Fixed by
   sampling at 1 Hz from `/proc` only.
2. **"The game is frozen" (CPU time stuck at 0.08 s).** The pid monitored was
   the Proton wrapper, sitting in `do_wait`; the real exe was a different
   process at 28% CPU. Fixed by selecting the busiest `SUPERHOT_VR.exe`.
3. **`body_hz: 1.0`.** The window sampler polled once per second, so it could
   never observe more than one tick per second — a ceiling of exactly 1.0 by
   construction. Fixed by counting mtime changes the way the standalone
   counter already did.

The pattern is the same in all three: the measurement harness, not the system,
was the thing under test.

## Limits

- **No in-game frame rate.** Game responsiveness is proxied by the exe's CPU
  time and GPU counters. There is no in-game counter, and reading game memory is
  an anti-goal.
- **No frame capture during the session.** The preview here is VNC-backed
  (5901 + proxy 6080) and `weston-screenshooter` blocks — the recorded
  `renderer_wait_for_present` stall. Pixel-motion measurement was therefore not
  available, which is why the CPU-time proxy was used. *(Superseded 2026-11-01:
  `weston-screenshooter` works while an authenticated VNC client is attached, and
  an X11 grab of the game window needs no VNC at all; see `research/capture.md`.
  The stall above was not reproducible, and the earlier "blocked" verdict was a
  wrong-place lookup plus a real no-VNC-client dependency.)*
- One machine, one resolution (640×480 body tiles, 1920×1080 session).
- Windows are 45 s. A longer run could surface thermal or memory growth that
  this does not see.
- The game sat in the tutorial; The Fall's heavier scenes were not running.
