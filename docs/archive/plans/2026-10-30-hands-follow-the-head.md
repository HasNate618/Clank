# Hands that follow the head, and a camera we can measure

> **Archived 2026-11-13.** Implementation plan, carried out. Kept for the task breakdown and the gates it defined; the code is the record now.
> Current behaviour: [`README.md`](../../../README.md) · agent rules: [`AGENTS.md`](../../../AGENTS.md).

**Goal:** make the wrist verbs move where their names say, make hand targets head-relative so a
hand stays in view when the body turns, and calibrate the HMD→capture map automatically — so
projection can drive a reach without a human watching the screen.

**Why now:** a hub pickup needs three numbers nobody has: which way the desk is, where the hand
lands in the captured image, and what the camera's field of view actually is. Today's verbs cannot
answer them. `aim` is measured to move the wrist *across the body*, not forward, so every reach
measurement taken through it measured the wrong axis.

## Measured facts this plan rests on

All MEASURED 2026-10-30, offline in a fresh `SonicExecutor` (no game, no live session):

- **The model frame is +X forward, +Y left, +Z up.** Clip wrist positions confirm it: left wrist
  `y = +0.190`, right wrist `y = −0.180`.
- **`hand_target` applies its offset on the wrong axes.** For `side="right"`:

  | argument | clip wrist | resulting target | axis actually moved |
  |---|---|---|---|
  | `forward_m=0.15` | (+0.084, −0.180, −0.252) | (+0.084, −0.030, −0.252) | **+Y (left, across the body)** |
  | `lateral_m=0.15` | (+0.084, −0.180, −0.252) | (+0.234, −0.180, −0.252) | **+X (forward)** |

  Its docstring ("+Y is forward, +X is the robot's left") states the wrong convention. This is the
  same defect already fixed in `dodge_target` by `f4ca2e7`, which documents the correct frame — the
  fix never propagated to `hand_target`.
- **Consequence:** `aim_target(0.22)` pushes the right wrist 0.22 m toward the body midline, which is
  the axis the body is weakest on (8.63 cm / 28.33° error at 30° off-axis), and it saturates. The
  operator's report — "aim just raises hand a tiny bit then goes back" — is this bug, not the TTL.
- **Forward reach was never measured.** Every "reach saturates ~0.15 m" figure was taken through
  `forward_m`, i.e. across the body. The real forward envelope is unknown and probably larger.
- **The hand can be raised into the view without head pitch.** Offline, a wrist target at
  x=0.30, z=+0.20 settles at 7° below the head's horizontal — inside any plausible vertical FOV.
  So the reported "FOV too small to see the hand" is a *target* problem (cross-body, low), not
  necessarily a geometry limit. Re-check after Task 1.
- **Head yaw is asymmetric:** look_left −28.6°, look_right +54.3° from a −86.8° baseline. If the desk
  sits beyond the small side, yaw alone cannot reach it.
- **`View` carries a per-view FOV and `per_view_data_valid`** (`clank/wire.py`), and `wire.fill`
  sets it `False`. The FOV is therefore *declarable*, not only measurable — see Task 3.

## Architecture

Three layers, one new one:

1. **Builders** (`spike/wbc/sonic_executor.py`) — turn named offsets into absolute root-frame wrist
   poses. Task 1 fixes their axes; Task 2 adds head-frame composition.
2. **Composition math** (`clank/projection.py`, stdlib) — yaw extraction and Z-rotation as pure
   functions, so the head-frame rule is testable without MuJoCo, numpy, or a body.
3. **Camera truth** (`clank/wire.py` + one experiment) — declare the FOV we want the game to use,
   instead of inferring it from images.
4. **Calibration** (`spike/wbc/calibrate_view.py`, venv) — sweep, capture, detect, fit. Writes a
   JSON file of measured numbers that later code reads; it never contains a guess.

## Tech stack

`clank/` standard library only. Image decoding and MuJoCo stay in `spike/wbc/` (venv:
`spike/e3/.venv/bin/python`). Detection operates on nested lists of ints, which is what `glow.py`
already takes.

## Spec

Extends `docs/archive/specs/2026-10-30-targeting-adapter.md` §S3 (FOV / crop map) and §8
(signals). Supersedes nothing. Play rules: `research/superhot-play.md`.

## Global constraints

- **`clank/` stays standard-library only.** No NumPy, no PIL, no OpenCV.
- **`BodyCommand` gains no image field.** Pixels remain the intent schema's business.
- **A frame is never fitted with fewer than 3 distinct hand positions.** Two points fit any line;
  the second axis would be pure invention.
- **A fit is reported with its residual.** A calibration file without residuals is not accepted.
- **Captures are only trusted when the game is forward.** Three captures in one session showed
  Steam's UI; the game must be verified, not assumed (Task 4's liveness check).
- **Assumptions stay labelled.** Monado honouring `per_view_data_valid` is an ASSUMPTION until the
  Task 3 experiment measures it.

---

## Task 1 — Fix the hand-offset axes, then re-measure forward reach

**Why first:** every later task moves the hand and reads where it went. On a swapped axis those
readings are garbage.

- [x] Add a stdlib helper in `clank/projection.py`:
      `root_offset(forward_m, up_m=0.0, lateral_m=0.0) -> tuple[float, float, float]` returning
      `(forward, lateral, up)` in the model's `(x, y, z)`. Reuse the existing `FORWARD`/`UP`/`LEFT`
      axis constants rather than restating them — the whole bug is one module stating a convention
      that another module contradicts.
- [x] Tests in `tests/test_projection.py` (write first, watch them fail):
      `root_offset(forward_m=0.15)` → `(0.15, 0, 0)`; `root_offset(lateral_m=0.15)` → `(0, 0.15, 0)`
      i.e. lateral is +Y = the robot's **left**; `root_offset(up_m=0.2)` → `(0, 0, 0.2)`.
      One test pins the *sign* of lateral against `dodge_target`'s documented frame.
- [x] Change `SonicExecutor.hand_target` to build its offset via `root_offset`, delete the wrong
      docstring, state the real convention, and note that `f4ca2e7` fixed the same bug in
      `dodge_target`.
- [ ] Keep `lateral_m`'s sign meaning **unchanged** (positive = robot's left) so `dodge_target` and
      the hand agree; `lateral_m` previously moved +X, so this is a real behaviour change and the
      commit message must say so.
- [x] Re-measure the forward envelope with a throwaway harness (`/tmp`, not committed): for
      `forward_m` in 0.10…0.40 and `up_m` in −0.10…+0.30, step 200 ticks per point, record achieved
      wrist, residual and `FLAG_SATURATED`. Record the largest `forward_m` that keeps residual under
      the 0.10 m tolerance, per `up_m`.
- [x] Write the numbers into `research/strike-results.md` as a dated section, including the retraction
      of the old ~0.15 m forward figure and why it was wrong.
- [ ] `aim_target`/`punch_target` defaults are **not** retuned here — they move onto the correct axis
      and their residuals must be re-read from the harness output before any default changes.

**Done when:** the axis tests pass, the offline harness shows the wrist moving +X for `forward_m`,
and the new envelope is written down with residuals.

## Task 2 — Hand targets relative to the head

**Why:** the game camera *is* the achieved head pose. A root-frame hand therefore swings out of view
whenever the body turns, and no amount of aiming fixes that.

- [x] `clank/projection.py`: add `yaw_of(q_xyzw) -> float` (heading, robust at gimbal-ish poses —
      document the choice) and `rotate_z(vec, yaw_rad)`. Pure, no numpy.
- [x] Tests: `yaw_of` identity → 0, +90° about Z → +π/2 (left, matching the yaw sign used by
      `look_target`), round-trips; `rotate_z` at 0 is identity to the bit (this is the compatibility
      guarantee below), at +90° sends +X to +Y.
- [x] `clank/body.py`: extend `FRAMES` to `("root", "head")` — `"world"` is currently accepted and
      silently ignored by the executor, which is a pre-existing hole; either implement or reject it,
      and say which in the commit message. Validate that `frame="head"` is meaningless for a head
      pose: reject `frame="head"` when `AUTH_HEAD` is set.
- [x] `clank/validate.py`: with `frame="head"`, treat the hand boxes as **offset magnitudes**
      around the clip wrist rather than absolute workspace points, documented as the same kind of
      sanity cap the module already describes ("not a reach model").
- [x] `SonicExecutor`: hold the clip's reference head yaw; when `cmd.frame == "head"`, rotate each
      hand offset by `Δ = yaw(achieved head) − yaw(reference head)` about Z before placing it.
      Compose from the **achieved** head pose in `self.last_state`, not the commanded one — the relay
      sends achieved head to the game, so a composed-from-commanded hand diverges from the camera
      whenever tracking lags. Accept the one-tick (20 ms) lag and document it.
- [x] **Compatibility guarantee:** `Δ = 0` must reproduce today's poses exactly. Test it by comparing
      the composed target against `hand_target`'s output at zero yaw, asserting exact equality, not
      approximate.
- [x] Carry the flag through: `clank/intent.py` gains nothing; the *verb* layer
      (`spike/wbc/body_view.py`) gains `frame` on `reach`, and `verb_to_intent` validates it.
      Pixels stay out of the body contract.
- [x] Verify offline: with a commanded head yaw of +35° and `frame="head"`, the right wrist target
      should rotate to the robot's left by 35° about Z at constant radius. Assert the radius is
      preserved to within 1e-9 and print the achieved wrist for the record.

**Done when:** `frame="head"` exists end to end (verb → command → executor), zero-yaw is bit-identical,
and a +35° yaw carries the wrist 35° with the radius unchanged.

## Task 3 — Declare the FOV instead of inferring it

**Why:** `projection.py` refuses to invent a FOV, correctly. But we *emit* the HMD data, so the FOV
can be a decision rather than a measurement — if Monado honours it.

- [x] Read the installed Monado `remote` driver source for how `per_view_data_valid` and `views[].fov`
      are consumed. Cite file and function in the commit. If the driver ignores them, stop this task,
      record the negative result, and fall back to Task 5's fit for FOV.
- [x] Add `wire.fill(..., fov=None)` writing symmetric `angle_left/right/up/down` from a single
      horizontal+vertical pair and setting `per_view_data_valid = True` when a FOV is supplied.
      Also write `views[].pose` for the eye offsets we assume, and document that assumption.
- [x] Tests in `tests/test_wire.py`: default keeps `per_view_data_valid = False` (unchanged
      behaviour); supplying a FOV marks it valid and writes all four half-angles; angles are radians
      and in range; layout still asserts 376 bytes.
- [ ] **Experiment (the measurement):** run the live session twice with two very different FOVs
      (e.g. 60° and 110° horizontal), capture a frame each with the body in `hold`, and compare.
      If the image changes as a wide-angle view should (same scene, more of it visible), Monado
      honours the field → record MEASURED. If the images are identical → negative result, record it,
      and the FOV comes from Task 5's fit instead.
- [ ] Record either result in `research/` with the two frames' paths and a one-line diff statistic.

**Done when:** either Monado honours a declared FOV (with the frame pair as evidence) or that route
is closed with a recorded negative result and no code left half-enabled.

## Task 4 — A capture path with a liveness check

**Why:** three captures this session silently returned Steam's UI. A calibration that fits Steam's
window would look like success and be worthless.

- [ ] Reuse `clank/preview.py:PreviewCapture.capture_once` for capture; do not write a second
      screenshooter wrapper. Verify it returns the path it wrote (the tool ignores its argument and
      writes a timestamped name in the cwd).
- [x] New `spike/wbc/frames.py` (venv): `load(path) -> Frame` where `Frame` holds `width`, `height`
      and `rows` (nested lists of `(r, g, b)` ints), plus `downscale(step)`. Decoding lives here
      because `clank/` may not import PIL.
- [x] `spike/wbc/calibrate_view.py`: `liveness(captures)` — capture at head yaw 0 and +20°, and require
      the frames to differ substantially (reuse the earlier frame-diff statistics: a real turn moved
      15k–17k pixels at mean 3.3–4.3, a stale frame moved ~1k at 0.17). Return a verdict, not a bool
      with a buried threshold: `("live", diff)` / `("stale", diff)` / `("steam", diff)`.
- [x] Unit-test the pure parts (`downscale`, the diff statistic) with synthetic frames; the capture and
      verdict run attended.
- [x] A `--dry-run` that captures one frame and prints the verdict and the green-gate reading **without
      moving the body** — the first thing to run, and it needs no approval to be safe.

**Deviation, measured.** The plan asked for a three-way verdict (`live` / `stale` / `steam`).
It cannot be honest: a Steam-UI frame and a game frame are nearly identical by mean colour
(73.1/105.8/117.1 vs 76.3/108.8/120.8, measured today), so no colour test separates them.
What separates them is motion — a Steam window does not follow the head — so the verdict is
`responding` / `not responding`, and *not responding* covers both a stale capture and the
wrong window. Telling those apart needs one human glance, which is cheaper than a
classifier that would be guessing. Shipped as `shot` / `motion` / `sweep` subcommands;
`sweep` is Task 6's tool as well.

**Done when:** `--dry-run` distinguishes a live game from a Steam window on demand, and the numbers
it prints are recorded in `research/`.

## Task 5 — Fit the hand in the image (crop map + FOV + projection validation)

**Why:** this is the only step that can prove `project_point` is right, and it simultaneously answers
"can we see the hand at all".

- [ ] With the game forward and the body in `hold`, capture N=5 frames → baseline noise mask
      (`|Δ| > 40` on any channel). Mask the CRT region by *measuring* which pixels are never stable,
      rather than hard-coding a rectangle.
- [ ] For each of ≥3 wrist targets (spread across forward/lateral/up, head yaw fixed), move with
      `frame="head"` off, settle, capture, subtract the baseline, take the changed-region centroid.
      Reject a sample whose changed fraction exceeds 25% (the body moved the whole view — a failed
      settle, not a hand).
- [ ] Fit `(u, v) = (cx + sx·x_n, cy + sy·y_n)` from the measured centroids against the *achieved*
      wrist positions from `body_state.json`, least squares. Report residuals; refuse to write the
      file below 3 samples or when residual RMS exceeds 5% of frame width.
- [ ] Use the fit to test `project_point`'s axis **directions**: the sign of `sx`/`sy` compared with
      the projection's +X-forward assumption. A sign mismatch is the expected failure mode and the
      main thing this task exists to catch.
- [ ] Optionally fit a FOV from the same samples if Task 3 failed; then report both and let later code
      prefer the declared FOV.
- [ ] Write `view_calibration.json` (frame size, crop rect, centre, scale, FOV if fitted, residuals,
      sample count, capture paths) next to the session, never into the repo.

**Done when:** a fit exists with residuals under the limit, the axis signs agree with `projection.py`,
and the hand's measured image position agrees with the predicted one to within the stated tolerance.

## Task 6 — Find the desk automatically

**Why:** the operator reports "we are looking at a wall". This answers where it is, with numbers.

- [ ] `calibrate_view.py sweep`: head yaw from −60° to +60° in 10° steps, dwell long enough to settle,
      capture each, run `green_centroid`. Print a table `yaw | green fraction | centroid | verdict`.
- [ ] Pick the yaw maximising green fraction subject to the fraction clearing `MIN_GREEN_FRACTION`;
      report the margin, not just the winner. If no yaw clears it, report **not found** and the
      per-yaw fractions — a negative result that immediately says whether the desk is beyond the yaw
      envelope (compare with the measured −28.6°/+54.3° asymmetry).
- [ ] Also measure the achievable yaw envelope in the same run (achieved head yaw per commanded yaw)
      and write it down; the answer decides whether the next plan needs `AUTH_ROOT_VEL`.
- [ ] Reuse `hold_pose.py` for the hold loop rather than duplicating the TTL-defeating re-send; if it
      needs a `--frame` flag or a capture hook, add it there once.
- [ ] Record the sweep table in `research/slot-green-calibration.md` as a dated addendum.

**Done when:** either the desk's yaw is found with margins (and `green_centroid` lands inside the
`view_calibration.json` crop as a cross-check), or "not found" is recorded with the reason the plan
can act on.

---

## Not in this plan

- The teleop panel, held-key control, gamepad input. Useful, but this plan makes them unnecessary for
  the calibration it targets.
- Head pitch. Task 1's harness may show it is needed; if it does, that is its own measurement.
- Retuning `aim_target`/`punch_target` defaults, and any change to the measured `--force-grab` path.
- The closed-loop reach-to-slot servo and the hub pickup sequence. That is the *next* plan, and it
  needs this plan's numbers.
- The brain (E5). It writes intents; nothing here changes its socket.
- Anything that moves the body without an operator present.

## Review focus

- **Does the axis fix have a test that pins the convention, not just the arithmetic?** The bug
  existed because two modules disagreed about a frame; a test that only checks `(0.15, 0, 0)` would
  not have caught it.
- **Is `frame="head"` composed from achieved or commanded head?** Commanded is easier and wrong.
- **Is zero-yaw bit-identical?** Any drift here silently invalidates every earlier measurement.
- **Does any fitted number enter the repo without a residual next to it?**
- **Is the liveness check able to fail?** Show the Steam-UI case returning `("steam", …)`.
