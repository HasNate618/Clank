# Pixel servo in the sim — it moves the hand, it does not arrive

- Date: 2026-11-01
- Question: when the brain names a pixel, does commanding the wrist to
  `unproject(pixel)` put the hand **on that pixel**?
- Probe: `spike/wbc_game/servo_sim_probe.py`, run under
  `spike/e3/.venv/bin/python`. Real MuJoCo + SONIC, standing clip
  `squat_001__A359` frame 0. Closed loop: project the achieved wrist → pixel
  error → `servo_step` → command the absolute wrist pose → step 20 ms.
- FOV: symmetric 90×90 **fixture** (the compositor's live FOV is still
  unmeasured). Target and projection share one FOV, so the loop is
  self-consistent — this measures convergence, not absolute pixel accuracy.
- Status: `[MEASURED]`, 2026-11-01, one run per cell, no repeats.

## 1. Where the hands actually are

Settled from the hold clip, root-relative:

| | value |
| --- | --- |
| head | (−0.002, 0.039, 0.402) |
| right wrist | (0.079, −0.308, −0.203) |
| wrist in head frame | forward 0.087, left −0.280, **up −0.638** |
| wrist pixel @ 90×90 | **(1.309, 1.414)** |
| gaze toward wrist | yaw −72.8°, pitch **−82.2°** |

The hand hangs 0.64 m below the head and 73°/82° off the gaze axis, so its pixel
is far outside `[0, 1]`: **under a 90° frustum the hold clip's hand is not in
view at all.** In-frame targets are therefore nowhere near the hand, and points
near the hand are not in frame. That alone makes the pixel servo's usual regime
unavailable from this clip.

## 2. Convergence

In-frame targets, servo depth 0.25 m, up to 120 ticks, deadband 0.005:

| target | start err | best | final | converged |
| --- | ---: | ---: | ---: | :---: |
| (0.35, 0.55) | 1.266 | 0.333 | 0.491 | no |
| (0.50, 0.60) | 0.873 | 0.525 | 0.570 | no |
| (0.65, 0.55) | 0.781 | 0.510 | 0.542 | no |
| (0.50, 0.35) | 0.535 | 0.346 | 0.347 | no |

**Directionally correct, never convergent.** Every target improves at first — by
0.3 to 0.9 of the frame — then stalls, and in three of four cases the final error
is *worse* than the best, so the loop dithers instead of settling. No falls.

## 3. Why: the arm saturates, and pixel error cannot see depth

Same loop with the depth varied, reporting the commanded-vs-achieved gap:

| target | depth | pixel err | \|cmd − ach\| | achieved wrist |
| --- | ---: | ---: | ---: | --- |
| (0.50, 0.60) | 0.25 | 0.513 | 0.316 | (0.426, −0.087, 0.604) |
| (0.50, 0.60) | 0.45 | 0.253 | 0.187 | (0.464, −0.260, 0.424) |
| (0.50, 0.60) | 0.70 | **0.074** | 0.200 | (0.158, 0.579, 0.148) |
| (0.50, 0.35) | 0.25 | 0.465 | 0.260 | (0.334, 0.313, 0.593) |
| (0.50, 0.35) | 0.45 | 0.289 | 0.203 | (0.139, 0.537, 0.513) |
| (0.50, 0.35) | 0.70 | **0.082** | 0.253 | (0.187, 0.600, 0.319) |

Two facts, both important:

1. **The arm never reaches the commanded point.** The gap is 0.19–0.32 m at every
   depth, matching the envelope probe's 0.10–0.18 m residual. The body saturates
   and the servo keeps commanding a point it cannot occupy.
2. **Pixel error falls with depth while the positional gap does not.** 0.51 → 0.07
   from 0.25 m to 0.70 m, because angular error shrinks with distance. A pixel
   error of zero means **on the ray**, not **at the point** — which the adapter
   spec already warned about in §7.3, now measured.

So "the pixel servo converged to 0.074" would be a true number about a fact that
does not matter: the hand was 0.20 m from where it was told to go.

## 3.5 It is not a sign error, and the signal can vanish

Checked before blaming the servo, because "moves a lot and never settles" is also
what a flipped axis looks like. `unproject` round-trips **exactly** against the
live head pose, not just the fixture:

| target pixel | depth | pixel → point → pixel | error |
| --- | ---: | --- | ---: |
| (0.5, 0.5) | 0.3 | (0.298, 0.037, 0.400) → (0.5, 0.5) | 0.0 |
| (0.2, 0.8) | 0.7 | (0.699, 0.350, 0.005) → (0.2, 0.8) | 0.0 |
| (0.8, 0.2) | 0.7 | (0.698, −0.283, 0.791) → (0.8, 0.2) | 0.0 |

So the commanded pose provably projects onto the target pixel. The convergence
failure is reachability, full stop.

It also exposed a failure mode the loop silently swallows: the wrist's forward
component is only 0.087 m, so it is often **at or behind the head's image plane**,
where `project_point` returns `None`. Nudging the wrist laterally, 2 of 3
positions gave `None` — the servo had no error signal at all, and the probe's
loop simply breaks. A pixel servo needs its wrist in front of its own camera.

## 4. What this means

- The pixel servo is **not validated for the task**. It has never terminated on
  its deadband in this stack.
- Depth is doing all the work and there is no depth sensor. The loop closes on
  the game's blue highlight, not on a reconstruction — which is the design, but
  it means the servo is a *ray aligner*, and the grab must be gated on the cue.
- **The hold clip is a poor base pose for reaching.** Hands down at the sides,
  out of the frustum, is fine for "hold still while the model thinks" and bad for
  "reach where you looked". Freeze-hold chose this clip for stability; a task
  that needs the hand visible wants a different anchor frame, or a pitch axis on
  `look` (which only yaws today).

## 5. Follow-up, in order

1. Measure the live FOV (adapter spec S3). Every pixel number above is scaled by
   an invented FOV.
2. Decide the base pose question in §4 before more servo tuning: no amount of
   gain fixes an arm that cannot see what it is reaching for.
3. Give `look` a pitch component, or pick an anchor frame whose hands sit inside
   the frustum.
4. Only then re-run this probe expecting convergence.

## 6. Negative and unmeasured

- One run per cell, one seed, no repeats, no confidence interval.
- FOV is a fixture; absolute pixel errors against the game are unknown.
- No left arm.
- No measurement of whether entering grab range (blue highlight) happens at these
  errors — that needs the game.
- The deadband (0.005) was never reached, so its value is untested.

## 7. Hands-in-view anchor: the servo lands, then leaves

Frame 0 hangs the wrist out of the frustum, so every in-frame target was
unreachable *and* every reachable point was out of frame. Frames 95 and 332 of the
same clip put the wrist 0.55 m forward with its pixel at (0.73, 0.72) — inside the
frame. Re-ran both the hold soak and the servo from there.

### 7.1 Hold stability now depends on the anchor

Through `bus.take()` with the latch fix, 20 s:

| anchor | net | RMS last 10 s | fall |
| --- | ---: | ---: | :---: |
| frame 0 | 0.0226 m | **0.000069 m/s** | no |
| frame 332 | 0.0432 m | 0.00768 m/s | no |
| frame 95 | 0.0478 m | 0.00594 m/s | no |

Frame 0 is a settled standing pose; 95 and 332 are mid-motion with the arm raised,
so their reference carries velocity the policy keeps acting on. That is a
**trade-off, not a win**: the steady anchor cannot see its own hand, and the
anchor that can see its hand is ~100× less steady.

### 7.2 The servo now reaches the pixel and then oscillates

Anchor 332, wrist pixel (0.734, 0.718), depth 0.544 m, 150 ticks, deadband 0.005:

| target | start | min | final | \|cmd − ach\| | converged |
| --- | ---: | ---: | ---: | ---: | :---: |
| (0.83, 0.72) | 0.097 | **0.005** | 0.102 | 0.114 | no |
| (0.63, 0.72) | 0.118 | 0.028 | 0.133 | 0.126 | no |
| (0.73, 0.87) | 0.143 | **0.005** | 0.109 | 0.119 | no |
| (0.50, 0.50) | 0.305 | 0.014 | 0.117 | 0.102 | no |

**It lands on the pixel (0.005) and then leaves it.** The aiming is right; the
loop does not hold. Tracking lag is 0.10–0.13 m, so the achieved wrist is always
a step behind the commanded point while the head is still settling.

### 7.3 Damping makes it worse

Hypothesis was overshoot from a unity-gain full step, so blend toward the target:

| target | gain 1.0 | 0.5 | 0.3 | 0.15 |
| --- | ---: | ---: | ---: | ---: |
| (0.50, 0.50) final | **0.146** | 0.540 | 0.668 | 9.9 |
| (0.83, 0.72) final | **0.095** | 0.345 | 9.9 | 9.9 |

9.9 is `project_point` returning `None` — the wrist left the frustum and the loop
lost its error signal. Unity gain is the best of the four, so the oscillation is
not an overshoot problem and **no gain fixes it**. This needs a control law, not a
knob, and that is where I stopped rather than keep fishing.

### 7.4 What this means

The pixel servo is a **ray aligner with ~0.10 of frame steady-state error**, not an
arrival mechanism. That is consistent with the design, not a failure of it: the
adapter spec already says a zero pixel error means *on the ray*, and the arrival
gate is the game's blue highlight. So:

- Do not set a pixel deadband tighter than ~0.10; it is unreachable and the loop
  will spin. The measured best-case steady state is 0.095–0.146.
- Terminate a reach on the glow cue, not on pixel error.
- Use frame 332 (or 95) as the *reaching* anchor and frame 0 for standing still
  during a PLAN call, if the adapter can switch. That is a design decision, not a
  measurement.

### 7.5 Negative and unmeasured

- One anchor pair, right arm, one seed, no repeats.
- No pitch on `look`, so the head never turns to help the ray.
- Not measured: whether the 0.10–0.13 m lag shrinks once the body settles, i.e.
  whether the oscillation is transient. The probe ran 3 s after a 1.2 s settle;
  the hold measurement says the body is still moving at 0.006–0.008 m/s at that
  anchor, so it probably is not settled.
- No cue-gated termination test, because that needs the game.
