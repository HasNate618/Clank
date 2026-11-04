# Reach: how the hand gets to a point, and why it kept missing

Measured 2026-11-03 on the live stack (session `20261104T000639Z-f772dd`), Job Simulator
running, relay up. Everything below is a number from a run, not an estimate.

## The thing that was wrong was the target, not the loop

The streaming integrator (`reach_ho.py`, `grab_ho.py`) is an oscillator by construction:
`offset += 0.7 * error` at 6–8 Hz against a plant that settles in 0.77–1.05 s with **+75 %
overshoot** on the up axis. That is the "dancing". But fixing the loop alone would not have
made the hand arrive, because five separate targeting errors were stacked underneath it.

### 1. The relay's yaw offset cancels — and applying it mirrors the target

The relay rotates the *whole player* by `--yaw-offset-deg` about the world origin. The depth
image is rendered from that rotated camera, and the object it shows has not moved relative to
the robot. So the rotation cancels exactly:

    p_robot = R^-1( R(head) + R(q_head) d n ) = head + q_head d n

Applying the offset to the camera puts the point in the rotated world and, without undoing it,
**mirrors the target about the robot**. Measured on the JobBot card:

| convention | point (pelvis frame) | right hand | left hand |
|---|---|---|---|
| rotated camera (+90°) | (0.55, **+0.33**, 0.12) | 0.78 m | **0.54 m** |
| **raw head (correct)** | (0.36, **−0.52**, 0.12) | **0.44 m** | 0.94 m |

The card is on the robot's **right**. The rotated convention put it on the left, which made the
left hand look closer and sent every reach in this session across the body. The error was
caught by the operator looking at the robot and saying the right hand was obviously nearer —
not by anything the loop reported, because the loop converged happily either way.

A second, smaller version of the same mistake: `point_from_pixel` already returns a
**pelvis-frame** point (it unprojects from a root-relative head), so converting it with
`to_rel` applies the root transform twice. That turned (0.356, −0.524, 0.125) into
(0.546, 0.345, −0.672) — a point near the floor on the wrong side.

### 2. `vr_pos` is the wrist link; the reported hand is link + lever

`_achieved()` reports `xpos[right_wrist_yaw_link] + R_link @ VR_OFFSETS[side]`. Measured with
a commanded point and the resulting state:

| quantity | value |
|---|---|
| commanded `vr_pos` | (0.168, −0.317, −0.125) |
| physical link | (0.166, −0.315, **+0.044**) |
| reported hand | (0.232, −0.290, −0.123) |
| lever at that moment | (0.067, 0.025, −0.167) |

The link tracks the command to **2 mm in x and y**; the reported hand is 0.18 m away, and the
lever's direction follows a wrist orientation the policy chooses (**28.5°** of wrist rotation
on the first command). So the residual is mostly a lever the loop cannot predict — and that
is the size of the error budget: 0.18 m of lever swinging by tens of degrees.

### 3. A lean makes a root-frame goal stale

Two to four centimetres of root motion per reach. A goal expressed in the pelvis frame is
stale by exactly that much after the lean; a pixel is valid only for the camera that saw it,
but *the surface it names does not move*. Pinning the goal in the world and re-expressing it
in the current root frame before each command removes that error entirely, and it lets the
step guard be set where a **step** is (>0.15 m) instead of where a lean is (0.03–0.06 m).

### 4. Pinning the head to the clip turns the camera

`look_target(0.0)` asks for a head yaw of 0 while the resting head measures **+21°**, so
"pinning" it swung the camera 21° mid-reach and moved the card ~140 px across the frame. Pin
the head to the pose it is in *now* (`last_state.head`) and the camera moves **0.7°** during a
reach (measured by cross-correlating the frame before and after).

This is the one place where an achieved pose is the correct reference: the goal is stability,
not a particular orientation.

### 5. The depth stream returns torn frames

The same card pixel read **0.29 m** and, a minute later, **1.20 m**; another read gave a
"goal" **2.21 m** away. Sample three frames and take the median point, and report the spread.

## What the loop is now

    rest  ->  fresh frame  ->  3-frame median depth  ->  unproject with the relay's camera
          ->  absolute VR target  ->  1.2 s dwell  ->  damped correction (gain 0.6)

plus a step guard on the root and a plausibility check on the goal.

## Measurements

| target | distance | result | commands |
|---|---|---|---|
| counter surface, left hand | 0.41 m | **26 mm** | 5 |
| JobBot card, right hand | 0.44 m | **94 mm** | 4 |
| JobBot card, left hand (cross-body) | 0.94 m | not attempted, unreachable | – |

The card stalls at 94 mm: the corrections keep asking for a lower point and the hand stays
6–7 cm high, so this is the arm's limit at that extension, not the loop's. The wrist lever is
0.17 m here (measured `(−0.054, −0.163, −0.059)`), so most of the residual is lever, and the
policy will not rotate the wrist further to give it back.

Root lean 2–3 cm per reach; no step. Gain 1.0 overshoots (152 → 88 → 56 → 42 mm); gain 0.6
converges (179 → 55 → 33 → 35 → 26 mm).

## The honest limit

**The card is 0.44 m from the resting right hand** and 0.94 m from the left. That is just
outside the 0.40 m steering clamp and inside the 0.41–0.51 m measured travel, which is why it
converges to 94 mm and stops: the arm is at its extension limit, and the last centimetres are
wrist-lever, not aiming. Getting closer needs either a body step or a wrist orientation the
policy will actually hold.

## Two false levers, both rejected by measurement

- **Raise the lateral clamp** (0.18 → 0.34 m). The cap became the oscillation amplitude; the
  original value was measured-safe.
- **Aim the link at `goal − lever`** instead of correcting the reported error. It assumes the
  policy puts the link where asked; on a raised target the link sat **0.17 m above** the
  command while the lever swung 0.20 m, and it diverged (452 → 150 → 310 mm). Correcting the
  measurement is worse-founded but better-behaved.

## Why IK is not the actuator (but is still useful)

`spike/wbc/arm_override_c3j.py` already implements damped Gauss-Newton on the arm inside the
policy, and `spike/wbc/logs/c3j_summary.json` has the verdict:

| variant | wrist position error |
|---|---|
| **policy, untouched** | **0.0114 m** |
| orient-only IK | 0.0279 m |
| hard IK override | 0.1440 m |

The arm's PD gains are `KP 14.25` against the legs' `99.1`, so the arm droops ~0.2 rad under
its own weight; an "exact" joint target lands 14 cm short, while the learned policy
pre-compensates for its own droop. **The policy is the right actuator.**

IK's real job is an oracle: offline, damped-least-squares on the seven arm joints converges to
**0.0 mm in 6–7 iterations** for reachable points and reports a **223 mm residual at the joint
limits** for an unreachable one — a reachability answer with zero motion and zero risk of
walking the body.

## Following the object instead of pinning it

The head will not hold a commanded pose. Held against one target for 10 s, the achieved
head wobbled **11.7 deg of yaw**, sat ~16 deg off the command, and its position drifted
**56 mm**. That is the mechanism behind every "the camera turned away" complaint, and it is
not fixable by commanding harder: the pin is a request the policy answers with something
else while it balances the arm.

Consequence: a pixel marked once is stale within a tick. Measured inside a single reach,
the card moved `1009 -> 1122 -> 865 -> 1007 px`.

So the pixel is re-derived instead. The object is fixed, the camera is measured every tick,
and projecting the object's own point through the measured head is accurate: predicting the
**hand's** pixel from its 3D position landed within **18 px** of where the game drew it.

`spike/wbc_game/track.py` does that, and `spike/wbc_game/watch_glow.py` runs beside a reach
and logs, per tick, the object's pixel, the hand's pixel, their separation in pixels, and
the game's glow state.

## The glow is a cliff, not a judgement

Same patch of card, measured on frames from this session:

| state | mean RGB | G − R | min(G−R, B−R) |
|---|---|---|---|
| hand at card | (130, 219, 219) | 89 | **97 – 108** |
| hand away | (153, 166, 169) | 13 | **5.9** |
| hand away, other view | (107, 149, 160) | 42 | **40.4** |

`GLOW_ON = 60` separates every sample with room on both sides. The human walkthrough pair
(`research/evidence/walkthrough-glow-*.png`) shows the same cue on the same family of
objects: the tray rim goes cyan.

**Searching for the glow does not work, and this is worth recording because it is the
obvious wrong design.** Within 120 px of the card, the brightest cyan patch in a frame whose
card was *not* glowing scored **163** against the card's own **5.9**. The room is painted in
the same hue - walls, counter strips, the JOB CART sign - so "most glow-like patch" locks
onto scenery. Only the change at a patch whose location is already known is informative,
which is why following has to come first.

The two walkthrough frames are 283 ms apart but **61 % of their pixels differ**: they are a
labelled pair for what the cue looks like, not for measuring a patch, because the camera had
moved.

## Where the hand actually is

Measured with the watcher during a reach to the card:

```
object (719, 435)   hand (678, 446)   gap 41.8 px   glow 37.6   state off
```

41.8 px at ~0.5 m is about **12 cm** from the patch centre, and the glow is off. The reach
reported a 67 mm residual for the same instant, so the two agree within their own
uncertainty. The gap that remains is the wrist lever: `[+0.133, +0.014, -0.123]`, a 0.18 m
offset whose direction is set by the wrist orientation, which nothing yet controls.

## The glow detector is not validated - correcting an earlier claim

The numbers that made the glow look like a cliff (97-108 glowing against 5.9-40.4 not) all
came from boxes **drawn by hand** on saved frames. Placed automatically, the separation does
not survive:

| sample | reading |
| --- | --- |
| fixed patch at the predicted pixel, 60 px off the glow | 59-93 |
| fixed patch on the glow, same frame | 144-168 |
| max over a +-45 px window, live, hand 10 cm away | 164-176 |
| window max, saved frame, **hand away** | **153.5** |
| window max, saved frame, **hand at the card** | **124.8** |

The window max is not selective: the room is painted in the same cyan, so the brightest patch
within 45 px is often signage. A single frame cannot separate a glowing card from a glowing
wall.

I also claimed the glow "pulses". It does not: over 6 s at a fixed pixel the value drifts
smoothly (144-168). What flickered was the patch drifting on and off the glow as the head
wobbled - the same 11.7 deg wobble that started this thread.

**Decision.** The glow does not gate an action. The reach's own residual does, because it is
measured against the same geometry the command was built from - 2-3 cm inside the envelope.
Calibrate the grab radius by *trying the grab* (close the hand, see whether the object moves)
rather than by watching for a glow, and keep the glow as an operator-visible confirmation.

## The grab: measured, and it failed

Procedure, 2026-11-04: rest, hold the camera, mark the card at (915, 495), reach with the right
hand, close the hand, retract toward the body, then measure where the card is.

```
reach residual at the card      79.0 mm
root drift during the reach     61.3 mm
close_right            ->        ok, closure [0.0, 1.0]
hand after retract              (0.277, -0.083, 0.158)
card after retract              (0.256, -0.521, 0.139)
|card - hand|                   0.439 m
```

**The card did not move.** Its point after the retract is 0.52 m to the robot's right where it
started, while the hand came back to 0.08 m from the centre line: the hand went to the card,
closed, and left without it.

So the last link is unproven, and the numbers say why it is unproven rather than mysterious:

- **It was never inside the grab radius.** 79 mm of residual is the largest of any run today,
  because the target sits at the edge of the arm's envelope and the root drifted 61 mm while
  the arm was out. Every successful reach today converged to 21-53 mm; this one did not.
- **Palm orientation is uncontrolled.** `hand_target_at` aims the wrist with the clip's own
  orientation plus an optional yaw. Nothing in the path chooses which way the palm faces, and
  a grab needs the palm on the object.

What to try next, cheapest first: re-run the same grab when the residual is under 30 mm (the
body has to be closer, or the target chosen from a run where it is inside 0.40 m), and if that
still fails, sweep `--yaw` for palm facing at the moment of closing.

### Retry: three attempts, best 49 mm

Same card, same pixel, fresh rest before each, tolerance 20 mm:

```
attempt 1   error 69.2 mm
attempt 2   error 49.0 mm
attempt 3   error 73.7 mm
```

None reached the 40 mm gate that would have triggered a grab. Earlier today the *same* card
converged to **21.4, 24.5, 27.9 and 32.9 mm** — so this is not a fixed limit, it is the body
being further away on this attempt. The difference between those runs and these is the root:
the successful ones had drifted 15-41 mm of lean into the reach, these had 61 mm and no lean.

That is the same conclusion the offline reachability probe reached from the other direction:
**~0.30 m of forward body displacement makes the card exactly reachable.** Without a controlled
step, whether the reach lands inside the grab radius is decided by where the policy happens to
have wandered, not by anything we command.

So the grab remains unproven for a reason that is now well characterised: not the loop, not the
aiming, not the detector - the body's distance, which nothing controls.
