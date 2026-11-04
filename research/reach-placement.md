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
