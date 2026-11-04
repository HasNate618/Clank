# Why the arm jerked, and what a smooth path measures

**STATUS: MEASURED, 2026-11-04.** Operator report: "~10 s of weird arm jerking to get there
instead of a smooth fast linear motion". Reproduced and diagnosed from the run logs.

## Cause, in order of size

1. **The loop re-aims during the settle.** `reach_vr` commanded, waited `--dwell 1.2 s`, then
   commanded again, for `--corrections 4` rounds - five commands, ~10 s. Each command is a step
   the body answers with 0.77-1.05 s of settling and **+75 % vertical overshoot**, so every
   command landed while the arm was still answering the previous one.
2. **No anti-windup, on a goal outside the envelope.** `target += 0.6 * error` integrates. When
   the arm cannot comply the error does not shrink, so the *commanded* point walks outward:
   `y = -0.647 -> -0.670 -> -0.716 -> -0.868 -> -0.922 m`. The lateral clamp is 0.18 m; the
   command went to 0.92 m. Nothing clamps, because `reach_vr` only *prints* "past the 0.40 m arm
   envelope: the legs will answer".
3. **The residual got worse as often as better:** 92.5 -> 109.6 -> 316.6 -> 127.8 -> 164.7 mm.
   Every fall is a direction reversal. That is the jerking.
4. **The axes are cross-coupled.** Measured Jacobian forward/lateral ≈ (0.72, 0.67): a correction
   in y drags x, and the next round drags it back.

## Fix

A straight line from the hand to the goal, **streamed** as waypoints every 0.1 s in world space
(the root moves), each waypoint clamped into `WRIST_BOX`, then at most two damped corrections,
with **best-so-far** and a stop on the first increase.

## Measured, same goal, same reset, three runs each

Free-air goal (0.35, -0.35, 0.20) in the root frame; "reversals" counts direction changes of the
dominant component of the sampled hand path (10 Hz), which is the jerk the operator saw.

| method | time | residual | reversals |
| --- | --- | --- | --- |
| smooth path only | 1.61 s | 64-93 mm | **2** |
| smooth + 2 corrections | 2.62 s | 34 mm | **4** |
| old step-and-correct | 2.51 s | 21-38 mm | **9-11** |

**Same accuracy and the same time, roughly half the direction reversals.** The smooth path alone
is the fastest and smoothest but stops short; the two corrections after it are what buys the
accuracy, and they cost two reversals rather than seven.

Run-to-run spread is real (the same method gave 64 mm and 93 mm on two runs), so treat the
residual column as ±30 mm and the reversal column as the stable part of the result.

## The other report: "the arm went in the opposite direction"

That was the AUTO MECHANIC attempt, and the arm was obeying orders. `locate` returned
**(367, 326)** - bottom *left* of the frame - which unprojects to `y = +0.387`, and positive y is
the robot's left. The tray is bottom-right at y ~ -0.5. The goal was 0.83 m away, past the
envelope, and the reach went left and diverged.

The cause is upstream: stage-3 verification had been switched off minutes earlier, because it had
**refused a pixel that was dead centre on the GOURMET CHEF label**. So verification was rejecting
correct answers and admitting a wrong one. A verifier that is wrong in both directions is not a
safety net, and the honest fix is not "turn it off" - it is the tight-crop size, so that a crop of
a label reads as a label.
