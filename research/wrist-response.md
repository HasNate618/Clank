# How the wrist responds to a step command

Every servo loop in this project used `time.sleep(1.6)` between commands, chosen to be
obviously longer than anything the body could take. That is now measured, and it was the
wrong way round: the arm is fast and the *axes are not alike*, which is what actually
mattered.

Method: `spike/wbc_game/step_response.py`. Head pinned with `look_yaw_deg=0.0` (see
`fix(body): a reach must not move the camera`), origin established by a zero-offset
`reach`, then one step on one axis, sampled from `body_state` at 20 ms. One trial per
point — not repeated, and gains are posture-dependent, so read these as magnitudes, not
constants.

## Measured 2026-11-02

| axis | command | final | gain | peak | rise to 63 % | settle (5 %) |
|---|---|---|---|---|---|---|
| forward | +0.250 | +0.274 | **1.10** | +0.301 (+9.9 %) | 0.10 s | **1.05 s** |
| up | +0.200 | +0.200 | **1.00** | +0.350 (**+75 %**) | 0.08 s | 0.77 s |
| lateral inward (+y) | +0.200 | +0.030 | **0.15** | +0.135 transient | 0.06 s | 3.28 s |
| lateral outward (−y) | −0.200 | −0.043 | **0.22** | — | 0.00 s | — |
| lateral outward (−y) | −0.100 | −0.032 | **0.32** | — | 0.00 s | — |

## What this says

1. **The arm is fast and the loop was slow.** 63 % of the motion arrives in ~0.1 s and the
   ring settles in 0.8–1.1 s. The 1.6 s sleep was about 50 % waste; **1.0 s is enough**,
   and a 6-step loop costs ~6 s instead of ~10 s.
2. **Commands land long on the forward axis**: 0.25 asked, 0.274 delivered. A single
   open-loop reach therefore misses by ~10 % of the requested distance *before* any
   posture error, which is a systematic, correctable component of the 0.10–0.20 m residual
   this project has been calling "arm error".
3. **The up axis overshoots 75 %** transiently (0.35 for 0.20) before settling exactly on
   target. Large up-steps therefore swing into the workspace clamp and, on the evidence of
   the lever attempts, into a body lean.
4. **Lateral is a fifth to a third as effective as forward, in both directions.** It tracks
   the sign and then largely springs back. This is the important one: the 3D corrector was
   issuing lateral fixes the arm would not deliver, and large lateral requests made the body
   lean, moving the wrist the *other* way. That is the mechanism behind the 0.46 m sideways
   swing observed on a command that only asked for "up".

## Consequences for placement

- **Pre-scale per axis** by the inverse gain (forward ÷1.10, up ÷1.00, lateral ÷~0.25).
  With that, one command plus one measured correction should land inside the ~0.05 m that
  matters, instead of the 5–6 step hunt.
- **Do not buy lateral motion with wrist offsets.** At a gain of ~0.25 the command needed is
  4× the desired move, which is exactly what provokes the lean. Lateral placement should
  come from **body yaw**, not from the wrist channel.
- **Expect the up-axis transient** when the target is near the clamp.

## Reproduce

```
spike/e3/.venv/bin/python spike/wbc_game/step_response.py fwd 4.0 0.20
spike/e3/.venv/bin/python spike/wbc_game/step_response.py up  4.0 0.20
spike/e3/.venv/bin/python spike/wbc_game/step_response.py lat 4.0 -0.20
```

## Related

`research/sonic-reach-envelope.md` measured *how far* the wrist goes. This measures *how
fast and how faithfully* it gets there, which is a different question and the one every
correction loop depends on.

## Placement with the measured response

`spike/wbc_game/place_wrist.py`: one probe command measures the gain **in that posture**,
then each step corrects `0.7 * error / gain`, capped at a gain of 1.6 and a lateral
request of 0.30 m. Settle 1.05 s, matching the measurement above.

Target 0.308 m from rest, aimed past the tucked-arm gain entirely:

| step | wrist | error |
|---|---|---|
| 1 (probe) | (+0.373, +0.009, −0.080) | 0.164 m |
| 2 | (+0.391, −0.104, −0.084) | 0.051 m |
| 3 | (+0.396, −0.125, −0.075) | **0.031 m** |

**3 commands, 3.2 s, 0.031 m.** The previous corrector used 6 commands, ~10 s and left
0.05–0.19 m. Forward and up are inside 0.01 m after the *first* command; every remaining
step is lateral.

### The gain is not a constant

Measured across postures in one session: **0.15–0.32** for a tucked arm, **~1.3** for an
extended arm, and **2.73** on a single probe with a small lateral command. A fixed matrix
built from the tucked-arm figure made the extended-arm loop diverge 5× (a +0.300 lateral
command produced +0.399 of motion). Hence: measure in situ, cap at 1.6, damp at 0.7.

### What this means for the demo

2–3 commands at ~1 s each is a *deliberate* motion, not a stutter, if each one is a visible
progress step rather than a hidden correction. The remaining work for smoothness is to
stream a path at a rate below the 0.4 s oscillation, not to shorten the loop further.

## The fall (2026-11-03)

The body fell during these measurements and the executor stopped itself:
`body_view: FALL detected, stopping`, exit code 3. The socket then refused connections,
which is how it was noticed - the dashboard still reported `running`, so a dead body and a
live one are indistinguishable from the state endpoint alone.

Cause, from the same session's numbers: a raw Jacobian inverse, asked for a modest
(0.20, 0.10, 0.12) move, produced the command

```
(+0.798, -0.761, +0.124) m of offset
```

That is nearly a metre, of which 0.76 m is **lateral** - the axis measured to be coupled
~45° into forward (Jacobian column `[+0.50, +0.71, +0.10]`) and the one the earlier
observations already blamed for the 0.46 m sideways swing and the body lean. The linear
model was fitted at a 0.15 m probe and extrapolated 5× beyond it; on this model the
off-diagonal terms are large enough that `J⁻¹` amplifies a small request into an
unreachable one.

The guard that should have caught it did not, because it lived in one tool and not the
other. It now lives in `clank.workspace.clamp_offset`: lateral capped at **0.18 m** on its
own axis first, then the whole vector scaled to **0.40 m** of magnitude, preserving
direction. `spike/wbc_game/jacobian.py` now *refuses* to send an offset the clamp would
change rather than silently sending a different command than it printed.

Two lessons worth keeping:

1. **`J⁻¹` is not a placement method here.** With off-diagonal terms this large, invert
   nothing: iterate a damped correction against measurement (`place_wrist.py`: 3 commands,
   0.031 m) and let the clamp bound each step.
2. **A "fine" lateral number is not a fine lateral command.** The per-axis table looks
   harmless because the numbers are small; the failure mode is the body, not the wrist.

## `look` moves the head, not the body - and the arm drifts when uncommanded

Two things that were assumed and are not true (measured 2026-11-03):

**1. Body yaw is not available through `look`.** A `look yaw +35` changed the root yaw by
**−0.1°** while the head clearly turned. So "aim laterally by turning the body" - which was
the natural next move given the ~45° coupling between forward and lateral - is not
reachable with the current verbs. `AUTH_WAIST` is in `UNSUPPORTED_IN_V1`, and
`AUTH_ROOT_VEL` is not honoured in a hold session, so the torso cannot be aimed either.
Lateral placement has to come from wrist offsets, iterated, at a gain of ~0.25.

**2. A body with no active command drifts.** After a fresh restart and no hand command at
all, the wrist relaxed 0.325 m over a few seconds toward the clip pose. Once a `reach` has
been issued the steady rate is much smaller, but it is not zero:

| keepalive | drift over 8 s | worst excursion |
|---|---|---|
| none | 0.039 m (4.9 mm/s) | 0.040 m |
| every 1.5 s | 0.029 m | 0.043 m |
| every 0.5 s | **0.009 m (1.1 mm/s)** | 0.026 m |

So re-sending the current target at ~2 Hz cuts drift about 4×. This is why a mark goes
stale while a frame is captured and thought about: the arm is not where the last command
put it, and nothing says so. Any loop that pauses between measurement and command should
keepalive - and the first command of a session must be sent before anything is measured,
because the arm is still relaxing toward the clip.

## Straight lines are not achievable inside the safe envelope

Three attempts at the same 0.300 m forward path, same start posture:

| method | worst cross-track | endpoint error |
|---|---|---|
| open-loop streaming | 0.162 m | 0.162 m |
| diagonal feedback (`move_path.py --feedback`) | 0.167 m | 0.162 m |
| damped `J⁻¹` feedback (+ clamp) | **0.286 m** | 0.140 m |

The mechanism is arithmetic, not tuning. Commanding 0.30 m forward requires a forward
offset near 0.39 m, and the measured Jacobian says that also produces ~0.67 of lateral
motion per unit. Cancelling it needs a lateral offset of

```
-0.67 * 0.388 / 0.71 ≈ -0.366 m
```

against a safety cap of **0.18 m** — the cap being 0.18 precisely because a 0.76 m lateral
request knocked the body over. So a straight forward line and a survivable command set are
incompatible on this arm. `J⁻¹` did worst of the three because it spent its authority trying.

This is not a bug to be fixed by more cleverness; it is a property of the arm's coupling.
The forward diagonal *is* the arm's natural direction.

**Consequence:** stop asking for straight lines. Ask for the endpoint (which works:
3 commands, 0.031 m) and let the path be whatever the arm's diagonal preference makes it.
A demo that reaches the object by the natural route looks better than one fighting its own
kinematics, and it is the same motion a person would make.
