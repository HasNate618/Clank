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
