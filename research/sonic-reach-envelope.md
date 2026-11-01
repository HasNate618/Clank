# SONIC standing reach envelope — is `FLAG_SATURATED` a reach signal?

- Date: 2026-11-01
- Question: can the simulated G1 put its right wrist where JobBot's floating card
  sits, and can `FLAG_SATURATED` tell us when it cannot?
- Probe: `spike/wbc_game/reach_envelope_sonic.py`, run under
  `spike/e3/.venv/bin/python`. Standing clip `squat_001__A359` frame 0, 60 ticks
  settle (1.2 s) then 100 ticks reach (2.0 s) per target, 30 ticks hold between.
  `hand_target("right", forward_m=...)` is an offset from the clip wrist, +X
  forward in the root frame.
- Status: `[MEASURED]`, 2026-11-01, one run, one seed, no repeat. Not yet a
  soak.

## 1. Where the wrist gets to

Settled right wrist at the standing clip: `(0.079, -0.308, -0.203)` m, root
frame.

| `forward_m` | achieved x | achieved y | achieved z | residual |
| ---: | ---: | ---: | ---: | ---: |
| 0.15 | 0.211 | -0.144 | -0.219 | 0.054 |
| 0.20 | 0.273 | -0.125 | -0.145 | 0.120 |
| 0.30 | 0.383 | -0.149 | -0.086 | 0.169 |
| 0.40 | 0.450 | -0.223 | -0.163 | 0.104 |
| 0.50 | 0.587 | -0.216 | -0.071 | 0.184 |
| 0.60 | 0.707 | -0.046 | 0.106 | 0.383 |
| 0.70 | 0.701 | -0.004 | 0.005 | 0.322 |

The wrist tracks forward monotonically to about **0.70 m forward of the root**,
then plateaus. That is *better* than the classical-QP figure carried in
`docs/2026-10-23-clank-design.md` §8 ("a probe at 0.70 m forward leaves the wrist
19.7–27.3 cm short") — that number is the mink QP, not SONIC, and must not be
applied to this controller.

**No fall** at any target (`ex.fell` false throughout).

**Caveat on the plateau rows.** 0.60 and 0.70 also lift and centre the wrist
(`z` +0.106, `y` −0.046) — the arm is straightening and swinging toward the body
midline rather than pressing forward. Those two rows are a different shape of
motion, not a longer reach, and the residual is 0.32–0.38 m. Do not quote "0.70 m
reach" as a positioned wrist; quote "the wrist translates to x ≈ 0.70 m and stops
tracking the target there".

## 2. `FLAG_SATURATED` is always on at the contract default

The flag is raised when the residual exceeds the *command's* `tolerance_m`,
default `0.02` m. Residuals measured here range 0.054–0.383 m, so:

| `forward_m` | residual | saturated @ 0.02 | @ 0.10 | @ 0.20 |
| ---: | ---: | :---: | :---: | :---: |
| 0.15 | 0.054 | yes | no | no |
| 0.20 | 0.120 | yes | yes | no |
| 0.30 | 0.169 | yes | yes | no |
| 0.40 | 0.104 | yes | yes | no |
| 0.50 | 0.184 | yes | yes | no |
| 0.60 | 0.383 | yes | yes | yes |
| 0.70 | 0.322 | yes | yes | yes |

**At the default 0.02 m tolerance the flag is set on every single target,
including one the body tracked well.** This is not a bug in the flag; it is the
tolerance being tighter than the controller. Measured aim error for this stack
is 3.51 cm / 7.22°, so 2 cm is below the noise floor of ordinary tracking.

Why this matters more than it looks: `clank/brain.py::outcome_from_flags` now
feeds this flag to the planner (P2b). Wired naively, the model is told
**"saturated" on every reach**, which per §11 means "change the action" — so the
brain would abandon every reach, including good ones. Nobody read this flag
before; this is the likely reason.

Residuals are also **not monotonic** in target distance (0.30 → 0.169, then
0.40 → 0.104). A single residual threshold is therefore a poor reach criterion
even at a sensible tolerance.

## 3. What this means for the milestone

- The card at arm's length is **probably reachable**: the wrist gets to 0.45–0.59 m
  forward with 0.10–0.18 m residual, and to ~0.70 m without falling.
- The **depth** of the card is still unknown. Everything above is a forward offset
  from the standing clip in the root frame, not a world position of an object in
  Job Simulator. `P2d` (envelope against the card) still needs the game.
- `P2d` is **also blocked on projection**: `clank/projection.py` has
  `project_point` (3D→pixel) and no pixel→wrist inverse, and the `reach` verb takes
  explicit offsets, not a pixel. A clicked pixel cannot be compiled to a reach
  today.

## 4. Recommended follow-up

1. Choose the saturation tolerance from a measurement, not the contract default.
   Candidates: per-skill tolerance, or a criterion that is not a residual at all
   (the game's own cue — Job Simulator lights the object blue when it is in grab
   range, which is a *better* reach oracle than tracking error).
2. Until then, **do not wire `FLAG_SATURATED` to a retry rule.** `P2b` composes
   the words correctly; the word is currently uninformative.
3. Re-run this probe with N ≥ 3 repeats and a longer hold, to see whether the
   0.45–0.59 m band is stable or seed-dependent.

## 5. Negative and unmeasured

- One run, one seed. No repeat, no soak, no confidence interval.
- No repeat with the left arm.
- No measurement of how far *laterally* the wrist can go at a reachable depth.
- No measurement of wrist **orientation** at those depths (the reference set
  quotes 17–32° error on uncommanded axes).
- The 0.02 m default was not changed; changing a contract default based on one
  run would be the same mistake in the other direction.
