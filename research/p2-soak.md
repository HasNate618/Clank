# P2 soak — freeze-hold does not hold, because the latch does not latch

- Date: 2026-11-01
- Question: `hold` is supposed to latch the last achieved poses and stand still
  while the model thinks. Does it?
- Probe: `spike/wbc_game/hold_soak.py`, real MuJoCo + SONIC, `squat_001__A359`
  frame 0, 1.2 s settle on `rest()` then 20 s measured at 50 Hz.
- Status: `[MEASURED]`, 2026-11-01. `hold()` numbers reproduced exactly three
  times (0.444 m / 0.041321 m/s each run).

## 1. Headline

| command | net 20 s | RMS last 10 s | p95 | fell |
| --- | ---: | ---: | ---: | :---: |
| `hold()` — freeze at last achieved | 0.444 m | **0.0413 m/s** | 0.1017 | no |
| `hold(state=snapshot)` — latched **once** | 0.0226 m | **0.000069 m/s** | 0.000093 | no |
| `rest()` — clip anchor | 0.0166 m | 0.0042 m/s | 0.0064 | no |

**FIXED the same day** — see §7. `clank/bus.py` now latches the snapshot once. Re-run end-to-end
through `bus.take()` + the executor: **0.000069 m/s, net 0.0226 m over 20 s**, on both the idle
path and the expired-TTL path a finished reach takes. The figures below are the pre-fix
measurement that motivated it, kept because they are the evidence.

`hold()` drifts 0.44 m over twenty seconds. The same command with the pose
latched once drifts 2.3 cm. **The latch is not a latch: it re-reads the live
achieved pose every tick**, so the target follows the body. There is no restoring
force, and any perturbation becomes the new target — a random walk at 4 cm/s.

That is the opposite of the documented intent. `docs/superpowers/plans/2026-11-01-freeze-hold.md`
says "freeze-hold latches last poses"; `clank/body.py::channel_pose` rule 3
returns `last`, and `clank/bus.py` passes `self._state`, which `publish_state()`
overwrites every tick.

## 2. It is not the ramp, and not the targets

Cross-checked against the recorded C3m recipe (`runner.run(motion, freeze=frame)`),
which holds the same clip and frame:

| driver | net 20 s | RMS last 10 s |
| --- | ---: | ---: |
| `runner.run(freeze=0)` | 0.108 m | 0.0034 m/s |
| `runner.run(freeze=0, vr_pos=…, vr_orn=…)` | 0.108 m | 0.0034 m/s |
| `runner.run(freeze=0, ramp=0)` | 0.121 m | 0.0040 m/s |
| `runner.run(freeze=0, ramp=0, vr_pos=…)` | 0.121 m | 0.0040 m/s |
| `SonicExecutor.step(hold())` | 0.444 m | 0.0413 m/s |

Explicit VR targets and the startup ramp each change nothing, so neither is the
difference. The only knob left is what `hold()` commands, and §1 shows it is.

Note also that `rest()` — the command introduced as the *less* desirable one —
holds 10× better than `hold()`. That inverts the relationship the plan assumed.

## 3. Consequence for the milestone

The claim needs the body to stand still while a 0.2–0.6 s PLAN call runs. As
implemented, `hold` moves the wrist ~1–2 cm during a typical call and ~0.44 m
across a task. Job Simulator has no state API, so the brain would be re-grounding
against a stale pixel for a hand that has since wandered.

`hold(state=snapshot)` fixes it and keeps the reached pose — no snap-back to the
clip, which is what `hold` was introduced for. 6.9×10⁻⁵ m/s is 60× better than
`rest()` and 370× better than the trailing hold.

## 4. Recommended fix

Latch at hold **start**, not per tick. Options, cheapest first:

1. `clank/bus.py`: when a freeze-hold is synthesised (idle or TTL branch), snapshot
   `self._state` once into `self._latched` and pass that on every subsequent
   tick; clear it when a command with real poses arrives.
2. `clank/body.py`: give `channel_pose` an explicit latched-state argument so the
   latching is a property of the contract rather than of one caller.

Option 1 is the smaller change and matches where the state already lives. Either
way the unit test is the same: two consecutive `take()` calls with different
`publish_state()` values must yield the **same** pose for an uncommanded channel.

## 5. Negative and unmeasured

- One clip, frame 0, right wrist only. No left arm, no other anchor frames.
- No soak *through the bus* — these calls bypass `bus.take()` and drive the
  executor directly. The bus's TTL/idle paths are unit-tested but not measured
  here.
- Not measured: whether the drift is monotone (a slow slide in one direction) or a
  random walk. The 1 s samples in the probe suggest monotone creep, but that is
  not established.
- The 0.0226 m net for the latched hold is ~1 cm/s over the first second before
  it settles; the steady-state figure is the 6.9×10⁻⁵ m/s.

## 7. Fix, and the re-run

`clank/bus.py` latches the state once (`_latched`) instead of handing the executor a
fresh snapshot per tick. The latch is taken on the first `take()` that needs it,
cleared when a command arrives that is not a freeze-hold (`is_freeze_hold`), and
baked into an explicit `hold` verb so the executor never reaches `channel_pose`
rule 3.

| path | net 20 s | RMS last 10 s | p95 |
| --- | ---: | ---: | ---: |
| bus idle path | 0.0226 m | **0.000069 m/s** | 0.000093 |
| bus, after a finished reach (expired TTL) | 0.0226 m | **0.000069 m/s** | 0.000093 |

Both match the hand-latched control exactly, so the bus is no longer the
difference. 370× better than the pre-fix 0.0413 m/s, and 60× better than `rest()`.

Unit-level, 7 new tests in `tests/test_bus.py::HoldLatchesOnce`. Mutation-checked:
restoring the trailing read fails 4, not clearing on a new command fails 2, and not
baking an explicit hold fails 1.

Still not measured: a left-arm soak, other anchor frames, or a soak with the game
attached. The latch is correct in the sim and that is all this claims.
