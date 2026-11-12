# The reach architecture, critiqued

- Date: 2026-11-11
- Subject: the ARDY reach design (`docs/superpowers/specs/2026-11-11-ardy-reach-design.md`)
- Method: three independent read-only critiques from outside the effort — a robotics/control lens, an
  ML/motion-generation lens, and a product/scope lens — asked for fragile parts, alternatives, and the
  experiment that would decide between them. Their findings are recorded here before any of them is acted on.

Tagging is strict, because two of these findings are bugs and the rest are opinions:

- **[CODE]** verified by reading the implementation in this repo. Reproducible.
- **[MEASURED]** a number from a committed test, probe, or research note; the source is named.
- **[SPECULATION]** a plausible mechanism nobody has measured.

## 1. The graded point is wrong by 0.18 m [CODE]

The hand constraint pins a *joint anchor*; the game and the clasp both look at the *relayed hand*. They are not
the same point.

- `ardy_gen.hand_position` returns `posed_joints[..., bone_index["right_wrist_yaw_skel"]]`
  (`spike/wbc/ardy_gen.py:424-439`), and `hand_target` builds the pinned point the same way (`:392-397`). The
  constraint displaces `skeleton.right_hand_joint_names` (`:379-380`), whose first member is that wrist joint.
- The relayed hand is that link **plus** `VR_OFFSETS[:2] = [0.18, ∓0.025, 0.0]` (`spike/wbc/sonic_runner.py:114-116`,
  applied at `:176-179`). The repo already says why this matters: "The point is the **VR point** … Commanding the
  physical link there would put it 0.18 m away" (`spike/wbc/sonic_executor.py:665-674`).

Two consequences, both live:

1. **Gate 5's 66 mm is wrist-to-wrist.** `probe_ardy_track` takes the lever back off on both sides
   (`spike/wbc_game/probe_ardy_track.py:340-343`), which is correct for a wrist-to-wrist comparison — and it means
   nothing in the repo has ever measured where the game's glove sits.
2. **`--grab` would never clasp.** `reach_ardy` pins the wrist at the object (`:112-121`) but tests the *levered*
   hand against that same object point (`:248-251`) before comparing to `CLASP_START_M = 0.08`
   (`spike/wbc/end_pose.py:30`). With the glove on the object the comparison reads ≈0.18 m, so the clasp gate
   cannot fire, while the printed `hand_residual` (wrist-based, `:124-132`) reports success.

**Fix:** build the delta on the VR point — `object − (wrist + R_wrist @ VR_OFFSETS[side])` — and make the residual,
the printed number, and the clasp all measure that same point. Report both wrist and glove residuals in the probe
so the lever can never hide again. ~10 lines.

## 2. The 80 mm threshold is uncalibrated, and it is 94 mm in the room [MEASURED + CODE]

`CLASP_START_M = TRACK_SUCCESS_M = 0.08` is a **guess**, pinned by a test
(`tests/test_reach_ardy.py::test_the_clasp_threshold_is_the_policy_constant`) and then treated as a requirement.
Nothing in the repo measures what Job Simulator accepts; `research/reach-placement.md:248` records the same gap.

Worse, the residual is measured in body space and the game grabs in game space. At `--player-scale 1.43` with hands
scaled (`clank/player_scale.py`, `research/g1-scale.md:87`), **66 mm of body-space error is ~94 mm of room-space
error** — past the gate the whole effort is graded against.

**Experiment (half a day):** hold the body at 2 / 5 / 8 / 12 cm from an object, squeeze, record whether the game
registers. That table converts an arbitrary constant into a requirement, and it may retroactively make 66 mm fine.

## 3. The baseline comparison is not like-for-like, and path shape is unmeasured [MEASURED]

| | residual | wall time | reversals |
| --- | --- | --- | --- |
| cruise, smooth path only (`research/motion.md:38-40`) | 64–93 mm | 1.61 s | 2 |
| cruise, smooth + 2 corrections (`:38-40`) | **34 mm** | 2.62 s | 4 |
| cruise, uncorrected sweep (`:89-91`) — the row this session kept quoting | — | 3.1–7.5 s | 13–27 |
| ARDY + SONIC mode 0 (gate 5) | **66 mm median** (52.5–92.2) | ~2.3 s | **not counted** |

The new pipeline is *inside the noise* of the thing it replaced on accuracy, slower to converge than the cruise's
best case, and better on exactly the property nobody measured. §10 of the design claims the win — "13–27 direction
changes → one generated window" — as an assertion; no reversal count exists for a generated path.

**Experiment:** add wrist path-length ÷ net, dominant-axis reversal count at 10 Hz, and torso pitch/roll excursion to
`probe_ardy_track.analyse`, then run it on the six stored clips and three `reach_vr` runs on the same targets.

## 4. What the architecture actually buys [CODE + MEASURED]

- Generation is **not** the latency problem: 0.07–0.20 s for 2.08 s of motion is 10–30× real time
  (`research/motion-generators.md` §9).
- v1 refuses `history_features` (`spike/wbc/ardy_gen.py:467-474`) and places a canonical stand at the body's root and
  heading. So a generated clip is a **pure function of (root, heading, relative target)** — live generation is an
  offline authoring step being run at the wrong moment, and every grab pays its sampling tail.
- The generator's adherence is 45.5 mm median with a 45–80 mm spread [MEASURED]. That spread has never been
  separated into sampling noise and bias, because the probe re-derives the target from *each run's* achieved hand
  (`spike/wbc_game/probe_ardy_track.py:461-475`). Until it is, "best-of-N will help" is arithmetic on an assumed
  distribution.
- The claim that clip frame 0 is within 1 mm / 0.8° covers the **root**, not the 29 hinges
  (`ardy_gen.frame0_delta`, `:193`). A second grab therefore starts with a step in the whole pose that the tracker
  must absorb — and no gate tests a second grab.

## 5. Recommended decomposition: generate → track → servo → clasp

The split between "a prior proposes shape" and "the policy owns physics" is right and should be kept — every attempt
to hand the 50 Hz process its own Cartesian or joint reference is measured negative
(`research/sonic-gate-results.md` §4: hard IK into SONIC 1.14 → 14.40 cm; 17-joint override delivering 33 mm of
235 mm; a classical QP that cannot walk).

But the boundary is drawn one stage too late. The design gives the generator responsibility for **final position** —
the one property a motion prior measures worst (45–80 mm) and the one the existing mode-1 servo measures well
(16.4 mm in-box; 3.51 cm at a fixed aim point; 37 mm end to end after one correction, `research/motion.md`).

```
ARDY generate          gross human shape, feet planted, waist reasonable      ~45-80 mm
  -> SONIC mode 0      physics and balance, no reset()                        ~66 mm
  -> VR servo closure  the last 8 cm, in the pelvis frame, recomputed/tick     ~16-35 mm  [SPECULATION for the composite]
  -> clasp             on the GLOVE point (§1)
```

**[SPECULATION]** composite ≤ 35 mm. The two error sources are orthogonal and the servo's 0.77–1.05 s settle is a
large-step property that should not bind on an ≤8 cm move. The discriminating experiment: 6 game-off runs —
generate, play, 0.6 s servo to the same point — measuring glove residual, falls, and whether the handover adds a
reversal.

## 6. Deployment: author offline, screen, then play [CODE]

Because §4's clip is a pure function of three quantities, a lattice of ~60 clips (bearing × distance × height,
52×36 float32 ≈ 7.5 kB each) can be generated offline, screened with `probe_ardy_track`, and the winners kept. That
turns the sampling tail into a one-off cost instead of a per-grab coin flip, and removes torch, 0.94 GB, the cold
start, and `LD_LIBRARY_PATH` from the live session entirely. It is a change behind `generate()`, not a rewrite.

It is also **the only remaining route to the look**. The four live attempts are measured dead (design §8.5): facing
the object with a constraint costs 25–110 mm of hand adherence because facing moves the object 0.35 → 0.53 m from the
right shoulder while the arm is 0.55 m. Authored offline, the same clip can be authored *from a stance that has
already turned and stepped closer*, where facing is free — and a clip that fails can be thrown away without a live
session. The design's "live, not offline" preference is a goal, not a requirement; nothing in the game's cadence
needs a fresh window per grab.

This supersedes the §14 fallback: ARDY as the author, not Kimodo. No new licence, no new tool.

## 7. What to cut [CODE]

- **The escalation policy.** Its entire measured history is one −0.03 m no-op and a rule written to stop it firing.
  It also has a real defect: `reach_ardy` commands `root_forward = placed[0]` and drops the turn (`:444-446`), and
  `step_root` places the root *behind* the object, so the forward component is **negative** whenever the object is
  nearer than `|HAND_PREF_xy| = 0.4482 m` — the body backs away while the hand stays pinned. `step_too_far` guards
  the 2-D placement while only the forward component is commanded. For v1: `if residual > tol: report and stop`.
  Keep `yaw_error` as a printed diagnostic.
- `end_pose.solve_planted` / `solve_stepped` — `NotImplementedError` stubs for a task that was replaced.
- `MAX_STEP_GEN`, `STEP_CHAIN_S`, `PLANTED_RESIDUAL_M` — dead once the chain is one-shot.
- `g1_pack.context4_from_ring` and the 32-pose ring: ARDY v1 reads `context[-1]` only, and the ring is a 32×36 JSON
  write plus `os.replace` every 20 ms in the one process that must not miss ticks (`sonic_executor.py:503-522`).
  Shrink to one pose.
- MotionBricks/Kimodo as *built* paths. §14's documentation stays.

## 8. Risks, ranked by probability × cost

| # | Risk | Evidence |
| --- | --- | --- |
| R1 | **Gate 6 has never run.** Bundles the pixel path (unexercised — gate 5 is `--world`), the game's grab at a generated pose, GPU co-residency, 0.89× realtime. | Task 7's Critical (`session_dir` never passed, so every new branch was dead in the only body process) was found by review, not by a test |
| R2 | **The 80 mm threshold is an uncalibrated proxy**, and ×1.43 makes 66 mm into 94 mm | §2 |
| R3 | **Out-of-reach escalation never proven**; generator 0.566 m of 0.60 m, tracked never | ledger, task-8 report |
| R4 | **The look is dead, not at risk** — 35–59°; best 15° at +32 mm | design §8.5 |
| R5 | **GPU contention** — every measurement had the GPU to itself | plan Task 9 Step 1 exists, undone |
| R6 | **50 Hz tick budget** — a 32×36 JSON ring per tick into a body at 0.89× realtime | §7 |
| R7 | **Left hand missing** — `--side left` exits 3, while the old path could reach either side | `reach_ardy.py:326-333` |
| R8 | **Wrist orientation is pinned as an equality**, contrary to §7.2's prose, so 3 DoF the prior would spend on looking human are fixed | `ardy_gen.hand_constraint:370-380` |
| R9 | Process split | **low** — best-tested part of the stack, two fix rounds |

## 9. Order of work

1. **Fix §1** (the lever). Without it gate 6 measures nothing.
2. **Measure the game's grab tolerance** (§2). Half a day; redefines "accurate enough".
3. **Measure path shape** (§3) on existing clips plus cruise controls.
4. **Separate noise from bias** in the generator spread: freeze one pose36, ten identical generations, report
   mean ± σ and the mean signed error. Decides whether best-of-N or a fixed offset is the instrument.
5. **Servo closure** (§5) behind a flag, then its discriminator.
6. **Gate 6**, with the cruise re-run first as a control so the swap is a comparison rather than a cliff.
7. **Offline library** (§6) — deployment and the look.
8. Defer: left hand (R7), text conditioning (never shown to matter), distillation, receding-horizon replanning.

## 10. Where the critics disagreed, and what I concluded

- *Is the generator the right instrument?* The ML critique says yes with two gaps (history, sampling). The control
  critique says it is being asked for the one job it does worst. Both are right: keep it, and stop asking it for the
  last 8 cm.
- *Live vs authored.* The design treats live generation as a goal (§3.3). Two critics independently showed it is an
  implementation detail given a history-free clip, and that authoring offline removes the entire sampling tail.
  Concluded: live generation stays available for arbitrary pixel targets; the demo should not depend on it.
- *Correction to this session's own claims.* "The body has never grabbed anything live" was wrong —
  `research/reach-placement.md:223` records a live Job Simulator grab on 2026-11-04: card marked, reached, closed,
  lifted, carried 0.29 m, released. It was the *cruise* path. And the "torso twists 100° the wrong way" figure was a
  frame mix-up (world-frame torso yaw against a body-relative bearing); the signed measurement is 0.2–24°
  (design §8.5).
