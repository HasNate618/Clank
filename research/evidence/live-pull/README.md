# A live pull, 2026-11-13: the wiring runs, the hand does not move

Session `20261113T223258Z-00f8ba`, game running, body up (`frame_age_s` 0.08), depth live
(seq 402). Goal: exercise `--move` from `docs/superpowers/plans/2026-11-13-directional-motion.md`
on the real body, without waiting for JobBot to reach the lever.

## What ran

```bash
# 1. from the resting hand
spike/ardy/.venv/bin/python spike/wbc_game/reach_ardy.py --move -0.25 0.0 0.0 --side right
# 2. hand moved in front with the cruise
spike/e3/.venv/bin/python spike/wbc_game/reach_vr.py --target 0.45 -0.15 0.0 --no-rest --no-clamp
# 3. the pull, twice
reach_ardy.py --move -0.25 0.0 0.0
reach_ardy.py --move -0.20 0.0 0.0
```

## What it reported

| run | target pelvis (m) | generated residual | "achieved" residual | hand before | hand after 4 s |
| --- | --- | --- | --- | --- | --- |
| 1 | x clamped to 0.079 (box floor) | 14.2 mm | 16.1 mm | (0.168, −0.271, −0.199) | (0.143, −0.293, −0.202) |
| 2 | (0.180, −0.168, 0.004) | 6.9 mm | 67.8 mm | (0.430, −0.168, 0.004) | (0.410, −0.199, −0.014) |
| 3 | (0.213, −0.230, −0.007) | 22.8 mm | 25.7 mm | (0.413, −0.230, −0.007) | (0.415, −0.214, −0.012) |

**The plumbing works.** `--move` reads the hand's own reported point, `clank.workspace.moved_point`
clamps it, `reach_vr.to_world` converts it, the clip generates in 0.77-1.21 s at 0.94 GB, deposits,
and plays to `done`.

**The pull does not.** In run 3 the commanded travel was 0.200 m backward and the relayed hand
moved 0.002 m; a 4 s settle changed nothing. The game draws `body_state.json`'s hand, so as built
this is a pull the game cannot see.

Run 1 also confirmed Review Focus 2 of the plan: a 0.25 m pull from the *resting* hand is capped by
`WRIST_BOX` x floor 0.079, so the real travel was 0.089 m. From a hand 0.43 m forward there is room,
and the cap does not bite - run 3's target is unclamped.

## Why the residual and the hand disagree, and what is actually wrong

`reach_ardy` measures "achieved" through `ardy_gen.glove_position()` on `pose36.json`'s last pose;
the relay reports the hand in `body_state.json`. Read at the same instant after run 3:

```text
FK glove  (ardy xyz): [-0.4502, 0.7537, -0.1214]
FK wrist  (ardy xyz): [-0.4018, 0.9117, -0.0458]
relay hand (pelvis xyz): [0.4150, -0.2145, -0.0118]
gap glove - wrist = 0.1817
```

The glove-to-wrist gap is the expected 0.18 m `VR_LEVER`. The gap between the **FK** hand and the
**relayed** hand is not a lever - it is a different arm configuration, tens of centimetres apart.
So `pose36.json`'s last pose is not the pose the relay is drawing, and a 25.7 mm "achieved"
residual measured on it says nothing about where the hand really is.

That is the defect, and it is not specific to `--move`: the generated reach has measured its own
success against the same ring since it was written, and the ARDY live grab's "24.7 mm achieved"
was read the same way. It needs its own investigation - which pose the ring's last row actually
holds (achieved qpos, or the mode-0 reference being played), and whether the relay's hand is
derived from the same row - before any generated motion can be trusted on the game's screen.

## Status

Task 6 (the live gate) **fails**. The machinery is committed and unit-tested (1473 tests), but a
pull as built would be invisible. Do not tune `MOVE_M` until the pose source is understood: the
commanded distance is not the variable that is wrong.

## Follow-up: the start point was wrong, and the arm does not follow the clip

Session `20261113T223258Z-00f8ba`, same body. Two more measurements, both with the achieved
pose read through `ardy_gen.glove_position` on `pose36.json`'s last row (the achieved qpos, per
`sonic_executor._publish_pose`) and expressed in the pelvis frame with `reach_vr.to_rel`.

**1. `body_state.json`'s hand is a command, not a position, and a pull must not start from it.**
With the body standing at rest after a `/api/body/stop` + `/start`, the relay reported the hand at
pelvis x `0.176` while the achieved glove was 0.5 m away in world x - the relay holds the last
*commanded* pose through a quiet body on purpose. `move_target` now reads the ring's achieved glove
first and falls back to the relay hand only when there is no ring. Committed with this note.

**2. The cruise moves the arm; a generated clip does not.** Measured back to back on the same
body:

```text
mode 1 (cruise, --target 0.46 -0.12 0.02)   achieved glove pelvis x  0.174 -> 0.472
mode 1, then 6 s of nothing                 0.486 -> 0.468 and flat for 6 s   (the hold holds)
mode 0 (generated clip, --move -0.20)       commanded 0.45 -> 0.237; achieved 0.463  (no travel)
```

The clip is not the problem: `mb_clip.npz` carries 0.74 rad of shoulder motion on `jp` rows 22, 24
and 28, smooth (max row-to-row change 0.045), 104 rows, and the body acks it `done` at frame
103/104. The mapping is `g1_pack.jp_from_pose36` (policy/CSV order, not index math). A clip with a
root goal does move the root - 28 mm - so mode 0 reaches the body and applies *some* of the clip.

What does not happen is the arm following the clip's arm reference. That is a pre-existing defect
of the generated-reach path, not of `--move`: it is the same path `grab_pixel` takes.

**Consequence for the record.** `reach_ardy`'s "achieved glove residual" is measured on this ring,
and on this body it reported 33-76 mm for a pull where the hand moved 13 mm. The 2026-11-12 grab's
"12.2 mm achieved" (`research/evidence/ardy-in-the-loop/`) was read the same way, and that run's
own evidence for success was the game's view (`after-card-taken.png`) - which is the only oracle
here that cannot be satisfied by the command.

## What to look at next, in order

1. Why mode 0's arm reference is not tracked while mode 1's VR target is. `build_encoder_obs` is
   called with `vr_pos=None, vr_orn=None` in mode 0 (`sonic_executor.step`), and the mode flag is
   `obs[0:4]`. The deployed ONNX may be dominated by the (zeroed) VR slots in mode 0; the 2026-11-12
   grab would not have shown it if its clip's arm motion was small.
2. `WRIST_BOX` x floor 0.079 is where the *resting* glove sits, so any backward travel from rest is
   clamped to zero. A pull needs the box's retraction limit measured separately from its forward
   reach.
