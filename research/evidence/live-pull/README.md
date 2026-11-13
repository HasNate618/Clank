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
