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

## The remaining contradiction, stated exactly

Two readings of "where is the hand" disagree, and the reach's aim depends on one of them:

| reading | source | value on a standing body |
| --- | --- | --- |
| achieved glove | `ardy_gen.glove_position(pose36[-1])`, world | x **−0.266**, y −0.166, z 0.596 |
| relay hand | `body_state.json` `right_hand.p`, pelvis | x **+0.177**, y −0.275, z −0.194 |

Put through the root transform both land at the same **height** (0.596 vs 0.593) and within 4 cm
in **y**, and differ by **0.5 m in x**. When the cruise moved the arm, *both* readings moved
forward by 0.2-0.3 m - so `glove_position` tracks the body correctly and this is a **constant
offset**, not a bad forward kinematics. A constant 0.5 m in x with y and z agreeing is not the
0.18 m `VR_LEVER`, so the two are not the same point under the same transform.

Ordering was checked and is **not** the explanation, contrary to the first guess in this file:

- `pose36` hinges are `g1_pack.QPOS_QADR` order = left leg, right leg, waist, left arm, right arm,
  which is `g1_pack.JOINT_NAMES` order.
- `ardy_gen._motion_pose` indexes the converter by pose36 slot, and
  `converter._mujoco_indices_to_ardy_indices` is `[1..6, 8..13, 15..24, 26..32]` - a slot-identity
  map that skips the root and the two hand groups. So ARDY's FK reads the ring's order correctly.
- `jp`/`jv` in a deposited clip are **policy/CSV** order (`g1_pack.jp_from_pose36`), which is the
  interleaved order `sonic_runner._jp` feeds the encoder. Also correct.

So both sides are internally consistent, and the 0.5 m is a *frame* question: which point the relay
calls the hand, and through which transform. That is the next thing to measure - not `MOVE_M`, and
not the clip.

## Harness for whoever picks this up

```bash
# achieved pose (world), and the same in the pelvis frame
LD_LIBRARY_PATH=/run/opengl-driver/lib spike/ardy/.venv/bin/python - <<'PY'
import json, pathlib, sys
sys.path.insert(0, 'spike/wbc'); sys.path.insert(0, 'spike/wbc_game')
import ardy_gen, reach_vr
p = pathlib.Path.home()/'.local/state/Clank/sessions/body'
ring = json.loads((p/'pose36.json').read_text()); st = json.loads((p/'body_state.json').read_text())
g = ardy_gen.glove_position(ring['poses'][-1], hand='right')
print('glove world', [round(v,3) for v in (g[2], g[0], g[1])])
print('glove pelvis', [round(v,3) for v in reach_vr.to_rel((g[2], g[0], g[1]), st)])
print('relay hand  ', [round(float(v),3) for v in st['right_hand']['p']])
PY
```

Poll that during a clip and during a cruise and the difference is visible in seconds. **Do not
trust `reach_ardy`'s "achieved glove residual" until it agrees with the relay on a still body.**

## The answer, 2026-11-13: a travel is a cruise, and the frame question was a red herring

Standalone body (`spike/wbc/body_view.py --no-render`, no game, no Monado), polled at 2 Hz with
the achieved glove read through `ardy_gen.glove_position` on `pose36.json`'s last row.

**1. `glove_position` and the relay's hand are the same point in the same frame.** The
contradiction above - 0.5 m apart in x - was measured while the relay's reading was **stale**,
which its own docstring says it will be ("a dead body must not release a grab"). On a freshly
started body, and on every tick of a cruise:

```text
fk glove pelvis  [0.465, -0.068, 0.025]
relay hand       [0.464, -0.066, 0.025]     1 mm, and they moved together
```

So there is no frame bug, `reach_vr.to_rel` is right, and `move_target` was right to start from
the ring. The ordering check that ruled out the qpos mapping was also right; nothing was wrong
except reading a command as a position.

**2. The generated clip is tracked, and then the arm walks home.** `--move -0.20` from pelvis
x 0.430:

```text
t=13.5  0.229   the clip is playing and the arm is where the clip says
t=15.0  0.270
t=16.5  0.464   clip done + hold sent; the arm goes back to the PREVIOUS reach's target
t=18.1  0.432
```

The clip is not the problem: `mb_clip.npz` asks for glove pelvis x 0.230 at frame 0 and 0.196 at
frame 103, and the body goes to 0.23. What fails is everything after `done`.

Three repairs were tried and all three lost the arm within a second:

- a single `hold` verb straight after `done`;
- the `hold` verb re-sent every 0.5 s for 3 s (the arm stays while it is re-sent, then leaves);
- `body_view`'s `hold` verb baking `executor.last_state` (a real defect: a bare `hold` carried no
  pose, so `CommandBus.take()` baked its **stale** latch - the snapshot from the last non-hold
  command - and the arm walked to that old target). Fixed, and it did not fix the pull.

The mechanism, read out of `clank/bus.py`: `push()` retires the freeze-hold snapshot only on a
real command, and **a deposited clip is not one**. So after a clip the snapshot is whatever the
last socket command left, and the freeze-hold that follows bakes it. The arm returns to the
previous reach's target, not to the clip's end.

**3. The cruise holds, so the travel goes through the cruise.** Same body, same hand, same 0.20 m:

| path | travel | 10 s after the process exited |
| --- | --- | --- |
| generated clip (`reach_ardy --move`) | 0.20 m, 67 mm residual | **back where it started** |
| cruise (`reach_vr --move`) | 0.174 m, 7.4 mm residual | **still there** |

`reach_vr` re-commands an absolute target every keepalive, which is why it survives its own exit.
A travel needs nothing a generator adds - no object to plan around, no root step - so
`auto._move_argv` now routes every travel to the cruise for **both** hands, and `--grab` keeps the
grip.

Two more defects found on the way, both fixed:

- `reach_vr.py`'s source guard did not know about `--move`, so the one flag that makes a pull a
  pull was rejected on the command line (`give a pixel, --target X Y Z or --world X Y Z`) while
  `moved_point` was already wired below it.
- `reach_ardy.py` pins the achieved pose for 3 s after a clip (`pin_where_it_is`). It does not
  make a generated clip hold, but it does keep a **grab** where the clip left it long enough to
  read, and a grab that walks home would drag the object with it.

**End to end, the exact argv `auto.py` builds:**

```text
cruise to (0.45, -0.15, 0.0)      achieved 0.422
reach_vr.py --move -0.200 0 0 --side right --no-rest --no-clamp --grab
                                  goal 0.250, achieved 0.268, error 37.6 mm
9 s after the process exited      achieved 0.282-0.297   (still pulled, hand closed)
```

**Still open:** the `CommandBus` freeze-hold snapshot is not retired when a clip is deposited.
That is the one thing that would let a generated clip hold its own result, and it matters for
every generated grab, not just a pull.

## The latch, 2026-11-13: a clip is not a command

The fix is `CommandBus.retire_latch()`, called from `SonicExecutor._finish_clip` through a
`body_view`-wired callback. `push()` retires the freeze-hold snapshot when a real command
arrives; a deposited clip arrives through the mailbox and never passes through `push()`, so the
snapshot still pointed at the pose from *before* the clip.

Verified on the live session body, restarted onto the new code:

```text
rest                        0.153
cruise to (0.45, -0.15, 0)  0.436 -> settles ~0.40
generated --move -0.20      0.195      (residual 29.7 mm)
7 s after the process exits 0.199 - 0.209   still there
```

Before the fix the same sequence read `0.229 -> 0.464` inside 2 s. That is the whole difference
between a pull the game can see and one it cannot, and it applies to every generated grab: a
grab that walked home would drag the card or the cartridge with it.
