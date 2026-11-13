# The lever: a saved world point, a pull, and the one-shot that proves it

2026-11-13, live session, Job Simulator running, depth stream live. Operator watching and
listening; the operator's report is the oracle here because the lever sits at the edge of the
frame where the hand is hard to see.

## The handle's world position, measured once from the default stance

The body was reset (`/api/body/stop` + `/start` + six `rest` verbs) so the measurement is
reproducible: root `[0.002, 0.016, 0.787]`, right hand `[0.153, -0.198, -0.219]`, hand open.

One still was saved **with its camera pose**, so the pixel can be unprojected later without the
head having drifted in between (`color.png`, `depth.npy`, `camera.json`, frame 12777). The knob
was marked by eye at pixel **(634, 390)**; depth there is **0.423 m** (median of a 7x7 patch,
0.414-0.433), which is the handle's front face.

```text
pelvis  [0.351, -0.291, 0.184]
world   [-0.275, -0.340, 0.979]      saved in lever.json
```

`lever.json` keeps the pixel, the frame id, the camera and root poses, and the depth it came
from. Reuse it with `reach_vr.py --world -0.2754 -0.3396 0.9786`, which is why `--world` now
works: it was parsed, documented in the help, allowed by the source guard, and **never handled** -
it fell through to the pixel branch and died on `args.px / cw`. Fixed in this commit.

## The pull

```bash
reach_ardy.py --world -0.2754 -0.3396 0.9786 --grab --side right
```

`reach_ardy` classified it `step` - the handle is **-39.6 deg off the body's heading** while the
torso already starts **+41.0 deg** off it - and generated a root goal that travelled **0.00 m
forward**: it chose a *turn*. Two generations, glove residual 47.1 then 28.0 mm, clasp fired at
**56 mm** from the object, achieved residual **50.8 mm**.

**The operator confirms the lever was pulled.** That matters because nothing in the robot's own
numbers says so: a 56 mm clasp is inside `CLASP_START_M` but on a 4 cm knob it is not proof of
contact, and the hand sits low and to the side of the frame.

## The one-shot: recorded, but not placed

`parecord --device=clank-game.monitor` across the attempt. 50 ms blocks, floor (p20) 596, median
844, peak 5331:

```text
event 31.10 s - 31.45 s   (0.35 s)   peak 5331
```

One event, 0.35 s long, no speech anywhere else. **It is not evidence of the pull, and the first
draft of this file said it was.** The clip finished at 18:28:55 (`mb_play` seq 2, the successful
run's second generation); the capture holds 54.0 s of audio written between 18:29:06 and 18:30:00
- 54 s of audio across ~105 s of wall clock, so the recorder stalled and its timeline cannot be
aligned to the body's. The transient sits after the pull, not at it, and the game's
`output_log.txt` has no lever line to arbitrate.

So the oracle for this pull is the **operator's eyes**, and the lesson for the next attempt is
procedural: timestamp the recorder's first byte, log the pull's own wall-clock time, and only then
call a transient the lever's sound. `pull-one-shot.wav` is kept as the candidate, not as proof.

**If it is the lever's sound, it is the success signal worth wiring**: `CLASP_START_M` fires at
80 mm on a 4 cm knob, while a sound effect fires only when the game agrees the handle moved. That
would be a `confirm()`-style short ungated window around the pull rather than asking the model
whether it thinks the pull worked. Unproven until the timing above is done properly.

## What the operator says is wrong with the reach

The lever is far enough that the body **twists** to get to it, and a twist is why the grab cannot
be seen. The robot should **walk forward or lean forward**, not twist. Today's classifier cannot
say that: `end_pose.step_root` puts the root where the preferred hand lands on the object, and
because the handle is 0.45 m away - inside the measured 0.41-0.51 m arm travel - it concludes the
step is 4 mm and only turns. The turn is the thing that does not happen (`AUTH_ROOT_VEL` is
ignored: measured +0.2 deg of 69 deg asked), so the body ends up twisted instead.

So the open item is a **forward root goal chosen for comfort, not for reachability**: bring the
target inside the arm's untwisted envelope instead of asking the torso to rotate. That is a
`reach_ardy` flag (`--root-forward M`) plus the same generator call, not new machinery.
