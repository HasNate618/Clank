# The hand actually closes now

Until 2026-11-06 the robot grabbed with an open hand. `clank/body.py` said so in a comment and
blamed nothing in particular; `research/g1-borrowable-code.md` ranked the fix as borrow #1 and named
the exact preset. This is that fix, measured.

## What was wrong

Three separate gaps, and the first one hid the other two:

1. **`SonicRunner.step_policy` writes `ctrl[:29]`.** The model has 43 actuators. The last 14 are the
   hand motors, so they read zero torque forever. Looked like a fingerless model; it is not - 7
   joints and 7 motors per hand, `left_hand_thumb_0..2`, `left_hand_index_0/1`,
   `left_hand_middle_0/1` and the same on the right.
2. **The hand motors are `<motor>` torque units**, `ctrlrange +-0.7 N m` (`+-2.45` on `thumb_0`),
   not position units. Commanding an angle does nothing; they need the same PD the legs get, with
   gains about 30x lower because the torque ceiling is 0.7 not 88.
3. **`closure` only ever travelled outward.** `SonicExecutor` latched it and republished it in
   `BodyState.closure`, which the relay turns into the controller's squeeze button. The game knew
   the hand was closed; the body did not.

## The numbers

`GRASP_Q` is GR00T's `grasp_qpos` (`g1_threefinger_hands.py:47,125`), checked against this XML's
joint ranges - every component inside them:

```
left  = [0.0,  1.0,  1.25, -1.2, -1.2, -1.2, -1.2]
right = [0.0, -1.0, -1.25,  1.2,  1.2,  1.2,  1.2]
```

Target is `g * GRASP_Q` per side, approached at **0.25 rad/tick** (GR00T's `writeOnce` slew), which
makes a grab a five-tick clasp at 50 Hz. Servo: `KP 3.0`, `KD 0.10`, torque clipped to ctrlrange.

Measured in sim, arm raised clear of the hip, 20 ticks after a grab:

| | thumb_0 | thumb_1 | thumb_2 | index_0 | index_1 | middle_0 | middle_1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| target (right) | 0.0 | -1.0 | -1.25 | 1.2 | 1.2 | 1.2 | 1.2 |
| achieved | **0.01** | **-0.96** | **-1.22** | **1.32** | **1.33** | **1.32** | **1.33** |

Max error **0.129 rad**. The fingers overshoot the preset by 0.13 rad because they are closing
against their own contact geometry, which is what a grasp looks like. Release ramps the same way
back to zero.

## The trap: at rest the hand is jammed against the hip

First attempt, arm hanging at the rest pose, the thumb never moved: `right_hand_thumb_1` sat at
**+0.730** - its upper limit - while its motor pushed **-0.700 N m**, the full negative torque.
MuJoCo reported the reason in the contact list:

```
contact right_hip_pitch_link <-> right_hand_thumb_2_link  dist=-0.0004
```

The down arm rests against the hip link and the thumb is pinned by it. So **a clasp test must raise
the arm first**; a grab happens with the hand away from the body and the thumb free. Anyone
re-measuring this in the rest pose will conclude the hand is broken when it is only leaning on the
robot.

## What this does not do

- `BodyState.closure` stays the **commanded** value, not measured. The game's squeeze button must
  fire the instant the body is told to grab; making the field measured would delay the button until
  the fingers arrive, and a 0.1 s late squeeze is a grab the game can miss. The joints are the
  honest part now, the field is the fast part.
- Rendering a close-up from this shell fails (`an OpenGL platform library has not been loaded`), so
  the picture of the clasp has to come from the live body view, not a headless MuJoCo render.

## Verified live (2026-11-06)

Restarted the body so the new executor loaded, then ran the reach with `--grab` at the operator's
clicked lever point (0.427, -0.358, 0.357; 37.3 mm on the first pass) and photographed the body view.

**`research/evidence/hand-clasp-closed.png`** - the robot's right hand is curled into a claw while the
left hangs open beside the torso. That is `closure = (0.0, 1.0)` with nothing else driving the hand.

**`research/evidence/hand-clasp-released.png`** - after `release_right`, the same hand is open and the
fingers are extended. Same arm pose, same camera; only the grip changed.

So the loop closes in both directions: the game's squeeze and the body's fingers are now the same
event, and releasing opens the hand instead of leaving it frozen shut.

## What it looks like up close

`research/evidence/hand-clasp-open-closed-open.png` is three live body-view frames from one fixed
camera, only the grip changing: **open, closed, released**.

The clasp is a **pinch, not a fist**. Open, the index and middle fingers splay apart in a V; closed,
they come parallel with the tips converged toward the opposing thumb; released, they splay again.
That is what a 7-DoF two-finger-plus-thumb hand does with `grasp_qpos`, and it is why the mid-frame
of that sheet looks straighter rather than shorter - the fingers converge, they do not curl over.
Anyone expecting a human fist will read the closed frame as "still open" from a distance.

## The other half: the game's glove, and the channel it listens on

The body closing was only half of it. In the game view the glove stayed open, and the operator's
reference was **manual override mode**, where a left click visibly turns the VR hand into a fist.

That is the comparison that found the answer. `spike/c0/c0_manual.py:365-374`, the path where the
click fisted the hand, sets three things together:

```python
st.squeeze = 1.0 if self.grip else 0.0
side.squeeze_force.x = 1.0 if self.grip else 0.0
for i in ...:
    side.hand_curl[i] = 1.0 if self.grip else 0.0
```

`clank/pose_relay.py` had `--grab-mode squeeze` as its **default**, which sends `squeeze_value` and
`squeeze_force` and leaves `hand_curl` at zero. Squeeze is enough to *grab* - that is why the card
lifted weeks ago - but it is `hand_curl` the game draws the fist from.

Measured 2026-11-08, hand held 0.42 m in front of the head so the glove fills the view:

| relay mode | wire carries | glove |
| --- | --- | --- |
| `squeeze` (old default) | squeeze 1.0, curl 0.0 | **open** - open and closed frames pixel-identical |
| `both` (new default) | squeeze 1.0, curl 1.0 | **fist** |

`research/evidence/hand-game-open-vs-fist.png` is that pair: left frame the splayed glove, right
frame the fist, same pose, same camera, only the grip changed.

The default is now `both`, so a session gets the fist without a flag. One wrong comparison cost an
hour: the first A/B ran open-vs-closed with `curl` already following the grip in *both* frames, so it
measured curl=1 against curl=1 and "proved" the game ignored it. The relay's own packet log
(`sessions/<id>/body_pose.csv`, columns `grip_l,grip_r,curl_l,curl_r`) is the record of what was
actually sent, and it disagreed with what I had assumed the process was running.

## Sequential test, both hands

Both hands raised 0.44 m in front of the head, then closed and opened one after the other. The
packet log is the whole story:

```
t=174.14s  grip L/R=1.0/0.0  curl L/R=1.0/0.0     left closed
t=177.14s  grip L/R=0.0/0.0  curl L/R=0.0/0.0     left open
t=180.14s  grip L/R=0.0/1.0  curl L/R=0.0/1.0     right closed
t=183.12s  grip L/R=0.0/0.0  curl L/R=0.0/0.0     right open
```

Three seconds each, one hand at a time, `curl` mirroring `grip` on both edges, and the body and the
glove both following. Run 2026-11-08, watched live on the body view and in the game.
