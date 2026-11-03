# The G1's scale against a game built for adults

Job Simulator is laid out for an adult. The G1 is not one. This note records the numbers,
the three separate effects, and the decision to leave the mismatch in place.

## Measured / cited

| Quantity | Value | Label |
|---|---|---|
| Eye height we emit as the HMD pose | **1.221 m** | MEASURED (`research/e1-spike-results.md`, `e2`) |
| G1 standing height | ~1.32 m | CITED (Unitree) |
| Right-arm wrist travel from rest | **0.41–0.51 m** | MEASURED (`research/sonic-reach-envelope.md`, §6) |
| A lever point we tried to reach | **0.627 m** from rest → 0.12–0.22 m **beyond** the arm | MEASURED (`research/live-pixel-servo.md`) |
| Per-eye FOV we declare | **140° H** (vfov 114.2°) | MEASURED (`research/camera-equals-hmd-pose.md` §7) |
| Adult eye height the game is designed for | ~1.7–1.8 m | ASSUMED |

Apparent size scales as `1 / tan(fov/2)`: `tan(55°) = 1.428` for a ~110° human per-eye FOV
against `tan(70°) = 2.747` for our 140°, so **everything in our renders appears about half the
size it would to a person at the same distance**. This is why "JobBot's head is ~270 px
wide, therefore 0.45–0.65 m away" is the least trustworthy number in this project: distance
estimated from apparent size is FOV-sensitive by roughly 2×.

## Three effects, only one of which is a problem

1. **Absolute height.** Counters sit ~0.5 m higher *relative to us* than intended. Reaching
   *up* is fine; this is not the constraint.
2. **Relative reach.** Our arm travels 0.45 m against ~0.65 m for an adult, so every object is
   proportionally ~1.4× further. **This is the real constraint**, and it is the same 0.627 m
   versus 0.45 m shortfall that every servo stall in this project has the shape of.
3. **The game's own calibration.** Many VR titles, Job Simulator included, adapt the world to
   the reported eye height. If it does, the desk is already pulled toward us and effect 2 is
   partly absorbed by the game rather than by us. **Untested.**

## Operator options and the decision

[OPERATOR] Job Simulator exposes a **"shorter human mode"** toggle which raises the player's
height in-game, at the cost of requiring more looking down. The simulated eye height and the
robot's size are also ours to change.

**Decision (2026-11-02): leave the scale as it is.** No rescaling of the relayed poses.

## Why not scale the relayed poses

We generate the HMD and controller poses, so emitting them `×k` would make the game see a
taller player and every object reachable in game units. It does not give the robot a longer
arm — it moves the *claimed* pose further than the hand can go. The game would render the hand
on the card while the sim shows the hand 0.2 m short of it, and both are screenshots we can
put side by side. A demo whose two views disagree falsifies itself, and "the robot's arm does
it" is the whole artifact. Available only as a labelled "if the avatar were adult-sized"
comparison, never as the claim.

## The open test

Launch at two different HMD heights and compare the counter's projected position and apparent
size at the same look yaw. If the game rescales, the mismatch is largely solved by the game's
own rules and the artifact gets *stronger*: a kid-sized robot in a game that calibrates to
kids. If it does not, the claim is scoped to the measured envelope, which is how the rest of
this project has handled every negative result.

## Related

`research/g1-borrowable-code.md` ranks the borrowable implementations. The missing head
**pitch** (`look_target` yaws only — `spike/wbc/sonic_executor.py:341`) is likely a concrete
cause of the 0.627 m miss: we were asking the arm to reach up to where the camera pointed,
because the camera could not point down.

## Player scaling, measured: the pivot and the head both decide whether it helps

**Result: scaling the hands (not the head) about the resting wrist cuts the reach the card
demands from 0.485 m to 0.339 m, and the height error goes to zero. Lateral survives as the
binding constraint.**

2026-11-03. The G1's eye is 1.221 m and Job Simulator is laid out for an adult, so the card
on JobBot's counter sits 0.485 m from the resting wrist - outside the 0.41-0.51 m of arm
travel once the hand has to start from the clip rest, which is why the body was stepping to
balance and dragging the camera (`research/wrist-response.md`).

`clank.wire.body_state_to_xr` now takes `scale` and a `pivot`, and both were measured rather
than chosen:

- **Scaling about the floor makes it worse.** Same body pose, card depth on the stream:
  `0.527 m` with the scale on against `0.342 m` with it off, so the game really does lift
  the camera to a taller player. Raising the pivot by 0.43 x 0.58 m moves the hand up 0.25 m
  while lengthening the arm only 0.43 x 0.45 = 0.19 m, and the card is *below* the shoulder,
  so the rise costs more than the longer arm earns. The demanded reach went 0.486 -> 0.566.
- **Scaling about the resting wrist is the useful pivot.** `travel = |O - H| / scale`, so
  the card needs `0.485 / 1.43 = 0.339 m`, inside the 0.40 m clamp.
- **But only if the head is left alone.** With the head scaled too, the camera rises 0.29 m,
  the depth it reports changes, and the object's apparent position moves with it: 1.43
  bought `0.500 -> 0.472 m`, four per cent. With `scale_head=False` the camera stays exactly
  where every measurement in `research/` was taken, the depth stays valid, and the hand still
  travels `scale` metres per real metre.

First live attempt with hands-only scaling about the wrist pivot, commanding the computed
0.339 m offset: the wrist moved 0.304 m, most of it up (+0.261) and forward (+0.147). The
emulated hand - `pivot + scale * (hand - pivot)` - came out at height 0.170 m against the
card's 0.177 m. **Height error: 7 mm.** The reach was 96 % vertical, and the scaling closed
it.

What is left is entirely lateral, about 0.20 m. The arm's up-command carries almost no
sideways motion (lateral gain ~0.2 at this posture), so the lateral **command** needed is
around 0.75 m, far beyond the 0.18 m clamp - the clamp that exists because a large lateral
offset is what leans and then walks the body. So the next blocker is not reach length and not
height; it is how this arm moves sideways, and whether the body may be allowed to help
(`touch_card.py` survived 0.34 m lateral with a fall-risk check at every step).

### Declared caveat, unchanged

The robot's own dimensions are untouched: MuJoCo still shows a 1.32 m humanoid whose arm
moved 0.30 m while the game drew its hand on the counter. The honest sentence stays "the
emulated player is 1.43x, so its arm reaches 1.43x further in the room", never "the G1
reaches the counter". `--player-scale-head on` gets the old whole-body behaviour back.
