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
