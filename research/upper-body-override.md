# The deploy has an arm escape hatch, and our driver is the problem

**STATUS: MEASURED in code, 2026-11-04.** Written after a scout pass over upstream issues,
docs and the SONIC paper. Two findings, one of which changes the plan.

## 1. The wrist-pose interface was never meant to carry steps

`docs/source/references/observation_config.md:225-246`: the VR 3-point observation is
`vr_3point_local_target` **9 floats of position** and `vr_3point_local_orn_target` **12 floats of
quaternion**, plus a compliance scalar. **No target velocity, no horizon.**

The motion-reference path, in contrast (`:99`, `:123-134`) carries
`motion_joint_velocities*` and `motion_joint_positions_10frame_step5` - 10 frames at 0.1 s, a
**0.9 s look-ahead**. So the policy can track a clip it can *see coming* and has no way to
anticipate a pose it is merely told about. A step on the VR path is a position setpoint with
zero velocity feedforward. That is the structural reason it rings.

NVIDIA's own numbers for that interface (paper, and the CSDN summary that quotes it at
`blog.csdn.net/v_july_v/article/details/156131`): **121.9 ms mean latency, 6 cm mean / 13.3 cm p95
right-wrist position error, 0.145 rad orientation error** over 300+ teleop trajectories.

**Our measured 0.77-1.05 s settle and +75 % overshoot is 6-8x outside that**, so it is our driver,
not the policy's floor. That reframes the last three sessions: the arm was never slow, the
reference was illegal.

Training-side reasons it overshoots rather than settling cleanly (paper Table S3):
end-effector position reward is `exp(-||p-p_goal||^2 / 0.1^2)` - a **10 cm kernel**, so at 0.3-0.5 m
error the gradient is nearly flat and the policy closes the gap at whatever speed the velocity
terms allow, then stops with momentum; and `r_shake` penalises wrist angular velocity above
**1.5 rad/s**, so a wrist that must traverse a large rotation in 0.3 s is fighting its own penalty.

## 2. There is a documented, in-tree arm override - a joint payload, not an IK mode

`gear_sonic_deploy/.../include/input_interface/input_command.hpp:60-96`: the planner message
carries **`upper_body_position[17]` and `upper_body_velocity[17]`**. When present,
`GatherMotionJointPositionsMultiFrame` (`src/g1_deploy_onnx_ref.cpp:738-800`, velocities at
`861-873`) copies the clip's 29 joints and then **overwrites these 17, for every look-ahead
frame**, while locomotion still comes from the planner. So an external caller can run its own arm
IK and hand the policy 17 joint angles; **legs and balance stay on the policy**.

The indices, verbatim from `policy_parameters.hpp:80-81`:

```
isaaclab order: {2, 5, 8, 11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28}
mujoco   order: {12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28}
```

17 = 3 waist + 14 arm joints.

Three independent deploys already do this: Unitree's own `xr_teleoperate`
(`G1_29_ArmIK`, CasADi/IPOPT, then `WeightedMovingFilter([0.4,0.3,0.2,0.1])` over 14 joints plus a
30 rad/s per-tick clamp), `GalaxyGeneralRobotics/OpenWBT` (imports that same class out-of-process),
and `kb131-kang/groot_teleop_dev` (XR -> G1 arm IK -> `rt/lowcmd`).

## Why this does not contradict our own IK regression

We measured `spike/wbc/arm_override_c3j.py`: policy untouched **0.0114 m**, hard IK **0.1440 m**.
That override replaced the policy's **ctrl** after inference. The deploy's contract is different in
the one way that matters: it replaces the **reference the policy tracks**, so the policy still
supplies its own compensation for its drooping arm (PD gap 14.25 vs 99.1). Overriding ctrl fights
the policy; overriding the reference *asks* it.

## What is implemented now

`spike/wbc/sonic_runner.py`: `UPPER_BODY_ISAACLAB_IDX` / `UPPER_BODY_MUJOCO_IDX`, a
`self.upper_body` slot, and `_jp()` routing the four full-29 observation writes through it.
Verified: exactly the 17 indices change, the 12 leg/other joints are untouched, and the clip row is
not mutated. With `upper_body = None` behaviour is identical to before.

## Next, and what decides it

Feed `upper_body` from the DLS IK already in `spike/wbc/arm_override_c3j.py`, solve for the same
Cartesian wrist targets the current reach uses, and measure the reported wrist point. If it lands
within a few cm where the pose path takes 3 commands and 65-84 mm, the reach becomes one shot and
the lever/overshoot disappear.

Open questions worth an A/B, from the scout:
- `upper_body_velocity` with a zero vector while positions move may itself drag. Compare zeros,
  finite differences, and the clip's own velocities.
- Whether the leg channel of the loaded clip keeps playing while standing (the override touches
  only the 17).

## Measured: the override does not work here, and VR mode cannot read it

Two experiments after wiring it up. Both are negative, and the second one corrects the claim
above, so read this before trusting the section before it.

### The 17 joints are never read in VR mode

`sonic_runner.build_encoder_obs`, default model, mode 1 (VR):

```python
o += 290 + 290            # 10 frames x 29 joint positions and velocities - SKIPPED
...
obs[o:o + 12] = motion["jp"][f, LOWER_CSV]   # only the 12 lower-body joints are written
```

So in VR mode the observation carries the 12 lower-body reference joints, the anchor, and the
3 VR poses - and **nothing** of the 29-joint reference. The upper-body override has nothing to
act on there. It is a motion-tracking channel: `upper_body_position[17]` in NVIDIA's deploy must
be paired with the motion-reference input type, not with the VR 3-point input.

That also means the `0.77-1.05 s` overshoot we measured is *not* fixable this way. The VR path is
the only arm channel we have, and it is a position setpoint with no velocity and no horizon.

### On the motion-tracking path the policy does read it - and does not adopt it

mode 0, clip frozen, 17 joints from IK substituted:

| quantity | measured |
| --- | --- |
| IK residual on the commanded point | 164 mm (target 235 mm from rest) |
| policy's joint tracking of the substituted reference | **0.125 rad mean, 0.299 rad max** |
| hand movement delivered out of 235 mm commanded | **~33 mm** |

So even where the policy can see the joints, it does not follow them: 0.125 rad is 7 deg mean
error per joint, which is the droop the policy normally corrects for when it is *choosing* its own
arms, and which it does not correct when the arm joints are handed to it from outside.

This is the same conclusion as `arm_override_c3j` reached at the *ctrl* level (policy 0.0114 m vs
hard IK 0.1440 m), now reproduced at the *reference* level. The arm is under-powered (PD 14.25
against the legs' 99.1) and the policy's compensation is part of its own action, not something
that survives an external joint target.

### The IK itself is fine

Damped Gauss-Newton with Levenberg-Marquardt and backtracking, on the 7 arm joints, aiming the
reported hand point (`position + R * lever`, with the lever term in the Jacobian):

| target offset from rest | IK residual |
| --- | --- |
| +0.05 z | **1.1 mm** |
| +0.05 x | **0.8 mm** |
| -0.10 z | **0.3 mm** |
| +0.10 x, +0.05 z | 50.6 mm |
| +0.14 x, -0.10 y, +0.16 z | 132.9 mm |

Sub-millimetre where the pose can reach, an honest residual where it cannot. Two bugs worth
recording because both cost a cycle: `mj_jacBody` reads `cdof`, which `mj_kinematics` does not
fill, so the Jacobian came back **exactly zero** and the solver reported its starting error
forever with `dq = 0`; and fixed damping on a near-singular arm Jacobian made the residual
oscillate 235 -> 147 -> 339 -> 116 mm.

### What this leaves

The reach stays on the wrist-pose channel with a cruise-limited reference (0.13 m/s), which is
measured at 2.3-2.4 s for 0.31-0.32 m and 65-84 mm after three commands. The remaining error is
the wrist lever, which turns with an orientation we only request. That is where the next attempt
belongs, not in joint-space injection.
