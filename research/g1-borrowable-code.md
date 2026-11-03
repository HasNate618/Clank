# Borrowable code for G1 hand placement, retargeting, servoing, and pose injection

Ranked by the brief's order. Evidence labels: **M** = I read the cited file/function or LICENSE; **C** = README/issue/API metadata says so, code not read; **A** = assumed whole or unverified. Paths are relative to the repo root unless a URL is given.

## Ranked table

| # | Need | Source (license, URL) | Exact file / function worth reading | Ev | Maps to ours |
|---|------|----------------------|--------------------------------------|----|--------------|
| 1 | Scalar closure -> 7 joint angles, and a valid full-grasp pose | GR00T-WholeBodyControl, vendored (source Apache-2.0) — https://github.com/NVlabs/GR00T-WholeBodyControl | `spike/wbc/vendor/GR00T-WholeBodyControl/decoupled_wbc/dexmg/gr00trobocasa/robocasa/models/grippers/g1_threefinger_hands.py`: `format_action` lines 32-41 (left) and 112-119 (right), `grasp_qpos` lines 47 and 125 | M | `spike/wbc/body_view.py:321` (`hand_button`) and `spike/wbc/sonic_executor.py:352` (`hand_target`); closure currently only reaches the controller button |
| 2 | Per-joint Dex3 limits, max-close scaling, delta clamp | GR00T, vendored (Apache-2.0) — https://github.com/NVlabs/GR00T-WholeBodyControl | `spike/wbc/vendor/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/dex3_hands.hpp`: `writeOnce` 116, `clipToMaxOpen` 421, `MAX/MIN_LIMITS_LEFT/RIGHT` 460-463, `close` 339, `open` 360 | M | Same two files; gives the safety envelope for a closure scalar and the 0.25 rad/tick slew limit |
| 3 | Per-finger closure presets from fingertip distances | GR00T, vendored (Apache-2.0) — https://github.com/NVlabs/GR00T-WholeBodyControl | `.../decoupled_wbc/control/teleop/solver/hand/g1_gripper_ik_solver.py`: `_get_index_close_q_desired`, `_get_middle_close_q_desired`, `__call__` | M | Demonstrates the "distance below threshold -> 7-vector preset" pattern; requires Manus-style fingertips we do not have, so lower value than #1 |
| 4 | Exact Dex3 joint order and a full retargeting config | `unitreerobotics/xr_teleoperate` (Apache-2.0) — https://github.com/unitreerobotics/xr_teleoperate | `teleop/robot_control/hand_retargeting.py`; `assets/unitree_hand/unitree_dex3.yml`; `Dex3_1_Left/Right_JointIndex` in `teleop/robot_control/robot_hand_unitree.py` | M | Confirms the deployed XML's order; the YAML is a ready-made DexPilot config if we ever get 21 human hand keypoints |
| 5 | XR wrist pose -> G1 arm IK, G1-specific | `unitreerobotics/xr_teleoperate` (Apache-2.0) — https://github.com/unitreerobotics/xr_teleoperate | `teleop/robot_control/robot_arm_ik.py`: `G1_29_ArmIK` line 18, `scale_arms` 245, `solve_ik` 253 | M | Reference for `spike/wbc/sonic_executor.py` wrist targets; its failure path returns the previous q (freeze), which is weaker than `clank/workspace.py`'s measured clamp |
| 6 | Unreachable-target handling by soft IK, not clamping | GR00T, vendored (Apache-2.0) — https://github.com/NVlabs/GR00T-WholeBodyControl | `.../decoupled_wbc/control/teleop/solver/body/body_ik_solver.py`: `BodyIKSolver.__call__`, `initialize`; `body_ik_solver_settings.py`: `link_costs`, `lm_damping`, posture weights | M | The principled version of what `clank/workspace.py` approximates: damp the step, add a posture bias, clip to limits; copy the constants, not the pinocchio/pink dependency |
| 7 | Eye-in-hand IBVS and depth from known target size | `lomash-relia/vision_mujoco` (MIT) — https://github.com/lomash-relia/vision_mujoco | `eye_in_hand_ibvs_v2.py`: `compute_image_jacobian` 222, `estimate_depth_from_marker` 305, `ibvs_control` 402; `docs/ibvs_math.md` | M | Directly informs `clank/projection.py`: `unproject` 255 and `servo_step` 300; the depth formula is `Z = S_physical * f_x / S_pixels` |
| 8 | Hand-tracking-style extension over the Monado wire | Monado (BSL-1.0) — https://gitlab.freedesktop.org/monado/monado | `src/xrt/drivers/remote/r_interface.h`: `r_remote_controller_data` line 75 and `hand_curl[5]` 81; `src/xrt/drivers/remote/r_device.c`: `r_device_get_hand_tracking` 139; `src/xrt/drivers/remote/r_hub.c`: `create_hand_tracker` 430, hand roles 497; `src/xrt/auxiliary/util/u_hand_simulation.c`: `u_hand_sim_simulate_for_valve_index_knuckles` 394 | M | `clank/pose_relay.py` and `clank/wire.py:92,377`: the wire already carries 5 curls, currently the same scalar on all five fingers; set index/middle/thumb from the 7-joint closure and ring/little to 0 |
| 9 | Prior art: a MuJoCo "simulated user" drives a Unity OpenXR app | `fl0fischer/sim2vr` (no license declared) — https://github.com/fl0fischer/sim2vr | `README.md`; `UnityEnv` / `UnityHeadset` / `SIM2VR Asset` | C | Architecturally the closest to our loop (poses out, image back), but **no license = do not copy**; read-only |
| 10 | Prior art: MuJoCo + OpenXR in one process | `spinworks-tech/RoboVRTele` (no license declared) — https://github.com/spinworks-tech/RoboVRTele | `app/main.cpp` (OpenXR + GLES + MuJoCo + damped-least-squares IK) | C | Reverse direction (MuJoCo renders to a Quest); **no license = do not copy** |
| 11 | Prior art: MuJoCo <-> OpenXR Python bridge | `SkytAsul/DeformableSimulation` (no license declared) — https://github.com/SkytAsul/DeformableSimulation | `mjxr_tests/mujoco_openxr.py`, `mjxr_tests/mujoco_xr_full.py` | C | Uses pyopenxr; also has finger retargeting from closure/abduction. **No license = do not copy** |
| 12 | OpenXR rendering inside MuJoCo `simulate` | `nishbo/mujoco`, branch `open-xr-vr-base`, PR #2124 (Apache-2.0 inherited from MuJoCo) — https://github.com/nishbo/mujoco/tree/open-xr-vr-base | `simulate` VR integration | C | Render-only and Windows-first; not a pose-injection bridge; not needed |
| 13 | Full hand retargeting library | `dexsuite/dex-retargeting` (MIT) — https://github.com/dexsuite/dex-retargeting | `README.md`, `pyproject.toml`, `src/dex_retargeting/` | C (M for license/deps) | Requires 21 human keypoints and pinocchio/nlopt; overkill for a click-only grab |
| 14 | Body retargeting only | `YanjieZe/GMR` (MIT) — https://github.com/YanjieZe/GMR | repo; issue #139 explicitly asks for G1 hands and is open | C | Body-only; no hand support, so not useful for #1 and redundant with SONIC for #2 |

## Concrete borrow #1: closure scalar to Dex3 joints

The deployed model is `spike/wbc/vendor/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof_with_hand.xml` (included by `scene_sonic_floor.xml`, which `spike/wbc/sonic_runner.py` loads). Its **joint declaration order** is, by name:

```
left : thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1
right: thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1
```

This is the order MuJoCo uses for `qpos`; the actuator list follows the same order. The robocasa hand XML uses `thumb, index, middle` for the left hand, but the two fingers receive the same sign and magnitude in every preset below, so the mistake is harmless. Look up by name if in doubt.

The GR00T commented scalar form (`g1_threefinger_hands.py:32-41` and `112-119`) is, for `g` in `[0,1]`:

```
left  = [0.0,  g,  g, -g, -g, -g, -g]
right = [0.0, -g, -g,  g,  g,  g,  g]
```

The full-closure endpoint (`grasp_qpos`, same file, lines 47 and 125) is:

```
left  = [0.0,  1.0,  1.25, -1.2, -1.2, -1.2, -1.2]
right = [0.0, -1.0, -1.25,  1.2,  1.2,  1.2,  1.2]
```

Both are inside the deployed XML limits. The simplest smooth mapping is `q = g * grasp_qpos` (elementwise; `thumb_0` stays 0). The literal GR00T form closes the fingers less at partial `g` (`-g` vs `-1.2g`); either is defensible, but pick one and keep it. Do **not** copy `Dex3Hands::close()`: it sets each joint to the midpoint of its range, which is a half-closed test pose, not a grasp. `Dex3Hands::open()` is simply `q = 0`.

The `max_close_ratio` in `dex3_hands.hpp` is a **limiter**, not the closure command: `clipToMaxOpen` clips a requested target to `ratio * limit`, and `writeOnce` additionally clamps each joint to `0.25 rad` per tick. The latter is a cheap, dependency-free safety rule worth copying.

Monado's `hand_curl[5]` order is **not** the Dex3 order: it is `[little, ring, middle, index, thumb]` (`r_device.c:156-160`). Our `clank/wire.py:377` currently writes the same scalar into all five, so the order has not mattered yet; it will as soon as we split the fingers.

## Deployed hand joint limits

Measured from the deployed XML (`gear_sonic_deploy/g1/g1_29dof_with_hand.xml`). `range` is the joint position limit in radians; `axis` is the MuJoCo joint axis.

| Joint (per side) | Left range | Right range | Axis |
|---|---|---|---|
| `thumb_0` | -1.0472 .. 1.0472 | -1.0472 .. 1.0472 | y |
| `thumb_1` | -0.724312 .. 1.0472 | -1.0472 .. 0.724312 | z |
| `thumb_2` | 0 .. 2.0944 | -2.0944 .. 0 | z |
| `index_0` | -1.8326 .. 0.191986 | -0.191986 .. 1.8326 | z |
| `index_1` | -2.0944 .. 0 | 0 .. 2.0944 | z |
| `middle_0` | -1.8326 .. 0.191986 | -0.191986 .. 1.8326 | z |
| `middle_1` | -2.0944 .. 0 | 0 .. 2.0944 | z |

Do not use the other copies blindly:

- `decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand.xml` has **tighter** ranges (`thumb_2` 0..1.74533, `middle_0` -1.5708..0, `middle_1` -1.74533..0, etc.).
- `g1_supplemental_info.py:140-153` repeats those tighter limits, not the deployed ones.
- The `<motor ctrlrange>` values are **torque** limits because these are `<motor>` actuators, not position actuators. `-0.7..0.7` does not mean a 0.7 rad joint limit.

## Concrete borrow #2: depth for a one-image servo

`vision_mujoco` (MIT) has the cleanest statement of the only principled single-image depth available without a sensor: if the target's physical size is known, `Z = S_physical * f_x / S_pixels` (`estimate_depth_from_marker`, line 305). With our tangent FOV, `f_x = (W/2) / tan(angle_right)`; the same code's interaction matrix (`compute_image_jacobian`, line 222) and damped pseudo-inverse (`ibvs_control`, line 402) can be reimplemented in stdlib `math` without numpy.

The harder truth for Job Simulator: we do not know arbitrary object sizes. Two honest options:

1. **Measure the object class once** (offline, from a frame at a known depth) and then use apparent size as the depth cue. This is a data task, not a code borrow.
2. **Use a nominal depth** from the measured workspace (e.g. `x ~= 0.45 m`) and servo on pixel error. For point features, a wrong depth changes the gain and path, not the image-space equilibrium; IBVS converges to the desired pixel under modest depth error. Use the game's grab-range highlight for the final approach distance. This is the right default when the goal is placement, not metric reconstruction.

The `vision_mujoco` loop also does the thing `clank/projection.py:300` currently warns against: it recomputes features and the interaction matrix from the **current** camera pose every tick, so the camera moving with the arm is handled by construction. Its `transform_velocity_cam_to_ee` includes the lever-arm term `v_ee = v_cam + p x omega`. If we keep the existing 3D corrector, the transferable lesson is to recompute the wrist pixel from the current HMD pose each tick, or to unproject once and then close the loop in 3D; the current docstring's "call once and hold the point" is the workaround for the former, not a substitute for image-space feedback.

## Negative findings

- **`LocoManip`**: no repository at `LocoManip/LocoManip` (GitHub API 404, MEASURED). Nearest names are `isri-aist/LocomanipController` (BSD-2, C++, humanoid loco-manipulation controller, no VR/retargeting) and `linchangyi1/LocoMan` (quadruped). Neither is a source for hand placement.
- **`AnyTeleop`**: `yzqin/AnyTeleop` and `AnyTeleop/AnyTeleop` both 404 (MEASURED). Its retargeting component is now `dex-retargeting` (MIT); the arm half is not needed because SONIC already supplies the body/wrist controller.
- **`unitree_sdk2_python` / `unitree_mujoco`**: BSD-3, but the Python trees are DDS bindings and a sim bridge (`simulate_python/unitree_mujoco.py`, `unitree_sdk2py_bridge.py`) only; no IK, retargeting, or hand-closure code (MEASURED file tree, C for "no such code"). The official `xr_teleoperate` repo is the one that contains the useful G1 IK/retargeting.
- **No learned or PCA synergy** for the G1 3-finger hand was found. The only scalar mappings are the hard-coded presets in GR00T/robocasa; `g1_gripper_ik_solver.py` is a distance-threshold preset selector, not a synergy.
- **No lightweight monocular metric-depth** code for unknown objects. Depth Anything / MiDaS are torch/GPU, not metric without calibration, and not worth the dependency. ViSP is GPL-2.0 and still wants a depth value (`vpFeatureDepth`); it does not solve single-image depth. The ROS IBVS examples are ROS1 and mostly unlicensed.
- **No other Monado remote-driver client** driving a closed-source PC VR game was found. The prior art is the Monado driver itself; our relay is already the client. `sim2vr` is the closest independent prior art, but it targets Unity's OpenXR plugin and has no license.
- **No maintained, permissively licensed, bidirectional MuJoCo <-> OpenXR bridge** was found. `sim2vr`, `RoboVRTele`, and `mjxr_tests/mujoco_openxr.py` declare no license; the nishbo MuJoCo branch is render-only.
- **SteamVR null driver / VR-Emulator-Driver** is already covered in `RESEARCH.md`; it is a SteamVR/Windows path, not applicable to our Monado `remote` driver, and the relevant repos were not re-verified for license this session.

## Dependency cost, if we import rather than reimplement

- GR00T hand constants and Monado `hand_curl` plumbing: **zero** new dependencies; copy literals / one assignment.
- `vision_mujoco` math: the original needs numpy, OpenCV, MuJoCo; the interaction matrix and depth formula are a few lines of `math` and can live in `clank/projection.py` with no new dependency.
- `xr_teleoperate` arm IK: pinocchio + CasADi + IPOPT (plus meshcat/zmq for the full app). Not worth adding to a stdlib-only `clank/`; use as a reference for limits and failure behavior.
- GR00T `BodyIKSolver`: pinocchio + pink + qpsolvers. The useful part (LM damping, posture weights, `clip_configuration`) is small enough to reimplement; do not import the stack.
- `dex-retargeting`: numpy, pytransform3d, `pin` (pinocchio), nlopt, anytree, pyyaml, lxml. Not justified for a button-driven grab.
- ViSP: GPL-2.0, C++ with Python bindings, still needs depth. Avoid.

## Smallest useful change set (no new dependencies)

1. In `spike/wbc/body_view.py`, replace the binary closure handed to `sonic_executor` with a `g` in `[0,1]` and the 7-vector from Concrete borrow #1; clamp each component to the deployed XML table.
2. In `clank/wire.py:377`, stop writing the same scalar to all five `hand_curl` slots. Set index/middle/thumb from the 7-joint closure (e.g. `g` for all three) and ring/little to `0`; that is the hand-tracking-style extension Monado already supports.
3. In `clank/projection.py`, keep `unproject` but make the depth argument explicit at the call site: use the apparent-size formula when a known object size exists, otherwise a nominal workspace depth. If the servo must track while the head moves, recompute the wrist pixel from the current HMD pose each tick rather than holding the first unprojected point.
4. Do not add pinocchio/casadi/dex-retargeting. The measured joint constants, the 0.25 rad/tick clamp, and the depth formula cover the actual gap.

## Gaze + reach: hand to a 3D point with the head camera on it

The architecture already has the right seam: the game camera is the head pose we relay, and
`_targets` accepts independent head and wrist poses. The missing piece is a **look-at
quaternion** for the head; the wrist target is the same 3D point. No detector or depth is
needed when the point is already known in 3D.

| # | Source (license, URL) | File / function | What to borrow |
|---|----------------------|-----------------|----------------|
| 1 | `kevinzakka/mink` (Apache-2.0) — https://github.com/kevinzakka/mink | `src/mink/tasks/look_at_task.py`: `LookAtTask._error_and_jacobian`, `compute_error`; `examples/humanoid_apollo.py`: `make_solver`; `examples/arm_ur5e_wrist_cam_lookat.py`: `_build_tasks` | The exact two-DOF look-at residual/Jacobian. It constrains only the line of sight and leaves roll free. Combine a `LookAtTask` on the head-camera site with a `FrameTask` on the wrist to the same point. Also lists G1 support. |
| 2 | `Rhoban/placo` (MIT) — https://github.com/Rhoban/placo | `src/placo/kinematics/axis_align_task.cpp`: `AxisAlignTask::fill`; docs `add_axisalign_task`; `placo-examples` humanoid ball example | Same idea in a QP: align the camera axis to `targetAxis_world`, leaving rotation about that axis free. Pair with `add_position_task`/`add_frame_task` for the wrist. Heavier C++/Python dependency than #1. |
| 3 | `pollen-robotics/grabette` (Apache-2.0) — https://github.com/pollen-robotics/grabette | `integrations/openarm/openarm_gripette_simu/examples/collect_reach_dataset.py`: `look_at_rotation`, `compute_reach_targets`, `LOOK_AT_STRENGTH` | A ~15-line, dependency-free look-at rotation with a world-up fallback, plus a "partially look at the target while reaching" blend. The camera is wrist-mounted, so port the math, not the architecture. |

**Local seams.** `spike/wbc/sonic_executor.py:341` `look_target(yaw_rad)` only yaws; it has no
pitch. `_targets` at `sonic_executor.py:190` already writes the head and both wrists as
independent `vr_pos`/`vr_orn` targets. `clank/projection.py:209` `project_point`, `:255`
`unproject`, and `:300` `servo_step` are the pixel-based path if the point starts as a pixel;
`servo_step`'s docstring already warns that the head moves during a reach.

**Minimal implementation.** Build a roll-fixed look-at quaternion from the head position `p`
to the target `T`: `fwd = normalize(T - p)`, `left = normalize(world_up x fwd)`,
`up = fwd x left`, `R = [fwd left up]`. That is the +X-forward head convention already
documented in `clank/projection.py`. Feed it as the head pose and `T` as the wrist target.
`mink.LookAtTask` intentionally leaves roll free, so if you borrow its residual rather than
this matrix, add an upright/roll task or the head can spin about its gaze axis. Recompute the
look-at from the **achieved** head pose each tick (the way `_head_twist` does) so the target
stays centred when the arm shifts the head; freeze the target point, not the head pose.

**Negative findings.** `vision_mujoco`, `hello-robot/stretch_visual_servoing`, and the
`ferromotion` example are eye-in-hand or fixed-camera: they centre the target in a **wrist**
camera while the hand approaches, not in the head camera. `TAVIS`, `AVR`, `GIAVA`, and
`AV-ALOHA` are IsaacLab/ROS/ALOHA training stacks; they need policy retraining and do not fit
the stdlib relay. `OpenRAL/rskill-moveit-look-at` implements the same roll-free gaze pose but
needs ROS 2/MoveIt/TF2, so it is out of scope. `mink` and `placo` are full IK/QP solvers; our
learned policy already accepts Cartesian head/wrist targets, so borrow the look-at math, not a
second control path, unless you deliberately want to bypass the policy.

## Chinese-language sources checked (follow-up)

The first pass relied on Exa, which returned mostly English/global pages. A second pass with
the local trawl engine (ddg) and GitHub Chinese queries (`视觉伺服`, `视觉伺服 机械臂`,
`头部 云台`, `注视 抓取`) returned genuinely Chinese pages and repos. The Chinese ecosystem
is rich in visual servoing and gimbal tracking, but still has no off-the-shelf "head-camera
look-at plus hand reach" controller. The relevant code:

- `dreamybroomxj-coder/ForceSight` (MIT) — Chinese; `src/gaze/gaze_tracker.py`
  `get_gaze_on_table`, `src/gaze/_geometry.py` `ray_intersect_plane`,
  `src/Yolo/table_lookup.py`, `main.py` `_execute_grasp`. Gaze -> table 3D point -> Piper
  arm grasp. Fixed external camera and user gaze, not head-camera centring, but the closest
  Chinese MIT code for "target a 3D point from gaze and reach".
- `GalaxyGeneralRobotics/OpenWBT` (Apache-2.0) — Chinese README; G1/H1 whole-body VR teleop,
  `deploy/teleop/robot_control/robot_arm_ik.py`, `deploy/helpers/ik_server.py`,
  `deploy/teleop/open_television/tv_wrapper.py`. G1-specific and MuJoCo-runnable, but it
  follows VR head/arm poses; it has no autonomous look-at. Closest Chinese G1 reference.
- `QiongMing-Intelligent/VISTA-Grasp` (Apache-2.0, Gitee) — WebXR head-pose + dual-arm IK
  (Pinocchio/CasADi) over ROS 1. Same limitation: operator-driven head, no gaze controller.
- `zc-xzc/robot_platform` (MIT) — PICO 4 head tracking -> 2-axis gimbal -> D415, PD yaw/pitch.
  A direct example of keeping a head camera on the operator's view, but it moves a gimbal,
  not an arm.
- `whx-prog/teleoperate-end_pose` (no license declared) — Quest -> Limx Oli, head pitch
  follow + arms. Do not copy; no license.
- `XuHaohui/RoboGrasp-Vision` (no license) and `2993845045zhy/ur3-visual-servoing`
  (NOASSERTION) — Chinese ROS 2/MuJoCo visual grasping and wrist-camera PBVS/IBVS. Do not
  copy.
- `kaka12331/2023-dian-sai-visual-gimbal` (no license) — two-axis gimbal visual servoing
  with STM32/K230; directly relevant to "keep the target centred", but no license.
- `haoyu-x/vision-in-action` (MIT) — active-perception neck and data pipeline; a training
  stack, not a drop-in controller.
- `HEAD` (hand-eye autonomous delivery, arXiv 2508.03068) — conceptually exact: a high-level
  planner commands eye + left/right hand targets, and a whole-body policy tracks them. I did
  not find a linked public code repo, so prior art only.

Most other Chinese results (`HuaYuXiao/Vision-Servo-PID-Control`,
`only-one-over/Dobot-VGS-RS400`, `XieMingyou/VisualServo`,
`polyethylene16/visual-motion-control-of-robotic-arms`) are eye-to-hand or wrist-camera
visual servoing. Local trawl also surfaced Chinese tutorials and forums (CSDN, Zhihu, BAAI,
Wanfang, 51CTO, fishros) on visual grasping and gimbal servoing: they confirm the pattern but
are not licensed code. No Chinese repo found combines head-camera look-at with hand reach in
the way we need. This does not change the recommendation above: `mink` (or `placo`) for the
look-at math, `grabette` for the direct quaternion.
