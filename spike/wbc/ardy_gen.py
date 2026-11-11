#!/usr/bin/env python3
"""State constraints, get motion: the ARDY-G1 adapter.

Everything below is a thin statement of intent over ARDY's own generation path. A reach is
`right hand at this world point` and/or `root walks this far`, not a hand-built pose for a
generator to interpolate towards - which is the reason this module exists instead of a mink
still frame (`research/motion-generators.md` §2, §9).

The call is the one `spike/ardy/probe_g1.py` measured, in the same order:

    build a RightHandConstraintSet from the latest achieved pose (CPU tensors)
    -> motion_rep.create_conditions_from_constraints_batched
    -> model.autoregressive_step  (init_global_translation / init_first_heading_angle)
    -> motion_rep.inverse        -> MujocoQposConverter.dict_to_qpos  => our (1, T, 36)

Three traps are encoded here because each one is silent or takes the GPU with it:

1. With `cfg_type="separated"` a **scalar** `cfg_weight` sets the constraint guidance weight
   to zero (`ardy_model.py:274-279`), so a constrained run under the default ignores its
   constraints - measured 870 mm of hand error instead of 60. `CFG_WEIGHT` is a pair.
2. `num_denoising_steps` above the checkpoint's `num_base_steps` trips a CUDA device-side
   assert. `check_steps` refuses in Python first.
3. Constraint tensors are assembled on the **CPU**: `create_pairs` (`constraints.py:40`)
   stacks frame indices with joint indices, so mixing a CPU index with a CUDA one raises.

Continuity is deliberately left out of v1. ARDY continues from `init_history_sequence`, which
the demo fills from its own previously generated *feature* tensor; a pose36 -> features
encoder does not exist (`motion_rep` has no `encode`), so `generate` starts from the body's
current root and heading instead (`init_global_translation`, `init_first_heading_angle`) and
lets SONIC absorb the pose offset, since it starts from the live `mjData` regardless.
`generate` therefore refuses `history_features` rather than accepting a tensor it would
misread: ARDY returns `init_history_len + gen_horizon_len` frames, while the hand pin and the
residual are indexed at `num_frames - 1`, which is then the last *history* frame - so a
history would silently measure and pin the wrong frame instead of erroring.

NixOS: torch needs `LD_LIBRARY_PATH=/run/opengl-driver/lib`, or it reports `cuda False` on a
working driver and everything runs on the CPU instead.

torch and ARDY are imported **inside** the calls that need them, so `import ardy_gen` is
cheap and `tests/test_ardy_gen.py` can hold the pure parts with the standard library alone.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: Where the checkpoint lives (`config.yaml` + `denoiser.safetensors` + `tokenizer.safetensors`).
ARDY_CKPT_DIR = Path.home() / ".local/state/Clank/ardy"
ARDY_MODEL = "ARDY-G1-RP-25FPS-Horizon52"
ARDY_FPS = 25
ARDY_HORIZON = 52
ARDY_TEXT_DIM = 4096

#: (text, constraint) guidance. The scalar form zeroes the constraint weight.
CFG_WEIGHT = (1.0, 2.0)

#: A reach in front of the chest, relative to where the hand already is.
HAND_DELTA_FWD = 0.35
HAND_DELTA_UP = 0.15

#: Floats in one pose: root xyz + root quat xyzw + 29 hinge angles (`g1_pack`).
POSE_LEN = 36

#: MuJoCo world (x forward, y left, z up) -> ARDY world (x left, y up, z forward): `ardy = M @ mujoco`.
MUJOCO_TO_ARDY = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))

_MODELS: dict[str, Any] = {}
_CONVERTERS: dict[int, Any] = {}
_SKELETON: Any = None


# --------------------------------------------------------------------------- pure parts


def hand_delta(forward_m: float, up_m: float) -> tuple[float, float, float]:
    """A hand displacement in ARDY space from a forward and an up distance, in metres."""
    return (0.0, float(up_m), float(forward_m))


def check_steps(steps: int, num_base_steps: int) -> int:
    """The denoising step count, or `ValueError` naming the checkpoint's limit."""
    limit = int(num_base_steps)
    count = int(steps)
    if count < 1 or count > limit:
        raise ValueError(
            f"num_denoising_steps must be between 1 and {limit}, the checkpoint's denoising "
            f"schedule; {count} would trip a CUDA device-side assert rather than an error"
        )
    return count


def check_num_frames(
    num_frames: int, gen_horizon_len: int, num_frames_per_token: int
) -> int:
    """The window length, or `ValueError` naming the checkpoint's horizon and token size.

    ARDY returns `gen_horizon_len` frames for a window whatever `num_frames` says, and asserts
    a whole number of tokens inside `autoregressive_step` (`ardy_model.py:736`) while building
    its future mask from the token count. Any other frame count is therefore either an
    `AssertionError` from deep inside the model or a full-horizon clip with the hand pinned at
    the wrong frame, so it is refused here first.
    """
    horizon = int(gen_horizon_len)
    per_token = int(num_frames_per_token)
    count = int(num_frames)
    if count != horizon:
        raise ValueError(
            f"num_frames must be the checkpoint's generated horizon, {horizon}: ARDY always "
            f"returns {horizon} frames for a window, so {count} would pin the hand at frame "
            f"{count - 1} of a {horizon}-frame clip"
        )
    if count % per_token != 0:
        raise ValueError(
            f"num_frames must be a multiple of {per_token}, the checkpoint's frames per token; "
            f"autoregressive_step asserts that and builds its future mask from the token count"
        )
    return count


def quat_matrix(quat_xyzw: Sequence[float]) -> tuple[tuple[float, float, float], ...]:
    """Rotation matrix of an `xyzw` quaternion, in the same convention as the pose."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def heading_angle(pose36: Sequence[float]) -> float:
    """The body's yaw in ARDY's convention: 0 faces ARDY +z, which is MuJoCo +x.

    The two world frames are the same rotation about the vertical axis (MuJoCo z becomes ARDY
    y), so this is the pose's ordinary yaw and no conversion is needed - which is what keeps
    `init_first_heading_angle` in the same frame as the constraints.
    """
    rot = quat_matrix(pose36[3:7])
    return math.atan2(rot[1][0], rot[0][0])


def rotation_angle_deg(
    first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]
) -> float:
    """The angle, in degrees, that `second` has to turn to become `first`.

    The rotation term of the FK round trip. A joint *anchor* does not move when the joint spins
    about its own axis, so a wrong axis or sign on a leaf joint (`*_wrist_yaw`, `*_ankle_roll`)
    is invisible to a position-only comparison - and `hand_constraint` pins the hand's global
    rotation, so such a bug would bias every reach. `trace(first^T @ second)` is the cosine, and
    the clamp keeps a matrix that is only orthonormal to float precision from raising.
    """
    cosine = (math.fsum(first[i][k] * second[i][k] for i in range(3) for k in range(3)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def world_root(pose36: Sequence[float]) -> tuple[float, float, float]:
    """The root in ARDY world metres, height zeroed.

    `init_global_translation` asserts a zero y: ARDY carries the root's height in its own
    features, and only the planar position is ours to place.
    """
    return (float(pose36[1]), 0.0, float(pose36[0]))


def root_waypoints(
    num_frames: int,
    forward_m: float,
    *,
    start_xy: tuple[float, float] = (0.0, 0.0),
    heading: float = 0.0,
) -> tuple[tuple[int, ...], tuple[tuple[float, float], ...]]:
    """A straight run of `forward_m`, as (frames, ARDY world xz points) for a root path.

    Three waypoints - first, middle and last frame of the window - are the sparsest path that
    still fixes a direction, and they are what `spike/ardy/probe_g1.py` used. `forward` is the
    body's own forward at `heading`, not world +z, because the object a reach is stepping
    towards is in front of the body.
    """
    last = int(num_frames) - 1
    if last < 1:
        raise ValueError(f"a root path needs at least two frames, got {num_frames}")
    distance = float(forward_m)
    dx, dz = math.sin(float(heading)), math.cos(float(heading))
    x0, z0 = float(start_xy[0]), float(start_xy[1])
    frames = (0, int(num_frames) // 2, last)
    points = tuple((x0 + dx * distance * t, z0 + dz * distance * t) for t in (0.0, 0.5, 1.0))
    return frames, points


def _rows(pose36_context: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    """Validate the context ring and hand it back as plain rows."""
    rows = tuple(tuple(float(v) for v in row) for row in pose36_context)
    if not rows:
        raise ValueError("pose36 context is empty: the body has not ticked yet")
    for row in rows:
        if len(row) != POSE_LEN:
            raise ValueError(f"each context pose must be {POSE_LEN} floats, got {len(row)}")
    return rows


# ------------------------------------------------------------------ lazy heavy imports


def _skeleton() -> Any:
    """The 34-joint G1 skeleton this checkpoint was trained on (a few kB, no weights)."""
    global _SKELETON
    if _SKELETON is None:
        from ardy.skeleton import G1Skeleton34  # pyright: ignore[reportMissingImports]

        _SKELETON = G1Skeleton34()
    return _SKELETON


def _converter(skeleton) -> Any:
    """`MujocoQposConverter` for `skeleton`, built once: it parses the model XML."""
    from ardy.exports.mujoco import MujocoQposConverter  # pyright: ignore[reportMissingImports]

    key = id(skeleton)
    converter = _CONVERTERS.get(key)
    if converter is None:
        converter = _CONVERTERS[key] = MujocoQposConverter(skeleton)
    return converter


def load_ardy(device: str = "cuda") -> Any:
    """The checkpoint's denoiser on `device`, without the text tower, loaded once per device.

    Text is a single pooled 4096-vector per sample in this checkpoint, and an empty prompt is
    already turned into zeros (`ardy_model.py:325`), so the tower buys nothing here.
    """
    from ardy.model import load_model  # pyright: ignore[reportMissingImports]

    model = _MODELS.get(device)
    if model is None:
        model = load_model(
            modelname=ARDY_MODEL,
            device=device,
            text_encoder=False,
            checkpoints_dir=str(ARDY_CKPT_DIR),
        )
        model.eval()
        _MODELS[device] = model
    return model


def zero_text(device: str):
    """Zero text features: one pooled 4096-d vector, which is what an empty prompt becomes."""
    import torch  # pyright: ignore[reportMissingImports]

    return torch.zeros(1, 1, ARDY_TEXT_DIM, device=device)


def _motion_pose(pose36: Sequence[float], skeleton, converter) -> dict:
    """Forward kinematics of one pose36 into ARDY's motion dict, on the CPU.

    ARDY only converts its own space to qpos, never back, so the inverse is reassembled from
    the exporter's own precomputed tables: a hinge's local rotation in ARDY space is
    `offset_j @ R(axis_j, angle_j)`, where `offset_j` is the body's fixed quaternion offset
    (`_rot_offsets_f2q` holds its inverse, because the exporter applies it leftwards) and
    `axis_j` is the hinge axis already mapped into ARDY space. Checked against ARDY's own
    `local_rot_mats` on generated motion: 0.7 mm mean joint error, 1-2 mm at either hand.
    """
    import torch  # pyright: ignore[reportMissingImports]
    from ardy.geometry import axis_angle_to_matrix  # pyright: ignore[reportMissingImports]
    from ardy.skeleton.kinematics import fk  # pyright: ignore[reportMissingImports]

    mujoco_to_ardy = torch.tensor(MUJOCO_TO_ARDY, dtype=torch.float32)
    local = torch.eye(3, dtype=torch.float32).repeat(1, 1, skeleton.nbjoints, 1, 1)
    local[0, 0, 0] = mujoco_to_ardy @ torch.tensor(
        quat_matrix(pose36[3:7]), dtype=torch.float32
    ) @ mujoco_to_ardy.T
    for hinge, joint_index in enumerate(converter._mujoco_indices_to_ardy_indices.tolist()):
        axis = converter._mujoco_joint_axis_values_ardy_space[hinge].to(torch.float32)
        axis_angle = (axis * float(pose36[7 + hinge])).reshape(1, 3)
        offset = converter._rot_offsets_f2q[joint_index].to(torch.float32).T
        local[0, 0, joint_index] = offset @ axis_angle_to_matrix(axis_angle)[0]
    # The full root, height included: the constraint has to know how high the body is holding
    # its hand. Only `init_global_translation` wants the height zeroed, and it uses `world_root`.
    x_m, y_m, z_m = (float(v) for v in pose36[:3])
    root = torch.tensor([[[y_m, z_m, x_m]]], dtype=torch.float32)  # [batch, time, xyz]
    global_rots, posed_joints, _ = fk(local, root, skeleton)
    return {
        "local_rot_mats": local,
        "global_rot_mats": global_rots,
        "posed_joints": posed_joints,
        "root_positions": root,
    }


def hand_constraint(
    skeleton,
    pose: dict,
    hand_delta: Sequence[float],
    *,
    frame: int | None = None,
    root_2d=None,
) -> list:
    """The right hand on `pose`'s hand plus `hand_delta`, at `frame` (default: `pose`'s last).

    `RightHandConstraintSet` takes whole-body arrays and uses only the named joints, so the
    pose this starts from has to be a full one - the achieved pose, not a hand point.
    Everything is CPU (`create_pairs` stacks frame indices with joint indices; trap 3).
    """
    import torch  # pyright: ignore[reportMissingImports]
    from ardy.constraints import RightHandConstraintSet  # pyright: ignore[reportMissingImports]

    source = pose["posed_joints"].shape[1] - 1
    index = skeleton.bone_index
    positions = pose["posed_joints"].cpu()[:, source : source + 1].clone()
    rotations = pose["global_rot_mats"].cpu()[:, source : source + 1].clone()
    offset = torch.tensor(hand_delta, dtype=positions.dtype)
    for name in skeleton.right_hand_joint_names:
        positions[:, :, index[name]] += offset
    return [
        RightHandConstraintSet(
            skeleton=skeleton,
            frame_indices=torch.tensor([int(source if frame is None else frame)]),
            global_joints_positions=positions.squeeze(0),
            global_joints_rots=rotations.squeeze(0),
            root_2d=root_2d,
        )
    ]


def hand_target(skeleton, pose: dict, hand_delta: Sequence[float]) -> tuple[float, float, float]:
    """The world point `hand_constraint` pins: `pose`'s right hand plus `hand_delta`."""
    hand = skeleton.right_hand_joint_names[0]
    x, y, z = pose["posed_joints"][0, -1, skeleton.bone_index[hand]].tolist()
    dx, dy, dz = (float(value) for value in hand_delta)
    return (x + dx, y + dy, z + dz)


def root_goal_constraint(
    skeleton,
    num_frames: int,
    forward_m: float,
    *,
    start_xy: tuple[float, float] = (0.0, 0.0),
    heading: float = 0.0,
) -> list:
    """A `root2d` path starting at `start_xy` and running `forward_m` along `heading`."""
    import torch  # pyright: ignore[reportMissingImports]
    from ardy.constraints import Root2DConstraintSet  # pyright: ignore[reportMissingImports]

    frames, points = root_waypoints(
        num_frames, forward_m, start_xy=start_xy, heading=heading
    )
    return [
        Root2DConstraintSet(
            skeleton=skeleton,
            frame_indices=torch.tensor(frames),
            root_2d=torch.tensor(points, dtype=torch.float32),
        )
    ]


def hand_position(pose36: Sequence[float], hand: str = "right") -> tuple[float, float, float]:
    """Where a pose36 puts one hand, in ARDY world metres.

    This is how a caller turns an object in the game's world into a `hand_delta`: subtract it
    from the object's point (after the same MuJoCo -> ARDY axis swap, `MUJOCO_TO_ARDY`).
    """
    skeleton = _skeleton()
    motion = _motion_pose(pose36, skeleton, _converter(skeleton))
    if hand == "right":
        names = skeleton.right_hand_joint_names
    elif hand == "left":
        names = skeleton.left_hand_joint_names
    else:
        raise ValueError(f"hand must be 'left' or 'right', got {hand!r}")
    x, y, z = motion["posed_joints"][0, -1, skeleton.bone_index[names[0]]].tolist()
    return (x, y, z)


def generate(
    pose36_context: Sequence[Sequence[float]],
    *,
    hand_delta: Sequence[float],
    root_forward_m: float | None,
    num_frames: int = ARDY_HORIZON,
    steps: int = 10,
    history_features=None,
) -> tuple[list[tuple[float, ...]], dict]:
    """Generate `num_frames` of 25 Hz pose36 motion from the latest achieved pose in the context.

    The context is the body's recent achieved ring (`mb_mailbox`, oldest first): its last pose
    is where the hand target starts from, which root the window is generated around, and which
    way the body faces. `root_forward_m` adds a root path - the out-of-reach case, where the
    body has to step towards the object as well as raise its hand.

    `num_frames` must be the checkpoint's own horizon (`check_num_frames`): ARDY generates
    that many frames regardless, so a different count would pin the hand at the wrong one.

    Returns `(frames, metrics)`, the frames being pose36 rows (`g1_pack` order, 25 Hz) and the
    metrics `{"seconds", "peak_vram_gb", "hand_residual_m", "fk_roundtrip_m",
    "fk_roundtrip_deg"}`. Every number is measured, not assumed, so a caller can decide whether
    another generation is worth it.
    """
    context = _rows(pose36_context)
    if history_features is not None:
        raise NotImplementedError(
            "history_features is not supported yet: ARDY continues from its own feature tensor "
            "and the pose36 -> feature encoder does not exist (motion_rep has no encode); pass "
            "None and let SONIC absorb the pose offset"
        )

    import torch  # pyright: ignore[reportMissingImports]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_ardy(device)
    count = check_steps(steps, model.diffusion.num_base_steps)
    horizon = check_num_frames(
        num_frames, model.gen_horizon_len, model.num_frames_per_token
    )
    skeleton = _skeleton()
    converter = _converter(skeleton)

    latest = context[-1]
    pose = _motion_pose(latest, skeleton, converter)
    start_xy = (float(pose["root_positions"][0, 0, 0]), float(pose["root_positions"][0, 0, 2]))
    heading = heading_angle(latest)
    last_frame = horizon - 1

    # The hand constraint pins the root at its own frame from the pose it was built on; when a
    # root path exists the two must agree or the path is ignored (measured: 0.03 m travelled
    # with two disagreeing root pins, 0.57 m when the hand carries the same waypoint).
    root_2d = None
    goals: list = []
    if root_forward_m is not None:
        _, points = root_waypoints(
            horizon, root_forward_m, start_xy=start_xy, heading=heading
        )
        root_2d = torch.tensor([points[-1]], dtype=torch.float32)
        goals = root_goal_constraint(
            skeleton, horizon, root_forward_m, start_xy=start_xy, heading=heading
        )
    constraints = hand_constraint(skeleton, pose, hand_delta, frame=last_frame, root_2d=root_2d)
    target = hand_target(skeleton, pose, hand_delta)

    lengths = torch.tensor([horizon], device=device)
    observed_motion, motion_mask = model.motion_rep.create_conditions_from_constraints_batched(
        constraints + goals, lengths, to_normalize=True, device=device
    )
    text_feat = zero_text(device)
    text_pad_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
    # No history: the window is generated around the body's own root and heading, so the
    # exported qpos is already in the game's world frame and the tracker has no jump.
    init_translation = torch.tensor([world_root(latest)], dtype=torch.float32, device=device)
    init_heading = torch.tensor([heading], dtype=torch.float32, device=device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        motion = model.autoregressive_step(
            num_frames=horizon,
            num_denoising_steps=count,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
            cfg_weight=CFG_WEIGHT,
            texts=None,
            text_feat=text_feat,
            text_pad_mask=text_pad_mask,
            init_history_sequence=history_features,
            init_global_translation=init_translation,
            init_first_heading_angle=init_heading,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        output = model.motion_rep.inverse(motion, is_normalized=True)
        as_tensors = {key: value for key, value in output.items() if torch.is_tensor(value)}
        qpos = converter.dict_to_qpos(
            as_tensors, device, root_quat_w_first=False, numpy=False
        )
        # `_motion_pose` builds the hand target the residual below is measured against, so a
        # systematic error in its FK cancels out of that residual. Round-tripping the last
        # generated frame back through it against ARDY's own posed joints is what catches a
        # wrong joint order, hinge axis or rotation offset - any of those biases every reach
        # while the residual still looks small. Most of the number that comes out is ARDY's
        # own export loss (see the probe), so it catches gross errors, not small ones.
        our_pose = _motion_pose(tuple(qpos[0, last_frame].tolist()), skeleton, converter)
        our_joints = our_pose["posed_joints"][0, 0]
        ardy_joints = output["posed_joints"][0, last_frame].to(
            device=our_joints.device, dtype=our_joints.dtype
        )
        fk_roundtrip = float(torch.linalg.norm(our_joints - ardy_joints, dim=-1).max())
        # Position alone cannot see a leaf joint turned about the wrong axis: its anchor does
        # not move. The rotation can, and it is what `hand_constraint` pins.
        fk_roundtrip_deg = max(
            rotation_angle_deg(mine, theirs)
            for mine, theirs in zip(
                our_pose["global_rot_mats"][0, 0].tolist(),
                output["global_rot_mats"][0, last_frame].tolist(),
                strict=True,
            )
        )

    hand = skeleton.bone_index[skeleton.right_hand_joint_names[0]]
    landed = output["posed_joints"][0, last_frame, hand]
    residual = float(
        torch.linalg.norm(landed - torch.tensor(target, device=landed.device, dtype=landed.dtype))
    )
    frames = [tuple(float(v) for v in row) for row in qpos[0].tolist()]
    metrics = {
        "seconds": seconds,
        "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else None,
        "hand_residual_m": residual,
        "fk_roundtrip_m": fk_roundtrip,
        "fk_roundtrip_deg": fk_roundtrip_deg,
    }
    return frames, metrics
