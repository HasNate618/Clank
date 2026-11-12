#!/usr/bin/env python3
"""Pack and unpack the 36-float MotionBricks pose against the deployed G1 model.

MotionBricks speaks `[root_x, root_y, root_z, qx, qy, qz, qw, 29 hinge angles]`. Nothing
else in the stack speaks that vector, and the three differences are all silent:

* The deployed `g1_29dof_with_hand.xml` puts a 7-joint Dex3 hand *between* the arms, so the
  29 body joints are not `qpos[7:36]`. `QPOS_QADR` is the real list: 22 contiguous body
  joints, then the right arm after the left hand.
* MuJoCo stores the free-joint quaternion `wxyz`; MotionBricks stores it `xyzw`.
* SONIC's `jp` arrays are in policy/CSV order, an interleaved left/right permutation of the
  same hinges (`CSV2MJ` in `sonic_runner.py`).

Standard library only: `tests/` imports this module and must stay stdlib.
"""

from __future__ import annotations

from collections.abc import Sequence

#: The 29 hinges in hardware order, also `SonicRunner.qadr` order. Fingers are excluded;
#: they stay on the clasp channel.
JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

#: policy/CSV joint index -> hardware joint index (`sonic_runner.py:CSV2MJ`).
CSV2MJ: tuple[int, ...] = (
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11,
    17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
)

#: qpos address of each of the 29 hinges: left leg, right leg, waist and left arm in
#: qpos[7:29], then the right arm in qpos[36:43] after the left hand.
QPOS_QADR: tuple[int, ...] = tuple(range(7, 29)) + tuple(range(36, 43))

#: The two Dex3 hands in qpos, left then right. Poses never read or write these.
HAND_QPOS = (slice(29, 36), slice(43, 50))

#: `g1_29dof_with_hand.xml` qpos length: free base 7 + 29 hinges + 14 finger joints.
QPOS_LEN = 50


def wxyz_to_xyzw(q: Sequence[float]) -> tuple[float, float, float, float]:
    """MuJoCo `qpos[3:7]` order to MotionBricks order."""
    return (q[1], q[2], q[3], q[0])


def xyzw_to_wxyz(q: Sequence[float]) -> tuple[float, float, float, float]:
    """MotionBricks order to MuJoCo `qpos[3:7]` order."""
    return (q[3], q[0], q[1], q[2])


def pose36_from_qpos(qpos: Sequence[float]) -> tuple[float, ...]:
    """Read a 36-float pose out of a full 50-value model `qpos`."""
    if len(qpos) != QPOS_LEN:
        raise ValueError(f"qpos must be {QPOS_LEN} values, got {len(qpos)}")
    return (qpos[0], qpos[1], qpos[2]) + wxyz_to_xyzw(qpos[3:7]) + tuple(
        qpos[a] for a in QPOS_QADR
    )


def apply_pose36(qpos: Sequence[float], pose: Sequence[float]) -> list[float]:
    """Write a 36-float pose into a copy of `qpos`; the hands keep their values."""
    if len(qpos) != QPOS_LEN:
        raise ValueError(f"qpos must be {QPOS_LEN} values, got {len(qpos)}")
    if len(pose) != 36:
        raise ValueError(f"pose must be 36 values, got {len(pose)}")
    out = list(qpos)
    out[0:3] = pose[0:3]
    out[3:7] = xyzw_to_wxyz(pose[3:7])
    for i, adr in enumerate(QPOS_QADR):
        out[adr] = pose[7 + i]
    return out


def jp_from_pose36(pose: Sequence[float]) -> tuple[float, ...]:
    """Reorder a pose's hinges into SONIC's policy/CSV order."""
    if len(pose) != 36:
        raise ValueError(f"pose must be 36 values, got {len(pose)}")
    return tuple(pose[7 + CSV2MJ[i]] for i in range(len(CSV2MJ)))


def pose36_from_jp(
    root_xyz: Sequence[float],
    root_xyzw: Sequence[float],
    jp: Sequence[float],
) -> tuple[float, ...]:
    """Assemble a pose from a root and a policy/CSV-order hinge vector."""
    if len(root_xyz) != 3:
        raise ValueError(f"root_xyz must be 3 values, got {len(root_xyz)}")
    if len(root_xyzw) != 4:
        raise ValueError(f"root_xyzw must be 4 values, got {len(root_xyzw)}")
    if len(jp) != len(CSV2MJ):
        raise ValueError(f"jp must be {len(CSV2MJ)} values, got {len(jp)}")
    hinges = [0.0] * len(CSV2MJ)
    for csv_i, hardware_i in enumerate(CSV2MJ):
        hinges[hardware_i] = jp[csv_i]
    return tuple(root_xyz) + tuple(root_xyzw) + tuple(hinges)


def context4_from_ring(
    ring: Sequence[Sequence[float]], hz: float = 50.0
) -> tuple[tuple[float, ...], ...]:
    """Pick MotionBricks' four context poses, oldest first, ending at the latest pose.

    The ring runs at the executor's rate `hz`; MotionBricks wants 30 Hz, so the stride is
    `round(hz / 30)`. A ring shorter than three strides back repeats its oldest pose rather
    than inventing one, which is also what makes a one-pose ring four copies of itself.
    """
    if not ring:
        raise ValueError("context ring is empty")
    stride = max(1, round(hz / 30.0))
    last = len(ring) - 1
    return tuple(
        tuple(ring[max(0, last - k * stride)]) for k in (3, 2, 1, 0)
    )
