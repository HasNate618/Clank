# pyright: reportAttributeAccessIssue=false
# mujoco ships stubs that omit MjData/mj_kinematics/mj_jacBody; the same report appears across
# every file in spike/wbc/ and is a stub gap, not a defect. Silenced here only.
"""Damped-least-squares IK for the 17 upper-body joints, in the deploy's override order.

The VR wrist-pose path has no velocity and no look-ahead (`observation_config.md:225-246`), so a
Cartesian step is a setpoint the policy can only react to one tick late; measured here that rings
for 0.77-1.05 s with +75 % overshoot. NVIDIA's own deploy never asks the policy to do that: it
substitutes 17 upper-body joint angles into the *reference motion* every look-ahead frame
(`upper_body_position[17]`, `GatherMotionJointPositionsMultiFrame`, `g1_deploy_onnx_ref.cpp:738`),
and the legs stay on the policy. This module produces those 17 numbers from a wrist point.

Order, which is the part that is easy to get wrong. There are three, and they are all different:

* **policy / CSV order** - `motion["jp"]`, interleaved left/right. `UPPER_BODY_ISAACLAB_IDX` in
  `sonic_runner` indexes this one.
* **hardware order** - `runner.qadr`, grouped: legs, waist, left arm, right arm.
  `UPPER_BODY_MUJOCO_IDX` indexes this one.
* **the 17-vector** - position `i` of the override payload is the joint that is
  `UPPER_BODY_ISAACLAB_IDX[i]` in policy order and `UPPER_BODY_MUJOCO_IDX[i]` in hardware order.
  Verified against `CSV2MJ` for all 17 (`CSV2MJ[11] == 15`, `[12] == 22`, `[15] == 16`, ...), so
  the two arrays really are the same joints in two orders.

Because the payload interleaves the arms (left at payload positions 3,5,7,...), never index it by
hand - `payload_slots()` derives the positions from the index vector.
"""

from __future__ import annotations

import mujoco  # pyright: ignore[reportMissingImports]
import numpy as np
import sonic_runner as sr

#: Hardware indices (the `runner.qadr` order) of the two 7-joint arms.
ARM_HW = {"left": list(range(15, 22)), "right": list(range(22, 29))}

#: The link the VR point is measured from. `_achieved` adds `VR_OFFSETS[side]` rotated by this
#: body's rotation, so aiming this body is aiming the reported point minus a rotating lever.
WRIST_BODY = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}


def payload_slots(side: str) -> list[int]:
    """Which positions of the 17-vector belong to `side`'s arm."""
    arm = set(ARM_HW[side])
    return [i for i, hw in enumerate(sr.UPPER_BODY_MUJOCO_IDX) if hw in arm]


def payload(runner, qpos: np.ndarray) -> np.ndarray:
    """The 17-vector the override wants, read out of a full `qpos`."""
    return np.array(
        [qpos[runner.qadr[hw]] for hw in sr.UPPER_BODY_MUJOCO_IDX], dtype=np.float64
    )


def vr_point(runner, side: str) -> np.ndarray:
    """Where the game draws this hand, in the pelvis frame - the quantity `reach_vr` corrects.

    Kept in step with `SonicExecutor._achieved`: body position plus a lever rotated by the body's
    own rotation, then expressed in the root frame.
    """
    d = runner.data
    base = d.qpos[0:3].copy()
    root_inv = sr.quat_conj(d.qpos[3:7])
    bid = runner.model.body(WRIST_BODY[side]).id
    offset = sr.VR_OFFSETS[0 if side == "left" else 1]
    pos = d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ offset - base
    return sr.quat_rotate(root_inv, pos)


def _joint_ids(model, runner, side: str) -> tuple[list[int], list[int]]:
    """(qpos addresses, dof addresses) of one arm, in hardware order."""
    qadr, dofs = [], []
    for hw in ARM_HW[side]:
        target = runner.qadr[hw]
        for jid in range(model.njnt):
            if model.jnt_qposadr[jid] == target:
                qadr.append(target)
                dofs.append(model.jnt_dofadr[jid])
                break
        else:
            raise RuntimeError(f"no joint at qpos address {target} for hardware index {hw}")
    return qadr, dofs


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
    )


def solve(
    runner,
    side: str,
    target_rel: np.ndarray,
    iters: int = 40,
    step: float = 0.35,
    tol: float = 0.002,
) -> tuple[np.ndarray, float]:
    """Solve one arm so the *reported* hand point lands on `target_rel`.

    `target_rel` is in the pelvis frame, which is what a depth unprojection and every reach so far
    has used. Returns the full 17-vector ready for `runner.upper_body`, plus the position residual
    in metres at the last iteration.

    The wrist orientation is left free. The 7 arm joints give 3 position constraints, so the solve
    is underdetermined and damping picks the smallest joint motion that does the job; constraining
    orientation as well was measured to matter less than the lever it adds.

    Damping is Levenberg-Marquardt with backtracking, not a fixed lambda. Fixed damping on a
    nearly singular arm Jacobian produced 1.5 rad steps and the residual walked
    235 -> 147 -> 339 -> 116 -> 158 mm, i.e. the solver oscillated instead of converging.
    """
    m, d = runner.model, runner.data
    bid = m.body(WRIST_BODY[side]).id
    arm_qadr, dofs = _joint_ids(m, runner, side)
    dofs_arr = np.array(dofs)

    ranges = np.array([m.jnt_range[m.dof_jntid[dd]] for dd in dofs])
    lo, hi = ranges[:, 0] + 0.02, ranges[:, 1] - 0.02

    scratch = mujoco.MjData(m)
    cur = np.array([d.qpos[a] for a in arm_qadr], dtype=np.float64)
    target = np.asarray(target_rel, dtype=np.float64)
    offset = sr.VR_OFFSETS[0 if side == "left" else 1]
    base = d.qpos[0:3]
    root = sr.quat_to_mat(d.qpos[3:7])
    p_des = base + root @ target

    def kinematics(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(error vector, point Jacobian, lever) with the arm set to `joints`."""
        scratch.qpos[:] = d.qpos
        for k, a in enumerate(arm_qadr):
            scratch.qpos[a] = joints[k]
        mujoco.mj_kinematics(m, scratch)
        # mj_jacBody reads `cdof`, which mj_kinematics does not fill. Without this the Jacobian
        # comes back exactly zero and the solver reports the starting error forever with dq = 0
        # (measured: six iterations, |dq| = 0.0000, residual stuck at 235.3 mm).
        mujoco.mj_comPos(m, scratch)
        rib = scratch.xmat[bid].reshape(3, 3)
        lever = rib @ offset
        ep = p_des - (scratch.xpos[bid] + lever)
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, scratch, jp, jr, bid)
        # The lever is rigid on this body, so the point Jacobian carries a lever term:
        # d(p + R o)/dq = Jp - [R o]x Jr. Dropping it leaves a ~0.09 m bias that shows up as
        # residual the loop cannot remove.
        jac = jp[:, dofs_arr] - _skew(lever) @ jr[:, dofs_arr]
        return ep, jac, lever

    ep, jac, _ = kinematics(cur)
    residual = float(np.linalg.norm(ep))
    lam = 1e-2
    for _ in range(iters):
        if residual < tol:
            break
        improved = False
        for _ in range(8):
            dq = jac.T @ np.linalg.solve(jac @ jac.T + lam * np.eye(3), ep)
            cand = np.clip(cur + np.clip(dq, -step, step), lo, hi)
            ep_c, jac_c, _ = kinematics(cand)
            r_c = float(np.linalg.norm(ep_c))
            if r_c < residual:
                cur, ep, jac, residual = cand, ep_c, jac_c, r_c
                lam = max(lam / 3.0, 1e-5)
                improved = True
                break
            lam *= 8.0
        if not improved:
            break

    vec = payload(runner, d.qpos)
    arm = ARM_HW[side]
    for i, hw in enumerate(sr.UPPER_BODY_MUJOCO_IDX):
        if hw in arm:
            vec[i] = cur[arm.index(hw)]
    return vec, residual
