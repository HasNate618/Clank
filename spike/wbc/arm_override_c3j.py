# pyright: reportAttributeAccessIssue=false
# mujoco's bundled stubs omit MjModel/MjData/mj_kinematics/mj_jacBody. Every file in
# spike/wbc/ raises the same report; it is a stub gap, not a defect.
#!/usr/bin/env python3
"""C3j — can SONIC's 14 arm joints be replaced by a sharper IK solve?

The hypothesis: SONIC's own arms are already driven by external VR targets in
its training distribution, so overriding them with a task-space IK solve toward
the *same* targets may be tolerated, where C3f's override of a balance-using
locomotion policy was not.

What is overridden: the position targets of the 14 arm joints only (shoulder
pitch/roll/yaw, elbow, wrist roll/pitch/yaw, both sides).  SONIC keeps the 12
leg joints, the 3 waist joints, and the base: they come from the policy's
decoder output exactly as before.  The IK is a damped Gauss-Newton solve over
the arm columns of the MuJoCo body Jacobian, started from the measured joint
state every control tick, so the actuator targets close a loop around the
achieved pose.

Variants (cheapest first):
  policy            no override (baseline)
  hard              arm joint targets := IK solution, no rate limit, no blend
  rate:<rad/s>      IK solution approached from the previous command at <= rate
  blend:<a>         target := (1-a)*policy_target + a*IK solution
  int:<g>           integral target: the applied target accumulates g*(IK minus
                    measured) each tick.  Needed because the arm PD is soft
                    (KP 14.25 Nm/rad at the shoulder) and a plain IK target
                    settles at a gravity droop of ~0.2 rad.
  orient            only the 6 wrist dofs move, matching commanded orientation;
                    shoulder/elbow keep the policy's position targets

Two target sets:
  motion            the VR targets the policy already receives from the reference
                    motion (tests whether IK tracking is at least as good)
  aim               a fixed root-relative arm stance pointing at a fixed world
                    point (the thing the game actually needs)

Run:  spike/e3/.venv/bin/python spike/wbc/arm_override_c3j.py
Out:  spike/wbc/logs/c3j_*.jsonl, spike/wbc/logs/c3j_summary.json
"""

import json
import os
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_common as G  # noqa: E402
import sonic_runner as S  # noqa: E402

LOGS = os.path.join(HERE, "logs")
os.makedirs(LOGS, exist_ok=True)

WALK = "walk_forward_amateur_001__A001"
WRIST_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
WRIST_COLS = [4, 5, 6, 11, 12, 13]  # wrist_roll/pitch/yaw for L then R


def rotvec(R):
    """Rotation vector of a 3x3 rotation matrix."""
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(cos_a)
    if ang < 1e-6:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(ang)
    if s < 1e-6:
        # ang ~ pi: use the largest diagonal
        i = int(np.argmax(np.diag(R)))
        v = np.zeros(3)
        v[i] = np.sqrt(max(0.0, (R[i, i] + 1.0) / 2.0))
        for j in range(3):
            if j != i:
                v[j] = R[i, j] / (2.0 * v[i] + 1e-12)
        return v / np.linalg.norm(v) * ang
    return axis / (2.0 * s) * ang


class ArmOverrideRunner(G.HookRunner):
    def __init__(self, variant="policy", model="default", ik_iters=12, ik_step=0.40):
        super().__init__(1, verbose=False, model=model)
        self.variant = variant
        self.ik_iters = ik_iters
        self.ik_step = ik_step
        self.arm_qadr = [self.qadr[j] for j in G.ARM_MJ]
        self.arm_dof = [self.dofadr[j] for j in G.ARM_MJ]
        lo = np.array([self.model.jnt_range[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                              mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                                self.model.dof_jntid[d]))][0]
            for d in self.arm_dof])
        hi = np.array([self.model.jnt_range[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                              mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                                self.model.dof_jntid[d]))][1]
            for d in self.arm_dof])
        self.arm_lo, self.arm_hi = lo + 0.02, hi - 0.02
        self.scratch = mujoco.MjData(self.model)
        self.q_applied = None
        self.ik_ms = []
        self.sat_frac = []

    # ---- target poses from the VR targets the policy is being given
    def _arm_targets(self):
        pts, orns = self.last_vr
        bp = self.data.qpos[0:3].copy()
        bq = self.data.qpos[3:7].copy()
        R_b = S.quat_to_mat(bq)
        out = []
        for k, (bi, off) in enumerate(((0, S.VR_OFFSETS[0]), (1, S.VR_OFFSETS[1]))):
            q_world = S.quat_mul(bq, np.asarray(orns[4 * k:4 * k + 4], dtype=np.float64))
            p_point = bp + R_b @ np.asarray(pts[3 * k:3 * k + 3], dtype=np.float64)
            p_body = p_point - S.quat_rotate(q_world, off)
            out.append((self.model.body(WRIST_BODIES[bi]).id, p_body, q_world))
        return out

    def _solve_ik(self, targets, cols=None, ori_only=False):
        q0 = np.array([self.data.qpos[a] for a in self.arm_qadr])
        q = q0.copy()
        s = self.scratch
        for _ in range(self.ik_iters):
            s.qpos[:] = self.data.qpos
            for k, a in enumerate(self.arm_qadr):
                s.qpos[a] = q[k]
            mujoco.mj_kinematics(self.model, s)
            mujoco.mj_comPos(self.model, s)
            Jl, el = [], []
            for bid, p_des, q_des in targets:
                jp = np.zeros((3, self.model.nv))
                jr = np.zeros((3, self.model.nv))
                mujoco.mj_jacBody(self.model, s, jp, jr, bid)
                cols_i = self.arm_dof if cols is None else [self.arm_dof[c] for c in cols]
                J = np.vstack([jp[:, cols_i], jr[:, cols_i]])
                ep = p_des - s.xpos[bid]
                Rerr = S.quat_to_mat(q_des) @ s.xmat[bid].reshape(3, 3).T
                eo = rotvec(Rerr)
                if ori_only:
                    J = J[3:]
                    e = eo
                else:
                    e = np.concatenate([ep, eo])
                Jl.append(J)
                el.append(e)
            Js = np.vstack(Jl)
            es = np.concatenate(el)
            lam2 = 1e-3
            dq = Js.T @ np.linalg.solve(Js @ Js.T + lam2 * np.eye(Js.shape[0]), es)
            dq = np.clip(dq, -self.ik_step, self.ik_step)
            if cols is None:
                q = np.clip(q + dq, self.arm_lo, self.arm_hi)
            else:
                for n, c in enumerate(cols):
                    q[c] = np.clip(q[c] + dq[n], self.arm_lo[c], self.arm_hi[c])
        return q

    def _new_arm_target(self, pol_arm):
        t0 = time.perf_counter()
        if self.variant == "orient":
            qk = self._solve_ik(self._arm_targets(), cols=WRIST_COLS, ori_only=True)
            out = pol_arm.copy()
            for c in WRIST_COLS:
                out[c] = qk[c]
        else:
            qk = self._solve_ik(self._arm_targets())
            v = self.variant
            if v == "hard":
                out = qk
            elif v.startswith("rate"):
                rate = float(v.split(":")[1])
                ref = self.q_applied if self.q_applied is not None else pol_arm
                d = np.clip(qk - ref, -rate * S.CONTROL_DT, rate * S.CONTROL_DT)
                out = ref + d
            elif v.startswith("blend"):
                a = float(v.split(":")[1])
                out = (1.0 - a) * pol_arm + a * qk
            elif v.startswith("int"):
                g = float(v.split(":")[1])
                ref = self.q_applied if self.q_applied is not None else pol_arm
                qm = np.array([self.data.qpos[a] for a in self.arm_qadr])
                out = np.clip(ref + g * (qk - qm), self.arm_lo, self.arm_hi)
            else:
                out = pol_arm
        self.ik_ms.append(1000.0 * (time.perf_counter() - t0))
        return out

    def step_policy(self, action):
        target = S.DEFAULT.copy()
        for p in range(29):
            target[S.CSV2MJ[p]] = S.DEFAULT[S.CSV2MJ[p]] + S.ACTION_SCALE[S.CSV2MJ[p]] * action[p]
        pol_arm = target[G.ARM_MJ].copy()
        if self.variant != "policy":
            new = self._new_arm_target(pol_arm)
            target[G.ARM_MJ] = new
            self.q_applied = new.copy()
        n_sat = 0
        for _ in range(S.STEPS_PER_CONTROL):
            q = np.array([self.data.qpos[i] for i in self.qadr])
            dq = np.array([self.data.qvel[i] for i in self.dofadr])
            tau = S.KP * (target - q) - S.KD * dq
            tau = np.clip(tau, self.ctrlrange[:, 0], self.ctrlrange[:, 1])
            n_sat += int(np.sum(np.abs(tau[G.ARM_MJ]) >= 0.99 * self.ctrlrange[G.ARM_MJ, 1]))
            self.data.ctrl[:29] = tau
            mujoco.mj_step(self.model, self.data)
        self.sat_frac.append(n_sat / (S.STEPS_PER_CONTROL * 14.0))

    def arm_track_err(self):
        if self.q_applied is None:
            return 0.0
        qm = np.array([self.data.qpos[a] for a in self.arm_qadr])
        return float(np.abs(qm - self.q_applied).max())

    def reset(self, motion, start=0):
        super().reset(motion, start=start)
        self.q_applied = None
        self.ik_ms = []
        self.sat_frac = []


# -------------------------------------------------------------------- metrics
def _tilt(up):
    return float(np.degrees(np.arccos(np.clip(up, -1.0, 1.0))))


def _qerr(a, b):
    return float(np.degrees(2 * np.arccos(np.clip(abs(float(np.dot(a, b))), 0, 1))))


def motion_track(rows):
    """Residual of the achieved wrists against the *motion's* VR targets."""
    a = G.series(rows)
    m = a["t"] >= 1.0
    pe = np.linalg.norm(a["rw"][m] - a["vr_t"][m, 1], axis=1)
    oe = np.array([_qerr(a["rw_q"][i], a["vr_o"][i, 1]) for i in np.where(m)[0]])
    return dict(wr_pos_mean=float(pe.mean()), wr_pos_p95=float(np.percentile(pe, 95)),
                wr_orn_mean=float(oe.mean()), wr_orn_p95=float(np.percentile(oe, 95)))


def walk_quality(runner, rows, n_expect):
    a = G.series(rows)
    th = np.array(runner.base_t)
    yaw = np.degrees(np.unwrap(np.radians(np.array(runner.base_yaw)))) if len(runner.base_yaw) else np.zeros(1)
    net = float(np.hypot(a["x"][-1] - a["x"][0], a["y"][-1] - a["y"][0]))
    return dict(
        n_ticks=len(rows), survived=len(rows) >= n_expect,
        fall_t=None if len(rows) >= n_expect else float(a["t"][-1]),
        net_disp=net,
        mean_speed=float(net / a["t"][-1]),
        tilt_max_deg=_tilt(a["up_z"].min()),
        min_z=float(a["z"].min()),
        base_yaw_drift_deg=float(np.mean(yaw[np.searchsorted(th, th[-1] - 1.0):])
                                 - np.mean(yaw[:np.searchsorted(th, 1.0)])),
        q_err_mean=float(a["q_err"][a["t"] >= 1.0].mean()),
    )


def run_case(label, variant, target, motion, start=600, seconds=10.0, freeze=None,
             extra=None, heading_P=None):
    runner = ArmOverrideRunner(variant)
    aim = None
    if target == "aim":
        if heading_P is None:
            bp = motion["bp"][start, S.BODY_PELVIS]
            fwd = S.quat_to_mat(motion["bq"][start, S.BODY_PELVIS])[:, 0]
            P = bp + fwd * 20.0
            P = np.array([P[0], P[1], bp[2] + 0.15])
        else:
            P = heading_P
        aim = G.AimTarget(P)

        def hook(mo, fr, bpos, bq, _aim=aim, _r=runner):
            pts, orns = S.vr_targets(mo, fr)
            p2, o2 = _aim.command(mo, fr, bpos, bq, pts, orns)
            _r.aim_cmd = _aim.last
            return p2, o2

        runner.vr_hook = hook
    rows = runner.run(motion, seconds=seconds, start=start, ramp=0.5, freeze=freeze,
                      log_path=os.path.join(LOGS, f"c3j_{label}.jsonl"),
                      freeze_schedule=(extra or {}).get("freeze_schedule"))
    n_expect = int(round(seconds / S.CONTROL_DT))
    out: dict = {"label": label, "variant": variant, "target": target}
    out.update(walk_quality(runner, rows, n_expect))
    out["arm_track_err_final"] = runner.arm_track_err()
    out["ik_ms_mean"] = float(np.mean(runner.ik_ms)) if runner.ik_ms else 0.0
    out["arm_torque_sat_frac"] = float(np.mean(runner.sat_frac)) if runner.sat_frac else 0.0
    if aim is not None and runner.diag:
        sel = [d for d in runner.diag if d["t"] >= 1.0]
        for key in ("wr_err_pos", "wr_err_orn", "wr_aim", "hd_aim"):
            v = np.array([d[key] for d in sel])
            out[key + "_mean"] = float(v.mean())
            out[key + "_p95"] = float(np.percentile(v, 95))
            out[key + "_max"] = float(v.max())
    else:
        out.update(motion_track(rows))
    return out


def main():
    walk = S.load_motion(WALK)
    res = []

    cases = [
        ("baseline_motion", "policy", "motion", {}),
        ("hard_motion", "hard", "motion", {}),
        ("baseline_aim", "policy", "aim", {}),
        ("hard_aim", "hard", "aim", {}),
        ("rate2_aim", "rate:2.0", "aim", {}),
        ("rate6_aim", "rate:6.0", "aim", {}),
        ("blend50_aim", "blend:0.5", "aim", {}),
        ("blend75_aim", "blend:0.75", "aim", {}),
        ("orient_aim", "orient", "aim", {}),
        # the PD-droop fix, reported separately from the naive variants above
        ("int50_motion", "int:0.5", "motion", {}),
        ("int100_motion", "int:1.0", "motion", {}),
        ("int50_aim", "int:0.5", "aim", {}),
        ("int100_aim", "int:1.0", "aim", {}),
        ("int100_aim_20s", "int:1.0", "aim", {"seconds": 20.0}),
        ("int15_aim", "int:0.15", "aim", {}),
    ]
    # hold / aim-in-place: the same arm-override machinery with the reference
    # frozen, for comparison against the classical QP's standing numbers.
    squat = S.load_motion("squat_001__A359")
    hold_cases = [
        ("hold_baseline", "policy"), ("hold_hard", "hard"),
        ("hold_orient", "orient"), ("hold_int15", "int:0.15"),
        ("hold_blend50", "blend:0.5"),
    ]
    for label, variant in hold_cases:
        out = run_case(label, variant, "aim", squat, start=0, freeze=0)
        res.append(out)
        print(json.dumps(out)[:400], flush=True)
    for label, variant, target, kw in cases:
        out = run_case(label, variant, target, walk, **kw)
        res.append(out)
        print(json.dumps(out)[:400], flush=True)

    # torso-yaw coupling: command a 30 deg head yaw during walking, with and
    # without the arm override.  Waist is policy-owned in both cases.
    for label, variant in (("yaw30_baseline", "policy"), ("yaw30_hard", "hard")):
        runner = ArmOverrideRunner(variant)
        rows = runner.run(walk, seconds=10.0, start=600, ramp=0.5,
                          vr_orn_delta=[G.I4, G.I4, G.quat_z(np.radians(30))],
                          log_path=os.path.join(LOGS, f"c3j_{label}.jsonl"))
        a = G.series(rows)
        m = a["t"] >= 5.0
        ty = [np.degrees(G.yaw_of(q)) for q in a["torso_q"][m]]
        out: dict = {"label": label, "variant": variant, "target": "motion+headyaw30"}
        out.update(walk_quality(runner, rows, 500))
        out["torso_yaw_mean_deg"] = float(np.mean(ty))
        out["torso_yaw_max_deg"] = float(np.max(ty))
        res.append(out)
        print(json.dumps(out)[:400], flush=True)

    with open(os.path.join(LOGS, "c3j_summary.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", os.path.join(LOGS, "c3j_summary.json"))


if __name__ == "__main__":
    main()
