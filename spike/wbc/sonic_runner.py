#!/usr/bin/env python3
"""Headless MuJoCo runner for the GEAR-SONIC G1 ONNX controller.

Reconstructs the observation assembly from
gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp
(GetObservationRegistry + gatherers) and drives the released
model_encoder.onnx / model_decoder.onnx at 50 Hz with 500 Hz PD torque.

The motion CSV joint order (the "policy order") was identified by matching
MuJoCo FK against body_pos/body_quat in the released example motions. It is:

  left_hip_pitch, right_hip_pitch, waist_yaw,
  left_hip_roll, right_hip_roll, waist_roll,
  left_hip_yaw, right_hip_yaw, waist_pitch,
  left_knee, right_knee,
  left_shoulder_pitch, right_shoulder_pitch,
  left_ankle_pitch, right_ankle_pitch,
  left_shoulder_roll, right_shoulder_roll,
  left_ankle_roll, right_ankle_roll,
  left_shoulder_yaw, right_shoulder_yaw,
  left_elbow, right_elbow,
  left_wrist_roll, right_wrist_roll,
  left_wrist_pitch, right_wrist_pitch,
  left_wrist_yaw, right_wrist_yaw
"""

import argparse
import json
import os
import time

import mujoco
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.path.join(HERE, "vendor", "GR00T-WholeBodyControl", "gear_sonic_deploy")
XML = os.path.join(DEPLOY, "g1", "scene_sonic_floor.xml")
MOTION_ROOT = os.path.join(DEPLOY, "reference", "example", "_dl")

#: The deploy's upper-body override, verbatim from
#: `gear_sonic_deploy/.../include/policy_parameters.hpp:80-81`. NVIDIA's own C++ deploy accepts
#: 17 upper-body joint angles (`upper_body_position[17]` on the planner topic) and substitutes
#: them into the *reference motion* every look-ahead frame, leaving the legs and balance on the
#: policy. `GatherMotionJointPositionsMultiFrame` is the code: it copies the clip's 29 joints,
#: then overwrites these 17. This is the escape hatch the wrist-pose interface lacks - the VR
#: path is 9 positions and 12 quaternion floats with no velocity and no horizon, which is why a
#: step target rings (measured here: 0.77-1.05 s settle, +75 % overshoot).
UPPER_BODY_ISAACLAB_IDX = (2, 5, 8, 11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28)
UPPER_BODY_MUJOCO_IDX = (12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28)
WEIGHTS = os.path.join(HERE, "weights", "sonic")

CONTROL_DT = 0.02
PHYSICS_DT = 0.002
STEPS_PER_CONTROL = int(round(CONTROL_DT / PHYSICS_DT))
TOKEN_DIM = 64

# policy/CSV joint index -> MuJoCo hardware joint index
CSV2MJ = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11,
          17, 24, 18, 25, 19, 26, 20, 27, 21, 28]

# hardware order, from policy_parameters.hpp (compiled via g++)
KP = np.array([99.0984268, 99.0984268, 40.1792374, 99.0984268, 28.5012455,
               28.5012455, 99.0984268, 99.0984268, 40.1792374, 99.0984268,
               28.5012455, 28.5012455, 40.1792374, 28.5012455, 28.5012455,
               14.2506227, 14.2506227, 14.2506227, 14.2506227, 14.2506227,
               16.7783279, 16.7783279, 14.2506227, 14.2506227, 14.2506227,
               14.2506227, 14.2506227, 16.7783279, 16.7783279])
KD = np.array([6.30880165, 6.30880165, 2.5578897, 6.30880165, 1.81444573,
               1.81444573, 6.30880165, 6.30880165, 2.5578897, 6.30880165,
               1.81444573, 1.81444573, 2.5578897, 1.81444573, 1.81444573,
               0.907222867, 0.907222867, 0.907222867, 0.907222867,
               0.907222867, 1.06814146, 1.06814146, 0.907222867,
               0.907222867, 0.907222867, 0.907222867, 0.907222867,
               1.06814146, 1.06814146])
ACTION_SCALE = np.array([0.350661466, 0.350661466, 0.547546465, 0.350661466,
                         0.438577314, 0.438577314, 0.350661466, 0.350661466,
                         0.547546465, 0.350661466, 0.438577314, 0.438577314,
                         0.547546465, 0.438577314, 0.438577314, 0.438577314,
                         0.438577314, 0.438577314, 0.438577314, 0.438577314,
                         0.0745008703, 0.0745008703, 0.438577314, 0.438577314,
                         0.438577314, 0.438577314, 0.438577314, 0.0745008703,
                         0.0745008703])
DEFAULT = np.array([-0.312, 0, 0, 0.669, -0.363, 0, -0.312, 0, 0, 0.669,
                    -0.363, 0, 0, 0, 0, 0.2, 0.2, 0, 0.6, 0, 0, 0, 0.2,
                    -0.2, 0, 0.6, 0, 0, 0])

# lower-body joints in hardware order, expressed as policy/CSV indices
LOWER_CSV = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18]
WRIST_CSV = [23, 24, 25, 26, 27, 28]

# body indices in the released 14-body motion arrays
BODY_PELVIS = 0
BODY_TORSO = 7
BODY_L_WRIST = 10
BODY_R_WRIST = 13
VR_OFFSETS = np.array([[0.18, -0.025, 0.0],
                       [0.18, 0.025, 0.0],
                       [0.0, 0.0, 0.35]])


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q, v):
    qv = np.array([0.0, v[0], v[1], v[2]])
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_to_6d(q):
    R = quat_to_mat(q)
    return np.array([R[0, 0], R[0, 1], R[1, 0], R[1, 1], R[2, 0], R[2, 1]])


def load_motion(name):
    d = os.path.join(MOTION_ROOT, name)
    jp = np.loadtxt(os.path.join(d, "joint_pos.csv"), delimiter=",", skiprows=1)
    jv = np.loadtxt(os.path.join(d, "joint_vel.csv"), delimiter=",", skiprows=1)
    bp = np.loadtxt(os.path.join(d, "body_pos.csv"), delimiter=",", skiprows=1).reshape(-1, 14, 3)
    bq = np.loadtxt(os.path.join(d, "body_quat.csv"), delimiter=",", skiprows=1).reshape(-1, 14, 4)
    bl = np.loadtxt(os.path.join(d, "body_lin_vel.csv"), delimiter=",", skiprows=1).reshape(-1, 14, 3)
    ba = np.loadtxt(os.path.join(d, "body_ang_vel.csv"), delimiter=",", skiprows=1).reshape(-1, 14, 3)
    bq = bq / np.linalg.norm(bq, axis=-1, keepdims=True)
    return dict(jp=jp, jv=jv, bp=bp, bq=bq, bl=bl, ba=ba, n=jp.shape[0])


def vr_targets(motion, frame):
    """Replicates GatherVR3PointPosition/Orientation (no external VR input)."""
    frame = min(frame, motion["n"] - 1)
    pos = motion["bp"][frame]
    quat = motion["bq"][frame]
    root_pos = pos[BODY_PELVIS]
    root_quat = quat[BODY_PELVIS]
    root_inv = quat_conj(root_quat)
    pts = np.zeros(9)
    orns = np.zeros(12)
    for i, body in enumerate((BODY_L_WRIST, BODY_R_WRIST, BODY_TORSO)):
        world = pos[body] + quat_rotate(quat[body], VR_OFFSETS[i])
        rel = quat_rotate(root_inv, world - root_pos)
        pts[3 * i:3 * i + 3] = rel
        orns[4 * i:4 * i + 4] = quat_mul(root_inv, quat[body])
    return pts, orns


def anchor_6d(motion, frame, base_quat):
    frame = min(frame, motion["n"] - 1)
    ref = motion["bq"][frame, BODY_PELVIS]
    return quat_to_6d(quat_mul(quat_conj(base_quat), ref))


def heading_quat(q):
    w, x, y, z = q
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def anchor_heading_6d(motion, frame, base_quat):
    frame = min(frame, motion["n"] - 1)
    ref = motion["bq"][frame, BODY_PELVIS]
    return quat_to_6d(quat_mul(quat_conj(heading_quat(base_quat)), ref))


def motion_frames(motion, frame, num=10, step=5):
    idx = np.minimum(frame + step * np.arange(num), motion["n"] - 1)
    return idx


class SonicRunner:
    def __init__(self, mode, verbose=False, model="default"):
        self.mode = mode
        self.verbose = verbose
        self.v11 = model == "v11"
        self.lowlat = model == "lowlat"
        #: 17 upper-body joint angles (rad, IsaacLab order) to substitute into the reference
        #: every tick, or None to leave the clip alone. Set by the executor from IK.
        self.upper_body = None
        self.model = mujoco.MjModel.from_xml_path(XML)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = PHYSICS_DT
        if self.v11:
            wdir = os.path.join(HERE, "weights", "sonic_v11")
        elif self.lowlat:
            wdir = os.path.join(HERE, "weights", "sonic_ll")
        else:
            wdir = WEIGHTS
        # MEASURED: with default settings onnxruntime opened one intra-op thread
        # per core and the body_view held ~19 of 20 logical CPUs (1876% CPU, 47
        # threads, load average 21). These nets are tiny - 1762->64 and 994->29 -
        # so a thread pool costs more in spin-wait than it saves in FLOPs.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.enc = ort.InferenceSession(os.path.join(wdir, "model_encoder.onnx"),
                                        sess_options=opts,
                                        providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(os.path.join(wdir, "model_decoder.onnx"),
                                        sess_options=opts,
                                        providers=["CPUExecutionProvider"])
        self.enc_in = self.enc.get_inputs()[0].name
        self.dec_in = self.dec.get_inputs()[0].name
        self.qadr = [self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in [
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
                "right_wrist_pitch_joint", "right_wrist_yaw_joint"]]
        self.dofadr = [self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in [
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
                "right_wrist_pitch_joint", "right_wrist_yaw_joint"]]
        self.ctrlrange = self.model.actuator_ctrlrange[:29].copy()
        self.hist = []  # (base_ang_vel, q_policy, dq_policy, last_action, gravity)

    def reset(self, motion, start=0):
        d = self.data
        d.qpos[:] = 0
        d.qpos[0:3] = motion["bp"][start, BODY_PELVIS]
        d.qpos[3:7] = motion["bq"][start, BODY_PELVIS]
        for csv_i, mj_i in enumerate(CSV2MJ):
            d.qpos[self.qadr[mj_i]] = motion["jp"][start, csv_i]
            d.qvel[self.dofadr[mj_i]] = motion["jv"][start, csv_i]
        d.qvel[0:3] = motion["bl"][start, BODY_PELVIS]
        d.qvel[3:6] = quat_rotate(quat_conj(motion["bq"][start, BODY_PELVIS]), motion["ba"][start, BODY_PELVIS])
        mujoco.mj_forward(self.model, d)
        self.hist = []
        for _ in range(10):
            self.hist.append(self._state_entry())

    def _q_policy(self):
        return np.array([self.data.qpos[self.qadr[mj]] for mj in CSV2MJ]) - DEFAULT[CSV2MJ]

    def _dq_policy(self):
        return np.array([self.data.qvel[self.dofadr[mj]] for mj in CSV2MJ])

    def _state_entry(self, last_action=None):
        d = self.data
        base_quat = d.qpos[3:7].copy()
        base_ang_vel = d.qvel[3:6].copy()
        gravity = quat_rotate(quat_conj(base_quat), np.array([0.0, 0.0, -1.0]))
        return dict(base_ang_vel=base_ang_vel, q=self._q_policy(), dq=self._dq_policy(),
                    action=np.zeros(29) if last_action is None else last_action.copy(),
                    gravity=gravity, base_quat=base_quat)

    def _jp(self, motion, f):
        """The reference joint row with the upper-body override applied, if any."""
        row = motion["jp"][f]
        if self.upper_body is None:
            return row
        out = np.array(row, dtype=np.float64, copy=True)
        out[list(UPPER_BODY_ISAACLAB_IDX)] = self.upper_body
        return out

    def build_encoder_obs(self, motion, frame, base_quat, vr_pos=None, vr_orn=None, vr_delta=None, vr_orn_delta=None):
        obs = np.zeros(1247 if self.lowlat else (1751 if self.v11 else 1762), dtype=np.float32)
        o = 0
        obs[o:o + 4] = [self.mode, 0, 0, 0]
        o += 4
        if self.lowlat:
            idx = motion_frames(motion, frame, step=1)
            if self.mode == 0:
                for f in idx:
                    obs[o:o + 29] = self._jp(motion, f)
                    o += 29
                for f in idx:
                    obs[o:o + 29] = motion["jv"][f]
                    o += 29
            else:
                for f in idx:
                    obs[o:o + 29] = self._jp(motion, f)
                    o += 29
                for f in idx:
                    obs[o:o + 29] = motion["jv"][f]
                    o += 29
            for f in idx:
                obs[o:o + 6] = anchor_6d(motion, f, base_quat)
                o += 6
            obs[o:o + 6] = anchor_6d(motion, frame, base_quat)
            o += 6
            if self.mode == 1:
                for f in idx:
                    obs[o:o + 12] = motion["jp"][f, LOWER_CSV]
                    o += 12
                for f in idx:
                    obs[o:o + 12] = motion["jv"][f, LOWER_CSV]
                    o += 12
                pts, orns = vr_targets(motion, frame)
                if vr_pos is not None: pts = np.asarray(vr_pos, dtype=np.float64)
                if vr_delta is not None: pts = pts + np.asarray(vr_delta, dtype=np.float64)
                if vr_orn is not None: orns = np.asarray(vr_orn, dtype=np.float64)
                if vr_orn_delta is not None:
                    oo = np.asarray(orns, dtype=np.float64).reshape(3, 4).copy()
                    for k in range(3):
                        oo[k] = quat_mul(np.asarray(vr_orn_delta[k], dtype=np.float64), oo[k])
                    orns = oo.reshape(-1)
                self.last_vr = (np.asarray(pts, dtype=np.float64).copy(), np.asarray(orns, dtype=np.float64).copy())
                obs[o:o + 9] = pts
                o += 9
                obs[o:o + 12] = orns
                o += 12
            return obs
        if self.v11:
            idx = motion_frames(motion, frame)
            if self.mode == 0:
                for f in idx:
                    obs[o:o + 29] = self._jp(motion, f)
                    o += 29
                for f in idx:
                    obs[o:o + 29] = motion["jv"][f]
                    o += 29
            else:
                o += 290 + 290
            for f in idx:
                obs[o:o + 6] = anchor_heading_6d(motion, f, base_quat)
                o += 6
            obs[o:o + 6] = anchor_heading_6d(motion, frame, base_quat)
            o += 6
            if self.mode == 1:
                idx = motion_frames(motion, frame)
                for f in idx:
                    obs[o:o + 12] = motion["jp"][f, LOWER_CSV]
                    o += 12
                for f in idx:
                    obs[o:o + 12] = motion["jv"][f, LOWER_CSV]
                    o += 12
                pts, orns = vr_targets(motion, frame)
                if vr_pos is not None: pts = np.asarray(vr_pos, dtype=np.float64)
                if vr_delta is not None: pts = pts + np.asarray(vr_delta, dtype=np.float64)
                if vr_orn is not None: orns = np.asarray(vr_orn, dtype=np.float64)
                if vr_orn_delta is not None:
                    oo = np.asarray(orns, dtype=np.float64).reshape(3, 4).copy()
                    for k in range(3):
                        oo[k] = quat_mul(np.asarray(vr_orn_delta[k], dtype=np.float64), oo[k])
                    orns = oo.reshape(-1)
                self.last_vr = (np.asarray(pts, dtype=np.float64).copy(), np.asarray(orns, dtype=np.float64).copy())
                obs[o:o + 9] = pts
                o += 9
                obs[o:o + 12] = orns
                o += 12
            return obs
        if self.mode == 0:
            idx = motion_frames(motion, frame)
            for f in idx:
                obs[o:o + 29] = self._jp(motion, f)
                o += 29
            for f in idx:
                obs[o:o + 29] = motion["jv"][f]
                o += 29
            o += 10  # root z multi-frame
            o += 1   # root z single
            obs[o:o + 6] = anchor_6d(motion, frame, base_quat)
            o += 6
            for f in idx:
                obs[o:o + 6] = anchor_6d(motion, f, base_quat)
                o += 6
            # lowerbody, vr, smpl, wrists not used by mode 0
        else:
            # teleop uses the single motion_anchor_orientation at offset 595
            obs[595:601] = anchor_6d(motion, frame, base_quat)
            o += 290 + 290 + 10 + 1 + 6 + 60
            idx = motion_frames(motion, frame)
            for f in idx:
                obs[o:o + 12] = motion["jp"][f, LOWER_CSV]
                o += 12
            for f in idx:
                obs[o:o + 12] = motion["jv"][f, LOWER_CSV]
                o += 12
            pts, orns = vr_targets(motion, frame)
            if vr_pos is not None: pts = np.asarray(vr_pos, dtype=np.float64)
            if vr_delta is not None: pts = pts + np.asarray(vr_delta, dtype=np.float64)
            if vr_orn is not None: orns = np.asarray(vr_orn, dtype=np.float64)
            if vr_orn_delta is not None:
                oo = np.asarray(orns, dtype=np.float64).reshape(3, 4).copy()
                for k in range(3):
                    oo[k] = quat_mul(np.asarray(vr_orn_delta[k], dtype=np.float64), oo[k])
                orns = oo.reshape(-1)
            self.last_vr = (np.asarray(pts, dtype=np.float64).copy(), np.asarray(orns, dtype=np.float64).copy())
            obs[o:o + 9] = pts
            o += 9
            obs[o:o + 12] = orns
            o += 12
            # smpl, smpl anchor, wrists not used by teleop
        return obs

    def build_decoder_obs(self, tokens):
        hist = self.hist[-10:]
        out = np.zeros(994, dtype=np.float32)
        out[:64] = tokens
        o = 64
        for key in ("base_ang_vel", "q", "dq", "action", "gravity"):
            for e in hist:
                v = e[key]
                out[o:o + len(v)] = v
                o += len(v)
        assert o == 994, o
        return out

    def step_policy(self, action):
        target = DEFAULT.copy()
        for p in range(29):
            target[CSV2MJ[p]] = DEFAULT[CSV2MJ[p]] + ACTION_SCALE[CSV2MJ[p]] * action[p]
        for _ in range(STEPS_PER_CONTROL):
            q = np.array([self.data.qpos[self.qadr[i]] for i in range(29)])
            dq = np.array([self.data.qvel[self.dofadr[i]] for i in range(29)])
            tau = KP * (target - q) - KD * dq
            tau = np.clip(tau, self.ctrlrange[:, 0], self.ctrlrange[:, 1])
            self.data.ctrl[:29] = tau
            mujoco.mj_step(self.model, self.data)

    def run(self, motion, seconds=10.0, ramp=0.5, log_path=None, freeze=None, vr_pos=None, vr_orn=None, vr_delta=None, vr_orn_delta=None, start=0,
            freeze_schedule=None, step_at=None, step_delta=None, step_orn_delta=None):
        self.reset(motion, start=start)
        n_ticks = int(round(seconds / CONTROL_DT))
        rows = []
        last_action = np.zeros(29)
        for tick in range(n_ticks):
            t = tick * CONTROL_DT
            frame = freeze if freeze is not None else min(start + tick, motion["n"] - 1)
            if freeze_schedule is not None:
                frame = min(start + tick, motion["n"] - 1)
                for t0, t1, f in freeze_schedule:
                    if t0 <= t < t1:
                        frame = f
                        break
            cur_delta = vr_delta
            cur_orn_delta = vr_orn_delta
            if step_at is not None and t >= step_at:
                if step_delta is not None:
                    cur_delta = (np.zeros(9) if vr_delta is None else np.asarray(vr_delta)) + np.asarray(step_delta)
                if step_orn_delta is not None:
                    cur_orn_delta = step_orn_delta if vr_orn_delta is None else [
                        quat_mul(np.asarray(step_orn_delta[k]), np.asarray(vr_orn_delta[k])) for k in range(3)]
            base_quat = self.data.qpos[3:7].copy()
            enc_obs = self.build_encoder_obs(motion, frame, base_quat, vr_pos=vr_pos, vr_orn=vr_orn, vr_delta=cur_delta, vr_orn_delta=cur_orn_delta)
            tokens = self.enc.run(None, {self.enc_in: enc_obs[None, :]})[0][0]
            dec_obs = self.build_decoder_obs(tokens)
            action = self.dec.run(None, {self.dec_in: dec_obs[None, :]})[0][0].astype(np.float64)
            if ramp > 0 and tick * CONTROL_DT < ramp:
                action = action * (tick * CONTROL_DT / ramp)
            self.step_policy(action)
            last_action = action
            e = self._state_entry(last_action)
            self.hist.append(e)
            if len(self.hist) > 20:
                self.hist.pop(0)
            # metrics
            q_meas = np.array([self.data.qpos[self.qadr[mj]] for mj in CSV2MJ])
            q_ref = motion["jp"][frame]
            base = self.data.qpos[0:3].copy()
            up = quat_rotate(self.data.qpos[3:7], np.array([0.0, 0.0, 1.0]))
            root_inv = quat_conj(self.data.qpos[3:7])
            lw = self.model.body('left_wrist_yaw_link').id
            rw = self.model.body('right_wrist_yaw_link').id
            lw_pos = quat_rotate(root_inv, self.data.xpos[lw] + self.data.xmat[lw].reshape(3,3) @ VR_OFFSETS[0] - base)
            rw_pos = quat_rotate(root_inv, self.data.xpos[rw] + self.data.xmat[rw].reshape(3,3) @ VR_OFFSETS[1] - base)
            tr = self.model.body('torso_link').id
            hw_pos = quat_rotate(root_inv, self.data.xpos[tr] + self.data.xmat[tr].reshape(3,3) @ VR_OFFSETS[2] - base)
            torso_q = quat_mul(root_inv, self.data.qpos[3:7]) if False else quat_mul(root_inv, self.data.xquat[tr])
            lw_q = quat_mul(root_inv, self.data.xquat[lw])
            rw_q = quat_mul(root_inv, self.data.xquat[rw])
            rows.append(dict(t=tick * CONTROL_DT, frame=frame,
                             lw=[float(v) for v in lw_pos], rw=[float(v) for v in rw_pos],
                             hw=[float(v) for v in hw_pos], torso_q=[float(v) for v in torso_q],
                             lw_q=[float(v) for v in lw_q], rw_q=[float(v) for v in rw_q],
                             vr_t=[float(v) for v in self.last_vr[0]], vr_o=[float(v) for v in self.last_vr[1]],
                             x=float(base[0]), y=float(base[1]), z=float(base[2]),
                             up_z=float(up[2]),
                             q_err=float(np.sqrt(np.mean((q_meas - q_ref) ** 2))),
                             act_norm=float(np.linalg.norm(action)),
                             ref_x=float(motion["bp"][frame, BODY_PELVIS, 0]),
                             ref_y=float(motion["bp"][frame, BODY_PELVIS, 1]),
                             ref_z=float(motion["bp"][frame, BODY_PELVIS, 2])))
            if self.verbose and tick % 25 == 0:
                print(f"t={tick*CONTROL_DT:5.2f} z={base[2]:.3f} up={up[2]:.3f} "
                      f"qerr={rows[-1]['q_err']:.3f} |a|={rows[-1]['act_norm']:.2f}")
            if base[2] < 0.25 or up[2] < 0.3:
                if self.verbose:
                    print(f"FALL at t={tick*CONTROL_DT:.2f} z={base[2]:.3f} up={up[2]:.3f}")
                break
        if log_path:
            with open(log_path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", default="walking_quip_360_R_002__A428")
    ap.add_argument("--mode", type=int, default=1, choices=[0, 1])
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--ramp", type=float, default=0.5)
    ap.add_argument("--log", default=None)
    ap.add_argument("--freeze", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--model", choices=["default", "v11", "lowlat"], default="default")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    motion = load_motion(args.motion)
    print(f"motion {args.motion}: {motion['n']} frames")
    runner = SonicRunner(args.mode, verbose=args.verbose, model=args.model)
    rows = runner.run(motion, seconds=args.seconds, ramp=args.ramp, log_path=args.log, freeze=args.freeze, start=args.start)
    last = rows[-1]
    print(f"ticks={len(rows)} t={last['t']:.2f} z={last['z']:.3f} up_z={last['up_z']:.3f} "
          f"q_err={last['q_err']:.3f} |a|={last['act_norm']:.2f}")
    print(f"root x,y=({last['x']:.3f},{last['y']:.3f}) ref=({last['ref_x']:.3f},{last['ref_y']:.3f})")


if __name__ == "__main__":
    main()
