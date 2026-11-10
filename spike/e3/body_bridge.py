#!/usr/bin/env python3
# mujoco ships no type information, so a checker reads every API access as an unknown
# attribute. That is a packaging gap, not a bug here: this file is exercised against a
# real MuJoCo model by the body gates, which is the check that counts.
# pyright: reportAttributeAccessIssue=false
import argparse, csv, ctypes as C, math, os, socket, struct, sys, time
import numpy as np
import mujoco

class Quat(C.Structure):
    _fields_ = [("x", C.c_float), ("y", C.c_float), ("z", C.c_float), ("w", C.c_float)]
class Vec3(C.Structure):
    _fields_ = [("x", C.c_float), ("y", C.c_float), ("z", C.c_float)]
class Vec2(C.Structure):
    _fields_ = [("x", C.c_float), ("y", C.c_float)]
class Vec1(C.Structure):
    _fields_ = [("x", C.c_float)]
class Pose(C.Structure):
    _fields_ = [("orientation", Quat), ("position", Vec3)]
class Fov(C.Structure):
    _fields_ = [("angle_left", C.c_float), ("angle_right", C.c_float), ("angle_up", C.c_float), ("angle_down", C.c_float)]
class View(C.Structure):
    _fields_ = [("fov", Fov), ("pose", Pose), ("_pad", C.c_uint32)]
class HeadData(C.Structure):
    _fields_ = [("views", View * 2), ("center", Pose), ("per_view_data_valid", C.c_bool), ("_pad0", C.c_bool), ("_pad1", C.c_bool), ("_pad2", C.c_bool)]
class ControllerData(C.Structure):
    _fields_ = [("pose", Pose), ("linear_velocity", Vec3), ("angular_velocity", Vec3), ("hand_curl", C.c_float * 5),
                ("trigger_value", Vec1), ("squeeze_value", Vec1), ("squeeze_force", Vec1), ("thumbstick", Vec2),
                ("trackpad_force", Vec1), ("trackpad", Vec2), ("hand_tracking_active", C.c_bool), ("active", C.c_bool),
                ("system_click", C.c_bool), ("system_touch", C.c_bool), ("a_click", C.c_bool), ("a_touch", C.c_bool),
                ("b_click", C.c_bool), ("b_touch", C.c_bool), ("trigger_click", C.c_bool), ("trigger_touch", C.c_bool),
                ("thumbstick_click", C.c_bool), ("thumbstick_touch", C.c_bool), ("trackpad_touch", C.c_bool),
                ("_pad0", C.c_bool), ("_pad1", C.c_bool), ("_pad2", C.c_bool)]
class RemoteData(C.Structure):
    _fields_ = [("header", C.c_uint64), ("head", HeadData), ("left", ControllerData), ("right", ControllerData)]

HEADER = struct.unpack("<Q", b"mndrmt3\0")[0]
SIZE = C.sizeof(RemoteData)
Q_ALIGN = np.array([-0.5, 0.5, 0.5, 0.5])

def assert_layout():
    assert C.sizeof(HeadData) == 128, C.sizeof(HeadData)
    assert C.sizeof(ControllerData) == 120, C.sizeof(ControllerData)
    assert SIZE == 376, SIZE

def qmul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw,
                     aw*bw - ax*bx - ay*by - az*bz])

def qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])

def qnorm(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])

def mj_to_xr_quat(q_wxyz):
    q = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return qnorm(qmul(qmul(Q_ALIGN, q), qconj(Q_ALIGN)))

def mj_to_xr_pos(p):
    return np.array([-p[1], p[2], -p[0]])

def xr_to_mj_pos(p):
    return np.array([-p[2], -p[0], p[1]])

def qrot(q_wxyz, v):
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z])
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)

def yaw_z(q_wxyz):
    w, x, y, z = q_wxyz
    return math.atan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))

def yaw_y(q_xyzw):
    x, y, z, w = q_xyzw
    return math.atan2(2.0 * (w*y + x*z), 1.0 - 2.0 * (y*y + z*z))

def wrap_deg(a):
    while a > 180.0: a -= 360.0
    while a < -180.0: a += 360.0
    return a

def blank_packet():
    d = RemoteData()
    d.header = HEADER
    d.head.center.position.y = 1.6
    d.head.center.orientation.w = 1.0
    d.head.per_view_data_valid = False
    for c in (d.left, d.right):
        c.active = True
        c.hand_tracking_active = False
        c.pose.orientation.w = 1.0
    return d

def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f"peer closed after {len(buf)}/{n}")
        buf.extend(chunk)
    return bytes(buf)

def set_pose(dst, q_wxyz, p_mj):
    q = mj_to_xr_quat(q_wxyz)
    p = mj_to_xr_pos(p_mj)
    dst.orientation.x, dst.orientation.y, dst.orientation.z, dst.orientation.w = map(float, q)
    dst.position.x, dst.position.y, dst.position.z = map(float, p)
    return q, p

PHASES = [
    ("stand", 2.0, None, 0.0),
    ("reach_left", 3.0, ("reach", "left"), 0.0),
    ("reach_right", 3.0, ("reach", "right"), 0.0),
    ("punch_left", 2.0, ("punch", "left"), 0.0),
    ("punch_right", 2.0, ("punch", "right"), 0.0),
    ("look_left", 2.0, None, 0.6),
    ("look_right", 2.0, None, -0.6),
    ("stand", 2.0, None, 0.0),
]
REACH = {"left": np.array([0.12, 0.38, 0.18]), "right": np.array([0.12, -0.38, 0.18])}
PUNCH = {"left": np.array([0.32, 0.14, 0.18]), "right": np.array([0.32, -0.14, 0.18])}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vendor/mujoco_menagerie/unitree_g1/scene.xml")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4242)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--no-net", action="store_true")
    ap.add_argument("--log", default="evidence/bridge.csv")
    ap.add_argument("--simshots-dir", default="")
    ap.add_argument("--max-joint-rate", type=float, default=2.0)
    args = ap.parse_args()
    assert_layout()
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)

    m = mujoco.MjModel.from_xml_path(args.model)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    sim_dt = float(m.opt.timestep)
    tick_dt = 1.0 / args.hz
    steps_per_tick = max(1, int(round(tick_dt / sim_dt)))
    tick_dt = steps_per_tick * sim_dt

    arm_names = {
        "left": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                 "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"],
        "right": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"],
    }
    arm_qadr = {s: np.array([m.jnt_qposadr[m.joint(n).id] for n in ns]) for s, ns in arm_names.items()}
    arm_dof = {s: np.array([m.jnt_dofadr[m.joint(n).id] for n in ns]) for s, ns in arm_names.items()}
    arm_jid = {s: np.array([m.joint(n).id for n in ns]) for s, ns in arm_names.items()}
    act_id = {m.actuator(i).name: i for i in range(m.nu)}
    arm_act = {s: np.array([act_id[n] for n in ns]) for s, ns in arm_names.items()}
    waist_act = act_id["waist_yaw_joint"]
    waist_qadr = m.jnt_qposadr[m.joint("waist_yaw_joint").id]

    wrist_bid = {s: m.body(f"{s}_wrist_yaw_link").id for s in ("left", "right")}
    torso_bid = m.body("torso_link").id
    head_mesh = m.mesh("head_link").id
    head_gid = next(g for g in range(m.ngeom) if m.geom_dataid[g] == head_mesh)

    default_arm = {s: np.array(d.qpos[arm_qadr[s]], copy=True) for s in ("left", "right")}
    q_target = {s: default_arm[s].copy() for s in ("left", "right")}
    waist_target = float(d.qpos[waist_qadr])

    sock = None
    sock_file = None
    if not args.no_net:
        sock = socket.create_connection((args.host, args.port), timeout=10)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        reset = recv_exact(sock, SIZE)
        latest = recv_exact(sock, SIZE)
        if struct.unpack_from("<Q", reset)[0] != HEADER or struct.unpack_from("<Q", latest)[0] != HEADER:
            sys.exit("handshake header mismatch")
        sock_file = sock.makefile("wb")
        print(f"CONNECTED {args.host}:{args.port} size={SIZE}", flush=True)

    log = open(args.log, "w", newline="", buffering=1)
    w = csv.writer(log)
    w.writerow(["wall_t", "sim_t", "phase", "phase_t", "mj_torso_yaw_deg", "packet_yaw_deg", "yaw_err_deg",
                "mj_head_x", "mj_head_y", "mj_head_z", "packet_head_x", "packet_head_y", "packet_head_z", "head_pos_err_m",
                "mj_left_x", "mj_left_y", "mj_left_z", "packet_left_x", "packet_left_y", "packet_left_z", "left_pos_err_m",
                "mj_right_x", "mj_right_y", "mj_right_z", "packet_right_x", "packet_right_y", "packet_right_z", "right_pos_err_m",
                "head_quat_err", "left_quat_err", "right_quat_err", "left_ik_residual_m", "right_ik_residual_m", "waist_target", "waist_actual", "base_x", "base_y", "base_z"])

    renderer = None
    simshot_phases = {"reach_left", "reach_right", "punch_left", "punch_right", "look_left", "look_right"}
    simshot_done = set()
    if args.simshots_dir:
        os.makedirs(args.simshots_dir, exist_ok=True)
        try:
            renderer = mujoco.Renderer(m, height=480, width=640)
        except Exception as e:
            print(f"SIMSHOT_DISABLED {e}", flush=True)
            renderer = None

    total = sum(p[1] for p in PHASES)
    duration = args.duration if args.duration > 0 else total + 2.0
    phase_idx = 0
    phase_start = 0.0
    phase_name = PHASES[0][0]
    print(f"PHASE {phase_name} sim_t={d.time:.3f} wall_t={time.time():.6f}", flush=True)

    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    max_step = args.max_joint_rate * tick_dt
    t_start = time.monotonic()
    wall0 = time.time()
    next_tick = t_start
    packet = blank_packet()
    n = 0
    try:
        while d.time < duration:
            while phase_idx < len(PHASES) and d.time >= phase_start + PHASES[phase_idx][1]:
                phase_start += PHASES[phase_idx][1]
                phase_idx += 1
                if phase_idx < len(PHASES):
                    phase_name = PHASES[phase_idx][0]
                    print(f"PHASE {phase_name} sim_t={d.time:.3f} wall_t={time.time():.6f}", flush=True)

            for _ in range(steps_per_tick):
                mujoco.mj_step(m, d)

            if phase_idx < len(PHASES):
                _, _, action, waist_des = PHASES[phase_idx]
            else:
                action, waist_des = None, 0.0

            torso_q = np.array(d.body(torso_bid).xquat, copy=True)
            torso_p = np.array(d.body(torso_bid).xpos, copy=True)
            head_p = np.array(d.geom_xpos[head_gid], copy=True)
            left_p = np.array(d.body(wrist_bid["left"]).xpos, copy=True)
            right_p = np.array(d.body(wrist_bid["right"]).xpos, copy=True)
            left_q = np.array(d.body(wrist_bid["left"]).xquat, copy=True)
            right_q = np.array(d.body(wrist_bid["right"]).xquat, copy=True)

            residuals = {"left": float("nan"), "right": float("nan")}
            for side in ("left", "right"):
                desired = default_arm[side].copy()
                if action is not None:
                    kind, which = action
                    if which == side:
                        local = REACH[side] if kind == "reach" else PUNCH[side]
                        target = torso_p + qrot(torso_q, local)
                        J = jacp[:, arm_dof[side]]
                        mujoco.mj_jacBody(m, d, jacp, jacr, wrist_bid[side])
                        J = jacp[:, arm_dof[side]]
                        e = target - np.array(d.body(wrist_bid[side]).xpos, copy=True)
                        A = J @ J.T + (0.05 ** 2) * np.eye(3)
                        dq = J.T @ np.linalg.solve(A, e)
                        nd = np.linalg.norm(dq)
                        if nd > 0.25:
                            dq *= 0.25 / nd
                        desired = np.array(d.qpos[arm_qadr[side]], copy=True) + dq
                        desired = np.clip(desired, m.jnt_range[arm_jid[side], 0], m.jnt_range[arm_jid[side], 1])
                        residuals[side] = float(np.linalg.norm(target - np.array(d.body(wrist_bid[side]).xpos, copy=True)))
                q_target[side] = q_target[side] + np.clip(desired - q_target[side], -max_step, max_step)
                q_target[side] = np.clip(q_target[side], m.jnt_range[arm_jid[side], 0], m.jnt_range[arm_jid[side], 1])
                d.ctrl[arm_act[side]] = q_target[side]

            waist_target += float(np.clip(waist_des - waist_target, -2.0 * tick_dt, 2.0 * tick_dt))
            d.ctrl[waist_act] = waist_target

            q_head, p_head = set_pose(packet.head.center, torso_q, head_p)
            q_left, p_left = set_pose(packet.left.pose, left_q, left_p)
            q_right, p_right = set_pose(packet.right.pose, right_q, right_p)
            packet.left.trigger_value.x = 1.0 if (action and action[0] == "punch" and action[1] == "left") else 0.0
            packet.right.trigger_value.x = 1.0 if (action and action[0] == "punch" and action[1] == "right") else 0.0
            for c in (packet.left, packet.right):
                for i in range(5):
                    c.hand_curl[i] = 0.3

            if sock_file is not None:
                sock_file.write(bytes(packet))
                sock_file.flush()

            pkt_yaw = math.degrees(yaw_y(q_head))
            mj_yaw = math.degrees(yaw_z(torso_q))
            head_err = float(np.linalg.norm(xr_to_mj_pos(p_head) - head_p))
            left_err = float(np.linalg.norm(xr_to_mj_pos(p_left) - left_p))
            right_err = float(np.linalg.norm(xr_to_mj_pos(p_right) - right_p))
            head_qerr = float(np.linalg.norm(np.array([packet.head.center.orientation.x, packet.head.center.orientation.y, packet.head.center.orientation.z, packet.head.center.orientation.w]) - q_head))
            left_qerr = float(np.linalg.norm(np.array([packet.left.pose.orientation.x, packet.left.pose.orientation.y, packet.left.pose.orientation.z, packet.left.pose.orientation.w]) - q_left))
            right_qerr = float(np.linalg.norm(np.array([packet.right.pose.orientation.x, packet.right.pose.orientation.y, packet.right.pose.orientation.z, packet.right.pose.orientation.w]) - q_right))
            w.writerow([f"{time.time():.6f}", f"{d.time:.6f}", phase_name, f"{d.time - phase_start:.3f}",
                        f"{mj_yaw:.6f}", f"{pkt_yaw:.6f}", f"{wrap_deg(pkt_yaw - mj_yaw):.6f}",
                        f"{head_p[0]:.6f}", f"{head_p[1]:.6f}", f"{head_p[2]:.6f}",
                        f"{p_head[0]:.6f}", f"{p_head[1]:.6f}", f"{p_head[2]:.6f}", f"{head_err:.6f}",
                        f"{left_p[0]:.6f}", f"{left_p[1]:.6f}", f"{left_p[2]:.6f}",
                        f"{p_left[0]:.6f}", f"{p_left[1]:.6f}", f"{p_left[2]:.6f}", f"{left_err:.6f}",
                        f"{right_p[0]:.6f}", f"{right_p[1]:.6f}", f"{right_p[2]:.6f}",
                        f"{p_right[0]:.6f}", f"{p_right[1]:.6f}", f"{p_right[2]:.6f}", f"{right_err:.6f}",
                        f"{head_qerr:.6f}", f"{left_qerr:.6f}", f"{right_qerr:.6f}",
                        f"{residuals['left']:.6f}", f"{residuals['right']:.6f}",
                        f"{waist_target:.6f}", f"{float(d.qpos[waist_qadr]):.6f}",
                        f"{d.qpos[0]:.6f}", f"{d.qpos[1]:.6f}", f"{d.qpos[2]:.6f}"])
            n += 1

            if renderer is not None and phase_name in simshot_phases and phase_name not in simshot_done and d.time - phase_start > 0.8:
                try:
                    cam = mujoco.MjvCamera()
                    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    cam.lookat[:] = [0.0, 0.0, 0.95]
                    cam.distance = 2.3
                    cam.azimuth = 140
                    cam.elevation = -12
                    renderer.update_scene(d, camera=cam)
                    import imageio.v3 as iio
                    iio.imwrite(os.path.join(args.simshots_dir, f"sim_{phase_name}.png"), renderer.render())
                    simshot_done.add(phase_name)
                    print(f"SIMSHOT {phase_name} sim_t={d.time:.3f}", flush=True)
                except Exception as e:
                    print(f"SIMSHOT_ERROR {e}", flush=True)
                    simshot_done.add(phase_name)

            next_tick += tick_dt
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        log.close()
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass
        if sock is not None:
            s = sock
            sock = None
            try:
                s.close()
            except Exception:
                pass
        print(f"SENT {n} packets sim_t={d.time:.3f} wall_end={time.time():.6f}", flush=True)

if __name__ == "__main__":
    main()
