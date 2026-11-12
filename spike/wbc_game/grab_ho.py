"""Reach the card with the scaled player, close on it, and bring it in front of the face.

The emulated hand is what the game draws and what counts: it sits at
`pivot + scale * (wrist - pivot)`, so a robot wrist 0.17 m short of the card is still the
game's hand *on* the card. That is the whole point of the scale, and it is also the caveat:
MuJoCo still shows the wrist short.

Bringing the card "in front of the face" is expressed in the game's frame too, because the
camera is the head: pick a point 0.33 m along the view axis and 0.20 m below it, then ask the
robot for `pivot + (that - pivot) / scale`, which is the wrist position whose *emulated* hand
lands there. Retracting by eye would land somewhere else entirely.
"""

from __future__ import annotations

import json
import math
import socket
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from clank.body import Pose  # noqa: E402
from clank.depth_stream import Stream  # noqa: E402
from clank.pixel3d import point_from_pixel  # noqa: E402
from clank.projection import Fov, project_point  # noqa: E402
from clank.workspace import clamp_offset  # noqa: E402

SOCK = Path.home() / ".local/state/Clank/sessions/body/control.sock"
STATE = Path.home() / ".local/state/Clank/sessions/body/body_state.json"
FOV = Fov.from_degrees(70.0, 70.0, 57.11, 57.11)
FLAG_FALL_RISK = 1 << 2
SCALE = 1.43
PIVOT = (-0.263, -0.150, 0.584)
MAX_M = 0.56
MAX_LATERAL_M = 0.34


def send(**kw: object) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(2.0)
        s.connect(str(SOCK))
        s.sendall((json.dumps(kw) + "\n").encode())
        try:
            return json.loads(s.recv(255).decode())
        except (TimeoutError, json.JSONDecodeError):
            return {}


def state() -> dict:
    return json.loads(STATE.read_text())


def wrist() -> tuple[float, float, float]:
    return tuple(state()["right_hand"]["p"])


def rotate(q, p):
    w, x, y, z = q
    t = (2 * (y * p[2] - z * p[1]), 2 * (z * p[0] - x * p[2]), 2 * (x * p[1] - y * p[0]))
    return (
        p[0] + w * t[0] + (y * t[2] - z * t[1]),
        p[1] + w * t[1] + (z * t[0] - x * t[2]),
        p[2] + w * t[2] + (x * t[1] - y * t[0]),
    )


def to_world(p, st):
    r = rotate(st["root_q"], p)
    root = st["root_p"]
    return (root[0] + r[0], root[1] + r[1], root[2] + r[2])


def to_rel(p, st):
    root = st["root_p"]
    d = (p[0] - root[0], p[1] - root[1], p[2] - root[2])
    w, x, y, z = st["root_q"]
    return rotate((w, -x, -y, -z), d)


def emulate(hand_world):
    return tuple(PIVOT[k] + SCALE * (hand_world[k] - PIVOT[k]) for k in range(3))


def stream_to(goal, rest, limit_s=10.0, tol=0.04, hz=8.0):
    """Drive the wrist to `goal` with the damped integrator; returns (err, offset)."""
    dt = 1.0 / hz
    offset = [0.0, 0.0, 0.0]
    t0 = time.monotonic()
    best = float("inf")
    while time.monotonic() - t0 < limit_s:
        if bool(int(state().get("flags") or 0) & FLAG_FALL_RISK):
            print("    FALL RISK - stopping")
            return best, offset
        got = wrist()
        err = math.dist(got, goal)
        best = min(best, err)
        if err < tol:
            break
        offset = [offset[k] + 0.7 * (goal[k] - got[k]) for k in range(3)]
        offset = list(
            clamp_offset(*offset, max_m=MAX_M, max_lateral_m=MAX_LATERAL_M)
        )
        send(verb="reach", side="right", forward_m=round(offset[0], 4),
             lateral_m=round(offset[1], 4), up_m=round(offset[2], 4),
             look_yaw_deg=0.0)
        time.sleep(dt)
    return best, offset


def main() -> int:
    px, py = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (800, 410)

    send(verb="look", yaw_deg=0.0)
    time.sleep(2.5)
    send(verb="reach", side="right", forward_m=0.0, lateral_m=0.0, up_m=0.0,
         look_yaw_deg=0.0)
    time.sleep(2.5)
    rest = wrist()
    st = state()
    head = Pose(p=tuple(st["head"]["p"]), q_xyzw=tuple(st["head"]["q_xyzw"]))

    with Stream() as stream:
        frame = stream.read(timeout=2.0)
    if frame is None:
        print("no depth frame")
        return 1
    cw, ch = int(frame["color_width"]), int(frame["color_height"])
    obj = point_from_pixel(frame, px / cw, py / ch, head, FOV)
    if obj is None:
        print("no surface there")
        return 1
    need = math.dist(rest, obj)
    goal = [rest[k] + (obj[k] - rest[k]) / SCALE for k in range(3)]
    print(f"card {[round(v, 3) for v in obj]}; |O-H| {need:.3f} m -> "
          f"{need / SCALE:.3f} m of arm")

    print("  reaching")
    err, offset = stream_to(goal, rest, limit_s=12.0)
    st = state()
    gw = to_world(wrist(), st)
    print(f"    wrist err {err:.3f} m; emulated hand "
          f"{[round(v, 3) for v in emulate(gw)]}, "
          f"{math.dist(emulate(gw), to_world(obj, st)):.3f} m from the card in the game")

    print("  closing:", send(verb="close_right"))
    time.sleep(1.4)
    print(f"    closure {state().get('closure')}")

    # Where the emulated hand should end up: 0.33 m along the view axis, 0.20 m below it.
    st = state()
    cam = to_world(tuple(st["head"]["p"]), st)
    target_game = (cam[0] + 0.33, cam[1], cam[2] - 0.20)
    robot_target_world = tuple(
        PIVOT[k] + (target_game[k] - PIVOT[k]) / SCALE for k in range(3)
    )
    robot_goal = to_rel(robot_target_world, st)
    print(f"  bringing it up: emulated hand to {[round(v,3) for v in target_game]}, "
          f"wrist to {[round(v,3) for v in robot_goal]}")
    k_off = list(
        clamp_offset(*[robot_goal[k] - rest[k] for k in range(3)],
                     max_m=MAX_M, max_lateral_m=MAX_LATERAL_M)
    )
    want = [rest[k] + k_off[k] for k in range(3)]
    err2, _ = stream_to(want, rest, limit_s=12.0)
    st = state()
    gw = to_world(wrist(), st)
    print(f"    wrist err {err2:.3f} m; emulated hand "
          f"{[round(v, 3) for v in emulate(gw)]}")

    with Stream() as stream:
        after = stream.read(timeout=2.0)
    if after is None:
        print("no depth frame after the reach")
        return 1
    cw, ch = int(after["color_width"]), int(after["color_height"])
    head = Pose(p=tuple(state()["head"]["p"]), q_xyzw=tuple(state()["head"]["q_xyzw"]))
    from PIL import Image

    import numpy as np

    rgba = np.frombuffer(bytes(after["color"]), dtype=np.uint8).reshape(ch, cw, 4)
    img = Image.fromarray(rgba[..., :3])
    img.save("/tmp/grab_ho_full.jpg", quality=93)
    img.crop((cw // 2 - 260, ch // 2 - 120, cw // 2 + 260, ch // 2 + 300)).save(
        "/tmp/grab_ho_face.jpg", quality=93
    )
    px_hand = project_point(head, wrist(), FOV)
    print(f"    sim wrist projects to "
          f"{None if px_hand is None else (round(px_hand[0]*cw), round(px_hand[1]*ch))}")
    print("  wrote /tmp/grab_ho_full.jpg and /tmp/grab_ho_face.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
