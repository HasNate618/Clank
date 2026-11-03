"""Reach the card, close on it, and bring it up in front of the face so it is visible.

The camera *is* the head, so "in front of the face" means the hand comes back toward the
eye line, where it fills the middle of the view. Everything is staged and every stage
checks `FLAG_FALL_RISK`, because an earlier unclamped command in this project put the body
on the floor.

Success is judged by measurement wherever possible: the depth at the card's original pixel
is re-sampled at the end. If the card left with the hand, that pixel now sees whatever was
behind it - or the hand itself.
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
from clank.pixel3d import point_from_pixel, sample_depth  # noqa: E402
from clank.projection import Fov  # noqa: E402
from clank.workspace import clamp_offset  # noqa: E402

SOCK = Path.home() / ".local/state/Clank/sessions/body/control.sock"
STATE = Path.home() / ".local/state/Clank/sessions/body/body_state.json"
FOV = Fov.from_degrees(70.0, 70.0, 57.11, 57.11)
FLAG_FALL_RISK = 1 << 2
RUN_MAX_M = 0.56
RUN_MAX_LATERAL_M = 0.34
PALM_LEAD_M = 0.08
#: Where to hold the grabbed card so the head camera sees it: forward, slightly to the
#: robot's right, a little below the eye line.
DISPLAY = (0.28, -0.16, 0.26)


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


def risky() -> bool:
    return bool(int(state().get("flags") or 0) & FLAG_FALL_RISK)


def drive_to(goal, rest, yaw_deg: float, label: str, tries: int = 2) -> bool:
    """Stage from the current pose to `goal` with feedback. False if fall risk appears."""
    current = wrist()
    for stage, frac in enumerate((0.4, 0.65, 0.85, 1.0, 1.0), start=1):
        if risky():
            print(f"    {label} {stage}: FALL RISK - stop")
            return False
        point = [current[k] + frac * (goal[k] - current[k]) for k in range(3)]
        offset = clamp_offset(
            *[point[k] - rest[k] for k in range(3)],
            max_m=RUN_MAX_M,
            max_lateral_m=RUN_MAX_LATERAL_M,
        )
        for i in range(tries):
            send(verb="reach", side="right", forward_m=round(offset[0], 4),
                 lateral_m=round(offset[1], 4), up_m=round(offset[2], 4),
                 look_yaw_deg=yaw_deg)
            time.sleep(1.05)
            got = wrist()
            err = math.dist(point, got)
            if risky():
                print(f"    {label} {stage}.{i}: FALL RISK - stop")
                return False
            if err < 0.05:
                break
            offset = [offset[k] + 0.7 * (point[k] - got[k]) for k in range(3)]
            offset = list(
                clamp_offset(
                    *offset, max_m=RUN_MAX_M, max_lateral_m=RUN_MAX_LATERAL_M
                )
            )
        print(f"    {label} {stage}: wrist ({got[0]:+.3f},{got[1]:+.3f},{got[2]:+.3f}) "
              f"err {err:.3f} m")
    return True


def main() -> int:
    px, py = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (782, 432)
    up_bias = float(sys.argv[3]) if len(sys.argv) > 3 else 0.07

    send(verb="look", yaw_deg=0.0)
    time.sleep(2.5)
    send(verb="reach", side="right", forward_m=0.0, lateral_m=0.0, up_m=0.0,
         look_yaw_deg=0.0)
    time.sleep(2.5)
    rest = wrist()

    with Stream() as stream:
        frame = stream.read(timeout=2.0)
    if frame is None:
        print("no depth frame")
        return 1
    cw, ch = int(frame["color_width"]), int(frame["color_height"])
    u, v = px / cw, py / ch
    head = Pose(p=tuple(state()["head"]["p"]), q_xyzw=tuple(state()["head"]["q_xyzw"]))
    card = point_from_pixel(frame, u, v, head, FOV)
    behind = sample_depth(frame["depth"], frame["width"], frame["height"], u, v)
    if card is None:
        print("no surface at that pixel")
        return 1
    dist = math.dist(rest, card)
    unit = [(card[k] - rest[k]) / dist for k in range(3)]
    aim = [card[k] - PALM_LEAD_M * unit[k] for k in range(3)]
    aim[2] += up_bias
    print(f"card at {dist:.3f} m (depth {behind:.3f}); aiming wrist at "
          f"({aim[0]:+.3f},{aim[1]:+.3f},{aim[2]:+.3f})")

    print("  approaching")
    if not drive_to(aim, rest, 0.0, "approach"):
        return 1
    print(f"    wrist -> card {math.dist(wrist(), card):.3f} m")

    print("  closing:", send(verb="close_right"))
    time.sleep(1.3)
    print(f"    closure {state().get('closure')}")

    print("  bringing it up in front of the face")
    if not drive_to(list(DISPLAY), rest, 0.0, "present"):
        return 1
    hold = [DISPLAY[k] - rest[k] for k in range(3)]
    hold = list(clamp_offset(*hold, max_m=RUN_MAX_M, max_lateral_m=RUN_MAX_LATERAL_M))
    send(verb="reach", side="right", forward_m=round(hold[0], 4),
         lateral_m=round(hold[1], 4), up_m=round(hold[2], 4), look_yaw_deg=0.0)

    with Stream() as stream:
        shown = stream.read(timeout=2.0)
    if shown is None:
        print("no frame for the reveal")
        return 1
    now_behind = sample_depth(shown["depth"], shown["width"], shown["height"], u, v)
    print(f"  depth at the card's old pixel: was {behind:.3f} m, now {now_behind}")
    print(f"  closure {state().get('closure')}, wrist "
          f"{[round(x, 3) for x in wrist()]}, flags {state().get('flags')}")

    from PIL import Image

    import numpy as np

    scw, sch = int(shown["color_width"]), int(shown["color_height"])
    rgba = np.frombuffer(bytes(shown["color"]), dtype=np.uint8).reshape(sch, scw, 4)
    img = Image.fromarray(rgba[..., :3])
    img.save("/tmp/present_full.jpg", quality=93)
    img.crop((scw // 2 - 230, sch // 2 - 170, scw // 2 + 230, sch // 2 + 190)).resize(
        (460, 360)
    ).save("/tmp/present_crop.jpg", quality=93)
    print("  wrote /tmp/present_full.jpg and /tmp/present_crop.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
