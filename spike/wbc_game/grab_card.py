"""Aim at a marked pixel, close, pull, and ask whether the object moved with the hand.

The whole point of this one: the card is 0.492 m from the wrist and the safe clamp is
0.40 m, so the aim stops ~9 cm short *by design*. Whether that is close enough is a
property of the game's hover radius, not of our arithmetic, so the only way to know is to
click and look.

The target point comes from the depth stream once, at the pinned camera, and is then held
in the root frame (`clank/pixel3d.py`); the camera cannot drift while the head is pinned.

Judged by measurement, not by hope: the card's patch is compared before and after. If the
card left with the hand the patch changes; if it is still there, the patch does not.
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


def fall_risk() -> bool:
    return bool(int(state().get("flags") or 0) & FLAG_FALL_RISK)


def frame_now():
    with Stream() as stream:
        return stream.read(timeout=2.0)


def card_patch(frame, u: float, v: float, radius: int = 40) -> list[int]:
    """The colour bytes in a square around normalised (u, v), for a before/after diff."""
    cw, ch = int(frame["color_width"]), int(frame["color_height"])
    cx, cy = int(u * cw), int(v * ch)
    out: list[int] = []
    for y in range(max(0, cy - radius), min(ch, cy + radius)):
        base = y * cw * 4
        for x in range(max(0, cx - radius), min(cw, cx + radius)):
            i = base + x * 4
            out.extend(frame["color"][i : i + 3])
    return out


def main() -> int:
    px, py = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (782, 432)
    label = sys.argv[3] if len(sys.argv) > 3 else "card"

    send(verb="look", yaw_deg=0.0)
    time.sleep(2.5)
    send(verb="reach", side="right", forward_m=0.0, lateral_m=0.0, up_m=0.0,
         look_yaw_deg=0.0)
    time.sleep(2.5)
    rest = wrist()

    frame = frame_now()
    if frame is None:
        print("no depth frame")
        return 1
    cw, ch = int(frame["color_width"]), int(frame["color_height"])
    u, v = px / cw, py / ch
    head = Pose(p=tuple(state()["head"]["p"]), q_xyzw=tuple(state()["head"]["q_xyzw"]))
    depth = sample_depth(frame["depth"], frame["width"], frame["height"], u, v)
    point = point_from_pixel(frame, u, v, head, FOV)
    if point is None or depth is None:
        print("no surface at that pixel")
        return 1
    delta = [point[k] - rest[k] for k in range(3)]
    aim = clamp_offset(*delta)
    short = math.dist(rest, point) - math.dist(rest, [rest[k] + aim[k] for k in range(3)])
    print(f"{label}: pixel ({px},{py}) depth {depth:.3f} m -> "
          f"({point[0]:+.3f},{point[1]:+.3f},{point[2]:+.3f})")
    print(f"  rest ({rest[0]:+.3f},{rest[1]:+.3f},{rest[2]:+.3f}), "
          f"target {math.dist(rest, point):.3f} m away, aiming {math.dist(rest, aim):.3f} m "
          f"-> {short * 100:.0f} cm short")
    before = card_patch(frame, u, v)

    offset = list(aim)
    for i in range(4):
        if fall_risk():
            print("  FALL RISK - aborting")
            return 1
        send(verb="reach", side="right", forward_m=round(offset[0], 4),
             lateral_m=round(offset[1], 4), up_m=round(offset[2], 4), look_yaw_deg=0.0)
        time.sleep(1.05)
        got = wrist()
        err = math.dist([rest[k] + aim[k] for k in range(3)], got)
        print(f"  {i}: wrist ({got[0]:+.3f},{got[1]:+.3f},{got[2]:+.3f}) "
              f"err to aim {err:.3f} m  flags {state().get('flags')}")
        if err < 0.05:
            break
        want = [rest[k] + aim[k] for k in range(3)]
        offset = [offset[k] + 0.7 * (want[k] - got[k]) for k in range(3)]
        offset = list(clamp_offset(*offset))

    print("  closing:", send(verb="close_right"))
    time.sleep(1.2)
    if fall_risk():
        print("  FALL RISK - aborting")
        return 1
    closed = wrist()
    print(f"  closed at ({closed[0]:+.3f},{closed[1]:+.3f},{closed[2]:+.3f}), "
          f"closure {state().get('closure')}")

    # Pull back along -forward, keeping the pin so the camera does not move.
    pull = [offset[0] - 0.12, offset[1], offset[2] + 0.02]
    pull = list(clamp_offset(*pull))
    send(verb="reach", side="right", forward_m=round(pull[0], 4),
         lateral_m=round(pull[1], 4), up_m=round(pull[2], 4), look_yaw_deg=0.0)
    time.sleep(2.0)
    print(f"  pulled to {[round(x, 3) for x in wrist()]}  closure {state().get('closure')}")

    after_frame = frame_now()
    if after_frame is None:
        print("no depth frame after the pull")
        return 1
    after = card_patch(after_frame, u, v)
    n = min(len(before), len(after))
    changed = sum(1 for i in range(n) if abs(before[i] - after[i]) > 40)
    print(f"  patch diff: {100.0 * changed / n:.1f}% of channel samples changed by >40")
    print("  (the hand also crosses the patch, so a change is not proof - the crops are)")

    from PIL import Image

    def crop(frame):
        import numpy as np

        cw, ch = int(frame["color_width"]), int(frame["color_height"])
        rgba = np.frombuffer(bytes(frame["color"]), dtype=np.uint8).reshape(ch, cw, 4)
        img = Image.fromarray(rgba[..., :3])
        cx, cy = int(u * cw), int(v * ch)
        return img.crop((cx - 90, cy - 90, cx + 90, cy + 90)).resize((360, 360))

    pair = Image.new("RGB", (728, 360), (10, 10, 10))
    pair.paste(crop(frame), (0, 0))
    pair.paste(crop(after_frame), (368, 0))
    pair.save(f"/tmp/grab_{label}.jpg", quality=92)
    print(f"  crops: /tmp/grab_{label}.jpg  (left = before, right = after)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
