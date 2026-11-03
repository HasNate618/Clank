"""Does the player scale actually shorten the reach? One command, then look.

The arithmetic got ambiguous: dividing the measured depth by the scale predicted a shorter
reach, but the card measured *further* (0.556 m against 0.486). That contradiction means one
of the assumptions about what the game does with the reported head height is wrong, and the
only way to settle it is to move the arm and look at where the hand lands.

The test is the one the whole change exists for: command a reach that would be too short
unscaled (the card used to sit 0.486 m out) and see whether the hand arrives at the card.
If the player really is 1.43x bigger, 0.34 m of arm covers 0.486 m of room.
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
from clank.projection import Fov, project_point  # noqa: E402
from clank.workspace import clamp_offset  # noqa: E402

SOCK = Path.home() / ".local/state/Clank/sessions/body/control.sock"
STATE = Path.home() / ".local/state/Clank/sessions/body/body_state.json"
FOV = Fov.from_degrees(70.0, 70.0, 57.11, 57.11)
FLAG_FALL_RISK = 1 << 2
MAX_M = 0.40
MAX_LATERAL_M = 0.18


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


def main() -> int:
    px, py = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (755, 385)
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.43
    up_bias = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0

    send(verb="look", yaw_deg=0.0)
    time.sleep(2.5)
    send(verb="reach", side="right", forward_m=0.0, lateral_m=0.0, up_m=0.0,
         look_yaw_deg=0.0)
    time.sleep(2.5)
    rest = tuple(state()["right_hand"]["p"])

    with Stream() as stream:
        before = stream.read(timeout=2.0)
    cw, ch = int(before["color_width"]), int(before["color_height"])
    head = Pose(p=tuple(state()["head"]["p"]), q_xyzw=tuple(state()["head"]["q_xyzw"]))
    raw = sample_depth(before["depth"], before["width"], before["height"], px / cw, py / ch)
    card = point_from_pixel(before, px / cw, py / ch, head, FOV, scale=scale)
    if card is None:
        print("no surface there")
        return 1
    delta = [card[k] - rest[k] for k in range(3)]
    want = math.dist(rest, card)
    print(f"card raw depth {raw:.3f} m; point {[round(v, 3) for v in card]}")
    print(f"reach demanded {want:.3f} m at scale {scale}")

    offset = clamp_offset(*delta, max_m=MAX_M, max_lateral_m=MAX_LATERAL_M)
    if any(abs(offset[k] - delta[k]) > 1e-9 for k in range(3)):
        print(f"  clamped: {[round(v, 3) for v in delta]} -> "
              f"{[round(v, 3) for v in offset]} (magnitude {math.dist((0,0,0), offset):.3f})")

    send(verb="reach", side="right", forward_m=round(offset[0], 4),
         lateral_m=round(offset[1], 4), up_m=round(offset[2] + up_bias, 4),
         look_yaw_deg=0.0)
    time.sleep(2.2)
    got = tuple(state()["right_hand"]["p"])
    print(f"  wrist went to {[round(v, 3) for v in got]} in ONE command "
          f"(flags {state().get('flags')})")
    if bool(int(state().get("flags") or 0) & FLAG_FALL_RISK):
        print("  FALL RISK")

    with Stream() as stream:
        after = stream.read(timeout=2.0)
    cw, ch = int(after["color_width"]), int(after["color_height"])
    head = Pose(p=tuple(state()["head"]["p"]), q_xyzw=tuple(state()["head"]["q_xyzw"]))
    hand_px = project_point(head, got, FOV)
    card_px = project_point(head, card, FOV)
    print(f"  hand at pixel {None if hand_px is None else (round(hand_px[0]*cw), round(hand_px[1]*ch))}, "
          f"card at {None if card_px is None else (round(card_px[0]*cw), round(card_px[1]*ch))}")

    from PIL import Image

    import numpy as np

    rgba = np.frombuffer(bytes(after["color"]), dtype=np.uint8).reshape(ch, cw, 4)
    img = Image.fromarray(rgba[..., :3])
    img.save("/tmp/scaled_after.jpg", quality=93)
    if hand_px is not None:
        hx, hy = int(hand_px[0] * cw), int(hand_px[1] * ch)
        img.crop((max(0, hx - 200), max(0, hy - 160), min(cw, hx + 200),
                  min(ch, hy + 200))).save("/tmp/scaled_hand.jpg", quality=93)
    print("  wrote /tmp/scaled_after.jpg and /tmp/scaled_hand.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
