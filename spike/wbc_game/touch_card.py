"""Put the hand on the card.

The card is 0.475 m from the wrist rest, which is past the 0.40 m safety clamp, so this
run deliberately exceeds it - in **stages**, checking `FLAG_FALL_RISK` at every step, so a
bad response stops the hand instead of putting the body on the floor. The 0.76 m lateral
request that already fell the robot is still far outside anything sent here.

Two physical details the previous attempts got wrong:

* the wrist is what we command, but the hand extends ~0.10 m beyond it, so the wrist is
  aimed 0.08 m **short** of the card surface along the approach, which is where the palm
  ends up;
* a grab also needs the **palm facing the object** (`hand_target(yaw_deg=...)` rotates the
  wrist about Z). That is not solved here - this run answers only "can it touch".
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
#: Wider than the standing cap, still far inside the 0.76 m that fell the body.
RUN_MAX_M = 0.56
RUN_MAX_LATERAL_M = 0.34
#: Where the palm sits relative to the wrist joint we command.
PALM_LEAD_M = 0.08


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


def risk() -> bool:
    return bool(int(state().get("flags") or 0) & FLAG_FALL_RISK)


def main() -> int:
    px, py = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (782, 432)
    yaw_deg = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    #: The hand hangs off the wrist along the wrist's own axis, not along the approach,
    #: so the palm lands low unless the wrist is biased up. Measured visually, one run.
    up_bias = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0

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
    depth = sample_depth(frame["depth"], frame["width"], frame["height"], u, v)
    if card is None or depth is None:
        print("no surface at that pixel")
        return 1
    to_card = [card[k] - rest[k] for k in range(3)]
    dist = math.dist(rest, card)
    unit = [c / dist for c in to_card]
    aim_point = [card[k] - PALM_LEAD_M * unit[k] for k in range(3)]
    aim_point[2] += up_bias
    print(f"card ({card[0]:+.3f},{card[1]:+.3f},{card[2]:+.3f}) at {dist:.3f} m "
          f"(depth {depth:.3f})")
    print(f"  wrist aim {[round(x, 3) for x in aim_point]} "
          f"({math.dist(rest, aim_point):.3f} m from rest)")

    target = list(aim_point)
    current = [0.0, 0.0, 0.0]
    for stage, frac in enumerate((0.55, 0.75, 0.9, 1.0, 1.0, 1.0), start=1):
        if risk():
            print(f"  stage {stage}: FALL RISK - stopping here")
            break
        goal = [rest[k] + frac * (target[k] - rest[k]) for k in range(3)]
        delta = [goal[k] - rest[k] for k in range(3)]
        offset = clamp_offset(*delta, max_m=RUN_MAX_M, max_lateral_m=RUN_MAX_LATERAL_M)
        for i in range(2):
            send(verb="reach", side="right", forward_m=round(offset[0], 4),
                 lateral_m=round(offset[1], 4), up_m=round(offset[2], 4),
                 look_yaw_deg=yaw_deg)
            time.sleep(1.05)
            got = wrist()
            err = math.dist(goal, got)
            print(f"  stage {stage}.{i}: wrist ({got[0]:+.3f},{got[1]:+.3f},"
                  f"{got[2]:+.3f}) err {err:.3f} m flags {state().get('flags')}")
            if risk():
                print("  FALL RISK - stopping")
                return 1
            if err < 0.05:
                break
            offset = [offset[k] + 0.7 * (goal[k] - got[k]) for k in range(3)]
            offset = list(
                clamp_offset(*offset, max_m=RUN_MAX_M, max_lateral_m=RUN_MAX_LATERAL_M)
            )
        current = list(offset)
        gap = math.dist(wrist(), card)
        print(f"    wrist -> card now {gap:.3f} m "
              f"(palm is ~{PALM_LEAD_M:.2f} m beyond it)")

    final = wrist()
    print(f"final wrist ({final[0]:+.3f},{final[1]:+.3f},{final[2]:+.3f}), "
          f"card is {math.dist(final, card):.3f} m away, closure {state().get('closure')}")

    from PIL import Image

    import numpy as np

    # A frame captured *before* the arm moved shows no hand. Read one now, and hold the
    # pose while it is read, or the TTL expires and the arm relaxes back to the clip.
    send(verb="reach", side="right", forward_m=round(current[0], 4),
         lateral_m=round(current[1], 4), up_m=round(current[2], 4), look_yaw_deg=yaw_deg)
    with Stream() as stream:
        shown = stream.read(timeout=2.0) or frame
    print(f"  fresh frame {shown['frame_id']}, wrist now {[round(x, 3) for x in wrist()]}")
    frame = shown
    cw, ch = int(frame["color_width"]), int(frame["color_height"])
    rgba = np.frombuffer(bytes(frame["color"]), dtype=np.uint8).reshape(ch, cw, 4)
    img = Image.fromarray(rgba[..., :3])
    cx, cy = int(u * cw), int(v * ch)
    box = (max(0, cx - 130), max(0, cy - 130), min(cw, cx + 130), min(ch, cy + 130))
    img.crop(box).resize((420, 420)).save("/tmp/touch_card.jpg", quality=93)
    print("  crop /tmp/touch_card.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
