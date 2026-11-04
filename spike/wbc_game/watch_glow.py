"""Watch one object in the game view: where it is, and whether it is glowing.

Run this beside a reach. It follows the object by prediction (the object is fixed, the
camera is measured every tick), samples a patch at the predicted pixel, and reports the
game's own "grabbable" answer as a glow state. The hand is followed the same way, so the
log carries the thing the eye actually judges - **how many pixels apart are the hand and
the object** - next to the state the game is showing.

    watch_glow.py --px 870 460 --seconds 20
    watch_glow.py --world 0.34 -0.41 0.12 --seconds 20 --hz 4

Why a separate tool: the reach cannot be its own witness. It reports the residual between
the hand it aimed at and the hand it achieved, which stayed small while the camera was
drifting 1009 -> 1122 -> 865 -> 1007 px across the card. This measures the object and the
hand through the *same* camera, so a reach that converges on a point the camera has left
is visible here and nowhere else.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from PIL import Image

from clank.depth_stream import Stream
from clank.pixel3d import point_from_pixel, sample_depth
from clank.projection import Fov, project_point

import track as T

FOV = Fov.from_degrees(70.0, 70.0, 57.11, 57.11)

#: Pixel separation under which the hand counts as covering the object, so the glow state
#: is "occluded" rather than "off". Half the patch window: any closer and the hand is
#: inside the sample.
COVER_PX = 40.0


def state_path() -> pathlib.Path:
    return pathlib.Path.home() / ".local/state/Clank/sessions/body/body_state.json"


def read_state() -> dict:
    return json.loads(state_path().read_text())


def resolve_world(frame: dict, st: dict, px: float, py: float) -> tuple[float, float, float]:
    """The object's pelvis-frame point, from the pixel, three depth samples, median."""
    cw, ch = int(frame["color_width"]), int(frame["color_height"])
    from clank.body import Pose

    head = Pose(p=tuple(st["head"]["p"]), q_xyzw=tuple(st["head"]["q_xyzw"]))
    points = []
    with Stream() as stream:
        for _ in range(3):
            f = stream.read(timeout=2.0)
            if f is None:
                continue
            p = point_from_pixel(f, px / cw, py / ch, head, FOV)
            if p is not None:
                points.append(p)
            time.sleep(0.1)
    if not points:
        raise SystemExit(f"no surface at ({px:.0f}, {py:.0f})")
    return tuple(sorted(q[k] for q in points)[len(points) // 2] for k in range(3))


def rgb_of(frame: dict) -> np.ndarray:
    cw, ch = int(frame["color_width"]), int(frame["color_height"])
    img = Image.frombytes("RGBA", (cw, ch), bytes(frame["color"])).convert("RGB")
    return np.asarray(img)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--px", nargs=2, type=float, metavar=("X", "Y"))
    ap.add_argument("--world", nargs=3, type=float, metavar=("X", "Y", "Z"))
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--hz", type=float, default=4.0)
    ap.add_argument("--side", default="right", choices=("left", "right"))
    args = ap.parse_args()
    if args.px is None and args.world is None:
        ap.error("give --px X Y (to resolve from the view) or --world X Y Z")

    st = read_state()
    with Stream() as stream:
        frame = stream.read(timeout=2.0)
    if frame is None:
        raise SystemExit("no depth frame")
    if args.world is not None:
        point_rel = tuple(args.world)
        source = "--world"
    else:
        point_rel = resolve_world(frame, st, args.px[0], args.px[1])
        source = f"pixel ({args.px[0]:.0f}, {args.px[1]:.0f})"
    print(f"object at {[round(v, 3) for v in point_rel]} (pelvis frame) from {source}", flush=True)

    watch = T.GlowWatch()
    previous = None
    end = time.monotonic() + args.seconds
    n = 0
    while time.monotonic() < end:
        n += 1
        st = read_state()
        with Stream() as stream:
            frame = stream.read(timeout=2.0)
        if frame is None:
            continue
        rgb = rgb_of(frame)
        head_p = tuple(st["head"]["p"])
        head_q = tuple(st["head"]["q_xyzw"])
        seen = T.follow(rgb, point_rel, head_p, head_q, FOV, previous)
        previous = seen.patch.pixel
        hand_p = T.predict(head_p, head_q, tuple(st[f"{args.side}_hand"]["p"]), FOV,
                           (rgb.shape[1], rgb.shape[0]))
        gap = float("nan") if hand_p is None else math.dist(hand_p, seen.patch.pixel)
        glow = watch.update(seen.patch, occluded=gap < COVER_PX)
        print(
            f"{n:3d}  object ({seen.patch.pixel[0]:4.0f},{seen.patch.pixel[1]:4.0f})"
            f"  hand {'off-frame' if hand_p is None else f'({hand_p[0]:4.0f},{hand_p[1]:4.0f})'}"
            f"  gap {gap:6.1f} px  glow {seen.patch.glow:6.1f}  state {glow}",
            flush=True,
        )
        time.sleep(max(0.0, 1.0 / args.hz))

    print(json.dumps({"object": point_rel, "watch": watch.to_dict()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
