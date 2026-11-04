"""Capture what the robot sees, faded to one distance, as a still for the model.

The player only ever interacts with what is in front of it, and the game fills the rest of
the frame with detail it will never touch - signage, a distant counter, a wall of white
screens. That detail is what the model keeps trying to interpret: the same grounding call
scores 39-65 px on a crop and 340-360 px on the full frame, and the crop's only real
advantage is that it removes the room.

This produces the frame a crop would have produced, from depth instead of a hand-drawn box.
It is **not** a crop: the model still sees the full field of view, so anything at the
object's own distance stays, and the connected component containing the object is the whole
console (`research/depth-fade.md`). What goes away is the far room.

    see.py --px 710 412 --out /tmp/see.jpg
    see.py --world 0.34 -0.41 0.12 --out /tmp/see.jpg   # resolves the pixel first
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from PIL import Image

from clank.depth_stream import Stream
from clank.projection import Fov, project_point
from clank.body import Pose

import fade as F

FOV = Fov.from_degrees(70.0, 70.0, 57.11, 57.11)


def state_path() -> pathlib.Path:
    return pathlib.Path.home() / ".local/state/Clank/sessions/body/body_state.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--px", nargs=2, type=float, metavar=("X", "Y"), help="object pixel")
    ap.add_argument("--world", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    help="object point in the pelvis frame, projected to a pixel")
    ap.add_argument("--near", type=float, default=F.NEAR_K)
    ap.add_argument("--far", type=float, default=F.FAR_K)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/tmp/see.jpg"))
    ap.add_argument("--quality", type=int, default=90)
    args = ap.parse_args()
    if args.px is None and args.world is None:
        ap.error("give --px X Y or --world X Y Z")

    with Stream() as stream:
        frame = stream.read(timeout=2.0)
    if frame is None:
        raise SystemExit("no depth frame")
    width, height = int(frame["color_width"]), int(frame["color_height"])
    rgb = np.frombuffer(bytes(frame["color"]), dtype=np.uint8).reshape(height, width, 4)[..., :3]
    depth = np.frombuffer(frame["depth"], dtype=np.float32).reshape(
        int(frame["height"]), int(frame["width"])
    )

    if args.world is not None:
        st = json.loads(state_path().read_text())
        head = Pose(p=tuple(st["head"]["p"]), q_xyzw=tuple(st["head"]["q_xyzw"]))
        uv = project_point(head, tuple(args.world), FOV)
        if uv is None:
            raise SystemExit("the object is behind the camera; nothing to fade to")
        px = (uv[0] * width, uv[1] * height)
    else:
        px = (args.px[0], args.px[1])

    found = F.nearest_surface(depth, (px[0], px[1], width, height))
    if found is None:
        raise SystemExit(f"no valid depth near ({px[0]:.0f}, {px[1]:.0f})")
    target, at = found
    out = F.fade(rgb, depth, target, args.near, args.far)
    Image.fromarray(out).save(args.out, quality=args.quality)
    kept = float((out.sum(axis=2) > sum(F.BACKDROP) + 6).mean())
    print(f"object pixel ({px[0]:.0f}, {px[1]:.0f})  nearest surface {target:.3f} m "
          f"at depth px {at}  -> {args.out}  ({100 * kept:.1f}% kept)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
