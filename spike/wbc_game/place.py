"""Place: hover the wrist above a destination and open the hand.

`place` is not a new motion. It is `reach` to a point that has nothing under it,
then `open`:

    locate(destination noun) -> pixel -> depth point -> +APPROACH_M in z -> reach -> open

Measured before it was written: free-air wrist goals land within 15-65 mm
(0.55 m up included), releasing is proven (the card drops cleanly), and the
approach height lives in `clank.workspace.APPROACH_M` rather than in a prompt.

    python3 spike/wbc_game/place.py --noun "the game console" [--dry-run]

Exit 2 and no motion when the destination cannot be located: a `place` with no
destination is the failure that looks like success.
"""

from __future__ import annotations

import argparse
import io
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clank import workspace  # noqa: E402
from clank.brain import BrainConfig, Frame  # noqa: E402
from clank.depth_stream import Stream  # noqa: E402
from clank.pixel3d import point_from_pixel, sample_depth  # noqa: E402
from clank.projection import Fov  # noqa: E402
from clank.body import Pose  # noqa: E402

import locate  # noqa: E402
import reach_vr as rv  # noqa: E402

FOV = Fov.from_degrees(70.0, 70.0, 57.11, 57.11)


def capture() -> tuple[bytes, tuple[int, int], dict]:
    """A 960-wide still plus the head pose that produced it, sampled together."""
    from PIL import Image

    with Stream() as stream:
        f = stream.read(timeout=2.0)
    if f is None:
        raise SystemExit("no depth frame: is the depth layer running?")
    image = Image.frombytes(
        "RGBA", (f["color_width"], f["color_height"]), bytes(f["color"])
    ).convert("RGB")
    width = 960
    height = round(width * image.height / image.width)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), (width, height), f


def hover(
    goal: tuple[float, float, float],
    side: str,
    camera: Pose,
    tolerance: float = 0.03,
    corrections: int = 4,
    gain: float = 0.6,
) -> tuple[tuple[float, float, float], float]:
    """Damped correction to a wrist target, the same loop the card reach uses.

    A single command overshoots by ~0.15 m (the emulated controller sits above the
    wrist), so one shot is never right; the loop closes that in 2-4 commands.
    """
    cmd = list(goal)
    done = 0.0
    for _ in range(corrections):
        rv.hold(tuple(cmd), 0.0, 0.45, 0.45, rv.root(), side)
        time.sleep(0.25)
        p = rv.wrist(side)
        done = math.dist(p, goal)
        if done <= tolerance:
            break
        cmd = [c + gain * (g - a) for c, g, a in zip(cmd, goal, p)]
    return rv.wrist(side), done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--noun", required=True, help="the destination, named")
    ap.add_argument("--side", default="right", choices=("left", "right"))
    ap.add_argument("--height", type=float, default=workspace.APPROACH_M)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    jpeg, size, f = capture()
    pathlib.Path("/tmp/place_frame.jpg").write_bytes(jpeg)
    st = rv.state()
    camera = Pose(p=tuple(st["head"]["p"]), q_xyzw=tuple(st["head"]["q_xyzw"]))
    frame = Frame(index=0, t_s=0.0, jpeg=jpeg, source="live")
    out = locate.locate(BrainConfig(), frame, args.noun, size)
    print(f"locate({args.noun!r}) -> quadrant={out['quadrant']} px={out['pixel_px']} "
          f"present={out.get('present')} error={out['error']!r}")
    if not out["pixel_px"] or out.get("present") is False:
        print("destination not visible - no motion")
        return 2

    u, v = out["pixel_px"][0] / size[0], out["pixel_px"][1] / size[1]
    dest = point_from_pixel(f, u, v, camera, FOV)
    if dest is None:
        print("no surface at the located pixel - no motion")
        return 2
    goal = workspace.approach_point(dest, args.height)
    print(f"destination {rv.fmt(dest)}  hover {rv.fmt(goal)}  "
          f"({args.height:.2f} m above, in free air)")
    if args.dry_run:
        return 0

    final, err = hover(goal, args.side, camera)
    print(f"hovered at {rv.fmt(final)}  error {err * 1000:.1f} mm")
    rv.send(verb=f"open_{args.side}")
    for _ in range(3):
        time.sleep(0.3)
        rv.send(verb=f"open_{args.side}")
    print(f"opened; closure {rv.state()['closure']}")
    return 0 if err <= 0.07 else 1


if __name__ == "__main__":
    raise SystemExit(main())
