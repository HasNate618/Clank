"""Frame -> pixel -> hand, with no human in between.

    python3 spike/wbc_game/find_and_reach.py --noun "JobBot ID card" --grab

Captures a still, asks `locate` where the noun is, and hands the pixel to `reach_vr`. If the
object is not visible the run stops with exit 2 and no motion: the refusal is the point, and a
guess here would move the arm at nothing.
"""

from __future__ import annotations

import argparse
import io
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clank import frames  # noqa: E402
from clank.brain import BrainConfig, Frame  # noqa: E402
from clank.depth_stream import Stream  # noqa: E402

import locate  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


def capture(width: int = 960) -> tuple[bytes, tuple[int, int]]:
    """One still from the depth stream's colour plane, as JPEG, resized to `width`.

    The live path everywhere else works on a 960-wide still, and locate is measured at that
    width: the only confident error in the fixture set was on an un-resized 1920 frame.
    """
    from PIL import Image

    with Stream() as stream:
        f = stream.read(timeout=2.0)
    image = Image.frombytes(
        "RGBA", (f["color_width"], f["color_height"]), bytes(f["color"])
    ).convert("RGB")
    height = round(width * image.height / image.width)
    image = image.resize((width, height), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), (width, height)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--noun", default="JobBot ID card")
    ap.add_argument("--side", default="right", choices=("left", "right"))
    ap.add_argument("--grab", action="store_true")
    ap.add_argument("--image", type=pathlib.Path, help="locate in this still instead of capturing")
    ap.add_argument("--dry-run", action="store_true", help="locate, print, do not move")
    args = ap.parse_args(argv)

    if args.image:
        from PIL import Image

        jpeg = args.image.read_bytes()
        size = Image.open(args.image).size
    else:
        jpeg, size = capture()
        pathlib.Path("/tmp/find_and_reach.jpg").write_bytes(jpeg)

    frame = Frame(index=0, t_s=0.0, jpeg=jpeg, source=str(args.image or "live"))
    out = locate.locate(BrainConfig(), frame, args.noun, size)
    print(f"locate: quadrant={out['quadrant']} px={out['pixel_px']} "
          f"present={out.get('present')} error={out['error']!r}")
    if not out["pixel_px"] or out.get("present") is False:
        print("not visible - no motion")
        return 2
    if args.dry_run:
        return 0

    x, y = out["pixel_px"]
    cmd = [
        sys.executable.replace("python3", "python3"),
        str(HERE / "reach_vr.py"), f"{x:.1f}", f"{y:.1f}",
        "--side", args.side, "--no-rest", "--tol", "0.02", "--corrections", "4",
        "--gain", "0.6", "--guard", "0.25",
    ]
    if args.grab:
        cmd.append("--grab")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
