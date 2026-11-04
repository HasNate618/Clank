"""One turn of work: look, decide, then run the steps in order.

The model is asked what it wants (goal + destination). It does not order the steps -
measured, it answers "insert the cartridge into INSERT CARTRIDGE" with empty hands, and
no phrasing fixed that - so `brain.sequence()` turns the goal into the ordered calls and
`--holding` says whether the hand is already full, because only the caller can know that.

    python3 spike/wbc_game/do_task.py --situation "The game wants a cartridge. The tray is \
        on the counter. Your hands are empty." [--holding] [--dry-run]
"""

from __future__ import annotations

import argparse
import io
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clank import brain  # noqa: E402
from clank.brain import BrainConfig, Frame  # noqa: E402
from clank.depth_stream import Stream  # noqa: E402

import locate  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
#: The interpreter that has PIL: locate and place both crop and encode.
VENV = str(HERE.parent / "e3" / ".venv" / "bin" / "python")


def capture(width: int = 960) -> tuple[bytes, tuple[int, int]]:
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
    pathlib.Path("/tmp/do_task_frame.jpg").write_bytes(buf.getvalue())
    return buf.getvalue(), (width, height)


def grab(noun: str, side: str) -> int:
    """locate the noun, then reach and close in one process, with no gap."""
    jpeg, size = capture()
    out = locate.locate(BrainConfig(), Frame(0, 0.0, jpeg, "live"), noun, size)
    print(f"   locate({noun!r}) -> {out['pixel_px']} quadrant={out['quadrant']} "
          f"present={out.get('present')} error={out['error']!r}")
    if not out["pixel_px"] or out.get("present") is False:
        print("   not visible - stopping")
        return 2
    x, y = out["pixel_px"]
    return subprocess.call([
        VENV, str(HERE / "reach_vr.py"), f"{x:.1f}", f"{y:.1f}", "--side", side,
        "--no-rest", "--tol", "0.02", "--corrections", "4", "--gain", "0.6",
        "--guard", "0.25", "--grab",
    ])


def place(noun: str, side: str) -> int:
    return subprocess.call([VENV, str(HERE / "place.py"), "--noun", noun, "--side", side])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--situation", required=True)
    ap.add_argument("--side", default="right", choices=("left", "right"))
    ap.add_argument("--holding", action="store_true", help="the hand already holds the object")
    ap.add_argument("--object", default="",
                    help="the noun the task names, used when the hand is empty and a grab "
                         "is needed; the model cannot name an object it is not holding")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    jpeg, size = capture()
    step = brain.next_step(BrainConfig(max_tokens=160),
                           Frame(0, 0.0, jpeg, "live"), args.situation,
                           default_side=args.side)
    print(f"decided: goal={step.goal!r} skill={step.skill} object={step.object!r} "
          f"destination={step.destination!r} error={step.error!r}")
    steps = brain.sequence(step, holding=args.holding, object_noun=args.object)
    print(f"sequence ({'holding' if args.holding else 'empty hand'}): "
          + " -> ".join(f"{s}({n})" for s, n in steps))
    if args.dry_run or not steps:
        return 0 if steps else 1
    for skill, noun in steps:
        print(f"-- {skill} {noun!r}")
        rc = grab(noun, args.side) if skill == "grab" else place(noun, args.side)
        if rc != 0:
            print(f"   {skill} returned {rc} - stopping")
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
