"""One turn of work: hear, look, decide, then run the steps in order.

The model is asked what it wants (goal + destination). It does not order the steps -
measured, it answers "insert the cartridge into INSERT CARTRIDGE" with empty hands, and
no phrasing fixed that - so `brain.sequence()` turns the goal into the ordered calls and
`--holding` says whether the hand is already full, because only the caller can know that.

With `--listen` the situation is not typed in: the game's own audio output is recorded
(clank.ears picks the JobSimulator node, so nothing else on the desktop is captured),
transcribed by the model, and the transcript becomes the situation. Every sentence the
brain produces goes to clank.brainlog, which writes a JSON line per message and echoes it
to stderr.

    spike/e3/.venv/bin/python spike/wbc_game/do_task.py --listen --dry-run
    spike/e3/.venv/bin/python spike/wbc_game/do_task.py --situation "Your hands are empty."
"""

from __future__ import annotations

import argparse
import io
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from clank import brain, brainlog, ears  # noqa: E402
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
    if f is None:
        raise SystemExit("no frame from the depth stream")
    image = Image.frombytes(
        "RGBA", (f["color_width"], f["color_height"]), bytes(f["color"])
    ).convert("RGB")
    height = round(width * image.height / image.width)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
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


def hear(pattern: str, max_s: float) -> str:
    """Record the game talking and return its words, or "" if it said nothing."""
    hearing = ears.listen(pattern=pattern, max_s=max_s)
    brainlog.log(
        "heard",
        f"{hearing.seconds:.1f}s captured from {hearing.node_name or 'the game'}",
        speech_s=hearing.speech_s, peak=hearing.peak, rms=hearing.rms,
    )
    if not hearing.ok:
        brainlog.log("error", hearing.error or "nothing was said")
        return ""
    heard = brain.transcribe(BrainConfig(), hearing.wav)
    if heard.error:
        brainlog.log("error", f"transcribe failed: {heard.error}")
        return ""
    if heard.empty:
        brainlog.log("heard", "(silence)")
        return ""
    brainlog.log("heard", heard.text, ms=heard.decode_ms, cut=heard.ellipsis)
    return heard.text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--situation", default="",
                    help="what is going on, when it is not coming from the game's audio")
    ap.add_argument("--listen", action="store_true",
                    help="hear the game instead of being told the situation")
    ap.add_argument("--pattern", default=ears.DEFAULT_PATTERN,
                    help="substring of the game's PipeWire application name")
    ap.add_argument("--max-s", type=float, default=25.0, dest="max_s")
    ap.add_argument("--side", default="right", choices=("left", "right"))
    ap.add_argument("--holding", action="store_true", help="the hand already holds the object")
    ap.add_argument("--object", default="",
                    help="the noun the task names, used when the hand is empty and a grab "
                         "is needed; the model cannot name an object it is not holding")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not args.listen and not args.situation:
        ap.error("--situation or --listen is required")

    situation = args.situation
    if args.listen:
        spoken = hear(args.pattern, args.max_s)
        if not spoken:
            return 3
        situation = f'JobBot said: "{spoken}"'

    jpeg, size = capture()
    step = brain.next_step(BrainConfig(max_tokens=160),
                           Frame(0, 0.0, jpeg, "live"), situation,
                           default_side=args.side)
    if step.error:
        brainlog.log("error", f"the model did not answer: {step.error}")
        return 4
    brainlog.log("think", f"I will {step.goal}", object=step.object,
                 destination=step.destination, side=step.side)
    steps = brain.sequence(step, holding=args.holding, object_noun=args.object)
    brainlog.log("think", "next: " + " -> ".join(f"{s}({n})" for s, n in steps)
                 or "nothing to do", hand="holding" if args.holding else "empty")
    if args.dry_run or not steps:
        return 0 if steps else 1
    for skill, noun in steps:
        brainlog.log("act", f"{skill} {noun!r}", skill=skill)
        rc = grab(noun, args.side) if skill == "grab" else place(noun, args.side)
        brainlog.log("result", f"{skill} {'ok' if rc == 0 else 'failed'}", skill=skill, code=rc)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
