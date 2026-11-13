"""Ask the planner one frozen question: what would you do, given this still and this speech?

This is the gate the loop rewrite passes through. The live loop used to decide the skill in Python
(`task_intent` read the verb out of JobBot's sentence), so whether the model would ever *choose*
`locate` was never measured on the live path. Asking it directly, with a still that has the card in
it, is the cheapest honest test.

    spike/e3/.venv/bin/python spike/wbc_game/eval_plan.py \
        --image ~/.local/state/Clank/auto/frame-185133.png \
        --heard 'Hey, Hillman, take my card.' --expect locate --contains card

Exit codes: 0 the expectation held, 1 it did not, 2 the model server is down. The model's own reply
is always printed as JSON, so a failure is readable without re-running anything.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from clank import brain  # noqa: E402
from clank.frames import Frame  # noqa: E402


def jpeg_bytes(path: Path) -> bytes:
    """The still as JPEG. ffmpeg is already how the replay path reads clips."""
    if path.suffix.lower() in (".jpg", ".jpeg"):
        return path.read_bytes()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as out:
        tmp = Path(out.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-q:v", "2", str(tmp)],
            check=True,
        )
        return tmp.read_bytes()
    finally:
        tmp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval_plan", description=__doc__)
    ap.add_argument("--image", required=True, help="a still the model has not seen")
    ap.add_argument("--heard", default="", help="what JobBot just said (empty = silence)")
    ap.add_argument("--expect", default="", help="required skill, e.g. locate")
    ap.add_argument("--contains", default="", help="required substring in target, lowercased")
    ap.add_argument("--last", default="(none)", help="the last line the model is told")
    ap.add_argument("--seed", default=brain.DEFAULT_SEED)
    ap.add_argument("--json", action="store_true", help="print only the JSON")
    ap.add_argument("--constitution", default="",
                    help="test a candidate constitution file instead of brain.CONSTITUTION")
    args = ap.parse_args(argv)

    status = brain.server_status(brain.BrainConfig().base_url)
    if not status.up:
        print(f"server down: {status.error}", file=sys.stderr)
        return 2

    path = Path(args.image).expanduser()
    frame = Frame(index=0, t_s=0.0, jpeg=jpeg_bytes(path), source=str(path))
    live = brain.live_block("empty", "empty", 0.0, args.last)
    if not args.json:
        print(f"image: {path}\nheard: {args.heard or '(none)'}\nlast:  {args.last}", flush=True)

    if args.constitution:
        # Measure a candidate wording without editing the module: the constitution is the thing
        # being designed here, and a variant that is never measured is a preference, not a result.
        original = brain.CONSTITUTION
        brain.CONSTITUTION = Path(args.constitution).expanduser().read_text()
        try:
            result = brain.plan(brain.BrainConfig(), args.seed, frame, live)
        finally:
            brain.CONSTITUTION = original
    else:
        result = brain.plan(brain.BrainConfig(), args.seed, frame, live)
    # What the model actually said, even when the parser refuses the word. Naming is being
    # measured here as much as the decision is: `find` and `locate` are the same tool, and the
    # parser only accepts the one the constitution offers.
    raw_skill = ""
    try:
        parsed = json.loads(result.raw) if result.raw.strip().startswith("{") else {}
        raw_skill = str(parsed.get("skill") or parsed.get("action") or "")
    except json.JSONDecodeError:
        raw_skill = ""
    report = {
        "skill": result.skill,
        "raw_skill": raw_skill,
        "side": result.side,
        "target": result.target,
        "error": result.error,
        "note": result.note,
        "prompt_ms": round(result.prompt_ms, 2),
        "decode_ms": round(result.decode_ms, 2),
        "raw": result.raw,
    }
    print(json.dumps(report, indent=2), flush=True)

    if args.expect and args.expect in (result.skill, raw_skill):
        if args.contains and args.contains.lower() not in (result.target or "").lower():
            print(f"FAIL: target {result.target!r} lacks {args.contains!r}", file=sys.stderr)
            return 1
        print("PASS" + ("" if result.ok else f" (as {raw_skill!r}, not a planner skill)"),
              flush=True)
        return 0
    if not result.ok:
        print("FAIL: no usable plan", file=sys.stderr)
        return 1
    if args.expect and result.skill != args.expect:
        print(f"FAIL: skill {result.skill!r} != {args.expect!r}", file=sys.stderr)
        return 1
    if args.contains and args.contains.lower() not in (result.target or "").lower():
        print(f"FAIL: target {result.target!r} lacks {args.contains!r}", file=sys.stderr)
        return 1
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
