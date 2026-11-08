"""Listen to the game and print what it says, one utterance at a time.

This is the ear on its own, with nothing acting on what it hears: useful while the brain is
being built, and the way to check the audio path without a body attached.

It records only the game's own PipeWire output (clank.ears resolves the JobSimulator node),
so a video on the desktop or the operator talking cannot reach it.

    spike/e3/.venv/bin/python spike/wbc_game/listen.py --for 120
    spike/e3/.venv/bin/python spike/wbc_game/listen.py --once
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from clank import brain, brainlog, ears  # noqa: E402

#: Stop listening this long before the deadline, so the last utterance is transcribed
#: inside the caller's time budget rather than cut off mid-call.
MARGIN_S = 3.0


def one_turn(config: brain.BrainConfig, pattern: str, max_s: float) -> str:
    """Record one utterance and return its words, or "" if the game said nothing."""
    try:
        hearing = ears.listen(pattern=pattern, max_s=max_s)
    except ears.EarsError as exc:
        # The game is not playing yet, or PipeWire is not up. Keep waiting: the caller's
        # deadline bounds this, and a session start takes a minute.
        brainlog.log("heard", f"waiting for the game ({exc})")
        return ""
    if not hearing.ok:
        brainlog.log("heard", f"nothing ({hearing.error})")
        return ""
    heard = brain.transcribe(config, hearing.wav)
    if heard.error:
        brainlog.log("error", f"transcribe failed: {heard.error}")
        return ""
    if heard.empty:
        brainlog.log("heard", f"({hearing.seconds:.1f}s of game audio, no words)")
        return ""
    brainlog.log("heard", heard.text, ms=heard.decode_ms, speech_s=hearing.speech_s,
                 peak=hearing.peak, cut=heard.ellipsis)
    return heard.text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--pattern", default=ears.DEFAULT_PATTERN,
                    help="substring of the game's PipeWire application name")
    ap.add_argument("--for", type=float, default=60.0, dest="duration",
                    help="how long to keep listening, seconds")
    ap.add_argument("--max-s", type=float, default=25.0, dest="max_s",
                    help="longest single utterance")
    ap.add_argument("--once", action="store_true", help="one utterance and stop")
    args = ap.parse_args(argv)

    config = brain.BrainConfig()
    deadline = time.monotonic() + args.duration
    heard = 0
    while True:
        budget = max(1.0, deadline - time.monotonic())
        text = one_turn(config, args.pattern, min(args.max_s, budget))
        if text:
            heard += 1
            if args.once:
                break
        if time.monotonic() + MARGIN_S >= deadline:
            break
    brainlog.log("result", f"{heard} utterance(s) in {args.duration:.0f}s")
    return 0 if heard else 1


if __name__ == "__main__":
    raise SystemExit(main())
