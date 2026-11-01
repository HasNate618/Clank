"""The scripted commander: the first `BodyCommand` writer.

The brain is a drop-in replacement for this on the same socket. Nothing below
it knows or cares which one is writing — that is the whole point of the
contract, and why the pipe must work scripted before a model is plugged in.

Script file is newline-delimited JSON. Each line is either a bare
`BodyCommand` object or `{"command": {...}, "sleep_ms": 1500}` when the step
needs its own dwell time. `seq` and `t_mono_ns` are stamped here, so a script
never has to reason about clocks.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from .body import SCHEMA, BodyCommand, rest
from .bus import send_command

DEFAULT_SLEEP_MS = 1000.0


def load_steps(path: str | Path) -> list[tuple[BodyCommand, float]]:
    """Parse a script file into (command, dwell seconds) pairs."""
    steps: list[tuple[BodyCommand, float]] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno}: {exc}") from exc
        if not isinstance(raw, dict):
            raise SystemExit(f"{path}:{lineno}: expected an object")

        if "command" in raw:
            sleep_ms = float(raw.get("sleep_ms", DEFAULT_SLEEP_MS))
            cmd_raw = raw["command"]
        else:
            sleep_ms = float(raw.get("sleep_ms", DEFAULT_SLEEP_MS))
            cmd_raw = {k: v for k, v in raw.items() if k != "sleep_ms"}

        cmd_raw.setdefault("schema", SCHEMA)
        cmd_raw.setdefault("source", "script")
        cmd = BodyCommand.from_dict(cmd_raw)
        steps.append((cmd, sleep_ms / 1000.0))
    return steps


def stamp(steps: list[tuple[BodyCommand, float]]) -> Iterator[tuple[BodyCommand, float]]:
    """Give each step a monotonic `seq` and an arrival time."""
    from dataclasses import replace

    for seq, (cmd, dwell) in enumerate(steps):
        yield replace(cmd, seq=seq, t_mono_ns=time.monotonic_ns()), dwell


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clank.scripted")
    parser.add_argument("--socket", required=True, help="command bus socket path")
    parser.add_argument("--file", required=True, help="newline-delimited JSON script")
    parser.add_argument("--loop", action="store_true", help="repeat the script forever")
    parser.add_argument("--speed", type=float, default=1.0, help="dwell time multiplier")
    parser.add_argument("--dry-run", action="store_true", help="print stamped commands only")
    args = parser.parse_args(argv)

    steps = load_steps(args.file)
    if not steps:
        print("scripted: no steps found", file=sys.stderr)
        return 2

    try:
        while True:
            for cmd, dwell in stamp(steps):
                if args.dry_run:
                    print(json.dumps(cmd.to_dict()))
                else:
                    send_command(args.socket, cmd)
                time.sleep(max(0.0, dwell / args.speed))
            if not args.loop:
                break
    except FileNotFoundError:
        print(f"scripted: no bus at {args.socket}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        if not args.dry_run:
            with contextlib.suppress(OSError):
                send_command(args.socket, rest(now_ns=time.monotonic_ns()))
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
