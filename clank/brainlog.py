"""What the brain is thinking, in one line at a time.

The claim the artifact has to make is "it hears JobBot and does the task", and that is a
sequence of sentences: *heard ...*, *I will pick up the card*, *reaching for the card*,
*picked it up*. Those lines are the demo, so they are written to disk as they happen
rather than reconstructed afterwards.

Two sinks, one row:

- `~/.local/state/Clank/brain/messages.jsonl`, one JSON object per line, append-only and
  readable with the stdlib. A dashboard can tail it; the dashboard here is being rebuilt,
  so nothing depends on a UI existing.
- stderr, one rendered line per row, so a run in a terminal shows the same thing live.

Rows are cheap and self-describing: `{"t": ..., "iso": ..., "kind": "think", "text": ...}`
plus whatever numbers the caller attaches. Nothing reads this file back to make a decision
- it is a diary, not state.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from typing import Any, TextIO

#: The kinds of row. `heard` is input, `think` is the decision, `act` is a skill starting,
#: `result` is how it went, `error` is the run failing.
KINDS = ("heard", "think", "act", "result", "error")

DEFAULT_PATH = pathlib.Path.home() / ".local" / "state" / "Clank" / "brain" / "messages.jsonl"


def default_path() -> pathlib.Path:
    """`$CLANK_BRAIN_LOG` if set, so tests and probes do not touch the live diary."""
    override = os.environ.get("CLANK_BRAIN_LOG", "")
    return pathlib.Path(override) if override else DEFAULT_PATH


def log(
    kind: str,
    text: str,
    *,
    path: pathlib.Path | None = None,
    echo: bool = True,
    stream: TextIO | None = None,
    **data: Any,
) -> dict[str, Any]:
    """Append one message. Never raises: a diary that breaks the run is a liability."""
    row: dict[str, Any] = {
        "t": round(time.time(), 3),
        "iso": time.strftime("%H:%M:%S"),
        "kind": kind if kind in KINDS else "think",
        "text": " ".join(str(text).split()),
    }
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, float):
            row[key] = round(value, 2)
        elif isinstance(value, (int, bool, str)):
            row[key] = value
        else:
            row[key] = str(value)

    target = path if path is not None else default_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        if echo:
            print(f"[brain] (cannot write {target}: {exc})", file=sys.stderr)
    if echo:
        print(render(row), file=stream or sys.stderr, flush=True)
    return row


def render(row: dict[str, Any]) -> str:
    """One row as a sentence a human can read across the room."""
    kind = str(row.get("kind", "think"))
    text = str(row.get("text", ""))
    extras = []
    for key in ("skill", "side", "object", "destination", "ms", "code"):
        if key in row:
            extras.append(f"{key}={row[key]}")
    suffix = f" ({', '.join(extras)})" if extras else ""
    return f"[brain] {row.get('iso', '')} {kind}: {text}{suffix}"


def read(path: pathlib.Path | None = None) -> list[dict[str, Any]]:
    """Every row written so far. Tolerates a torn last line."""
    target = path if path is not None else default_path()
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def tail(count: int = 20, path: pathlib.Path | None = None) -> list[dict[str, Any]]:
    rows = read(path)
    return rows[-count:] if count > 0 else rows


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Read the brain's message diary.")
    ap.add_argument("--tail", type=int, default=20)
    ap.add_argument("--path", default="")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    target = pathlib.Path(args.path) if args.path else None
    rows = tail(args.tail, target)
    print(f"brain: {len(rows)} message(s) from {target or default_path()}", file=sys.stderr)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False) if args.json else render(row))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
