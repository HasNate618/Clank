#!/usr/bin/env python3
"""A resident ARDY generator: load torch and the checkpoint once, answer clips until idle.

Why this exists, measured 2026-11-13 on the live stack: one reach paid **0.90 s** to import
torch and read the body, **1.99 s** to put the 740 MB checkpoint into VRAM and about **1.0 s**
on the first CUDA call, to run **0.22 s** of denoising. The hand therefore started moving 4.3 s
after the act was decided, on every act, because each act spawns a fresh process. The body was
never the slow part - it picks a deposited clip up in 0.06-0.2 s.

Requests are stateless: the caller sends the body's recent pose36 ring and the target, so this
process holds no session state and a restarted daemon cannot continue from a pose the body no
longer has. One connection is served at a time, which is enough - a body conducts one reach.

Protocol, newline-delimited JSON over a Unix socket (`ardy_client.py` is the other half):

    {"op": "ping"}                                  -> {"ok": true, "pid": 123}
    {"op": "generate", "poses": [[36 floats]...],   -> {"ok": true,
     "target": [x, y, z],                              "frames": [[36 floats]...],
     "root_forward_m": 0.05 | null,                    "metrics": {...}}
     "steps": 10}
    {"op": "shutdown"}                              -> {"ok": true, "stopping": true}

A bad request is answered, not fatal: the daemon must outlive a caller that sent nonsense,
because the alternative is that every act after it pays the cold start again.

Usage:
    ardy_daemon.py [--socket PATH] [--idle SECONDS] [--steps N] [--lazy] [--once]

The model is loaded at startup unless `--lazy` is given; `--once` serves a single request and
exits, which is what the tests use. Under NixOS the process needs
`LD_LIBRARY_PATH=/run/opengl-driver/lib`, or torch reports `cuda False` on a working driver.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_WBC = Path(__file__).resolve().parent
if str(_WBC) not in sys.path:
    sys.path.insert(0, str(_WBC))
# The generators own the hand-delta definition the adapter measures against; `reach_ardy` is
# where it lives, so the daemon imports it rather than restating the lever and drifting.
_WBC_GAME = Path(__file__).resolve().parents[1] / "wbc_game"
if str(_WBC_GAME) not in sys.path:
    sys.path.insert(0, str(_WBC_GAME))

#: The socket the body's mailbox lives beside, when no `--socket` is given.
DEFAULT_SOCKET = Path.home() / ".local/state/Clank/sessions/body/ardy.sock"
#: A run listens for 900 s and a body holds a pose between jobs, so a generator that released
#: its VRAM after every reach would re-pay the cold start on the next one. Fifteen minutes of
#: silence means the run is over.
IDLE_S = 900.0
STEPS = 10
#: Long enough for one generation including a first CUDA call, short enough that a half-open
#: connection cannot wedge the daemon for a whole run.
READ_TIMEOUT_S = 120.0

#: `(poses, target, root_forward_m, steps) -> (frames, metrics)`.
Generate = Callable[[Sequence[Sequence[float]], Sequence[float], float | None, int],
                    tuple[list[list[float]], dict[str, Any]]]


def default_generator(steps_default: int = STEPS) -> Generate:
    """The real generator, importing torch only when this is called.

    Importing `ardy_gen` is cheap (0.02 s); it is torch and the checkpoint that cost, and they
    are loaded inside `ardy_gen.generate`. `reach_ardy.hand_delta_for` is reused rather than
    re-derived so the delta the daemon pins cannot drift from the one the adapter measures.
    """
    import ardy_gen  # pyright: ignore[reportMissingImports]
    import reach_ardy  # pyright: ignore[reportMissingImports]

    def generate(
        poses: Sequence[Sequence[float]],
        target: Sequence[float],
        root_forward_m: float | None,
        steps: int,
    ) -> tuple[list[list[float]], dict[str, Any]]:
        delta = reach_ardy.hand_delta_for(poses[-1], tuple(float(v) for v in target))
        frames, metrics = ardy_gen.generate(
            [list(pose) for pose in poses],
            hand_delta=delta,
            root_forward_m=root_forward_m,
            steps=steps or steps_default,
        )
        return [list(row) for row in frames], dict(metrics)

    return generate


def _as_floats(value: Any, length: int | None, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of numbers")
    numbers = [float(v) for v in value]
    if not numbers:
        raise ValueError(f"{name} is empty")
    if length is not None and len(numbers) != length:
        raise ValueError(f"{name} must be {length} numbers, got {len(numbers)}")
    return numbers


def parse_request(request: Any) -> dict[str, Any]:
    """A validated generate request. Raises `ValueError` for anything a caller got wrong."""
    if not isinstance(request, dict):
        raise ValueError("a request must be a JSON object")
    poses = request.get("poses")
    if not isinstance(poses, (list, tuple)) or not poses:
        raise ValueError("poses must be a non-empty list of pose36 rows")
    rows = [_as_floats(row, None, "each pose") for row in poses]
    targets = _as_floats(request.get("target"), 3, "target")
    root = request.get("root_forward_m")
    if root is not None:
        root = float(root)
    steps = int(request.get("steps") or STEPS)
    return {"poses": rows, "target": targets, "root_forward_m": root, "steps": steps}


def handle(request: Any, generate: Generate | None) -> dict[str, Any]:
    """One request to one reply. Never raises: a caller's mistake is not a reason to die."""
    if not isinstance(request, dict):
        return {"ok": False, "why": "a request must be a JSON object"}
    op = request.get("op")
    if op == "ping":
        return {"ok": True, "pid": os.getpid(), "socket": True}
    if op == "shutdown":
        return {"ok": True, "stopping": True}
    if op != "generate":
        return {"ok": False, "why": f"unknown op {op!r}"}
    if generate is None:
        return {"ok": False, "why": "this daemon has no generator"}
    try:
        parsed = parse_request(request)
    except (TypeError, ValueError) as exc:
        return {"ok": False, "why": f"bad request: {exc}"}
    try:
        frames, metrics = generate(
            parsed["poses"], parsed["target"], parsed["root_forward_m"], parsed["steps"]
        )
    except Exception as exc:  # noqa: BLE001 - one bad generation must not stop the daemon
        return {"ok": False, "why": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "frames": frames, "metrics": metrics}


def _read_line(conn: socket.socket) -> bytes:
    chunks: list[bytes] = []
    while True:
        block = conn.recv(65536)
        if not block:
            break
        chunks.append(block)
        if block.endswith(b"\n"):
            break
    return b"".join(chunks)


def serve(
    sock: Path | str,
    generate: Generate | None,
    idle_s: float = IDLE_S,
    once: bool = False,
    stop: Any | None = None,
) -> int:
    """Serve until idle, `shutdown`, `once`, or `stop` is set. Returns the request count."""
    path = Path(sock)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A socket file left by a daemon that was killed is not a listener: unlink it, or bind fails
    # with EADDRINUSE and every later reach pays the cold start.
    with contextlib.suppress(FileNotFoundError):
        path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(4)
    server.settimeout(1.0)
    served = 0
    last = time.monotonic()
    try:
        while True:
            if stop is not None and stop.is_set():
                break
            try:
                conn, _ = server.accept()
            except TimeoutError:
                if idle_s and time.monotonic() - last > idle_s:
                    break
                continue
            last = time.monotonic()
            with conn:
                conn.settimeout(READ_TIMEOUT_S)
                try:
                    raw = _read_line(conn)
                except (OSError, TimeoutError):
                    continue
                try:
                    request = json.loads(raw) if raw.strip() else None
                except ValueError:
                    request = None
                reply = handle(request, generate)
                served += 1
                # The caller may have given up between asking and being answered. A queued
                # answer is not a failure; the pose it asked about is gone either way.
                with contextlib.suppress(OSError):
                    conn.sendall(json.dumps(reply).encode("utf-8") + b"\n")
                if reply.get("stopping") or once:
                    break
    finally:
        server.close()
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
    return served


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ardy_daemon", description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)
    parser.add_argument("--idle", type=float, default=IDLE_S,
                        help="exit after this many seconds with no request; 0 never exits")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--lazy", action="store_true",
                        help="do not load the checkpoint until the first request")
    parser.add_argument("--once", action="store_true", help="serve one request, then exit")
    args = parser.parse_args(argv)

    generate: Generate | None = None
    if not args.lazy:
        generate = default_generator(args.steps)
        # Load now, so the first reach is not the one that pays for it. A generator that cannot
        # load is answered as unsupported; the adapter then does the work in-process and says
        # the same thing, which is the behaviour that existed before this daemon.
        try:
            import torch  # pyright: ignore[reportMissingImports]

            if torch.cuda.is_available():
                import ardy_gen  # pyright: ignore[reportMissingImports]

                ardy_gen.load_ardy()
        except Exception as exc:  # noqa: BLE001 - unsupported is not a crash
            print(f"cannot warm the generator: {type(exc).__name__}: {exc}", flush=True, file=sys.stderr)
            generate = None

    served = serve(args.socket, generate, idle_s=args.idle, once=args.once)
    print(f"served {served} request(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
