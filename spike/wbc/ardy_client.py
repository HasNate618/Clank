#!/usr/bin/env python3
"""Ask the resident ARDY generator for a clip, or bring it up, or say it is not there.

Measured 2026-11-13, on a warm cache, per act: a new interpreter spends **0.90 s** importing
torch and reading the body, **1.99 s** putting the 740 MB checkpoint into VRAM, and about
**1.0 s** on the first CUDA call, before **0.22 s** of denoising. That is 4.3 s before the hand
moves, and it was paid again on every act because each act is a fresh `reach_ardy.py` process.

The body's share of the same act is 0.06-0.2 s: 2.14 s of wall time for a 2.08 s clip. So the
delay was never the body; it was the generator loading itself.

This client is deliberately stdlib. The heavy side is the daemon, which lives in the ARDY venv;
anything that can write a JSON line can ask it for a clip. `Unavailable` is not an error to
report to the operator - it is the signal to generate in-process instead, because on this
machine the GPU is often off and a reach does not care how the hand arrived.
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: Beside the body's other mailbox files: one per session directory, never a TCP port.
SOCKET_NAME = "ardy.sock"
#: Where a daemon this client spawned writes its own stdout, for the day it refuses to start.
LOG_NAME = "ardy_daemon.log"

CONNECT_TIMEOUT_S = 0.5
#: One generation is 0.3 s warm and a first call after a cold start is a few seconds; this is
#: a liveness bound, not a performance target.
REQUEST_TIMEOUT_S = 60.0


class Unavailable(RuntimeError):
    """No resident generator answered. The caller should do the work itself."""


def ask(sock: Path | str, payload: dict[str, Any],
        timeout: float = REQUEST_TIMEOUT_S) -> dict[str, Any]:
    """One request, one reply. Raises `Unavailable` on anything that is not an answer."""
    path = Path(sock)
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(timeout)
    try:
        conn.connect(str(path))
        conn.sendall(json.dumps(payload).encode("utf-8") + b"\n")
        chunks: list[bytes] = []
        while True:
            block = conn.recv(65536)
            if not block:
                break
            chunks.append(block)
            if block.endswith(b"\n"):
                break
        raw = b"".join(chunks)
    except (OSError, TimeoutError) as exc:
        raise Unavailable(f"{path.name}: {exc}") from exc
    finally:
        conn.close()
    if not raw.strip():
        raise Unavailable(f"{path.name}: closed without a reply")
    try:
        reply = json.loads(raw)
    except ValueError as exc:
        raise Unavailable(f"{path.name}: reply is not JSON ({exc})") from exc
    if not isinstance(reply, dict):
        raise Unavailable(f"{path.name}: reply is {type(reply).__name__}, not an object")
    return reply


def alive(sock: Path | str, timeout: float = CONNECT_TIMEOUT_S) -> bool:
    """Whether something is serving this socket right now."""
    try:
        return bool(ask(sock, {"op": "ping"}, timeout=timeout).get("ok"))
    except Unavailable:
        return False


def ensure(
    sock: Path | str,
    spawn: Sequence[str],
    wait_s: float = 30.0,
    log: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Make a daemon be answering this socket, spawning `spawn` if nothing is.

    Returns False rather than raising: a generator that will not come up is a fallback, not a
    failure. A spawn that exits at once (no CUDA, a missing checkpoint, another daemon already
    holding the socket) returns in milliseconds rather than waiting out `wait_s`, which matters
    because the GPU being off is this machine's normal state and not a 30 s shutdown.
    """
    path = Path(sock)
    if alive(path):
        return True
    log_path = Path(log) if log is not None else path.parent / LOG_NAME
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = log_path.open("ab")
    except OSError:
        return False
    try:
        child = subprocess.Popen(  # noqa: S603 - the argv comes from the caller, not a shell
            list(spawn),
            env=None if env is None else dict(env),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError:
        return False
    finally:
        handle.close()
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if alive(path):
            return True
        if child.poll() is not None:
            return False
        time.sleep(0.1)
    return False


def generate(
    sock: Path | str,
    poses: Sequence[Sequence[float]],
    target: Sequence[float],
    root_forward_m: float | None,
    steps: int,
    timeout: float = REQUEST_TIMEOUT_S,
) -> tuple[list[list[float]], dict[str, Any]]:
    """One generated window from the resident generator.

    The request is stateless by design: it carries the ring's recent poses and the target, so a
    daemon that was restarted, or one serving a second session, cannot answer from a pose the
    body no longer holds.
    """
    reply = ask(
        sock,
        {
            "op": "generate",
            "poses": [list(map(float, pose)) for pose in poses],
            "target": [float(v) for v in target],
            "root_forward_m": None if root_forward_m is None else float(root_forward_m),
            "steps": int(steps),
        },
        timeout=timeout,
    )
    if not reply.get("ok"):
        raise Unavailable(str(reply.get("why") or "the generator refused"))
    frames = reply.get("frames")
    metrics = reply.get("metrics")
    if not isinstance(frames, list) or not frames:
        raise Unavailable("the generator answered with no frames")
    if not isinstance(metrics, dict):
        raise Unavailable("the generator answered with no metrics")
    return [list(map(float, row)) for row in frames], metrics
