"""The command bus: latest-wins in, achieved state out.

Design §6 semantics, which this module exists to enforce:

- **TTL gates acceptance, not execution.** A command that arrives expired is
  dropped and the body holds.
- **Continuous commands (`duration_ms == 0`) are latest-wins sample-and-hold.**
  A newer one always replaces an older one; a stale `seq` never rewinds.
- **A non-preemptible ballistic runs to its duration.** Newer commands queue
  behind it rather than blending into it mid-skill.

The brain never sees this file's timing rules directly; it just writes commands
and reads `BodyState`.

Transport is newline-delimited JSON over a Unix socket, one command per line.
One writer, one reader, one session — matching the Monado pipe's constraint.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .body import BodyCommand, BodyState, hold
from .validate import ValidationResult, validate_command


class CommandBus:
    """Thread-safe latest-wins queue with ballistic queueing and an invalid counter."""

    def __init__(self, clock: Callable[[], int] = time.monotonic_ns) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._latest: BodyCommand | None = None
        self._state: BodyState | None = None
        self._invalid = 0
        self._seq_seen = -1
        self._ballistic: BodyCommand | None = None
        self._ballistic_end_ns = 0

    # ---- writes ---------------------------------------------------------

    def push(self, cmd: BodyCommand) -> bool:
        """Accept a command. Returns False when it was dropped as stale."""
        with self._lock:
            if cmd.seq < self._seq_seen:
                return False
            self._seq_seen = cmd.seq

            now = self._clock()
            if self._ballistic is not None and now < self._ballistic_end_ns and not cmd.preempt:
                # Queue behind the running skill; take() keeps returning it.
                self._latest = cmd
                return True

            if self._ballistic is not None and (cmd.preempt or now >= self._ballistic_end_ns):
                self._ballistic = None

            if cmd.is_blocking():
                self._ballistic = cmd
                self._ballistic_end_ns = cmd.t_mono_ns + int(cmd.duration_ms) * 1_000_000

            self._latest = cmd
            return True

    def push_json(self, payload: str) -> ValidationResult:
        """Parse, validate and push one JSON line. Never raises on bad input."""
        try:
            raw = json.loads(payload)
            cmd = BodyCommand.from_dict(raw)
        except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
            with self._lock:
                self._invalid += 1
            return ValidationResult(None, f"unparseable command: {exc}")

        result = validate_command(cmd, self._clock())
        if not result.ok:
            with self._lock:
                self._invalid += 1
            return result

        self.push(cmd)
        return result

    def publish_state(self, state: BodyState) -> None:
        with self._lock:
            self._state = state

    # ---- reads ----------------------------------------------------------

    def take(self, now_ns: int | None = None) -> BodyCommand:
        """The command to execute now. Always returns something: hold if unsure."""
        now = self._clock() if now_ns is None else now_ns
        with self._lock:
            if self._ballistic is not None and now < self._ballistic_end_ns:
                return self._ballistic
            if self._ballistic is not None:
                self._ballistic = None

            cmd = self._latest
            if cmd is None:
                return hold(now_ns=now, state=self._state)
            if cmd.expires_at_ns() < now:
                # Freeze-hold at what the body actually achieved, not at the
                # standing clip. This is the path a finished reach takes, and an
                # empty mask here would snap the wrist back and drop what it held.
                return hold(seq=cmd.seq, now_ns=now, state=self._state)
            return cmd

    def latest_state(self) -> BodyState | None:
        with self._lock:
            return self._state

    @property
    def invalid_count(self) -> int:
        with self._lock:
            return self._invalid

    def idle(self) -> bool:
        with self._lock:
            return self._latest is None and self._ballistic is None


class UnixCommandServer:
    """Serve newline-delimited JSON `BodyCommand`s from one socket client."""

    def __init__(self, bus: CommandBus, path: str | Path) -> None:
        self.bus = bus
        self.path = Path(path)
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.errors: list[str] = []

    def start(self) -> None:
        if self._sock is not None:
            return
        if self.path.exists():
            self.path.unlink()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(self.path))
        sock.listen(1)
        sock.settimeout(0.2)
        self._sock = sock
        self._thread = threading.Thread(target=self._serve, name="command-bus", daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        sock = self._sock
        if sock is None:
            return
        while not self._stop.is_set():
            try:
                conn, _ = sock.accept()
            except (TimeoutError, OSError):
                continue
            with conn:
                stream = conn.makefile("r", encoding="utf-8")
                for line in stream:
                    line = line.strip()
                    if not line:
                        continue
                    result = self.bus.push_json(line)
                    if not result.ok and result.reason:
                        self.errors.append(result.reason)

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()


def send_command(path: str | Path, cmd: BodyCommand) -> None:
    """One-shot writer, used by the scripted commander and by tests."""
    payload = (json.dumps(cmd.to_dict()) + "\n").encode("utf-8")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(str(path))
        sock.sendall(payload)


def bus_socket_path(runtime_dir: str | os.PathLike[str]) -> Path:
    """Where a session's bus lives: inside the session runtime directory."""
    return Path(runtime_dir) / "body.sock"
