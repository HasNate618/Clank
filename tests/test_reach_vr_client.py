"""`reach_vr.send` must survive the body view recreating its control socket.

The sidecar deletes and rebinds `control.sock` when it restarts, and `ready()` is
the first thing a brain loop does: a connect that lands in the gap killed a whole
live run twice, 0.2 s into the brain.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any

WBC = Path(__file__).resolve().parent.parent / "spike" / "wbc_game"


def load(name: str) -> ModuleType:
    """Import a spike module by path; `spike/` is not on the suite's sys.path."""
    spec = importlib.util.spec_from_file_location(name, WBC / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reach_vr: Any = load("reach_vr")


class ControlServer:
    """A control socket that binds late, like the sidecar does."""

    def __init__(self, path: Path, delay: float = 0.0) -> None:
        self.path = Path(path)
        self.delay = delay
        self.verbs: list[dict] = []
        self._stop = threading.Event()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        if self.delay and self._stop.wait(self.delay):
            return
        self._sock.bind(str(self.path))
        self._sock.listen(4)
        self._sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with conn:
                data = b""
                while b"\n" not in data:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                try:
                    self.verbs.append(json.loads(data.decode() or "{}"))
                except ValueError:
                    self.verbs.append({})
                conn.sendall(b'{"ok": true}\n')

    def close(self) -> None:
        self._stop.set()
        with contextlib.suppress(OSError):
            self._sock.close()
        self._thread.join(timeout=2.0)


class SendWaitsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "control.sock"
        self.original_sock = reach_vr.SOCK
        self.original_wait = reach_vr.SOCK_WAIT_S
        reach_vr.SOCK = self.path
        self.addCleanup(self.restore)

    def restore(self) -> None:
        reach_vr.SOCK = self.original_sock
        reach_vr.SOCK_WAIT_S = self.original_wait

    def test_send_waits_for_a_socket_that_is_being_recreated(self) -> None:
        server = ControlServer(self.path, delay=0.4)
        self.addCleanup(server.close)
        started = time.monotonic()
        reply = reach_vr.send(verb="rest")
        elapsed = time.monotonic() - started
        self.assertTrue(reply.get("ok"))
        self.assertGreaterEqual(elapsed, 0.3)
        self.assertEqual([verb.get("verb") for verb in server.verbs], ["rest"])

    def test_send_gives_up_after_the_bounded_wait(self) -> None:
        reach_vr.SOCK_WAIT_S = 0.3
        started = time.monotonic()
        with self.assertRaises(OSError):
            reach_vr.send(verb="rest")
        self.assertGreaterEqual(time.monotonic() - started, 0.25)


if __name__ == "__main__":
    unittest.main()
