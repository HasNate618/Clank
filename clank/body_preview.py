"""Third-person body view as a dashboard sidecar.

Deliberately **not** part of `SessionManager`. The body view is a MuJoCo process
that has nothing to do with Monado, Weston, Steam or the game: it can run with
no session at all, it must not delay a session start, and if it dies the session
must not notice. Keeping it out of the session state machine also keeps that
machine's tested process invariants intact.

What it owns: one child process, one output directory, and a health snapshot.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from .config import BodyConfig

# The verbs the viewer knows how to turn into a BodyCommand. Kept here so the
# dashboard rejects a typo before it reaches the socket; the authoritative map
# is `verb_command()` in spike/wbc/body_view.py.
BODY_VIEWS: tuple[str, ...] = ("three_quarter", "front", "side", "back")

BODY_VERBS: tuple[str, ...] = (
    "hold",
    "look_left",
    "look_right",
    "aim",
    "punch",
    "dodge_left",
    "dodge_right",
    # Buttons. Superhot grabs on a squeeze and fires on a trigger, and the body
    # has neither, so each is a verb with a side: a grab with no side cannot be
    # bound to anything.
    "grab_left",
    "grab_right",
    "release_left",
    "release_right",
    "fire_left",
    "fire_right",
)


def send_control(path: Path | str, payload: dict, timeout: float = 2.0) -> dict:
    """Send one control message and return the reply. Raises OSError if unreachable."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(path))
            sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            buf = b""
            while b"\n" not in buf:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
        reply = json.loads(buf.decode("utf-8", "replace") or "{}")
    except (OSError, ValueError, UnicodeError) as exc:
        raise OSError(f"body control {path} failed: {exc}") from exc
    return reply if isinstance(reply, dict) else {"ok": False, "error": "bad reply"}


class BodyPreview:
    """Runs `body_view.py` and exposes the newest frame it wrote."""

    def __init__(self, cfg: BodyConfig, out_dir: Path, log_path: Path | None = None) -> None:
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.log_path = Path(log_path) if log_path else self.out_dir / "body_view.log"
        self._proc: subprocess.Popen[bytes] | None = None
        self._error = ""

    # ---- lifecycle ------------------------------------------------------

    def argv(self) -> list[str]:
        return [
            self.cfg.python,
            self.cfg.script,
            "--out", str(self.out_dir),
            "--fps", f"{self.cfg.fps:g}",
            "--width", str(self.cfg.width),
            "--height", str(self.cfg.height),
            "--control", str(self.control_path()),
            "--views", ",".join(self.cfg.views),
        ] + (["--reel"] if self.cfg.reel else []) + ([] if self.cfg.render else ["--no-render"])

    def control_path(self) -> Path:
        return self.out_dir / "control.sock"

    def state_path(self) -> Path:
        """Where `body_view.py` publishes what it achieved, and the relay reads it."""
        return self.out_dir / "body_state.json"

    def forget_state(self) -> None:
        """Delete the last run's achieved state, so nothing can latch it.

        `pose_relay.py` holds the last state it read through a quiet body, buttons
        included - deliberately, because a dead body must not release a grab. That
        is right mid-run and wrong between runs: for the seconds a reset takes, the
        relay would keep sending the previous run's final pose, and if that run
        ended with a closed hand, a held squeeze at the start of the next one. With
        no file at all the relay sends `wire.REST_POSES` and both buttons up, which
        is the only honest thing to send while the new sim loads.
        """
        for name in ("body_state.json", "body_latest.jpg"):
            with contextlib.suppress(OSError):
                (self.out_dir / name).unlink()

    def owner_elsewhere(self) -> bool:
        """True when a process this object did not spawn is serving the socket.

        `stop()` cannot touch a viewer it never started (`pick_pixel.py`, a gate, a
        `body_view.py` an operator left running, one that outlived a dashboard
        restart), so a run that carries on either loses a start to the busy socket
        or - worse - has its `rest` answered by a stranger. Ask once, and say what
        to do instead of retrying a race.
        """
        if self.running():
            return False
        path = self.control_path()
        try:
            if not path.exists():
                return False
        except OSError:
            return False
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.5)
        try:
            probe.connect(str(path))
        except OSError:
            return False
        finally:
            probe.close()
        return True

    def preflight(self) -> str:
        """Empty string when the sidecar can start, else why it cannot."""
        if not Path(self.cfg.python).exists():
            return f"body python not found: {self.cfg.python}"
        if not Path(self.cfg.script).exists():
            return f"body view not found: {self.cfg.script}"
        return ""

    def start(self) -> str:
        if self._proc is not None and self._proc.poll() is None:
            return ""
        problem = self.preflight()
        if problem:
            self._error = problem
            return problem
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._error = ""
        # The sidecar runs its own interpreter (a SONIC venv), so the parent's
        # PYTHONPATH must not leak in: a dashboard started through `nix develop`
        # carries the devshell's 3.14 site-packages, and the venv's 3.12 Python
        # would import that numpy and die on a cpython-314 extension.
        env = os.environ.copy()
        for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            env.pop(key, None)
        with self.log_path.open("ab") as log:
            self._proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                self.argv(),
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                start_new_session=True,  # own process group, so stop() is clean
                cwd=str(Path.cwd()),
            )
        return ""

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5)

    # ---- reads ----------------------------------------------------------

    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def frame(self, view: str | None = None) -> Path | None:
        """The newest JPEG for one angle. None means the primary tile."""
        name = "body_latest.jpg" if not view else f"body_{view}.jpg"
        path = self.out_dir / name
        try:
            if path.is_file():
                return path
        except OSError:
            return None
        return None

    def latest(self) -> Path | None:
        return self.frame()

    def _log_tail_line(self) -> str:
        """The last non-empty line the viewer wrote, for summarizing a crash."""
        try:
            with self.log_path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 4096))
                text = handle.read().decode("utf-8", "replace")
        except OSError:
            return ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1][:200] if lines else ""

    def camera(self) -> dict:
        """What the viewer reports about its angles; empty when it has not yet."""
        try:
            raw = (self.out_dir / "body_camera.json").read_text(encoding="utf-8")
        except OSError:
            return {}
        try:
            doc = json.loads(raw)
        except ValueError:
            return {}
        return doc if isinstance(doc, dict) else {}

    def status(self) -> dict:
        frame = self.latest()
        cam = self.camera()
        views = cam.get("views") or list(self.cfg.views)
        proc = self._proc
        alive = proc is not None and proc.poll() is None
        exit_code = None if proc is None or alive else proc.returncode
        age = None if frame is None else round(time.time() - frame.stat().st_mtime, 2)
        error = self._error
        if not error and exit_code is not None:
            tail = self._log_tail_line()
            error = f"body view exited ({exit_code})" + (f": {tail}" if tail else "")
        return {
            "enabled": self.cfg.enabled,
            "running": alive,
            "exit_code": exit_code,
            # A frame can be fresh while our child is dead: an older viewer may
            # still be writing to the shared output directory.
            "frames_unmanaged": bool(not alive and age is not None and age < 5.0),
            "frame": str(frame) if frame else None,
            "frame_age_s": age,
            "views": views,
            "primary": cam.get("primary") or (views[0] if views else "three_quarter"),
            "follow": bool(cam.get("follow", True)),
            "lookat": cam.get("lookat", {}),
            "error": error,
            "log": str(self.log_path),
        }
