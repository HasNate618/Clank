from __future__ import annotations

import contextlib
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from .capture import capture_frame
from .config import Config


def newest_png(directory: Path, started_ns: int) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    try:
        paths = list(Path(directory).glob("*.png"))
    except OSError:
        return None
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_mtime_ns >= started_ns:
            candidates.append((stat.st_mtime_ns, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def atomic_replace(tmp: Path, destination: Path) -> None:
    os.replace(tmp, destination)


class PreviewCapture:
    def __init__(
        self,
        *,
        width: int,
        height: int,
        panel_height: int,
        fps: float,
        screenshooter: str,
        runtime_dir: Path,
        wayland_display: str,
        work_dir: Path,
        publish_path: Path,
        stop_event: threading.Event | None = None,
        run: Callable = subprocess.run,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.panel_height = panel_height
        self.fps = fps
        self.screenshooter = screenshooter
        self.runtime_dir = Path(runtime_dir)
        self.wayland_display = wayland_display
        self.work_dir = Path(work_dir)
        self.publish_path = Path(publish_path)
        self.stop_event = stop_event or threading.Event()
        self.run = run
        self.log = log or (lambda message: None)
        self._thread: threading.Thread | None = None
        self._state = "new"
        self._state_lock = threading.Lock()
        self._lock = threading.Lock()
        self._frames = 0
        self._errors = 0
        self._dropped = 0
        self._last_frame_at: str | None = None
        self._warned_at = 0.0
        self._started_monotonic = time.monotonic()

    def start(self) -> None:
        with self._state_lock:
            if self._state != "new" or self.stop_event.is_set():
                raise RuntimeError("preview capture already started")
            self.publish_path.parent.mkdir(parents=True, exist_ok=True)
            self._started_monotonic = time.monotonic()
            thread = threading.Thread(target=self._run, name="clank-preview", daemon=True)
            try:
                thread.start()
            except BaseException:
                self._state = "new"
                raise
            self._thread = thread
            self._state = "started"

    def stop(self, timeout: float = 5.0) -> None:
        with self._state_lock:
            self._state = "stopped"
            self.stop_event.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                self.log("preview: capture thread did not stop within timeout")

    def status(self) -> dict:
        with self._lock:
            elapsed = max(0.001, time.monotonic() - self._started_monotonic)
            return {
                "width": self.width,
                "height": self.height - self.panel_height,
                "fps_target": self.fps,
                "fps_actual": round(self._frames / elapsed, 1),
                "last_frame_at": self._last_frame_at,
                "errors": self._errors,
                "frames": self._frames,
            }

    def _run(self) -> None:
        interval = 1.0 / self.fps
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                self.capture_once()
            except Exception as exc:
                with self._lock:
                    self._errors += 1
                self._warn(f"preview loop: {exc}")
            remaining = interval - (time.monotonic() - started)
            if remaining > 0:
                self.stop_event.wait(remaining)
            else:
                with self._lock:
                    self._dropped += 1

    def capture_once(self) -> bool:
        try:
            capture_frame(
                screenshooter=self.screenshooter,
                runtime_dir=self.runtime_dir,
                wayland_display=self.wayland_display,
                work_dir=self.work_dir,
                destination=self.publish_path,
                timeout=5,
                run=self.run,
            )
            with self._lock:
                self._frames += 1
                self._last_frame_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            return True
        except Exception as exc:
            with self._lock:
                self._errors += 1
            self._warn(str(exc))
            return False

    def _warn(self, message: str) -> None:
        now = time.monotonic()
        should_log = False
        with self._lock:
            if now - self._warned_at >= 10.0:
                self._warned_at = now
                should_log = True
        if should_log:
            self.log(f"preview: {message}")


def _kill_group(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(pid), signal.SIGTERM)


def preview_smoke(cfg: Config) -> int:
    weston = shutil.which(cfg.paths.weston_bin)
    screenshooter = shutil.which(cfg.paths.weston_screenshooter_bin)
    magick = shutil.which("magick")
    mpv = shutil.which("mpv")
    if not (weston and screenshooter and mpv):
        print("preview smoke skipped: weston, weston-screenshooter, or mpv unavailable")
        return 0

    runtime = Path(os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}")
    socket_name = f"clank-smoke-{os.getpid()}"
    weston_proc: subprocess.Popen | None = None
    mpv_proc: subprocess.Popen | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="clank-smoke-") as raw:
            work = Path(raw)
            weston_log = work / "weston.log"
            env = os.environ.copy()
            env["XDG_RUNTIME_DIR"] = str(runtime)
            weston_proc = subprocess.Popen(
                [
                    weston,
                    "--backend=headless",
                    "--renderer=gl",
                    "--width=1280",
                    "--height=720",
                    f"--socket={socket_name}",
                    "--no-config",
                    "--xwayland",
                    "--debug",
                    f"--log={weston_log}",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
            )
            deadline = time.monotonic() + 10
            display = None
            while time.monotonic() < deadline:
                if (runtime / socket_name).exists() and weston_log.exists():
                    text = weston_log.read_text(errors="replace")
                    match = re.search(r"xserver listening on display (:\d+)", text)
                    if match:
                        display = match.group(1)
                        break
                time.sleep(0.2)
            if display is None:
                print("preview smoke failed: Weston did not publish an Xwayland display")
                return 1
            mpv_env = env.copy()
            mpv_env["WAYLAND_DISPLAY"] = socket_name
            mpv_env["DISPLAY"] = display
            mpv_proc = subprocess.Popen(
                [
                    mpv,
                    "--no-config",
                    "--really-quiet",
                    "--loop=inf",
                    "--no-audio",
                    "--vo=x11",
                    "--force-window=yes",
                    "av://lavfi:testsrc2=size=1280x720:rate=10",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=mpv_env,
                start_new_session=True,
            )
            time.sleep(2)
            loop_stop = threading.Event()
            capture = PreviewCapture(
                width=1280,
                height=720,
                panel_height=32,
                fps=12,
                screenshooter=screenshooter,
                runtime_dir=runtime,
                wayland_display=socket_name,
                work_dir=work,
                publish_path=work / "latest.png",
                stop_event=loop_stop,
                log=print,
            )
            capture.start()
            time.sleep(2)
            capture.stop()
            latest = work / "latest.png"
            ok = False
            if latest.exists() and latest.stat().st_size > 1000:
                if magick is None:
                    ok = True
                else:
                    result = subprocess.run(
                        [magick, str(latest), "-colorspace", "gray", "-format", "%[fx:mean]", "info:"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result.returncode == 0:
                        try:
                            ok = float(result.stdout.strip()) > 0.01
                        except ValueError:
                            ok = False
            print("preview smoke ok" if ok else "preview smoke failed")
            return 0 if ok else 1
    finally:
        for proc in (mpv_proc, weston_proc):
            if proc is not None:
                _kill_group(proc.pid)
        for proc in (mpv_proc, weston_proc):
            if proc is not None:
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait(timeout=3)
        with contextlib.suppress(FileNotFoundError):
            (runtime / socket_name).unlink()
