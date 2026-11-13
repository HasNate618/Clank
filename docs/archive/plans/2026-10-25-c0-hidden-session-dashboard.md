# C0 Hidden Session Dashboard Implementation Plan

> **Archived 2026-11-13.** Implementation plan, carried out. Kept for the task breakdown and the gates it defined; the code is the record now.
> Current behaviour: [`README.md`](../../../README.md) · agent rules: [`AGENTS.md`](../../../AGENTS.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local dashboard that launches, monitors, previews, and cleanly stops one hidden C0 session (Monado, Weston headless, Steam, Superhot VR, and the pose client).

**Architecture:** A stdlib Python package owns the lifecycle. `SessionManager` runs a single state machine and process groups; a `ThreadingHTTPServer` exposes JSON controls and an MJPEG preview; `PreviewCapture` screenshots Weston headlessly, crops the panel, and publishes JPEG frames atomically.

**Tech Stack:** Python 3 stdlib (`tomllib`, `unittest`, `http.server`, `subprocess`, `threading`), Weston 16 (`weston`, `weston-screenshooter`, `weston-debug`), ImageMagick `magick`, Steam/Proton, and Monado.

---

## File Structure

- `config.toml` — machine-local defaults from the approved spec.
- `run.sh` — puts Weston on `PATH` when needed, then starts the dashboard.
- `clank/__init__.py` — package version.
- `clank/__main__.py` — CLI entry point and `--preview-smoke`.
- `clank/config.py` — TOML/env loading, type and bounds validation, required-path checks.
- `clank/processes.py` — process groups, logs, `/proc` probes, pointer file, sockets.
- `clank/preview.py` — screenshot loop, crop math, atomic JPEG publication, smoke helper.
- `clank/orchestrator.py` — state machine, startup/shutdown, mode switching, health snapshot.
- `clank/dashboard.py` — HTTP routes, JSON, MJPEG, static files.
- `clank/static/index.html`, `style.css`, `app.js` — dashboard UI.
- `tests/__init__.py`, `tests/test_config.py`, `tests/test_processes.py`, `tests/test_preview.py`, `tests/test_orchestrator.py`, `tests/test_dashboard.py`.

The approved design is `docs/archive/specs/2026-10-25-c0-hidden-session-dashboard-design.md`.

---

### Task 1: Configuration and defaults

**Files:**
- Create: `config.toml`
- Create: `clank/__init__.py`
- Create: `clank/config.py`
- Create: `tests/__init__.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing config tests**

Create `tests/__init__.py` as an empty file.

Create `tests/test_config.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clank.config import ConfigError, load_config, missing_requirements


class ConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = load_config(None, env={})
        self.assertEqual(cfg.preview.width, 1280)
        self.assertEqual(cfg.preview.height, 720)
        self.assertEqual(cfg.preview.panel_height, 32)
        self.assertEqual(cfg.preview.fps, 8.0)
        self.assertEqual(cfg.dashboard.port, 8765)
        self.assertEqual(cfg.launch.shortcut_id, "18251483899710406656")
        self.assertTrue(cfg.paths.pose_client.endswith("spike/c0/c0_client.py"))

    def test_toml_and_env_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.toml"
            path.write_text("[preview]\nfps = 6\n\n[dashboard]\nport = 9000\n")
            cfg = load_config(path, env={"CLANK_PREVIEW_FPS": "12"})
            self.assertEqual(cfg.preview.fps, 12.0)
            self.assertEqual(cfg.dashboard.port, 9000)

    def test_unknown_toml_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.toml"
            path.write_text("[preview]\nnope = 1\n")
            with self.assertRaises(ConfigError):
                load_config(path, env={})

    def test_unknown_env_key_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            load_config(None, env={"CLANK_NOPE_X": "1"})

    def test_wrong_type_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.toml"
            path.write_text('[preview]\nfps = "fast"\n')
            with self.assertRaises(ConfigError):
                load_config(path, env={})

    def test_bounds_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.toml"
            path.write_text("[preview]\npanel_height = 720\n")
            with self.assertRaises(ConfigError):
                load_config(path, env={})

    def test_weston_siblings_are_derived(self) -> None:
        cfg = load_config(
            None,
            env={"CLANK_PATHS_WESTON_BIN": "/opt/weston/bin/weston"},
        )
        self.assertEqual(cfg.paths.weston_screenshooter_bin, "/opt/weston/bin/weston-screenshooter")
        self.assertEqual(cfg.paths.weston_debug_bin, "/opt/weston/bin/weston-debug")

    def test_missing_requirements_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            names = [
                "monado-service",
                "weston",
                "weston-screenshooter",
                "weston-debug",
                "magick",
                "steam",
                "game.exe",
                "pose.py",
                "runtime.json",
            ]
            for name in names:
                path = base / name
                path.write_text("")
                path.chmod(0o755)
            env = {
                "CLANK_PATHS_MONADO_BIN": str(base / "monado-service"),
                "CLANK_PATHS_WESTON_BIN": str(base / "weston"),
                "CLANK_PATHS_MAGICK_BIN": str(base / "magick"),
                "CLANK_PATHS_STEAM_BIN": str(base / "steam"),
                "CLANK_PATHS_GAME_EXE": str(base / "game.exe"),
                "CLANK_PATHS_POSE_CLIENT": str(base / "pose.py"),
                "CLANK_PATHS_OPENXR_ACTIVE_RUNTIME": str(base / "runtime.json"),
                "CLANK_PATHS_SESSION_ROOT": str(base / "sessions"),
            }
            cfg = load_config(None, env=env)
            self.assertEqual(missing_requirements(cfg), [])
            (base / "magick").unlink()
            self.assertTrue(any("magick" in item for item in missing_requirements(cfg)))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_config -v`

Expected: `ModuleNotFoundError: No module named 'clank'` or an import error for `ConfigError`.

- [ ] **Step 3: Create the package and config module**

Create `clank/__init__.py`:

```python
"""Clank hidden C0 session orchestrator."""

__version__ = "0.1.0"
```

Create `clank/config.py`:

```python
from __future__ import annotations

import os
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when config.toml or a CLANK_ override is invalid."""


@dataclass(frozen=True)
class PathsConfig:
    monado_bin: str
    weston_bin: str
    weston_screenshooter_bin: str
    weston_debug_bin: str
    magick_bin: str
    steam_bin: str
    game_exe: str
    pose_client: str
    openxr_active_runtime: str
    session_root: str


@dataclass(frozen=True)
class LaunchConfig:
    shortcut_id: str
    proton_app_id: str
    steam_extra_args: tuple[str, ...]
    steam_settle_sec: float
    steam_ready_timeout_sec: float
    game_process_timeout_sec: float
    openxr_client_timeout_sec: float
    startup_timeout_sec: float


@dataclass(frozen=True)
class PreviewConfig:
    width: int
    height: int
    panel_height: int
    fps: float
    jpeg_quality: int


@dataclass(frozen=True)
class DashboardConfig:
    host: str
    port: int


@dataclass(frozen=True)
class TimeoutsConfig:
    monado_startup_sec: float
    weston_startup_sec: float
    pose_handshake_sec: float
    stop_sec: float


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    launch: LaunchConfig
    preview: PreviewConfig
    dashboard: DashboardConfig
    timeouts: TimeoutsConfig


DEFAULTS: dict[str, dict[str, Any]] = {
    "paths": {
        "monado_bin": "/run/wrappers/bin/monado-service",
        "weston_bin": "weston",
        "weston_screenshooter_bin": "weston-screenshooter",
        "weston_debug_bin": "weston-debug",
        "magick_bin": "magick",
        "steam_bin": "/home/nate/.local/bin/steam",
        "game_exe": "/home/nate/Games/SUPERHOT VR/SUPERHOT_VR.exe",
        "pose_client": "spike/c0/c0_client.py",
        "openxr_active_runtime": "/home/nate/.config/openxr/1/active_runtime.json",
        "session_root": "/home/nate/.local/state/Clank/sessions",
    },
    "launch": {
        "shortcut_id": "18251483899710406656",
        "proton_app_id": "4249504744",
        "steam_extra_args": ["-cef-disable-gpu"],
        "steam_settle_sec": 60.0,
        "steam_ready_timeout_sec": 120.0,
        "game_process_timeout_sec": 60.0,
        "openxr_client_timeout_sec": 60.0,
        "startup_timeout_sec": 300.0,
    },
    "preview": {
        "width": 1280,
        "height": 720,
        "panel_height": 32,
        "fps": 8.0,
        "jpeg_quality": 80,
    },
    "dashboard": {
        "host": "127.0.0.1",
        "port": 8765,
    },
    "timeouts": {
        "monado_startup_sec": 20.0,
        "weston_startup_sec": 20.0,
        "pose_handshake_sec": 10.0,
        "stop_sec": 60.0,
    },
}


def _coerce(value: Any, default: Any, where: str) -> Any:
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"{where} must be a boolean")
        return value
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where} must be an integer")
        return value
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where} must be a number")
        return float(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise ConfigError(f"{where} must be a string")
        return value
    if isinstance(default, (list, tuple)):
        if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
            raise ConfigError(f"{where} must be a list of strings")
        return tuple(value)
    raise ConfigError(f"unsupported default for {where}")


def _parse_env_value(section: str, key: str, raw: str, default: Any) -> Any:
    where = f"{section}.{key}"
    try:
        if isinstance(default, bool):
            lowered = raw.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
            raise ConfigError(f"{where} must be a boolean")
        if isinstance(default, int):
            return int(raw)
        if isinstance(default, float):
            return float(raw)
        if isinstance(default, (list, tuple)):
            return tuple(raw.split())
        return raw
    except ValueError as exc:
        raise ConfigError(f"invalid value for {where}: {raw!r}") from exc


def _validate(cfg: Config) -> None:
    if cfg.preview.width <= 0 or cfg.preview.height <= 0:
        raise ConfigError("preview width and height must be positive")
    if cfg.preview.panel_height < 0 or cfg.preview.panel_height >= cfg.preview.height:
        raise ConfigError("preview.panel_height must be >= 0 and less than preview.height")
    if cfg.preview.fps <= 0:
        raise ConfigError("preview.fps must be positive")
    if not 1 <= cfg.preview.jpeg_quality <= 100:
        raise ConfigError("preview.jpeg_quality must be between 1 and 100")
    if not 0 <= cfg.dashboard.port <= 65535:
        raise ConfigError("dashboard.port must be between 0 and 65535")
    for field_name in (
        "steam_settle_sec",
        "steam_ready_timeout_sec",
        "game_process_timeout_sec",
        "openxr_client_timeout_sec",
        "startup_timeout_sec",
    ):
        if getattr(cfg.launch, field_name) <= 0:
            raise ConfigError(f"launch.{field_name} must be positive")
    for field_name in (
        "monado_startup_sec",
        "weston_startup_sec",
        "pose_handshake_sec",
        "stop_sec",
    ):
        if getattr(cfg.timeouts, field_name) <= 0:
            raise ConfigError(f"timeouts.{field_name} must be positive")


def load_config(
    path: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    base_dir: Path | str | None = None,
) -> Config:
    base = Path(base_dir or Path.cwd())
    file_data: dict[str, Any] = {}
    if path is not None:
        candidate = Path(path)
        if candidate.exists():
            with candidate.open("rb") as handle:
                file_data = tomllib.load(handle)

    unknown_sections = set(file_data) - set(DEFAULTS)
    if unknown_sections:
        raise ConfigError(f"unknown config sections: {', '.join(sorted(unknown_sections))}")

    merged = {section: dict(values) for section, values in DEFAULTS.items()}
    for section, values in file_data.items():
        if not isinstance(values, dict):
            raise ConfigError(f"{section} must be a table")
        unknown = set(values) - set(DEFAULTS[section])
        if unknown:
            raise ConfigError(f"unknown keys in {section}: {', '.join(sorted(unknown))}")
        for key, value in values.items():
            merged[section][key] = _coerce(value, DEFAULTS[section][key], f"{section}.{key}")

    for env_key, raw in (env or {}).items():
        if not env_key.startswith("CLANK_"):
            continue
        rest = env_key[len("CLANK_") :]
        if "_" not in rest:
            raise ConfigError(f"invalid environment override {env_key}")
        raw_section, raw_key = rest.split("_", 1)
        section = raw_section.lower()
        key = raw_key.lower()
        if section not in DEFAULTS or key not in DEFAULTS[section]:
            raise ConfigError(f"unknown environment override {env_key}")
        default = DEFAULTS[section][key]
        value = _parse_env_value(section, key, raw, default)
        merged[section][key] = _coerce(value, default, f"{section}.{key}")

    paths = merged["paths"]
    if os.sep in paths["weston_bin"]:
        parent = Path(paths["weston_bin"]).parent
        for key, sibling in (
            ("weston_screenshooter_bin", "weston-screenshooter"),
            ("weston_debug_bin", "weston-debug"),
        ):
            if paths[key] == DEFAULTS["paths"][key]:
                paths[key] = str(parent / sibling)

    for key in ("pose_client", "session_root"):
        candidate = Path(paths[key])
        if not candidate.is_absolute():
            paths[key] = str((base / candidate).resolve())

    cfg = Config(
        paths=PathsConfig(**merged["paths"]),
        launch=LaunchConfig(**merged["launch"]),
        preview=PreviewConfig(**merged["preview"]),
        dashboard=DashboardConfig(**merged["dashboard"]),
        timeouts=TimeoutsConfig(**merged["timeouts"]),
    )
    _validate(cfg)
    return cfg


def _resolve_executable(value: str) -> Path | None:
    if os.sep in value:
        path = Path(value)
        return path if path.exists() else None
    found = shutil.which(value)
    return Path(found) if found else None


def missing_requirements(cfg: Config) -> list[str]:
    checks = [
        ("monado", cfg.paths.monado_bin, os.X_OK),
        ("weston", cfg.paths.weston_bin, os.X_OK),
        ("weston-screenshooter", cfg.paths.weston_screenshooter_bin, os.X_OK),
        ("weston-debug", cfg.paths.weston_debug_bin, os.X_OK),
        ("magick", cfg.paths.magick_bin, os.X_OK),
        ("steam", cfg.paths.steam_bin, os.X_OK),
        ("game", cfg.paths.game_exe, os.R_OK),
        ("openxr runtime", cfg.paths.openxr_active_runtime, os.R_OK),
        ("pose client", cfg.paths.pose_client, os.R_OK),
    ]
    missing: list[str] = []
    for label, value, mode in checks:
        resolved = _resolve_executable(value)
        if resolved is None or not os.access(resolved, mode):
            missing.append(f"{label} ({value})")
    return missing
```

Create `config.toml`:

```toml
[paths]
monado_bin = "/run/wrappers/bin/monado-service"
weston_bin = "weston"
weston_screenshooter_bin = "weston-screenshooter"
weston_debug_bin = "weston-debug"
magick_bin = "magick"
steam_bin = "/home/nate/.local/bin/steam"
game_exe = "/home/nate/Games/SUPERHOT VR/SUPERHOT_VR.exe"
pose_client = "spike/c0/c0_client.py"
openxr_active_runtime = "/home/nate/.config/openxr/1/active_runtime.json"
session_root = "/home/nate/.local/state/Clank/sessions"

[launch]
shortcut_id = "18251483899710406656"
proton_app_id = "4249504744"
steam_extra_args = ["-cef-disable-gpu"]
steam_settle_sec = 60
steam_ready_timeout_sec = 120
game_process_timeout_sec = 60
openxr_client_timeout_sec = 60
startup_timeout_sec = 300

[preview]
width = 1280
height = 720
panel_height = 32
fps = 8
jpeg_quality = 80

[dashboard]
host = "127.0.0.1"
port = 8765

[timeouts]
monado_startup_sec = 20
weston_startup_sec = 20
pose_handshake_sec = 10
stop_sec = 60
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_config -v`

Expected: 8 tests, `OK`.

- [ ] **Step 5: Commit**

```bash
git add config.toml clank/__init__.py clank/config.py tests/__init__.py tests/test_config.py
git commit -m "feat: add clank configuration"
```

---

### Task 2: Process primitives and pointer file

**Files:**
- Create: `clank/processes.py`
- Test: `tests/test_processes.py`

- [ ] **Step 1: Write the failing process tests**

Create `tests/test_processes.py`:

```python
from __future__ import annotations

import json
import os
import socket
import tempfile
import unittest
from pathlib import Path

from clank.processes import ProcessRunner


class ProcessTests(unittest.TestCase):
    def test_pointer_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            pointer = Path(raw) / "current-session.json"
            runner = ProcessRunner(pointer)
            self.assertIsNone(runner.read_pointer())
            runner.write_pointer({"session": "/tmp/x", "processes": {}})
            self.assertEqual(runner.read_pointer()["session"], "/tmp/x")
            runner.remove_pointer()
            self.assertIsNone(runner.read_pointer())

    def test_proc_scan_and_cmdline(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            proc_root = Path(raw)
            pid_dir = proc_root / "123"
            pid_dir.mkdir()
            argv0 = r"Z:\home\nate\Games\SUPERHOT VR\SUPERHOT_VR.exe"
            (pid_dir / "cmdline").write_bytes(argv0.encode() + b"\0")
            runner = ProcessRunner(Path(raw) / "pointer.json", proc_root=proc_root)
            self.assertEqual(runner.pids_by_argv0(argv0), [123])
            self.assertEqual(runner.pid_cmdline(123)[0], argv0)
            self.assertTrue(runner.pid_alive(123))
            self.assertFalse(runner.pid_alive(999))

    def test_remove_stale_socket(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "monado_comp_ipc"
            sock = socket.socket(socket.AF_UNIX)
            sock.bind(str(path))
            sock.close()
            runner = ProcessRunner(Path(raw) / "pointer.json")
            self.assertTrue(runner.remove_stale_socket(path))
            self.assertFalse(path.exists())
            self.assertFalse(runner.remove_stale_socket(path))

    def test_start_and_stop_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            runner = ProcessRunner(base / "pointer.json")
            handle = runner.start(
                "sleeper",
                ["/bin/sh", "-c", "sleep 30"],
                os.environ.copy(),
                base,
                base / "sleeper.log",
            )
            self.assertTrue(handle.alive())
            self.assertTrue((base / "sleeper.log").exists())
            handle.stop(timeout=1)
            self.assertFalse(handle.alive())

    def test_tail(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "log.txt"
            path.write_text("\n".join(str(i) for i in range(300)))
            runner = ProcessRunner(Path(raw) / "pointer.json")
            text = runner.tail(path, 5)
            self.assertEqual(text.splitlines(), ["295", "296", "297", "298", "299"])

    def test_pointer_is_json_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            pointer = Path(raw) / "pointer.json"
            runner = ProcessRunner(pointer)
            runner.write_pointer({"a": 1})
            self.assertEqual(json.loads(pointer.read_text()), {"a": 1})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_processes -v`

Expected: `ModuleNotFoundError` for `clank.processes`.

- [ ] **Step 3: Implement process primitives**

Create `clank/processes.py`:

```python
from __future__ import annotations

import json
import os
import signal
import socket
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterator


class Aborted(RuntimeError):
    """Raised when a wait is interrupted by a session stop request."""


class ProcessError(RuntimeError):
    """Raised for process lifecycle failures."""


@dataclass
class ProcessHandle:
    name: str
    pid: int
    pgid: int
    log_path: Path
    _popen: subprocess.Popen | None = None
    _handles: tuple[IO[bytes], ...] = ()

    def alive(self) -> bool:
        if self._popen is not None:
            return self._popen.poll() is None
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def signal_group(self, sig: int) -> None:
        try:
            os.killpg(self.pgid, sig)
        except ProcessLookupError:
            pass

    def stop(self, timeout: float = 5.0) -> None:
        if not self.alive():
            self._close()
            return
        self.signal_group(signal.SIGTERM)
        done = False
        if self._popen is not None:
            try:
                self._popen.wait(timeout=timeout)
                done = True
            except subprocess.TimeoutExpired:
                pass
        else:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if not self.alive():
                    done = True
                    break
                time.sleep(0.1)
        if not done and self.alive():
            self.signal_group(signal.SIGKILL)
            if self._popen is not None:
                try:
                    self._popen.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        self._close()

    def _close(self) -> None:
        for handle in self._handles:
            try:
                handle.close()
            except OSError:
                pass
        self._handles = ()


class ProcessRunner:
    def __init__(self, pointer_path: Path, proc_root: Path = Path("/proc")) -> None:
        self.pointer_path = Path(pointer_path)
        self.proc_root = Path(proc_root)

    def start(
        self,
        name: str,
        argv: list[str] | tuple[str, ...],
        env: dict[str, str],
        cwd: Path,
        log_path: Path,
        stderr_path: Path | None = None,
    ) -> ProcessHandle:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        out = log_path.open("ab", buffering=0)
        handles: list[IO[bytes]] = [out]
        err = out
        if stderr_path is not None:
            stderr_path = Path(stderr_path)
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            err = stderr_path.open("ab", buffering=0)
            handles.append(err)
        try:
            popen = subprocess.Popen(
                [str(item) for item in argv],
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                cwd=str(cwd),
                env={str(key): str(value) for key, value in env.items()},
                start_new_session=True,
            )
        except Exception:
            for handle in handles:
                handle.close()
            raise
        pgid = os.getpgid(popen.pid)
        return ProcessHandle(name, popen.pid, pgid, log_path, popen, tuple(handles))

    def run(
        self,
        argv: list[str] | tuple[str, ...],
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float = 10.0,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(item) for item in argv],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=str(cwd) if cwd is not None else None,
            timeout=timeout,
            check=check,
        )

    def process_name_running(self, name: str) -> bool:
        result = subprocess.run(
            ["pgrep", "-x", name],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0

    def systemd_unit_active(self, unit: str) -> bool:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", unit],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0

    def port_in_use(self, host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex((host, port)) == 0

    def wait_port(self, host: str, port: int, timeout: float, abort=None) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if abort is not None and abort.is_set():
                raise Aborted("session stop requested")
            if self.port_in_use(host, port):
                return
            time.sleep(0.25)
        raise TimeoutError(f"timed out waiting for {host}:{port}")

    def wait_log(self, path: Path, needle: str, timeout: float, abort=None) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if abort is not None and abort.is_set():
                raise Aborted("session stop requested")
            try:
                text = Path(path).read_text(errors="replace")
            except OSError:
                text = ""
            if needle in text:
                return
            time.sleep(0.25)
        raise TimeoutError(f"timed out waiting for {needle!r} in {path}")

    def wait_pids(self, argv0: str, timeout: float, abort=None) -> list[int]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if abort is not None and abort.is_set():
                raise Aborted("session stop requested")
            pids = self.pids_by_argv0(argv0)
            if pids:
                return pids
            time.sleep(0.5)
        raise TimeoutError(f"timed out waiting for process {argv0}")

    def scan_processes(self) -> Iterator[tuple[int, list[str]]]:
        try:
            entries = list(self.proc_root.iterdir())
        except OSError:
            return
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / "cmdline").read_bytes()
            except OSError:
                continue
            args = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
            if args:
                yield int(entry.name), args

    def pids_by_argv0(self, argv0: str) -> list[int]:
        expected = argv0.lower()
        pids: list[int] = []
        for pid, args in self.scan_processes():
            actual = args[0].lower()
            if actual == expected or actual.endswith(expected):
                pids.append(pid)
        return sorted(pids)

    def pid_alive(self, pid: int) -> bool:
        return (self.proc_root / str(pid)).exists()

    def pid_cmdline(self, pid: int) -> list[str]:
        try:
            raw = (self.proc_root / str(pid) / "cmdline").read_bytes()
        except OSError:
            return []
        return [part.decode(errors="replace") for part in raw.split(b"\0") if part]

    def kill_pid(self, pid: int) -> None:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def debug_scene_graph(self, debug_bin: str, env: dict[str, str], timeout: float = 5.0) -> str | None:
        try:
            result = self.run([debug_bin, "scene-graph"], env=env, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        return result.stdout

    def remove_stale_socket(self, path: Path) -> bool:
        path = Path(path)
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            return False
        if not stat.S_ISSOCK(mode):
            return False
        path.unlink()
        return True

    def read_pointer(self) -> dict | None:
        try:
            return json.loads(self.pointer_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def write_pointer(self, data: dict) -> None:
        self.pointer_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.pointer_path.with_suffix(self.pointer_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        os.replace(tmp, self.pointer_path)

    def remove_pointer(self) -> None:
        try:
            self.pointer_path.unlink()
        except FileNotFoundError:
            pass

    def tail(self, path: Path, lines: int = 200) -> str:
        try:
            data = Path(path).read_text(errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(data[-lines:])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_processes -v`

Expected: 6 tests, `OK`.

- [ ] **Step 5: Commit**

```bash
git add clank/processes.py tests/test_processes.py
git commit -m "feat: add process group helpers"
```

---

### Task 3: Preview capture

**Files:**
- Create: `clank/preview.py`
- Test: `tests/test_preview.py`

- [ ] **Step 1: Write the failing preview tests**

Create `tests/test_preview.py`:

```python
from __future__ import annotations

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from clank.preview import PreviewCapture, atomic_replace, crop_filter, newest_png


class FakeRun:
    def __init__(self, fail: str | None = None) -> None:
        self.fail = fail
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        self.calls.append((argv, kwargs))
        if argv[0] == "weston-screenshooter":
            if self.fail == "screen":
                return subprocess.CompletedProcess(argv, 1, "", "screen failed")
            cwd = Path(kwargs["cwd"])
            shot = cwd / "wayland-screenshot.png"
            shot.write_bytes(b"PNG")
            now = time.time_ns()
            os.utime(shot, ns=(now, now))
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[0] == "magick":
            if self.fail == "magick":
                return subprocess.CompletedProcess(argv, 1, "", "magick failed")
            out = Path(argv[-1][4:])
            out.write_bytes(b"JPEG")
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise AssertionError(f"unexpected command: {argv}")


class PreviewTests(unittest.TestCase):
    def make_capture(self, base: Path, run) -> PreviewCapture:
        return PreviewCapture(
            width=1280,
            height=720,
            panel_height=32,
            fps=8,
            jpeg_quality=80,
            screenshooter="weston-screenshooter",
            magick="magick",
            runtime_dir=base,
            wayland_display="wayland-test",
            work_dir=base,
            publish_path=base / "latest.jpg",
            run=run,
            log=lambda message: None,
        )

    def test_crop_filter(self) -> None:
        self.assertEqual(crop_filter(1280, 720, 32), "1280x688+0+32")

    def test_newest_png_rejects_stale_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            stale = base / "old.png"
            stale.write_bytes(b"old")
            old = time.time_ns() - 10_000_000_000
            os.utime(stale, ns=(old, old))
            self.assertIsNone(newest_png(base, time.time_ns()))

    def test_atomic_replace(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            tmp = base / "latest.jpg.tmp"
            dest = base / "latest.jpg"
            tmp.write_bytes(b"JPEG")
            atomic_replace(tmp, dest)
            self.assertEqual(dest.read_bytes(), b"JPEG")
            self.assertFalse(tmp.exists())

    def test_capture_once_publishes_frame(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            run = FakeRun()
            capture = self.make_capture(base, run)
            self.assertTrue(capture.capture_once())
            self.assertEqual((base / "latest.jpg").read_bytes(), b"JPEG")
            self.assertEqual(capture.status()["errors"], 0)
            self.assertEqual(capture.status()["frames"], 1)
            magick_call = next(call for call in run.calls if call[0][0] == "magick")
            self.assertIn("1280x688+0+32", magick_call[0])
            self.assertTrue(magick_call[0][-1].startswith("jpg:"))

    def test_failed_frame_keeps_previous_frame(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            run = FakeRun()
            capture = self.make_capture(base, run)
            self.assertTrue(capture.capture_once())
            before = (base / "latest.jpg").read_bytes()
            run.fail = "screen"
            self.assertFalse(capture.capture_once())
            self.assertEqual((base / "latest.jpg").read_bytes(), before)
            self.assertEqual(capture.status()["errors"], 1)

    def test_failed_encode_keeps_previous_frame(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            run = FakeRun()
            capture = self.make_capture(base, run)
            self.assertTrue(capture.capture_once())
            before = (base / "latest.jpg").read_bytes()
            run.fail = "magick"
            self.assertFalse(capture.capture_once())
            self.assertEqual((base / "latest.jpg").read_bytes(), before)
            self.assertEqual(capture.status()["errors"], 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_preview -v`

Expected: `ModuleNotFoundError` for `clank.preview`.

- [ ] **Step 3: Implement preview capture**

Create `clank/preview.py`:

```python
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import Config


def crop_filter(width: int, height: int, panel_height: int) -> str:
    return f"{width}x{height - panel_height}+0+{panel_height}"


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
        jpeg_quality: int,
        screenshooter: str,
        magick: str,
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
        self.jpeg_quality = jpeg_quality
        self.screenshooter = screenshooter
        self.magick = magick
        self.runtime_dir = Path(runtime_dir)
        self.wayland_display = wayland_display
        self.work_dir = Path(work_dir)
        self.publish_path = Path(publish_path)
        self.stop_event = stop_event or threading.Event()
        self.run = run
        self.log = log or (lambda message: None)
        self._capture_root = self.work_dir / "preview-capture"
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._frames = 0
        self._errors = 0
        self._dropped = 0
        self._last_frame_at: str | None = None
        self._warned_at = 0.0
        self._started_monotonic = time.monotonic()

    def start(self) -> None:
        self.publish_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="clank-preview", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout)

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
            self.capture_once()
            remaining = interval - (time.monotonic() - started)
            if remaining > 0:
                self.stop_event.wait(remaining)
            else:
                with self._lock:
                    self._dropped += 1

    def capture_once(self) -> bool:
        frame_dir = self._capture_root / str(time.monotonic_ns())
        frame_dir.mkdir(parents=True, exist_ok=True)
        started_ns = time.time_ns()
        try:
            env = os.environ.copy()
            env["XDG_RUNTIME_DIR"] = str(self.runtime_dir)
            env["WAYLAND_DISPLAY"] = self.wayland_display
            result = self.run(
                [self.screenshooter],
                cwd=str(frame_dir),
                env=env,
                timeout=5,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode != 0:
                raise RuntimeError(f"screenshooter exited {result.returncode}")
            png = newest_png(frame_dir, started_ns)
            if png is None:
                raise RuntimeError("screenshooter produced no new PNG")
            tmp = self.publish_path.with_name(self.publish_path.name + ".tmp")
            result = self.run(
                [
                    self.magick,
                    str(png),
                    "-crop",
                    crop_filter(self.width, self.height, self.panel_height),
                    "+repage",
                    "-quality",
                    str(self.jpeg_quality),
                    f"jpg:{tmp}",
                ],
                cwd=str(frame_dir),
                timeout=10,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode != 0:
                raise RuntimeError(f"magick exited {result.returncode}")
            atomic_replace(tmp, self.publish_path)
            with self._lock:
                self._frames += 1
                self._last_frame_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            return True
        except Exception as exc:
            with self._lock:
                self._errors += 1
            self._warn(str(exc))
            return False
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)

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
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        pass


def preview_smoke(cfg: Config) -> int:
    weston = shutil.which(cfg.paths.weston_bin)
    screenshooter = shutil.which(cfg.paths.weston_screenshooter_bin)
    magick = shutil.which(cfg.paths.magick_bin)
    mpv = shutil.which("mpv")
    if not (weston and screenshooter and magick and mpv):
        print("preview smoke skipped: weston, weston-screenshooter, mpv, or magick unavailable")
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
                fps=8,
                jpeg_quality=80,
                screenshooter=screenshooter,
                magick=magick,
                runtime_dir=runtime,
                wayland_display=socket_name,
                work_dir=work,
                publish_path=work / "latest.jpg",
                stop_event=loop_stop,
                log=lambda message: None,
            )
            capture.start()
            time.sleep(2)
            capture.stop()
            latest = work / "latest.jpg"
            ok = False
            if latest.exists():
                result = subprocess.run(
                    [magick, str(latest), "-colorspace", "gray", "-format", "%[fx:mean]", "info:"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
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
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=3)
        try:
            (runtime / socket_name).unlink()
        except FileNotFoundError:
            pass
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_preview -v`

Expected: 6 tests, `OK`.

- [ ] **Step 5: Commit**

```bash
git add clank/preview.py tests/test_preview.py
git commit -m "feat: add headless preview capture"
```

---

### Task 4: Session manager startup

**Files:**
- Create: `clank/orchestrator.py`
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: Write the failing startup tests**

Create `tests/test_orchestrator.py`:

```python
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from clank.config import load_config
from clank.orchestrator import (
    ConflictError,
    PreflightError,
    SessionManager,
    parse_openxr_client,
)


class FakeHandle:
    def __init__(self, name: str, pid: int) -> None:
        self.name = name
        self.pid = pid
        self.pgid = pid
        self.log_path = Path("/tmp") / f"{name}.log"
        self._alive = True

    def alive(self) -> bool:
        return self._alive

    def stop(self, timeout: float = 5.0) -> None:
        self._alive = False

    def signal_group(self, sig: int) -> None:
        self._alive = False


class FakePreview:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self, timeout: float = 5.0) -> None:
        self.stopped = True

    def status(self) -> dict:
        return {
            "width": 1280,
            "height": 688,
            "fps_target": 8,
            "fps_actual": 7.4,
            "last_frame_at": "2026-10-25T22:39:00Z",
            "errors": 0,
            "frames": 10,
        }


class FakeRunner:
    def __init__(self) -> None:
        self.handles: dict[str, FakeHandle] = {}
        self.next_pid = 100
        self.host_steam = False
        self.units: set[str] = set()
        self.port = False
        self.pointer = None
        self.game_started = False
        self.scene_graph: str | None = "window: title: Steam"
        self.fail_at: str | None = None
        self.block_wait: threading.Event | None = None
        self.runs: list[list[str]] = []
        self.alive_pids: set[int] = set()
        self.removed_socket = False

    def start(self, name, argv, env, cwd, log_path, stderr_path=None):
        if self.fail_at == f"start:{name}":
            raise RuntimeError(f"{name} failed to start")
        if name == "weston":
            for arg in argv:
                text = str(arg)
                if text.startswith("--log="):
                    Path(text[len("--log="):]).write_text("xserver listening on display :2\n")
        handle = FakeHandle(name, self.next_pid)
        self.next_pid += 1
        self.handles[name] = handle
        return handle

    def wait_port(self, host, port, timeout, abort=None):
        if self.fail_at == "wait_port":
            raise RuntimeError("monado port timeout")
        if self.block_wait is not None:
            self.block_wait.wait(5)

    def wait_log(self, path, needle, timeout, abort=None):
        if self.fail_at == f"wait_log:{needle}":
            raise RuntimeError(f"log timeout for {needle}")

    def wait_pids(self, argv0, timeout, abort=None):
        if self.fail_at == "wait_pids":
            raise RuntimeError("game process timeout")
        self.game_started = True
        return [555]

    def process_name_running(self, name):
        return self.host_steam

    def systemd_unit_active(self, unit):
        return unit in self.units

    def port_in_use(self, host, port):
        return self.port

    def remove_stale_socket(self, path):
        self.removed_socket = True
        return True

    def pids_by_argv0(self, argv0):
        return [555] if self.game_started else []

    def pid_alive(self, pid):
        if pid in self.alive_pids:
            return True
        return any(handle.pid == pid and handle.alive() for handle in self.handles.values())

    def pid_cmdline(self, pid):
        return ["/nix/store/example/bin/monado-service"]

    def kill_pid(self, pid):
        if pid == 555:
            self.game_started = False

    def run(self, argv, env=None, cwd=None, timeout=10, check=False):
        self.runs.append([str(item) for item in argv])
        return subprocess.CompletedProcess(argv, 0, "", "")

    def debug_scene_graph(self, debug_bin, env, timeout=5):
        return self.scene_graph

    def read_pointer(self):
        return self.pointer

    def write_pointer(self, data):
        self.pointer = json.loads(json.dumps(data))

    def remove_pointer(self):
        self.pointer = None

    def tail(self, path, lines=200):
        return ""


class OrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        cfg = load_config(None, env={})
        cfg = replace(
            cfg,
            paths=replace(cfg.paths, session_root=str(self.base / "sessions")),
            launch=replace(
                cfg.launch,
                steam_settle_sec=0.01,
                steam_ready_timeout_sec=0.05,
                game_process_timeout_sec=1,
                openxr_client_timeout_sec=1,
            ),
            timeouts=replace(
                cfg.timeouts,
                monado_startup_sec=1,
                weston_startup_sec=1,
                pose_handshake_sec=1,
                stop_sec=1,
            ),
        )
        self.cfg = cfg
        self.runner = FakeRunner()
        self.preview = FakePreview()
        self.manager = SessionManager(
            cfg,
            runner=self.runner,
            preview_factory=lambda **kwargs: self.preview,
        )

    def tearDown(self) -> None:
        self.manager.stop()
        self.manager.wait_idle(2)
        self.temp.cleanup()

    def start(self) -> None:
        with patch("clank.orchestrator.missing_requirements", return_value=[]):
            self.manager.start()
        self.assertTrue(self.manager.wait_idle(2))
        self.assertEqual(self.manager.state_value(), "running")

    def test_start_reaches_running(self) -> None:
        self.start()
        self.assertEqual(set(self.runner.handles), {"monado", "weston", "steam", "pose"})
        self.assertTrue(self.preview.started)
        self.assertIsNotNone(self.runner.pointer)

    def test_snapshot_reports_health(self) -> None:
        self.start()
        log_path = self.manager.log_path("monado")
        assert log_path is not None
        log_path.write_text(
            " INFO [client_connected] Client 1 connected\n"
            "\tapplication_name: 'Unity Application'\n"
        )
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["state"], "running")
        self.assertEqual(snapshot["openxr_client"], "Unity Application")
        self.assertTrue(snapshot["processes"]["game"]["alive"])
        self.assertEqual(snapshot["preview"]["width"], 1280)
        self.assertTrue(snapshot["player_log"].endswith("Player.log"))

    def test_host_steam_blocks_start(self) -> None:
        self.runner.host_steam = True
        with patch("clank.orchestrator.missing_requirements", return_value=[]):
            with self.assertRaises(PreflightError):
                self.manager.start()
        self.assertEqual(self.manager.state_value(), "failed")
        self.assertIsNone(self.runner.pointer)
        self.assertEqual(self.runner.handles, {})

    def test_stale_pointer_blocks_start(self) -> None:
        self.runner.alive_pids.add(999)
        self.runner.pointer = {
            "session": "/old",
            "processes": {"monado": {"pid": 999, "command": "monado-service"}},
        }
        with patch("clank.orchestrator.missing_requirements", return_value=[]):
            with self.assertRaises(PreflightError) as raised:
                self.manager.start()
        self.assertIn("stale", str(raised.exception))
        self.assertEqual(self.manager.state_value(), "failed")

    def test_startup_failure_cleans_up(self) -> None:
        self.runner.fail_at = "wait_port"
        with patch("clank.orchestrator.missing_requirements", return_value=[]):
            self.manager.start()
        self.assertTrue(self.manager.wait_idle(2))
        self.assertEqual(self.manager.state_value(), "failed")
        self.assertIn("monado", self.manager.snapshot()["last_error"])
        self.assertIsNone(self.runner.pointer)
        self.assertTrue(all(not handle.alive() for handle in self.runner.handles.values()))

    def test_concurrent_start_is_rejected(self) -> None:
        self.runner.block_wait = threading.Event()
        with patch("clank.orchestrator.missing_requirements", return_value=[]):
            self.manager.start()
            with self.assertRaises(ConflictError):
                self.manager.start()
        self.runner.block_wait.set()
        self.assertTrue(self.manager.wait_idle(2))

    def test_stop_is_idempotent(self) -> None:
        self.start()
        self.manager.stop()
        self.assertTrue(self.manager.wait_idle(2))
        self.manager.stop()
        self.assertEqual(self.manager.state_value(), "stopped")
        self.assertIsNone(self.runner.pointer)
        self.assertTrue(self.preview.stopped)

    def test_mode_switch_restarts_pose_only(self) -> None:
        self.start()
        old_pose = self.runner.handles["pose"]
        self.manager.set_mode("yaw")
        self.assertFalse(old_pose.alive())
        self.assertTrue(self.runner.handles["pose"].alive())
        self.assertEqual(self.manager.snapshot()["mode"], "yaw")

    def test_monado_death_fails_and_cleans_up(self) -> None:
        self.start()
        self.runner.handles["monado"]._alive = False
        self.assertTrue(self.manager.wait_for_state("failed", 3))
        self.assertIn("monado", self.manager.snapshot()["last_error"])
        self.assertIsNone(self.runner.pointer)
        self.assertTrue(all(not handle.alive() for handle in self.runner.handles.values()))


class OpenXRTests(unittest.TestCase):
    def test_parser_handles_multiple_clients(self) -> None:
        log = (
            " INFO [client_connected] Client 1 connected\n"
            "\tapplication_name: 'steam'\n"
            " INFO [client_connected] Client 2 connected\n"
            "\tapplication_name: 'Unity Application'\n"
        )
        self.assertEqual(parse_openxr_client(log), "Unity Application")

    def test_parser_returns_none_after_disconnect(self) -> None:
        log = (
            " INFO [client_connected] Client 1 connected\n"
            "\tapplication_name: 'Unity Application'\n"
            " INFO [client_disconnected] Client 1 disconnected\n"
        )
        self.assertIsNone(parse_openxr_client(log))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_orchestrator -v`

Expected: `ModuleNotFoundError` for `clank.orchestrator`.

- [ ] **Step 3: Implement the session manager**

Create `clank/orchestrator.py`:

```python
from __future__ import annotations

import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable

from .config import Config, missing_requirements
from .preview import PreviewCapture
from .processes import Aborted, ProcessHandle, ProcessRunner


MODES = ("hold", "yaw", "trigger", "dodge", "walk", "clock")


class State(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


class ConflictError(RuntimeError):
    pass


class PreflightError(RuntimeError):
    pass


class StartupError(RuntimeError):
    pass


def parse_openxr_client(text: str) -> str | None:
    active: dict[str, str | None] = {}
    pending: str | None = None
    for line in text.splitlines():
        match = re.search(r"\[client_connected\]\s+Client\s+(\d+)\s+connected", line)
        if match:
            pending = match.group(1)
            continue
        match = re.search(r"application_name:\s*'([^']*)'", line)
        if match and pending is not None:
            name = match.group(1).strip()
            active[pending] = name or None
            pending = None
            continue
        match = re.search(r"\[client_disconnected\]\s+Client\s+(\d+)\s+disconnected", line)
        if match:
            active.pop(match.group(1), None)
    for name in reversed(list(active.values())):
        if name:
            return name
    return None


class SessionManager:
    LOG_FILES = {
        "monado": "monado.log",
        "weston": "weston.log",
        "weston-stdout": "weston-stdout.log",
        "steam": "steam.log",
        "pose": "pose.log",
        "session": "session.log",
    }

    def __init__(
        self,
        cfg: Config,
        runner: ProcessRunner | None = None,
        preview_factory: Callable[..., PreviewCapture] | None = None,
    ) -> None:
        self.cfg = cfg
        self.runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}")
        session_root = Path(cfg.paths.session_root)
        self.runner = runner or ProcessRunner(session_root.parent / "current-session.json")
        self.preview_factory = preview_factory or PreviewCapture
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._lifecycle_lock = threading.Lock()
        self._state = State.STOPPED
        self._session_id: str | None = None
        self._session_dir: Path | None = None
        self._started_at: datetime | None = None
        self._last_error: str | None = None
        self._mode = "hold"
        self._handles: dict[str, ProcessHandle] = {}
        self._game_pids: list[int] = []
        self._display: str | None = None
        self._socket_name: str | None = None
        self._preview: PreviewCapture | None = None
        self._pointer: dict = {}
        self._stop_event = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._thread: threading.Thread | None = None
        self._monitor_thread: threading.Thread | None = None
        self._startup_deadline: float | None = None

    def state_value(self) -> str:
        with self._lock:
            return self._state.value

    def wait_idle(self, timeout: float = 10.0) -> bool:
        return self._idle.wait(timeout)

    def wait_for_state(self, state: str, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._state.value != state:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
        return True

    def log_path(self, name: str) -> Path | None:
        filename = self.LOG_FILES.get(name)
        if filename is None or self._session_dir is None:
            return None
        return self._session_dir / filename

    def preview_path(self) -> Path | None:
        if self._session_dir is None:
            return None
        return self._session_dir / "latest.jpg"

    def start(self) -> None:
        with self._lock:
            if self._state not in (State.STOPPED, State.FAILED):
                raise ConflictError("session is not stopped")
            self._state = State.STARTING
            self._last_error = None
            self._stop_event.clear()
            self._idle.clear()
        try:
            self._preflight()
        except Exception as exc:
            self._last_error = str(exc)
            self._cleanup_safe()
            self._set_state(State.FAILED)
            self._idle.set()
            if isinstance(exc, PreflightError):
                raise
            raise PreflightError(str(exc)) from exc
        self._thread = threading.Thread(target=self._start_worker, name="clank-start", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if self._state == State.STOPPED:
                return
            if self._state == State.FAILED:
                self.runner.remove_pointer()
                self._pointer = {}
                self._set_state(State.STOPPED)
                return
            if self._state == State.STOPPING:
                return
            self._stop_event.set()
            if self._state == State.STARTING:
                self._set_state(State.STOPPING)
                return
            self._idle.clear()
            self._set_state(State.STOPPING)
            self._thread = threading.Thread(target=self._stop_worker, name="clank-stop", daemon=True)
        self._thread.start()

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        with self._lifecycle_lock:
            with self._lock:
                if self._state != State.RUNNING:
                    raise ConflictError("session is not running")
            self._stop_handle("pose", 2)
            try:
                self._start_pose(mode)
            except Exception as exc:
                self._last_error = str(exc)
                self._cleanup_safe()
                self._set_state(State.FAILED)
                raise
            with self._lock:
                self._mode = mode

    def kill(self, target: str) -> None:
        with self._lock:
            if self._state != State.RUNNING:
                raise ConflictError("session is not running")
        if target == "game":
            for pid in self.runner.pids_by_argv0(self._game_argv0()):
                self.runner.kill_pid(pid)
            self._game_pids = []
            return
        if target == "steam":
            try:
                self.runner.run(
                    [self.cfg.paths.steam_bin, "-shutdown"],
                    env=self._steam_env(),
                    cwd=Path.home(),
                    timeout=20,
                )
            except Exception:
                pass
            handle = self._handles.get("steam")
            if handle is not None:
                handle.stop(5)
            for pid in self.runner.pids_by_argv0(self._game_argv0()):
                self.runner.kill_pid(pid)
            self._game_pids = []
            return
        raise ValueError(f"unknown target {target!r}")

    def snapshot(self) -> dict:
        with self._lock:
            state = self._state.value
            session_id = self._session_id
            started_at = self._started_at
            mode = self._mode
            last_error = self._last_error
            handles = dict(self._handles)
            display = self._display
            preview = self._preview
            session_dir = self._session_dir
        game_pids = self.runner.pids_by_argv0(self._game_argv0()) if state in ("running", "stopping") else []
        processes: dict[str, dict] = {}
        for name in ("monado", "weston", "steam", "pose"):
            handle = handles.get(name)
            processes[name] = {
                "alive": bool(handle and handle.alive()),
                "pid": handle.pid if handle is not None else None,
            }
            if name == "weston":
                processes[name]["display"] = display
        processes["game"] = {
            "alive": bool(game_pids),
            "pid": game_pids[0] if game_pids else None,
            "exe": Path(self.cfg.paths.game_exe).name,
        }
        openxr_client = None
        monado_log = self.log_path("monado")
        if monado_log is not None:
            try:
                openxr_client = parse_openxr_client(monado_log.read_text(errors="replace"))
            except OSError:
                openxr_client = None
        if preview is not None:
            preview_status = preview.status()
        else:
            preview_status = {
                "width": self.cfg.preview.width,
                "height": self.cfg.preview.height - self.cfg.preview.panel_height,
                "fps_target": self.cfg.preview.fps,
                "fps_actual": 0,
                "last_frame_at": None,
                "errors": 0,
                "frames": 0,
            }
        logs = {
            name: str(self.log_path(name)) if self.log_path(name) is not None else None
            for name in ("session", "monado", "weston", "steam", "pose")
        }
        player_log = (
            Path.home()
            / ".local/share/Steam/steamapps/compatdata"
            / self.cfg.launch.proton_app_id
            / "pfx/drive_c/users/steamuser/AppData/LocalLow/SUPERHOT_Team/SUPERHOT_VR/Player.log"
        )
        return {
            "state": state,
            "session_id": session_id,
            "started_at": started_at.isoformat().replace("+00:00", "Z") if started_at else None,
            "mode": mode,
            "processes": processes,
            "openxr_client": openxr_client,
            "preview": preview_status,
            "last_error": last_error,
            "logs": logs,
            "session_dir": str(session_dir) if session_dir is not None else None,
            "player_log": str(player_log),
        }

    def _set_state(self, state: State) -> None:
        with self._lock:
            self._state = state
            self._condition.notify_all()

    def _abort_if_stopped(self) -> None:
        if self._stop_event.is_set():
            raise Aborted("session stop requested")
        if self._startup_deadline is not None and time.monotonic() > self._startup_deadline:
            raise StartupError("startup timeout exceeded")

    def _sleep_abort(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, seconds)
        while True:
            self._abort_if_stopped()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.25, remaining))

    def _base_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["XDG_RUNTIME_DIR"] = str(self.runtime_dir)
        return env

    def _steam_env(self) -> dict[str, str]:
        env = self._base_env()
        if self._display:
            env["DISPLAY"] = self._display
        if self._socket_name:
            env["WAYLAND_DISPLAY"] = self._socket_name
        return env

    def _log(self, name: str) -> Path:
        path = self.log_path(name)
        if path is None:
            raise StartupError("no active session directory")
        return path

    def _session_log(self, message: str) -> None:
        if self._session_dir is None:
            return
        path = self._session_dir / "session.log"
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with path.open("a") as handle:
            handle.write(f"{timestamp} {message}\n")

    def _game_argv0(self) -> str:
        linux_path = Path(self.cfg.paths.game_exe).as_posix()
        return ("Z:" + linux_path).replace("/", "\\")

    def _matches_command(self, actual: str, expected: str) -> bool:
        if not expected:
            return True
        return (
            actual == expected
            or os.path.basename(actual) == os.path.basename(expected)
            or actual.endswith(expected)
        )

    def _preflight(self) -> None:
        pointer = self.runner.read_pointer()
        stale: list[str] = []
        if pointer:
            for name, record in pointer.get("processes", {}).items():
                pid = int(record.get("pid") or 0)
                if pid and self.runner.pid_alive(pid):
                    args = self.runner.pid_cmdline(pid)
                    if args and self._matches_command(args[0], str(record.get("command", ""))):
                        stale.append(f"{name} pid {pid}")
            if stale:
                raise PreflightError(
                    "stale session processes: " + ", ".join(stale) + "; stop them before starting"
                )
            self.runner.remove_pointer()

        if self.runner.process_name_running("steam"):
            raise PreflightError("Close the host Steam client before starting a hidden C0 session.")
        for unit in ("monado.service", "monado.socket"):
            if self.runner.systemd_unit_active(unit):
                raise PreflightError(f"Stop the user unit {unit} before starting a hidden C0 session.")
        if self.runner.port_in_use("127.0.0.1", 4242):
            raise PreflightError("TCP 4242 is already in use")
        self.runner.remove_stale_socket(self.runtime_dir / "monado_comp_ipc")

        missing = missing_requirements(self.cfg)
        if missing:
            raise PreflightError("missing requirements: " + ", ".join(missing))

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._session_id = f"{stamp}-{os.urandom(3).hex()}"
        self._session_dir = Path(self.cfg.paths.session_root) / self._session_id
        self._session_dir.mkdir(parents=True, exist_ok=False)
        self._pointer = {"session": str(self._session_dir), "processes": {}}
        self.runner.write_pointer(self._pointer)

    def _start(self, name: str, argv, env, cwd: Path, log_path: Path, stderr_path: Path | None = None):
        handle = self.runner.start(name, argv, env, cwd, log_path, stderr_path)
        with self._lock:
            self._handles[name] = handle
        self._pointer.setdefault("processes", {})[name] = {
            "pid": handle.pid,
            "pgid": handle.pgid,
            "command": str(argv[0]),
        }
        self.runner.write_pointer(self._pointer)
        return handle

    def _start_monado(self) -> None:
        env = self._base_env()
        env.update(
            {
                "XRT_COMPOSITOR_NULL": "1",
                "P_OVERRIDE_ACTIVE_CONFIG": "remote",
                "IPC_EXIT_ON_DISCONNECT": "off",
                "XRT_NO_STDIN": "1",
            }
        )
        self._start("monado", [self.cfg.paths.monado_bin], env, Path.home(), self._log("monado"))
        self.runner.wait_port(
            "127.0.0.1",
            4242,
            self.cfg.timeouts.monado_startup_sec,
            self._stop_event,
        )

    def _start_weston(self) -> None:
        self._socket_name = f"clank-{self._session_id.split('-')[-1]}"
        weston_log = self._log("weston")
        argv = [
            self.cfg.paths.weston_bin,
            "--backend=headless",
            "--renderer=gl",
            f"--width={self.cfg.preview.width}",
            f"--height={self.cfg.preview.height}",
            f"--socket={self._socket_name}",
            "--no-config",
            "--xwayland",
            "--debug",
            f"--log={weston_log}",
        ]
        handle = self._start("weston", argv, self._base_env(), Path.home(), self._log("weston-stdout"))
        self._wait_weston_display(handle, weston_log)

    def _wait_weston_display(self, handle: ProcessHandle, log_path: Path) -> None:
        deadline = time.monotonic() + self.cfg.timeouts.weston_startup_sec
        pattern = re.compile(r"xserver listening on display (:\d+)")
        while time.monotonic() < deadline:
            self._abort_if_stopped()
            if not handle.alive():
                raise StartupError("Weston exited during startup")
            try:
                text = log_path.read_text(errors="replace")
            except OSError:
                text = ""
            match = pattern.search(text)
            if match:
                self._display = match.group(1)
                return
            time.sleep(0.2)
        raise StartupError("Weston did not publish an Xwayland display")

    def _start_preview(self) -> None:
        self._preview = self.preview_factory(
            width=self.cfg.preview.width,
            height=self.cfg.preview.height,
            panel_height=self.cfg.preview.panel_height,
            fps=self.cfg.preview.fps,
            jpeg_quality=self.cfg.preview.jpeg_quality,
            screenshooter=self.cfg.paths.weston_screenshooter_bin,
            magick=self.cfg.paths.magick_bin,
            runtime_dir=self.runtime_dir,
            wayland_display=self._socket_name,
            work_dir=self._session_dir,
            publish_path=self._session_dir / "latest.jpg",
            stop_event=self._stop_event,
            log=self._session_log,
        )
        self._preview.start()

    def _start_pose(self, mode: str) -> None:
        pose_log = self._log("pose")
        pose_log.unlink(missing_ok=True)
        argv = [
            sys.executable,
            self.cfg.paths.pose_client,
            "--mode",
            mode,
            "--duration",
            "3600",
            "--log",
            str(self._session_dir / f"c0_{mode}.csv"),
        ]
        self._start("pose", argv, self._base_env(), Path.cwd(), pose_log)
        self.runner.wait_log(pose_log, "handshake ok", self.cfg.timeouts.pose_handshake_sec, self._stop_event)

    def _start_steam(self) -> None:
        argv = [self.cfg.paths.steam_bin, *self.cfg.launch.steam_extra_args]
        self._start("steam", argv, self._steam_env(), Path.home(), self._log("steam"))
        self._wait_steam_ready()

    def _wait_steam_ready(self) -> None:
        handle = self._handles["steam"]
        started = time.monotonic()
        deadline = started + self.cfg.launch.steam_ready_timeout_sec
        failures = 0
        while time.monotonic() < deadline:
            self._abort_if_stopped()
            if not handle.alive():
                raise StartupError("Steam exited during startup")
            graph = self.runner.debug_scene_graph(self.cfg.paths.weston_debug_bin, self._steam_env())
            if graph is None:
                failures += 1
                if failures >= 3:
                    break
            else:
                failures = 0
                if "steam" in graph.lower():
                    return
            self._sleep_abort(2)
        self._session_log("Steam window not found; continuing after settle period")
        remaining = self.cfg.launch.steam_settle_sec - (time.monotonic() - started)
        if remaining > 0:
            self._sleep_abort(remaining)
        if not handle.alive():
            raise StartupError("Steam exited during startup")

    def _launch_game(self) -> None:
        result = self.runner.run(
            [self.cfg.paths.steam_bin, f"steam://rungameid/{self.cfg.launch.shortcut_id}"],
            env=self._steam_env(),
            cwd=Path.home(),
            timeout=15,
        )
        if result.returncode != 0:
            raise StartupError(f"steam rungameid failed: {result.stderr.strip()}")
        pids = self.runner.wait_pids(
            self._game_argv0(),
            self.cfg.launch.game_process_timeout_sec,
            self._stop_event,
        )
        self._game_pids = pids
        self._pointer.setdefault("processes", {})["game"] = {
            "pid": pids[0],
            "command": self._game_argv0(),
        }
        self.runner.write_pointer(self._pointer)
        self.runner.wait_log(
            self._log("monado"),
            "application_name: 'Unity Application'",
            self.cfg.launch.openxr_client_timeout_sec,
            self._stop_event,
        )

    def _startup(self) -> None:
        self._abort_if_stopped()
        self._start_monado()
        self._abort_if_stopped()
        self._start_weston()
        self._abort_if_stopped()
        self._start_preview()
        self._abort_if_stopped()
        self._start_pose(self._mode)
        self._abort_if_stopped()
        self._start_steam()
        self._abort_if_stopped()
        self._launch_game()
        self._abort_if_stopped()

    def _start_worker(self) -> None:
        with self._lifecycle_lock:
            self._startup_deadline = time.monotonic() + self.cfg.launch.startup_timeout_sec
            try:
                self._startup()
            except Aborted:
                self._cleanup_safe()
                self._set_state(State.STOPPED)
            except Exception as exc:
                self._last_error = str(exc)
                self._session_log(f"startup failed: {exc}")
                self._cleanup_safe()
                self._set_state(State.STOPPED if self._stop_event.is_set() else State.FAILED)
            else:
                with self._lock:
                    self._started_at = datetime.now(timezone.utc)
                self._set_state(State.RUNNING)
                self._monitor_thread = threading.Thread(
                    target=self._monitor_worker, name="clank-monitor", daemon=True
                )
                self._monitor_thread.start()
            finally:
                self._startup_deadline = None
                self._idle.set()

    def _monitor_worker(self) -> None:
        while not self._stop_event.wait(1.0):
            with self._lock:
                if self._state != State.RUNNING:
                    return
                monado = self._handles.get("monado")
                weston = self._handles.get("weston")
            dead = None
            if monado is None or not monado.alive():
                dead = "monado"
            elif weston is None or not weston.alive():
                dead = "weston"
            if dead is None:
                continue
            with self._lock:
                if self._state != State.RUNNING:
                    return
                self._state = State.STOPPING
                self._last_error = f"{dead} exited while running"
                self._idle.clear()
            with self._lifecycle_lock:
                self._cleanup_safe()
            self._set_state(State.FAILED)
            self._idle.set()
            return

    def _stop_worker(self) -> None:
        with self._lifecycle_lock:
            try:
                self._cleanup_safe()
            finally:
                self._set_state(State.STOPPED)
                self._idle.set()

    def _cleanup_safe(self) -> None:
        try:
            self._cleanup()
        except Exception as exc:
            self._last_error = f"cleanup failed: {exc}"

    def _stop_handle(self, name: str, timeout: float):
        handle = self._handles.pop(name, None)
        if handle is not None:
            handle.stop(timeout)
        return handle

    def _cleanup(self) -> None:
        deadline = time.monotonic() + self.cfg.timeouts.stop_sec

        def budget(cap: float) -> float:
            return max(0.1, min(cap, deadline - time.monotonic()))

        self._stop_handle("pose", budget(2))
        if "steam" in self._handles:
            try:
                self.runner.run(
                    [self.cfg.paths.steam_bin, "-shutdown"],
                    env=self._steam_env(),
                    cwd=Path.home(),
                    timeout=budget(20.0),
                )
            except Exception:
                pass
        self._stop_handle("steam", budget(5))
        for pid in list(self._game_pids):
            self.runner.kill_pid(pid)
        self._game_pids = []
        for pid in self.runner.pids_by_argv0(self._game_argv0()):
            self.runner.kill_pid(pid)
        self._stop_handle("weston", budget(5))
        monado = self._stop_handle("monado", budget(5))
        if monado is None or not monado.alive():
            self.runner.remove_stale_socket(self.runtime_dir / "monado_comp_ipc")
        if self._preview is not None:
            try:
                self._preview.stop(5)
            except Exception:
                pass
            self._preview = None
        self.runner.remove_pointer()
        self._pointer = {}
        self._handles.clear()
        self._display = None
        self._session_log("cleanup complete")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_orchestrator -v`

Expected: 11 tests, `OK`.

- [ ] **Step 5: Commit**

```bash
git add clank/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: add C0 session manager"
```

---

### Task 5: Dashboard HTTP API

**Files:**
- Create: `clank/dashboard.py`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing dashboard tests**

Create `tests/test_dashboard.py`:

```python
from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from clank.config import load_config
from clank.dashboard import DashboardServer
from clank.orchestrator import ConflictError, MODES, PreflightError


class FakeManager:
    def __init__(self, base: Path) -> None:
        self.base = base
        self.state = "stopped"
        self.session_id = "20261025T223900Z-a1b2c3"
        self.started_at = None
        self.mode = "hold"
        self.last_error = None
        self.fail_start = False
        self.start_calls = 0
        self.stop_calls = 0
        self.kill_calls: list[str] = []
        self.logs = {name: base / f"{name}.log" for name in ("session", "monado", "weston", "steam", "pose")}
        for path in self.logs.values():
            path.write_text("")
        self.latest = base / "latest.jpg"

    def start(self) -> None:
        self.start_calls += 1
        if self.fail_start and self.start_calls == 1:
            raise PreflightError("Close the host Steam client before starting a hidden C0 session.")
        if self.state not in ("stopped", "failed"):
            raise ConflictError("session is not stopped")
        self.state = "starting"

    def stop(self) -> None:
        self.stop_calls += 1
        self.state = "stopping"

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        if self.state != "running":
            raise ConflictError("session is not running")
        self.mode = mode

    def kill(self, target: str) -> None:
        if target not in ("game", "steam"):
            raise ValueError(f"unknown target {target!r}")
        if self.state != "running":
            raise ConflictError("session is not running")
        self.kill_calls.append(target)

    def state_value(self) -> str:
        return self.state

    def log_path(self, name: str) -> Path | None:
        return self.logs.get(name)

    def preview_path(self) -> Path:
        return self.latest

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "mode": self.mode,
            "processes": {
                "monado": {"alive": True, "pid": 1},
                "weston": {"alive": True, "pid": 2, "display": ":2"},
                "steam": {"alive": True, "pid": 3},
                "game": {"alive": True, "pid": 4, "exe": "SUPERHOT_VR.exe"},
                "pose": {"alive": True, "pid": 5},
            },
            "openxr_client": "Unity Application",
            "preview": {
                "width": 1280,
                "height": 688,
                "fps_target": 8,
                "fps_actual": 7.4,
                "last_frame_at": "2026-10-25T22:39:00Z",
                "errors": 0,
                "frames": 10,
            },
            "last_error": self.last_error,
            "logs": {name: str(path) for name, path in self.logs.items()},
        }


class DashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        (self.base / "index.html").write_text("<h1>C0</h1>")
        (self.base / "style.css").write_text("body{}")
        (self.base / "app.js").write_text("console.log('x')")
        self.manager = FakeManager(self.base)
        self.cfg = load_config(None, env={})
        self.server = DashboardServer(("127.0.0.1", 0), self.manager, self.cfg, self.base)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.temp.cleanup()

    def request(self, method: str, path: str, body=None, raw: bytes | None = None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def test_state(self) -> None:
        status, body = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["openxr_client"], "Unity Application")
        self.assertEqual(payload["preview"]["height"], 688)

    def test_static_index(self) -> None:
        status, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"C0", body)

    def test_frame_404_then_200(self) -> None:
        status, _ = self.request("GET", "/api/frame.jpg")
        self.assertEqual(status, 404)
        self.manager.latest.write_bytes(b"JPEG")
        status, body = self.request("GET", "/api/frame.jpg")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"JPEG")

    def test_start_returns_202_then_conflict(self) -> None:
        status, _ = self.request("POST", "/api/session/start")
        self.assertEqual(status, 202)
        status, body = self.request("POST", "/api/session/start")
        self.assertEqual(status, 409)
        self.assertIn("error", json.loads(body))

    def test_preflight_failure_returns_400(self) -> None:
        self.manager.fail_start = True
        status, body = self.request("POST", "/api/session/start")
        self.assertEqual(status, 400)
        self.assertIn("host Steam", json.loads(body)["error"])

    def test_stop_returns_202(self) -> None:
        status, _ = self.request("POST", "/api/session/stop")
        self.assertEqual(status, 202)
        self.assertEqual(self.manager.stop_calls, 1)

    def test_mode_and_kill_validation(self) -> None:
        status, _ = self.request("POST", "/api/session/mode", {"mode": "yaw"})
        self.assertEqual(status, 409)
        self.manager.state = "running"
        status, _ = self.request("POST", "/api/session/mode", {"mode": "yaw"})
        self.assertEqual(status, 200)
        status, _ = self.request("POST", "/api/session/mode", {"mode": "nope"})
        self.assertEqual(status, 400)
        status, _ = self.request("POST", "/api/session/kill", {"target": "game"})
        self.assertEqual(status, 200)
        self.assertEqual(self.manager.kill_calls, ["game"])
        status, _ = self.request("POST", "/api/session/kill", {"target": "nope"})
        self.assertEqual(status, 400)

    def test_logs_allowlist(self) -> None:
        self.manager.logs["monado"].write_text("hello monado")
        status, body = self.request("GET", "/api/logs/monado")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"hello monado")
        status, _ = self.request("GET", "/api/logs/nope")
        self.assertEqual(status, 404)

    def test_malformed_json(self) -> None:
        status, _ = self.request("POST", "/api/session/mode", raw=b"{")
        self.assertEqual(status, 400)

    def test_unknown_route(self) -> None:
        status, _ = self.request("GET", "/nope")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_dashboard -v`

Expected: `ModuleNotFoundError` for `clank.dashboard`.

- [ ] **Step 3: Implement the dashboard**

Create `clank/dashboard.py`:

```python
from __future__ import annotations

import json
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .config import Config
from .orchestrator import ConflictError, PreflightError, SessionManager


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, manager, config: Config, static_dir: Path) -> None:
        self.manager = manager
        self.config = config
        self.static_dir = Path(static_dir)
        super().__init__(address, DashboardHandler)


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "clank/0.1"

    def log_message(self, format, *args) -> None:
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._static("index.html", "text/html; charset=utf-8")
        if path == "/style.css":
            return self._static("style.css", "text/css; charset=utf-8")
        if path == "/app.js":
            return self._static("app.js", "application/javascript; charset=utf-8")
        if path == "/api/state":
            return self._json(self.server.manager.snapshot())
        if path == "/api/frame.jpg":
            latest = self.server.manager.preview_path()
            if latest is None or not latest.exists():
                return self._error(404, "no preview frame")
            return self._bytes(200, latest.read_bytes(), "image/jpeg")
        if path == "/preview.mjpg":
            return self._mjpeg()
        if path.startswith("/api/logs/"):
            return self._log(path.rsplit("/", 1)[-1])
        return self._error(404, "not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/session/start":
            try:
                self.server.manager.start()
            except ConflictError as exc:
                return self._error(409, str(exc))
            except PreflightError as exc:
                return self._error(400, str(exc))
            except Exception as exc:
                return self._error(500, str(exc))
            return self._json({"accepted": True}, status=202)
        if path == "/api/session/stop":
            self.server.manager.stop()
            return self._json({"accepted": True}, status=202)
        try:
            body = self._read_body()
        except ValueError as exc:
            return self._error(400, str(exc))
        if path == "/api/session/mode":
            try:
                self.server.manager.set_mode(body.get("mode"))
            except ValueError as exc:
                return self._error(400, str(exc))
            except ConflictError as exc:
                return self._error(409, str(exc))
            return self._json({"mode": body.get("mode")})
        if path == "/api/session/kill":
            try:
                self.server.manager.kill(body.get("target"))
            except ValueError as exc:
                return self._error(400, str(exc))
            except ConflictError as exc:
                return self._error(409, str(exc))
            return self._json({"killed": body.get("target")})
        return self._error(404, "not found")

    def _read_body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length > 1_000_000:
            raise ValueError("request body too large")
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON object required")
        return data

    def _json(self, payload, status: int = 200) -> None:
        self._bytes(status, json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status=status)

    def _bytes(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _static(self, name: str, content_type: str) -> None:
        path = self.server.static_dir / name
        try:
            data = path.read_bytes()
        except OSError:
            return self._error(404, "not found")
        self._bytes(200, data, content_type)

    def _log(self, name: str) -> None:
        path = self.server.manager.log_path(name)
        if path is None:
            return self._error(404, "unknown log")
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            return self._error(404, "log not found")
        data = "\n".join(lines[-200:]).encode()
        self._bytes(200, data, "text/plain; charset=utf-8")

    def _mjpeg(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        boundary = b"--frame\r\n"
        interval = 1.0 / max(1.0, self.server.config.preview.fps)
        while True:
            if self.server.manager.state_value() in ("stopped", "failed"):
                break
            latest = self.server.manager.preview_path()
            data = None
            if latest is not None:
                try:
                    data = latest.read_bytes()
                except OSError:
                    data = None
            if data:
                try:
                    self.wfile.write(boundary)
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
            time.sleep(interval)


def run_dashboard(cfg: Config) -> None:
    manager = SessionManager(cfg)
    static_dir = Path(__file__).resolve().parent / "static"
    server = DashboardServer((cfg.dashboard.host, cfg.dashboard.port), manager, cfg, static_dir)

    def request_shutdown(signum, frame) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    host, port = server.server_address[:2]
    print(f"clank dashboard: http://{host}:{port}")
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        manager.stop()
        manager.wait_idle(70)
        server.server_close()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_dashboard -v`

Expected: 10 tests, `OK`.

- [ ] **Step 5: Commit**

```bash
git add clank/dashboard.py tests/test_dashboard.py
git commit -m "feat: add dashboard HTTP API"
```

---

### Task 6: Dashboard UI

**Files:**
- Create: `clank/static/index.html`
- Create: `clank/static/style.css`
- Create: `clank/static/app.js`

- [ ] **Step 1: Add a static-content assertion to the dashboard test**

Append this test to `tests/test_dashboard.py` inside `DashboardTests`:

```python
    def test_ui_assets_are_served(self) -> None:
        for path in ("/style.css", "/app.js"):
            status, _ = self.request("GET", path)
            self.assertEqual(status, 200)
```

- [ ] **Step 2: Create the UI files**

Create `clank/static/index.html`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Clank C0</title>
  <link rel="stylesheet" href="/style.css">
</head>
<body>
  <header>
    <h1>C0 session</h1>
    <span id="state" class="badge">stopped</span>
    <span id="session-id"></span>
    <span id="uptime"></span>
    <button id="start">Start</button>
    <button id="stop">Stop</button>
  </header>
  <main>
    <section class="preview">
      <img id="preview" src="/preview.mjpg" alt="live C0 preview">
      <div id="preview-wait">waiting for the first frame…</div>
    </section>

    <section>
      <h2>Pose mode</h2>
      <div id="modes">
        <button data-mode="hold">hold</button>
        <button data-mode="yaw">yaw</button>
        <button data-mode="trigger">trigger</button>
        <button data-mode="dodge">dodge</button>
        <button data-mode="walk">walk</button>
        <button data-mode="clock">clock</button>
      </div>
    </section>

    <section>
      <h2>Health</h2>
      <div id="health"></div>
    </section>

    <section class="danger">
      <h2>Danger zone</h2>
      <button id="kill-game">Kill game</button>
      <button id="kill-steam">Shutdown Steam</button>
    </section>

    <section>
      <h2>Logs</h2>
      <select id="log-name">
        <option value="session">session</option>
        <option value="monado">monado</option>
        <option value="weston">weston</option>
        <option value="steam">steam</option>
        <option value="pose">pose</option>
      </select>
      <button id="log-refresh">Refresh</button>
      <pre id="log-tail"></pre>
    </section>

    <p id="error" class="error"></p>
  </main>
  <script src="/app.js"></script>
</body>
</html>
```

Create `clank/static/style.css`:

```css
:root {
  color-scheme: dark;
  --bg: #111417;
  --panel: #1b2026;
  --text: #e8edf2;
  --muted: #8c98a6;
  --accent: #59b7ff;
  --ok: #51d88a;
  --bad: #ff6b6b;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 15px/1.45 system-ui, sans-serif;
}

header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 18px;
  border-bottom: 1px solid #2b333d;
  background: var(--panel);
  position: sticky;
  top: 0;
  z-index: 2;
}

h1 { font-size: 18px; margin: 0 8px 0 0; }
h2 { font-size: 14px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }

main {
  max-width: 1100px;
  margin: 0 auto;
  padding: 18px;
  display: grid;
  gap: 18px;
}

section {
  background: var(--panel);
  border: 1px solid #2b333d;
  border-radius: 10px;
  padding: 14px;
}

.preview { position: relative; padding: 0; overflow: hidden; }
.preview img { display: block; width: 100%; aspect-ratio: 1280 / 688; object-fit: contain; background: #000; }
#preview-wait {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  color: var(--muted);
  background: #000c;
}

button, select {
  background: #26303a;
  color: var(--text);
  border: 1px solid #3a4653;
  border-radius: 7px;
  padding: 7px 11px;
  font: inherit;
  cursor: pointer;
}

button:hover:not(:disabled) { border-color: var(--accent); }
button:disabled { opacity: 0.45; cursor: default; }
button.active { border-color: var(--accent); color: var(--accent); }

#modes { display: flex; flex-wrap: wrap; gap: 8px; }

.badge {
  padding: 2px 9px;
  border-radius: 999px;
  background: #303943;
  color: var(--muted);
  text-transform: capitalize;
}
.badge.running { background: #153b2a; color: var(--ok); }
.badge.starting, .badge.stopping { background: #3b3215; color: #ffd166; }
.badge.failed { background: #3b1b1b; color: var(--bad); }

#session-id, #uptime { color: var(--muted); font-family: ui-monospace, monospace; font-size: 12px; }

#health { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; }
.card { border: 1px solid #303943; border-radius: 8px; padding: 9px; display: grid; gap: 3px; }
.card strong { font-size: 12px; text-transform: uppercase; color: var(--muted); }
.card small { color: var(--muted); font-family: ui-monospace, monospace; }
.ok { color: var(--ok); }
.bad { color: var(--bad); }

.danger { border-color: #5a2a2a; }
.danger button { border-color: #7a3434; }

#log-tail {
  min-height: 180px;
  max-height: 360px;
  overflow: auto;
  background: #0c0f12;
  border: 1px solid #2b333d;
  border-radius: 8px;
  padding: 10px;
  white-space: pre-wrap;
  font-family: ui-monospace, monospace;
  font-size: 12px;
}

.error { display: none; color: var(--bad); border: 1px solid #5a2a2a; background: #2a1515; padding: 10px; border-radius: 8px; }
```

Create `clank/static/app.js`:

```js
const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[char]));
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let message = response.statusText;
    try {
      message = (await response.json()).error || message;
    } catch (_) {
      // keep the status text
    }
    throw new Error(message);
  }
  return response.json();
}

async function post(path, body) {
  return api(path, {
    method: "POST",
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

function showError(message) {
  $("error").textContent = message || "";
  $("error").style.display = message ? "block" : "none";
}

function uptime(startedAt) {
  if (!startedAt) return "";
  const seconds = Math.max(0, Math.floor((Date.now() - new Date(startedAt).getTime()) / 1000));
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

function render(state) {
  $("state").textContent = state.state;
  $("state").className = `badge ${state.state}`;
  $("session-id").textContent = state.session_id || "";
  $("uptime").textContent = uptime(state.started_at);
  $("start").disabled = !["stopped", "failed"].includes(state.state);
  $("stop").disabled = state.state === "stopped";
  $("preview-wait").style.display = state.preview.last_frame_at ? "none" : "block";
  showError(state.last_error);

  document.querySelectorAll("#modes button").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === state.mode);
    button.disabled = state.state !== "running";
  });

  $("kill-game").disabled = state.state !== "running";
  $("kill-steam").disabled = state.state !== "running";

  const cards = Object.entries(state.processes).map(([name, process]) => `
    <div class="card">
      <strong>${escapeHtml(name)}</strong>
      <span class="${process.alive ? "ok" : "bad"}">${process.alive ? "alive" : "down"}</span>
      <small>${escapeHtml(process.pid ?? "")}</small>
    </div>
  `);
  cards.push(`
    <div class="card">
      <strong>OpenXR</strong>
      <span>${escapeHtml(state.openxr_client || "none")}</span>
    </div>
  `);
  cards.push(`
    <div class="card">
      <strong>Preview</strong>
      <span>${escapeHtml(state.preview.fps_actual)} fps</span>
    </div>
  `);
  $("health").innerHTML = cards.join("");
}

async function refresh() {
  try {
    render(await api("/api/state"));
  } catch (error) {
    showError(error.message);
  }
}

$("start").onclick = async () => {
  try {
    await post("/api/session/start");
    await refresh();
  } catch (error) {
    showError(error.message);
  }
};

$("stop").onclick = async () => {
  try {
    await post("/api/session/stop");
    await refresh();
  } catch (error) {
    showError(error.message);
  }
};

document.querySelectorAll("#modes button").forEach((button) => {
  button.onclick = async () => {
    try {
      await post("/api/session/mode", { mode: button.dataset.mode });
      await refresh();
    } catch (error) {
      showError(error.message);
    }
  };
});

$("kill-game").onclick = async () => {
  if (!confirm("Kill the game process?")) return;
  try {
    await post("/api/session/kill", { target: "game" });
    await refresh();
  } catch (error) {
    showError(error.message);
  }
};

$("kill-steam").onclick = async () => {
  if (!confirm("Shut down the hidden Steam client?")) return;
  try {
    await post("/api/session/kill", { target: "steam" });
    await refresh();
  } catch (error) {
    showError(error.message);
  }
};

$("log-refresh").onclick = async () => {
  try {
    const name = $("log-name").value;
    const response = await fetch(`/api/logs/${name}`);
    $("log-tail").textContent = response.ok ? await response.text() : `failed: ${response.status}`;
  } catch (error) {
    $("log-tail").textContent = error.message;
  }
};

refresh();
setInterval(refresh, 1000);
```

- [ ] **Step 3: Run the dashboard tests**

Run: `python3 -m unittest tests.test_dashboard -v`

Expected: 11 tests, `OK`.

- [ ] **Step 4: Commit**

```bash
git add clank/static tests/test_dashboard.py
git commit -m "feat: add dashboard UI"
```

---

### Task 7: CLI and launcher

**Files:**
- Create: `clank/__main__.py`
- Create: `run.sh`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing CLI test**

Create `tests/test_cli.py`:

```python
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class CliTests(unittest.TestCase):
    def test_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "clank", "--help"],
            cwd=Path(__file__).resolve().parent.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--preview-smoke", result.stdout)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m unittest tests.test_cli -v`

Expected: failure because `clank.__main__` does not exist.

- [ ] **Step 3: Implement the CLI and launcher**

Create `clank/__main__.py`:

```python
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

from .config import ConfigError, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clank")
    parser.add_argument("--config", default="config.toml", help="TOML config path")
    parser.add_argument("--host", help="override dashboard host")
    parser.add_argument("--port", type=int, help="override dashboard port")
    parser.add_argument("--preview-smoke", action="store_true", help="run the Weston/mpv preview smoke test")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(Path(args.config), env=os.environ)
    except (ConfigError, OSError) as exc:
        print(f"clank: {exc}", file=sys.stderr)
        return 2

    if args.host is not None or args.port is not None:
        cfg = replace(
            cfg,
            dashboard=replace(
                cfg.dashboard,
                host=args.host if args.host is not None else cfg.dashboard.host,
                port=args.port if args.port is not None else cfg.dashboard.port,
            ),
        )

    if args.preview_smoke:
        from .preview import preview_smoke

        return preview_smoke(cfg)

    from .dashboard import run_dashboard

    run_dashboard(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `run.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if command -v weston >/dev/null 2>&1 \
  && command -v weston-screenshooter >/dev/null 2>&1 \
  && command -v weston-debug >/dev/null 2>&1; then
  exec python3 -m clank "$@"
fi

exec nix shell nixpkgs#weston -c python3 -m clank "$@"
```

Make it executable:

```bash
chmod +x run.sh
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_cli -v`

Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add run.sh clank/__main__.py tests/test_cli.py
git commit -m "feat: add clank CLI and launcher"
```

---

### Task 8: Full verification

**Files:**
- No new files.

- [ ] **Step 1: Run the complete unit suite**

Run: `python3 -m unittest discover -s tests -v`

Expected: all tests pass, `OK`.

- [ ] **Step 2: Review the implementation against the spec**

Check each item against `docs/archive/specs/2026-10-25-c0-hidden-session-dashboard-design.md`:

- preflight checks host Steam, Monado units, stale pointer, TCP 4242, stale socket, required binaries;
- startup order is Monado, Weston, preview, pose, Steam, game;
- the running monitor transitions to `failed` and cleans up if Monado or Weston exits;
- startup and shutdown obey their outer time bounds;
- Steam readiness uses `weston-debug scene-graph` with a settle fallback;
- game launch uses `steam://rungameid/18251483899710406656`;
- preview crops 32 px, publishes atomically, targets 8 fps;
- shutdown stops pose, Steam/game, Weston, Monado, preview, and removes the pointer;
- API routes and status codes match the spec;
- dashboard is bound to `127.0.0.1`.

Fix any mismatch before proceeding.

- [ ] **Step 3: Run the preview smoke test**

Run: `python3 -m clank --preview-smoke`

Expected: `preview smoke ok`, or a `skipped` message if Weston/mpv/magick is unavailable. This must not launch Monado, Steam, or the game.

- [ ] **Step 4: Manual acceptance with the real session**

1. Close the host Steam client.
2. Run `./run.sh` and open the printed URL.
3. Click Start. Confirm no Steam or Superhot window appears in Hyprland.
4. Confirm the dashboard reaches `running`, the game card is alive, and OpenXR shows `Unity Application`.
5. Confirm the preview shows the game (not the Weston panel).
6. Select `yaw`, confirm the view rotates; select `hold`, confirm it stops.
7. Select each other mode once and confirm the pose client restarts without killing the game.
8. Click Stop. Confirm `pgrep -af 'SUPERHOT_VR.exe|monado-service|weston'` shows no session processes.
9. Start again after Stop and confirm it reaches `running` without manual cleanup.

- [ ] **Step 5: Commit any verification fixes**

```bash
git status --short
git add config.toml run.sh clank tests
git commit -m "fix: address verification findings"
```

If there are no changes, skip the commit.

---

## Notes for the Implementer

- Run all commands from the repository root.
- `weston`, `weston-screenshooter`, and `weston-debug` are not on the default NixOS `PATH`; use `run.sh` or `nix shell nixpkgs#weston`.
- Never start Steam with a working directory under `/tmp`; use `$HOME`.
- Do not use `--shell=kiosk`; it captured black frames in the spike.
- The Steam shortcut ID is `18251483899710406656`; `4249504744` is the Proton AppId, not a launchable store AppID.
- Keep the dashboard on `127.0.0.1`; do not add authentication or LAN exposure.
- Do not commit `spike/c0/c0_manual.py` from the earlier spike unless the user asks.
