"""One attempt at the job, repeatable while the stack stays up.

A **run** is what a person repeats: a fresh directory, a fresh diary, the game's saved
job wiped and the game relaunched, the body reset, and then the brain loop. The
**stack** — monado-service, Weston, websockify, the pose relay, Steam — is what costs a
cold boot. A run **boots it when it is down and never touches one that is up**: that is
why Start run is one press, and why Stop run leaves a stack standing.

The split is deliberate. `SessionManager` owns the stack and can only restart the game
(`restart_game`), so booting it is a request this module makes (`_stack`) and waits on,
not something it does itself. This module owns everything that accumulates across
attempts. A run's failure leaves the stack running, because the alternative — one bad
attempt costing a boot — is the thing this replaces.

Design: `docs/superpowers/specs/2026-11-12-runs-design.md`.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from . import brain, brainlog, ears
from .body_preview import send_control
from .config import Config
from .depth_stream import Stream
from .orchestrator import Aborted, ConflictError
from .recording import Recorder, read_manifest, run_composite

#: `<HHMMSS>-<6 hex>`, the directory name and the id the dashboard shows.
RUN_ID_RE = re.compile(r"^\d{6}-[0-9a-f]{6}$")

#: What every phase is waiting for, in words. The topbar shows the phase's phrase
#: while the spinner turns; nothing parses these.
PHASE_DETAILS: dict[str, str] = {
    "checking": "checking the model server…",
    "stack": "starting the stack…",
    "preparing": "preparing the run…",
    "game": "restarting the game…",
    "body": "resetting the body…",
    "brain": "starting the brain loop…",
    "running": "listening…",
    "stopping": "stopping the run…",
}


class RunState(StrEnum):
    IDLE = "idle"
    CHECKING = "checking"
    STACK = "stack"
    PREPARING = "preparing"
    GAME = "game"
    BODY = "body"
    BRAIN = "brain"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


class RunError(RuntimeError):
    """A run cannot start at all (bad interpreter, bad script) — not a phase failure."""


class _Cancelled(Exception):
    """The operator pressed Stop; the worker unwinds without calling it a failure."""


def _now(monotonic: float | None = None) -> str:
    moment = datetime.now(UTC) if monotonic is None else datetime.fromtimestamp(monotonic, UTC)
    return moment.isoformat().replace("+00:00", "Z")


def _short_model(status: brain.ServerStatus) -> str:
    return status.model or "nothing"


class RunManager:
    """A run state machine over a stack it boots when it has to.

    Collaborators are injected so the machine is testable without a game:
    `manager` needs `snapshot()`, `restart_game()` and `set_mode()`; `body` is a
    `BodyPreview` or None; `overlay` needs `clear()`; `probe` returns a
    `brain.ServerStatus`; `spawn(**kwargs)` starts the brain loop; `recorder` is a
    `Recorder` (None builds the real one when the run reaches it); `composite` is
    the post-run encode, detached into its own thread.
    """

    def __init__(
        self,
        manager: Any,
        cfg: Config,
        body: Any = None,
        overlay: Any = None,
        *,
        probe: Callable[[], brain.ServerStatus] | None = None,
        spawn: Callable[..., Any] | None = None,
        recorder: Any = None,
        composite: Callable[[Path, Config], Any] | None = None,
    ) -> None:
        self._manager = manager
        self._cfg = cfg
        self._body = body
        self._overlay = overlay
        self._probe = probe or (lambda: brain.server_status())
        self._spawn = spawn if spawn is not None else spawn_child
        self._recorder = recorder
        self._composite = composite if composite is not None else run_composite
        self._recording = False
        #: The run's one audio recorder. It is opened before the game is relaunched, because
        #: that relaunch is what makes the game talk.
        self._ring: ears.Ring | None = None

        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._generation = 0

        self._state = RunState.IDLE
        self._phase = RunState.IDLE
        self._detail = ""
        self._id: str | None = None
        self._dir: Path | None = None
        self._session_id: str | None = None
        self._last_diary: Path | None = None
        self._started_at: str | None = None
        self._started_monotonic: float | None = None
        self._ended_at: str | None = None
        self._error: str | None = None
        self._exit_code: int | None = None
        self._child: Any = None

    # ---- queries -------------------------------------------------------

    def preflight(self) -> str:
        """Empty when the brain loop can start, else why it cannot.

        Both paths are relative to the dashboard's cwd (the repo root), like the
        body sidecar's own preflight: `BodyPreview` resolves its venv the same way.
        """
        for name, value in (
            ("brain_python", self._cfg.run.brain_python),
            ("brain_script", self._cfg.run.brain_script),
        ):
            if not Path(value).exists():
                return f"run.{name} not found: {value}"
        # A body that cannot start is a run that cannot act, and its own preflight
        # is free: checking it here refuses before a game restart rather than after.
        body = self._body
        check = getattr(body, "preflight", None)
        if check is not None:
            problem = check()
            if problem:
                return f"body view cannot start: {problem}"
        return ""

    def snapshot(self) -> dict:
        with self._lock:
            state = self._state.value
            phase = self._phase.value
            busy = self._state not in (RunState.IDLE, RunState.FAILED)
            detail = self._detail if busy else ""
            if busy and not detail:
                detail = PHASE_DETAILS.get(phase, "")
            ended = self._ended_at
            duration = self._duration()
            run_dir = self._dir
            snapshot = {
                "state": state,
                "phase": phase,
                "detail": detail,
                "id": self._id,
                "dir": str(run_dir) if run_dir is not None else None,
                "session_id": self._session_id,
                "started_at": self._started_at,
                "ended_at": ended,
                "duration_s": duration,
                "error": self._error,
                "exit_code": self._exit_code,
            }
        # Read outside the lock: the dashboard polls this, and a manifest read is
        # a file read. None means no recording, which is also every run before Task 5.
        snapshot["recording"] = read_manifest(run_dir) if run_dir is not None else None
        return snapshot

    def current_diary(self) -> Path:
        """The diary the panel should read: this run, then the newest on disk.

        A restarted dashboard has no memory, so the newest `runs/*/brain.jsonl`
        under the session root is the fallback: the panel opens on the last run
        instead of on the global file the runs no longer write.
        """
        with self._lock:
            last = self._last_diary
        if last is not None:
            return last
        newest = self._newest_diary()
        return newest if newest is not None else brainlog.default_path()

    def _newest_diary(self) -> Path | None:
        best: tuple[float, Path] | None = None
        try:
            candidates = Path(self._cfg.paths.session_root).glob("*/runs/*/brain.jsonl")
            for path in candidates:
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                if best is None or mtime > best[0]:
                    best = (mtime, path)
        except OSError:
            return None
        return best[1] if best is not None else None

    def brain_argv(self) -> list[str]:
        return [
            self._cfg.run.brain_python,
            self._cfg.run.brain_script,
            "--for",
            f"{self._cfg.run.brain_duration_sec:g}",
        ]

    def brain_env(self, run_dir: Path) -> dict[str, str]:
        """The child's environment: the run's own paths, and none of the parent's.

        `PYTHONPATH`/`VIRTUAL_ENV` must not survive into the venv interpreter for the
        same reason the body sidecar drops them: the dashboard can be started through
        `nix develop`, whose site-packages belongs to a different Python.
        """
        env = os.environ.copy()
        for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            env.pop(key, None)
        env["CLANK_BRAIN_LOG"] = str(run_dir / "brain.jsonl")
        env["CLANK_AUTO_STATE"] = str(run_dir / "frames")
        # Not a CLANK_ name: `clank.config` rejects unknown CLANK_* as a bad override, and this
        # is not configuration, it is where this run put its recorder.
        env["AUDIO_RING_PATH"] = str(run_dir / "audio-ring.raw")
        return env

    # ---- lifecycle -----------------------------------------------------

    def start(self) -> None:
        """Take the run on, from whatever the stack is doing.

        Nothing here waits: the stack boot, the gate and every phase run on the
        worker thread, so the dashboard's HTTP call returns while the spinner turns.
        A run starts from a stopped stack now, which is why `start()` no longer
        refuses one -- it boots it.
        """
        problem = self.preflight()
        if problem:
            raise RunError(problem)
        with self._lock:
            if self._state not in (RunState.IDLE, RunState.FAILED):
                raise ConflictError("a run is already active")
            self._generation += 1
            generation = self._generation
            self._cancel.clear()
            self._idle.clear()
            self._id = None
            self._dir = None
            self._session_id = None
            self._state = RunState.CHECKING
            self._phase = RunState.CHECKING
            self._detail = PHASE_DETAILS["checking"]
            self._started_at = _now()
            self._started_monotonic = time.monotonic()
            self._ended_at = None
            self._ended_monotonic = None
            self._error = None
            self._exit_code = None
            self._child = None
            self._thread = threading.Thread(
                target=self._worker,
                args=(generation,),
                name="clank-run",
                daemon=True,
            )
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if self._state in (RunState.IDLE, RunState.FAILED):
                return
            self._cancel.set()
            if self._state != RunState.STOPPING:
                self._state = RunState.STOPPING
                self._phase = RunState.STOPPING
                self._detail = PHASE_DETAILS["stopping"]
            child = self._child
        if child is not None:
            child.stop(self._cfg.run.stop_timeout_sec)

    def wait_idle(self, timeout: float = 10.0) -> bool:
        return self._idle.wait(timeout)

    # ---- the worker ----------------------------------------------------

    def _worker(self, generation: int) -> None:
        try:
            # A session that already exists names the directory now, so a refusal
            # at the gate still leaves a `run.json` of its own.
            self._claim_dir()
            self._gate(generation)
            run_dir = self._stack(generation)
            self._set_phase(RunState.PREPARING)
            self._open_run(run_dir)
            # Before the game: the relaunch is what makes it talk, so a recorder opened after it
            # is a recorder opened too late. MEASURED 2026-11-14, run 025412-cf7309 - the brain
            # loop's own ring began at 02:54:28 and the greeting's first words, "Hello human!
            # Welcome to the Job Simulator.", were not in the file at all.
            self._open_ring(run_dir)
            self._check_cancel(generation)

            self._set_phase(RunState.GAME)
            self._manager.restart_game()
            self._check_cancel(generation)

            self._set_phase(RunState.BODY)
            self._reset_body(generation)
            self._check_cancel(generation)

            self._set_phase(RunState.BRAIN)
            child = self._start_brain(run_dir)
            with self._lock:
                self._child = child
            self._set_phase(RunState.RUNNING)
            self._start_recording(run_dir)
            self._monitor(generation, child, run_dir)
        except (_Cancelled, Aborted):
            self._finish(RunState.IDLE, "")
        except Exception as exc:  # noqa: BLE001 - every phase failure is the run's
            self._finish(RunState.FAILED, str(exc))
        finally:
            with self._lock:
                self._child = None
            self._idle.set()

    def _check_cancelled(self, generation: int) -> None:
        """The operator's Stop, and a newer run's generation -- nothing else.

        A stack that is still starting is not a reason to stop: `_stack` is the
        phase that makes it run, so it cannot demand one that already does.
        """
        if generation != self._generation or self._cancel.is_set():
            raise _Cancelled()

    def _check_cancel(self, generation: int) -> None:
        self._check_cancelled(generation)
        # The stack can be shut down from outside this object (curl, the
        # orchestrator's own shutdown). A run that kept walking its phases would
        # spawn a brain loop against a game that is already gone.
        if self._manager.snapshot().get("state") != "running":
            raise _Cancelled()

    def _claim_dir(self) -> Path | None:
        """Name the run's directory as soon as a session exists to name it after.

        Free while the stack is down -- there is no session to put a run under yet --
        and called again after a boot, because that is when one appears. Claiming
        twice is the same as claiming once: the id is minted only the first time.
        """
        with self._lock:
            if self._dir is not None:
                return self._dir
        info = self._manager.snapshot()
        session_id = info.get("session_id")
        if info.get("state") != "running" or not session_id:
            return None
        run_id = f"{datetime.now().strftime('%H%M%S')}-{os.urandom(3).hex()}"
        run_dir = Path(self._cfg.paths.session_root) / str(session_id) / "runs" / run_id
        with self._lock:
            if self._dir is not None:      # another caller claimed it first
                return self._dir
            self._id = run_id
            self._dir = run_dir
            self._session_id = str(session_id)
        self._ensure_run_dir(run_dir)
        return run_dir

    def _stack(self, generation: int) -> Path:
        """Boot the stack when it is down, and never touch one that is up.

        This is the phase that makes Start run one press. It returns the run's
        directory, because the directory is named after the session and a booted
        stack is the first thing to have one.

        A cancelled boot shuts nothing down: the operator asked for a run, not for a
        machine, and leaving a half-started stack for the next press is cheaper than
        tearing down a healthy one. A boot that fails is the orchestrator's own
        error, carried into `run.json` -- the stack is left as it is, because its
        log is the evidence.
        """
        info = self._manager.snapshot()
        if info.get("state") != "running":
            self._set_phase(RunState.STACK)
            # A stack that is already starting (or stopping) is not an error to
            # hand back: `start()` refuses those states, and the wait below is the
            # answer for both -- one becomes `running`, the other `stopped`.
            with contextlib.suppress(ConflictError):
                self._manager.start()
            deadline = time.monotonic() + self._cfg.run.session_start_sec
            while True:
                self._check_cancelled(generation)
                info = self._manager.snapshot()
                state = info.get("state")
                if state == "running" and info.get("session_id"):
                    break
                if state in ("failed", "stopped"):
                    reason = info.get("last_error") or state
                    raise RunError(f"the stack did not start: {reason}")
                if time.monotonic() > deadline:
                    raise RunError(
                        "the stack did not come up within "
                        f"{self._cfg.run.session_start_sec:g} s"
                    )
                self._cancel.wait(0.2)
        run_dir = self._claim_dir()
        if run_dir is None:
            raise RunError("the stack is running without a session id")
        return run_dir

    def _set_phase(self, state: RunState) -> None:
        with self._lock:
            self._state = state
            self._phase = state
            self._detail = PHASE_DETAILS.get(state.value, "")

    def _set_detail(self, detail: str) -> None:
        with self._lock:
            self._detail = detail

    # ---- recording -----------------------------------------------------

    def _start_recording(self, run_dir: Path) -> None:
        """Begin the run's recording, or say why there will not be one.

        A recording is evidence, not a phase: ffmpeg missing, a depth stream that
        will not open, a disk that will not take the file -- none of it ends the
        run. The reason goes on the detail line, so the topbar says why there is
        no video instead of the manifest being the only place that knows.
        """
        if not self._cfg.recording.enabled:
            return
        try:
            if self._recorder is None:
                # Both sources are passed, never defaulted: `Recorder` reads a None
                # factory as "not available here", which records nothing and says
                # so only in the manifest.
                self._recorder = Recorder(
                    self._cfg, run_dir, stream_factory=Stream, capture=ears.capture
                )
            self._recorder.start()
            self._recording = True
        except Exception as exc:  # noqa: BLE001 - a recording never fails a run
            self._recorder = None
            self._set_detail(f"recording unavailable: {exc}")

    def _open_ring(self, run_dir: Path) -> None:
        """Start the run's audio recorder, or say why there is not one.

        A recorder is evidence, like a recording: no parecord, no PulseAudio, no sink - none of
        it fails a run, and the brain loop falls back to recording each window itself.
        """
        if self._ring is not None:
            return
        try:
            ears.route()
            ring = ears.Ring(run_dir / "audio-ring.raw")
            if not ring.start():
                self._set_detail(f"no audio ring: {ring.error}")
                return
            self._ring = ring
        except Exception as exc:  # noqa: BLE001 - a recorder never fails a run
            self._set_detail(f"no audio ring: {exc}")

    def _stop_recording(self) -> None:
        """Stop the recorder, then hand the encode to a thread of its own.

        `stop()` flushes three encoders and is expected back quickly; the composite
        re-encodes the whole run and is not. Detaching it keeps `wait_idle`'s 15 s
        honest -- the run is over when the brain loop is, not when the video is --
        and the page reads `composing` from the manifest in the meantime.
        """
        recorder, self._recorder = self._recorder, None
        if recorder is None or not self._recording:
            return
        self._recording = False
        try:
            recorder.stop()
        except Exception as exc:  # noqa: BLE001 - the run is over either way
            print(f"clank: the recorder failed to stop: {exc}", file=sys.stderr)
        run_dir = self._dir
        if run_dir is None:
            return
        threading.Thread(
            target=self._composite_quietly,
            args=(run_dir,),
            name="clank-composite",
            daemon=True,
        ).start()

    def _composite_quietly(self, run_dir: Path) -> None:
        try:
            self._composite(run_dir, self._cfg)
        except Exception as exc:  # noqa: BLE001 - a detached thread has no caller
            print(f"clank: the composite failed: {exc}", file=sys.stderr)

    def _finish(self, state: RunState, error: str) -> None:
        if self._ring is not None:
            self._ring.stop()
            self._ring = None
        self._stop_recording()
        with self._lock:
            self._state = state
            self._phase = state
            self._detail = ""
            self._error = error or None
            self._ended_at = _now()
            self._ended_monotonic = time.monotonic()
        self._write_run_json()

    def _monitor(self, generation: int, child: Any, run_dir: Path) -> None:
        while True:
            if generation != self._generation or self._cancel.is_set():
                child.stop(self._cfg.run.stop_timeout_sec)
                self._finish(RunState.IDLE, "")
                return
            if not child.alive():
                code = child.exit_code()
                with self._lock:
                    self._exit_code = code
                if code in (0, None):
                    self._finish(RunState.IDLE, "")
                else:
                    tail = self._tail(run_dir / "brain.log")
                    self._finish(RunState.FAILED, f"the brain loop exited with code {code}{tail}")
                return
            self._cancel.wait(0.5)

    def _tail(self, path: Path, lines: int = 12) -> str:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        rows = [row for row in text.splitlines() if row.strip()][-lines:]
        return "\n--- brain.log tail ---\n" + "\n".join(rows) if rows else ""

    # ---- phases --------------------------------------------------------

    def _gate(self, generation: int) -> None:
        """Refuse the run unless the planner is serving the model it needs.

        A down server is a failure before any game restart is paid for; a server
        that is still loading is a wait, because a 2B Q8_0 load takes tens of
        seconds and that is the normal state after a reboot.
        """
        started = time.monotonic()
        autostarted = False
        required = self._cfg.run.required_model
        hint = self._cfg.run.model_start_cmd or "llama-server --model <LFM3-Omni GGUF> ..."
        while True:
            self._check_cancelled(generation)
            status = self._probe()
            if status.up and status.status == "ok":
                if required.lower() in status.model.lower():
                    return
                raise RuntimeError(
                    f"the model server is serving {_short_model(status)}, "
                    f"need {required}"
                )
            if status.up:
                self._set_detail("waiting for the model to load…")
            elif (
                not autostarted
                and self._cfg.run.model_autostart
                and self._cfg.run.model_start_cmd
            ):
                self._set_detail("starting the model server…")
                self._start_model()
                autostarted = True
            elif not autostarted:
                raise RuntimeError(
                    "the llama.cpp server is not up "
                    f"({status.error or status.status or 'no answer'}); "
                    f"start it with: {hint}"
                )
            if time.monotonic() - started > self._cfg.run.model_timeout_sec:
                raise RuntimeError(
                    "the llama.cpp server did not become ready within "
                    f"{self._cfg.run.model_timeout_sec:g}s; start it with: {hint}"
                )
            self._cancel.wait(2.0)

    def _start_model(self) -> None:
        argv = shlex.split(self._cfg.run.model_start_cmd)
        if not argv:
            raise RuntimeError("run.model_start_cmd is empty")
        log_path = (self._dir or Path.cwd()) / "model.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("ab") as log:
                subprocess.Popen(  # noqa: S603 - the operator's own command
                    argv,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    cwd=str(Path.cwd()),
                )
        except OSError as exc:
            raise RuntimeError(f"could not start the model server: {exc}") from exc

    def _ensure_run_dir(self, run_dir: Path) -> None:
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "frames").mkdir(exist_ok=True)
        except OSError as exc:
            raise RunError(f"could not create the run directory {run_dir}: {exc}") from exc
        self._write_run_json()

    def _open_run(self, run_dir: Path) -> None:
        self._ensure_run_dir(run_dir)
        diary = run_dir / "brain.jsonl"
        diary.touch()
        with self._lock:
            self._last_diary = diary
        if self._overlay is not None:
            self._overlay.clear()
        self._write_run_json()

    def _duration(self) -> float | None:
        """Seconds from start to now (or to the end), None before the first start."""
        if self._started_monotonic is None:
            return None
        end = self._ended_monotonic
        if end is None:
            end = time.monotonic()
        return round(max(0.0, end - self._started_monotonic), 2)

    def _write_run_json(self) -> None:
        run_dir = self._dir
        if run_dir is None:
            return
        with self._lock:
            payload = {
                "run_id": self._id,
                "duration_s": self._duration(),
                "session_id": self._session_id,
                "started_at": self._started_at,
                "ended_at": self._ended_at,
                "mode": "body",
                "brain_argv": self.brain_argv(),
                "diary": str(run_dir / "brain.jsonl"),
                "frames": str(run_dir / "frames"),
                "exit_code": self._exit_code,
                "error": self._error,
            }
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            tmp = run_dir / "run.json.tmp"
            tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, run_dir / "run.json")
        except OSError:
            # A run whose directory cannot be written is still a run in flight; the
            # failure that matters is in `last_error`, not in this bookkeeping.
            pass

    def _reset_body(self, generation: int) -> None:
        """Stand the robot up from wherever the last attempt left it.

        The order is the measured one (`auto.ready`): reset the process first, then
        ask for the rest pose repeatedly, because a restart alone leaves the arms in
        whatever pose the last command pinned.

        Two things make it a *reset* rather than a request to stand up. The process
        is killed and respawned, so the sim, the policy state and the clip all start
        at frame 0 - the `rest` verbs are a settle, not the reset. And the previous
        run's `body_state.json` is deleted before the new process exists: the relay
        holds the last state through a quiet body on purpose, buttons included, so
        without that it would spend the restart holding the previous run's final
        pose - and a closed hand with it. The relay is switched last, once there is
        a fresh body to relay.
        """
        if self._body is None:
            return
        self._body.stop()
        forget = getattr(self._body, "forget_state", None)
        if forget is not None:
            forget()
        # A socket answered by a process this dashboard never spawned cannot be
        # replaced by `stop()`, and the retry below would either lose every start
        # to the busy socket or drive a body on its way out.
        foreign = getattr(self._body, "owner_elsewhere", None)
        if foreign is not None and foreign():
            raise RuntimeError(
                f"another body view is already listening on "
                f"{self._body.control_path()}; stop it (`pkill -f body_view.py`) "
                f"and start the run again"
            )
        problem = self._body.start()
        if problem:
            raise RuntimeError(f"body view did not start: {problem}")
        # `start()` only spawns the sidecar; MuJoCo and the weights load after it,
        # and `auto.py` dies on its own first `rest` if nothing is listening. So the
        # first rest is retried until it is answered, then the rest of the ticks go
        # out as before.
        path = self._body.control_path()
        resting = 0
        starts = 1
        deadline = time.monotonic() + self._cfg.run.body_timeout_sec
        while resting < max(1, self._cfg.run.rest_ticks):
            self._check_cancel(generation)
            if not self._body.running():
                # A spawn that finds the dying viewer still holding the socket
                # prints "another body view is already listening" and exits at
                # once. The old process then answers our rest and unlinks the
                # socket as it goes, so "the socket replied" is not "my child is
                # up": notice the dead child and start another.
                starts += 1
                if starts > 4:
                    log = getattr(self._body, "log_path", None)
                    raise RuntimeError(
                        "the body view kept exiting"
                        + (f"; see {log}" if log else "")
                    )
                self._set_detail("restarting the body view…")
                problem = self._body.start()
                if problem:
                    raise RuntimeError(f"body view did not start: {problem}")
                self._cancel.wait(0.5)
                continue
            try:
                send_control(path, {"verb": "rest"})
            except OSError as exc:
                if resting == 0 and time.monotonic() < deadline:
                    self._set_detail("waiting for the body view…")
                    self._cancel.wait(0.5)
                    continue
                log = getattr(self._body, "log_path", None)
                where = f"; see {log}" if log else ""
                if resting == 0:
                    raise RuntimeError(
                        f"the body view did not answer on {path} within "
                        f"{self._cfg.run.body_timeout_sec:g}s{where}"
                    ) from None
                raise RuntimeError(f"the body view stopped answering: {exc}{where}") from exc
            resting += 1
            if resting < self._cfg.run.rest_ticks:
                time.sleep(0.25)
        # The relay last: it reads `body_state.json` as it starts, and by now the
        # file is the new process's or absent, never the previous run's.
        self._manager.set_mode("body")

    def _start_brain(self, run_dir: Path) -> Any:
        return self._spawn(
            argv=self.brain_argv(),
            env=self.brain_env(run_dir),
            cwd=Path.cwd(),
            log_path=run_dir / "brain.log",
        )


class _Child:
    """The brain loop as its own process group, with one log file.

    A process group, not a bare child: `auto.py` starts reaches and audio capture of
    its own, and stopping only the interpreter would leave those running against a
    game nobody is driving.
    """

    def __init__(self, argv: list[str], env: dict[str, str], cwd: Path, log_path: Path) -> None:
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._log_path.open("ab")
        try:
            self._proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                argv,
                cwd=str(cwd),
                env=env,
                stdout=self._handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            self._handle.close()
            raise
        self.pid = self._proc.pid

    def alive(self) -> bool:
        return self._proc.poll() is None

    def exit_code(self) -> int | None:
        return self._proc.poll()

    def stop(self, timeout: float = 10.0) -> None:
        if self._proc.poll() is not None:
            self._close()
            return
        with contextlib.suppress(OSError, ProcessLookupError):
            os.killpg(self._proc.pid, signal.SIGTERM)
        try:
            self._proc.wait(timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError, ProcessLookupError):
                os.killpg(self._proc.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(2.0)
        self._close()

    def _close(self) -> None:
        if not self._handle.closed:
            self._handle.close()


def spawn_child(argv: list[str], env: dict[str, str], cwd: Path, log_path: Path) -> _Child:
    """Start the brain loop, logging into the run directory."""
    return _Child(argv, env, Path(cwd), Path(log_path))
