"""The run state machine: one attempt, from model gate to brain loop."""

from __future__ import annotations

import contextlib
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from clank import brain
from clank import run as run_mod
from clank.config import load_config
from clank.orchestrator import ConflictError
from clank.run import RunManager


class FakeManager:
    """What `RunManager` is allowed to ask the stack."""

    def __init__(
        self,
        state: str = "running",
        session_id: str | None = "20261112T101010Z-abc123",
        fail_restart: bool = False,
    ):
        self.state_value = state
        self.session_id = session_id
        self.fail_restart = fail_restart
        self.calls: list[str] = []
        self.started = 0
        self.stopped = 0
        #: The state a boot lands on: "running", "failed", "starting" (a boot that
        #: never finishes) or "never" (a call that does nothing at all).
        self.start_result = "running"
        #: False models a stack that runs but never names its session.
        self.mints_session_id = True
        #: Seconds a boot takes, so `stack` is a phase a test can see.
        self.start_delay = 0.0
        self.last_error: str | None = None

    def snapshot(self) -> dict:
        return {
            "state": self.state_value,
            "session_id": self.session_id,
            "last_error": self.last_error,
        }

    def start(self) -> None:
        """Booting the stack, as `SessionManager.start()` does it: non-blocking.

        `start_result` is the state the boot lands on; "never" models a stack that
        accepts the call and then does nothing, which is what a cancel has to
        survive without waiting out the whole deadline.
        """
        self.calls.append("stack")
        self.started += 1
        if self.start_delay:
            time.sleep(self.start_delay)     # a real boot takes minutes, not microseconds
        if self.start_result == "never":
            return
        self.state_value = self.start_result
        if self.start_result == "running" and self.mints_session_id:
            self.session_id = self.session_id or "20261113T101010Z-b00t01"

    def stop(self) -> None:
        self.calls.append("shutdown")
        self.stopped += 1
        self.state_value = "stopped"

    def restart_game(self) -> None:
        self.calls.append("game")
        if self.fail_restart:
            raise RuntimeError("steam refused")

    def set_mode(self, mode: str) -> None:
        self.calls.append(f"mode:{mode}")


class BodyServer:
    """A real AF_UNIX control socket that binds late, like the sidecar does."""

    def __init__(self, path: Path, delay: float = 0.0) -> None:
        self.path = Path(path)
        self.delay = delay
        self.verbs: list[dict] = []
        self._stop = threading.Event()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        if self.delay and self._stop.wait(self.delay):
            return
        try:
            self._sock.bind(str(self.path))
            self._sock.listen(8)
            # Closing a socket in another thread does not wake a blocked accept(),
            # so the loop polls and close() only has to set the flag.
            self._sock.settimeout(0.2)
        except OSError:
            return
        self._ready.set()
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


class FakeBody:
    def __init__(self, path: Path | None = None) -> None:
        self.alive = False
        self.path = path or Path("/nonexistent/body-control.sock")
        self.starts = 0
        # A viewer another tool started (or one that outlived a dashboard restart)
        # answers the socket while this object owns nothing.
        self.foreign = False
        self.forgotten = 0
        # A sidecar that loses the socket race exits at once: the first spawn
        # answers "another body view is already listening" and dies.
        self.dies_after_start = False

    def start(self) -> str:
        self.starts += 1
        self.alive = not self.dies_after_start
        return ""

    def running(self) -> bool:
        return self.alive

    def stop(self) -> None:
        self.alive = False

    def control_path(self) -> Path:
        return self.path

    def preflight(self) -> str:
        return ""

    def forget_state(self) -> None:
        self.forgotten += 1

    def owner_elsewhere(self) -> bool:
        return self.foreign


class FakeChild:
    def __init__(self) -> None:
        self.alive_flag = True
        self.exit_code_value: int | None = None
        self.stopped = False

    def alive(self) -> bool:
        return self.alive_flag

    def stop(self, timeout: float) -> None:
        self.stopped = True
        self.alive_flag = False

    def exit_code(self) -> int | None:
        return self.exit_code_value


class FakeOverlay:
    def __init__(self) -> None:
        self.cleared = 0

    def clear(self) -> None:
        self.cleared += 1


_FRESH_RECORDER = object()


class FakeRecorder:
    """Stands in for `Recorder`: the state machine does not need ffmpeg."""

    def __init__(self, manifest: dict | None = None):
        self.started = 0
        self.stopped = 0
        self.manifest = (
            manifest if manifest is not None else {"state": "ready", "file": "recording.mp4"}
        )

    def start(self) -> None:
        self.started += 1

    def stop(self) -> dict:
        self.stopped += 1
        return self.manifest


class UnstartableRecorder(FakeRecorder):
    """The ffmpeg-absent case: a recording that cannot begin must not end the run."""

    def start(self) -> None:
        self.started += 1
        raise OSError("ffmpeg is not installed")


class SlowComposite:
    """A composite that outlives the run: `wait_idle` must not wait for it."""

    def __init__(self, seconds: float = 3.0):
        self.seconds = seconds
        self.started = threading.Event()
        self.args: tuple | None = None

    def __call__(self, run_dir: Path, cfg) -> dict:
        self.args = (run_dir, cfg)
        self.started.set()
        time.sleep(self.seconds)
        return {"state": "ready"}


class RunHarness(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        cfg = load_config(None, env={})
        cfg = replace(
            cfg,
            paths=replace(cfg.paths, session_root=str(self.base / "sessions")),
            run=replace(
                cfg.run,
                brain_python=sys.executable,
                brain_script="tests/test_run.py",
                model_timeout_sec=5.0,
                stop_timeout_sec=1.0,
                rest_ticks=1,
            ),
        )
        self.cfg = cfg
        self.runs_root = Path(cfg.paths.session_root)
        self.manager = FakeManager()
        self.body_socket = self.base / "body-control.sock"
        self.body_server = BodyServer(self.body_socket, delay=0.0)
        self.addCleanup(self.body_server.close)
        self.body = FakeBody(self.body_socket)
        self.overlay = FakeOverlay()
        self.child = FakeChild()
        self.spawned: list[str] = []
        self.spawn_kwargs: dict = {}
        self.runs: list[RunManager] = []
        self.composites: list[Path] = []
        self.phases: dict[RunManager, list[str]] = {}
        self._watchers: list[threading.Event] = []

    def spawn(self, **kwargs):
        self.spawned.append("brain")
        self.spawn_kwargs.update(kwargs)
        return self.child

    def make(
        self,
        manager=None,
        probe=None,
        model_timeout_sec=None,
        session_start_sec=None,
        body: str | None = "default",
        spawn=None,
        recorder: object = _FRESH_RECORDER,
        composite=None,
        recording: bool = True,
    ):
        cfg = self.cfg
        if model_timeout_sec is not None:
            cfg = replace(cfg, run=replace(cfg.run, model_timeout_sec=model_timeout_sec))
        if session_start_sec is not None:
            cfg = replace(cfg, run=replace(cfg.run, session_start_sec=session_start_sec))
        if not recording:
            cfg = replace(cfg, recording=replace(cfg.recording, enabled=False))
        # A run must not need ffmpeg (or a depth stream) to be tested, so the
        # harness hands it a fake recorder unless a test wants the real one.
        if recorder is _FRESH_RECORDER:
            recorder = FakeRecorder()
        self.recorder = recorder
        run = RunManager(
            manager or self.manager,
            cfg,
            body=self.body if body == "default" else body,
            overlay=self.overlay,
            probe=probe
            or (lambda: brain.ServerStatus(up=True, status="ok", model="LFM3-Omni-2B-Q8_0")),
            spawn=spawn or self.spawn,
            recorder=recorder,
            composite=composite or (lambda run_dir, cfg: self.composites.append(run_dir)),
        )
        self.runs.append(run)
        self._watch(run)
        self.addCleanup(self._stop_run, run)
        return run

    def _watch(self, run: RunManager) -> None:
        """Record the phases this run passes through, read from its own snapshot.

        Sampled, not instrumented: `snapshot()` is exactly what the page sees, so a
        phase the sampler missed is a phase the operator would not have been shown.
        """
        seen: list[str] = []
        self.phases[run] = seen
        stop = threading.Event()
        self._watchers.append(stop)

        def loop() -> None:
            while not stop.is_set():
                phase = run.snapshot().get("phase")
                if phase and (not seen or seen[-1] != phase):
                    seen.append(phase)
                stop.wait(0.002)

        threading.Thread(target=loop, name="clank-watch-phases", daemon=True).start()

    def phases_seen(self, run: RunManager | None = None) -> list[str]:
        return list(self.phases.get(run or self.runs[-1], []))

    def wait_phase(self, phase: str, run: RunManager | None = None, timeout: float = 5.0) -> bool:
        target = run or self.runs[-1]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if phase in self.phases_seen(target):
                return True
            time.sleep(0.005)
        return False

    def _stop_run(self, run: RunManager) -> None:
        for stop in self._watchers:
            stop.set()
        run.stop()
        run.wait_idle(5)

    def wait_state(self, state: str, timeout: float = 5.0, run: RunManager | None = None) -> str:
        target = run or self.runs[-1]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if target.snapshot()["state"] == state:
                return state
            time.sleep(0.01)
        return target.snapshot()["state"]


class RunManagerTests(RunHarness):
    def test_start_builds_the_run_dir_and_diary(self) -> None:
        run = self.make()
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        snapshot = run.snapshot()
        run_dir = Path(snapshot["dir"])
        data = json.loads((run_dir / "run.json").read_text())
        self.assertEqual(data["run_id"], snapshot["id"])
        self.assertEqual(data["session_id"], self.manager.session_id)
        self.assertTrue((run_dir / "brain.jsonl").exists())
        self.assertEqual(Path(data["diary"]), run_dir / "brain.jsonl")
        self.assertEqual(Path(data["frames"]), run_dir / "frames")
        self.assertEqual(self.overlay.cleared, 1)

    def test_phases_run_game_then_body_then_brain(self) -> None:
        run = self.make()
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        self.assertEqual(self.manager.calls, ["game", "mode:body"])
        self.assertEqual(self.spawned, ["brain"])

    def test_mode_is_set_after_the_body_is_fresh(self) -> None:
        """The relay holds the last state it read through a quiet body, buttons
        included, so a relay restarted *before* the sim is replaced starts on the
        previous run's final pose - a closed hand included. Order: kill, forget,
        start, rest, then switch the relay."""
        run = self.make()
        order: list[str] = []

        def set_mode(mode):
            order.append(f"mode:{mode}")

        self.manager.set_mode = set_mode
        self.body.stop = lambda: order.append("body-stop")
        self.body.forget_state = lambda: order.append("forget")

        def start() -> str:
            order.append("body-start")
            self.body.alive = True
            return ""

        self.body.start = start
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        self.assertEqual(order[:3], ["body-stop", "forget", "body-start"])
        self.assertEqual(order[-1], "mode:body")
        self.assertGreater(order.index("mode:body"), order.index("body-start"))

    def test_a_disabled_body_does_not_block_the_run(self) -> None:
        run = self.make(body=None)
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        self.assertEqual(self.spawned, ["brain"])

    def test_a_run_starts_the_stack_when_it_is_down(self) -> None:
        manager = FakeManager(state="stopped", session_id=None)
        manager.start_delay = 0.05
        run = self.make(manager=manager)
        run.start()
        self.assertEqual(self.wait_state("running", run=run), "running")
        self.assertEqual(manager.started, 1)            # the stack was booted
        self.assertIn("stack", self.phases_seen(run))   # and it said so

    def test_a_run_does_not_touch_a_running_stack(self) -> None:
        run = self.make()
        run.start()
        self.assertEqual(self.wait_state("running", run=run), "running")
        self.assertEqual(self.manager.started, 0)
        self.assertNotIn("stack", self.phases_seen(run))

    def test_a_stack_that_fails_to_start_fails_the_run_with_its_reason(self) -> None:
        manager = FakeManager(state="stopped", session_id=None)
        manager.start_result = "failed"
        manager.last_error = "monado refused to load the driver"
        run = self.make(manager=manager)
        run.start()
        self.assertTrue(run.wait_idle(20))
        info = run.snapshot()
        self.assertEqual(info["state"], "failed")
        self.assertIn("stack", info["error"].lower())
        self.assertIn("monado refused", info["error"])
        self.assertEqual(self.spawned, [])          # no game, no body, no brain

    def test_a_run_cancelled_while_the_stack_starts_leaves_it_alone(self) -> None:
        manager = FakeManager(state="stopped", session_id=None)
        manager.start_result = "starting"       # a boot that never gets there
        run = self.make(manager=manager)
        run.start()
        deadline = time.monotonic() + 5
        while manager.started == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(manager.started, 1, "the stack was never asked to boot")
        self.assertTrue(self.wait_phase("stack", run=run), "the phase was never shown")
        run.stop()                                  # cancel during `stack`
        self.assertTrue(run.wait_idle(20))
        self.assertEqual(manager.stopped, 0)        # nothing was shut down
        self.assertEqual(run.snapshot()["state"], "idle")
        self.assertEqual(self.spawned, [])

    def test_a_stack_whose_start_is_ignored_fails_at_once(self) -> None:
        """A start that leaves the stack stopped never began: do not wait it out."""
        manager = FakeManager(state="stopped", session_id=None)
        manager.start_result = "never"
        run = self.make(manager=manager)
        started = time.monotonic()
        run.start()
        self.assertTrue(run.wait_idle(5))
        info = run.snapshot()
        self.assertEqual(info["state"], "failed")
        self.assertIn("stopped", info["error"])
        self.assertLess(time.monotonic() - started, 3.0)

    def test_a_stack_that_never_names_its_session_fails_the_run(self) -> None:
        manager = FakeManager(state="stopped", session_id=None)
        manager.mints_session_id = False
        run = self.make(manager=manager, session_start_sec=0.3)
        run.start()
        self.assertTrue(run.wait_idle(20))
        info = run.snapshot()
        self.assertEqual(info["state"], "failed")
        self.assertIn("0.3", info["error"])
        self.assertIsNone(info["id"])               # no session, so no run directory

    def test_start_twice_conflicts(self) -> None:
        run = self.make()
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        with self.assertRaises(ConflictError):
            run.start()

    def test_gate_refuses_a_down_server_before_any_game_restart(self) -> None:
        run = self.make(
            probe=lambda: brain.ServerStatus(up=False, status="down", error="connection refused")
        )
        run.start()
        self.assertEqual(self.wait_state("failed"), "failed")
        self.assertIn("llama", run.snapshot()["error"])
        self.assertEqual(self.manager.calls, [])
        self.assertEqual(self.spawned, [])
        self.assertTrue(Path(run.snapshot()["dir"]).is_dir())

    def test_gate_refuses_the_wrong_model(self) -> None:
        run = self.make(
            probe=lambda: brain.ServerStatus(up=True, status="ok", model="LFM3-VL-2B-Q8_0")
        )
        run.start()
        self.assertEqual(self.wait_state("failed"), "failed")
        self.assertIn("LFM3-Omni", run.snapshot()["error"])
        self.assertIn("LFM3-VL-2B", run.snapshot()["error"])
        self.assertEqual(self.manager.calls, [])

    def test_gate_waits_while_the_model_loads(self) -> None:
        answers = iter(
            [
                brain.ServerStatus(up=True, status="loading"),
                brain.ServerStatus(up=True, status="ok", model="LFM3-Omni-2B-Q8_0"),
            ]
        )
        run = self.make(
            probe=lambda: next(answers, brain.ServerStatus(up=True, status="ok", model="LFM3-Omni-2B-Q8_0"))
        )
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        self.assertEqual(self.spawned, ["brain"])

    def test_detail_names_the_current_wait(self) -> None:
        gate = threading.Event()
        run = self.make(
            probe=lambda: (gate.wait(2), brain.ServerStatus(up=True, status="loading"))[1],
            model_timeout_sec=30.0,
        )
        run.start()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and "model" not in run.snapshot()["detail"]:
            time.sleep(0.01)
        self.assertIn("model", run.snapshot()["detail"])
        gate.set()
        run.stop()
        self.assertTrue(run.wait_idle(5))
        self.assertEqual(run.snapshot()["detail"], "")

    def test_game_failure_never_stops_the_stack(self) -> None:
        run = self.make(manager=FakeManager(fail_restart=True))
        run.start()
        self.assertEqual(self.wait_state("failed"), "failed")
        self.assertIn("steam refused", run.snapshot()["error"])
        self.assertEqual(self.spawned, [])

    def test_stop_before_the_brain_phase_spawns_nothing(self) -> None:
        gate = threading.Event()
        run = self.make(
            probe=lambda: (gate.wait(2), brain.ServerStatus(up=True, status="ok", model="LFM3-Omni"))[1]
        )
        run.start()
        time.sleep(0.05)
        run.stop()
        gate.set()
        self.assertTrue(run.wait_idle(5))
        self.assertEqual(run.snapshot()["state"], "idle")
        self.assertEqual(self.spawned, [])
        self.assertEqual(self.manager.calls, [])
        self.assertIsNone(run.snapshot()["error"])

    def test_child_exit_ends_the_run(self) -> None:
        run = self.make()
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        self.child.alive_flag = False
        self.child.exit_code_value = 0
        self.assertTrue(run.wait_idle(5))
        snapshot = run.snapshot()
        self.assertEqual(snapshot["state"], "idle")
        self.assertEqual(snapshot["exit_code"], 0)
        self.assertIsNotNone(snapshot["ended_at"])

    def test_snapshot_while_idle_keeps_the_last_run(self) -> None:
        run = self.make()
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        run_id, run_dir = run.snapshot()["id"], run.snapshot()["dir"]
        self.child.alive_flag = False
        self.assertTrue(run.wait_idle(5))
        self.assertEqual(run.snapshot()["id"], run_id)
        self.assertEqual(run.snapshot()["dir"], run_dir)
        self.assertEqual(run.current_diary(), Path(run_dir) / "brain.jsonl")

    def test_current_diary_falls_back_to_the_newest_run_on_disk(self) -> None:
        newest = (
            self.runs_root
            / "20261112T101010Z-abc123"
            / "runs"
            / "999999-abcdef"
        )
        newest.mkdir(parents=True)
        (newest / "brain.jsonl").write_text('{"kind": "think", "text": "last run"}\n')
        run = self.make()
        self.assertEqual(run.current_diary(), newest / "brain.jsonl")

    def test_argv_and_env_name_the_run(self) -> None:
        run = self.make()
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        self.assertEqual(self.spawn_kwargs["argv"], run.brain_argv())
        self.assertEqual(self.spawn_kwargs["argv"][1], self.cfg.run.brain_script)
        self.assertIn("--for", self.spawn_kwargs["argv"])
        env = self.spawn_kwargs["env"]
        self.assertTrue(env["CLANK_BRAIN_LOG"].endswith("brain.jsonl"))
        self.assertTrue(env["CLANK_AUTO_STATE"].endswith("frames"))
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("VIRTUAL_ENV", env)
        self.assertEqual(Path(self.spawn_kwargs["log_path"]).name, "brain.log")

    def test_phases_include_preparing(self) -> None:
        run = self.make()
        seen: list[str] = []
        original = run._set_phase

        def record(state):
            seen.append(state.value)
            original(state)

        run._set_phase = record
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        self.assertEqual(seen[:4], ["preparing", "game", "body", "brain"])

    def test_run_json_records_the_duration(self) -> None:
        run = self.make()
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        self.child.alive_flag = False
        self.assertTrue(run.wait_idle(5))
        data = json.loads((Path(run.snapshot()["dir"]) / "run.json").read_text())
        self.assertIsInstance(data["duration_s"], float)
        self.assertGreaterEqual(data["duration_s"], 0.0)

    def test_stop_terminates_the_child_before_returning(self) -> None:
        run = self.make()
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        run.stop()
        # Synchronous: the monitor's next tick is up to 0.5 s away, and the stack
        # may be shutting Steam down inside that window.
        self.assertTrue(self.child.stopped)

    def test_a_run_directory_failure_fails_the_run_not_the_manager(self) -> None:
        blocker = self.base / "blocked"
        blocker.write_text("not a directory")
        cfg = replace(self.cfg, paths=replace(self.cfg.paths, session_root=str(blocker)))
        run = RunManager(
            self.manager,
            cfg,
            body=self.body,
            overlay=self.overlay,
            probe=lambda: brain.ServerStatus(up=True, status="ok", model="LFM3-Omni"),
            spawn=self.spawn,
        )
        self.addCleanup(run.wait_idle, 2)
        # The directory is made on the worker, after the boot: nothing waits on the
        # HTTP thread, so the failure is the run's rather than the caller's.
        run.start()
        self.assertTrue(run.wait_idle(5))
        info = run.snapshot()
        self.assertEqual(info["state"], "failed")
        self.assertIn("blocked", info["error"])
        # Not a wedged manager: a second attempt fails the same way, it does not
        # claim "a run is already active" forever.
        run.start()
        self.assertTrue(run.wait_idle(5))
        self.assertEqual(run.snapshot()["state"], "failed")
        self.assertIn("blocked", run.snapshot()["error"])

    def test_body_phase_waits_for_the_control_socket(self) -> None:
        """The sidecar returns from start() before it binds; a rest sent into the
        void is a body that was never reset, and the brain loop then dies on its
        own first rest."""
        late_path = self.base / "late-body-control.sock"
        late = BodyServer(late_path, delay=1.0)
        self.addCleanup(late.close)
        self.body.path = late_path
        run = self.make()
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        self.assertEqual([v.get("verb") for v in late.verbs], ["rest"])

    def test_body_phase_restarts_a_sidecar_that_lost_the_socket_race(self) -> None:
        """A spawn that finds the dying viewer still holding the socket exits at
        once. The run must notice its own child is gone and start another, not
        call the reset done because the *old* process answered one rest."""
        self.body.dies_after_start = True
        real_start = self.body.start

        def start_and_heal() -> str:
            reply = real_start()
            if self.body.starts >= 2:
                server = BodyServer(self.body.path)
                self.addCleanup(server.close)
                self.body.dies_after_start = False
            return reply

        self.body.start = start_and_heal
        run = self.make()
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        self.assertGreaterEqual(self.body.starts, 2)
        self.assertTrue(self.body.running())

    def test_a_body_view_from_elsewhere_is_a_named_failure(self) -> None:
        """`stop()` is a no-op when this dashboard never spawned the view, so a run
        that carries on restarts nothing and gets its `rest` answered by a process
        it does not own - the corpse race in AGENTS.md, one `pkill` away from gone."""
        self.body.foreign = True
        run = self.make()
        run.start()
        self.assertTrue(run.wait_idle(5))
        snapshot = run.snapshot()
        self.assertEqual(snapshot["state"], "failed")
        self.assertIn("already listening", snapshot["error"])
        self.assertIn("pkill", snapshot["error"])
        # Nothing was started and nothing was rested on the stranger's body.
        self.assertEqual(self.body.starts, 0)

    def test_a_reset_forgets_the_previous_run_state(self) -> None:
        run = self.make()
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        self.assertEqual(self.body.forgotten, 1)

    def test_a_silent_body_view_fails_the_run_before_the_brain(self) -> None:
        self.body.path = self.base / "never.sock"
        cfg = replace(self.cfg, run=replace(self.cfg.run, body_timeout_sec=0.6))
        run = RunManager(
            self.manager,
            cfg,
            body=self.body,
            overlay=self.overlay,
            probe=lambda: brain.ServerStatus(up=True, status="ok", model="LFM3-Omni"),
            spawn=self.spawn,
        )
        self.addCleanup(run.wait_idle, 2)
        run.start()
        self.assertTrue(run.wait_idle(5))
        snapshot = run.snapshot()
        self.assertEqual(snapshot["state"], "failed")
        self.assertIn("never.sock", snapshot["error"])
        self.assertIn("did not answer", snapshot["error"])
        # No body, no relay: switching the pose client onto a sim that never came
        # up would have the game hold the previous run's pose for the whole failure.
        self.assertEqual(self.spawned, [])
        # The game restart is paid first; the reset that failed is the body's. The
        # relay is not switched at all: with no body the client would hold the
        # previous run's pose for the whole failure.
        self.assertEqual(self.manager.calls, ["game"])

    def test_a_stopped_stack_cancels_the_run(self) -> None:
        def stop_the_stack() -> str:
            self.manager.state_value = "stopped"
            self.body.alive = True
            return ""

        self.body.start = stop_the_stack
        run = self.make()
        run.start()
        self.assertTrue(run.wait_idle(5))
        self.assertEqual(run.snapshot()["state"], "idle")
        self.assertEqual(self.spawned, [])

    def test_stop_reports_a_stopped_child_as_a_stop_not_a_failure(self) -> None:
        run = self.make()
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        run.stop()
        self.assertTrue(run.wait_idle(5))
        self.assertTrue(self.child.stopped)
        self.assertEqual(run.snapshot()["state"], "idle")
        self.assertIsNone(run.snapshot()["error"])


class PreflightTests(RunHarness):
    def test_a_broken_body_view_is_refused_before_anything_starts(self) -> None:
        from clank.run import RunError

        self.body.preflight = lambda: "body python not found: /nonexistent"
        run = self.make()
        with self.assertRaises(RunError):
            run.start()
        self.assertEqual(self.manager.calls, [])

    def test_missing_brain_python_is_refused_before_anything_starts(self) -> None:
        from clank.run import RunError

        cfg = replace(self.cfg, run=replace(self.cfg.run, brain_python="/nonexistent/python"))
        run = RunManager(
            self.manager,
            cfg,
            body=self.body,
            overlay=self.overlay,
            probe=lambda: brain.ServerStatus(up=True, status="ok", model="LFM3-Omni"),
            spawn=self.spawn,
        )
        self.addCleanup(run.wait_idle, 2)
        with self.assertRaises(RunError):
            run.start()
        self.assertEqual(self.manager.calls, [])


class ChildTests(unittest.TestCase):
    """The brain loop as a process: its exit code, and stopping it for real."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.log = Path(self.temp.name) / "brain.log"

    def child(self, code: str):
        return run_mod.spawn_child(
            [sys.executable, "-c", code], os.environ.copy(), Path.cwd(), self.log
        )

    def test_exit_code_is_recorded(self) -> None:
        child = self.child("raise SystemExit(3)")
        self.addCleanup(child.stop, 2)
        deadline = time.monotonic() + 10
        while child.alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertFalse(child.alive())
        self.assertEqual(child.exit_code(), 3)

    def test_stop_terminates_a_lingering_child(self) -> None:
        child = self.child("import time; time.sleep(30)")
        self.addCleanup(child.stop, 2)
        self.assertTrue(child.alive())
        started = time.monotonic()
        child.stop(2.0)
        self.assertFalse(child.alive())
        self.assertLess(time.monotonic() - started, 5.0)

    def test_output_goes_to_the_run_log(self) -> None:
        child = self.child("print('heard jobbot')")
        self.addCleanup(child.stop, 2)
        deadline = time.monotonic() + 10
        while child.alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertIn("heard jobbot", self.log.read_text())


class RecorderWiringTests(RunHarness):
    def test_the_recorder_starts_with_the_brain_and_stops_with_the_run(self) -> None:
        recorder = FakeRecorder()
        run = self.make(recorder=recorder)
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        self.assertEqual(recorder.started, 1)
        self.child.alive_flag = False
        self.assertTrue(run.wait_idle(5))
        self.assertEqual(recorder.stopped, 1)

    def test_a_run_refused_at_the_gate_never_starts_a_recorder(self) -> None:
        recorder = FakeRecorder()
        run = self.make(
            recorder=recorder,
            probe=lambda: brain.ServerStatus(up=False, status="down", model=""),
        )
        run.start()
        self.assertTrue(run.wait_idle(5))
        self.assertEqual(run.snapshot()["state"], "failed")
        self.assertEqual(recorder.started, 0)

    def test_the_composite_does_not_hold_the_run_open(self) -> None:
        """A run that stops must not wait for the encode that follows it."""
        composite = SlowComposite(3.0)
        run = self.make(composite=composite)
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        self.child.alive_flag = False
        started = time.monotonic()
        self.assertTrue(run.wait_idle(5))
        elapsed = time.monotonic() - started
        self.assertTrue(composite.started.wait(2), "the composite never ran")
        self.assertLess(elapsed, 1.5, f"stop waited {elapsed:.2f}s for the composite")

    def test_a_recording_that_cannot_start_does_not_fail_the_run(self) -> None:
        recorder = UnstartableRecorder()
        run = self.make(recorder=recorder)
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        self.assertEqual(run.snapshot()["state"], "running")
        self.assertEqual(recorder.stopped, 0)  # nothing was started, so nothing is stopped
        self.child.alive_flag = False
        self.assertTrue(run.wait_idle(5))

    def test_a_disabled_recording_is_never_started(self) -> None:
        recorder = FakeRecorder()
        run = self.make(recorder=recorder, recording=False)
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        self.assertEqual(recorder.started, 0)
        self.child.alive_flag = False
        self.assertTrue(run.wait_idle(5))

    def test_a_real_recorder_gets_the_stream_and_the_ears(self) -> None:
        """The two sources a real recorder needs are passed, not defaulted."""
        run = RunManager(
            self.manager,
            self.cfg,
            body=self.body,
            overlay=self.overlay,
            probe=lambda: brain.ServerStatus(up=True, status="ok", model="LFM3-Omni-2B-Q8_0"),
            spawn=self.spawn,
            recorder=None,
            composite=lambda run_dir, cfg: None,
        )
        self.runs.append(run)
        self.addCleanup(self._stop_run, run)
        with mock.patch("clank.run.Recorder") as recorder_cls:
            run.start()
            self.assertEqual(self.wait_state("running"), "running")
            self.child.alive_flag = False
            self.assertTrue(run.wait_idle(5))
        called = recorder_cls.call_args
        kwargs = called.kwargs
        self.assertIs(kwargs["stream_factory"], run_mod.Stream)
        self.assertIs(kwargs["capture"], run_mod.ears.capture)
        self.assertEqual(
            Path(called.args[1]).resolve(), Path(run.snapshot()["dir"]).resolve()
        )

    def test_the_snapshot_carries_the_manifest_the_recorder_left(self) -> None:
        run = self.make()
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        self.assertIsNone(run.snapshot()["recording"])
        run_dir = Path(run.snapshot()["dir"])
        (run_dir / "recording.json").write_text(
            json.dumps({"state": "recording", "sources": {"game": "ok"}})
        )
        self.assertEqual(run.snapshot()["recording"]["state"], "recording")
        self.child.alive_flag = False
        self.assertTrue(run.wait_idle(5))
