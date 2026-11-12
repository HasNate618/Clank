"""The run state machine: one attempt, from model gate to brain loop."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

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

    def snapshot(self) -> dict:
        return {"state": self.state_value, "session_id": self.session_id}

    def restart_game(self) -> None:
        self.calls.append("game")
        if self.fail_restart:
            raise RuntimeError("steam refused")

    def set_mode(self, mode: str) -> None:
        self.calls.append(f"mode:{mode}")


class FakeBody:
    def __init__(self) -> None:
        self.running = False

    def start(self) -> str:
        self.running = True
        return ""

    def stop(self) -> None:
        self.running = False

    def control_path(self) -> Path:
        return Path("/nonexistent/body-control.sock")

    def preflight(self) -> str:
        return ""


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
        self.body = FakeBody()
        self.overlay = FakeOverlay()
        self.child = FakeChild()
        self.spawned: list[str] = []
        self.spawn_kwargs: dict = {}
        self.runs: list[RunManager] = []

    def spawn(self, **kwargs):
        self.spawned.append("brain")
        self.spawn_kwargs.update(kwargs)
        return self.child

    def make(self, manager=None, probe=None, model_timeout_sec=None, body: str | None = "default", spawn=None):
        cfg = self.cfg
        if model_timeout_sec is not None:
            cfg = replace(cfg, run=replace(cfg.run, model_timeout_sec=model_timeout_sec))
        run = RunManager(
            manager or self.manager,
            cfg,
            body=self.body if body == "default" else body,
            overlay=self.overlay,
            probe=probe
            or (lambda: brain.ServerStatus(up=True, status="ok", model="LFM3-Omni-2B-Q8_0")),
            spawn=spawn or self.spawn,
        )
        self.runs.append(run)
        self.addCleanup(self._stop_run, run)
        return run

    def _stop_run(self, run: RunManager) -> None:
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

    def test_mode_is_set_before_the_body_is_reset(self) -> None:
        run = self.make()
        order: list[str] = []

        def set_mode(mode):
            order.append(f"mode:{mode}")

        self.manager.set_mode = set_mode
        self.body.stop = lambda: order.append("body-stop")
        self.body.start = lambda: order.append("body-start") or ""
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        self.assertEqual(order[:2], ["mode:body", "body-stop"])

    def test_a_disabled_body_does_not_block_the_run(self) -> None:
        run = self.make(body=None)
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        self.assertEqual(self.spawned, ["brain"])

    def test_start_needs_a_running_stack(self) -> None:
        run = self.make(manager=FakeManager(state="stopped"))
        with self.assertRaises(ConflictError):
            run.start()
        self.assertEqual(self.spawned, [])
        self.assertFalse(self.runs_root.exists())

    def test_start_needs_a_session_id(self) -> None:
        run = self.make(manager=FakeManager(session_id=None))
        with self.assertRaises(ConflictError):
            run.start()

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

    def test_a_run_directory_failure_leaves_the_manager_usable(self) -> None:
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
        with self.assertRaises(run_mod.RunError):
            run.start()
        self.assertIn(run.snapshot()["state"], ("idle", "failed"))
        # Not a wedged manager: a second attempt fails the same way, it does not
        # claim "a run is already active" forever.
        with self.assertRaises(run_mod.RunError):
            run.start()

    def test_a_stopped_stack_cancels_the_run(self) -> None:
        def stop_the_stack() -> str:
            self.manager.state_value = "stopped"
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
