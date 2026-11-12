from __future__ import annotations

import collections
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from clank import brain
from clank.config import load_config
from clank.processes import ProcessError
from clank.orchestrator import (
    ConflictError,
    PreflightError,
    SessionManager,
    State,
    StartupError,
    _OpenXRLogParser,
    initial_mode,
    parse_openxr_client,
    proton_prefix,
)


class FakeHandle:
    def __init__(self, name: str, pid: int) -> None:
        self.name = name
        self.pid = pid
        self.pgid = pid
        self.log_path = Path(tempfile.gettempdir()) / f"{name}.log"
        self._alive = True
        self._group_alive = False

    def alive(self) -> bool:
        return self._alive

    def group_alive(self) -> bool:
        return self._group_alive

    def stop(self, timeout: float = 5.0) -> None:
        self._alive = False

    def signal_group(self, sig: int) -> None:
        self._alive = False


class FakeRunner:
    def __init__(self) -> None:
        self.handles: dict[str, FakeHandle] = {}
        self.next_pid = 100
        self.host_steam = False
        self.units: set[str] = set()
        self.ports_in_use: set[tuple[str, int]] = set()
        self.pointer = None
        self.game_started = False
        self.scene_graph: str | None = "window: title: Steam"
        self.fail_at: str | None = None
        self.block_wait: threading.Event | None = None
        self.runs: list[list[str]] = []
        self.starts: list[tuple[str, list[str]]] = []
        self.waits: list[tuple[str, int]] = []
        self.alive_pids: set[int] = set()
        self.removed_socket = False
        self.stopped_units: list[str] = []
        self.reclaimed_ports: list[tuple[int, str]] = []

    def start(self, name, argv, env, cwd, log_path, stderr_path=None):
        self.starts.append((name, [str(item) for item in argv]))
        if self.fail_at == f"start:{name}":
            raise RuntimeError(f"{name} failed to start")
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch()
        if stderr_path is not None:
            stderr_path = Path(stderr_path)
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            stderr_path.touch()
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
        self.waits.append((host, port))
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

    def stop_systemd_unit(self, unit):
        self.stopped_units.append(unit)
        self.units.discard(unit)

    def port_in_use(self, host, port):
        return (host, port) in self.ports_in_use

    def reclaim_port(self, port, expect="monado", net_text=None, timeout=3.0):
        self.reclaimed_ports.append((port, expect))
        if self.fail_at == "reclaim":
            raise ProcessError(
                "port 4242 is held by pid 7 (game-server --port 4242), "
                "which is not a monado process"
            )
        self.ports_in_use.discard(("127.0.0.1", port))
        return [1410672]

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
        argv = [str(item) for item in argv]
        self.runs.append(argv)
        # The fullscreen step polls `xdotool search`; answer it so startup does not
        # sit on the 30 s deadline in every test.
        if argv and argv[0].endswith("xdotool") and "search" in argv:
            return subprocess.CompletedProcess(argv, 0, "4242\n", "")
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
        try:
            with Path(path).open("r", errors="replace") as handle:
                data = collections.deque(handle, maxlen=lines)
        except OSError:
            return ""
        text = "".join(data)
        if text.endswith("\n"):
            text = text[:-1]
        return text


class SessionHarness(unittest.TestCase):
    """A started-looking session with fake processes, shared by several suites."""

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
            run=replace(cfg.run, game_restart_timeout_sec=2.0),
        )
        self.cfg = cfg
        self.runner = FakeRunner()
        self.manager = SessionManager(cfg, runner=self.runner)  # type: ignore[arg-type]

    def tearDown(self) -> None:
        self.manager.stop()
        self.manager.wait_idle(2)
        self.temp.cleanup()

    def start(self) -> None:
        with patch("clank.orchestrator.missing_requirements", return_value=[]):
            self.manager.start()
        self.assertTrue(self.manager.wait_idle(2))
        self.assertEqual(self.manager.state_value(), "running")


class OrchestratorTests(SessionHarness):
    def test_start_reaches_running(self) -> None:
        self.start()
        self.assertEqual(
            set(self.runner.handles),
            {"monado", "weston", "proxy", "steam", "pose"},
        )
        self.assertIsNotNone(self.runner.pointer)
        snapshot = self.manager.snapshot()
        self.assertIsNotNone(snapshot["session_id"])
        self.assertIsNotNone(snapshot["started_at"])

    def test_session_starts_in_body_mode_when_body_enabled(self) -> None:
        cfg = replace(self.cfg, body=replace(self.cfg.body, enabled=True))
        self.assertEqual(initial_mode(cfg), "body")

    def test_session_starts_in_hold_mode_when_body_disabled(self) -> None:
        # load_config's default: no body view, so the static pose stands in.
        self.assertEqual(initial_mode(self.cfg), "hold")

    def test_steam_env_carries_the_viewport_scale(self) -> None:
        self.manager.cfg = replace(
            self.cfg,
            launch=replace(self.cfg.launch, viewport_scale_percentage=150.0),
        )
        self.assertEqual(
            self.manager._steam_env()["OXR_VIEWPORT_SCALE_PERCENTAGE"], "150"
        )

    def test_game_audio_muted_reads_pipewire_when_the_game_has_streams(self) -> None:
        # module-stream-restore can mute the game behind the dashboard's back,
        # so the reported state comes from wpctl, not from the flag last written.
        with (
            patch.object(self.manager, "_game_audio_node_ids", return_value=[55]),
            patch("clank.orchestrator.subprocess.run") as run,
        ):
            run.return_value = subprocess.CompletedProcess(
                [], 0, stdout="Volume: 1.00 [MUTED]\n", stderr=""
            )
            self.assertTrue(self.manager.game_audio_muted())
            run.return_value = subprocess.CompletedProcess(
                [], 0, stdout="Volume: 0.80\n", stderr=""
            )
            self.assertFalse(self.manager.game_audio_muted())

    def test_game_audio_muted_falls_back_to_the_flag_without_streams(self) -> None:
        with patch.object(self.manager, "_game_audio_node_ids", return_value=[]):
            self.assertFalse(self.manager.game_audio_muted())
            self.manager.set_game_audio_muted(True)
            self.assertTrue(self.manager.game_audio_muted())

    def test_apply_mute_pushes_the_flag_not_the_live_state(self) -> None:
        # Asking PipeWire for the value to write would return the state being
        # replaced and undo the toggle mid-flight.
        calls: list[list[str]] = []

        def fake_run(argv, **_kwargs):
            calls.append(list(argv))
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        with (
            patch.object(self.manager, "_game_audio_node_ids", return_value=[55]),
            patch("clank.orchestrator.subprocess.run", side_effect=fake_run),
        ):
            self.manager.set_game_audio_muted(True)
        self.assertIn(["wpctl", "set-mute", "55", "1"], calls)

    def test_steam_env_omits_a_native_scale(self) -> None:
        self.manager.cfg = replace(
            self.cfg,
            launch=replace(self.cfg.launch, viewport_scale_percentage=100.0),
        )
        self.assertNotIn(
            "OXR_VIEWPORT_SCALE_PERCENTAGE", self.manager._steam_env()
        )

    def test_steam_env_imports_the_host_openxr_runtime(self) -> None:
        self.manager.cfg = replace(
            self.cfg, paths=replace(self.cfg.paths, vr_override="/shim")
        )
        env = self.manager._steam_env()
        self.assertEqual(env["PRESSURE_VESSEL_IMPORT_OPENXR_1_RUNTIMES"], "1")
        # Pinning the client with XR_RUNTIME_JSON makes OpenComposite fail to
        # enumerate extensions, so the host's registered runtime must be imported.
        self.assertNotIn("XR_RUNTIME_JSON", env)

    def test_launch_asks_the_game_window_to_fullscreen(self) -> None:
        self.start()
        calls = [
            argv for argv in self.runner.runs if argv and argv[0].endswith("xdotool")
        ]
        self.assertTrue(any("search" in argv for argv in calls), calls)
        windowstate = [argv for argv in calls if "windowstate" in argv]
        self.assertTrue(windowstate, calls)
        self.assertIn("fullscreen", windowstate[-1])
        self.assertIn("4242", windowstate[-1])

    def test_start_uses_vnc_and_proxy(self) -> None:
        self.start()
        self.assertIn("proxy", self.runner.handles)
        weston_argv = next(
            argv for name, argv in self.runner.starts if name == "weston"
        )
        for expected in (
            "--backend=vnc",
            "--renderer=gl",
            "--address=127.0.0.1",
            "--port=5901",
            "--disable-transport-layer-security",
        ):
            self.assertIn(expected, weston_argv)
        config_arg = next(arg for arg in weston_argv if arg.startswith("--config="))
        config_path = Path(config_arg.split("=", 1)[1])
        config = config_path.read_text()
        self.assertIn("[core]", config)
        self.assertIn("idle-time=0", config)
        self.assertIn("[vnc]", config)
        self.assertIn("refresh-rate=30", config)
        proxy_argv = next(
            argv for name, argv in self.runner.starts if name == "proxy"
        )
        self.assertEqual(proxy_argv[-2:], ["127.0.0.1:6080", "127.0.0.1:5901"])
        self.assertIn(("127.0.0.1", 5901), self.runner.waits)
        self.assertIn(("127.0.0.1", 6080), self.runner.waits)

    def test_snapshot_reports_vnc_transport(self) -> None:
        self.start()
        snapshot = self.manager.snapshot()
        preview = snapshot["preview"]
        self.assertEqual(preview["mode"], "vnc")
        self.assertEqual(preview["width"], self.cfg.preview.width)
        # The VNC clip is the full framebuffer: no panel is subtracted.
        self.assertEqual(preview["height"], self.cfg.preview.height)
        self.assertEqual(preview["fps_target"], 30.0)
        self.assertEqual(preview["vnc_port"], 5901)
        self.assertEqual(preview["proxy_port"], 6080)
        self.assertTrue(preview["vnc_alive"])
        self.assertTrue(preview["proxy_alive"])
        self.assertIn("proxy", snapshot["processes"])
        self.assertTrue(snapshot["processes"]["proxy"]["alive"])
        self.assertIn("proxy", snapshot["logs"])

    def test_vnc_port_conflict_fails(self) -> None:
        self.runner.ports_in_use.add(("127.0.0.1", 5901))
        with patch("clank.orchestrator.missing_requirements", return_value=[]):
            self.manager.start()
        self.assertTrue(self.manager.wait_idle(2))
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["state"], "failed")
        self.assertIn("5901", snapshot["last_error"])
        self.assertIsNone(self.runner.pointer)
        self.assertTrue(
            all(not handle.alive() for handle in self.runner.handles.values())
        )

    def test_proxy_port_conflict_fails(self) -> None:
        self.runner.ports_in_use.add(("127.0.0.1", 6080))
        with patch("clank.orchestrator.missing_requirements", return_value=[]):
            self.manager.start()
        self.assertTrue(self.manager.wait_idle(2))
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["state"], "failed")
        self.assertIn("6080", snapshot["last_error"])
        self.assertIsNone(self.runner.pointer)
        self.assertTrue(
            all(not handle.alive() for handle in self.runner.handles.values())
        )

    def test_weston_stdout_log_is_written_but_not_public(self) -> None:
        self.start()
        snapshot = self.manager.snapshot()
        self.assertTrue((Path(snapshot["session_dir"]) / "weston-stdout.log").exists())
        self.assertIsNone(self.manager.log_path("weston-stdout"))

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
        self.assertTrue(snapshot["player_log"].endswith("output_log.txt"))

    def test_model_status_is_probed_once_per_ttl(self) -> None:
        # The dashboard polls /api/state every second; the planner's server must
        # not be asked that often, or a wedged server holds every poll.
        calls = []

        def status():
            calls.append(1)
            return brain.ServerStatus(up=True, status="ok", model="LFM3-Omni-2B")

        with patch("clank.orchestrator.brain.server_status", side_effect=status):
            first = self.manager.model_status()
            second = self.manager.model_status()
        self.assertEqual(len(calls), 1)
        self.assertEqual(first["model"], "LFM3-Omni-2B")
        self.assertTrue(second["up"])

    def test_snapshot_carries_the_model_status(self) -> None:
        with patch(
            "clank.orchestrator.brain.server_status",
            return_value=brain.ServerStatus(up=True, status="ok", model="omni"),
        ):
            snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["model"]["status"], "ok")
        self.assertEqual(snapshot["model"]["model"], "omni")

    def test_snapshot_keeps_openxr_client_across_log_growth(self) -> None:
        self.start()
        log_path = self.manager.log_path("monado")
        assert log_path is not None
        log_path.write_text(
            " INFO [client_connected] Client 1 connected\n"
            "\tapplication_name: 'Unity Application'\n"
        )
        self.assertEqual(self.manager.snapshot()["openxr_client"], "Unity Application")
        with log_path.open("a") as handle:
            for index in range(300):
                handle.write(f"chatter {index}\n")
        self.assertEqual(self.manager.snapshot()["openxr_client"], "Unity Application")

    def test_snapshot_reads_openxr_log_in_bounded_chunks(self) -> None:
        self.start()
        log_path = self.manager.log_path("monado")
        assert log_path is not None
        log_path.write_text(
            "chatter\n" * 20000
            + " INFO [client_connected] Client 1 connected\n"
            + "\tapplication_name: 'Unity Application'\n"
        )
        reads: list[int] = []
        real_open = Path.open

        class CountingHandle:
            def __init__(self, handle) -> None:
                self.handle = handle

            def __enter__(self):
                return self

            def __exit__(self, *args) -> None:
                self.handle.close()

            def seek(self, *args):
                return self.handle.seek(*args)

            def tell(self):
                return self.handle.tell()

            def read(self, amount: int = -1):
                reads.append(amount)
                return self.handle.read(amount)

        def counting_open(path, *args, **kwargs):
            return CountingHandle(real_open(path, *args, **kwargs))

        with patch.object(Path, "open", counting_open):
            client = self.manager.snapshot()["openxr_client"]
        self.assertEqual(client, "Unity Application")
        self.assertTrue(reads)
        self.assertTrue(all(0 <= amount <= 65536 for amount in reads))

    def test_snapshot_clears_openxr_client_on_disconnect(self) -> None:
        self.start()
        log_path = self.manager.log_path("monado")
        assert log_path is not None
        log_path.write_text(
            " INFO [client_connected] Client 1 connected\n"
            "\tapplication_name: 'Unity Application'\n"
        )
        self.assertEqual(self.manager.snapshot()["openxr_client"], "Unity Application")
        with log_path.open("a") as handle:
            handle.write(" INFO [client_disconnected] Client 1 disconnected\n")
        self.assertIsNone(self.manager.snapshot()["openxr_client"])

    def test_snapshot_parses_openxr_client_split_across_polls(self) -> None:
        self.start()
        log_path = self.manager.log_path("monado")
        assert log_path is not None
        log_path.write_text(" INFO [client_connected] Client 1 conn")
        self.assertIsNone(self.manager.snapshot()["openxr_client"])
        with log_path.open("a") as handle:
            handle.write("ected\n\tapplication_name: 'Unity Application'")
        self.assertEqual(self.manager.snapshot()["openxr_client"], "Unity Application")
        with log_path.open("a") as handle:
            handle.write("\n")
        self.assertEqual(self.manager.snapshot()["openxr_client"], "Unity Application")

    def test_snapshot_does_not_advance_offset_when_parser_feed_fails(self) -> None:
        self.start()
        log_path = self.manager.log_path("monado")
        assert log_path is not None
        log_path.write_text(
            " INFO [client_connected] Client 1 connected\n"
            "\tapplication_name: 'APP A'\n"
        )
        self.assertEqual(self.manager.snapshot()["openxr_client"], "APP A")
        parser = self.manager._openxr_parser
        real_feed = parser.feed
        calls = 0

        def failing_feed(text: str) -> str | None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("parser failure")
            return real_feed(text)

        with log_path.open("a") as handle:
            handle.write(
                " INFO [client_connected] Client 2 connected\n"
                "\tapplication_name: 'APP B'\n"
            )
        with patch.object(parser, "feed", side_effect=failing_feed):
            with self.assertRaises(RuntimeError):
                self.manager.snapshot()
        self.assertEqual(self.manager.snapshot()["openxr_client"], "APP B")

    def test_snapshot_resets_openxr_state_after_truncation(self) -> None:
        self.start()
        log_path = self.manager.log_path("monado")
        assert log_path is not None
        log_path.write_text(
            " INFO [client_connected] Client 1 connected\n"
            "\tapplication_name: 'Unity Application'\n"
        )
        self.assertEqual(self.manager.snapshot()["openxr_client"], "Unity Application")
        log_path.write_text("")
        self.assertIsNone(self.manager.snapshot()["openxr_client"])
        log_path.write_text(
            " INFO [client_connected] Client 2 connected\n"
            "\tapplication_name: 'Other App'\n"
        )
        self.assertEqual(self.manager.snapshot()["openxr_client"], "Other App")

    def test_snapshot_detects_truncate_and_regrow(self) -> None:
        self.start()
        log_path = self.manager.log_path("monado")
        assert log_path is not None
        log_path.write_text(
            " INFO [client_connected] Client 1 connected\n"
            "\tapplication_name: 'APP A'\n"
        )
        self.assertEqual(self.manager.snapshot()["openxr_client"], "APP A")
        log_path.write_text(
            " INFO [client_connected] Client 2 connected\n"
            "\tapplication_name: 'APP B'\n"
            + "chatter\n" * 40
        )
        self.assertEqual(self.manager.snapshot()["openxr_client"], "APP B")

    def test_snapshot_detects_same_size_rewrite_with_identical_tail(self) -> None:
        self.start()
        log_path = self.manager.log_path("monado")
        assert log_path is not None
        filler = ("z" * 63 + "\n") * 4
        log_path.write_text(
            "[client_connected] Client 1 connected\n"
            "\tapplication_name: 'APP A'\n"
            + filler
        )
        self.assertEqual(self.manager.snapshot()["openxr_client"], "APP A")
        log_path.write_text(
            "[client_connected] Client 1 connected\n"
            "\tapplication_name: 'APP B'\n"
            + filler
        )
        self.assertEqual(self.manager.snapshot()["openxr_client"], "APP B")

    def test_snapshot_hides_previous_openxr_client_while_start_prepares(self) -> None:
        self.start()
        log_path = self.manager.log_path("monado")
        assert log_path is not None
        log_path.write_text(
            " INFO [client_connected] Client 1 connected\n"
            "\tapplication_name: 'OLD APP'\n"
        )
        self.assertEqual(self.manager.snapshot()["openxr_client"], "OLD APP")
        self.manager.stop()
        self.assertTrue(self.manager.wait_idle(2))

        entered = threading.Event()
        release = threading.Event()
        original_preflight = self.manager._preflight

        def blocking_preflight(gen):
            entered.set()
            self.assertTrue(release.wait(2))
            return original_preflight(gen)

        with (
            patch.object(self.manager, "_preflight", side_effect=blocking_preflight),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            thread = threading.Thread(target=self.manager.start, daemon=True)
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                self.assertEqual(self.manager.state_value(), "starting")
                snapshot = self.manager.snapshot()
                self.assertIsNone(snapshot["openxr_client"])
                self.assertIsNone(snapshot["session_id"])
                self.assertIsNone(snapshot["started_at"])
            finally:
                release.set()
                thread.join(2)
        self.assertTrue(self.manager.wait_idle(2))

    def test_failed_preflight_clears_previous_session_paths(self) -> None:
        self.start()
        log_path = self.manager.log_path("monado")
        assert log_path is not None
        log_path.write_text(
            " INFO [client_connected] Client 1 connected\n"
            "\tapplication_name: 'OLD APP'\n"
        )
        self.assertEqual(self.manager.snapshot()["openxr_client"], "OLD APP")
        self.assertIsNotNone(self.manager.snapshot()["session_dir"])
        self.assertIsNotNone(self.manager.snapshot()["session_id"])
        self.assertIsNotNone(self.manager.snapshot()["started_at"])
        self.manager.stop()
        self.assertTrue(self.manager.wait_idle(2))

        self.runner.host_steam = True
        with patch("clank.orchestrator.missing_requirements", return_value=[]):
            with self.assertRaises(PreflightError):
                self.manager.start()
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["state"], "failed")
        self.assertIsNone(snapshot["openxr_client"])
        self.assertIsNone(snapshot["session_dir"])
        self.assertIsNone(snapshot["session_id"])
        self.assertIsNone(snapshot["started_at"])
        self.assertIsNone(self.manager.log_path("weston"))

    def test_snapshot_clears_openxr_client_after_stop(self) -> None:
        self.start()
        log_path = self.manager.log_path("monado")
        assert log_path is not None
        log_path.write_text(
            " INFO [client_connected] Client 1 connected\n"
            "\tapplication_name: 'Unity Application'\n"
        )
        self.assertEqual(self.manager.snapshot()["openxr_client"], "Unity Application")
        self.manager.stop()
        self.assertTrue(self.manager.wait_idle(2))
        self.assertIsNone(self.manager.snapshot()["openxr_client"])

    def test_snapshot_resets_openxr_state_after_rotation(self) -> None:
        self.start()
        log_path = self.manager.log_path("monado")
        assert log_path is not None
        log_path.write_text(
            " INFO [client_connected] Client 1 connected\n"
            "\tapplication_name: 'Unity Application'\n"
        )
        self.assertEqual(self.manager.snapshot()["openxr_client"], "Unity Application")
        rotated = log_path.with_name("monado.log.rotated")
        rotated.write_text("chatter " * 40)
        os.replace(rotated, log_path)
        self.assertIsNone(self.manager.snapshot()["openxr_client"])

    def test_host_steam_blocks_start(self) -> None:
        self.runner.host_steam = True
        with patch("clank.orchestrator.missing_requirements", return_value=[]):
            with self.assertRaises(PreflightError):
                self.manager.start()
        self.assertEqual(self.manager.state_value(), "failed")
        self.assertIsNone(self.runner.pointer)
        self.assertEqual(self.runner.handles, {})

    def test_probe_failure_blocks_start(self) -> None:
        with (
            patch.object(
                self.runner,
                "process_name_running",
                side_effect=OSError("pgrep missing"),
            ),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            with self.assertRaises(PreflightError) as raised:
                self.manager.start()
        self.assertIn("pgrep missing", str(raised.exception))
        self.assertEqual(self.manager.state_value(), "failed")
        self.assertIsNone(self.runner.pointer)
        self.assertEqual(self.runner.handles, {})

    def test_preflight_reclaims_a_leftover_monado_on_4242(self) -> None:
        # A previous session's monado outliving its teardown matches no unit, so it
        # used to make every later start refuse with "TCP 4242 is already in use".
        self.runner.ports_in_use.add(("127.0.0.1", 4242))
        with patch("clank.orchestrator.missing_requirements", return_value=[]):
            try:
                self.manager.start()
            except Exception:
                pass  # only the preflight decision is under test here
        self.assertEqual(self.runner.reclaimed_ports, [(4242, "monado")])
        self.assertNotIn(("127.0.0.1", 4242), self.runner.ports_in_use)

    def test_preflight_refuses_when_4242_belongs_to_a_stranger(self) -> None:
        self.runner.ports_in_use.add(("127.0.0.1", 4242))
        self.runner.fail_at = "reclaim"
        with patch("clank.orchestrator.missing_requirements", return_value=[]):
            with self.assertRaises(PreflightError) as raised:
                self.manager.start()
        self.assertIn("game-server", str(raised.exception))
        self.assertEqual(self.manager.state_value(), "failed")
        self.assertIsNone(self.runner.pointer)

    def test_systemd_probe_failure_blocks_start(self) -> None:
        with (
            patch.object(
                self.runner,
                "systemd_unit_active",
                side_effect=OSError("systemctl missing"),
            ),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            with self.assertRaises(PreflightError) as raised:
                self.manager.start()
        self.assertIn("systemctl missing", str(raised.exception))
        self.assertEqual(self.manager.state_value(), "failed")
        self.assertIsNone(self.runner.pointer)
        self.assertEqual(self.runner.handles, {})

    def test_active_monado_socket_is_reclaimed_not_refused(self) -> None:
        self.runner.units.add("monado.socket")
        self.start()
        self.assertEqual(self.runner.stopped_units, ["monado.socket"])
        self.assertEqual(self.runner.units, set())

    def test_running_monado_service_is_stopped_before_its_socket(self) -> None:
        self.runner.units.update({"monado.service", "monado.socket"})
        self.start()
        self.assertEqual(self.runner.stopped_units, ["monado.service", "monado.socket"])

    def test_unit_that_will_not_stop_blocks_start(self) -> None:
        self.runner.units.add("monado.socket")
        with (
            patch.object(self.runner, "stop_systemd_unit", return_value=None),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            with self.assertRaises(PreflightError) as raised:
                self.manager.start()
        self.assertIn("still active", str(raised.exception))
        self.assertEqual(self.manager.state_value(), "failed")
        self.assertIsNone(self.runner.pointer)

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

    def test_stale_proxy_wrapper_blocks_start(self) -> None:
        self.runner.alive_pids.add(999)
        self.runner.pointer = {
            "session": "/old",
            "processes": {
                "proxy": {
                    "pid": 999,
                    "command": "/nix/store/example/bin/websockify",
                }
            },
        }
        with (
            patch.object(
                self.runner,
                "pid_cmdline",
                return_value=[
                    "/nix/store/example/bin/python3",
                    "/nix/store/example/bin/websockify",
                    "127.0.0.1:6080",
                    "127.0.0.1:5901",
                ],
            ),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            with self.assertRaises(PreflightError) as raised:
                self.manager.start()
        self.assertIn("stale", str(raised.exception))
        self.assertIn("proxy", str(raised.exception))
        self.assertEqual(self.manager.state_value(), "failed")

    def test_non_matching_stale_pointer_process_is_ignored(self) -> None:
        self.runner.alive_pids.add(999)
        self.runner.pointer = {
            "session": "/old",
            "processes": {
                "proxy": {
                    "pid": 999,
                    "command": "/nix/store/example/bin/websockify",
                }
            },
        }
        with (
            patch.object(
                self.runner,
                "pid_cmdline",
                return_value=[
                    "/nix/store/example/bin/python3",
                    "/nix/store/example/bin/unrelated",
                ],
            ),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            self.manager.start()
        self.assertTrue(self.manager.wait_idle(2))
        self.assertEqual(self.manager.state_value(), "running")

    def test_startup_failure_cleans_up(self) -> None:
        self.runner.fail_at = "wait_port"
        with patch("clank.orchestrator.missing_requirements", return_value=[]):
            self.manager.start()
        self.assertTrue(self.manager.wait_idle(2))
        self.assertEqual(self.manager.state_value(), "failed")
        self.assertIn("monado", self.manager.snapshot()["last_error"])
        self.assertIsNone(self.runner.pointer)
        self.assertTrue(all(not handle.alive() for handle in self.runner.handles.values()))

    def test_startup_failure_includes_step_and_log_tail(self) -> None:
        original_wait_log = self.runner.wait_log

        def failing_wait_log(path, needle, timeout, abort=None):
            if needle == "handshake ok":
                Path(path).write_text("pose handshake detail\n")
                raise RuntimeError("pose handshake timeout")
            return original_wait_log(path, needle, timeout, abort)

        with (
            patch.object(self.runner, "wait_log", side_effect=failing_wait_log),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            self.manager.start()
            self.assertTrue(self.manager.wait_idle(2))
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["state"], "failed")
        self.assertIn("pose failed", snapshot["last_error"])
        self.assertIn("pose handshake detail", snapshot["last_error"])

    def test_startup_failure_uses_weston_stdout_fallback_tail(self) -> None:
        def failing_start_weston() -> None:
            self.manager._log("weston").write_text("")
            self.manager._log("weston-stdout").write_text("weston stdout detail\n")
            raise RuntimeError("weston display timeout")

        with (
            patch.object(self.manager, "_start_weston", side_effect=failing_start_weston),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            self.manager.start()
            self.assertTrue(self.manager.wait_idle(2))
        last_error = self.manager.snapshot()["last_error"]
        self.assertIn("weston failed", last_error)
        self.assertIn("weston stdout detail", last_error)
        self.assertIn("--- weston-stdout tail ---", last_error)
        self.assertNotIn("--- weston tail ---", last_error)

    def test_startup_timeout_at_step_boundary_includes_step_and_tail(self) -> None:
        self.manager.cfg = replace(
            self.cfg,
            launch=replace(self.cfg.launch, startup_timeout_sec=0.05),
        )

        def slow_wait_port(host, port, timeout, abort=None) -> None:
            self.manager._log("monado").write_text("monado startup detail\n")
            time.sleep(0.15)

        with (
            patch.object(self.runner, "wait_port", side_effect=slow_wait_port),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            self.manager.start()
            self.assertTrue(self.manager.wait_idle(2))
        last_error = self.manager.snapshot()["last_error"]
        self.assertEqual(self.manager.state_value(), "failed")
        self.assertIn("monado failed", last_error)
        self.assertIn("startup timeout exceeded", last_error)
        self.assertIn("monado startup detail", last_error)

    def test_startup_tail_read_failure_keeps_original_error(self) -> None:
        self.runner.fail_at = "wait_port"
        with (
            patch.object(self.runner, "tail", side_effect=RuntimeError("tail boom")),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            self.manager.start()
            self.assertTrue(self.manager.wait_idle(2))
        last_error = self.manager.snapshot()["last_error"]
        self.assertEqual(self.manager.state_value(), "failed")
        self.assertIn("monado failed", last_error)
        self.assertIn("monado port timeout", last_error)
        self.assertNotIn("tail boom", last_error)

    def test_startup_timeout_bounds_slow_step(self) -> None:
        self.manager.cfg = replace(
            self.cfg,
            launch=replace(self.cfg.launch, startup_timeout_sec=0.25),
            timeouts=replace(self.cfg.timeouts, monado_startup_sec=2),
        )

        def slow_wait_port(host, port, timeout, abort=None) -> None:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                time.sleep(0.01)
            raise TimeoutError("monado port timeout")

        started = time.monotonic()
        with (
            patch.object(self.runner, "wait_port", side_effect=slow_wait_port),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            self.manager.start()
            self.assertTrue(self.manager.wait_idle(5))
        elapsed = time.monotonic() - started
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["state"], "failed")
        self.assertIn("monado failed", snapshot["last_error"])
        self.assertLess(elapsed, 1.0)

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
        self.assertFalse(self.runner.handles["proxy"].alive())

    def test_clean_stop_reports_no_survivors(self) -> None:
        self.start()
        self.manager.stop()
        self.assertTrue(self.manager.wait_idle(2))
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["state"], "stopped")
        self.assertEqual(snapshot["survivors"], [])
        session_log = Path(snapshot["session_dir"]) / "session.log"
        self.assertNotIn("surviving processes", session_log.read_text())

    def test_stop_reports_surviving_process_group(self) -> None:
        self.start()
        self.runner.handles["steam"]._group_alive = True
        self.manager.stop()
        self.assertTrue(self.manager.wait_idle(2))
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["state"], "stopped")
        self.assertIn("steam", snapshot["survivors"])
        session_log = Path(snapshot["session_dir"]) / "session.log"
        self.assertIn("warning: surviving processes: steam", session_log.read_text())

    def test_start_clears_previous_survivors(self) -> None:
        self.manager._survivors = ["steam"]
        self.start()
        self.assertEqual(self.manager.snapshot()["survivors"], [])

    def test_start_resets_stale_handles_and_game_pids(self) -> None:
        self.manager._handles = {"ghost": FakeHandle("ghost", 900)}  # type: ignore[dict-item]
        self.manager._game_pids = [777]
        self.manager._survivors = ["ghost"]
        self.start()
        self.assertNotIn("ghost", self.manager._handles)
        self.assertNotIn(777, self.manager._game_pids)
        self.assertEqual(self.manager.snapshot()["survivors"], [])

    def test_cleanup_failure_still_clears_state(self) -> None:
        self.runner.fail_at = "wait_port"
        original_remove = self.runner.remove_stale_socket
        calls = 0

        def flaky_remove(path):
            nonlocal calls
            calls += 1
            if calls >= 2:
                raise RuntimeError("cleanup boom")
            return original_remove(path)

        with (
            patch.object(self.runner, "remove_stale_socket", side_effect=flaky_remove),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            self.manager.start()
            self.assertTrue(self.manager.wait_idle(2))
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["state"], "failed")
        self.assertIsNone(self.runner.pointer)
        self.assertEqual(self.manager._handles, {})
        self.assertEqual(self.manager._game_pids, [])
        self.assertEqual(snapshot["survivors"], [])

    def test_cleanup_phase_failure_does_not_skip_remaining_phases(self) -> None:
        self.start()
        self.runner.handles["steam"]._group_alive = True
        with patch.object(
            self.runner,
            "remove_stale_socket",
            side_effect=RuntimeError("socket boom"),
        ):
            self.manager.stop()
            self.assertTrue(self.manager.wait_idle(2))
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["state"], "stopped")
        self.assertFalse(self.runner.handles["proxy"].alive())
        self.assertTrue(all(not handle.alive() for handle in self.runner.handles.values()))
        self.assertEqual(snapshot["survivors"], ["steam"])
        session_log = Path(snapshot["session_dir"]) / "session.log"
        text = session_log.read_text()
        self.assertIn("socket boom", text)
        self.assertIn("warning: surviving processes: steam", text)

    def test_cleanup_first_phase_failure_still_stops_later_handles(self) -> None:
        self.start()
        pose = self.runner.handles["pose"]
        later = {
            name: self.runner.handles[name] for name in ("steam", "weston", "monado")
        }

        def failing_stop(timeout: float = 5.0) -> None:
            raise RuntimeError("pose stop boom")

        pose.stop = failing_stop
        self.manager.stop()
        self.assertTrue(self.manager.wait_idle(2))
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["state"], "stopped")
        self.assertFalse(self.runner.handles["proxy"].alive())
        self.assertIn("pose", snapshot["survivors"])
        self.assertTrue(all(not handle.alive() for handle in later.values()))
        session_log = Path(snapshot["session_dir"]) / "session.log"
        text = session_log.read_text()
        self.assertIn("cleanup phase pose failed: pose stop boom", text)
        self.assertIn("warning: surviving processes: pose", text)

    def test_startup_failure_preserves_step_error_when_cleanup_fails(self) -> None:
        original_wait_log = self.runner.wait_log

        def failing_wait_log(path, needle, timeout, abort=None):
            if needle == "handshake ok":
                Path(path).write_text("pose handshake detail\n")
                raise RuntimeError("pose handshake timeout")
            return original_wait_log(path, needle, timeout, abort)

        original_remove = self.runner.remove_stale_socket
        calls = 0

        def flaky_remove(path):
            nonlocal calls
            calls += 1
            if calls >= 2:
                raise RuntimeError("cleanup boom")
            return original_remove(path)

        with (
            patch.object(self.runner, "wait_log", side_effect=failing_wait_log),
            patch.object(self.runner, "remove_stale_socket", side_effect=flaky_remove),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            self.manager.start()
            self.assertTrue(self.manager.wait_idle(2))
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["state"], "failed")
        self.assertIn("pose failed", snapshot["last_error"])
        self.assertIn("pose handshake detail", snapshot["last_error"])
        self.assertNotIn("cleanup failed", snapshot["last_error"])
        session_log = Path(snapshot["session_dir"]) / "session.log"
        self.assertIn("cleanup boom", session_log.read_text())

    def test_stop_cleanup_failure_without_prior_error_reports_and_logs(self) -> None:
        self.start()
        with patch.object(
            self.runner,
            "remove_stale_socket",
            side_effect=RuntimeError("socket boom"),
        ):
            self.manager.stop()
            self.assertTrue(self.manager.wait_idle(2))
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["state"], "stopped")
        self.assertIn("cleanup failed", snapshot["last_error"])
        self.assertIn("socket boom", snapshot["last_error"])
        session_log = Path(snapshot["session_dir"]) / "session.log"
        self.assertIn("socket boom", session_log.read_text())

    def test_stop_during_startup_clears_cleanup_error(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def blocking_wait_port(host, port, timeout, abort=None) -> None:
            entered.set()
            release.wait(2)

        original_remove = self.runner.remove_stale_socket
        calls = 0

        def flaky_remove(path):
            nonlocal calls
            calls += 1
            if calls >= 2:
                raise RuntimeError("cleanup boom")
            return original_remove(path)

        with (
            patch.object(self.runner, "wait_port", side_effect=blocking_wait_port),
            patch.object(self.runner, "remove_stale_socket", side_effect=flaky_remove),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            self.manager.start()
            self.assertTrue(entered.wait(2))
            self.manager.stop()
            release.set()
            self.assertTrue(self.manager.wait_idle(2))
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["state"], "stopped")
        self.assertIsNone(snapshot["last_error"])
        session_log = Path(snapshot["session_dir"]) / "session.log"
        self.assertIn("cleanup boom", session_log.read_text())

    def test_stop_during_preflight_removes_pointer(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        original_preflight = self.manager._preflight

        def blocking_preflight(gen):
            session_dir = original_preflight(gen)
            entered.set()
            self.assertTrue(release.wait(2))
            return session_dir

        with (
            patch.object(self.manager, "_preflight", side_effect=blocking_preflight),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            thread = threading.Thread(target=self.manager.start, daemon=True)
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                session_dir = self.manager._session_dir
                self.assertIsNotNone(session_dir)
                assert session_dir is not None
                self.assertTrue(session_dir.exists())
                self.manager.stop()
                self.assertTrue(self.manager.wait_idle(2))
                release.set()
                thread.join(2)
                worker = self.manager._thread
                self.assertIsNotNone(worker)
                assert worker is not None
                worker.join(2)
            finally:
                release.set()
                thread.join(2)
        self.assertTrue(self.manager.wait_idle(2))
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["state"], "stopped")
        self.assertIsNone(self.runner.pointer)
        self.assertIsNone(snapshot["session_dir"])
        self.assertFalse(session_dir.exists())

    def test_stop_after_final_startup_checkpoint_ends_stopped(self) -> None:
        reached = threading.Event()
        release = threading.Event()
        cleanup_calls: list[bool] = []
        original_startup = self.manager._startup
        original_cleanup = self.manager._cleanup

        def paused_startup() -> None:
            original_startup()
            reached.set()
            self.assertTrue(release.wait(2))

        def counting_cleanup() -> None:
            cleanup_calls.append(True)
            original_cleanup()

        with (
            patch.object(self.manager, "_startup", side_effect=paused_startup),
            patch.object(self.manager, "_cleanup", side_effect=counting_cleanup),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            self.manager.start()
            self.assertTrue(reached.wait(2))
            self.manager.stop()
            release.set()
            self.assertTrue(self.manager.wait_idle(2))
        self.assertEqual(self.manager.state_value(), "stopped")
        self.assertEqual(len(cleanup_calls), 1)
        self.assertFalse(self.runner.handles["proxy"].alive())
        self.assertIsNone(self.runner.pointer)

    def test_stop_during_starting_cleans_up_once(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        cleanup_calls: list[bool] = []
        original_cleanup = self.manager._cleanup

        def blocking_wait_port(host, port, timeout, abort=None) -> None:
            entered.set()
            release.wait(2)

        def counting_cleanup() -> None:
            cleanup_calls.append(True)
            original_cleanup()

        with (
            patch.object(self.runner, "wait_port", side_effect=blocking_wait_port),
            patch.object(self.manager, "_cleanup", side_effect=counting_cleanup),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            self.manager.start()
            self.assertTrue(entered.wait(2))
            self.manager.stop()
            release.set()
            self.assertTrue(self.manager.wait_idle(2))
        self.assertEqual(self.manager.state_value(), "stopped")
        self.assertEqual(len(cleanup_calls), 1)
        self.assertNotIn("proxy", self.runner.handles)
        self.assertIsNone(self.runner.pointer)

    def test_stop_during_starting_has_no_failure_tail(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def blocking_wait_port(host, port, timeout, abort=None) -> None:
            entered.set()
            release.wait(2)

        with (
            patch.object(self.runner, "wait_port", side_effect=blocking_wait_port),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            self.manager.start()
            self.assertTrue(entered.wait(2))
            self.manager.stop()
            release.set()
            self.assertTrue(self.manager.wait_idle(2))
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["state"], "stopped")
        self.assertIsNone(snapshot["last_error"])

    def test_stop_before_start_worker_cleans_up_once(self) -> None:
        allow = threading.Event()
        finished = threading.Event()
        cleanup_calls: list[bool] = []
        original_start_worker = self.manager._start_worker
        original_cleanup = self.manager._cleanup

        def delayed_start_worker(gen, session_dir) -> None:
            self.assertTrue(allow.wait(2))
            original_start_worker(gen, session_dir)
            finished.set()

        def counting_cleanup() -> None:
            cleanup_calls.append(True)
            original_cleanup()

        with (
            patch.object(self.manager, "_start_worker", side_effect=delayed_start_worker),
            patch.object(self.manager, "_cleanup", side_effect=counting_cleanup),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            self.manager.start()
            captured = self.manager._session_dir
            self.assertIsNotNone(captured)
            assert captured is not None
            self.assertTrue(captured.exists())
            self.manager.stop()
            allow.set()
            self.assertTrue(finished.wait(2))
            self.assertTrue(self.manager.wait_idle(2))
        self.assertEqual(self.manager.state_value(), "stopped")
        self.assertEqual(len(cleanup_calls), 1)
        self.assertIsNone(self.runner.pointer)
        self.assertFalse(captured.exists())

    def test_stop_during_preflight_then_start_preserves_new_session(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        original_preflight = self.manager._preflight
        calls = 0
        calls_lock = threading.Lock()

        def blocking_first_preflight(gen):
            nonlocal calls
            with calls_lock:
                calls += 1
                first = calls == 1
            if first:
                entered.set()
                self.assertTrue(release.wait(5))
            return original_preflight(gen)

        with (
            patch.object(
                self.manager, "_preflight", side_effect=blocking_first_preflight
            ),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            thread_a = threading.Thread(target=self.manager.start, daemon=True)
            thread_a.start()
            self.assertTrue(entered.wait(2))
            self.manager.stop()
            self.assertTrue(self.manager.wait_idle(2))

            self.manager.start()
            self.assertTrue(self.manager.wait_idle(2))
            self.assertEqual(self.manager.state_value(), "running")
            session_dir = Path(self.manager.snapshot()["session_dir"])
            pointer = self.runner.pointer
            self.assertIsNotNone(pointer)
            marker = session_dir / "marker.log"
            marker.write_text("new session\n")

            release.set()
            thread_a.join(5)

        self.assertFalse(thread_a.is_alive())
        self.assertTrue(session_dir.exists())
        self.assertTrue(marker.exists())
        self.assertEqual(self.runner.pointer, pointer)
        self.assertEqual(self.manager.state_value(), "running")
        self.assertEqual(self.manager.snapshot()["session_dir"], str(session_dir))

    def test_late_preflight_failure_does_not_overwrite_successor(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        discard_entered = threading.Event()
        allow_discard = threading.Event()
        a_errors: list[BaseException] = []
        original_preflight = self.manager._preflight
        original_discard = self.manager._discard_attempt

        def failing_preflight(gen):
            if gen == 1:
                entered.set()
                self.assertTrue(release.wait(5))
                raise PreflightError("late preflight failure")
            return original_preflight(gen)

        def delayed_discard(gen, session_dir) -> None:
            if gen == 1 and not discard_entered.is_set():
                discard_entered.set()
                self.assertTrue(allow_discard.wait(5))
            original_discard(gen, session_dir)

        def run_a() -> None:
            try:
                self.manager.start()
            except BaseException as exc:
                a_errors.append(exc)

        with (
            patch.object(
                self.manager, "_preflight", side_effect=failing_preflight
            ),
            patch.object(
                self.manager, "_discard_attempt", side_effect=delayed_discard
            ),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            thread_a = threading.Thread(target=run_a, daemon=True)
            thread_a.start()
            self.assertTrue(entered.wait(2))
            self.manager.stop()
            self.assertTrue(self.manager.wait_idle(2))
            release.set()
            self.assertTrue(discard_entered.wait(2))

            self.manager.start()
            self.assertEqual(self.manager.state_value(), "starting")
            session_dir = Path(self.manager._session_dir or "")
            pointer = self.runner.pointer
            self.assertIsNotNone(pointer)
            assert pointer is not None
            marker = session_dir / "marker.log"
            marker.write_text("new session\n")

            allow_discard.set()
            thread_a.join(5)
            self.assertTrue(self.manager.wait_idle(2))

        self.assertFalse(thread_a.is_alive())
        self.assertEqual(a_errors, [])
        self.assertEqual(self.manager.state_value(), "running")
        self.assertTrue(session_dir.exists())
        self.assertTrue(marker.exists())
        self.assertIsNotNone(self.runner.pointer)
        # Bound once: `pointer` is a property, so two reads are not the same value to a
        # type checker and are not guaranteed to be the same pointer either.
        written = self.runner.pointer
        assert written is not None
        self.assertEqual(written["session"], pointer["session"])
        self.assertEqual(self.manager.snapshot()["session_dir"], str(session_dir))

    def test_stale_preflight_does_not_remove_successor_socket(self) -> None:
        runtime_dir = self.base / "runtime"
        runtime_dir.mkdir()
        self.manager.runtime_dir = runtime_dir
        socket_path = runtime_dir / "monado_comp_ipc"
        entered = threading.Event()
        release = threading.Event()
        a_errors: list[BaseException] = []
        sockets: list[socket.socket] = []
        original_start = self.runner.start
        calls = 0
        calls_lock = threading.Lock()

        def blocking_port(host, port) -> bool:
            nonlocal calls
            with calls_lock:
                calls += 1
                call = calls
            if call == 1:
                entered.set()
                self.assertTrue(release.wait(5))
            return False

        def removing_socket(path) -> bool:
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass
            return True

        def creating_start(name, argv, env, cwd, log_path, stderr_path=None):
            handle = original_start(name, argv, env, cwd, log_path, stderr_path)
            if name == "monado":
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.bind(str(socket_path))
                sockets.append(sock)
            return handle

        def run_a() -> None:
            try:
                self.manager.start()
            except BaseException as exc:
                a_errors.append(exc)

        with (
            patch.object(self.runner, "port_in_use", side_effect=blocking_port),
            patch.object(self.runner, "remove_stale_socket", side_effect=removing_socket),
            patch.object(self.runner, "start", side_effect=creating_start),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            thread_a = threading.Thread(target=run_a, daemon=True)
            thread_a.start()
            self.assertTrue(entered.wait(2))
            self.manager.stop()
            self.assertTrue(self.manager.wait_idle(2))

            self.manager.start()
            self.assertTrue(self.manager.wait_idle(2))
            self.assertEqual(self.manager.state_value(), "running")
            self.assertTrue(socket_path.exists())
            session_dir = Path(self.manager.snapshot()["session_dir"])
            pointer = self.runner.pointer

            release.set()
            thread_a.join(5)

        try:
            self.assertFalse(thread_a.is_alive())
            self.assertTrue(a_errors)
            self.assertIsInstance(a_errors[0], ConflictError)
            self.assertTrue(socket_path.exists())
            self.assertEqual(self.manager.state_value(), "running")
            self.assertTrue(session_dir.exists())
            self.assertEqual(self.runner.pointer, pointer)
            self.assertEqual(self.manager.snapshot()["session_dir"], str(session_dir))
        finally:
            for sock in sockets:
                sock.close()

    def test_stale_monitor_does_not_tear_down_successor(self) -> None:
        self.start()
        old_monado = self.runner.handles["monado"]
        captured = threading.Event()
        release = threading.Event()
        first = True
        first_lock = threading.Lock()
        original_alive = old_monado.alive

        def blocking_alive() -> bool:
            nonlocal first
            with first_lock:
                block = first
                first = False
            if block:
                captured.set()
                self.assertTrue(release.wait(5))
                return False
            return original_alive()

        old_monado.alive = blocking_alive
        old_monitor = self.manager._monitor_thread
        self.assertIsNotNone(old_monitor)
        assert old_monitor is not None
        self.assertTrue(captured.wait(3))

        self.manager.stop()
        self.assertTrue(self.manager.wait_idle(2))
        self.start()

        successor_monado = self.manager._handles["monado"]
        session_dir = Path(self.manager.snapshot()["session_dir"])
        pointer = self.runner.pointer

        release.set()
        old_monitor.join(3)

        self.assertFalse(old_monitor.is_alive())
        self.assertEqual(self.manager.state_value(), "running")
        self.assertIs(self.manager._handles["monado"], successor_monado)
        self.assertTrue(successor_monado.alive())
        self.assertTrue(session_dir.exists())
        self.assertEqual(self.runner.pointer, pointer)
        self.assertEqual(self.manager.snapshot()["session_dir"], str(session_dir))
        self.assertIsNone(self.manager.snapshot()["last_error"])

    def test_stale_worker_removes_abandoned_directory(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        original_start_worker = self.manager._start_worker

        def delayed_first_worker(gen, session_dir) -> None:
            if gen == 1:
                entered.set()
                self.assertTrue(release.wait(5))
            original_start_worker(gen, session_dir)
            if gen == 1:
                finished.set()

        with (
            patch.object(
                self.manager, "_start_worker", side_effect=delayed_first_worker
            ),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            self.manager.start()
            self.assertTrue(entered.wait(2))
            abandoned = self.manager._session_dir
            self.assertIsNotNone(abandoned)
            assert abandoned is not None
            self.assertTrue(abandoned.exists())
            self.manager.stop()
            self.assertTrue(self.manager.wait_idle(2))

            self.manager.start()
            self.assertTrue(self.manager.wait_idle(2))
            self.assertEqual(self.manager.state_value(), "running")
            session_dir = Path(self.manager.snapshot()["session_dir"])
            pointer = self.runner.pointer

            release.set()
            self.assertTrue(finished.wait(2))

        self.assertFalse(abandoned.exists())
        self.assertTrue(session_dir.exists())
        self.assertEqual(self.runner.pointer, pointer)
        self.assertEqual(self.manager.state_value(), "running")
        self.assertEqual(self.manager.snapshot()["session_dir"], str(session_dir))

    def test_stop_start_idle_race_keeps_idle_clear_while_starting(self) -> None:
        self.start()
        entered_b = threading.Event()
        release_b = threading.Event()
        idle_done = threading.Event()
        allow_idle = threading.Event()
        original_preflight = self.manager._preflight
        original_idle_set = self.manager._idle.set

        def blocking_second_preflight(gen):
            if gen == 2:
                entered_b.set()
                self.assertTrue(release_b.wait(5))
            return original_preflight(gen)

        def guarded_idle_set() -> None:
            if not self.manager._lock._is_owned():  # type: ignore[attr-defined]
                self.assertTrue(allow_idle.wait(5))
            original_idle_set()
            idle_done.set()

        with (
            patch.object(
                self.manager, "_preflight", side_effect=blocking_second_preflight
            ),
            patch.object(self.manager._idle, "set", side_effect=guarded_idle_set),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            self.manager.stop()
            self.assertTrue(self.manager.wait_for_state("stopped", 2))
            thread_b = threading.Thread(target=self.manager.start, daemon=True)
            thread_b.start()
            self.assertTrue(entered_b.wait(2))
            self.assertEqual(self.manager.state_value(), "starting")

            allow_idle.set()
            self.assertTrue(idle_done.wait(2))
            self.assertFalse(self.manager.wait_idle(0.2))
            self.assertEqual(self.manager.state_value(), "starting")

            release_b.set()
            thread_b.join(5)

        self.assertTrue(self.manager.wait_idle(2))
        self.assertEqual(self.manager.state_value(), "running")

    def test_discard_logs_directory_removal_failure(self) -> None:
        allow = threading.Event()
        finished = threading.Event()
        original_start_worker = self.manager._start_worker

        def delayed_start_worker(gen, session_dir) -> None:
            self.assertTrue(allow.wait(2))
            original_start_worker(gen, session_dir)
            finished.set()

        with (
            patch.object(self.manager, "_start_worker", side_effect=delayed_start_worker),
            patch(
                "clank.orchestrator.shutil.rmtree",
                side_effect=OSError("rmtree boom"),
            ),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            self.manager.start()
            captured = self.manager._session_dir
            self.assertIsNotNone(captured)
            assert captured is not None
            self.manager.stop()
            allow.set()
            self.assertTrue(finished.wait(2))
            self.assertTrue(self.manager.wait_idle(2))

        text = (captured / "session.log").read_text()
        self.assertIn("session directory removal failed: rmtree boom", text)

    def test_cleanup_pointer_failure_is_not_reported_complete(self) -> None:
        self.start()
        with patch.object(
            self.runner, "remove_pointer", side_effect=RuntimeError("pointer boom")
        ):
            self.manager.stop()
            self.assertTrue(self.manager.wait_idle(2))

        snapshot = self.manager.snapshot()
        text = (Path(snapshot["session_dir"]) / "session.log").read_text()
        self.assertIn("pointer removal failed: pointer boom", text)
        self.assertNotIn("cleanup complete", text)
        self.assertIn("cleanup finished (with", text)

    def test_stop_during_preflight_then_start_does_not_duplicate_processes(
        self,
    ) -> None:
        entered = threading.Event()
        release_preflight = threading.Event()
        worker_entered = threading.Event()
        release_worker = threading.Event()
        worker_finished = threading.Event()
        original_preflight = self.manager._preflight
        original_start_worker = self.manager._start_worker
        original_start = self.runner.start
        starts: list[str] = []

        def blocking_preflight(gen):
            entered.set()
            self.assertTrue(release_preflight.wait(5))
            return original_preflight(gen)

        def delayed_first_worker(gen, session_dir) -> None:
            if gen == 1:
                worker_entered.set()
                self.assertTrue(release_worker.wait(5))
            original_start_worker(gen, session_dir)
            if gen == 1:
                worker_finished.set()

        def recording_start(name, argv, env, cwd, log_path, stderr_path=None):
            starts.append(name)
            return original_start(name, argv, env, cwd, log_path, stderr_path)

        with (
            patch.object(
                self.manager, "_preflight", side_effect=blocking_preflight
            ),
            patch.object(
                self.manager, "_start_worker", side_effect=delayed_first_worker
            ),
            patch.object(self.runner, "start", side_effect=recording_start),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            thread_a = threading.Thread(target=self.manager.start, daemon=True)
            thread_a.start()
            self.assertTrue(entered.wait(2))
            self.manager.stop()
            self.assertTrue(self.manager.wait_idle(2))
            release_preflight.set()
            thread_a.join(5)
            self.assertFalse(thread_a.is_alive())
            self.assertTrue(worker_entered.wait(2))

            self.manager.start()
            self.assertTrue(self.manager.wait_idle(2))
            self.assertEqual(self.manager.state_value(), "running")
            session_dir = Path(self.manager.snapshot()["session_dir"])
            pointer = self.runner.pointer
            self.assertIsNotNone(pointer)

            release_worker.set()
            self.assertTrue(worker_finished.wait(2))

        self.assertEqual(self.manager.state_value(), "running")
        self.assertTrue(session_dir.exists())
        self.assertEqual(self.runner.pointer, pointer)
        self.assertEqual(self.manager.snapshot()["session_dir"], str(session_dir))
        self.assertEqual(
            sorted(starts),
            ["monado", "pose", "proxy", "steam", "weston"],
        )

    def test_startup_failure_log_error_still_cleans_up(self) -> None:
        original_log = self.manager._session_log

        def failing_log(message: str) -> None:
            if message.startswith("startup failed:"):
                raise OSError("log failure")
            original_log(message)

        self.runner.fail_at = "wait_port"
        with (
            patch.object(self.manager, "_session_log", side_effect=failing_log),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            self.manager.start()
            self.assertTrue(self.manager.wait_idle(2))
        snapshot = self.manager.snapshot()
        self.assertIn(snapshot["state"], ("failed", "stopped"))
        self.assertNotEqual(snapshot["state"], "starting")
        self.assertIsNone(self.runner.pointer)
        self.assertEqual(self.manager._handles, {})

    def test_preflight_failure_after_stop_ends_stopped(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def failing_preflight(gen) -> None:
            entered.set()
            self.assertTrue(release.wait(5))
            raise PreflightError("late preflight failure")

        with (
            patch.object(self.manager, "_preflight", side_effect=failing_preflight),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            thread = threading.Thread(target=self.manager.start, daemon=True)
            thread.start()
            self.assertTrue(entered.wait(2))
            self.manager.stop()
            self.assertTrue(self.manager.wait_idle(2))
            release.set()
            thread.join(5)

        self.assertFalse(thread.is_alive())
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["state"], "stopped")
        self.assertIsNone(snapshot["last_error"])
        self.assertIsNone(self.runner.pointer)

    def test_stop_from_failed_clears_last_error(self) -> None:
        self.runner.host_steam = True
        with patch("clank.orchestrator.missing_requirements", return_value=[]):
            with self.assertRaises(PreflightError):
                self.manager.start()
        self.assertEqual(self.manager.snapshot()["state"], "failed")
        self.assertIsNotNone(self.manager.snapshot()["last_error"])

        self.manager.stop()

        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["state"], "stopped")
        self.assertIsNone(snapshot["last_error"])

    def test_stop_from_failed_logs_pointer_removal_failure(self) -> None:
        self.runner.fail_at = "wait_port"
        with patch("clank.orchestrator.missing_requirements", return_value=[]):
            self.manager.start()
            self.assertTrue(self.manager.wait_idle(2))
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["state"], "failed")
        self.assertIsNotNone(snapshot["last_error"])
        session_dir = Path(snapshot["session_dir"])

        with patch.object(
            self.runner, "remove_pointer", side_effect=RuntimeError("pointer boom")
        ):
            self.manager.stop()

        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["state"], "stopped")
        self.assertIsNone(snapshot["last_error"])
        self.assertEqual(self.manager._pointer, {})
        text = (session_dir / "session.log").read_text()
        self.assertIn("pointer removal failed: pointer boom", text)

    def test_superseded_successful_preflight_discards_own_directory(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        a_dirs: list[Path | None] = []
        original_preflight = self.manager._preflight
        calls = 0
        calls_lock = threading.Lock()

        def pausing_preflight(gen):
            nonlocal calls
            with calls_lock:
                calls += 1
                call = calls
            session_dir = original_preflight(gen)
            if call == 1:
                a_dirs.append(session_dir)
                entered.set()
                self.assertTrue(release.wait(5))
            return session_dir

        with (
            patch.object(self.manager, "_preflight", side_effect=pausing_preflight),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            thread_a = threading.Thread(target=self.manager.start, daemon=True)
            thread_a.start()
            self.assertTrue(entered.wait(2))
            self.assertTrue(a_dirs)
            a_dir = a_dirs[0]
            self.assertIsNotNone(a_dir)
            assert a_dir is not None
            self.assertTrue(a_dir.exists())

            self.manager.stop()
            self.assertTrue(self.manager.wait_idle(2))
            self.manager.start()
            self.assertTrue(self.manager.wait_idle(2))
            self.assertEqual(self.manager.state_value(), "running")
            b_dir = Path(self.manager.snapshot()["session_dir"])
            pointer = self.runner.pointer
            marker = b_dir / "marker.log"
            marker.write_text("new session\n")

            release.set()
            thread_a.join(5)

        self.assertFalse(thread_a.is_alive())
        self.assertFalse(a_dir.exists())
        self.assertTrue(b_dir.exists())
        self.assertTrue(marker.exists())
        self.assertEqual(self.runner.pointer, pointer)
        self.assertEqual(self.manager.state_value(), "running")
        self.assertEqual(self.manager.snapshot()["session_dir"], str(b_dir))

    def test_preflight_write_failure_discards_own_directory(self) -> None:
        a_dirs: list[Path] = []

        def failing_write(data):
            a_dirs.append(Path(data["session"]))
            raise RuntimeError("pointer write boom")

        self.manager._start_generation = 1
        self.manager._state = State.STARTING
        self.manager._started_at = object()  # type: ignore[assignment]
        self.manager._socket_name = "clank-old"
        with (
            patch.object(self.runner, "write_pointer", side_effect=failing_write),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            with self.assertRaises(RuntimeError):
                self.manager._preflight(1)

        self.assertEqual(len(a_dirs), 1)
        self.assertFalse(a_dirs[0].exists())
        self.assertIsNone(self.manager._session_id)
        self.assertIsNone(self.manager._session_dir)
        self.assertIsNone(self.manager._started_at)
        self.assertIsNone(self.manager._socket_name)
        self.assertIsNone(self.runner.pointer)
        self.manager._state = State.STOPPED

    def test_preflight_write_failure_after_supersession_keeps_successor(self) -> None:
        a_dirs: list[Path] = []
        b_dir = Path(self.cfg.paths.session_root) / "session-b"
        b_dir.mkdir(parents=True)
        marker = b_dir / "marker.log"
        marker.write_text("new session\n")
        b_pointer = {"session": str(b_dir), "processes": {}}
        b_started = object()

        def superseding_write(data):
            a_dirs.append(Path(data["session"]))
            with self.manager._lock:
                self.manager._start_generation += 1
                self.manager._session_id = "session-b"
                self.manager._session_dir = b_dir
                self.manager._started_at = b_started  # type: ignore[assignment]
                self.manager._socket_name = "clank-b"
            self.runner.pointer = b_pointer
            raise RuntimeError("pointer write boom")

        self.manager._start_generation = 1
        self.manager._state = State.STARTING
        with (
            patch.object(self.runner, "write_pointer", side_effect=superseding_write),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            with self.assertRaises(RuntimeError):
                self.manager._preflight(1)

        self.assertEqual(len(a_dirs), 1)
        self.assertFalse(a_dirs[0].exists())
        self.assertTrue(b_dir.exists())
        self.assertTrue(marker.exists())
        self.assertEqual(self.runner.pointer, b_pointer)
        self.assertEqual(self.manager._session_id, "session-b")
        self.assertEqual(self.manager._session_dir, b_dir)
        self.assertIs(self.manager._started_at, b_started)
        self.assertEqual(self.manager._socket_name, "clank-b")
        self.manager._state = State.STOPPED

    def test_preflight_mkdir_collision_preserves_existing_directory(self) -> None:
        fixed_now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        collision = Path(self.cfg.paths.session_root) / "20260101T000000Z-000000"
        collision.mkdir(parents=True)
        marker = collision / "keep.log"
        marker.write_text("historical session\n")

        with (
            patch("clank.orchestrator.os.urandom", return_value=b"\x00\x00\x00"),
            patch("clank.orchestrator.datetime") as fake_datetime,
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            fake_datetime.now.return_value = fixed_now
            with self.assertRaises(PreflightError):
                self.manager.start()

        self.assertTrue(collision.exists())
        self.assertEqual(marker.read_text(), "historical session\n")
        self.assertIsNone(self.manager._session_dir)
        self.assertEqual(self.manager.state_value(), "failed")

    def test_preflight_pointer_write_failure_discards_directory(self) -> None:
        with (
            patch.object(
                self.runner,
                "write_pointer",
                side_effect=RuntimeError("pointer write boom"),
            ),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            with self.assertRaises(PreflightError) as caught:
                self.manager.start()

        self.assertIn("pointer write boom", str(caught.exception))
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["state"], "failed")
        self.assertIn("pointer write boom", snapshot["last_error"])
        self.assertIsNone(snapshot["session_dir"])
        self.assertIsNone(self.manager._session_dir)
        self.assertIsNone(self.runner.pointer)
        session_root = Path(self.cfg.paths.session_root)
        self.assertEqual(list(session_root.iterdir()) if session_root.exists() else [], [])

    def test_stop_during_preflight_leaves_no_orphan_directories(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        original_preflight = self.manager._preflight

        def blocking_preflight(gen):
            session_dir = original_preflight(gen)
            entered.set()
            self.assertTrue(release.wait(5))
            return session_dir

        with (
            patch.object(self.manager, "_preflight", side_effect=blocking_preflight),
            patch("clank.orchestrator.missing_requirements", return_value=[]),
        ):
            thread = threading.Thread(target=self.manager.start, daemon=True)
            thread.start()
            self.assertTrue(entered.wait(2))
            self.manager.stop()
            self.assertTrue(self.manager.wait_idle(2))
            release.set()
            thread.join(5)
            worker = self.manager._thread
            self.assertIsNotNone(worker)
            assert worker is not None
            worker.join(2)

        self.assertFalse(thread.is_alive())
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["state"], "stopped")
        self.assertIsNone(self.runner.pointer)
        self.assertIsNone(snapshot["session_dir"])
        session_root = Path(self.cfg.paths.session_root)
        self.assertEqual(list(session_root.iterdir()) if session_root.exists() else [], [])

    def test_mode_switch_restarts_pose_only(self) -> None:
        self.start()
        old_pose = self.runner.handles["pose"]
        self.manager.set_mode("yaw")
        self.assertFalse(old_pose.alive())
        self.assertTrue(self.runner.handles["pose"].alive())
        self.assertEqual(self.manager.snapshot()["mode"], "yaw")

    def test_manual_mode_starts_manual_client(self) -> None:
        self.start()
        self.manager.set_mode("manual")
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["mode"], "manual")
        self.assertIn("manual", snapshot)
        argv = self.runner.starts[-1][1]
        self.assertTrue(argv[1].endswith("c0_manual.py"), argv)
        self.assertNotIn("--mode", argv)
        self.assertIn("--input-socket", argv)
        socket_path = Path(argv[argv.index("--input-socket") + 1])
        self.assertEqual(socket_path.name, f"{self.manager._socket_name}-manual.sock")

    def test_manual_mode_waits_for_the_handshake(self) -> None:
        self.start()
        seen: list[str] = []
        original = self.runner.wait_log

        def record(path, needle, timeout, abort=None):
            seen.append(needle)
            return original(path, needle, timeout, abort)

        with patch.object(self.runner, "wait_log", side_effect=record):
            self.manager.set_mode("manual")
        self.assertIn("[manual] handshake ok", seen)

    def test_body_mode_starts_the_pose_relay(self) -> None:
        self.start()
        self.manager.set_mode("body")
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["mode"], "body")
        argv = self.runner.starts[-1][1]
        self.assertTrue(argv[1].endswith("pose_relay.py"), argv)
        self.assertNotIn("--mode", argv)
        state = Path(argv[argv.index("--state") + 1])
        self.assertEqual(state.name, "body_state.json")
        self.assertEqual(state.parent.name, "body")
        self.assertIn("--log", argv)
        # The heading offset must reach the relay even when it is zero, so an
        # operator reading the session log can see which heading was in force.
        self.assertIn("--yaw-offset-deg", argv)
        self.assertEqual(float(argv[argv.index("--yaw-offset-deg") + 1]), 0.0)
        # Undeclared by default: the driver's own field of view stays in force, which
        # is the behaviour every earlier measurement was taken under.
        self.assertNotIn("--fov-h-deg", argv)
        self.assertNotIn("--fov-v-deg", argv)

    def test_body_mode_declares_the_configured_field_of_view(self) -> None:
        self.manager.cfg = replace(
            self.cfg,
            body=replace(self.cfg.body, fov_h_deg=110.0, fov_v_deg=78.0),
        )
        self.start()
        self.manager.set_mode("body")
        argv = self.runner.starts[-1][1]
        self.assertEqual(float(argv[argv.index("--fov-h-deg") + 1]), 110.0)
        self.assertEqual(float(argv[argv.index("--fov-v-deg") + 1]), 78.0)

    def test_body_mode_leaves_the_player_unscaled_by_default(self) -> None:
        # Undeclared by default, and the relay writes its declaration either way so a
        # reader can tell "unscaled" from "nobody said".
        self.start()
        self.manager.set_mode("body")
        argv = self.runner.starts[-1][1]
        self.assertNotIn("--player-scale", argv)
        self.assertNotIn("--player-pivot", argv)
        self.assertNotIn("--player-scale-head", argv)

    def test_body_mode_declares_the_configured_player_scale(self) -> None:
        self.manager.cfg = replace(
            self.cfg,
            body=replace(self.cfg.body, player_scale=1.43, player_pivot=(0.1, -0.2, 0.0),
                         player_scale_head=False),
        )
        self.start()
        self.manager.set_mode("body")
        argv = self.runner.starts[-1][1]
        self.assertEqual(argv[argv.index("--player-scale") + 1], "1.43")
        self.assertEqual(argv[argv.index("--player-pivot") + 1], "0.1,-0.2,0")
        self.assertEqual(argv[argv.index("--player-scale-head") + 1], "off")

    def test_manual_mode_declares_the_configured_field_of_view(self) -> None:
        # The frustum is a property of the session, not the mode: switching into
        # body after the game connected must not change the view configuration,
        # or the client keeps rendering against swapchains built for the old one.
        self.manager.cfg = replace(
            self.cfg,
            body=replace(self.cfg.body, fov_h_deg=110.0, fov_v_deg=78.0),
        )
        self.start()
        self.manager.set_mode("manual")
        argv = self.runner.starts[-1][1]
        self.assertEqual(float(argv[argv.index("--fov-h-deg") + 1]), 110.0)
        self.assertEqual(float(argv[argv.index("--fov-v-deg") + 1]), 78.0)

    def test_scripted_mode_declares_the_configured_field_of_view(self) -> None:
        self.manager.cfg = replace(
            self.cfg,
            body=replace(self.cfg.body, fov_h_deg=110.0, fov_v_deg=78.0),
        )
        self.start()
        self.manager.set_mode("hold")
        argv = self.runner.starts[-1][1]
        self.assertEqual(float(argv[argv.index("--fov-h-deg") + 1]), 110.0)
        self.assertEqual(float(argv[argv.index("--fov-v-deg") + 1]), 78.0)

    def test_manual_input_requires_manual_mode(self) -> None:
        self.start()
        with self.assertRaisesRegex(ConflictError, "manual"):
            self.manager.manual_input({"client": "browser-1234", "armed": False})

    def test_manual_input_forwards_to_manual_socket(self) -> None:
        self.start()
        self.manager.set_mode("manual")
        argv = self.runner.starts[-1][1]
        socket_path = Path(argv[argv.index("--input-socket") + 1])
        receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        receiver.bind(str(socket_path))
        receiver.settimeout(1)
        try:
            self.assertTrue(
                self.manager.manual_input(
                    {
                        "client": "browser-1234",
                        "armed": True,
                        "keys": ["w"],
                        "buttons": [],
                        "dx": 2,
                        "dy": -1,
                        "wheel": 0,
                    }
                )
            )
            payload = json.loads(receiver.recv(65536))
            self.assertEqual(payload["keys"], ["w"])
            self.assertTrue(self.manager.snapshot()["manual"]["armed"])
        finally:
            receiver.close()
            socket_path.unlink(missing_ok=True)

    def test_manual_mode_switches_back_to_scripted_client(self) -> None:
        self.start()
        self.manager.set_mode("manual")
        self.manager.set_mode("hold")
        self.assertEqual(self.manager.snapshot()["mode"], "hold")
        argv = self.runner.starts[-1][1]
        self.assertTrue(argv[1].endswith("c0_client.py"), argv)
        self.assertIn("--mode", argv)

    def test_monado_death_fails_and_cleans_up(self) -> None:
        self.start()
        self.runner.handles["monado"]._alive = False
        self.assertTrue(self.manager.wait_for_state("failed", 3))
        self.assertIn("monado", self.manager.snapshot()["last_error"])
        self.assertIsNone(self.runner.pointer)
        self.assertTrue(all(not handle.alive() for handle in self.runner.handles.values()))

    def test_proxy_death_fails_and_cleans_up(self) -> None:
        self.start()
        self.runner.handles["proxy"]._alive = False
        self.assertTrue(self.manager.wait_for_state("failed", 3))
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["state"], "failed")
        self.assertIn("proxy exited while running", snapshot["last_error"])
        self.assertIsNone(self.runner.pointer)
        self.assertFalse(snapshot["processes"]["game"]["alive"])
        self.assertFalse(self.runner.game_started)
        self.assertEqual(snapshot["survivors"], [])
        self.assertNotIn("proxy", snapshot["survivors"])
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

    def test_parser_handles_final_line_without_newline(self) -> None:
        log = (
            " INFO [client_connected] Client 1 connected\n"
            "\tapplication_name: 'Unity Application'"
        )
        self.assertEqual(parse_openxr_client(log), "Unity Application")

    def test_incremental_parser_reports_unterminated_application_line(self) -> None:
        parser = _OpenXRLogParser()
        parser.feed(" INFO [client_connected] Client 1 connected\n")
        self.assertIsNone(parser.value())
        parser.feed("\tapplication_name: 'X'")
        self.assertEqual(parser.value(), "X")
        parser.feed("\n")
        self.assertEqual(parser.value(), "X")
        parser.feed(" INFO [client_disconnected] Client 1 disconnected\n")
        self.assertIsNone(parser.value())

    def test_incremental_parser_caps_unterminated_buffer(self) -> None:
        parser = _OpenXRLogParser()
        parser.feed("x" * (1024 * 1024 + 128))
        self.assertLessEqual(len(parser._buffer), 1024 * 1024)
        parser.feed(
            "\n INFO [client_connected] Client 1 connected\n"
            "\tapplication_name: 'Unity Application'\n"
        )
        self.assertEqual(parser.value(), "Unity Application")


if __name__ == "__main__":
    unittest.main()


class SaveWipeTests(unittest.TestCase):
    """A wipe is the only thing here that deletes files, so it gets its own harness."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        cfg = load_config(None, env={})
        cfg = replace(
            cfg,
            paths=replace(cfg.paths, session_root=str(self.base / "sessions")),
        )
        self.cfg = cfg
        self.prefix = self.base / "pfx"
        self.prefix.mkdir()
        self.manager = SessionManager(cfg, runner=FakeRunner())  # type: ignore[arg-type]
        patcher = patch("clank.orchestrator.proton_prefix", return_value=self.prefix)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.save = (
            self.prefix
            / "drive_c/users/steamuser/AppData/LocalLow/Owlchemy Labs/Job Simulator/User"
        )
        self.save.mkdir(parents=True)
        (self.save / "JobSimulatorSave.save.bak-002050").write_bytes(b"x" * 40)
        (self.save.parent / "output_log.txt").write_text("log")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_proton_prefix_is_derived_from_the_config(self) -> None:
        expected = (
            Path.home()
            / ".local/share/Steam/steamapps/compatdata"
            / self.cfg.launch.proton_app_id
            / "pfx"
        )
        self.assertEqual(proton_prefix(self.cfg), expected)

    def test_wipe_removes_the_save_but_not_the_log(self) -> None:
        self.assertEqual(self.manager.wipe_game_saves(), {"files": 1, "bytes": 40})
        self.assertTrue(self.save.is_dir())
        self.assertEqual(list(self.save.iterdir()), [])
        self.assertTrue((self.save.parent / "output_log.txt").exists())

    def test_wipe_tolerates_a_missing_dir(self) -> None:
        shutil.rmtree(self.prefix / "drive_c")
        self.assertEqual(self.manager.wipe_game_saves(), {"files": 0, "bytes": 0})

    def test_wipe_clears_a_second_configured_dir(self) -> None:
        cache = self.prefix / "drive_c/users/steamuser/AppData/LocalLow/Unity/Owlchemy Labs_Job Simulator"
        cache.mkdir(parents=True)
        (cache / "__data").write_bytes(b"y" * 10)
        self.assertEqual(self.manager.wipe_game_saves(), {"files": 2, "bytes": 50})

    def test_wipe_refuses_a_symlink_out_of_the_prefix(self) -> None:
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "keep").write_text("keep")
        (self.prefix / "drive_c/escape").symlink_to(outside)
        cfg = replace(
            self.cfg, paths=replace(self.cfg.paths, game_save_dirs=("drive_c/escape",))
        )
        manager = SessionManager(cfg, runner=FakeRunner())  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "outside the Proton prefix"):
            manager.wipe_game_saves()
        self.assertTrue((outside / "keep").exists())


class RestartGameTests(SessionHarness):
    def test_restart_requires_a_running_stack(self) -> None:
        with self.assertRaises(ConflictError):
            self.manager.restart_game()

    def test_restart_resets_the_openxr_parser_before_waiting(self) -> None:
        self.start()
        self.manager._openxr_parser.feed(
            "[client_connected] Client 3 connected\napplication_name: 'OpenComposite_wine'\n"
        )
        self.assertEqual(self.manager._openxr_parser.value(), "OpenComposite_wine")
        seen = []
        original = self.runner.wait_log

        def record(path, needle, timeout, abort=None):
            seen.append(self.manager._openxr_parser.value())
            return original(path, needle, timeout, abort)

        self.runner.wait_log = record
        self.manager.restart_game()
        self.assertEqual(seen, [None])

    def test_restart_kills_then_wipes_then_relaunches(self) -> None:
        self.start()
        order = []

        def kill(pid):
            order.append("kill")
            self.runner.game_started = False

        original = self.runner.wait_pids

        def wait(argv0, timeout, abort=None):
            order.append("rungameid")
            return original(argv0, timeout, abort)

        self.runner.kill_pid = kill
        self.runner.wait_pids = wait
        self.manager.wipe_game_saves = lambda: order.append("wipe") or {"files": 0, "bytes": 0}
        self.manager.restart_game()
        self.assertEqual(order, ["kill", "wipe", "rungameid"])

    def test_restart_failure_leaves_the_stack_running(self) -> None:
        self.start()
        self.runner.fail_at = "wait_pids"
        with self.assertRaises(StartupError):
            self.manager.restart_game()
        self.assertEqual(self.manager.state_value(), "running")
