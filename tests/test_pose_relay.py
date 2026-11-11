# Re-indexed: forced_pair, ClickEdge and buttons_from_state live in
# clank/pose_relay.py; run() takes force_grab/force_trigger/grab_mode.
from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

from clank import pose_relay, wire


def sample_state(root_p=(0.0, 0.0, 0.79), head_p=(-0.001, 0.02, 0.40)) -> dict:
    return {
        "root_p": list(root_p),
        "root_q": [1.0, 0.0, 0.0, 0.0],
        "head": {"p": list(head_p), "q_xyzw": [0.0, 0.0, 0.0, 1.0]},
        "left_hand": {"p": [0.12, 0.26, -0.23], "q_xyzw": [0.0, 0.0, 0.0, 1.0]},
        "right_hand": {"p": [0.16, -0.25, -0.21], "q_xyzw": [0.0, 0.0, 0.0, 1.0]},
    }


class StateSourceTests(unittest.TestCase):
    def test_missing_file_is_not_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source = pose_relay.StateSource(Path(raw) / "nope.json")
            state, age = source.read(1.0)
        self.assertIsNone(state)
        self.assertIsNone(age)

    def test_reads_and_reports_age(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "body_state.json"
            path.write_text(json.dumps(sample_state()), encoding="utf-8")
            source = pose_relay.StateSource(path)
            state, age = source.read(10.0)
            self.assertIsNotNone(state)
            self.assertIsNotNone(age)
            self.assertEqual(age, 0.0)
            _, later = source.read(10.5)
            self.assertAlmostEqual(later or -1, 0.5)

    def test_new_content_replaces_the_cached_document(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "body_state.json"
            path.write_text(json.dumps(sample_state(root_p=(1.0, 0.0, 0.79))), encoding="utf-8")
            source = pose_relay.StateSource(path)
            first, _ = source.read(1.0)
            assert first is not None
            path.write_text(json.dumps(sample_state(root_p=(2.0, 0.0, 0.79))), encoding="utf-8")
            second, age = source.read(2.0)
            assert second is not None
            self.assertEqual(first["root_p"][0], 1.0)
            self.assertEqual(second["root_p"][0], 2.0)
            self.assertIsNotNone(age)
            self.assertEqual(age, 0.0)

    def test_broken_json_keeps_the_previous_state(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "body_state.json"
            path.write_text(json.dumps(sample_state()), encoding="utf-8")
            source = pose_relay.StateSource(path)
            source.read(1.0)
            path.write_text("{not json", encoding="utf-8")
            state, _ = source.read(2.0)
        self.assertIsNotNone(state)


class PacketSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "body_state.json"
        self.source = pose_relay.PacketSource(pose_relay.StateSource(self.path), stale_after=0.3)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_rest_pose_before_any_state_exists(self) -> None:
        poses, note, age = self.source.poses(1.0)
        self.assertEqual(note, "rest")
        self.assertIsNone(age)
        self.assertEqual(poses["head"][0], wire.REST_POSES["head"][0])

    def test_live_state_is_converted(self) -> None:
        self.path.write_text(json.dumps(sample_state()), encoding="utf-8")
        poses, note, age = self.source.poses(1.0)
        self.assertEqual(note, "live")
        self.assertIsNotNone(age)
        self.assertEqual(age, 0.0)
        # head world z = 0.79 + 0.40 -> xr_y
        self.assertAlmostEqual(poses["head"][0][1], 1.19, places=6)

    def test_state_goes_stale_and_the_last_pose_is_held(self) -> None:
        self.path.write_text(json.dumps(sample_state()), encoding="utf-8")
        live, _, _ = self.source.poses(1.0)
        held, note, age = self.source.poses(1.5)
        self.assertEqual(note, "stale")
        self.assertAlmostEqual(age or -1, 0.5)
        self.assertEqual(held, live)

    def test_malformed_state_keeps_the_last_good_pose(self) -> None:
        self.path.write_text(json.dumps(sample_state()), encoding="utf-8")
        live, _, _ = self.source.poses(1.0)
        self.path.write_text(json.dumps({"root_p": [0, 0, 0]}), encoding="utf-8")
        held, note, _ = self.source.poses(2.0)
        self.assertEqual(note, "malformed")
        self.assertEqual(held, live)


class HandshakeTests(unittest.TestCase):
    def test_accepts_two_packets(self) -> None:
        a, b = socket.socketpair()
        try:
            payload = bytes(wire.blank_packet()) * 2
            threading.Thread(target=lambda: b.sendall(payload), daemon=True).start()
            pose_relay.read_handshake(a)
        finally:
            a.close()
            b.close()

    def test_rejects_a_wrong_header(self) -> None:
        a, b = socket.socketpair()
        try:
            bad = bytes(wire.SIZE * 2)
            threading.Thread(target=lambda: b.sendall(bad), daemon=True).start()
            with self.assertRaises(ValueError):
                pose_relay.read_handshake(a)
        finally:
            a.close()
            b.close()

    def test_reports_a_closed_peer(self) -> None:
        a, b = socket.socketpair()
        try:
            b.close()
            with self.assertRaises(ConnectionError):
                pose_relay.read_handshake(a)
        finally:
            a.close()


class ConnectTests(unittest.TestCase):
    def _listener(self) -> socket.socket:
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        self.addCleanup(listener.close)
        return listener

    def test_retries_while_the_driver_is_busy_with_another_client(self) -> None:
        listener = self._listener()
        port = listener.getsockname()[1]

        def serve() -> None:
            first, _ = listener.accept()
            try:
                time.sleep(0.30)
            finally:
                first.close()
            second, _ = listener.accept()
            try:
                second.sendall(bytes(wire.blank_packet()) * 2)
                time.sleep(0.20)
            finally:
                second.close()

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        sock = pose_relay.connect(
            "127.0.0.1", port, timeout=2.0, handshake_timeout=0.10
        )
        try:
            self.assertIsNotNone(sock)
        finally:
            sock.close()
        thread.join(2)

    def test_gives_up_when_no_handshake_arrives(self) -> None:
        listener = self._listener()
        port = listener.getsockname()[1]
        release = threading.Event()

        def serve() -> None:
            conn, _ = listener.accept()
            try:
                release.wait(2)
            finally:
                conn.close()

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        start = time.monotonic()
        try:
            with self.assertRaises(OSError):
                pose_relay.connect(
                    "127.0.0.1", port, timeout=0.35, handshake_timeout=0.10
                )
            self.assertLess(time.monotonic() - start, 1.5)
        finally:
            release.set()
            thread.join(2)


class DriverButtonsTests(unittest.TestCase):
    def test_forced_pair_is_per_hand(self) -> None:
        self.assertEqual(pose_relay.forced_pair("left"), (1.0, 0.0))
        self.assertEqual(pose_relay.forced_pair("right"), (0.0, 1.0))
        self.assertEqual(pose_relay.forced_pair("both"), (1.0, 1.0))
        self.assertEqual(pose_relay.forced_pair("none"), (0.0, 0.0))

    def test_click_lasts_long_enough_to_survive_a_slow_game_frame(self) -> None:
        # Measured: one click packet in 30. Monado latches only the newest packet,
        # so a single 20 ms window can be overwritten before the game samples it.
        edge = pose_relay.ClickEdge(hold_packets=2)
        self.assertEqual(edge.update((1.0, 0.0)), (True, False), "rising edge")
        self.assertEqual(edge.update((1.0, 0.0)), (True, False), "still held for one more")
        self.assertEqual(edge.update((1.0, 0.0)), (False, False), "then released")

    def test_click_is_the_rising_edge_only(self) -> None:
        # hold_packets=1 isolates the edge from the click's width (see the test below).
        edge = pose_relay.ClickEdge(hold_packets=1)
        self.assertEqual(edge.update((1.0, 0.0)), (True, False), "first press clicks")
        self.assertEqual(edge.update((1.0, 0.0)), (False, False), "holding does not re-click")
        self.assertEqual(edge.update((0.0, 0.0)), (False, False))
        self.assertEqual(edge.update((1.0, 0.0)), (True, False), "re-press clicks again")

    def test_state_without_buttons_still_sends_zero(self) -> None:
        grip, fire = pose_relay.buttons_from_state(sample_state())
        self.assertEqual((grip, fire), ((0.0, 0.0), (0.0, 0.0)))

    def test_buttons_from_a_state_with_nonsense_fields(self) -> None:
        grip, fire = pose_relay.buttons_from_state({"closure": "x", "trigger": [1.0]})
        self.assertEqual((grip, fire), ((0.0, 0.0), (0.0, 0.0)))


class FakeDriver(threading.Thread):
    """Stands in for the Monado remote driver: handshake, then count packets."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.port = self.listener.getsockname()[1]
        self.packets = 0
        self.closed = threading.Event()

    def run(self) -> None:
        conn, _ = self.listener.accept()
        try:
            conn.sendall(bytes(wire.blank_packet()))
            conn.sendall(bytes(wire.blank_packet()))
            buf = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
                self.packets += len(buf) // wire.SIZE
                buf = buf[len(buf) // wire.SIZE * wire.SIZE:]
        finally:
            conn.close()
            self.listener.close()
            self.closed.set()


class RecordingDriver(FakeDriver):
    """FakeDriver that also remembers the last packet's per-hand button values."""

    def __init__(self) -> None:
        super().__init__()
        self.last_squeeze = (0.0, 0.0)
        self.last_trigger = (0.0, 0.0)
        self.last_system = (False, False)
        self.system_seen = False

    def run(self) -> None:
        conn, _ = self.listener.accept()
        try:
            conn.sendall(bytes(wire.blank_packet()))
            conn.sendall(bytes(wire.blank_packet()))
            buf = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while len(buf) >= wire.SIZE:
                    packet = wire.RemoteData.from_buffer_copy(buf[: wire.SIZE])
                    buf = buf[wire.SIZE:]
                    self.packets += 1
                    self.last_squeeze = (
                        packet.left.squeeze_value.x,
                        packet.right.squeeze_value.x,
                    )
                    self.last_trigger = (
                        packet.left.trigger_value.x,
                        packet.right.trigger_value.x,
                    )
                    self.last_system = (packet.left.system_click, packet.right.system_click)
                    self.system_seen = self.system_seen or any(self.last_system)
        finally:
            conn.close()
            self.listener.close()
            self.closed.set()


class RelayRunTests(unittest.TestCase):
    def test_sends_packets_at_the_requested_rate(self) -> None:
        driver = FakeDriver()
        driver.start()
        with tempfile.TemporaryDirectory() as raw:
            state_path = Path(raw) / "body_state.json"
            state_path.write_text(json.dumps(sample_state()), encoding="utf-8")
            log_path = Path(raw) / "poses.csv"
            code = pose_relay.run("127.0.0.1", driver.port, state_path, log_path,
                                  duration=0.6, hz=50.0)
            driver.closed.wait(5)
            rows = list(csv.DictReader(log_path.read_text().splitlines()))
        self.assertEqual(code, 0)
        self.assertGreaterEqual(driver.packets, 15)
        self.assertGreaterEqual(len(rows), 15)
        self.assertEqual(rows[0]["note"], "live")
        self.assertGreater(float(rows[0]["hy"]), 1.0)  # head above the hands

    def test_forced_grab_reaches_the_packet_and_the_log(self) -> None:
        """A grab that misses must still be readable as a grab that was sent."""
        driver = RecordingDriver()
        driver.start()
        with tempfile.TemporaryDirectory() as raw:
            state_path = Path(raw) / "body_state.json"
            state_path.write_text(json.dumps(sample_state()), encoding="utf-8")
            log_path = Path(raw) / "poses.csv"
            pose_relay.run("127.0.0.1", driver.port, state_path, log_path,
                           duration=0.3, hz=50.0, force_grab="right",
                           force_trigger="left")
            driver.closed.wait(5)
            rows = list(csv.DictReader(log_path.read_text().splitlines()))
        self.assertAlmostEqual(driver.last_squeeze[0], 0.0)
        self.assertAlmostEqual(driver.last_squeeze[1], 1.0)
        self.assertAlmostEqual(driver.last_trigger[0], 1.0)
        self.assertAlmostEqual(driver.last_trigger[1], 0.0)
        self.assertEqual(rows[-1]["grip_r"], "1.0")
        self.assertEqual(rows[-1]["fire_l"], "1.0")

    def test_log_records_the_masked_field_not_the_request(self) -> None:
        # Under --grab-mode curl the wire carries squeeze 0, so logging the raw
        # request would claim a grip the game never saw.
        driver = RecordingDriver()
        driver.start()
        with tempfile.TemporaryDirectory() as raw:
            state_path = Path(raw) / "body_state.json"
            state_path.write_text(json.dumps(sample_state()), encoding="utf-8")
            log_path = Path(raw) / "poses.csv"
            pose_relay.run("127.0.0.1", driver.port, state_path, log_path,
                           duration=0.3, hz=50.0, force_grab="right", grab_mode="curl")
            driver.closed.wait(5)
            rows = list(csv.DictReader(log_path.read_text().splitlines()))
        self.assertEqual(rows[-1]["grip_r"], "0.0", "squeeze was masked off")
        self.assertEqual(rows[-1]["curl_r"], "1.0", "the curl is what was sent")
        self.assertAlmostEqual(driver.last_squeeze[1], 0.0)

    def test_state_buttons_are_forwarded_when_not_forced(self) -> None:
        driver = RecordingDriver()
        driver.start()
        with tempfile.TemporaryDirectory() as raw:
            state_path = Path(raw) / "body_state.json"
            pressed = {**sample_state(), "closure": [1.0, 0.0]}
            state_path.write_text(json.dumps(pressed), encoding="utf-8")
            pose_relay.run("127.0.0.1", driver.port, state_path, None,
                           duration=0.3, hz=50.0)
            driver.closed.wait(5)
        self.assertAlmostEqual(driver.last_squeeze[0], 1.0)
        self.assertAlmostEqual(driver.last_squeeze[1], 0.0)
        self.assertAlmostEqual(driver.last_trigger[0], 0.0)

    def test_log_is_readable_while_the_relay_runs(self) -> None:
        # A relay that runs for an hour must not hold its evidence in a buffer:
        # the CSV is the only record of what was actually sent.
        driver = FakeDriver()
        driver.start()
        header_bytes = len(",".join(pose_relay.LOG_HEADER))
        with tempfile.TemporaryDirectory() as raw:
            state_path = Path(raw) / "body_state.json"
            state_path.write_text(json.dumps(sample_state()), encoding="utf-8")
            log_path = Path(raw) / "poses.csv"
            threading.Thread(
                target=pose_relay.run,
                args=("127.0.0.1", driver.port, state_path, log_path, 3.0, 50.0),
                daemon=True,
            ).start()
            deadline = time.monotonic() + 8.0
            rows = 0
            while time.monotonic() < deadline:
                time.sleep(0.2)
                if log_path.exists() and log_path.stat().st_size > header_bytes + 20:
                    rows = len(log_path.read_text().splitlines())
                    break
        self.assertGreater(rows, 1, "the CSV stayed empty while the relay was running")

    def test_exits_cleanly_when_the_driver_hangs_up(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state_path = Path(raw) / "body_state.json"
            state_path.write_text(json.dumps(sample_state()), encoding="utf-8")
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]

            def serve() -> None:
                conn, _ = listener.accept()
                conn.sendall(bytes(wire.blank_packet()) * 2)
                conn.recv(wire.SIZE)
                conn.close()
                listener.close()

            threading.Thread(target=serve, daemon=True).start()
            code = pose_relay.run("127.0.0.1", port, state_path, None, duration=5.0, hz=50.0)
        self.assertEqual(code, 1)


class YawOffsetTests(unittest.TestCase):
    """The robot's heading in the virtual room, at the relay's boundary.

    The desk needs a 90 deg left turn and the body cannot twist that far, so the
    heading is chosen rather than walked. What matters here is that the relay is the
    *only* place it is applied - applying it twice, or to the head alone, would put
    the hands somewhere no camera can see them.
    """

    def _source(self, **kw):
        from clank.pose_relay import PacketSource, StateSource

        return PacketSource(StateSource(Path("/nonexistent/state.json")), **kw)

    def _state_file(self) -> Path:
        import json
        import tempfile

        payload = {
            "root_p": [0.0, 0.0, 0.8],
            "root_q": [1.0, 0.0, 0.0, 0.0],
            "head": {"p": [0.0, 0.0, 0.4], "q_xyzw": [0.0, 0.0, 0.0, 1.0]},
            "left_hand": {"p": [0.0, 0.2, -0.2], "q_xyzw": [0.0, 0.0, 0.0, 1.0]},
            "right_hand": {"p": [0.0, -0.2, -0.2], "q_xyzw": [0.0, 0.0, 0.0, 1.0]},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
        return Path(handle.name)

    def test_the_default_offset_leaves_poses_untouched(self) -> None:
        from clank.pose_relay import PacketSource, StateSource

        path = self._state_file()
        plain = PacketSource(StateSource(path)).poses(0.0)[0]
        zero = PacketSource(StateSource(path), yaw_offset_rad=0.0).poses(0.0)[0]
        self.assertEqual(plain, zero)

    def test_a_turned_relay_reports_a_different_heading(self) -> None:
        from clank.pose_relay import PacketSource, StateSource

        path = self._state_file()
        plain = PacketSource(StateSource(path)).poses(0.0)[0]
        turned = PacketSource(
            StateSource(path), yaw_offset_rad=math.radians(90.0)
        ).poses(0.0)[0]
        self.assertNotEqual(plain["head"], turned["head"])
        # The hand must move with the head, or the view and the arms disagree.
        self.assertNotEqual(plain["right"], turned["right"])

    def test_a_non_finite_offset_is_refused(self) -> None:
        # A nan heading would emit nan poses and look like a body fault.
        with self.assertRaises(ValueError):
            self._source(yaw_offset_rad=float("nan"))


class YawOffsetCli(unittest.TestCase):
    """`--yaw-offset-deg` from the command line to `run`.

    This exists because the flag was once added to the orchestrator's argv but not to
    the relay's parser - an edit that replaced a line with itself and looked applied.
    Tests passed; body mode would have failed to start in the field. So the check here
    is that the flag is *parsed and forwarded*, not merely that argv mentions it.
    """

    def _capture(self, argv: list[str], *, stderr: bool = False) -> dict:
        from clank import pose_relay

        seen: dict = {}

        def fake_run(host, port, state, log, duration, hz, **kw):
            seen["yaw_offset_rad"] = kw.get("yaw_offset_rad")
            return 0

        original = pose_relay.run
        pose_relay.run = fake_run
        try:
            pose_relay.main(argv)
        finally:
            pose_relay.run = original
        return seen

    def test_the_orchestrators_argv_is_accepted_and_converted(self) -> None:
        # The exact shape the orchestrator builds, so the two sides cannot drift again.
        seen = self._capture(
            ["--state", "/nonexistent/body_state.json", "--duration", "3600",
             "--log", "/nonexistent/body_pose.csv", "--yaw-offset-deg", "90.0"]
        )
        self.assertAlmostEqual(seen["yaw_offset_rad"], math.pi / 2, places=9)

    def test_ninety_degrees_is_the_robots_left(self) -> None:
        # Sign convention: positive is the robot's left, matching every other yaw here.
        seen = self._capture(["--state", "/nonexistent/s.json", "--yaw-offset-deg", "-45"])
        self.assertAlmostEqual(seen["yaw_offset_rad"], -math.pi / 4, places=9)

    def test_the_default_is_zero(self) -> None:
        seen = self._capture(["--state", "/nonexistent/s.json"])
        self.assertEqual(seen["yaw_offset_rad"], 0.0)

    def test_a_non_finite_offset_is_refused_by_name(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self._capture(["--state", "/nonexistent/s.json", "--yaw-offset-deg", "nan"])
        self.assertNotEqual(caught.exception.code, 0)



class DeclaredFovCli(unittest.TestCase):
    """Widening the view: `--fov-h-deg` / `--fov-v-deg` to `run`.

    Nothing declares a field of view by default, so `per_view_data_valid` is False and
    the remote driver falls back to its own default - which is why the preview looks
    narrow. Declaring one is the knob; whether this build honours it is the experiment.

    Args and config are FULL angles, because 110 deg is how anyone describes a headset.
    `wire.fill` wants half-angles in radians, and the halving happens exactly once, in
    main().
    """

    def _capture(self, argv: list[str], *, sink: dict | None = None) -> dict:
        """Run main() with `run` stubbed out; never opens a socket or a file."""
        from clank import pose_relay

        seen: dict = {}

        def fake_run(host, port, state, log, duration, hz, **kw):
            seen["fov_rad"] = kw.get("fov_rad")
            return 0

        original = pose_relay.run
        pose_relay.run = fake_run
        buf = io.StringIO()
        try:
            with contextlib.redirect_stderr(buf):
                pose_relay.main(argv)
        finally:
            pose_relay.run = original
            # Written to the caller's sink before any SystemExit propagates, or the
            # assertion would read an unbound name instead of a message.
            if sink is not None:
                sink["stderr"] = buf.getvalue()
        return seen

    def test_a_declared_fov_arrives_as_half_angle_radians(self) -> None:
        seen = self._capture(
            ["--state", "/nonexistent/s.json", "--fov-h-deg", "110", "--fov-v-deg", "78"]
        )
        half_h, half_v = seen["fov_rad"]
        self.assertAlmostEqual(half_h, math.radians(55.0), places=9)
        self.assertAlmostEqual(half_v, math.radians(39.0), places=9)

    def test_undeclared_stays_undeclared(self) -> None:
        # None, not (0, 0): a zero angle would be a declared FOV of nothing.
        self.assertIsNone(self._capture(["--state", "/nonexistent/s.json"])["fov_rad"])

    def test_half_a_field_of_view_is_refused(self) -> None:
        # One angle alone would need an invented aspect ratio, and this module's whole
        # discipline is not inventing numbers.
        sink: dict = {}
        with self.assertRaises(SystemExit) as caught:
            self._capture(["--state", "/nonexistent/s.json", "--fov-h-deg", "110"], sink=sink)
        self.assertNotEqual(caught.exception.code, 0)
        self.assertIn("both", sink["stderr"])

    def test_an_impossible_angle_is_refused(self) -> None:
        sink: dict = {}
        with self.assertRaises(SystemExit) as caught:
            self._capture(
                ["--state", "/nonexistent/s.json",
                 "--fov-h-deg", "200", "--fov-v-deg", "78"], sink=sink
            )
        self.assertNotEqual(caught.exception.code, 0)
        self.assertIn("between 0 and 180", sink["stderr"])

    def test_a_negative_angle_is_refused(self) -> None:
        sink: dict = {}
        with self.assertRaises(SystemExit) as caught:
            self._capture(
                ["--state", "/nonexistent/s.json",
                 "--fov-h-deg", "-110", "--fov-v-deg", "78"], sink=sink
            )
        self.assertNotEqual(caught.exception.code, 0)
        self.assertIn("between 0 and 180", sink["stderr"])



class DurationTests(unittest.TestCase):
    """A bounded relay must not exit quietly.

    The measured failure this guards against: the relay hit its duration, exited
    cleanly, and the game held a frozen pose for 17 minutes while the dashboard
    still reported `state: running, last_error: None`. Nothing in the stack
    noticed, because the only thing that knew was a process that had already
    succeeded at everything it had been asked to do.
    """

    def test_zero_and_negative_mean_run_until_killed(self) -> None:
        for d in (0.0, -1.0, -3600.0):
            self.assertTrue(pose_relay.unlimited(d), d)
            self.assertEqual(pose_relay.deadline_state(1e9, d), "unlimited")

    def test_nan_is_treated_as_unlimited_not_as_expired(self) -> None:
        """A NaN duration must not compare its way into an instant exit."""
        self.assertTrue(pose_relay.unlimited(float("nan")))

    def test_a_bounded_run_expires_at_its_duration(self) -> None:
        self.assertEqual(pose_relay.deadline_state(3599.0, 3600.0), "warning")
        self.assertEqual(pose_relay.deadline_state(3600.0, 3600.0), "expired")
        self.assertEqual(pose_relay.deadline_state(3601.0, 3600.0), "expired")

    def test_it_warns_before_it_expires(self) -> None:
        self.assertEqual(pose_relay.deadline_state(100.0, 3600.0), "running")
        self.assertEqual(
            pose_relay.deadline_state(3600.0 - pose_relay.EXPIRY_WARN_S + 1.0, 3600.0),
            "warning",
        )

    def test_the_expiry_note_names_the_consequence(self) -> None:
        """Not "done": the reader has to learn the game is now frozen."""
        note = pose_relay.EXPIRY_NOTE
        self.assertIn("hold the last pose", note)
        self.assertIn("relay", note)
        self.assertGreater(len(note), 60)


if __name__ == "__main__":
    unittest.main()


class PlayerScaleTests(unittest.TestCase):
    """The relay must carry the scale through to the packets, not just accept it.

    A scale that is validated at the door and then dropped would look exactly like a
    working system: every pose would still be plausible, just adult-sized or not, and the
    only symptom would be distances that do not match the reach measurements.
    """

    def _source(self, scale: float):
        tmp = tempfile.mkdtemp()
        path = Path(tmp) / "body_state.json"
        path.write_text(
            json.dumps(
                {
                    "root_p": [0.0, 0.0, 0.787],
                    "root_q": [1.0, 0.0, 0.0, 0.0],
                    "head": {"p": [0.0, 0.0, 0.403], "q_xyzw": [0.0, 0.0, 0.0, 1.0]},
                    "left_hand": {"p": [0.0, 0.2, 0.0], "q_xyzw": [0.0, 0.0, 0.0, 1.0]},
                    "right_hand": {"p": [0.3, -0.2, 0.1], "q_xyzw": [0.0, 0.0, 0.0, 1.0]},
                }
            ),
            encoding="utf-8",
        )
        return pose_relay.PacketSource(pose_relay.StateSource(path), player_scale=scale)

    def test_the_eye_comes_out_at_the_scaled_height(self) -> None:
        plain, _, _ = self._source(1.0).poses(time.monotonic())
        tall, _, _ = self._source(1.43).poses(time.monotonic())
        self.assertAlmostEqual(tall["head"][0][1] / plain["head"][0][1], 1.43, places=6)

    def test_a_bad_scale_is_refused_at_construction(self) -> None:
        for bad in (0.0, -1.43, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self._source(bad)


class MenuClickTests(unittest.TestCase):
    """The menu button is on the controller, not on the body: a request, then a short press."""

    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.path = Path(holder.name) / "click.json"

    def test_the_wire_carries_the_system_bit(self) -> None:
        packet = wire.fill(
            wire.blank_packet(),
            {"head": ((0.0, 0.0, 1.2), (0.0, 0.0, 0.0, 1.0)),
             "left": ((0.1, 0.2, 0.0), (0.0, 0.0, 0.0, 1.0)),
             "right": ((0.1, -0.2, 0.0), (0.0, 0.0, 0.0, 1.0))},
            system_click=(False, True),
        )
        self.assertFalse(packet.left.system_click)
        self.assertTrue(packet.right.system_click)

    def test_a_request_is_one_short_press(self) -> None:
        click = pose_relay.ClickFile(self.path, hold_s=0.15)
        self.assertEqual(click.pressed(0.0), (False, False))
        pose_relay.request_click("left", path=self.path, hold_s=0.15)
        self.assertEqual(click.pressed(0.0), (True, False))
        self.assertEqual(click.pressed(0.10), (True, False))
        self.assertEqual(click.pressed(0.20), (False, False), "a press is not a level")

    def test_one_request_is_not_a_press_per_packet(self) -> None:
        # Otherwise a single click would fire the menu again and again at 50 Hz.
        pose_relay.request_click("right", path=self.path, hold_s=0.15)
        click = pose_relay.ClickFile(self.path, hold_s=0.15)
        self.assertEqual(click.pressed(0.0), (False, True))
        self.assertEqual(click.pressed(0.30), (False, False))
        self.assertEqual(click.pressed(9.0), (False, False))

    def test_a_malformed_request_is_not_a_press(self) -> None:
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(pose_relay.ClickFile(self.path).pressed(0.0), (False, False))

    def test_an_absent_file_is_not_a_press(self) -> None:
        self.assertEqual(pose_relay.ClickFile(self.path).pressed(0.0), (False, False))

    def test_the_relay_puts_the_click_in_a_packet(self) -> None:
        driver = RecordingDriver()
        driver.start()
        with tempfile.TemporaryDirectory() as raw:
            state_path = Path(raw) / "body_state.json"
            state_path.write_text(json.dumps(sample_state()), encoding="utf-8")
            click_path = Path(raw) / "click.json"
            pose_relay.request_click("right", path=click_path)
            pose_relay.run("127.0.0.1", driver.port, state_path, None,
                           duration=0.4, hz=50.0, click_file=str(click_path))
            driver.closed.wait(5)
        self.assertTrue(driver.system_seen, "the menu click never reached the wire")
        self.assertEqual(driver.last_system, (False, False), "the press was released")
