import json
import tempfile
import time
import unittest
from pathlib import Path

from clank.body import (
    AUTH_ROOT_VEL,
    FREEZE_HOLD_AUTH,
    BodyCommand,
    BodyState,
    Pose,
    hold,
    rest,
)
from clank.bus import CommandBus, UnixCommandServer, send_command

REACHED_RIGHT = Pose((0.18, -0.20, 0.40), (0.0, 0.0, 0.0, 1.0))
REACHED_LEFT = Pose((0.18, 0.20, 0.40), (0.0, 0.0, 0.0, 1.0))
LOOKED_HEAD = Pose((0.0, 0.0, 0.40), (0.0, 0.0, 0.1, 0.995))


def published_state() -> BodyState:
    """A body that has reached with its right hand and turned its head."""
    return BodyState(
        seq_ack=7,
        t_mono_ns=123456789,
        root_p=(0.0, 0.0, 0.793),
        root_q=(1.0, 0.0, 0.0, 0.0),
        v_xy=(0.0, 0.0),
        yaw_rate=0.0,
        q=tuple(0.0 for _ in range(29)),
        qd=tuple(0.0 for _ in range(29)),
        head=LOOKED_HEAD,
        left_hand=REACHED_LEFT,
        right_hand=REACHED_RIGHT,
    )


class FakeClock:
    """Deterministic clock so ballistic timing tests are not wall-clock flaky."""

    def __init__(self, now_ns: int = 0) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns


class BusTestCase(unittest.TestCase):
    def test_latest_wins_drops_older_seq(self) -> None:
        bus = CommandBus(clock=FakeClock())
        self.assertTrue(bus.push(BodyCommand(seq=2, authority=AUTH_ROOT_VEL, v_xy=(0.3, 0.0))))
        self.assertFalse(bus.push(BodyCommand(seq=1, authority=AUTH_ROOT_VEL, v_xy=(0.9, 0.0))))
        self.assertEqual(bus.take(now_ns=0).seq, 2)

    def test_expired_ttl_returns_hold(self) -> None:
        bus = CommandBus(clock=FakeClock())
        bus.push(BodyCommand(seq=3, t_mono_ns=0, ttl_ms=200, authority=AUTH_ROOT_VEL, v_xy=(0.5, 0.0)))
        out = bus.take(now_ns=10 * 1_000_000_000)
        self.assertEqual(out.authority, FREEZE_HOLD_AUTH)
        self.assertEqual(out.seq, 3)

    def test_expired_ttl_copies_published_poses(self) -> None:
        # The production path: body_view pushes a verb, it expires, and the
        # executor must be told to stay where it actually is.
        bus = CommandBus(clock=FakeClock())
        bus.push(BodyCommand(seq=3, t_mono_ns=0, ttl_ms=200, authority=AUTH_ROOT_VEL, v_xy=(0.5, 0.0)))
        bus.publish_state(published_state())
        out = bus.take(now_ns=10 * 1_000_000_000)
        self.assertEqual(out.authority, FREEZE_HOLD_AUTH)
        self.assertEqual(out.head, LOOKED_HEAD)
        self.assertEqual(out.left_hand, REACHED_LEFT)
        self.assertEqual(out.right_hand, REACHED_RIGHT)

    def test_expired_ttl_without_state_claims_the_channels(self) -> None:
        # Nothing achieved yet, so the poses are absent - but the claim is still
        # freeze-hold, not rest. An empty mask here would be the old bug.
        bus = CommandBus(clock=FakeClock())
        bus.push(BodyCommand(seq=3, t_mono_ns=0, ttl_ms=200, authority=AUTH_ROOT_VEL, v_xy=(0.5, 0.0)))
        out = bus.take(now_ns=10 * 1_000_000_000)
        self.assertEqual(out.authority, FREEZE_HOLD_AUTH)
        self.assertIsNone(out.head)
        self.assertIsNone(out.right_hand)

    def test_fresh_command_is_returned(self) -> None:
        bus = CommandBus(clock=FakeClock())
        bus.push(BodyCommand(seq=1, t_mono_ns=0, ttl_ms=5000, authority=AUTH_ROOT_VEL, v_xy=(0.4, 0.1)))
        out = bus.take(now_ns=1_000_000)
        self.assertEqual(out.v_xy, (0.4, 0.1))
        self.assertEqual(out.authority, AUTH_ROOT_VEL)

    def test_empty_bus_is_hold(self) -> None:
        bus = CommandBus(clock=FakeClock())
        self.assertEqual(bus.take(now_ns=0).authority, FREEZE_HOLD_AUTH)
        self.assertTrue(bus.idle())

    def test_ballistic_not_preempted_until_done(self) -> None:
        clock = FakeClock(0)
        bus = CommandBus(clock=clock)
        punch = BodyCommand(seq=1, t_mono_ns=0, ttl_ms=100_000, duration_ms=2000,
                            preempt=False, authority=AUTH_ROOT_VEL, v_xy=(0.5, 0.0))
        walk = BodyCommand(seq=2, t_mono_ns=0, ttl_ms=100_000, duration_ms=0,
                           authority=AUTH_ROOT_VEL, v_xy=(0.0, 0.7))
        bus.push(punch)
        bus.push(walk)

        # Mid-skill: the ballistic wins, the walk only queues.
        self.assertEqual(bus.take(now_ns=1_500_000_000).seq, 1)
        # Past its duration: the queued command takes over.
        self.assertEqual(bus.take(now_ns=2_500_000_000).seq, 2)

    def test_preempting_ballistic_is_replaced_immediately(self) -> None:
        bus = CommandBus(clock=FakeClock(0))
        bus.push(BodyCommand(seq=1, t_mono_ns=0, ttl_ms=100_000, duration_ms=2000, preempt=False))
        bus.push(BodyCommand(seq=2, t_mono_ns=0, ttl_ms=100_000, duration_ms=2000, preempt=True))
        self.assertEqual(bus.take(now_ns=1_000_000).seq, 2)

    def test_invalid_json_increments_counter(self) -> None:
        bus = CommandBus(clock=FakeClock())
        result = bus.push_json("{not json")
        self.assertFalse(result.ok)
        self.assertEqual(bus.invalid_count, 1)

    def test_valid_json_is_accepted_and_counted_once(self) -> None:
        bus = CommandBus(clock=FakeClock())
        result = bus.push_json(json.dumps({"seq": 1, "authority": 0}))
        self.assertTrue(result.ok, result.reason)
        self.assertEqual(bus.invalid_count, 0)

    def test_rejected_command_is_not_pushed(self) -> None:
        bus = CommandBus(clock=FakeClock())
        result = bus.push_json(json.dumps({"seq": 1, "schema": "wrong"}))
        self.assertFalse(result.ok)
        self.assertEqual(bus.invalid_count, 1)
        self.assertEqual(bus.take(now_ns=0).authority, FREEZE_HOLD_AUTH)


class SocketTestCase(unittest.TestCase):
    def test_socket_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "body.sock"
            bus = CommandBus()
            server = UnixCommandServer(bus, path)
            server.start()
            try:
                send_command(path, BodyCommand(seq=1, ttl_ms=10_000,
                                               t_mono_ns=time.monotonic_ns(),
                                               authority=AUTH_ROOT_VEL, v_xy=(0.25, 0.0)))
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline and bus.take().authority != AUTH_ROOT_VEL:
                    time.sleep(0.02)
                out = bus.take()
                self.assertEqual(out.authority, AUTH_ROOT_VEL)
                self.assertEqual(out.v_xy, (0.25, 0.0))
                self.assertEqual(bus.invalid_count, 0)
            finally:
                server.stop()
            self.assertFalse(path.exists())

    def test_socket_reports_invalid_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "body.sock"
            bus = CommandBus()
            server = UnixCommandServer(bus, path)
            server.start()
            try:
                import socket

                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.connect(str(path))
                    sock.sendall(b'{"schema": "nope"}\n')
                    time.sleep(0.1)
                self.assertEqual(bus.invalid_count, 1)
                self.assertTrue(server.errors)
            finally:
                server.stop()

    def test_hold_sent_over_socket_replaces_with_freeze_hold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "body.sock"
            bus = CommandBus()
            server = UnixCommandServer(bus, path)
            server.start()
            try:
                send_command(path, BodyCommand(seq=1, ttl_ms=10_000,
                                               t_mono_ns=time.monotonic_ns(),
                                               authority=AUTH_ROOT_VEL, v_xy=(0.5, 0.0)))
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline and bus.take().authority != AUTH_ROOT_VEL:
                    time.sleep(0.02)
                send_command(path, hold(seq=2, now_ns=time.monotonic_ns()))
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline and bus.take().authority != FREEZE_HOLD_AUTH:
                    time.sleep(0.02)
                # A hold stops the walk without freeing the clip.
                self.assertEqual(bus.take().authority, FREEZE_HOLD_AUTH)
            finally:
                server.stop()

    def test_rest_sent_over_socket_clears_authority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "body.sock"
            bus = CommandBus()
            server = UnixCommandServer(bus, path)
            server.start()
            try:
                send_command(path, BodyCommand(seq=1, ttl_ms=10_000,
                                               t_mono_ns=time.monotonic_ns(),
                                               authority=AUTH_ROOT_VEL, v_xy=(0.5, 0.0)))
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline and bus.take().authority != AUTH_ROOT_VEL:
                    time.sleep(0.02)
                send_command(path, rest(seq=2, now_ns=time.monotonic_ns()))
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline and bus.take().authority != 0:
                    time.sleep(0.02)
                self.assertEqual(bus.take().authority, 0)
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
