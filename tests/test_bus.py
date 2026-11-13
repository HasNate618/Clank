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


class HoldLatchesOnce(unittest.TestCase):
    """A freeze-hold must pin a pose, not chase the body.

    MEASURED in the sim (`research/p2-soak.md`): `hold()` driven by a live
    `publish_state()` every tick drifts 0.444 m over 20 s at 0.0413 m/s RMS,
    because `channel_pose` rule 3 returns the LAST ACHIEVED pose - the target
    follows the body, so nothing pulls it back. The same command with the pose
    latched once drifts 0.0226 m at 6.9e-5 m/s. The corruption is here: the bus
    handed the executor a fresh snapshot every call.
    """

    def moved_state(self) -> BodyState:
        """A body that has drifted a long way from `published_state()`."""
        base = published_state()
        far = Pose((0.60, -0.55, 0.20), (0.0, 0.0, 0.0, 1.0))
        return BodyState(
            **{
                **base.__dict__,
                "right_hand": far,
                "left_hand": far,
                "head": far,
                "t_mono_ns": base.t_mono_ns + 20_000_000_000,
            }
        )

    def test_the_hold_does_not_follow_the_body(self) -> None:
        clock = FakeClock(0)
        bus = CommandBus(clock=clock)
        bus.publish_state(published_state())
        first = bus.take(now_ns=0)
        self.assertEqual(first.right_hand, REACHED_RIGHT)

        # The body moves a long way while the model thinks. The hold must not.
        bus.publish_state(self.moved_state())
        clock.now_ns = 1_000_000_000
        second = bus.take(now_ns=1_000_000_000)
        self.assertEqual(second.authority, FREEZE_HOLD_AUTH)
        self.assertEqual(second.right_hand, REACHED_RIGHT)

    def test_the_latch_survives_many_ticks(self) -> None:
        bus = CommandBus(clock=FakeClock(0))
        bus.publish_state(published_state())
        bus.take(now_ns=0)
        for i in range(1, 50):
            bus.publish_state(self.moved_state())
            cmd = bus.take(now_ns=i * 20_000_000)
            self.assertEqual(cmd.right_hand, REACHED_RIGHT)

    def test_an_expired_command_latches_too(self) -> None:
        clock = FakeClock(0)
        bus = CommandBus(clock=clock)
        bus.publish_state(published_state())
        bus.push(BodyCommand(seq=1, t_mono_ns=0, ttl_ms=100, authority=AUTH_ROOT_VEL,
                             v_xy=(0.4, 0.0)))
        clock.now_ns = 500_000_000
        expired = bus.take(now_ns=clock.now_ns)
        self.assertEqual(expired.authority, FREEZE_HOLD_AUTH)
        bus.publish_state(self.moved_state())
        clock.now_ns = 600_000_000
        later = bus.take(now_ns=clock.now_ns)
        self.assertEqual(later.right_hand, REACHED_RIGHT)

    def test_a_retired_latch_pins_the_pose_it_was_retired_at(self) -> None:
        """A clip is not a command, so the latch has to be retired by the body explicitly.

        MEASURED 2026-11-13 on a live body: a generated clip moved the glove to pelvis x 0.26,
        and the arm was back at 0.43 - the previous reach's target - within 2 s. `push()` retires
        the snapshot only on a real command, and a deposited clip never goes through `push()`, so
        the freeze-hold that follows a clip baked the pose from *before* it. Re-sending the hold,
        pinning `executor.last_state` into the verb and a keepalive were each tried and each lost
        the arm; the snapshot was the stale thing all along.
        """
        clock = FakeClock(0)
        bus = CommandBus(clock=clock)
        bus.publish_state(published_state())
        bus.take(now_ns=0)
        # The clip plays: the body reaches a different place, with no command to say so.
        bus.publish_state(self.moved_state())
        bus.retire_latch()
        clock.now_ns = 1_000_000_000
        held = bus.take(now_ns=clock.now_ns)
        self.assertEqual(held.authority, FREEZE_HOLD_AUTH)
        self.assertEqual(held.right_hand, self.moved_state().right_hand)
        # ... and it still latches exactly once.
        bus.publish_state(published_state())
        again = bus.take(now_ns=2_000_000_000)
        self.assertEqual(again.right_hand, self.moved_state().right_hand)

    def test_a_new_command_clears_the_latch(self) -> None:
        clock = FakeClock(0)
        bus = CommandBus(clock=clock)
        bus.publish_state(published_state())
        bus.take(now_ns=0)
        bus.push(BodyCommand(seq=2, t_mono_ns=0, ttl_ms=100,
                             authority=AUTH_ROOT_VEL, v_xy=(0.4, 0.0)))
        bus.take(now_ns=0)
        # A new pose has been reached, so the next hold must latch the new one.
        bus.publish_state(self.moved_state())
        clock.now_ns = 400_000_000
        fresh = bus.take(now_ns=clock.now_ns)
        self.assertEqual(fresh.right_hand.p, (0.60, -0.55, 0.20))

    def test_rest_clears_the_latch(self) -> None:
        clock = FakeClock(0)
        bus = CommandBus(clock=clock)
        bus.publish_state(published_state())
        bus.take(now_ns=0)
        bus.push(rest(seq=2, now_ns=0))
        self.assertEqual(bus.take(now_ns=0).authority, 0)
        bus.publish_state(self.moved_state())
        clock.now_ns = 400_000_000
        fresh = bus.take(now_ns=clock.now_ns)
        self.assertEqual(fresh.right_hand.p, (0.60, -0.55, 0.20))

    def test_an_explicit_hold_command_is_latched_too(self) -> None:
        clock = FakeClock(0)
        bus = CommandBus(clock=clock)
        bus.publish_state(published_state())
        bus.push(hold(seq=1, now_ns=0, ttl_ms=10_000))
        self.assertEqual(bus.take(now_ns=0).right_hand, REACHED_RIGHT)
        bus.publish_state(self.moved_state())
        clock.now_ns = 20_000_000
        later = bus.take(now_ns=clock.now_ns)
        self.assertEqual(later.authority, FREEZE_HOLD_AUTH)
        self.assertEqual(later.right_hand, REACHED_RIGHT)

    def test_the_latch_is_not_shared_between_buses(self) -> None:
        one, two = CommandBus(clock=FakeClock(0)), CommandBus(clock=FakeClock(0))
        one.publish_state(published_state())
        two.publish_state(published_state())
        one.take(now_ns=0)
        two.publish_state(self.moved_state())
        self.assertEqual(two.take(now_ns=0).right_hand.p, (0.60, -0.55, 0.20))
