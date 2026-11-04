from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import socket
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

MANUAL_PATH = Path(__file__).resolve().parents[1] / "spike" / "c0" / "c0_manual.py"
SPEC = importlib.util.spec_from_file_location("c0_manual_test", MANUAL_PATH)
assert SPEC is not None and SPEC.loader is not None
manual = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manual)


class FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class HandshakeTests(unittest.TestCase):
    @staticmethod
    def packet() -> bytes:
        return manual.c0.HEADER.to_bytes(8, "little") + bytes(manual.c0.SIZE - 8)

    def test_reads_both_packets_before_reporting_success(self) -> None:
        a, b = socket.socketpair()
        a.settimeout(1.0)
        output = io.StringIO()
        errors: list[BaseException] = []

        def run() -> None:
            try:
                with contextlib.redirect_stdout(output):
                    manual.read_handshake(a)
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        try:
            time.sleep(0.05)
            self.assertNotIn("handshake ok", output.getvalue())
            b.sendall(self.packet() * 2)
            thread.join(1.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertIn("[manual] handshake ok", output.getvalue())
        finally:
            a.close()
            b.close()

    def test_rejects_a_wrong_header(self) -> None:
        a, b = socket.socketpair()
        try:
            b.sendall(bytes(manual.c0.SIZE * 2))
            with self.assertRaisesRegex(ValueError, "header"):
                manual.read_handshake(a)
        finally:
            a.close()
            b.close()


class BrowserInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.path = self.base / "manual.sock"
        self.clock = FakeClock()
        self.input = manual.BrowserInput(self.path, clock=self.clock)
        self.client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.client.connect(str(self.path))

    def tearDown(self) -> None:
        self.client.close()
        self.input.close()
        self.temp.cleanup()

    def send(self, payload) -> None:
        self.client.send(json.dumps(payload).encode())

    def test_armed_batch_sets_keys_buttons_and_mouse(self) -> None:
        self.send(
            {
                "armed": True,
                "keys": ["w", "shift"],
                "buttons": [0, 2],
                "dx": 4,
                "dy": -2,
                "wheel": 1,
            }
        )
        self.input.read()
        self.assertTrue(self.input.armed)
        self.assertTrue(self.input.down("w"))
        self.assertTrue(self.input.down("shift"))
        self.assertEqual(self.input.buttons, {0, 2})
        self.assertEqual(self.input.take_mouse(), (4, -2, 1))
        self.assertEqual(self.input.take_mouse(), (0, 0, 0))

    def test_disarm_clears_state_and_ignores_input(self) -> None:
        self.send({"armed": True, "keys": ["w"], "buttons": [0], "dx": 2, "dy": 3, "wheel": 1})
        self.input.read()
        self.send({"armed": False})
        self.input.read()
        self.assertFalse(self.input.armed)
        self.assertFalse(self.input.down("w"))
        self.assertEqual(self.input.buttons, set())
        self.assertEqual(self.input.take_mouse(), (0, 0, 0))

    def test_stale_batch_disarms(self) -> None:
        self.send({"armed": True, "keys": ["w"], "buttons": [0], "dx": 1, "dy": 2, "wheel": 0})
        self.input.read()
        self.clock.now += manual.INPUT_LEASE_SECONDS + 0.1
        self.input.read()
        self.assertFalse(self.input.armed)
        self.assertFalse(self.input.down("w"))

    def test_malformed_and_unknown_keys_are_ignored(self) -> None:
        self.client.send(b"{")
        self.input.read()
        self.assertFalse(self.input.armed)
        self.send({"armed": True, "keys": ["nope", "w"], "buttons": [99], "dx": 1, "dy": 0, "wheel": 0})
        self.input.read()
        self.assertTrue(self.input.down("w"))
        self.assertEqual(self.input.buttons, set())

    def test_close_removes_socket(self) -> None:
        self.input.close()
        self.assertFalse(self.path.exists())


class KeyInput:
    def __init__(self, keys=()) -> None:
        self.keys = set(keys)
        self.buttons: set[int] = set()
        self.armed = True

    def down(self, name: str) -> bool:
        return name in self.keys

    def take_mouse(self):
        return 0.0, 0.0, 0.0


def make_manual():
    args = types.SimpleNamespace(
        hand=(0.0, 0.056, -0.702),
        sens=0.072,
        invert_y=False,
        speed=0.8,
        hand_speed=0.5,
        fov_h_deg=0.0,
        fov_v_deg=0.0,
    )
    return manual.Manual(args)


class ManualPunchTests(unittest.TestCase):
    DT = 1.0 / 90.0

    def run_for(self, m, inp, seconds):
        steps = max(1, round(seconds / self.DT))
        packet = None
        for _ in range(steps):
            m.update(inp, self.DT)
            packet = m.packet()
        assert packet is not None
        return packet

    def test_f_punches_active_hand_forward_then_returns(self) -> None:
        m = make_manual()
        inp = KeyInput(["f"])
        base = list(m.hand)

        packet = self.run_for(m, inp, 0.14)
        self.assertTrue(m.punch_active)
        self.assertTrue(m.grip)
        self.assertLess(packet.right.pose.position.z, -1.0)
        self.assertEqual(m.hand, base)

        packet = self.run_for(m, inp, 0.25)
        self.assertFalse(m.punch_active)
        self.assertFalse(m.grip)
        self.assertAlmostEqual(packet.right.pose.position.z, base[2], places=5)

    def test_punch_uses_active_left_hand_only(self) -> None:
        m = make_manual()
        inp = KeyInput(["2"])
        m.update(inp, self.DT)
        right_base = m.packet().right.pose.position.z
        inp.keys.add("f")

        packet = self.run_for(m, inp, 0.14)
        self.assertLess(packet.left.pose.position.z, -0.6)
        self.assertAlmostEqual(packet.right.pose.position.z, right_base, places=5)

    def test_punch_does_not_retrigger_while_key_is_held(self) -> None:
        m = make_manual()
        inp = KeyInput(["f"])
        self.run_for(m, inp, 0.14)
        self.assertTrue(m.punch_active)
        self.run_for(m, inp, 0.25)
        self.assertFalse(m.punch_active)

        self.run_for(m, inp, 0.20)
        self.assertFalse(m.punch_active, "holding F must not loop the punch")

        inp.keys.clear()
        m.update(inp, self.DT)
        inp.keys.add("f")
        m.update(inp, self.DT)
        self.assertTrue(m.punch_active)

    def test_reset_cancels_punch_and_restores_hand(self) -> None:
        m = make_manual()
        inp = KeyInput(["f"])
        self.run_for(m, inp, 0.14)
        self.assertTrue(m.punch_active)

        inp.keys.add("r")
        m.update(inp, self.DT)
        packet = m.packet()
        self.assertFalse(m.punch_active)
        self.assertFalse(m.grip)
        self.assertAlmostEqual(packet.right.pose.position.z, -0.702, places=5)

    def test_punch_sets_active_hand_linear_velocity(self) -> None:
        m = make_manual()
        inp = KeyInput(["f"])
        m.update(inp, self.DT)
        m.packet()

        packet = self.run_for(m, inp, 0.10)
        self.assertLess(packet.right.linear_velocity.z, -1.0)
        self.assertAlmostEqual(packet.left.linear_velocity.z, 0.0, places=5)

        packet = self.run_for(m, inp, 0.30)
        self.assertAlmostEqual(packet.right.linear_velocity.z, 0.0, places=5)

    def test_status_reports_punch(self) -> None:
        m = make_manual()
        inp = KeyInput(["f"])
        m.update(inp, self.DT)
        self.assertIn("punch=1", m.status())


if __name__ == "__main__":
    unittest.main()
