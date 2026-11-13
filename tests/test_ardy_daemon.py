"""The resident generator's protocol, exercised over a real socket with a fake generator.

No torch anywhere: the reason the protocol lives in its own module is that everything except the
final call into `ardy_gen` can be tested on a machine with the GPU off, which is this machine's
normal state. The generator is injected, so these tests measure the daemon's own behaviour -
what it answers, what it survives, and when it lets go of the VRAM.
"""

from __future__ import annotations

import contextlib
import importlib.util
import pathlib
import socket
import sys
import tempfile
import threading
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:  # pragma: no cover - a missing file, not a branch
        raise RuntimeError(f"could not load {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


daemon = _load("clank_ardy_daemon", "spike/wbc/ardy_daemon.py")
client = _load("clank_ardy_client", "spike/wbc/ardy_client.py")

POSES = [[0.0] * 36 for _ in range(3)]
FRAMES = [[1.0] * 36 for _ in range(52)]


def fake_generator(poses, target, root_forward_m, steps):
    """Echo what it was asked, so the round trip is visible in the answer."""
    return FRAMES, {
        "seconds": 0.0,
        "hand_residual_m": 0.016,
        "wrist_residual_m": 0.023,
        "peak_vram_gb": 0.94,
        "asked_poses": len(poses),
        "asked_target": list(target),
        "asked_root": root_forward_m,
        "asked_steps": steps,
    }


def _wait_for(predicate, timeout=10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class TheAnswer(unittest.TestCase):
    """`handle` is pure: a request in, a reply out, and no way for a caller to kill it."""

    def test_ping_answers(self) -> None:
        reply = daemon.handle({"op": "ping"}, fake_generator)
        self.assertTrue(reply["ok"])
        self.assertIsInstance(reply["pid"], int)

    def test_an_unknown_op_is_answered_not_fatal(self) -> None:
        reply = daemon.handle({"op": "dance"}, fake_generator)
        self.assertFalse(reply["ok"])
        self.assertIn("dance", reply["why"])

    def test_a_request_that_is_not_an_object_is_answered(self) -> None:
        for nonsense in (None, [], "generate", 7):
            reply = daemon.handle(nonsense, fake_generator)
            self.assertFalse(reply["ok"], nonsense)

    def test_a_generate_request_carries_the_target_and_the_root(self) -> None:
        reply = daemon.handle(
            {"op": "generate", "poses": POSES, "target": [1, 2, 3],
             "root_forward_m": 0.05, "steps": 4},
            fake_generator,
        )
        self.assertTrue(reply["ok"], reply)
        metrics = reply["metrics"]
        self.assertEqual(metrics["asked_poses"], 3)
        self.assertEqual(metrics["asked_target"], [1.0, 2.0, 3.0])
        self.assertEqual(metrics["asked_root"], 0.05)
        self.assertEqual(metrics["asked_steps"], 4)
        self.assertEqual(len(reply["frames"]), 52)

    def test_a_target_is_three_numbers(self) -> None:
        for bad in ([1, 2], [1, 2, 3, 4], "abc", []):
            reply = daemon.handle(
                {"op": "generate", "poses": POSES, "target": bad}, fake_generator
            )
            self.assertFalse(reply["ok"], bad)

    def test_poses_are_required(self) -> None:
        reply = daemon.handle({"op": "generate", "target": [1, 2, 3]}, fake_generator)
        self.assertFalse(reply["ok"])
        self.assertIn("poses", reply["why"])

    def test_a_generator_that_raises_is_reported_not_propagated(self) -> None:
        def explode(poses, target, root_forward_m, steps):
            raise RuntimeError("cuda is unavailable")

        reply = daemon.handle(
            {"op": "generate", "poses": POSES, "target": [1, 2, 3]}, explode
        )
        self.assertFalse(reply["ok"])
        self.assertIn("cuda is unavailable", reply["why"])

    def test_a_daemon_with_no_generator_says_so(self) -> None:
        reply = daemon.handle(
            {"op": "generate", "poses": POSES, "target": [1, 2, 3]}, None
        )
        self.assertFalse(reply["ok"])
        self.assertIn("no generator", reply["why"])


class TheSocket(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.sock = pathlib.Path(self.dir.name) / "ardy.sock"
        self.thread = threading.Thread(
            target=daemon.serve, args=(self.sock, fake_generator),
            kwargs={"idle_s": 10.0}, daemon=True,
        )
        self.thread.start()
        self.assertTrue(_wait_for(lambda: client.alive(self.sock)),
                        "the daemon never answered a ping")

    def tearDown(self) -> None:
        # Every test in this class leaves a live daemon of its own; the shutdown test kills it
        # early, so a refused ask here is the expected case rather than a failure.
        with contextlib.suppress(client.Unavailable):
            client.ask(self.sock, {"op": "shutdown"}, timeout=1.0)
        self.thread.join(timeout=5.0)
        self.assertFalse(self.thread.is_alive(), "the daemon did not exit")

    def test_a_reach_round_trips(self) -> None:
        frames, metrics = client.generate(self.sock, POSES, (0.1, 0.2, 0.3), 0.05, 10)
        self.assertEqual(len(frames), 52)
        self.assertAlmostEqual(metrics["hand_residual_m"], 0.016)

    def test_shutdown_releases_the_socket(self) -> None:
        reply = client.ask(self.sock, {"op": "shutdown"})
        self.assertTrue(reply.get("stopping"))
        self.assertTrue(_wait_for(lambda: not self.sock.exists()),
                        "the socket file outlived the daemon")
        self.assertFalse(client.alive(self.sock))


class TheLifetime(unittest.TestCase):
    def test_an_idle_daemon_lets_go(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sock = pathlib.Path(tmp) / "ardy.sock"
            started = time.monotonic()
            served = daemon.serve(sock, fake_generator, idle_s=0.2)
            self.assertEqual(served, 0)
            self.assertLess(time.monotonic() - started, 5.0)
            self.assertFalse(sock.exists())

    def test_a_socket_left_by_a_dead_daemon_does_not_block_a_new_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sock = pathlib.Path(tmp) / "ardy.sock"
            sock.write_text("stale\n")  # what a SIGKILLed daemon leaves behind
            thread = threading.Thread(
                target=daemon.serve, args=(sock, fake_generator),
                kwargs={"idle_s": 5.0}, daemon=True,
            )
            thread.start()
            self.assertTrue(_wait_for(lambda: client.alive(sock)),
                            "a stale socket file was treated as a listener")
            client.ask(sock, {"op": "shutdown"})
            thread.join(timeout=5.0)

    def test_once_serves_one_request_and_exits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sock = pathlib.Path(tmp) / "ardy.sock"
            result: list[int] = []
            thread = threading.Thread(
                target=lambda: result.append(daemon.serve(sock, fake_generator, once=True)),
                daemon=True,
            )
            thread.start()
            # The readiness ping *is* the single request this daemon serves: a successful
            # `alive` means one ping was answered, and that is what `once` allows.
            self.assertTrue(_wait_for(lambda: client.alive(sock)))
            thread.join(timeout=5.0)
            self.assertEqual(result, [1])


class TheClientWithoutADaemon(unittest.TestCase):
    def test_generating_with_nothing_listening_is_unavailable_not_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
                self.assertRaises(client.Unavailable):
            client.generate(pathlib.Path(tmp) / "nothing.sock", POSES, (0, 0, 0), None, 10)

    def test_a_reply_that_is_not_json_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
                socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            sock = pathlib.Path(tmp) / "rude.sock"
            server.bind(str(sock))
            server.listen(1)

            def answer() -> None:
                conn, _ = server.accept()
                with conn:
                    conn.recv(4096)
                    conn.sendall(b"not json\n")

            thread = threading.Thread(target=answer, daemon=True)
            thread.start()
            with self.assertRaises(client.Unavailable):
                client.ask(sock, {"op": "ping"}, timeout=5.0)
            thread.join(timeout=5.0)


class TheSpawn(unittest.TestCase):
    """`ensure` is the only thing that starts a process, so its two failure modes matter."""

    def test_it_spawns_a_daemon_and_waits_for_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            sock = tmp_path / "ardy.sock"
            helper = tmp_path / "helper.py"
            helper.write_text(
                "import importlib.util, pathlib\n"
                f"spec = importlib.util.spec_from_file_location('d', {daemon.__file__!r})\n"
                "d = importlib.util.module_from_spec(spec)\n"
                "spec.loader.exec_module(d)\n"
                "d.serve(pathlib.Path(" + repr(str(sock)) + "),\n"
                "        lambda p, t, r, s: ([[0.0] * 36], {'seconds': 0.0}),\n"
                "        idle_s=30.0)\n"
            )
            self.assertTrue(client.ensure(sock, [sys.executable, str(helper)], wait_s=20.0))
            self.assertTrue(client.alive(sock))
            client.ask(sock, {"op": "shutdown"})

    def test_a_spawn_that_dies_at_once_is_not_waited_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sock = pathlib.Path(tmp) / "ardy.sock"
            started = time.monotonic()
            ok = client.ensure(
                sock, [sys.executable, "-c", "raise SystemExit(1)"], wait_s=30.0
            )
            self.assertFalse(ok)
            self.assertLess(time.monotonic() - started, 5.0,
                            "a dead spawn was waited out instead of noticed")


if __name__ == "__main__":
    unittest.main()
