from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

from clank.preview import PreviewCapture, atomic_replace, newest_png

#: A real PNG header, because the capture primitive rejects an empty or truncated
#: file even when the screenshooter exits 0.
PNG_BYTES = b"\x89PNG\r\n\x1a\nfixture"


class FakeRun:
    def __init__(self, fail: str | None = None) -> None:
        self.fail = fail
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        self.calls.append((argv, kwargs))
        if argv[0] != "weston-screenshooter":
            raise AssertionError(f"unexpected command: {argv}")
        if self.fail == "screen":
            return subprocess.CompletedProcess(argv, 1, "", "screen failed")
        cwd = Path(kwargs["cwd"])
        shot = cwd / "wayland-screenshot.png"
        shot.write_bytes(PNG_BYTES)
        now = time.time_ns()
        os.utime(shot, ns=(now, now))
        return subprocess.CompletedProcess(argv, 0, "", "")


class ExplodingCapture(PreviewCapture):
    def capture_once(self) -> bool:
        raise RuntimeError("boom")


class TimeoutRun:
    def __call__(self, argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs.get("timeout", 5))


class BlockingCapture(PreviewCapture):
    def _run(self) -> None:
        self.stop_event.wait(2)


class PreviewTests(unittest.TestCase):
    def make_capture(
        self,
        base: Path,
        run,
        log=None,
        capture_type=PreviewCapture,
    ) -> PreviewCapture:
        return capture_type(
            width=1280,
            height=720,
            panel_height=32,
            fps=12,
            screenshooter="weston-screenshooter",
            runtime_dir=base,
            wayland_display="wayland-test",
            work_dir=base,
            publish_path=base / "latest.png",
            run=run,
            log=log or (lambda message: None),
        )

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
            tmp = base / "latest.png.tmp"
            dest = base / "latest.png"
            tmp.write_bytes(b"PNG")
            atomic_replace(tmp, dest)
            self.assertEqual(dest.read_bytes(), b"PNG")
            self.assertFalse(tmp.exists())

    def test_frame_directory_failure_is_counted(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            work_file = base / "not-a-directory"
            work_file.write_bytes(b"x")
            run = FakeRun()
            capture = self.make_capture(work_file, run)
            self.assertFalse(capture.capture_once())
            self.assertEqual(capture.status()["errors"], 1)
            self.assertEqual(run.calls, [])

    def test_loop_survives_capture_crash(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            capture = self.make_capture(
                base,
                FakeRun(),
                capture_type=ExplodingCapture,
            )
            capture.start()
            try:
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline and capture.status()["errors"] < 1:
                    time.sleep(0.01)
                self.assertIsNotNone(capture._thread)
                self.assertTrue(capture._thread.is_alive())
                self.assertGreaterEqual(capture.status()["errors"], 1)
            finally:
                capture.stop()

    def test_start_twice_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            capture = self.make_capture(base, FakeRun())
            capture.start()
            try:
                with self.assertRaisesRegex(RuntimeError, "already started"):
                    capture.start()
            finally:
                capture.stop()

    def test_start_after_stop_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            capture = self.make_capture(base, FakeRun())
            capture.start()
            capture.stop()
            with self.assertRaisesRegex(RuntimeError, "already started"):
                capture.start()

    def test_concurrent_start_allows_only_one(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            capture = self.make_capture(
                base,
                FakeRun(),
                capture_type=BlockingCapture,
            )
            barrier = threading.Barrier(8)
            results: list[object] = [None] * 8

            def attempt(index: int) -> None:
                barrier.wait()
                try:
                    capture.start()
                except Exception as exc:
                    results[index] = exc

            threads = [threading.Thread(target=attempt, args=(index,)) for index in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5)
            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(sum(result is None for result in results), 1)
            self.assertEqual(sum(isinstance(result, RuntimeError) for result in results), 7)
            capture.stop(timeout=2)

    def test_stop_before_start_prevents_start(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            capture = self.make_capture(base, FakeRun())
            capture.stop()
            self.assertIsNone(capture._thread)
            with self.assertRaisesRegex(RuntimeError, "already started"):
                capture.start()
            self.assertIsNone(capture._thread)

    def test_stop_joins_thread(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            capture = self.make_capture(base, FakeRun())
            capture.start()
            capture.stop(timeout=2)
            self.assertIsNotNone(capture._thread)
            self.assertFalse(capture._thread.is_alive())

    def test_screenshooter_failure_logs_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            logs: list[str] = []
            capture = self.make_capture(base, FakeRun("screen"), log=logs.append)
            self.assertFalse(capture.capture_once())
            self.assertTrue(any("screenshooter exited 1" in entry for entry in logs))
            self.assertTrue(any("screen failed" in entry for entry in logs))

    def test_run_timeout_is_counted(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            capture = self.make_capture(base, TimeoutRun())
            self.assertFalse(capture.capture_once())
            self.assertEqual(capture.status()["errors"], 1)

    def test_capture_once_publishes_raw_png(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            run = FakeRun()
            capture = self.make_capture(base, run)
            self.assertTrue(capture.capture_once())
            self.assertEqual((base / "latest.png").read_bytes(), PNG_BYTES)
            self.assertEqual(capture.status()["errors"], 0)
            self.assertEqual(capture.status()["frames"], 1)
            self.assertEqual([call[0] for call in run.calls], [["weston-screenshooter"]])
            self.assertEqual(list(base.glob("latest.png*.tmp")), [])

    def test_failed_frame_keeps_previous_frame(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            run = FakeRun()
            capture = self.make_capture(base, run)
            self.assertTrue(capture.capture_once())
            before = (base / "latest.png").read_bytes()
            run.fail = "screen"
            self.assertFalse(capture.capture_once())
            self.assertEqual((base / "latest.png").read_bytes(), before)
            self.assertEqual(capture.status()["errors"], 1)


if __name__ == "__main__":
    unittest.main()
