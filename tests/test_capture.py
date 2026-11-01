from __future__ import annotations

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from clank.capture import PNG_SIGNATURE, CaptureError, capture_frame

#: A PNG header plus a little payload. `capture_frame` validates the signature, so
#: fixtures must look like a PNG rather than like the string "PNG".
PNG_BYTES = PNG_SIGNATURE + b"fixture"


class FakeScreenshooter:
    """Stands in for `weston-screenshooter`, including its measured quirks."""

    def __init__(
        self,
        *,
        write: bool = True,
        payload: bytes = PNG_BYTES,
        returncode: int = 0,
        stderr: str = "",
        names: tuple[str, ...] = ("wayland-screenshot-2026-11-01_18-25-02.png",),
        timeouts: bool = False,
    ) -> None:
        self.write = write
        self.payload = payload
        self.returncode = returncode
        self.stderr = stderr
        self.names = names
        self.timeouts = timeouts
        self.calls: list[tuple[list[str], dict]] = []
        self.cwd_existed: bool | None = None

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        self.calls.append((argv, kwargs))
        if kwargs.get("cwd") is not None:
            self.cwd_existed = Path(kwargs["cwd"]).is_dir()
        if self.timeouts:
            timeout = kwargs.get("timeout")
            raise subprocess.TimeoutExpired(argv, float(timeout) if timeout else 5.0)
        # The real tool ignores any output argument and writes a timestamped file
        # into the cwd, so this fake does exactly that and nothing about argv.
        if self.write and kwargs["cwd"] is not None:
            directory = Path(kwargs["cwd"])
            for index, name in enumerate(self.names):
                shot = directory / name
                shot.write_bytes(self.payload)
                now = time.time_ns() + index
                os.utime(shot, ns=(now, now))
        return subprocess.CompletedProcess(argv, self.returncode, b"", self.stderr)


class CaptureFrameTests(unittest.TestCase):
    def capture(self, base: Path, run, **overrides) -> Path:
        options = {
            "screenshooter": "weston-screenshooter",
            "runtime_dir": "/run/user/1000",
            "wayland_display": "clank-test",
            "work_dir": base / "work",
            "destination": base / "out" / "game.png",
            "run": run,
        }
        options.update(overrides)
        return capture_frame(**options)

    def test_publishes_the_png_and_returns_the_destination(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            run = FakeScreenshooter()
            path = self.capture(base, run)
            self.assertEqual(path, base / "out" / "game.png")
            self.assertEqual(path.read_bytes(), PNG_BYTES)

    def test_the_output_argument_is_not_used(self) -> None:
        # The measured failure mode: a path argument is accepted, ignored, and the
        # PNG lands in the cwd. The primitive must never rely on the argument.
        with tempfile.TemporaryDirectory() as raw:
            run = FakeScreenshooter()
            self.capture(Path(raw), run)
            argv, _ = run.calls[0]
            self.assertEqual(argv, ["weston-screenshooter"])

    def test_runs_in_a_fresh_working_directory_inside_work_dir(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            run = FakeScreenshooter()
            self.capture(base, run)
            _, kwargs = run.calls[0]
            cwd = Path(kwargs["cwd"])
            self.assertTrue(run.cwd_existed)
            self.assertEqual(cwd.parent, base / "work")
            # The frame was moved out and the scratch directory removed.
            self.assertFalse(cwd.exists())
            self.assertEqual(list((base / "work").glob("*")), [])

    def test_sets_the_wayland_environment_and_drops_an_inherited_socket(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            previous = os.environ.get("WAYLAND_SOCKET")
            os.environ["WAYLAND_SOCKET"] = "999"
            try:
                run = FakeScreenshooter()
                self.capture(base, run)
            finally:
                if previous is None:
                    os.environ.pop("WAYLAND_SOCKET", None)
                else:
                    os.environ["WAYLAND_SOCKET"] = previous
            _, kwargs = run.calls[0]
            env = kwargs["env"]
            self.assertIsNot(env, os.environ)
            self.assertEqual(env["XDG_RUNTIME_DIR"], "/run/user/1000")
            self.assertEqual(env["WAYLAND_DISPLAY"], "clank-test")
            self.assertNotIn("WAYLAND_SOCKET", env)

    def test_creates_missing_work_and_destination_directories(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            path = self.capture(base, FakeScreenshooter())
            self.assertTrue(path.parent.is_dir())

    def test_picks_the_newest_png_when_the_tool_writes_several(self) -> None:
        # A multi-output compositor writes one file per output.
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            run = FakeScreenshooter(
                names=(
                    "wayland-screenshot-output-0-old.png",
                    "wayland-screenshot-output-1-new.png",
                )
            )
            path = self.capture(base, run)
            self.assertEqual(path.read_bytes(), PNG_BYTES)

    def test_failure_leaves_the_previous_frame_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            destination = base / "out" / "game.png"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(PNG_SIGNATURE + b"previous")

            with self.assertRaisesRegex(CaptureError, "screenshooter exited 1"):
                self.capture(
                    base,
                    FakeScreenshooter(returncode=1, stderr="cannot connect to display"),
                )
            self.assertEqual(destination.read_bytes(), PNG_SIGNATURE + b"previous")

    def test_stderr_is_carried_in_the_error(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            self.assertRaisesRegex(CaptureError, "cannot connect to display"),
        ):
            self.capture(
                Path(raw),
                FakeScreenshooter(returncode=1, stderr="cannot connect to display"),
            )

    def test_no_file_despite_exit_zero_is_an_error(self) -> None:
        # The exact stale claim this path was buried under: exit 0, no artifact.
        with (
            tempfile.TemporaryDirectory() as raw,
            self.assertRaisesRegex(CaptureError, "wrote no"),
        ):
            self.capture(Path(raw), FakeScreenshooter(write=False))

    def test_a_non_png_payload_is_rejected(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            self.assertRaisesRegex(CaptureError, "non-PNG"),
        ):
            self.capture(Path(raw), FakeScreenshooter(payload=b"not a png"))

    def test_timeout_is_not_swallowed(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            self.assertRaises(subprocess.TimeoutExpired),
        ):
            self.capture(Path(raw), FakeScreenshooter(timeouts=True))


if __name__ == "__main__":
    unittest.main()
