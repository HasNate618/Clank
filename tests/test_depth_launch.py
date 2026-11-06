from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LAUNCH = REPO / "spike/e5/launch.py"
RUN_SH = REPO / "run.sh"


class DepthLaunchTests(unittest.TestCase):
    """run.sh's launcher: layer by default, plain dashboard on request or failure.

    Every case passes an explicit --config, because the repository config.toml is
    machine-specific and absent from the sandboxed check.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.home = Path(self._tmp.name)
        self.config = self.home / "config.toml"
        self.config.write_text('[launch]\nproton_app_id = "3415720001"\n', encoding="utf-8")
        self.layer = self.home / "depth_stream_layer.so"
        self.layer.write_bytes(b"")
        self.compatdata = (
            self.home / ".local/share/Steam/steamapps/compatdata/3415720001"
        )
        self.compatdata.mkdir(parents=True)

    def run_launch(self, *args: str, home: Path | None = None) -> subprocess.CompletedProcess[str]:
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("DEPTH_STREAM_", "DISABLE_DEPTH_STREAM"))
            and key != "VK_LOADER_LAYERS_DISABLE"
        }
        env["HOME"] = str(home or self.home)
        env["DEPTH_STREAM_LAYER"] = str(self.layer)
        return subprocess.run(
            [sys.executable, str(LAUNCH), *args],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def command(self, result: subprocess.CompletedProcess[str]) -> str:
        for line in result.stdout.splitlines():
            if line.startswith("command: "):
                return line[len("command: ") :]
        self.fail(f"no command line in stdout:\n{result.stdout}")

    def test_forwards_clank_flags(self) -> None:
        result = self.run_launch(
            "--dry-run",
            "--replace",
            "--config",
            str(self.config),
            "--host",
            "127.0.0.1",
            "--port",
            "9001",
            "--preview-smoke",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        command = self.command(result)
        for expected in (
            "-m clank",
            "--replace",
            f"--config {self.config}",
            "--host 127.0.0.1",
            "--port 9001",
            "--preview-smoke",
        ):
            self.assertIn(expected, command)
        # the layer path is the difference from the plain dashboard
        self.assertIn("DEPTH_STREAM_PATH=", result.stdout)

    def test_config_selects_the_compatdata(self) -> None:
        # A config pointing at another AppId must not be ignored by the manifest.
        other = self.home / "other.toml"
        other.write_text('[launch]\nproton_app_id = "1111111111"\n', encoding="utf-8")
        (self.home / ".local/share/Steam/steamapps/compatdata/1111111111").mkdir()

        result = self.run_launch("--dry-run", "--config", str(other))

        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = self.home / ".local/state/Clank/depth-layer/vulkan/implicit_layer.d/depth_stream.json"
        self.assertIn("1111111111", manifest.read_text(encoding="utf-8"))

    def test_no_depth_starts_plain_dashboard(self) -> None:
        result = self.run_launch("--dry-run", "--no-depth", "--config", str(self.config))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.command(result), f"{sys.executable} -m clank --config {self.config}"
        )
        self.assertNotIn("DEPTH_STREAM_PATH", result.stdout)
        self.assertFalse(
            (self.home / ".local/state/Clank/depth-layer/vulkan/implicit_layer.d/depth_stream.json").exists()
        )

    def test_missing_layer_falls_back(self) -> None:
        self.layer.unlink()

        result = self.run_launch("--dry-run", "--config", str(self.config))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("depth layer not built", result.stderr)
        self.assertIn("starting the plain dashboard", result.stderr)
        self.assertNotIn("DEPTH_STREAM_PATH", result.stdout)

    def test_missing_compatdata_falls_back(self) -> None:
        empty_home = self.home / "empty"
        empty_home.mkdir()

        result = self.run_launch(
            "--dry-run", "--config", str(self.config), home=empty_home
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("compatdata path does not exist", result.stderr)
        self.assertIn("starting the plain dashboard", result.stderr)

    def test_unknown_flag_is_rejected(self) -> None:
        result = self.run_launch("--dry-run", "--nope")

        self.assertEqual(result.returncode, 2)
        self.assertIn("--nope", result.stderr)

    def test_run_sh_targets_the_launcher(self) -> None:
        script = RUN_SH.read_text(encoding="utf-8")
        self.assertIn("spike/e5/launch.py", script)
        self.assertNotIn("-m clank", script)


if __name__ == "__main__":
    unittest.main()
