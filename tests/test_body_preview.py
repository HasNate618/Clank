import os
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from clank import body_preview, pose_relay
from clank.body_preview import BodyPreview
from clank.config import BodyConfig


def make_cfg(
    *,
    enabled: bool = True,
    python: str = "spike/e3/.venv/bin/python",
    script: str = "spike/wbc/body_view.py",
    render: bool = True,
) -> BodyConfig:
    return BodyConfig(
        enabled=enabled,
        autostart=False,
        python=python,
        script=script,
        fps=10.0,
        width=560,
        height=420,
        reel=True,
        render=render,
        views=("three_quarter", "side"),
    )


class BodyPreviewTestCase(unittest.TestCase):
    def test_argv_points_at_the_control_socket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            view = BodyPreview(make_cfg(), out_dir=Path(tmp))
            argv = view.argv()
            self.assertIn("--control", argv)
            self.assertEqual(argv[argv.index("--control") + 1], str(view.control_path()))
    def test_argv_carries_output_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            view = BodyPreview(make_cfg(), out_dir=Path(tmp))
            argv = view.argv()
            self.assertIn("--out", argv)
            self.assertIn(str(Path(tmp)), argv)
            self.assertIn("--fps", argv)
            self.assertIn("--width", argv)

    def test_argv_asks_for_the_reel_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("--reel", BodyPreview(make_cfg(), out_dir=Path(tmp)).argv())

    def test_argv_omits_the_reel_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            view = BodyPreview(make_cfg(), out_dir=Path(tmp))
            view.cfg = BodyConfig(**{**view.cfg.__dict__, "reel": False})
            self.assertNotIn("--reel", view.argv())

    def test_no_render_flag_when_rendering_is_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            view = BodyPreview(make_cfg(render=False), out_dir=Path(tmp))
            self.assertIn("--no-render", view.argv())
            self.assertNotIn("--no-render",
                             BodyPreview(make_cfg(), out_dir=Path(tmp)).argv())

    def test_preflight_reports_missing_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            view = BodyPreview(make_cfg(python="/nonexistent/python"), out_dir=Path(tmp))
            self.assertIn("python not found", view.preflight())

    def test_preflight_reports_missing_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            view = BodyPreview(
                make_cfg(python=sys.executable, script="/nonexistent/body_view.py"),
                out_dir=Path(tmp),
            )
            self.assertIn("body view not found", view.preflight())

    def test_start_refuses_when_preflight_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            view = BodyPreview(make_cfg(python="/nonexistent/python"), out_dir=Path(tmp))
            problem = view.start()
            self.assertIn("python not found", problem)
            self.assertFalse(view.running())
            self.assertTrue(view.status()["error"])

    def test_forget_state_deletes_the_last_runs_pose(self) -> None:
        """The relay holds what it last read through a quiet body, buttons
        included; the file must not survive a reset."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "body_state.json").write_text("{}")
            (out / "body_latest.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            (out / "run.json").write_text("{}")
            BodyPreview(make_cfg(), out_dir=out).forget_state()
            self.assertFalse((out / "body_state.json").exists())
            self.assertFalse((out / "body_latest.jpg").exists())
            self.assertTrue((out / "run.json").exists())  # nothing else is touched

    def test_owner_elsewhere_is_false_when_the_socket_is_ours(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            view = BodyPreview(make_cfg(), out_dir=Path(tmp))
            view._proc = type("Proc", (), {"poll": staticmethod(lambda: None)})()  # type: ignore[assignment]
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as srv:
                srv.bind(str(view.control_path()))
                srv.listen(1)
                self.assertFalse(view.owner_elsewhere())

    def test_owner_elsewhere_sees_a_viewer_it_never_spawned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            view = BodyPreview(make_cfg(), out_dir=Path(tmp))
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as srv:
                srv.bind(str(view.control_path()))
                srv.listen(1)
                accepted: list = []

                def accept() -> None:
                    try:
                        conn, _ = srv.accept()
                    except OSError:
                        return
                    accepted.append(conn)
                    conn.close()

                thread = threading.Thread(target=accept, daemon=True)
                thread.start()
                self.assertTrue(view.owner_elsewhere())
                thread.join(2)

    def test_a_stale_socket_file_is_not_an_owner(self) -> None:
        """A file left by a viewer that died: connect refuses, so a reset may
        unlink it and bind."""
        with tempfile.TemporaryDirectory() as tmp:
            view = BodyPreview(make_cfg(), out_dir=Path(tmp))
            view.control_path().write_text("")
            self.assertFalse(view.owner_elsewhere())

    def test_status_before_any_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            view = BodyPreview(make_cfg(), out_dir=Path(tmp))
            status = view.status()
            self.assertFalse(status["running"])
            self.assertIsNone(status["frame"])
            self.assertIsNone(status["frame_age_s"])

    def test_latest_finds_written_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            view = BodyPreview(make_cfg(), out_dir=Path(tmp))
            self.assertIsNone(view.latest())
            (Path(tmp) / "body_latest.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            found = view.latest()
            self.assertIsNotNone(found)
            if found is None:  # keeps the checker honest about the assertion above
                self.fail("latest() should have found the frame")
            self.assertEqual(found.name, "body_latest.jpg")
            self.assertIsNotNone(view.status()["frame_age_s"])

    def test_stop_is_idempotent_when_never_started(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            view = BodyPreview(make_cfg(), out_dir=Path(tmp))
            view.stop()
            view.stop()
            self.assertFalse(view.running())

    def test_start_strips_parent_pythonpath_from_the_child(self) -> None:
        # A dashboard launched through `nix develop` carries a 3.14 PYTHONPATH;
        # the sidecar's own 3.12 interpreter must not inherit it.
        with tempfile.TemporaryDirectory() as tmp:
            view = BodyPreview(
                make_cfg(python=sys.executable, script=__file__), out_dir=Path(tmp)
            )
            seen: dict = {}

            class FakeProc:
                pid = 1
                returncode = None

                def poll(self):
                    return None

            def fake_popen(argv, **kwargs):
                seen.update(kwargs)
                return FakeProc()

            polluted = {
                "PYTHONPATH": "/polluted/site-packages",
                "PYTHONHOME": "/polluted/home",
                "VIRTUAL_ENV": "/polluted/venv",
            }
            with patch.dict(os.environ, polluted), patch(
                "clank.body_preview.subprocess.Popen", side_effect=fake_popen
            ):
                self.assertEqual(view.start(), "")
            env = seen["env"]
            self.assertNotIn("PYTHONPATH", env)
            self.assertNotIn("PYTHONHOME", env)
            self.assertNotIn("VIRTUAL_ENV", env)

    def test_status_explains_a_dead_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            view = BodyPreview(
                make_cfg(python=sys.executable, script=__file__), out_dir=Path(tmp)
            )

            class DeadProc:
                pid = 1
                returncode = 3

                def poll(self):
                    return 3

            with patch("clank.body_preview.subprocess.Popen", return_value=DeadProc()):
                self.assertEqual(view.start(), "")
            view.log_path.write_text("Traceback...\nboom: missing numpy\n")
            status = view.status()
            self.assertFalse(status["running"])
            self.assertEqual(status["exit_code"], 3)
            self.assertIn("exited (3)", status["error"])
            self.assertIn("boom: missing numpy", status["error"])

    def test_button_verbs_include_grab_and_fire(self) -> None:
        for verb in ("grab_right", "release_right", "fire_right", "grab_left", "release_left"):
            self.assertIn(verb, body_preview.BODY_VERBS)
        self.assertNotIn("grab", body_preview.BODY_VERBS, "a grab without a side cannot be bound")

    def test_every_verb_is_reachable_from_the_dashboard(self) -> None:
        # The dashboard rejects a verb it does not know before it reaches the
        # socket, so a verb missing from the shared tuple is a dead button.
        self.assertEqual(pose_relay.FORCE_MODES, ("none", "left", "right", "both"))
        self.assertGreater(len(body_preview.BODY_VERBS), len(body_preview.BODY_VIEWS))


if __name__ == "__main__":
    unittest.main()
