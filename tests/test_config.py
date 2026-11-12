from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from clank.config import (
    ConfigError,
    body_pose_flags,
    load_config,
    missing_requirements,
    validate_config,
)


class ConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = load_config(None, env={})
        self.assertEqual(cfg.preview.width, 1280)
        self.assertEqual(cfg.preview.height, 720)
        self.assertEqual(cfg.preview.panel_height, 32)
        self.assertEqual(cfg.preview.fps, 30.0)
        self.assertEqual(cfg.dashboard.port, 8765)
        self.assertEqual(cfg.launch.shortcut_id, "14670405696621641728")
        self.assertEqual(cfg.launch.openxr_client_name, "OpenComposite")
        self.assertEqual(cfg.launch.viewport_scale_percentage, 100.0)
        self.assertEqual(cfg.launch.game_window, "Job Simulator")
        self.assertEqual(cfg.paths.xdotool_bin, "xdotool")
        self.assertTrue(cfg.paths.game_player_log.endswith("output_log.txt"))
        self.assertTrue(cfg.paths.pose_client.endswith("spike/c0/c0_client.py"))

    def test_vnc_defaults(self) -> None:
        cfg = load_config(None, env={})
        self.assertEqual(cfg.preview.vnc_port, 5901)
        self.assertEqual(cfg.preview.proxy_port, 6080)
        self.assertEqual(cfg.paths.novnc_bin, "novnc")
        self.assertEqual(cfg.paths.websockify_bin, "websockify")

    def test_preview_fps_must_be_whole(self) -> None:
        cfg = load_config(None, env={})
        with self.assertRaises(ConfigError):
            validate_config(
                replace(cfg, preview=replace(cfg.preview, fps=29.97))
            )

    def test_viewport_scale_is_bounded(self) -> None:
        cfg = load_config(None, env={})
        for value in (0.0, -10.0, 500.0):
            with self.assertRaises(ConfigError):
                validate_config(
                    replace(
                        cfg,
                        launch=replace(cfg.launch, viewport_scale_percentage=value),
                    )
                )

    def test_vnc_ports_must_differ(self) -> None:
        cfg = load_config(None, env={})
        with self.assertRaises(ConfigError):
            validate_config(
                replace(
                    cfg,
                    preview=replace(cfg.preview, proxy_port=cfg.preview.vnc_port),
                )
            )

    def test_vnc_ports_are_bounded(self) -> None:
        cfg = load_config(None, env={})
        for key in ("vnc_port", "proxy_port"):
            with self.subTest(key=key):
                with self.assertRaises(ConfigError):
                    validate_config(replace(cfg, preview=replace(cfg.preview, **{key: 0})))
                with self.assertRaises(ConfigError):
                    validate_config(replace(cfg, preview=replace(cfg.preview, **{key: 70000})))

    def test_vnc_port_boundaries_are_accepted(self) -> None:
        cfg = load_config(None, env={})
        for key in ("vnc_port", "proxy_port"):
            for value in (1, 65535):
                with self.subTest(key=key, value=value):
                    validate_config(
                        replace(cfg, preview=replace(cfg.preview, **{key: value}))
                    )

    def test_dashboard_port_conflicts_are_rejected(self) -> None:
        cfg = load_config(None, env={})
        for key in ("vnc_port", "proxy_port"):
            with self.subTest(key=key), self.assertRaises(ConfigError):
                validate_config(
                    replace(
                        cfg,
                        preview=replace(
                            cfg.preview, **{key: cfg.dashboard.port}
                        ),
                    )
                )
        validate_config(
            replace(
                cfg,
                dashboard=replace(cfg.dashboard, port=0),
                preview=replace(cfg.preview, vnc_port=8765),
            )
        )

    def test_novnc_web_root_is_derived(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            bin_dir = base / "novnc" / "bin"
            web = base / "novnc" / "share" / "webapps" / "novnc"
            (web / "core").mkdir(parents=True)
            (web / "core" / "rfb.js").write_text("")
            bin_dir.mkdir(parents=True)
            tool = bin_dir / "novnc"
            tool.write_text("")
            tool.chmod(0o755)
            cfg = load_config(None, env={"CLANK_PATHS_NOVNC_BIN": str(tool)})
            self.assertEqual(cfg.paths.novnc_web_root, str(web))

    def test_explicit_novnc_web_root_wins(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            bin_dir = base / "novnc" / "bin"
            web = base / "novnc" / "share" / "webapps" / "novnc"
            (web / "core").mkdir(parents=True)
            (web / "core" / "rfb.js").write_text("")
            bin_dir.mkdir(parents=True)
            tool = bin_dir / "novnc"
            tool.write_text("")
            tool.chmod(0o755)
            explicit = base / "explicit-web-root"
            explicit.mkdir()
            cfg = load_config(
                None,
                env={
                    "CLANK_PATHS_NOVNC_BIN": str(tool),
                    "CLANK_PATHS_NOVNC_WEB_ROOT": str(explicit),
                },
            )
            self.assertEqual(cfg.paths.novnc_web_root, str(explicit))

    def test_steam_extra_args_default_is_tuple(self) -> None:
        cfg = load_config(None, env={})
        self.assertIsInstance(cfg.launch.steam_extra_args, tuple)
        self.assertEqual(cfg.launch.steam_extra_args, ("-cef-disable-gpu",))

    def test_toml_and_env_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.toml"
            path.write_text("[preview]\nfps = 6\n\n[dashboard]\nport = 9000\n")
            cfg = load_config(path, env={"CLANK_PREVIEW_FPS": "12"})
            self.assertEqual(cfg.preview.fps, 12.0)
            self.assertEqual(cfg.dashboard.port, 9000)

    def test_unknown_toml_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.toml"
            path.write_text("[preview]\nnope = 1\n")
            with self.assertRaises(ConfigError):
                load_config(path, env={})

    def test_malformed_toml_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.toml"
            path.write_text("[preview\nfps = 6\n")
            with self.assertRaises(ConfigError):
                load_config(path, env={})

    def test_missing_explicit_config_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "missing.toml"
            with self.assertRaises(ConfigError):
                load_config(path, env={})

    def test_relative_paths_resolve_from_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            path = base / "config.toml"
            path.write_text('[paths]\npose_client = "pose.py"\n')
            cfg = load_config(path, env={})
            self.assertEqual(cfg.paths.pose_client, str((base / "pose.py").resolve()))

    def test_relative_novnc_web_root_resolves_from_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            path = base / "config.toml"
            path.write_text('[paths]\nnovnc_web_root = "web"\n')
            cfg = load_config(
                path,
                env={"CLANK_PATHS_NOVNC_BIN": "/nonexistent/novnc"},
            )
            self.assertEqual(cfg.paths.novnc_web_root, str((base / "web").resolve()))

    def test_explicit_base_dir_wins_for_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as raw, tempfile.TemporaryDirectory() as other:
            path = Path(raw) / "config.toml"
            path.write_text('[paths]\npose_client = "pose.py"\n')
            cfg = load_config(path, env={}, base_dir=other)
            self.assertEqual(cfg.paths.pose_client, str((Path(other) / "pose.py").resolve()))

    def test_unknown_env_key_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            load_config(None, env={"CLANK_NOPE_X": "1"})

    def test_wrong_type_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.toml"
            path.write_text('[preview]\nfps = "fast"\n')
            with self.assertRaises(ConfigError):
                load_config(path, env={})

    def test_bounds_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.toml"
            path.write_text("[preview]\npanel_height = 720\n")
            with self.assertRaises(ConfigError):
                load_config(path, env={})

    def test_empty_dashboard_host_is_rejected(self) -> None:
        cfg = load_config(None, env={})
        for host in ("", "   "):
            with self.subTest(host=host), self.assertRaises(ConfigError):
                validate_config(
                    replace(cfg, dashboard=replace(cfg.dashboard, host=host))
                )

    def test_wildcard_dashboard_host_is_rejected(self) -> None:
        cfg = load_config(None, env={})
        # S104: the wildcard literals are this test's INPUTS - it asserts they are
        # refused - so the noqa is for the linter, not a real binding.
        for host in ("0.0.0.0", "::", "[::]"):  # noqa: S104
            with self.subTest(host=host), self.assertRaises(ConfigError):
                validate_config(
                    replace(cfg, dashboard=replace(cfg.dashboard, host=host))
                )

    def test_specific_dashboard_host_is_accepted(self) -> None:
        cfg = load_config(None, env={})
        for host in ("127.0.0.1", "192.168.1.10"):
            with self.subTest(host=host):
                validate_config(
                    replace(cfg, dashboard=replace(cfg.dashboard, host=host))
                )

    def test_non_finite_floats_are_rejected(self) -> None:
        float_env_keys = [
            "CLANK_PREVIEW_FPS",
            "CLANK_LAUNCH_STEAM_SETTLE_SEC",
            "CLANK_LAUNCH_STEAM_READY_TIMEOUT_SEC",
            "CLANK_LAUNCH_GAME_PROCESS_TIMEOUT_SEC",
            "CLANK_LAUNCH_OPENXR_CLIENT_TIMEOUT_SEC",
            "CLANK_LAUNCH_STARTUP_TIMEOUT_SEC",
            "CLANK_TIMEOUTS_MONADO_STARTUP_SEC",
            "CLANK_TIMEOUTS_WESTON_STARTUP_SEC",
            "CLANK_TIMEOUTS_POSE_HANDSHAKE_SEC",
            "CLANK_TIMEOUTS_STOP_SEC",
        ]
        for env_key in float_env_keys:
            for raw in ("nan", "inf", "-inf"):
                with (
                    self.subTest(env_key=env_key, raw=raw),
                    self.assertRaises(ConfigError),
                ):
                    load_config(None, env={env_key: raw})

    def test_weston_siblings_are_derived(self) -> None:
        cfg = load_config(
            None,
            env={"CLANK_PATHS_WESTON_BIN": "/opt/weston/bin/weston"},
        )
        self.assertEqual(cfg.paths.weston_screenshooter_bin, "/opt/weston/bin/weston-screenshooter")
        self.assertEqual(cfg.paths.weston_debug_bin, "/opt/weston/bin/weston-debug")

    def test_missing_requirements_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            names = [
                "monado-service",
                "weston",
                "weston-screenshooter",
                "weston-debug",
                "steam",
                "websockify",
                "game.exe",
                "pose.py",
                "runtime.json",
            ]
            for name in names:
                path = base / name
                path.write_text("")
                path.chmod(0o755)
            novnc_bin = base / "novnc" / "bin"
            novnc_web = base / "novnc" / "share" / "webapps" / "novnc"
            (novnc_web / "core").mkdir(parents=True)
            (novnc_web / "core" / "rfb.js").write_text("")
            novnc_bin.mkdir(parents=True)
            novnc_tool = novnc_bin / "novnc"
            novnc_tool.write_text("")
            novnc_tool.chmod(0o755)
            env = {
                "CLANK_PATHS_MONADO_BIN": str(base / "monado-service"),
                "CLANK_PATHS_WESTON_BIN": str(base / "weston"),
                "CLANK_PATHS_STEAM_BIN": str(base / "steam"),
                "CLANK_PATHS_NOVNC_BIN": str(novnc_tool),
                "CLANK_PATHS_WEBSOCKIFY_BIN": str(base / "websockify"),
                "CLANK_PATHS_GAME_EXE": str(base / "game.exe"),
                "CLANK_PATHS_POSE_CLIENT": str(base / "pose.py"),
                "CLANK_PATHS_OPENXR_ACTIVE_RUNTIME": str(base / "runtime.json"),
                "CLANK_PATHS_SESSION_ROOT": str(base / "sessions"),
            }
            cfg = load_config(None, env=env)
            self.assertEqual(missing_requirements(cfg), [])
            (base / "steam").unlink()
            self.assertTrue(any("steam" in item for item in missing_requirements(cfg)))
            (base / "steam").write_text("")
            (base / "steam").chmod(0o755)
            novnc_tool.unlink()
            self.assertTrue(
                any(item.startswith("novnc (") for item in missing_requirements(cfg))
            )
            novnc_tool.write_text("")
            novnc_tool.chmod(0o755)
            (base / "websockify").unlink()
            self.assertTrue(
                any(
                    item.startswith("websockify (")
                    for item in missing_requirements(cfg)
                )
            )
            (base / "websockify").write_text("")
            (base / "websockify").chmod(0o755)
            (novnc_web / "core" / "rfb.js").unlink()
            self.assertTrue(
                any(
                    item.startswith("noVNC assets")
                    for item in missing_requirements(cfg)
                )
            )


if __name__ == "__main__":
    unittest.main()


class YawOffsetConfig(unittest.TestCase):
    """The robot's heading in the virtual room, as a config knob.

    The desk needs a 90 deg left turn and this body cannot twist that far (measured:
    -28.6 deg left, +54.3 right), so the heading is chosen rather than walked. It lives
    in config so a session is reproducible from a file rather than from shell history.
    """

    def test_the_default_is_zero_and_changes_nothing(self) -> None:
        self.assertEqual(load_config(None, env={}).body.yaw_offset_deg, 0.0)

    def test_it_reads_from_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.toml"
            path.write_text("[body]\nyaw_offset_deg = 90.0\n", encoding="utf-8")
            self.assertEqual(load_config(path, env={}).body.yaw_offset_deg, 90.0)

    def test_a_misspelt_key_is_still_refused(self) -> None:
        # The knob is only reproducible if a typo is loud. `yaw_offset_deg` and
        # `yaw_offset` differing silently would send 0 deg and look like a body fault.
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.toml"
            path.write_text("[body]\nyaw_offset = 90.0\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path, env={})


class FieldOfViewConfig(unittest.TestCase):
    """The declared headset field of view, as a config knob.

    Undeclared by default, because that is the behaviour every earlier measurement was
    taken under. Two angles or none: a lone horizontal angle would need an invented
    aspect ratio.
    """

    def test_nothing_is_declared_by_default(self) -> None:
        body = load_config(None, env={}).body
        self.assertEqual((body.fov_h_deg, body.fov_v_deg), (0.0, 0.0))

    def test_it_reads_from_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.toml"
            path.write_text("[body]\nfov_h_deg = 110.0\nfov_v_deg = 78.0\n", encoding="utf-8")
            body = load_config(path, env={}).body
            self.assertEqual((body.fov_h_deg, body.fov_v_deg), (110.0, 78.0))

    def test_half_a_field_of_view_is_refused_at_load(self) -> None:
        # Caught here rather than at body-mode start, so the session never opens.
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.toml"
            path.write_text("[body]\nfov_h_deg = 110.0\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path, env={})


class PlayerScaleConfig(unittest.TestCase):
    """How big the emulated player is reported to the game, as a config knob.

    Unscaled by default, because that is the behaviour every earlier measurement was taken
    under. The pivot is a numeric triple, which is the one shape the config loader's
    list-of-strings rule does not cover: a two-value "point" has to be refused at load
    rather than reach the relay as half a pivot.
    """

    def test_the_player_is_the_robot_as_it_is_by_default(self) -> None:
        body = load_config(None, env={}).body
        self.assertEqual(body.player_scale, 1.0)
        self.assertEqual(body.player_pivot, (0.0, 0.0, 0.0))
        self.assertTrue(body.player_scale_head)

    def test_it_reads_from_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.toml"
            path.write_text(
                "[body]\nplayer_scale = 1.43\nplayer_pivot = [0.0, -0.24, 0.0]\n"
                "player_scale_head = false\n",
                encoding="utf-8",
            )
            body = load_config(path, env={}).body
            self.assertEqual(body.player_scale, 1.43)
            self.assertEqual(body.player_pivot, (0.0, -0.24, 0.0))
            self.assertFalse(body.player_scale_head)

    def test_half_a_pivot_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.toml"
            path.write_text("[body]\nplayer_pivot = [0.0, -0.24]\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path, env={})

    def test_a_pivot_that_is_not_numbers_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.toml"
            path.write_text("[body]\nplayer_pivot = ['a', 'b', 'c']\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path, env={})

    def test_a_mirroring_scale_is_refused(self) -> None:
        for bad in (0.0, -1.43):
            with tempfile.TemporaryDirectory() as raw:
                path = Path(raw) / "config.toml"
                path.write_text(f"[body]\nplayer_scale = {bad}\n", encoding="utf-8")
                with self.assertRaises(ConfigError):
                    load_config(path, env={})


class BodyPoseFlags(unittest.TestCase):
    """The relay argv the config derives, so "0 means undeclared" has one owner."""

    def test_undeclared_produces_no_flags(self) -> None:
        self.assertEqual(body_pose_flags(load_config(None, env={}).body), [])

    def test_declared_produces_both_flags(self) -> None:
        body = replace(load_config(None, env={}).body, fov_h_deg=110.0, fov_v_deg=78.0)
        self.assertEqual(
            body_pose_flags(body), ["--fov-h-deg", "110", "--fov-v-deg", "78"]
        )
