from __future__ import annotations

import array
import contextlib
import http.client
import io
import json
import os
import socket
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

from clank import dashboard, pose_relay
from clank.body_preview import BodyPreview
from clank.config import load_config
from clank.dashboard import DashboardHandler, DashboardServer, OverlayBoard, shutdown_all
from clank.manual_input import ManualInputError, ManualInputUnavailable
from clank.orchestrator import ConflictError, MODES, PreflightError, SessionManager


class FakeManager:
    LOG_FILES = SessionManager.LOG_FILES

    def __init__(self, base: Path) -> None:
        self.base = base
        self.state = "stopped"
        self.session_id = "20261025T223900Z-a1b2c3"
        self.started_at = None
        self.mode = "hold"
        self.last_error = None
        self.fail_start = False
        self.start_calls = 0
        self.stop_calls = 0
        self.manual_calls: list[dict] = []
        self.logs = {
            name: base / f"{name}.log"
            for name in ("session", "monado", "weston", "proxy", "steam", "pose")
        }
        for path in self.logs.values():
            path.write_text("")

    def start(self) -> None:
        self.start_calls += 1
        if self.fail_start and self.start_calls == 1:
            raise PreflightError("Close the host Steam client before starting a hidden C0 session.")
        if self.state not in ("stopped", "failed"):
            raise ConflictError("session is not stopped")
        self.state = "starting"

    def stop(self) -> None:
        self.stop_calls += 1
        self.state = "stopping"

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        if self.state != "running":
            raise ConflictError("session is not running")
        self.mode = mode

    def manual_input(self, payload: dict) -> bool:
        if self.state != "running":
            raise ConflictError("session is not running")
        if self.mode != "manual":
            raise ConflictError("manual mode is not active")
        self.manual_calls.append(payload)
        return bool(payload.get("armed"))

    def state_value(self) -> str:
        return self.state

    def log_path(self, name: str) -> Path | None:
        filename = self.LOG_FILES.get(name)
        if filename is None:
            return None
        return self.base / filename

    def snapshot(self) -> dict:
        transport_alive = self.state in ("starting", "running", "stopping")
        return {
            "state": self.state,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "mode": self.mode,
            "processes": {
                "monado": {"alive": True, "pid": 1},
                "weston": {"alive": True, "pid": 2, "display": ":2"},
                "steam": {"alive": True, "pid": 3},
                "game": {"alive": True, "pid": 4, "exe": "SUPERHOT_VR.exe"},
                "pose": {"alive": True, "pid": 5},
            },
            "openxr_client": "Unity Application",
            "preview": {
                "mode": "vnc",
                "width": 1280,
                "height": 688,
                "fps_target": 30,
                "vnc_port": 5901,
                "proxy_port": 6080,
                "vnc_alive": transport_alive,
                "proxy_alive": transport_alive,
            },
            "last_error": self.last_error,
            "logs": {name: str(path) for name, path in self.logs.items()},
        }


class DashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        (self.base / "index.html").write_text("<h1>C0</h1>")
        (self.base / "style.css").write_text("body{}")
        (self.base / "app.js").write_text("console.log('x')")
        self.novnc_root = self.base / "novnc"
        (self.novnc_root / "core").mkdir(parents=True)
        (self.novnc_root / "core" / "rfb.js").write_bytes(b"/* test noVNC */")
        self.manager = FakeManager(self.base)
        cfg = load_config(None, env={})
        self.cfg = replace(cfg, paths=replace(cfg.paths, novnc_web_root=str(self.novnc_root)))
        self.server = DashboardServer(("127.0.0.1", 0), self.manager, self.cfg, self.base)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.temp.cleanup()

    def request(self, method: str, path: str, body=None, raw: bytes | None = None, headers=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        request_headers = {"Content-Type": "application/json"}
        request_headers.update(headers or {})
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data,
            method=method,
            headers=request_headers,
        )
        try:
            # Local test server only; the scheme is fixed above.
            with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, exc.read()
            finally:
                exc.close()

    def request_with_headers(self, path: str):
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, exc.headers, exc.read()
            finally:
                exc.close()

    def raw_request(self, method: str, path: str, headers: dict[str, str] | None = None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
        try:
            connection.putrequest(method, path)
            for name, value in (headers or {}).items():
                connection.putheader(name, value)
            connection.endheaders()
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def test_state(self) -> None:
        self.manager.state = "running"
        status, body = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["openxr_client"], "Unity Application")
        preview = payload["preview"]
        self.assertEqual(preview["mode"], "vnc")
        self.assertEqual(preview["height"], 688)
        self.assertEqual(preview["vnc_port"], 5901)
        self.assertEqual(preview["proxy_port"], 6080)
        self.assertTrue(preview["vnc_alive"])
        self.assertTrue(preview["proxy_alive"])

    def test_preview_config(self) -> None:
        status, body = self.request("GET", "/api/preview/config")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["mode"], "vnc")
        self.assertEqual(payload["vnc_port"], 5901)
        self.assertEqual(payload["proxy_port"], 6080)
        self.assertEqual(payload["ws_url"], "ws://127.0.0.1:6080/")
        self.assertTrue(payload["username"])
        self.assertFalse(payload["available"])
        self.assertFalse(payload["transport_alive"])
        self.assertTrue(payload["assets_available"])

    def test_novnc_asset_and_traversal(self) -> None:
        status, body = self.request("GET", "/novnc/core/rfb.js")
        self.assertEqual(status, 200)
        self.assertIn(b"test noVNC", body)
        paths = (
            "/novnc/../../etc/passwd",
            "/novnc/%2e%2e%2fetc%2fpasswd",
            "/novnc/%2fetc%2fpasswd",
            "/novnc//etc/passwd",
        )
        for path in paths:
            with self.subTest(path=path):
                status, _ = self.request("GET", path)
                self.assertEqual(status, 404)
        outside = self.base / "outside.txt"
        outside.write_text("secret")
        link = self.novnc_root / "core" / "outside.js"
        with self.subTest(path="/novnc/core/outside.js"):
            try:
                os.symlink(outside, link)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            status, _ = self.request("GET", "/novnc/core/outside.js")
            self.assertEqual(status, 404)

    def test_novnc_rejects_dotfiles(self) -> None:
        (self.novnc_root / ".env").write_text("SECRET=1")
        status, _ = self.request("GET", "/novnc/.env")
        self.assertEqual(status, 404)

    def test_novnc_mime_types(self) -> None:
        status, headers, _ = self.request_with_headers("/novnc/core/rfb.js")
        self.assertEqual(status, 200)
        self.assertIn(
            headers.get_content_type(),
            ("text/javascript", "application/javascript"),
        )
        (self.novnc_root / "blob.bin").write_bytes(b"\x00\x01")
        (self.novnc_root / "blob.unknownext").write_bytes(b"\x02\x03")
        for name in ("blob.bin", "blob.unknownext"):
            with self.subTest(name=name):
                status, headers, _ = self.request_with_headers(f"/novnc/{name}")
                self.assertEqual(status, 200)
                self.assertEqual(headers.get_content_type(), "application/octet-stream")

    def test_removed_frame_routes_return_404(self) -> None:
        for path in ("/preview.png", "/api/frame.png"):
            with self.subTest(path=path):
                status, _ = self.request("GET", path)
                self.assertEqual(status, 404)

    def test_novnc_missing_assets(self) -> None:
        original = self.server.config
        try:
            for value in ("", str(self.base / "missing")):
                with self.subTest(novnc_web_root=value):
                    self.server.config = replace(
                        original,
                        paths=replace(original.paths, novnc_web_root=value),
                    )
                    status, body = self.request("GET", "/novnc/core/rfb.js")
                    self.assertEqual(status, 404)
                    self.assertIn("error", json.loads(body))
        finally:
            self.server.config = original

    def test_static_index(self) -> None:
        status, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"C0", body)

    def test_ui_assets_are_served(self) -> None:
        for path in ("/style.css", "/app.js"):
            status, _ = self.request("GET", path)
            self.assertEqual(status, 200)

    def test_start_returns_202_then_conflict(self) -> None:
        status, _ = self.request("POST", "/api/session/start")
        self.assertEqual(status, 202)
        status, body = self.request("POST", "/api/session/start")
        self.assertEqual(status, 409)
        self.assertIn("error", json.loads(body))

    def test_preflight_failure_returns_400(self) -> None:
        self.manager.fail_start = True
        status, body = self.request("POST", "/api/session/start")
        self.assertEqual(status, 400)
        self.assertIn("host Steam", json.loads(body)["error"])

    def test_stop_returns_202(self) -> None:
        status, _ = self.request("POST", "/api/session/stop")
        self.assertEqual(status, 202)
        self.assertEqual(self.manager.stop_calls, 1)

    def test_mode_validation(self) -> None:
        status, _ = self.request("POST", "/api/session/mode", {"mode": "yaw"})
        self.assertEqual(status, 409)
        self.manager.state = "running"
        status, _ = self.request("POST", "/api/session/mode", {"mode": "yaw"})
        self.assertEqual(status, 200)
        self.assertEqual(self.manager.mode, "yaw")
        status, _ = self.request("POST", "/api/session/mode", {"mode": "nope"})
        self.assertEqual(status, 400)

    def test_manual_input_requires_running_manual_mode(self) -> None:
        body = {"client": "browser-1234", "armed": True, "keys": [], "buttons": []}
        status, _ = self.request("POST", "/api/manual/input", body)
        self.assertEqual(status, 409)
        self.manager.state = "running"
        self.manager.mode = "hold"
        status, _ = self.request("POST", "/api/manual/input", body)
        self.assertEqual(status, 409)

    def test_manual_input_forwards_when_armed(self) -> None:
        self.manager.state = "running"
        self.manager.mode = "manual"
        body = {
            "client": "browser-1234",
            "armed": True,
            "keys": ["w"],
            "buttons": [],
            "dx": 1,
            "dy": 2,
            "wheel": 0,
        }
        status, response = self.request("POST", "/api/manual/input", body)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(response)["armed"], True)
        self.assertEqual(self.manager.manual_calls, [body])

    def test_manual_input_validation_errors_are_400(self) -> None:
        self.manager.state = "running"
        self.manager.mode = "manual"
        with patch.object(self.manager, "manual_input", side_effect=ManualInputError("bad key")):
            status, body = self.request("POST", "/api/manual/input", {"client": "browser-1234", "armed": True})
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"], "bad key")

    def test_manual_input_unavailable_is_503(self) -> None:
        self.manager.state = "running"
        self.manager.mode = "manual"
        with patch.object(self.manager, "manual_input", side_effect=ManualInputUnavailable("missing")):
            status, body = self.request("POST", "/api/manual/input", {"client": "browser-1234", "armed": True})
        self.assertEqual(status, 503)
        self.assertIn("missing", json.loads(body)["error"])

    def test_manual_input_oversized_body_is_413(self) -> None:
        self.manager.state = "running"
        self.manager.mode = "manual"
        status, body = self.request("POST", "/api/manual/input", raw=b"x" * 1_000_001)
        self.assertEqual(status, 413)
        self.assertEqual(json.loads(body)["error"], "request body too large")

    def test_manual_input_rejects_cross_origin(self) -> None:
        self.manager.state = "running"
        self.manager.mode = "manual"
        status, _ = self.request(
            "POST",
            "/api/manual/input",
            {"client": "browser-1234", "armed": False},
            headers={"Origin": "http://evil.example"},
        )
        self.assertEqual(status, 403)

    def test_logs_allowlist(self) -> None:
        self.manager.logs["monado"].write_text("hello monado")
        status, body = self.request("GET", "/api/logs/monado")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"hello monado")
        status, _ = self.request("GET", "/api/logs/nope")
        self.assertEqual(status, 404)

    def test_logs_carry_the_brain_diary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"CLANK_BRAIN_LOG": str(Path(tmp) / "m.jsonl")}):
            from clank import brainlog

            brainlog.log("think", "I will pick up the card", echo=False,
                         path=Path(tmp) / "m.jsonl")
            status, body = self.request("GET", "/api/logs")
        self.assertEqual(status, 200)
        logs = json.loads(body)["logs"]
        self.assertIn("brain", logs)
        self.assertTrue(any("I will pick up the card" in line for line in logs["brain"]))

    def test_logs_omit_the_brain_source_when_the_diary_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"CLANK_BRAIN_LOG": str(Path(tmp) / "none.jsonl")}):
            status, body = self.request("GET", "/api/logs")
        self.assertEqual(status, 200)
        self.assertNotIn("brain", json.loads(body)["logs"])

    def test_an_unreadable_diary_does_not_break_the_log_panel(self) -> None:
        blocked = Path(self.temp.name) / "blocked"
        blocked.mkdir()
        with patch.dict(os.environ, {"CLANK_BRAIN_LOG": str(blocked)}), \
                patch("clank.brainlog.tail", side_effect=OSError("nope")):
            status, body = self.request("GET", "/api/logs")
        self.assertEqual(status, 200)
        self.assertIn("logs", json.loads(body))

    def test_logs_weston_stdout_is_not_public(self) -> None:
        (self.base / "weston-stdout.log").write_text("internal weston stdout")
        status, body = self.request("GET", "/api/logs/weston-stdout")
        self.assertEqual(status, 404)
        self.assertNotIn(b"internal weston stdout", body)

    def test_logs_reject_traversal(self) -> None:
        self.manager.logs["monado"].write_text("secret")
        for path in ("/api/logs/../monado", "/api/logs/%2e%2e%2fmonado"):
            with self.subTest(path=path):
                status, body = self.raw_request("GET", path)
                self.assertEqual(status, 404)
                self.assertNotIn(b"secret", body)

    def test_logs_missing_or_unreadable_returns_404(self) -> None:
        path = self.manager.logs["monado"]
        path.unlink()
        status, _ = self.request("GET", "/api/logs/monado")
        self.assertEqual(status, 404)
        path.mkdir()
        status, _ = self.request("GET", "/api/logs/monado")
        self.assertEqual(status, 404)

    def test_logs_returns_last_200_lines(self) -> None:
        self.manager.logs["monado"].write_text("\n".join(f"line {index}" for index in range(250)))
        status, body = self.request("GET", "/api/logs/monado")
        self.assertEqual(status, 200)
        lines = body.decode().splitlines()
        self.assertEqual(len(lines), 200)
        self.assertEqual(lines[0], "line 50")
        self.assertEqual(lines[-1], "line 249")

    def test_merged_log_stream_carries_every_source(self) -> None:
        self.manager.logs["monado"].write_text("monado says hi\nsecond line")
        self.manager.logs["weston"].write_text("weston says hi")
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"CLANK_BRAIN_LOG": str(Path(tmp) / "none.jsonl")}):
            status, body = self.request("GET", "/api/logs")
        self.assertEqual(status, 200)
        logs = json.loads(body)["logs"]
        self.assertEqual(set(logs), set(FakeManager.LOG_FILES))
        self.assertEqual(logs["monado"], ["monado says hi", "second line"])
        self.assertEqual(logs["weston"], ["weston says hi"])

    def test_the_brain_is_an_extra_source_beyond_the_log_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            diary = Path(tmp) / "m.jsonl"
            from clank import brainlog

            brainlog.log("think", "I will pick up the card", echo=False, path=diary)
            with patch.dict(os.environ, {"CLANK_BRAIN_LOG": str(diary)}):
                status, body = self.request("GET", "/api/logs")
        self.assertEqual(status, 200)
        logs = json.loads(body)["logs"]
        self.assertLessEqual(set(FakeManager.LOG_FILES), set(logs))
        self.assertEqual(set(logs) - set(FakeManager.LOG_FILES), {"brain"})

    def test_merged_log_stream_skips_missing_files(self) -> None:
        self.manager.logs["steam"].unlink()
        status, body = self.request("GET", "/api/logs")
        self.assertEqual(status, 200)
        self.assertNotIn("steam", json.loads(body)["logs"])

    def test_malformed_json(self) -> None:
        status, _ = self.request("POST", "/api/session/mode", raw=b"{")
        self.assertEqual(status, 400)

    def test_brain_messages_returns_diary_rows(self) -> None:
        diary = self.base / "brain.jsonl"
        diary.write_text(
            '{"t": 1.0, "iso": "11:27:33", "kind": "heard", "text": "Hello human"}\n'
            '{"t": 2.0, "iso": "11:27:34", "kind": "think", "text": "I will grab the card"}\n',
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"CLANK_BRAIN_LOG": str(diary)}):
            status, body = self.request("GET", "/api/brain/messages")
        self.assertEqual(status, 200)
        kinds = [row["kind"] for row in json.loads(body)["messages"]]
        self.assertEqual(kinds, ["heard", "think"])

    def test_brain_messages_without_diary_is_empty_not_an_error(self) -> None:
        absent = self.base / "absent-brain.jsonl"
        with patch.dict(os.environ, {"CLANK_BRAIN_LOG": str(absent)}):
            status, body = self.request("GET", "/api/brain/messages")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["messages"], [])

    def test_overlay_round_trip(self) -> None:
        claim = {
            "noun": "the JobBot ID card",
            "box": [0.70, 0.70, 0.90, 0.90],
            "pixel": [768.0, 432.0],
            "size": [960, 540],
            "spread": 34.1,
            "present": True,
        }
        status, body = self.request("POST", "/api/overlay", claim)
        self.assertEqual(status, 202)
        self.assertTrue(json.loads(body)["ok"])
        status, body = self.request("GET", "/api/overlay")
        self.assertEqual(status, 200)
        items = json.loads(body)["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["noun"], "the JobBot ID card")
        self.assertEqual(items[0]["box"], claim["box"])
        self.assertEqual(items[0]["pixel"], claim["pixel"])
        self.assertEqual(items[0]["spread"], 34.1)
        self.assertTrue(items[0]["present"])
        self.assertGreaterEqual(items[0]["age_s"], 0)

    def test_overlay_without_claims_is_an_empty_list(self) -> None:
        status, body = self.request("GET", "/api/overlay")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["items"], [])

    def test_overlay_rejects_an_inverted_box(self) -> None:
        status, body = self.request("POST", "/api/overlay", {
            "noun": "card",
            "box": [0.9, 0.9, 0.2, 0.2],
            "pixel": [1.0, 2.0],
            "size": [960, 540],
        })
        self.assertEqual(status, 400)
        self.assertIn("box", json.loads(body)["error"])

    def test_overlay_rejects_a_missing_pixel(self) -> None:
        status, body = self.request("POST", "/api/overlay", {
            "noun": "card",
            "box": [0.1, 0.1, 0.2, 0.2],
            "size": [960, 540],
        })
        self.assertEqual(status, 400)
        self.assertIn("pixel", json.loads(body)["error"])

    def test_live_depth_is_raw_float32_metres(self) -> None:
        frame = {
            "width": 2,
            "height": 2,
            "depth": array.array("f", [0.4, 1.2, 3.0, 0.0]),
            "color": None,
            "color_width": 0,
            "color_height": 0,
        }
        with patch("clank.dashboard.latest", return_value=frame):
            status, headers, body = self.request_with_headers("/api/preview/live.depth")
        self.assertEqual(status, 200)
        self.assertEqual(headers["X-Frame-Width"], "2")
        self.assertEqual(headers["X-Frame-Height"], "2")
        self.assertEqual(len(body), 2 * 2 * 4)
        values = array.array("f")
        values.frombytes(body)
        self.assertAlmostEqual(values[1], 1.2, places=5)

    def test_live_depth_without_a_frame_is_404(self) -> None:
        with patch("clank.dashboard.latest", return_value=None):
            status, _ = self.request("GET", "/api/preview/live.depth")
        self.assertEqual(status, 404)

    def test_negative_content_length_returns_400(self) -> None:
        status, _ = self.raw_request(
            "POST",
            "/api/session/mode",
            {"Content-Type": "application/json", "Content-Length": "-1"},
        )
        self.assertEqual(status, 400)

    def test_mode_exception_returns_500_without_detail(self) -> None:
        self.manager.state = "running"
        with patch.object(self.manager, "set_mode", side_effect=RuntimeError("boom")):
            status, body = self.request("POST", "/api/session/mode", {"mode": "yaw"})
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(body)["error"], "internal server error")
        self.assertNotIn(b"boom", body)

    def test_stop_exception_returns_500_without_detail(self) -> None:
        with patch.object(self.manager, "stop", side_effect=RuntimeError("boom")):
            status, body = self.request("POST", "/api/session/stop")
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(body)["error"], "internal server error")
        self.assertNotIn(b"boom", body)

    def test_start_exception_returns_500_without_detail(self) -> None:
        with patch.object(self.manager, "start", side_effect=RuntimeError("boom")):
            status, body = self.request("POST", "/api/session/start")
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(body)["error"], "internal server error")
        self.assertNotIn(b"boom", body)

    def test_stalled_body_returns_400_json(self) -> None:
        with patch.object(DashboardHandler, "timeout", 0.1):
            sock = socket.create_connection(("127.0.0.1", self.port), timeout=3)
            response = b""
            try:
                sock.sendall(
                    b"POST /api/session/mode HTTP/1.1\r\n"
                    b"Host: 127.0.0.1\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Content-Length: 10\r\n\r\n"
                )
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    response += chunk
            finally:
                sock.close()
        self.assertIn(b" 400 ", response)
        self.assertIn(b"request body timeout", response)

    def test_unknown_route(self) -> None:
        status, _ = self.request("GET", "/nope")
        self.assertEqual(status, 404)


class ShippedStaticAssetsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.manager = FakeManager(Path(self.temp.name))
        self.cfg = load_config(None, env={})
        static_dir = Path(__file__).resolve().parents[1] / "clank" / "static"
        self.server = DashboardServer(("127.0.0.1", 0), self.manager, self.cfg, static_dir)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.temp.cleanup()

    def request(self, path: str):
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, exc.read()
            finally:
                exc.close()

    def test_shipped_assets_are_served(self) -> None:
        for path in ("/", "/app.js", "/style.css"):
            with self.subTest(path=path):
                status, body = self.request(path)
                self.assertEqual(status, 200)
                self.assertTrue(body)
        _, index = self.request("/")
        self.assertIn(b"/app.js", index)
        self.assertIn(b"/style.css", index)
        self.assertIn(b'id="mode-toggle"', index)
        self.assertIn(b'id="theme-toggle"', index)
        self.assertIn(b'id="brain-messages"', index)
        self.assertIn(b'id="preview-overlay"', index)
        self.assertIn(b'data-layout="side"', index)

    def test_run_controls_are_shipped(self) -> None:
        _, index = self.request("/")
        for element in (
            "run-state",
            "run-spinner",
            "start-run",
            "stop-run",
            "brain-run",
            "preview-wait-text",
            "preview-spinner",
        ):
            with self.subTest(element=element):
                self.assertIn(f'id="{element}"'.encode(), index)
        self.assertIn(b"Shut down", index)
        _, js = self.request("/app.js")
        code = js.decode()
        self.assertIn("/api/run/start", code)
        self.assertIn("/api/run/stop", code)
        self.assertIn("run-spinner", code)
        self.assertIn("run.detail", code)
        self.assertIn("run.error", code)

    def test_spinner_is_styled_and_respects_reduced_motion(self) -> None:
        _, css = self.request("/style.css")
        text = css.decode()
        self.assertIn("@keyframes clank-spin", text)
        self.assertIn("prefers-reduced-motion", text)

    def test_vendored_fonts_are_served(self) -> None:
        for name in ("inter-var.woff2", "plex-mono-400.woff2", "plex-mono-600.woff2", "fredoka-600.woff2"):
            with self.subTest(name=name):
                status, body = self.request(f"/fonts/{name}")
                self.assertEqual(status, 200)
                self.assertTrue(body)

    def test_font_route_rejects_traversal(self) -> None:
        for path in ("/fonts/%2e%2e%2fstyle.css", "/fonts/..%2fstyle.css", "/fonts/"):
            with self.subTest(path=path):
                status, _ = self.request(path)
                self.assertEqual(status, 404)


class _FakeBody:
    """Stands in for BodyPreview: only what the command route touches."""

    out_dir: Path | None = None

    def __init__(self, path: Path, running: bool = True) -> None:
        self.path = path
        self._running = running

    def running(self) -> bool:
        return self._running

    def control_path(self) -> Path:
        return self.path

    def frame(self, view=None):
        name = "body_latest.jpg" if not view else f"body_{view}.jpg"
        candidate = getattr(self, "out_dir", Path(".")) / name
        return candidate if candidate.is_file() else None

    def status(self) -> dict:
        return {"enabled": True, "running": self._running}


class _VerbSocket:
    """A one-connection Unix socket that answers like the real VerbServer."""

    def __init__(self, path: Path, reply: dict) -> None:
        self.path = path
        self.reply = reply
        self.seen: list[dict] = []
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(str(path))
        self.sock.listen(2)
        self.sock.settimeout(5)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            conn, _ = self.sock.accept()
        except (TimeoutError, OSError):
            return
        with conn:
            buf = b""
            while b"\n" not in buf:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                buf += chunk
            line, _ = buf.split(b"\n", 1)
            self.seen.append(json.loads(line.decode()))
            conn.sendall((json.dumps(self.reply) + "\n").encode())

    def close(self) -> None:
        self.sock.close()


class BodyCommandRouteTestCase(unittest.TestCase):
    """POST /api/body/command is the only path from a button to the body."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = load_config(None, base_dir=Path(self.tmp.name))
        self.base = Path(self.tmp.name) / "static"
        self.base.mkdir(exist_ok=True)
        self.manager = FakeManager(self.base)
        self.sock_path = Path(self.tmp.name) / "control.sock"

    def _serve(self, body, doc=None) -> tuple[int, dict]:
        server = DashboardServer(("127.0.0.1", 0), self.manager, self.cfg, self.base, body=body)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                                  daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        payload = json.dumps(doc or {}).encode()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/api/body/command", payload,
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        try:
            return response.status, json.loads(raw)
        except json.JSONDecodeError:
            return response.status, {"raw": raw.decode(errors="replace")}

    def test_verb_reaches_the_body_socket(self) -> None:
        verbs = _VerbSocket(self.sock_path, {"ok": True, "verb": "punch", "seq": 3})
        self.addCleanup(verbs.close)
        status, doc = self._serve(_FakeBody(self.sock_path), {"verb": "punch"})
        self.assertEqual(status, 200)
        self.assertEqual(doc.get("verb"), "punch")
        self.assertEqual(verbs.seen, [{"verb": "punch"}])

    def test_unknown_verb_is_rejected_before_the_socket(self) -> None:
        status, doc = self._serve(_FakeBody(self.sock_path), {"verb": "teleport"})
        self.assertEqual(status, 400)
        self.assertIn("unknown verb", doc.get("error", ""))

    def test_command_reaches_a_viewer_the_dashboard_did_not_start(self) -> None:
        # The handle only knows about processes this dashboard spawned. A viewer
        # started another way (or one that outlived a restart) holds the socket
        # perfectly well, and refusing on the handle alone broke the buttons.
        verbs = _VerbSocket(self.sock_path, {"ok": True, "verb": "look_left", "seq": 1})
        self.addCleanup(verbs.close)
        status, doc = self._serve(_FakeBody(self.sock_path, running=False), {"verb": "look_left"})
        self.assertEqual(status, 200)
        self.assertEqual(doc.get("verb"), "look_left")
        self.assertEqual(verbs.seen, [{"verb": "look_left"}])

    def test_dead_body_socket_returns_503_with_the_reason(self) -> None:
        status, doc = self._serve(_FakeBody(self.sock_path, running=False), {"verb": "punch"})
        self.assertEqual(status, 503)
        self.assertIn("body control unavailable", doc.get("error", ""))

    def test_view_switch_reaches_the_body_socket(self) -> None:
        verbs = _VerbSocket(self.sock_path, {"ok": True, "view": "side"})
        self.addCleanup(verbs.close)
        status, doc = self._serve(_FakeBody(self.sock_path), {"view": "side"})
        self.assertEqual(status, 200)
        self.assertEqual(doc.get("view"), "side")
        self.assertEqual(verbs.seen, [{"view": "side"}])

    def test_unknown_view_is_rejected(self) -> None:
        status, doc = self._serve(_FakeBody(self.sock_path), {"view": "drone"})
        self.assertEqual(status, 400)

    def test_view_param_serves_that_angle(self) -> None:
        import tempfile as _tf
        from pathlib import Path as _P
        with _tf.TemporaryDirectory() as tmp:
            for name in ("body_three_quarter.jpg", "body_side.jpg"):
                (_P(tmp) / name).write_bytes(b"\xff\xd8\xff\xd9")
            body = _FakeBody(_P(tmp) / "control.sock")
            body.out_dir = _P(tmp)
            server = DashboardServer(("127.0.0.1", 0), self.manager, self.cfg, self.base,
                                     body=cast(BodyPreview | None, body))
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever,
                                      kwargs={"poll_interval": 0.05}, daemon=True)
            thread.start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
            import http.client as _hc
            for view, want in (("side", 200), ("back", 404), ("drone", 400)):
                conn = _hc.HTTPConnection("127.0.0.1", port, timeout=5)
                conn.request("GET", f"/api/body.jpg?view={view}")
                response = conn.getresponse()
                response.read()
                conn.close()
                self.assertEqual(response.status, want, view)

    def test_missing_body_returns_400(self) -> None:
        status, _ = self._serve(None, {"verb": "punch"})
        self.assertEqual(status, 400)


class OverlayBoardTests(unittest.TestCase):
    """The board itself, without a server in the way."""

    def claim(self, noun: str = "card") -> dict:
        return {
            "noun": noun,
            "box": [0.0, 0.0, 1.0, 1.0],
            "pixel": [10.0, 10.0],
            "size": [960, 540],
        }

    def test_items_expire(self) -> None:
        board = OverlayBoard(ttl_s=0.0)
        board.add(self.claim())
        # A claim that outlives its evidence is a lie about what the robot sees, so the
        # board forgets rather than growing.
        self.assertEqual(board.snapshot(), [])

    def test_the_oldest_claims_are_dropped_first(self) -> None:
        board = OverlayBoard(ttl_s=60.0, limit=2)
        for noun in ("a", "b", "c"):
            board.add(self.claim(noun))
        self.assertEqual([item["noun"] for item in board.snapshot()], ["b", "c"])

    def test_clear_forgets_every_claim(self) -> None:
        board = OverlayBoard(ttl_s=60.0)
        board.add(self.claim("card"))
        board.clear()
        # A new run starts with an empty picture: the last attempt's boxes describe
        # a scene that no longer exists.
        self.assertEqual(board.snapshot(), [])

    def test_a_claim_must_be_finite_and_ordered(self) -> None:
        board = OverlayBoard()
        with self.assertRaises(ValueError):
            board.add({**self.claim(), "box": [0.5, 0.5, 0.5, 0.9]})
        with self.assertRaises(ValueError):
            board.add({**self.claim(), "pixel": [float("nan"), 1.0]})
        with self.assertRaises(ValueError):
            board.add({**self.claim(), "size": [0, 540]})


if __name__ == "__main__":
    unittest.main()


class MenuRouteTestCase(unittest.TestCase):
    """POST /api/menu is the dashboard's press of the controller's menu button.

    The button is not on the body: no joint of the simulated G1 can press it, so the request goes
    to the relay as a file and the relay turns it into a short system_click. This test pins that
    the dashboard writes a request and never pretends to have pressed a button itself.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = load_config(None, base_dir=Path(self.tmp.name))
        self.base = Path(self.tmp.name) / "static"
        self.base.mkdir(exist_ok=True)
        self.manager = FakeManager(self.base)
        self.click_file = Path(self.tmp.name) / "relay-click.json"

    def _post(self, doc: dict | None = None) -> tuple[int, dict]:
        server = DashboardServer(("127.0.0.1", 0), self.manager, self.cfg, self.base)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                                  daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        with patch.object(pose_relay, "DEFAULT_CLICK_FILE", str(self.click_file)):
            conn.request("POST", "/api/menu", json.dumps(doc or {}).encode(),
                         {"Content-Type": "application/json"})
            response = conn.getresponse()
            raw = response.read()
            conn.close()
        try:
            return response.status, json.loads(raw)
        except json.JSONDecodeError:
            return response.status, {"raw": raw.decode(errors="replace")}

    def test_a_press_writes_a_request_for_the_relay(self) -> None:
        status, doc = self._post()
        self.assertEqual(status, 200, doc)
        self.assertTrue(doc.get("accepted"))
        saved = json.loads(self.click_file.read_text())
        self.assertEqual(saved["hand"], "right")
        self.assertIsInstance(saved["seq"], int)
        self.assertGreater(saved["seq"], 0)

    def test_the_hand_can_be_chosen(self) -> None:
        status, _ = self._post({"hand": "left"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(self.click_file.read_text())["hand"], "left")

    def test_an_unknown_hand_is_rejected_before_anything_is_written(self) -> None:
        status, doc = self._post({"hand": "both"})
        self.assertEqual(status, 400)
        self.assertIn("unknown hand", doc.get("error", ""))
        self.assertFalse(self.click_file.exists())


class FakeRun:
    """A RunManager as the dashboard sees it: four methods and a dict."""

    def __init__(self, state: str = "idle", diary: Path | None = None) -> None:
        self.state_value = state
        self.diary = diary
        self.started = 0
        self.stopped = 0
        self.waits = 0

    def snapshot(self) -> dict:
        return {
            "state": self.state_value,
            "phase": self.state_value,
            "detail": "",
            "id": "101010-a1b2c3",
            "dir": str(self.diary.parent) if self.diary else None,
            "session_id": None,
            "started_at": None,
            "ended_at": None,
            "duration_s": None,
            "error": None,
            "exit_code": None,
        }

    def start(self) -> None:
        self.started += 1
        if self.state_value == "conflict":
            raise ConflictError("session is not running")
        if self.state_value == "bad":
            from clank.run import RunError

            raise RunError("run.brain_python not found: /nonexistent")

    def stop(self) -> None:
        self.stopped += 1

    def current_diary(self) -> Path:
        return self.diary or Path("/nonexistent/brain.jsonl")

    def wait_idle(self, timeout: float = 10.0) -> bool:
        self.waits += 1
        return True


class RunRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.manager = FakeManager(self.base)
        self.manager.state = "running"
        cfg = load_config(None, env={})
        self.cfg = replace(cfg, paths=replace(cfg.paths, session_root=str(self.base)))
        self.run_dir = self.base / "session" / "runs" / "101010-a1b2c3"
        self.run_dir.mkdir(parents=True)
        self.diary = self.run_dir / "brain.jsonl"
        self.diary.write_text(
            json.dumps({"kind": "think", "text": "this run only"}) + "\n", encoding="utf-8"
        )
        self.fake_run = FakeRun(diary=self.diary)
        self.server = DashboardServer(
            ("127.0.0.1", 0),
            _SessionDirManager(self.manager, self.base),
            self.cfg,
            self.base,
            overlay=OverlayBoard(),
            run=self.fake_run,  # type: ignore[arg-type] - a four-method stand-in
        )
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.thread.join, 2)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, method: str, path: str, body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read() or b"{}")
            finally:
                exc.close()

    def test_state_includes_the_run(self) -> None:
        status, body = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(body["run"]["id"], "101010-a1b2c3")
        self.assertEqual(body["run"]["state"], "idle")

    def test_run_start_and_stop_are_accepted(self) -> None:
        status, body = self.request("POST", "/api/run/start", {})
        self.assertEqual(status, 202)
        self.assertTrue(body["accepted"])
        status, _ = self.request("POST", "/api/run/stop", {})
        self.assertEqual(status, 202)
        self.assertEqual(self.fake_run.started, 1)
        self.assertEqual(self.fake_run.stopped, 1)

    def test_run_start_conflict_is_409(self) -> None:
        self.fake_run.state_value = "conflict"
        status, body = self.request("POST", "/api/run/start", {})
        self.assertEqual(status, 409)
        self.assertIn("session is not running", body["error"])

    def test_run_start_with_a_bad_interpreter_is_400(self) -> None:
        self.fake_run.state_value = "bad"
        status, body = self.request("POST", "/api/run/start", {})
        self.assertEqual(status, 400)
        self.assertIn("brain_python", body["error"])

    def test_runs_disabled_is_503(self) -> None:
        self.server.run = None
        status, _ = self.request("POST", "/api/run/start", {})
        self.assertEqual(status, 503)

    def test_session_stop_stops_the_run_first(self) -> None:
        order: list[str] = []
        self.fake_run.stop = lambda: order.append("run")
        self.server.manager.stop = lambda: order.append("stack")
        status, _ = self.request("POST", "/api/session/stop", {})
        self.assertEqual(status, 202)
        # Steam must not be shut down underneath a run that is about to restart or
        # restart-monitor the game: the run goes first.
        self.assertEqual(order, ["run", "stack"])

    def test_past_run_opens_without_a_current_session(self) -> None:
        self.server.manager.snapshot = lambda: {**self.manager.snapshot(), "session_dir": None}
        status, body = self.request("GET", "/api/brain/messages?run=101010-a1b2c3")
        self.assertEqual(status, 200)
        self.assertEqual(body["messages"][0]["text"], "this run only")

    def test_brain_messages_read_the_run_diary(self) -> None:
        status, body = self.request("GET", "/api/brain/messages")
        self.assertEqual(status, 200)
        self.assertEqual([row["text"] for row in body["messages"]], ["this run only"])

    def test_brain_messages_accept_a_past_run(self) -> None:
        status, body = self.request("GET", "/api/brain/messages?run=101010-a1b2c3")
        self.assertEqual(status, 200)
        self.assertEqual(body["messages"][0]["text"], "this run only")

    def test_brain_messages_refuse_a_traversing_run_id(self) -> None:
        for bad in ("..%2F..%2Fetc", "101010-a1b2c3/../../x", "nope"):
            with self.subTest(run=bad):
                status, _ = self.request("GET", f"/api/brain/messages?run={bad}")
                self.assertEqual(status, 404)


class _SessionDirManager(FakeManager):
    """FakeManager plus the session_dir the run-diary route needs."""

    def __init__(self, inner: FakeManager, base: Path) -> None:
        super().__init__(inner.base)
        self.inner = inner
        self.state = inner.state
        self.session_id = inner.session_id

    def snapshot(self) -> dict:
        return {**self.inner.snapshot(), "session_dir": str(self.base / "session")}


class ShutdownOrderTests(unittest.TestCase):
    def test_the_run_stops_before_the_stack(self) -> None:
        order: list[str] = []

        class Body:
            def stop(self) -> None:
                order.append("body")

        class Run:
            def stop(self) -> None:
                order.append("run")

            def wait_idle(self, timeout: float = 10.0) -> bool:
                order.append("run-wait")
                return True

        class Manager:
            def stop(self) -> None:
                order.append("stack")

        shutdown_all(  # type: ignore[arg-type] - the three collaborators, ordered
            Body(),  # type: ignore[arg-type]
            Run(),  # type: ignore[arg-type]
            Manager(),
        )
        # A run that outlives its Steam leaves a brain loop driving nothing.
        self.assertEqual(order, ["body", "run", "run-wait", "stack"])


class RecordingSweepTests(unittest.TestCase):
    """A dashboard that died mid-run leaves parts nobody owns: the next start
    clears them before the page serves, and says how many it cleared."""

    sessions_root: str = ""

    def run_dashboard_with(self, order: list, stdout, server=None) -> list[Path]:
        class FakeServer:
            def __init__(self, address, manager, cfg, static_dir, **kwargs) -> None:
                order.append("bind")
                self.server_address = address

            def serve_forever(self, poll_interval: float = 0.5) -> None:
                order.append("serve")

            def server_close(self) -> None:
                pass

        server = server or FakeServer

        class FakeManager:
            def __init__(self, cfg) -> None:
                self.cfg = cfg

            def stop(self) -> None:
                pass

            def wait_idle(self, timeout: float = 70.0) -> bool:
                return True

        class FakeRun:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def stop(self) -> None:
                pass

            def wait_idle(self, timeout: float = 15.0) -> bool:
                return True

        with tempfile.TemporaryDirectory() as temp:
            self.sessions_root = temp
            cfg = load_config(None, env={})
            cfg = replace(cfg, paths=replace(cfg.paths, session_root=temp))
            with contextlib.redirect_stdout(stdout), \
                    patch.object(dashboard, "SessionManager", FakeManager), \
                    patch.object(dashboard, "RunManager", FakeRun), \
                    patch.object(dashboard, "DashboardServer", server), \
                    patch.object(dashboard, "signal"), \
                    patch.object(dashboard, "sweep_stale", autospec=True) as sweep:
                swept = Path(temp) / "swept"
                sweep.side_effect = lambda root: order.append("sweep") or [swept]
                dashboard.run_dashboard(cfg)
            return [call.args[0] for call in sweep.call_args_list]

    def test_the_sweep_runs_before_the_page_is_served(self):
        order: list[str] = []
        out = io.StringIO()
        args = self.run_dashboard_with(order, out)
        # After the bind: a dashboard that cannot take its port deletes nothing.
        self.assertEqual(order, ["bind", "sweep", "serve"])
        # Once, on the configured sessions root: nothing else sweeps.
        self.assertEqual(len(args), 1)
        self.assertEqual(Path(args[0]), Path(self.sessions_root))
        # And the number is in the log, so a sweep that found nothing is visible
        # as a zero rather than as a silence.
        self.assertIn("swept 1 stale recording(s)", out.getvalue())

    def test_a_dashboard_that_cannot_bind_sweeps_nothing(self):
        """A second dashboard on the same session root is refused at the bind; it
        must not have deleted the live one's parts on the way there."""
        class Occupied:
            def __init__(self, *args, **kwargs) -> None:
                raise OSError(98, "Address already in use")

        order: list[str] = []
        with self.assertRaises(OSError):
            self.run_dashboard_with(order, io.StringIO(), server=Occupied)
        self.assertEqual(order, [])  # nothing bound, and nothing swept


class RecordingRouteTests(unittest.TestCase):
    """The listing, and the range behaviour a video player needs to seek."""

    size = 4096

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.root = base / "sessions"
        static = base / "static"
        static.mkdir()
        (static / "recordings.html").write_text("<h1>Recordings</h1>", encoding="utf-8")
        (static / "recordings.js").write_text("// recordings\n", encoding="utf-8")
        self.manager = FakeManager(base)
        self.manager.state = "running"
        cfg = load_config(None, env={})
        self.cfg = replace(cfg, paths=replace(cfg.paths, session_root=str(self.root)))
        self.write_run("100000-aaaaaa", "ready", "2026-11-12T10:00:00Z", video=True)
        self.write_run("110000-bbbbbb", "composing", "2026-11-12T11:00:00Z", video=False)
        self.server = DashboardServer(("127.0.0.1", 0), self.manager, self.cfg, static)
        self.port = self.server.server_address[1]
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def write_run(self, run_id: str, state: str, started: str, video: bool) -> Path:
        run_dir = self.root / "20261112T100000Z-abc123" / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": run_id,
            "state": state,
            "started_at": started,
            "duration_s": 12.5,
            "bytes": self.size if video else 0,
            "file": "recording.mp4" if video else None,
        }
        (run_dir / "recording.json").write_text(json.dumps(manifest), encoding="utf-8")
        if video:
            (run_dir / "recording.mp4").write_bytes(b"\x01" * self.size)
        return run_dir

    def get(self, path: str, headers: dict[str, str] | None = None):
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", headers=headers or {})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, exc.headers, exc.read()
            finally:
                exc.close()

    def test_the_listing_is_newest_first(self) -> None:
        status, _, body = self.get("/api/recordings")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(
            [row["run_id"] for row in payload["recordings"]],
            ["110000-bbbbbb", "100000-aaaaaa"],
        )
        self.assertTrue(payload["recording_available"])

    def test_the_page_is_served(self) -> None:
        status, _, body = self.get("/recordings")
        self.assertEqual(status, 200)
        self.assertIn(b"Recordings", body)

    def test_a_range_request_gets_a_206_slice(self) -> None:
        status, headers, body = self.get(
            "/api/recordings/100000-aaaaaa/video", headers={"Range": "bytes=0-1"}
        )
        self.assertEqual(status, 206)
        self.assertEqual(headers["Content-Range"], f"bytes 0-1/{self.size}")
        self.assertEqual(headers["Accept-Ranges"], "bytes")
        self.assertEqual(len(body), 2)

    def test_a_suffix_range_and_a_multi_range(self) -> None:
        status, headers, _ = self.get(
            "/api/recordings/100000-aaaaaa/video", headers={"Range": "bytes=-500"}
        )
        self.assertEqual(status, 206)
        self.assertEqual(
            headers["Content-Range"], f"bytes {self.size - 500}-{self.size - 1}/{self.size}"
        )
        # A multi-range is served whole rather than guessed at.
        status, _, body = self.get(
            "/api/recordings/100000-aaaaaa/video", headers={"Range": "bytes=0-1,5-6"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(body), self.size)

    def test_a_whole_request_is_a_200(self) -> None:
        status, headers, body = self.get("/api/recordings/100000-aaaaaa/video")
        self.assertEqual(status, 200)
        self.assertEqual(len(body), self.size)
        self.assertEqual(headers["Content-Type"], "video/mp4")

    def test_an_impossible_range_is_a_416(self) -> None:
        status, headers, _ = self.get(
            "/api/recordings/100000-aaaaaa/video", headers={"Range": f"bytes={self.size * 10}-"}
        )
        self.assertEqual(status, 416)
        self.assertEqual(headers["Content-Range"], f"bytes */{self.size}")

    def test_an_unknown_or_unfinished_run_is_a_404(self) -> None:
        self.assertEqual(self.get("/api/recordings/nope/video")[0], 404)
        self.assertEqual(self.get("/api/recordings/110000-bbbbbb/video")[0], 404)
        # The id must be the pinned shape, and the leaf must be `video`.
        self.assertEqual(self.get("/api/recordings/..%2F..%2Fetc/video")[0], 404)
        self.assertEqual(self.get("/api/recordings/100000-aaaaaa/brain.jsonl")[0], 404)

    def test_recording_off_is_reported_not_an_empty_list(self) -> None:
        off = replace(self.cfg, recording=replace(self.cfg.recording, enabled=False))
        server = DashboardServer(("127.0.0.1", 0), self.manager, off, Path(self.temp.name) / "static")
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        request = urllib.request.Request(f"http://127.0.0.1:{port}/api/recordings")
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
            payload = json.loads(response.read())
        self.assertFalse(payload["recording_available"])
