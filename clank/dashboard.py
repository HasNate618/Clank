from __future__ import annotations

import array
import contextlib
import getpass
import json
import mimetypes
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import brainlog
from .body_preview import BODY_VERBS, BODY_VIEWS, BodyPreview, send_control
from .config import Config
from .depth_stream import bmp, latest
from .manual_input import ManualInputError, ManualInputUnavailable
from .orchestrator import ConflictError, PreflightError, SessionManager
from .pose_relay import request_click
from .processes import tail_stream
from .recording import RECORDING_NAME, list_recordings, parse_range, sweep_stale
from .run import RUN_ID_RE, RunError, RunManager


class PayloadTooLarge(ValueError):
    pass


#: Lines of each log the merged stream carries. Enough to read a failure, small
#: enough that the whole payload stays well under a megabyte.
LOG_TAIL_LINES = 250

#: Bytes per socket write when a recording is served. Small enough that a browser
#: aborting a seek costs one read, large enough that a 100 MB run is a few hundred
#: writes instead of forty thousand.
VIDEO_CHUNK_BYTES = 256 * 1024

#: How long a grounding claim stays drawable, and how many are kept. The overlay answers
#: "what did the brain just claim", not "what is true now", so items expire: a box left on
#: the picture after the fact is a lie about what the robot currently sees.
OVERLAY_TTL_S = 6.0
OVERLAY_MAX = 12
OVERLAY_NOUN_MAX = 120


def _numbers(value, count: int, name: str, positive: bool = False) -> list[float]:
    """`value` as `count` floats, or ValueError. Rejects bools, which are ints in Python."""
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"{name} needs {count} numbers")
    out: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{name} needs numbers")
        number = float(item)
        if number != number or number in (float("inf"), float("-inf")):
            raise ValueError(f"{name} needs finite numbers")
        if positive and number <= 0:
            raise ValueError(f"{name} must be positive")
        out.append(number)
    return out


class OverlayBoard:
    """Recent grounding claims, for the preview overlay to draw and then forget.

    The brain locates an object and this is where it says where: a normalised box, the
    point it chose, and how far its own repeated reads disagreed. It is a board rather
    than state because nothing reads it back to decide anything - and it expires, because
    the picture it describes moves on.
    """

    def __init__(self, ttl_s: float = OVERLAY_TTL_S, limit: int = OVERLAY_MAX) -> None:
        self._lock = threading.Lock()
        self._ttl_s = float(ttl_s)
        self._limit = max(1, int(limit))
        self._items: list[dict] = []
        self._seq = 0

    def add(self, payload: dict) -> dict:
        """Validate and remember one claim. Raises ValueError on anything malformed."""
        box = _numbers(payload.get("box"), 4, "box")
        if not (0.0 <= box[0] < box[2] <= 1.0 and 0.0 <= box[1] < box[3] <= 1.0):
            raise ValueError("box must be four normalised edges, in order")
        pixel = _numbers(payload.get("pixel"), 2, "pixel")
        size = _numbers(payload.get("size"), 2, "size", positive=True)
        noun = payload.get("noun", "object")
        if not isinstance(noun, str) or not noun.strip():
            raise ValueError("noun required")
        spread = payload.get("spread")
        if spread is not None:
            spread = _numbers([spread], 1, "spread")[0]
        present = payload.get("present")
        with self._lock:
            self._seq += 1
            item = {
                "id": self._seq,
                "t": time.time(),
                "noun": noun.strip()[:OVERLAY_NOUN_MAX],
                "box": box,
                "pixel": pixel,
                "size": size,
                "spread": spread,
                "present": None if present is None else bool(present),
            }
            self._items.append(item)
            del self._items[: -self._limit]
            return self._view(item, item["t"])

    def snapshot(self) -> list[dict]:
        now = time.time()
        with self._lock:
            self._prune(now)
            return [self._view(item, now) for item in self._items]

    def clear(self) -> None:
        """Forget every claim: a new run starts with an empty picture."""
        with self._lock:
            self._items.clear()

    def _prune(self, now: float) -> None:
        cutoff = now - self._ttl_s
        kept = [item for item in self._items if item["t"] >= cutoff]
        self._items[:] = kept

    def _view(self, item: dict, now: float) -> dict:
        return {
            "id": item["id"],
            "age_s": round(now - item["t"], 2),
            "noun": item["noun"],
            "box": item["box"],
            "pixel": item["pixel"],
            "size": item["size"],
            "spread": item["spread"],
            "present": item["present"],
        }


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, manager, config: Config, static_dir: Path,
                 body: BodyPreview | None = None, overlay: OverlayBoard | None = None,
                 run: RunManager | None = None) -> None:
        self.manager = manager
        self.config = config
        self.static_dir = Path(static_dir)
        self.body = body
        self.overlay = overlay or OverlayBoard()
        self.run = run
        self.shutdown_event = threading.Event()
        super().__init__(address, DashboardHandler)

    def shutdown(self) -> None:
        self.shutdown_event.set()
        super().shutdown()


class DashboardHandler(BaseHTTPRequestHandler):
    # BaseHTTPRequestHandler types .server as BaseServer, which has neither of
    # ours. This annotation narrows it for the checker and costs nothing at runtime.
    server: DashboardServer

    server_version = "clank/0.1"
    timeout = 10

    def log_message(self, format, *args) -> None:
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._static("index.html", "text/html; charset=utf-8")
        if path == "/style.css":
            return self._static("style.css", "text/css; charset=utf-8")
        if path == "/app.js":
            return self._static("app.js", "application/javascript; charset=utf-8")
        if path == "/api/state":
            state = self.server.manager.snapshot()
            body = self.server.body
            if body is not None:
                state = {**state, "body": body.status()}
            # The run is a separate lifetime from the session: a stopped stack can
            # still have a last run to show, and a running stack can have none.
            run = self.server.run
            state = {**state, "run": run.snapshot() if run is not None else None}
            return self._json(state)
        if path == "/api/preview/config":
            return self._preview_config()
        if path == "/api/preview/live.bmp":
            return self._preview_live()
        if path == "/api/preview/live.rgba":
            return self._preview_live_rgba()
        if path == "/api/preview/live.depth":
            return self._preview_live_depth()
        if path.startswith("/novnc/"):
            return self._novnc(path[len("/novnc/"):])
        if path == "/api/body.jpg":
            body = self.server.body
            if body is None:
                return self._error(404, "no body frame")
            view = parse_qs(urlparse(self.path).query).get("view", [None])[0]
            if view is not None and view not in BODY_VIEWS:
                return self._error(400, f"unknown view: {view}")
            frame = body.frame(view)
            if frame is None:
                return self._error(404, "no body frame")
            try:
                return self._bytes(200, frame.read_bytes(), "image/jpeg")
            except OSError:
                return self._error(404, "no body frame")
        if path == "/api/recordings":
            return self._recordings_index()
        if path.startswith("/api/recordings/"):
            return self._recording_video(path)
        if path == "/recordings":
            return self._static("recordings.html", "text/html; charset=utf-8")
        if path == "/recordings.js":
            return self._static("recordings.js", "application/javascript; charset=utf-8")
        if path == "/api/logs":
            return self._log_stream()
        if path == "/api/brain/messages":
            # Structured diary rows for the Brain panel; /api/logs keeps its own
            # rendered copy of the same diary for the plain log view. `?run=<id>`
            # reads a past run, which is how a finished attempt is inspected.
            query = parse_qs(urlparse(self.path).query)
            wanted = (query.get("run") or [None])[0]
            target = self._diary()
            if wanted is not None:
                target = self._run_diary(wanted)
                if target is None:
                    return self._error(404, "unknown run")
            try:
                rows = brainlog.tail(50, target)
            except Exception:  # noqa: BLE001 - a broken diary must not break the page
                rows = []
            return self._json({"messages": rows})
        if path == "/api/overlay":
            # What the brain last claimed it could see, for the boxes it draws on the
            # preview. Expires by itself, so a stale box cannot outlive its evidence.
            return self._json({"items": self.server.overlay.snapshot()})
        if path.startswith("/fonts/"):
            return self._font(path[len("/fonts/"):])
        if path.startswith("/api/logs/"):
            name = path[len("/api/logs/"):]
            decoded = unquote(name)
            if "/" in decoded or "\\" in decoded or decoded in (".", ".."):
                return self._error(404, "unknown log")
            return self._log(name)
        return self._error(404, "not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/session/start":
            try:
                self.server.manager.start()
            except ConflictError as exc:
                return self._error(409, str(exc))
            except PreflightError as exc:
                return self._error(400, str(exc))
            except Exception:
                return self._error(500, "internal server error")
            return self._json({"accepted": True}, status=202)
        if path == "/api/session/stop":
            # The run goes first, and not just on process exit: `stop()` ends the
            # brain child inline, so Steam is never shut down while a loop that
            # drives it is still alive. `shutdown_all` waits for idle; here the
            # request must not block, and the stack's own lifecycle lock already
            # keeps an in-flight game restart and the teardown in order.
            run = self.server.run
            if run is not None:
                run.stop()
            try:
                self.server.manager.stop()
            except Exception:
                return self._error(500, "internal server error")
            return self._json({"accepted": True}, status=202)
        if path == "/api/run/start" or path == "/api/run/stop":
            run = self.server.run
            if run is None:
                return self._error(503, "runs disabled")
            try:
                run.start() if path.endswith("start") else run.stop()
            except ConflictError as exc:
                return self._error(409, str(exc))
            except RunError as exc:
                return self._error(400, str(exc))
            except Exception:
                return self._error(500, "internal server error")
            return self._json({"accepted": True}, status=202)
        try:
            body = self._read_body()
        except PayloadTooLarge as exc:
            return self._error(413, str(exc))
        except ValueError as exc:
            return self._error(400, str(exc))
        if path == "/api/overlay":
            try:
                self.server.overlay.add(body)
            except ValueError as exc:
                return self._error(400, str(exc))
            return self._json({"ok": True, "items": self.server.overlay.snapshot()}, status=202)
        if path == "/api/manual/input":
            if not self._origin_allowed():
                return self._error(403, "cross-origin input rejected")
            try:
                armed = self.server.manager.manual_input(body)
            except ManualInputError as exc:
                return self._error(400, str(exc))
            except ConflictError as exc:
                return self._error(409, str(exc))
            except ManualInputUnavailable as exc:
                return self._error(503, str(exc))
            except Exception:
                return self._error(500, "internal server error")
            return self._json({"accepted": True, "armed": bool(armed)})
        if path == "/api/menu":
            # The menu button lives on the controller, not on the body: no joint of the
            # simulated G1 can press it, and JobBot asks a player to press it early. So the
            # dashboard asks the relay for one press, and the relay turns that into a short
            # a_click on the wire (XRT_INPUT_INDEX_A_CLICK / X), the button the game
            # actually reads as its menu button.
            if not self._origin_allowed():
                return self._error(403, "cross-origin input rejected")
            hand = body.get("hand", "right")
            if not isinstance(hand, str) or hand not in ("left", "right"):
                return self._error(400, f"unknown hand: {hand}")
            try:
                request = request_click(hand)
            except OSError as exc:
                return self._error(503, f"cannot write the press request: {exc}")
            return self._json({"accepted": True, **request})
        if path == "/api/body/start":
            body = self.server.body
            if body is None:
                return self._error(400, "body view disabled")
            problem = body.start()
            if problem:
                return self._error(400, problem)
            return self._json({"ok": True, "body": body.status()}, status=202)
        if path == "/api/body/stop":
            body = self.server.body
            if body is None:
                return self._error(400, "body view disabled")
            body.stop()
            return self._json({"ok": True, "body": body.status()})
        if path == "/api/body/command":
            view = self.server.body
            if view is None:
                return self._error(400, "body view disabled")
            # No running() gate: that only knows about the process *this* dashboard
            # spawned, and a viewer started another way (or one that outlived a
            # restart) publishes frames and holds the socket perfectly well. The
            # socket is the authority - if nothing is listening, send_control says so
            # with the real reason instead of a guess about our own handle.
            if "view" in body and "verb" not in body:
                name = body.get("view")
                if not isinstance(name, str) or name not in BODY_VIEWS:
                    return self._error(400, f"unknown view: {name}")
                try:
                    reply = send_control(view.control_path(), {"view": name})
                except (OSError, ValueError) as exc:
                    return self._error(503, f"body control unavailable: {exc}")
                return self._json(reply)
            verb = body.get("verb")
            if not isinstance(verb, str) or verb not in BODY_VERBS:
                return self._error(400, f"unknown verb: {verb}")
            try:
                reply = send_control(view.control_path(), {"verb": verb})
            except (OSError, ValueError) as exc:
                return self._error(503, f"body control unavailable: {exc}")
            return self._json(reply)
        if path == "/api/session/mode":
            try:
                self.server.manager.set_mode(body.get("mode"))
            except ValueError as exc:
                return self._error(400, str(exc))
            except ConflictError as exc:
                return self._error(409, str(exc))
            except Exception:
                return self._error(500, "internal server error")
            return self._json({"mode": body.get("mode")})
        if path in ("/api/audio/mute", "/api/audio", "/api/game/audio", "/api/game/audio/mute", "/api/audio/toggle"):
            try:
                if path.endswith("/toggle"):
                    muted = self.server.manager.toggle_game_audio_mute()
                elif "muted" in body:
                    muted = self.server.manager.set_game_audio_muted(bool(body.get("muted")))
                elif "mute" in body:
                    muted = self.server.manager.set_game_audio_muted(bool(body.get("mute")))
                else:
                    muted = self.server.manager.set_game_audio_muted(True)
            except Exception:
                return self._error(500, "internal server error")
            return self._json({"muted": bool(muted), "audio": self.server.manager.snapshot().get("audio", {})})
        return self._error(404, "not found")

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        parsed = urlparse(origin)
        return parsed.scheme in ("http", "https") and parsed.netloc == self.headers.get("Host", "")

    def _read_body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length < 0:
            raise ValueError("invalid Content-Length")
        if length > 1_000_000:
            raise PayloadTooLarge("request body too large")
        try:
            raw = self.rfile.read(length) if length else b""
        except (TimeoutError, OSError) as exc:
            raise ValueError("request body timeout") from exc
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON object required")
        return data

    def _json(self, payload, status: int = 200) -> None:
        self._bytes(status, json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status=status)

    def _bytes(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _static(self, name: str, content_type: str) -> None:
        path = self.server.static_dir / name
        try:
            data = path.read_bytes()
        except OSError:
            return self._error(404, "not found")
        self._bytes(200, data, content_type)

    def _log_stream(self) -> None:
        """Every public log in one payload, so the UI can merge and filter them."""
        streams: dict[str, list[str]] = {}
        for name in self.server.manager.LOG_FILES:
            path = self.server.manager.log_path(name)
            if path is None:
                continue
            try:
                with path.open("rb") as handle:
                    text = tail_stream(handle, lines=LOG_TAIL_LINES)
            except OSError:
                continue
            streams[name] = text.splitlines()
        brain = self._brain_lines()
        if brain:
            # Same panel, same chips: the brain is just another source, so the operator
            # reads "I will grab the card" next to the pose log without a second view.
            streams["brain"] = brain
        self._json({"logs": streams})

    def _brain_lines(self) -> list[str]:
        """The brain's diary as log lines. Never raises: a diary must not break the UI."""
        try:
            rows = brainlog.tail(LOG_TAIL_LINES, self._diary())
        except Exception:  # noqa: BLE001 - a broken diary must not take out /api/logs
            return []
        lines: list[str] = []
        for row in rows:
            rendered = brainlog.render(row)
            lines.append(rendered.removeprefix("[brain] "))
        return lines

    def _diary(self) -> Path:
        """The diary this page shows: the run's, or the global one with no runs yet.

        The override has to be explicit: `brainlog.default_path()` reads the
        dashboard's own environment, which has no `CLANK_BRAIN_LOG`, so a run's
        diary would be invisible if the reader relied on it.
        """
        run = self.server.run
        if run is not None:
            with contextlib.suppress(Exception):
                path = run.current_diary()
                if path is not None:
                    return Path(path)
        return brainlog.default_path()

    def _run_diary(self, run_id: str) -> Path | None:
        """A past run's diary, or None. The id is the only input, so it is pinned."""
        if not RUN_ID_RE.fullmatch(str(run_id)):
            return None
        session_dir = self.server.manager.snapshot().get("session_dir")
        if session_dir:
            path = Path(session_dir) / "runs" / str(run_id) / "brain.jsonl"
            if path.is_file():
                return path
        # A dashboard restarted with the stack stopped has no session directory,
        # but the run it is showing is still on disk: find it by its pinned id.
        root = Path(self.server.config.paths.session_root)
        for candidate in root.glob(f"*/runs/{run_id}/brain.jsonl"):
            if candidate.is_file():
                return candidate
        return None

    def _recordings_index(self) -> None:
        """Every run that left a manifest, newest first.

        `recording_available` is the page's off-state and it is the config switch,
        not a guess: ffmpeg is not in the flake, so a machine can have runs and no
        video at all, and the page says that once instead of showing an empty list
        that looks like a bug.
        """
        try:
            rows = list_recordings(Path(self.server.config.paths.session_root))
        except OSError:
            rows = []
        return self._json({
            "recordings": rows,
            "recording_available": bool(self.server.config.recording.enabled),
        })

    def _recording_row(self, run_id: str) -> dict | None:
        """One playable recording, found by id. Never by joining the URL to a path."""
        if not RUN_ID_RE.fullmatch(str(run_id)):
            return None
        try:
            rows = list_recordings(Path(self.server.config.paths.session_root))
        except OSError:
            return None
        for row in rows:
            if (
                row.get("run_id") == run_id
                and row.get("state") == "ready"
                and row.get("has_file")
            ):
                return row
        return None

    def _recording_video(self, path: str) -> None:
        """One recording's bytes, seekable.

        A player seeks with a `Range` request, so a usable range is a 206 with the
        slice, an unsatisfiable one is a 416, and no range is the whole file. The
        file is streamed rather than read: at 24 MB a minute a long run is bigger
        than the process should hold, and a browser aborts half of these requests
        the moment the operator scrubs.
        """
        run_id, _, leaf = path[len("/api/recordings/"):].partition("/")
        if leaf != "video":
            return self._error(404, "not found")
        row = self._recording_row(run_id)
        if row is None:
            return self._error(404, "no recording")
        target = Path(row["dir"]) / RECORDING_NAME
        try:
            size = target.stat().st_size
        except OSError:
            return self._error(404, "no recording")
        header = self.headers.get("Range")
        try:
            span = parse_range(header, size) if header else None
        except ValueError:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start, end = (0, size - 1) if span is None else span
        length = max(0, end - start + 1)
        try:
            handle = target.open("rb")
        except OSError:
            return self._error(404, "no recording")
        with handle:
            handle.seek(start)
            self.send_response(200 if span is None else 206)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if span is not None:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            # Long, and immutable once ready: the player may cache it.
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            remaining = length
            try:
                while remaining > 0:
                    chunk = handle.read(min(VIDEO_CHUNK_BYTES, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except OSError:
                # The browser scrubbed away and closed the socket. Not an error.
                pass

    def _font(self, name: str) -> None:
        # Only the vendored font directory is served, and never a traversal.
        if not name or "/" in name or "\\" in name or name in (".", ".."):
            return self._error(404, "not found")
        return self._static(f"fonts/{name}", "font/woff2")

    def _log(self, name: str) -> None:
        path = self.server.manager.log_path(name)
        if path is None:
            return self._error(404, "unknown log")
        try:
            with path.open("rb") as handle:
                data = tail_stream(handle).encode()
        except OSError:
            return self._error(404, "log not found")
        self._bytes(200, data, "text/plain; charset=utf-8")

    def _preview_live(self) -> None:
        """The newest colour frame from the depth-stream layer, as a BMP.

        The game presents its desktop window once and then renders only to the
        OpenXR views, so Weston's VNC preview is a frozen frame. The layer is
        the only live source, and a BMP needs no encoder on either side.
        """
        frame = latest(max_age=2.0)
        if frame is None or frame.get("color") is None:
            return self._error(404, "no live frame")
        width = int(frame["color_width"])
        height = int(frame["color_height"])
        self._bytes(200, bmp(frame["color"], width, height), "image/bmp")

    def _preview_live_rgba(self) -> None:
        """The newest colour frame as raw RGBA, for the dashboard canvas.

        Sending the bytes untransformed avoids a per-pixel BMP encode in Python
        and lets the browser draw at 20 fps without reload flicker. The frame is
        already top-down, so the dimensions travel in headers.
        """
        frame = latest(max_age=2.0)
        if frame is None or frame.get("color") is None:
            return self._error(404, "no live frame")
        width = int(frame["color_width"])
        height = int(frame["color_height"])
        data = frame["color"]
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Frame-Width", str(width))
        self.send_header("X-Frame-Height", str(height))
        self.end_headers()
        self.wfile.write(data)

    def _preview_live_depth(self) -> None:
        """The newest depth map as raw float32 metres, for the model-view fade.

        The browser rebuilds what `spike/wbc_game/fade.py` does to the frame it
        hands the model: a smoothstep mask from the foreground depth, blending
        the far room toward a dark backdrop. Depth travels exactly as the layer
        stores it - quantising to millimetres here would mean a per-pixel Python
        loop per request, and a Float32Array costs the browser nothing.

        Byte order is the reader's, which is little-endian on every machine this
        runs on; the client checks the length before trusting it.
        """
        frame = latest(max_age=2.0)
        if frame is None or frame.get("depth") is None:
            return self._error(404, "no live frame")
        width = int(frame["width"])
        height = int(frame["height"])
        depth = frame["depth"]
        data = depth.tobytes() if isinstance(depth, array.array) else array.array("f", depth).tobytes()
        if len(data) != width * height * 4:
            return self._error(500, "depth frame is the wrong size")
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Frame-Width", str(width))
        self.send_header("X-Frame-Height", str(height))
        self.end_headers()
        self.wfile.write(data)

    def _preview_config(self) -> None:
        preview = self.server.manager.snapshot().get("preview") or {}
        web_root = self.server.config.paths.novnc_web_root
        rfb = Path(web_root) / "core" / "rfb.js" if web_root else None
        try:
            username = getpass.getuser()
        except Exception:
            username = ""
        try:
            assets = bool(rfb is not None and rfb.is_file() and os.access(rfb, os.R_OK))
        except OSError:
            assets = False
        self._json({
            "mode": "vnc",
            "vnc_port": self.server.config.preview.vnc_port,
            "proxy_port": self.server.config.preview.proxy_port,
            "ws_url": f"ws://127.0.0.1:{self.server.config.preview.proxy_port}/",
            "username": username,
            "available": self.server.manager.state_value()
            in ("starting", "running", "stopping"),
            "transport_alive": bool(preview.get("vnc_alive"))
            and bool(preview.get("proxy_alive")),
            "assets_available": assets,
        })

    def _novnc(self, name: str) -> None:
        web_root = self.server.config.paths.novnc_web_root
        if not web_root:
            return self._error(404, "noVNC assets unavailable")
        root = Path(web_root)
        if not (root / "core" / "rfb.js").is_file():
            return self._error(404, "noVNC assets unavailable")
        decoded = unquote(name)
        if any(part.startswith(".") for part in decoded.split("/") if part):
            return self._error(404, "not found")
        try:
            target = (root / decoded).resolve()
            target.relative_to(root.resolve())
        except (OSError, ValueError):
            return self._error(404, "not found")
        if not target.is_file():
            return self._error(404, "not found")
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        try:
            data = target.read_bytes()
        except OSError:
            return self._error(404, "not found")
        self._bytes(200, data, content_type)


def dashboard_pidfile(cfg: Config) -> Path:
    """Where a running dashboard records its pid. --replace reads this."""
    return Path(cfg.paths.session_root) / "dashboard.pid"


def shutdown_all(body: BodyPreview | None, run: RunManager | None, manager) -> None:
    """Stop everything in the one order that leaves no ghost.

    The run goes before the stack: its last phase may be restarting the game or
    waiting on a game restart, and shutting Steam down underneath that is how a
    half-killed game and an orphan brain loop happen.
    """
    if body is not None:
        body.stop()
    if run is not None:
        run.stop()
        run.wait_idle(15)
    manager.stop()


def run_dashboard(cfg: Config) -> None:
    manager = SessionManager(cfg)
    static_dir = Path(__file__).resolve().parent / "static"
    body = None
    if cfg.body.enabled:
        body = BodyPreview(cfg.body, out_dir=Path(cfg.paths.session_root) / "body")
        if cfg.body.autostart:
            problem = body.start()
            if problem:
                print(f"clank dashboard: body view: {problem}", file=sys.stderr)
    overlay = OverlayBoard()
    run = RunManager(manager, cfg, body=body, overlay=overlay)
    server = DashboardServer(
        (cfg.dashboard.host, cfg.dashboard.port), manager, cfg, static_dir,
        body=body, overlay=overlay, run=run,
    )
    # After the bind, so a dashboard refused its port deletes nothing; before the
    # page serves and before a run can exist, so nothing it meets is live.
    stale = sweep_stale(Path(cfg.paths.session_root))
    print(f"clank dashboard: swept {len(stale)} stale recording(s)", flush=True)
    pidfile = dashboard_pidfile(cfg)
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pidfile.write_text(str(os.getpid()), encoding="utf-8")

    def request_shutdown(signum, frame) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    host, port = server.server_address[:2]
    print(f"clank dashboard: http://{host}:{port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        print("clank dashboard: shutting down", flush=True)
        if body is not None:
            print(f"clank dashboard: stopping body view (running={body.running()})", flush=True)
        shutdown_all(body, run, manager)
        if not manager.wait_idle(70):
            print("clank dashboard: session did not stop within 70s", file=sys.stderr)
        server.server_close()
        try:
            owned = pidfile.read_text(encoding="utf-8").strip() == str(os.getpid())
        except OSError:
            owned = False
        if owned:
            with contextlib.suppress(OSError):
                pidfile.unlink()
