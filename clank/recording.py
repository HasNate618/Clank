"""Turn a live run into one watchable file.

A run produces three encoded parts: the game's picture, the robot's picture,
and the game's audio. This module holds the argv for the one ffmpeg pass that
stacks them, the HTTP ``Range`` parser the dashboard's video route answers
with, and ``Recorder``, which runs an encoder per source while the run is live.
Both of the first two are pure, so the recorder and the composite that run
them are tested around them rather than through them.

Nothing here raises into a run. A missing encoder, a dead pipe, an absent
depth stream: each becomes a manifest state or a source status, never an
exception the run has to survive.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clank.ears import level

GAME_WIDTH = 960
GAME_HEIGHT = 540
ROBOT_WIDTH = 720
HEIGHT = GAME_HEIGHT
WIDTH = GAME_WIDTH + ROBOT_WIDTH

#: The manifest Task 4 composites from, written twice: at start and at stop.
MANIFEST_NAME = "recording.json"
PARTS_DIR = "parts"
PART_NAMES = {"game": "game.mp4", "robot": "robot.mp4", "audio": "audio.m4a"}
SOURCES = ("game", "robot", "audio")
#: 16 kHz mono s16le; half a second is what `parecord` flushes at a time.
AUDIO_BLOCK = 8000
#: How long one encoder is given to finish after its stdin closes.
ENCODER_EXIT_S = 10.0
#: A frame tick never sleeps longer than this, so no wait is unbounded.
TICK_S = 0.01
#: The most the final audio drain may take before stop() gives up on it.
AUDIO_FLUSH_S = 1.0
#: Threads are joined in bounded slices; a wedged writer must not hang a run.
THREAD_JOIN_S = 2.0
#: A depth stream that has raised on this many reads running has died, not
#: blinked: at the 10 ms tick that is a third of a second of no picture.
STREAM_ERROR_LIMIT = 30

#: The threads that write the encoders' pipes, and the encoders each one holds.
#: A writer wedged in a full pipe takes its own encoders down, and only its own.
PUMP_THREAD = "clank-recording"
AUDIO_THREAD = "clank-recording-audio"
WEDGED_SOURCES = {PUMP_THREAD: ("game", "robot"), AUDIO_THREAD: ("audio",)}


def composite_argv(
    *,
    game: Path,
    robot: Path | None,
    audio: Path | None,
    out: Path,
    fps: int,
    crf: int,
    preset: str,
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    """The ffmpeg pass that stacks the parts side by side into `out`.

    A part that is missing is black padding of the same size, not a narrower
    file: the layout is fixed so every recording looks the same to a player.
    """
    inputs = ["-i", str(game)]
    input_count = 1
    audio_index: int | None = None
    if robot is not None:
        inputs += ["-i", str(robot)]
        input_count += 1
    if audio is not None:
        audio_index = input_count
        inputs += ["-i", str(audio)]

    left = f"[0:v]scale={GAME_WIDTH}:{GAME_HEIGHT}[g]"
    if robot is None:
        # An endless black pad, so the game's picture decides where the stack ends.
        right = f"color=black:s={ROBOT_WIDTH}x{HEIGHT}:r={fps}[r]"
        stack = "[g][r]hstack=inputs=2:shortest=1[v]"
    else:
        right = f"[1:v]scale={ROBOT_WIDTH}:{HEIGHT}[r]"
        stack = "[g][r]hstack=inputs=2[v]"

    argv = [
        ffmpeg, "-y", *inputs,
        "-filter_complex", ";".join([left, right, stack]),
        "-map", "[v]",
    ]
    if audio_index is not None:
        argv += ["-map", f"{audio_index}:a"]
    argv += [
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-r", str(fps),
    ]
    if audio_index is not None:
        argv += ["-c:a", "aac", "-b:a", "96k", "-shortest"]
    argv += ["-movflags", "+faststart", str(out)]
    return argv


class Recorder:
    """Three encoders writing `<run_dir>/parts/`, and the manifest that names them.

    ``sources`` says what actually happened to each one, and is exactly one of:

    - ``ok``: it produced at least one frame or block, so its part is playable;
    - ``missing``: there was nothing to record (no depth stream, no body tile,
      no capture device), so its part is absent and the composite pads it;
    - ``failed``: it was tried and broke (no encoder, a dead pipe, a depth
      stream that stopped answering).

    ``notes`` carries one line per source that gave a reason, whether or not the
    source recovered: a read error that came and went still explains a gap.

    ``stream_factory`` and ``capture`` default to None, which means the source
    is not available in this process: the caller that owns a live stack passes
    ``clank.depth_stream.Stream`` and ``clank.ears.capture``.
    """

    def __init__(
        self,
        cfg: Any,
        run_dir: Path,
        *,
        stream_factory: Callable[[], Any] | None = None,
        capture: Callable[[], Any] | None = None,
        spawn: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self._recording = cfg.recording
        self._stream_factory = stream_factory
        self._capture = capture
        self._spawn = spawn or subprocess.Popen
        self._clock = clock
        self._period = 1.0 / cfg.recording.fps
        self._robot_period = 1.0 / cfg.recording.robot_fps

        self._audio_lock = threading.Lock()
        self._tick_lock = threading.Lock()  # one tick at a time, whoever pumps
        self._idle = threading.Event()   # set by stop() to wake every wait
        self._stopped = threading.Event()
        self._threads: list[threading.Thread] = []

        self._stream = None
        self._capture_ctx = None
        self._reader = None
        self._seq = 0
        self._procs: dict[str, Any] = {}
        self._units: dict[str, int] = dict.fromkeys(SOURCES, 0)
        self._sources: dict[str, str] = dict.fromkeys(SOURCES, "missing")
        self._notes: dict[str, str] = {}

        self._color: bytes | None = None
        self._color_errors = 0
        self._tile: bytes | None = None
        self._tile_stamp: tuple[int, int] | None = None
        self._peak = 0

        self._started = False
        self._finished = False
        self._started_monotonic = 0.0
        self._started_at = ""
        self._ended_at: str | None = None
        self._duration = 0.0
        self._encoder = ""
        self._error: str | None = None
        self._manifest: dict = {}
        self._next_frame_at = 0.0
        self._next_robot_at = 0.0

    # ---- lifecycle ------------------------------------------------------

    def manifest(self) -> Path:
        return self.run_dir / MANIFEST_NAME

    def start(self) -> None:
        """Start an encoder per source. Never raises: a failure is a manifest."""
        if self._started:
            return
        self._started = True
        self._started_monotonic = self._clock()
        self._started_at = _utc_now()
        self._next_frame_at = self._started_monotonic
        self._next_robot_at = self._started_monotonic
        try:
            (self.run_dir / PARTS_DIR).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._fatal(f"could not create the parts directory: {exc}")
            return
        encoder = self._probe_encoder()
        if encoder is None:
            return  # _fatal already wrote the reason
        self._encoder = encoder
        self._start_game()
        self._start_robot()
        self._start_audio()
        self._write_manifest("recording")
        if self._sources["game"] == "ok" or self._sources["robot"] == "ok":
            self._start_thread(self._pump_loop, PUMP_THREAD)
        if self._sources["audio"] == "ok":
            self._start_thread(self._audio_loop, AUDIO_THREAD)

    def pump(self, seconds: float) -> None:
        """Run the pictures' schedule for ``seconds`` of wall time.

        This is the body the recording thread runs, and what a test drives
        directly so the suite needs no sleep-based race. Every frame the clock
        says is due is written; when no new one arrived, the last one is written
        again. A rawvideo or image2pipe feed is constant-rate, so a drop would
        shorten the part and slide every later frame away from the diary event
        that names it.

        The tick is serialised: a second caller waits for the one in progress
        and then finds the slots it consumed already taken, so a frame is never
        written twice and the part is as long as the run however many pumps
        there are.
        """
        deadline = self._clock() + max(0.0, float(seconds))
        while not self._stopped.is_set():
            now = self._clock()
            if now >= deadline:
                return
            with self._tick_lock:
                self._game_tick(now)
                self._robot_tick(now)
            self._idle.wait(min(max(deadline - now, 0.0), TICK_S))

    def stop(self) -> dict:
        """Close the encoders, write the manifest, and return it. Never raises.

        The composite is Task 4's: this leaves the run in ``composing`` with its
        parts on disk and a manifest that names them.
        """
        if not self._started or self._finished:
            found = read_manifest(self.run_dir)
            return found if found is not None else dict(self._manifest)
        self._finished = True
        self._stopped.set()
        self._idle.set()
        stuck = self._join_threads(THREAD_JOIN_S)
        wedged = {thread.name for thread in stuck}
        # A writer wedged in a full pipe: kill the encoders it holds so its write
        # ends with EPIPE rather than waiting on an ffmpeg that is not reading.
        # A source whose writer finished keeps its encoder and its part.
        for thread_name in wedged:
            for name in WEDGED_SOURCES.get(thread_name, ()):
                self._kill_encoder(name)
        if stuck:
            self._join_threads(1.0)
        if PUMP_THREAD not in wedged:
            self._flush_tile()
        if AUDIO_THREAD not in wedged:
            self._flush_audio()
        for name in SOURCES:
            self._close_stdin(name)
        for name in SOURCES:
            self._wait_encoder(name)
        self._join_threads(1.0)
        self._close_capture()
        self._close_stream()
        self._settle_sources()
        for name in SOURCES:
            self._close_encoder(name)
        self._ended_at = _utc_now()
        self._duration = round(self._clock() - self._started_monotonic, 2)
        return self._write_manifest("failed" if self._error else "composing")

    # ---- the three sources ----------------------------------------------

    def _probe_encoder(self) -> str | None:
        """The encoder's first `-version` line, or None with the failure recorded."""
        try:
            done = subprocess.run(
                [self._recording.ffmpeg, "-version"],
                capture_output=True, stdin=subprocess.DEVNULL, timeout=ENCODER_EXIT_S,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self._fatal(f"ffmpeg is not runnable: {exc}")
            return None
        if done.returncode != 0:
            lines = _lines(done.stderr) or _lines(done.stdout)
            detail = lines[0] if lines else f"exit {done.returncode}"
            self._fatal(f"ffmpeg -version failed: {detail}")
            return None
        lines = _lines(done.stdout) or _lines(done.stderr)
        return lines[0] if lines else self._recording.ffmpeg

    def _start_game(self) -> None:
        if self._stream_factory is None:
            self._note("game", "no depth stream")
            return
        try:
            self._stream = self._stream_factory()
        except (OSError, ValueError) as exc:
            self._note("game", f"no depth stream: {exc}")
            return
        proc = self._spawn_encoder(self._game_argv(self._part_path("game")), "game")
        if proc is None:
            self._close_stream()
            return
        self._procs["game"] = proc
        self._sources["game"] = "ok"

    def _start_robot(self) -> None:
        # The tile may not exist yet: the body view may still be loading. The
        # encoder waits on its pipe, and a run that never gets a tile is marked
        # missing at stop rather than recorded as an empty picture.
        proc = self._spawn_encoder(self._robot_argv(self._part_path("robot")), "robot")
        if proc is None:
            return
        self._procs["robot"] = proc
        self._sources["robot"] = "ok"

    def _start_audio(self) -> None:
        if self._capture is None:
            self._note("audio", "no capture available")
            return
        try:
            # `capture()` is a context manager, and the recorder cannot be one: it
            # is held open by the run, so `stop()` closes it.
            ctx = self._capture()
            reader, _proc, _info = ctx.__enter__()
        except Exception as exc:  # EarsError, OSError, anything the audio tools raise
            self._note("audio", f"no game audio: {exc}")
            return
        self._capture_ctx = ctx
        self._reader = reader
        proc = self._spawn_encoder(self._audio_argv(self._part_path("audio")), "audio")
        if proc is None:
            self._close_capture()
            return
        self._procs["audio"] = proc
        self._sources["audio"] = "ok"

    def _spawn_encoder(self, argv: list[str], name: str) -> Any | None:
        try:
            return self._spawn(
                argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            self._sources[name] = "failed"
            self._note(name, f"could not start the {name} encoder: {exc}")
            return None

    # ---- encoder argv ---------------------------------------------------

    def _game_argv(self, out: Path) -> list[str]:
        rec = self._recording
        return [
            rec.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgba",
            "-s", f"{GAME_WIDTH}x{GAME_HEIGHT}", "-r", str(rec.fps), "-i", "-",
            # The reader runs unflipped (210 fps against 58): ffmpeg turns it over.
            "-vf", "vflip",
            "-c:v", "libx264", "-preset", rec.preset, "-crf", str(rec.crf),
            "-pix_fmt", "yuv420p", str(out),
        ]

    def _robot_argv(self, out: Path) -> list[str]:
        rec = self._recording
        return [
            rec.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "image2pipe", "-vcodec", "mjpeg", "-framerate", str(rec.robot_fps),
            "-i", "-",
            "-vf", f"scale={ROBOT_WIDTH}:{HEIGHT}",
            "-c:v", "libx264", "-preset", rec.preset, "-crf", str(rec.crf),
            "-pix_fmt", "yuv420p", str(out),
        ]

    def _audio_argv(self, out: Path) -> list[str]:
        return [
            self._recording.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "-",
            "-c:a", "aac", "-b:a", "96k", str(out),
        ]

    # ---- the tick bodies ------------------------------------------------

    def _pump_loop(self) -> None:
        while not self._stopped.is_set():
            self.pump(self._period)

    def _audio_loop(self) -> None:
        while not self._stopped.is_set():
            if not self._audio_tick():
                self._idle.wait(TICK_S)

    def _game_tick(self, now: float) -> None:
        if self._sources["game"] == "ok":
            color = self._read_color()
            if color is not None:
                self._color = color
        due = 0
        while now >= self._next_frame_at:
            self._next_frame_at += self._period
            due += 1
        if not due or self._color is None or self._sources["game"] != "ok":
            return
        for _ in range(due):
            if not self._push("game", self._color):
                return

    def _robot_tick(self, now: float, force: bool = False) -> None:
        if self._sources["robot"] != "ok":
            return
        due = 0
        if force:
            due = 1
        else:
            while now >= self._next_robot_at:
                self._next_robot_at += self._robot_period
                due += 1
        if not due:
            return
        self._refresh_tile()
        if self._tile is None:
            return
        for _ in range(due):
            if not self._push("robot", self._tile):
                return

    def _audio_tick(self) -> bool:
        if not self._audio_lock.acquire(timeout=1.0):
            return False
        try:
            return self._audio_block()
        finally:
            self._audio_lock.release()

    def _audio_block(self) -> bool:
        if self._reader is None or self._sources["audio"] != "ok":
            return False
        chunk = self._reader.read(AUDIO_BLOCK)
        if not chunk:
            return False
        peak, _rms = level(chunk)
        self._peak = max(self._peak, peak)
        return self._push("audio", chunk)

    # ---- reading the two pictures ---------------------------------------

    def _read_color(self) -> bytes | None:
        """The newest RGBA frame, unflipped, or None when none is newer."""
        if self._stream is None:
            return None
        try:
            frame = self._stream.read(timeout=0.0, after_seq=self._seq, flip=False)
        except (OSError, ValueError) as exc:
            self._note_color_error(exc)
            return None
        self._color_errors = 0
        if not frame:
            return None
        self._seq = int(frame.get("seq") or 0)
        color = frame.get("color")
        if not color or len(color) != GAME_WIDTH * GAME_HEIGHT * 4:
            return None
        return color

    def _note_color_error(self, exc: Exception) -> None:
        """A read that raised is a note at first and a failed source at the bound.

        One bad read is a blink: the note keeps the reason for the gap, and the
        run carries on. A stream that has raised on every read for
        `STREAM_ERROR_LIMIT` ticks is dead, and padding its last frame forever
        under ``ok`` would report a still picture as a healthy recording.
        """
        self._color_errors += 1
        detail = f"the depth stream stopped answering: {exc}"
        if self._color_errors >= STREAM_ERROR_LIMIT and self._sources["game"] == "ok":
            self._sources["game"] = "failed"
            detail = (f"the depth stream stopped answering "
                      f"{self._color_errors} reads in a row: {exc}")
        self._note("game", detail)

    def _newest_tile(self) -> Path | None:
        directory = Path(self.cfg.paths.session_root) / "body"
        primary = directory / "body_latest.jpg"
        try:
            if primary.is_file():
                return primary
            tiles = [path for path in directory.glob("body_*.jpg") if path.is_file()]
            if not tiles:
                return None
            return max(tiles, key=lambda path: path.stat().st_mtime_ns)
        except OSError:
            return None

    def _refresh_tile(self) -> None:
        """Copy the newest JPEG, but only when the body view rewrote it."""
        path = self._newest_tile()
        if path is None:
            return
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if stamp == self._tile_stamp:
                return
            data = path.read_bytes()
        except OSError:
            return
        if data:
            self._tile, self._tile_stamp = data, stamp

    # ---- pushing to the encoders ----------------------------------------

    def _push(self, name: str, data: bytes) -> bool:
        proc = self._procs.get(name)
        if proc is None or not data:
            return False
        try:
            proc.stdin.write(data)
            proc.stdin.flush()
        except (OSError, ValueError) as exc:
            if not self._stopped.is_set():
                self._sources[name] = "failed"
                self._note(name, f"the {name} encoder stopped: {exc}")
            return False
        self._units[name] += 1
        return True

    def _flush_tile(self) -> None:
        """Write a tile that moved just before stop, so a run shorter than a tick
        still has the robot's picture."""
        self._robot_tick(self._clock(), force=True)

    def _flush_audio(self) -> None:
        """Drain the last PCM, so the peak in the manifest covers the whole run."""
        if self._reader is None or self._sources["audio"] != "ok":
            return
        if not self._audio_lock.acquire(timeout=1.0):
            return
        try:
            deadline = self._clock() + AUDIO_FLUSH_S
            while self._clock() < deadline and self._audio_block():
                pass
        finally:
            self._audio_lock.release()

    # ---- closing --------------------------------------------------------

    def _join_threads(self, timeout: float) -> list[threading.Thread]:
        deadline = time.monotonic() + timeout
        alive = []
        for thread in self._threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                alive.append(thread)
        return alive

    def _start_thread(self, target: Callable[[], None], name: str) -> None:
        thread = threading.Thread(target=target, name=name, daemon=True)
        self._threads.append(thread)
        thread.start()

    def _close_stdin(self, name: str) -> None:
        proc = self._procs.get(name)
        if proc is None or proc.stdin is None:
            return
        with contextlib.suppress(OSError, ValueError):
            proc.stdin.close()

    def _wait_encoder(self, name: str) -> None:
        proc = self._procs.get(name)
        if proc is None:
            return
        try:
            proc.wait(timeout=ENCODER_EXIT_S)
        except subprocess.TimeoutExpired:
            self._kill_encoder(name)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=2.0)

    def _kill_encoder(self, name: str) -> None:
        proc = self._procs.get(name)
        if proc is None or proc.poll() is not None:
            return
        try:
            # Each encoder leads its own session, so the group is the encoder.
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            with contextlib.suppress(OSError):
                proc.kill()

    def _close_encoder(self, name: str) -> None:
        """Close the encoder's pipes: its exit status and stderr have been read."""
        proc = self._procs.get(name)
        if proc is None:
            return
        for pipe in (proc.stdin, proc.stderr):
            if pipe is not None:
                with contextlib.suppress(OSError, ValueError):
                    pipe.close()

    def _close_capture(self) -> None:
        ctx, self._capture_ctx = self._capture_ctx, None
        if ctx is None:
            return
        with contextlib.suppress(Exception):
            ctx.__exit__(None, None, None)

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        with contextlib.suppress(OSError, ValueError):
            stream.close()

    def _settle_sources(self) -> None:
        """Decide what each source really produced, and drop a part that is not it."""
        for name in SOURCES:
            proc = self._procs.get(name)
            status = self._sources[name]
            if status == "ok" and proc is not None:
                code = proc.returncode
                if self._units[name] == 0:
                    status = "missing"
                    self._note(name, "nothing arrived")
                elif code not in (0, None):
                    status = "failed"
                    self._note(name, _stderr_line(proc) or f"the {name} encoder exited {code}")
                self._sources[name] = status
            if status != "ok":
                with contextlib.suppress(OSError):
                    (self.run_dir / self._part(name)).unlink()

    # ---- the manifest ---------------------------------------------------

    def _part(self, name: str) -> str:
        return f"{PARTS_DIR}/{PART_NAMES[name]}"

    def _part_path(self, name: str) -> Path:
        return self.run_dir / self._part(name)

    def _note(self, name: str, text: str) -> None:
        self._notes[name] = text

    def _fatal(self, reason: str) -> None:
        self._error = reason
        self._ended_at = _utc_now()
        self._duration = round(self._clock() - self._started_monotonic, 2)
        self._write_manifest("failed")

    def _write_manifest(self, state: str) -> dict:
        parts = {name: self._part(name) for name in SOURCES if self._sources[name] == "ok"}
        total = 0
        for name in parts:
            with contextlib.suppress(OSError):
                total += self._part_path(name).stat().st_size
        doc = {
            "available": bool(parts) and state != "failed",
            "file": None,          # Task 4 names the composite when it exists
            "started_at": self._started_at,
            "ended_at": self._ended_at,
            "duration_s": self._duration,
            "state": state,
            "width": WIDTH,
            "height": HEIGHT,
            "fps": self._recording.fps,
            "bytes": total,
            "sources": dict(self._sources),
            "audio_peak": int(self._peak),
            "encoder": self._encoder,
            "parts": parts,
            "notes": {name: text for name, text in self._notes.items() if text},
            "error": self._error,
        }
        self._manifest = doc
        try:
            path = self.manifest()
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            pass  # the run is worth more than its recording
        return dict(doc)


def read_manifest(run_dir: Path) -> dict | None:
    """A run's manifest, or None when it has none (or it cannot be read)."""
    try:
        raw = (Path(run_dir) / MANIFEST_NAME).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return doc if isinstance(doc, dict) else None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _lines(raw: bytes | None) -> list[str]:
    if not raw:
        return []
    return [line.strip() for line in raw.decode("utf-8", "replace").splitlines() if line.strip()]


def _stderr_line(proc: Any) -> str:
    """The encoder's last complaint, once it has exited."""
    if proc.stderr is None or proc.poll() is None:
        return ""
    try:
        lines = _lines(proc.stderr.read())
    except (OSError, ValueError):
        return ""
    return lines[-1][:200] if lines else ""


def parse_range(value: str, size: int) -> tuple[int, int] | None:
    """Inclusive (start, end) for a single ``bytes=`` range, or None for all.

    None means the header names no range this module serves as one: it is
    absent, in another unit, or a multi-range, and answering with the whole
    file is always legal. A malformed or unsatisfiable range raises ValueError
    so the caller can answer 416 rather than guess at what was meant.
    """
    value = value.strip()
    if not value.startswith("bytes="):
        return None
    spec = value[len("bytes="):].strip()
    if "," in spec:
        return None
    first, dash, last = spec.partition("-")
    if not dash:
        raise ValueError(f"not a byte range: {value!r}")

    if not first:
        if not last.isdigit():
            raise ValueError(f"unparsable suffix length: {value!r}")
        suffix = int(last)
        if suffix == 0 or size <= 0:
            raise ValueError(f"unsatisfiable range: {value!r}")
        return (max(size - suffix, 0), size - 1)

    if not first.isdigit() or (last and not last.isdigit()):
        raise ValueError(f"unparsable range: {value!r}")
    start = int(first)
    end = int(last) if last else size - 1
    if start >= size:
        raise ValueError(f"unsatisfiable range: {value!r}")
    if end < start:
        raise ValueError(f"empty range: {value!r}")
    return (start, min(end, size - 1))
