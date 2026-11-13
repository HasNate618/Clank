"""Turn a live run into one watchable file.

A run produces three encoded parts: the game's picture, the robot's picture,
and the game's audio. This module holds the argv for the one ffmpeg pass that
stacks them, the HTTP ``Range`` parser the dashboard's video route answers
with, and ``Recorder``, which runs an encoder per source while the run is live.
Both of the first two are pure, so the recorder and the composite that run
them are tested around them rather than through them.

``Recorder`` is the live half. The offline half finishes the job once the run
has ended: ``run_composite`` probes the parts, composites them and publishes
``recording.mp4``; ``sweep_stale`` cleans up after a dashboard that died mid-run;
``prune`` holds the recordings under their disk cap; ``list_recordings`` is what
the dashboard's page reads.

Nothing here raises into a run. A missing encoder, a dead pipe, an absent
depth stream: each becomes a manifest state or a source status, never an
exception the run has to survive.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from clank.ears import level

if TYPE_CHECKING:  # imported for the annotations only: config never imports this module
    from clank.config import Config

def tile_filter(width: int, height: int, source_width: int | None = None) -> str:
    """Fit a picture into `width`x`height` without changing its shape.

    The body render's size is the operator's own `[body] width/height`, and
    `config.toml` is git-ignored and machine-specific, so the recording cannot
    assume the render is already portrait: a 4:3 render squeezed into a 9:16 tile
    is a 33 % horizontal squash that no manifest records and no page can show.
    Pillarboxing is the honest outcome, and it is the same doctrine as the
    missing-source case: pad, never crop or stretch.
    """
    scale = f"scale={width}:{height}:force_original_aspect_ratio=decrease"
    return f"{scale},pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"


GAME_WIDTH = 960
GAME_HEIGHT = 540
#: The robot tile is 9:16 beside the game's 16:9. It is rendered portrait by the
#: body view ([body] width/height) rather than cropped to portrait here: cropping
#: a 4:3 render to 9:16 cuts the arms off.
ROBOT_WIDTH = 304
HEIGHT = GAME_HEIGHT
WIDTH = GAME_WIDTH + ROBOT_WIDTH

#: The manifest Task 4 composites from, written twice: at start and at stop.
MANIFEST_NAME = "recording.json"
PARTS_DIR = "parts"
PART_NAMES = {"game": "game.mp4", "robot": "robot.mp4", "audio": "audio.m4a"}
SOURCES = ("game", "robot", "audio")
#: The file a run's composite lands in, and the name a `ready` manifest carries.
RECORDING_NAME = "recording.mp4"
#: A manifest in either state belongs to a dashboard that died mid-run.
STALE_STATES = ("recording", "composing")
#: The composite's own bound: a re-encode that outlasts this is a wedged encoder.
COMPOSITE_TIMEOUT_S = 1800.0
#: Who is composing a run, as a pid. A live one means the work is in flight
#: somewhere else; a dead one means a dashboard exited mid-composite and the
#: next start may finish the job instead of deleting the evidence.
COMPOSITE_LOCK = "composite.pid"
#: A probe reads the container's index, never the frames, so this is generous.
PROBE_TIMEOUT_S = 60.0
#: Hold past the run's length, so `-t duration_s` never cuts inside a stream
#: whose encoded length landed a frame short of the schedule's.
PAD_MARGIN_S = 1.0
#: 16 kHz mono s16le, which is also the rate the audio encoder's pipe is read at.
AUDIO_RATE = 16000
AUDIO_BYTES_PER_S = AUDIO_RATE * 2
#: The most one read takes from the capture file: a quarter second of that pipe.
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
    game: Path | None,
    robot: Path | None,
    audio: Path | None,
    out: Path,
    fps: int,
    crf: int,
    preset: str,
    ffmpeg: str = "ffmpeg",
    pad: dict[str, float] | None = None,
    duration: float | None = None,
) -> list[str]:
    """The ffmpeg pass that stacks the parts side by side into `out`.

    A part that is missing is black padding of the same size, not a narrower
    file: the layout is fixed so every recording looks the same to a player.

    `pad` holds a picture's last frame for that many seconds, which is what
    keeps a part demoted mid-run from shortening the whole recording: without
    it a black pad makes the stack end with the picture that is real, and
    `-shortest` ends the file with the shortest stream. `duration` then cuts
    the result at the run's own length, which replaces `-shortest` -- an audio
    part that died early must leave a quiet tail, not a truncated picture.

    Nothing here reads the filesystem: the caller measures the parts, and this
    function writes down what to do about them.
    """
    sources = ((name, path) for name, path in (("game", game), ("robot", robot),
                                               ("audio", audio)) if path is not None)
    index: dict[str, int] = {}
    inputs: list[str] = []
    for name, path in sources:
        index[name] = len(index)
        inputs += ["-i", str(path)]

    holds = pad or {}
    left = f"[{index['game']}:v]{tile_filter(GAME_WIDTH, GAME_HEIGHT)}" if "game" in index \
        else f"color=black:s={GAME_WIDTH}x{GAME_HEIGHT}:r={fps}"
    left += _hold(holds.get("game", 0.0)) + "[g]"
    right = f"[{index['robot']}:v]{tile_filter(ROBOT_WIDTH, HEIGHT)}" if "robot" in index \
        else f"color=black:s={ROBOT_WIDTH}x{HEIGHT}:r={fps}"
    right += _hold(holds.get("robot", 0.0)) + "[r]"
    # An endless black pad cannot decide where the stack ends; without a target
    # the picture that is real does.
    black = "game" not in index or "robot" not in index
    stack = "[g][r]hstack=inputs=2:shortest=1[v]" if black else "[g][r]hstack=inputs=2[v]"

    argv = [
        ffmpeg, "-y", *inputs,
        "-filter_complex", ";".join([left, right, stack]),
        "-map", "[v]",
    ]
    if "audio" in index:
        argv += ["-map", f"{index['audio']}:a"]
    argv += [
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-r", str(fps),
    ]
    if "audio" in index:
        argv += ["-c:a", "aac", "-b:a", "96k"]
        if duration is None:
            argv += ["-shortest"]
    if duration is not None:
        argv += ["-t", f"{duration:.3f}"]
    argv += ["-movflags", "+faststart", str(out)]
    return argv


def _hold(seconds: float) -> str:
    """The filter that holds a picture's last frame for `seconds`, or nothing."""
    if seconds <= 0:
        return ""
    return f",tpad=stop_mode=clone:stop_duration={seconds:.3f}"


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

    A source demoted mid-run keeps the part it wrote, and ``parts`` is built
    from what is on disk, not from ``sources``: ten recorded minutes before the
    depth layer died are the run's evidence, and ``sources`` plus ``notes`` say
    the rest of the run is missing from it. ``part_seconds`` gives each kept
    part's real length. Two consequences for the composite (Task 4):
    ``composite_argv`` ends at ``-shortest``, so a demoted part truncates the
    whole recording unless it is padded or looped to ``duration_s``; and an
    encoder SIGKILLed mid-write (the wedged-writer path) leaves an MP4 with no
    moov atom, so probe a kept part before trusting it to play.

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
        self._written: dict[str, int] = dict.fromkeys(SOURCES, 0)
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
            "-vf", tile_filter(ROBOT_WIDTH, HEIGHT),
            "-c:v", "libx264", "-preset", rec.preset, "-crf", str(rec.crf),
            "-pix_fmt", "yuv420p", str(out),
        ]

    def _audio_argv(self, out: Path) -> list[str]:
        return [
            self._recording.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "s16le", "-ar", str(AUDIO_RATE), "-ac", "1", "-i", "-",
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
            # A live stream of the wrong size is a launch flag (`--no-color`, a
            # different `--color-scale`), not a dead layer. It goes through the
            # same bound as a read that raises, so the manifest says which size
            # it got instead of "nothing arrived".
            detail = (
                f"the colour frame is {frame.get('color_width')}x"
                f"{frame.get('color_height')}, the composite needs "
                f"{GAME_WIDTH}x{GAME_HEIGHT}"
            )
            self._note_color_error(ValueError(detail))
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
        self._written[name] += len(data)
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
        """Decide what each source really produced, and drop a part that holds nothing.

        Deletion is about bytes, never about status. A source demoted mid-run
        keeps the part it wrote, because the minutes before the depth layer died
        are the evidence and ``sources``/``notes`` already say the rest is gone.
        Only a part with no bytes is deleted: Task 4 cannot composite an empty
        file, and that is the ``missing`` case.
        """
        for name in SOURCES:
            proc = self._procs.get(name)
            status = self._sources[name]
            if status == "ok" and proc is not None:
                code = proc.returncode
                if self._units[name] == 0:
                    status = "missing"
                    # A source that already said why (a stream of the wrong
                    # size, a layer that never appeared) keeps that reason:
                    # "nothing arrived" is the answer of last resort, not an
                    # improvement on a specific one.
                    if not self._notes.get(name):
                        self._note(name, "nothing arrived")
                elif code not in (0, None):
                    status = "failed"
                    self._note(name, _stderr_line(proc) or f"the {name} encoder exited {code}")
                self._sources[name] = status
            if self._units[name] == 0 or not self._part_written(name):
                with contextlib.suppress(OSError):
                    (self.run_dir / self._part(name)).unlink()

    # ---- the manifest ---------------------------------------------------

    def _part(self, name: str) -> str:
        return f"{PARTS_DIR}/{PART_NAMES[name]}"

    def _part_path(self, name: str) -> Path:
        return self.run_dir / self._part(name)

    def _part_written(self, name: str) -> bool:
        """Whether a part is on disk with bytes in it, whatever its source says."""
        try:
            return self._part_path(name).stat().st_size > 0
        except OSError:
            return False

    def _part_seconds(self, name: str) -> float:
        """How long a part really is, from what the recorder already counted.

        The two pictures are constant-rate pipes whose every frame is written,
        so their length is the schedule's exactly: one written frame is one
        period. The audio pipe is a byte stream of known rate, so its length is
        the bytes fed over that rate. Guessing from the part's file size would
        not work: an MP4's size is its encoder's bitrate, not its duration.
        """
        if name == "game":
            return round(self._units[name] * self._period, 2)
        if name == "robot":
            return round(self._units[name] * self._robot_period, 2)
        return round(self._written[name] / AUDIO_BYTES_PER_S, 2)

    def _note(self, name: str, text: str) -> None:
        self._notes[name] = text

    def _fatal(self, reason: str) -> None:
        self._error = reason
        self._ended_at = _utc_now()
        self._duration = round(self._clock() - self._started_monotonic, 2)
        self._write_manifest("failed")

    def _write_manifest(self, state: str) -> dict:
        # A part is named because it is on disk with bytes in it, not because
        # its source is `ok`: a demoted source keeps what it recorded.
        parts = {name: self._part(name) for name in SOURCES if self._part_written(name)}
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
            "part_seconds": {name: self._part_seconds(name) for name in parts},
            "notes": {name: text for name, text in self._notes.items() if text},
            "error": self._error,
        }
        self._manifest = doc
        _save_manifest(self.run_dir, doc)
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


def probe_part(path: Path, *, want: str, ffprobe: str = "ffprobe") -> tuple[dict | None, str]:
    """What the demuxer says a part holds, or (None, why it is unusable).

    An encoder SIGKILLed mid-write leaves an MP4 with no moov atom, which no
    player and no muxer can demux. `ffprobe` reads the container's index and
    not its frames, so this is a whole part for the price of a header; a part
    that fails here is dropped before it can fail the composite.

    `want` is the stream kind the part exists for: ``video`` for the two
    pictures, ``audio`` for the sound.
    """
    argv = [
        ffprobe, "-v", "error", "-print_format", "json",
        "-show_entries", "stream=codec_type:format=duration", str(path),
    ]
    try:
        done = subprocess.run(
            argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"ffprobe is not runnable: {exc}"
    if done.returncode != 0:
        lines = _lines(done.stderr)
        return None, (lines[-1][:200] if lines else f"ffprobe exited {done.returncode}")
    try:
        doc = json.loads(done.stdout)
    except ValueError:
        return None, "ffprobe printed no stream information"
    if not isinstance(doc, dict):
        return None, "ffprobe printed no stream information"
    streams = [item.get("codec_type") for item in doc.get("streams") or []
               if isinstance(item, dict)]
    if want not in streams:
        return None, f"no {want} stream"
    try:
        seconds = float((doc.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        seconds = 0.0
    return {"streams": streams, "duration_s": seconds}, ""


def _pid_alive(pid: int) -> bool:
    """Whether a process exists. EPERM means it does and is not ours."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _lock_owner(run_dir: Path) -> int | None:
    try:
        text = (Path(run_dir) / COMPOSITE_LOCK).read_text(encoding="utf-8").strip()
        return int(text)
    except (OSError, ValueError):
        return None


def composite_is_live(run_dir: Path) -> bool:
    """True when a process that still exists holds this run's composite."""
    owner = _lock_owner(run_dir)
    return owner is not None and _pid_alive(owner)


def _acquire_composite(run_dir: Path) -> bool:
    """Claim the composite for this process, or report that it is taken.

    O_EXCL, so two dashboards cannot both compose one run. A lock naming a
    process that is gone is litter left by a dashboard that exited mid-encode:
    remove it and take over, which is exactly how an interrupted composite gets
    finished instead of abandoned.
    """
    path = Path(run_dir) / COMPOSITE_LOCK
    for _attempt in (0, 1):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            if composite_is_live(run_dir):
                return False
            with contextlib.suppress(OSError):
                path.unlink()
            continue
        except OSError:
            return True  # cannot lock here; do not stand in the way of composing
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        return True
    return False


def _release_composite(run_dir: Path) -> None:
    path = Path(run_dir) / COMPOSITE_LOCK
    if _lock_owner(run_dir) != os.getpid():
        return
    with contextlib.suppress(OSError):
        path.unlink()


def _temp_name() -> str:
    """The composite's output before it is published.

    It keeps the `.mp4` suffix on purpose: ffmpeg infers the container from the
    extension, and a bare `.part.<pid>` is refused as an invalid output. The pid
    keeps a resumed attempt from interleaving with an orphan of a dead dashboard.
    """
    return f"{Path(RECORDING_NAME).stem}.part.{os.getpid()}.mp4"


def _has_parts(run_dir: Path) -> bool:
    """Whether this run still holds bytes a composite could be built from."""
    parts = Path(run_dir) / PARTS_DIR
    if parts.is_symlink():
        return False  # a link out of the run is never something to build from
    try:
        return any(item.stat().st_size > 0 for item in parts.iterdir() if item.is_file())
    except OSError:
        return False


def _sweep_temporaries(run_dir: Path) -> None:
    """Remove composite outputs an exited process left unfinished.

    A composite writes beside its target and renames on success, so a leftover
    `.part.<pid>` is by definition a file nobody finished. One whose pid is
    still alive is a compose happening right now and is left alone.
    """
    for leftover in Path(run_dir).glob(f"{Path(RECORDING_NAME).stem}.part.*.mp4"):
        bits = leftover.name.split(".")
        suffix = bits[-2] if len(bits) >= 2 else ""
        if suffix.isdigit() and _pid_alive(int(suffix)):
            continue
        _remove_tree(leftover)


def run_composite(run_dir: Path, cfg: Config) -> dict:
    """Composite a finished recording's parts into `recording.mp4`, and publish it.

    Synchronous and bounded, because the caller detaches it into a thread once
    the run has gone idle. It probes every part the manifest names before it
    composites them, drops the ones that cannot be demuxed with a note, and
    probes the composite's own output before it publishes it: ``ready`` means a
    file that demuxes and carries a picture, not a process that exited 0. Every
    kept picture is padded to the run's own length so a part demoted mid-run
    cannot shorten the file, and ``duration_s`` is the length the output probe
    measured. The parts are deleted unless `keep_parts`; a failed composite
    keeps them, because they are what the failure is diagnosed from.

    Returns the manifest it wrote. Prunes the recordings afterwards, never
    deleting the run it just published.
    """
    run_dir = Path(run_dir).resolve()
    manifest = read_manifest(run_dir)
    if manifest is None:
        return {
            "state": "failed", "error": "no recording manifest to compose",
            "available": False, "file": None,
        }
    if manifest.get("state") == "ready":
        return manifest  # already published; a second call must not delete parts
    if manifest.get("state") == "failed":
        # The recorder's own failure (no encoder, no parts directory) already
        # says why, and there is nothing to compose: composing it again would
        # replace a precise reason with "nothing survived the probe".
        return manifest
    if not _acquire_composite(run_dir):
        # Another process is composing this run; its call publishes. Returning
        # the manifest unchanged is the honest answer to "did I compose this".
        return manifest
    try:
        return _composite(run_dir, manifest, cfg)
    except Exception as exc:  # a detached thread must still leave a manifest
        return _fail(run_dir, manifest, f"the composite failed: {exc}", {}, {}, {})
    finally:
        _release_composite(run_dir)


def _composite(run_dir: Path, manifest: dict, cfg: Config) -> dict:
    rec = cfg.recording
    notes = dict(manifest.get("notes") or {})
    recorded = dict(manifest.get("part_seconds") or {})
    named = dict(manifest.get("parts") or {})
    ffprobe = _ffprobe(rec.ffmpeg)
    kept: dict[str, Path] = {}
    lengths: dict[str, float] = {}
    for name in SOURCES:
        relative = named.get(name)
        if not relative:
            continue
        # Absolute, always: the dashboard's cwd is not the run directory.
        path = (run_dir / relative).resolve()
        if not path.is_file():
            notes[name] = "the part is gone"
            continue
        probed, problem = probe_part(
            path, want=("audio" if name == "audio" else "video"), ffprobe=ffprobe,
        )
        if probed is None:
            notes[name] = f"the part cannot be demuxed: {problem}"
            continue
        kept[name] = path
        lengths[name] = float(probed.get("duration_s") or 0.0) or float(recorded.get(name) or 0.0)
    parts = {name: named[name] for name in SOURCES if name in kept}
    seconds = {name: lengths[name] for name in parts}

    if "game" not in kept and "robot" not in kept:
        detail = notes.get("game") or notes.get("robot") or "no parts"
        return _fail(run_dir, manifest, f"no picture survived the probe: {detail}",
                     notes, {}, {})

    length = max([float(manifest.get("duration_s") or 0.0), *lengths.values()])
    duration = round(length, 3) if length > 0 else None
    pad: dict[str, float] = {}
    if duration is not None:
        for name in ("game", "robot"):
            if name in kept:
                pad[name] = max(0.0, duration - lengths.get(name, 0.0)) + PAD_MARGIN_S

    out = run_dir / RECORDING_NAME
    # Write beside the target and rename: an interrupted composite must never
    # leave a `recording.mp4` that looks published, and the pid keeps a resumed
    # attempt from interleaving with an orphan of the dashboard that died.
    temp = run_dir / _temp_name()
    argv = composite_argv(
        game=kept.get("game"), robot=kept.get("robot"), audio=kept.get("audio"),
        out=temp, fps=rec.fps, crf=rec.crf, preset=rec.preset, ffmpeg=rec.ffmpeg,
        pad=pad, duration=duration,
    )
    # A half-written, unplayable or empty `recording.mp4` is worse than none:
    # `_fail` publishes `file: null`, so neither the sweep (only STALE_STATES)
    # nor the prune (only `ready`) would ever come back for it.
    try:
        done = subprocess.run(
            argv, stdin=subprocess.DEVNULL, capture_output=True,
            timeout=COMPOSITE_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _remove_tree(temp)
        return _fail(run_dir, manifest, f"the composite could not run: {exc}",
                     notes, parts, seconds)
    if done.returncode != 0:
        lines = _lines(done.stderr)
        detail = lines[-1][:200] if lines else f"ffmpeg exited {done.returncode}"
        _remove_tree(temp)
        return _fail(run_dir, manifest, f"the composite failed: {detail}", notes, parts, seconds)
    try:
        size = temp.stat().st_size
    except OSError:
        size = 0
    if size <= 0:
        _remove_tree(temp)
        return _fail(run_dir, manifest, "the composite wrote nothing", notes, parts, seconds)
    probed_out, problem = probe_part(temp, want="video", ffprobe=ffprobe)
    if probed_out is None:
        _remove_tree(temp)
        return _fail(run_dir, manifest,
                     f"the composite wrote no playable video: {problem}",
                     notes, parts, seconds)
    try:
        os.replace(temp, out)
    except OSError as exc:
        _remove_tree(temp)
        return _fail(run_dir, manifest,
                     f"the composite could not be published: {exc}",
                     notes, parts, seconds)
    measured = float(probed_out.get("duration_s") or 0.0)
    # A measurement beats a target: the run asked for `duration`, the file is
    # what it is, and the page shows the length a player will actually run.
    measured_s = round(measured, 2) if measured > 0 else (
        round(duration, 2) if duration is not None else manifest.get("duration_s")
    )

    doc = dict(manifest)
    doc.update({
        "state": "ready",
        "file": RECORDING_NAME,
        "available": True,
        "error": None,
        "bytes": size,
        "width": WIDTH,
        "height": HEIGHT,
        "fps": rec.fps,
        "duration_s": measured_s,
        "notes": notes,
        "parts": parts,
        "part_seconds": {name: round(value, 2) for name, value in seconds.items()},
    })
    if not rec.keep_parts:
        _remove_tree(run_dir / PARTS_DIR)
        doc["parts"] = {}
        doc["part_seconds"] = {}
    published = _save_manifest(run_dir, doc)
    # Housekeeping, not part of the publication: a prune that raised here would
    # hand this call's blanket `except` a `ready` run to demote to `failed`.
    with contextlib.suppress(Exception):
        prune(Path(cfg.paths.session_root), rec.max_total_mb, keep=run_dir)
    return published


def _fail(
    run_dir: Path, manifest: dict, error: str, notes: dict, parts: dict, seconds: dict,
) -> dict:
    """Publish the failure: what could not be done, and what is left on disk.

    The parts stay on purpose: a failed recording is diagnosed from them, and
    they are the one thing a re-probe or a retry has to work from.
    """
    doc = dict(manifest)
    doc.update({
        "state": "failed",
        "error": error,
        "available": False,
        "file": None,
        "notes": notes,
        "parts": parts,
        "part_seconds": seconds,
    })
    if not doc.get("ended_at"):
        doc["ended_at"] = _utc_now()
    return _save_manifest(run_dir, doc)


def sweep_stale(sessions_root: Path) -> list[Path]:
    """Clean up after a dashboard that died with a run in flight.

    Called once at dashboard start, after the server has bound its port and
    before any run exists: a dashboard refused its port deletes nothing, and
    nothing it meets can be live yet. One dashboard per ``session_root`` is the
    rule this rests on -- two on different ports would still both sweep.

    **An unfinished composite is not a corpse.** A manifest left ``recording``
    or ``composing`` whose parts are still on disk is handed to
    `pending_composites`, which the next dashboard finishes: a whole run is
    re-encoded in minutes, so exiting inside that window is ordinary, and
    deleting the video *and* the parts that could rebuild it loses the only copy
    of what the run did. Only a run nothing can be rebuilt from -- no manifest,
    no parts, an encoder that never wrote -- is marked ``failed: dashboard
    exited``. A composite still running under a live pid is left entirely alone.
    Leftover ``recording.mp4.part.<pid>`` files from dead processes are removed.

    A ``ready`` run is untouched, so its ``recording.mp4`` survives; a
    ``failed`` run keeps its parts as evidence.

    Returns the run directories it swept. Deletion is confined to a run
    directory found under the root: a symlinked session, ``runs`` or run
    directory is skipped rather than followed.
    """
    swept: list[Path] = []
    for run_dir, _session in _runs(sessions_root):
        _sweep_temporaries(run_dir)
        manifest = read_manifest(run_dir)
        if manifest is not None and manifest.get("state") not in STALE_STATES:
            continue
        if manifest is None and not (run_dir / PARTS_DIR).exists():
            continue  # nothing was ever recorded here
        if composite_is_live(run_dir):
            continue  # composing right now, in a process that still exists
        if manifest is not None and _has_parts(run_dir):
            continue  # unfinished, not dead: pending_composites() resumes it
        doc = dict(manifest or {})
        # Capture a time before the parts go: a run with no manifest has no
        # `started_at` of its own, and an undated row sorts to the bottom of the
        # page as if it were the oldest thing on it.
        started = doc.get("started_at") or _mtime_iso(run_dir / PARTS_DIR)
        _remove_tree(run_dir / PARTS_DIR)
        _remove_tree(run_dir / RECORDING_NAME)
        doc.update({
            "state": "failed",
            "error": "dashboard exited",
            "available": False,
            "file": None,
            "parts": {},
            "part_seconds": {},
            "bytes": 0,
        })
        if started:
            doc["started_at"] = started
        if not doc.get("ended_at"):
            doc["ended_at"] = _utc_now()
        _save_manifest(run_dir, doc)
        swept.append(run_dir)
    return swept


def pending_composites(sessions_root: Path) -> list[Path]:
    """Runs left mid-composite by a dashboard that exited, ready to be finished.

    The work outlives the process that started it: the parts are on disk, the
    manifest names them, and `run_composite` is idempotent and probes before it
    publishes. So the next dashboard finishes the encode rather than deleting
    the evidence. A run whose lock names a live process is somebody else's
    in-flight composite and is skipped, which is what keeps two dashboards on
    one session root from racing.
    """
    pending: list[Path] = []
    for run_dir, _session in _runs(sessions_root):
        manifest = read_manifest(run_dir)
        if manifest is None or manifest.get("state") not in STALE_STATES:
            continue
        if not _has_parts(run_dir) or composite_is_live(run_dir):
            continue
        pending.append(run_dir)
    return pending


def prune(sessions_root: Path, max_total_mb: int, keep: Path | None = None) -> list[Path]:
    """Reclaim the oldest finished recordings' video until the total fits the cap.

    Only a ``ready`` recording counts and only a ``ready`` recording is pruned:
    a run whose composite has not finished has no size to count yet, and one
    that failed has no video. ``keep`` is the run that just composed and is
    never pruned, even when it alone is over the cap: deleting the thing that
    was just asked for is not pruning. ``0`` disables pruning.

    The cap measures video, so the video is what goes: ``recording.mp4`` and the
    ``parts/`` it was cut from. The run's diary (``brain.jsonl``, ``brain.log``),
    ``run.json`` and ``frames/`` stay -- kilobytes beside the megabytes, and the
    operator's other named requirement -- and the manifest is rewritten as
    ``pruned`` so the page lists the run and says why there is nothing to play.
    One line per prune names the run and its bytes in the session's
    ``session.log``.

    Returns the run directories it pruned.
    """
    if max_total_mb <= 0:
        return []
    cap = max_total_mb * 1024 * 1024
    keep_dir = Path(keep).resolve() if keep is not None else None
    ready: list[tuple[str, Path, Path, int, dict]] = []
    for run_dir, session_dir in _runs(sessions_root):
        manifest = read_manifest(run_dir)
        if manifest is None or manifest.get("state") != "ready":
            continue
        try:
            size = (run_dir / RECORDING_NAME).stat().st_size
        except OSError:
            continue  # nothing on disk to reclaim
        ready.append((str(manifest.get("started_at") or ""), run_dir, session_dir, size,
                      manifest))
    total = sum(size for *_rest, size, _manifest in ready)
    removed: list[Path] = []
    for _started, run_dir, session_dir, size, manifest in sorted(
        ready, key=lambda row: (row[0], row[1].name)
    ):
        if total <= cap:
            break
        if keep_dir is not None and run_dir.resolve() == keep_dir:
            continue
        _remove_tree(run_dir / RECORDING_NAME)
        _remove_tree(run_dir / PARTS_DIR)
        doc = dict(manifest)
        doc.update({
            "state": "pruned",
            "available": False,
            "file": None,
            "error": "pruned to make room",
            "bytes": 0,
            "parts": {},
            "part_seconds": {},
        })
        _save_manifest(run_dir, doc)
        total -= size
        _append_session_log(session_dir, f"recording pruned: {run_dir.name} {size} bytes")
        removed.append(run_dir)
    return removed


def list_recordings(sessions_root: Path) -> list[dict]:
    """Every run worth listing, newest first: its video if it has one, its diary if not.

    ``has_file`` is whether the video is really there and ``has_diary`` whether
    the run left one, so the page can say what a run is missing instead of
    offering a player that cannot start. A run with a diary and no manifest is
    listed with ``state: None``: recording can be off on a machine, or a run can
    predate the feature, and the diary is half of what this page is for --
    hiding those runs makes the operator's second requirement unreachable.
    """
    rows: list[dict] = []
    for run_dir, session_dir in _runs(sessions_root):
        manifest = read_manifest(run_dir)
        if manifest is None:
            diary = run_dir / "brain.jsonl"
            if not diary.is_file():
                continue
            rows.append({
                "run_id": run_dir.name,
                "session_id": session_dir.name,
                "dir": str(run_dir),
                "state": None,
                "started_at": _mtime_iso(diary),
                "ended_at": None,
                "duration_s": None,
                "bytes": None,
                "width": None,
                "height": None,
                "fps": None,
                "audio_peak": None,
                "error": None,
                "has_file": False,
                "has_diary": True,
            })
            continue
        name = manifest.get("file")
        rows.append({
            "run_id": run_dir.name,
            "session_id": session_dir.name,
            "dir": str(run_dir),
            "state": manifest.get("state"),
            "started_at": manifest.get("started_at"),
            "ended_at": manifest.get("ended_at"),
            "duration_s": manifest.get("duration_s"),
            "bytes": manifest.get("bytes"),
            "width": manifest.get("width"),
            "height": manifest.get("height"),
            "fps": manifest.get("fps"),
            "audio_peak": manifest.get("audio_peak"),
            "error": manifest.get("error"),
            "has_file": bool(name) and (run_dir / str(name)).is_file(),
            "has_diary": (run_dir / "brain.jsonl").is_file(),
        })
    rows.sort(key=lambda row: (str(row.get("started_at") or ""), str(row.get("run_id"))),
              reverse=True)
    return rows


def _mtime_iso(path: Path) -> str | None:
    """When a file was last written, in the manifests' own ISO form.

    A run with a diary and no manifest has no `started_at` of its own, and the
    diary's mtime is the next honest thing: it is when the run last said
    something. None if the file cannot be stat'ed, which sorts it out of the way.
    """
    try:
        moment = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    except OSError:
        return None
    return moment.isoformat().replace("+00:00", "Z")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _save_manifest(run_dir: Path, doc: dict) -> dict:
    """The manifest replaced in one step, so a reader never sees half a file."""
    path = Path(run_dir) / MANIFEST_NAME
    try:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass  # the evidence on disk is worth more than its index
    return doc


def _ffprobe(ffmpeg: str) -> str:
    """The probe beside the configured encoder, else the one on `PATH`.

    ffmpeg and ffprobe ship together, so an operator (or a test) who names one
    by path has the other in the same directory; a bare `ffmpeg` is resolved on
    `PATH` like the encoder itself.
    """
    beside = Path(ffmpeg).with_name("ffprobe")
    if Path(ffmpeg).parent != Path(".") and beside.is_file():
        return str(beside)
    return "ffprobe"


def _runs(sessions_root: Path) -> list[tuple[Path, Path]]:
    """Every real run directory under the root, with its session directory.

    The only place the sweep and the prune learn a path, and the reason neither
    can leave the root: a symlinked session, ``runs`` or run directory is not
    one of ours, so it is skipped rather than followed.
    """
    found: list[tuple[Path, Path]] = []
    for session_dir in _subdirs(Path(sessions_root)):
        runs_dir = session_dir / "runs"
        if runs_dir.is_symlink() or not runs_dir.is_dir():
            continue
        for run_dir in _subdirs(runs_dir):
            found.append((run_dir, session_dir))
    return found


def _subdirs(path: Path) -> list[Path]:
    try:
        entries = sorted(path.iterdir())
    except OSError:
        return []
    return [entry for entry in entries if not entry.is_symlink() and entry.is_dir()]


def _remove_tree(path: Path) -> None:
    """Delete a file or a directory, and never delete through a symlink."""
    try:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
            return
        shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


def _append_session_log(session_dir: Path, message: str) -> None:
    """One line in the session's log, the shape `orchestrator` writes."""
    try:
        with (session_dir / "session.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{_utc_now()} {message}\n")
    except OSError:
        pass


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
