"""Hear the game, and nothing else.

The game is a PipeWire client like any other, so its audio can be isolated - but doing it
took three measured corrections, and the third one is the reason this module does not use
`pw-record`:

1. `pw-record --target <node>` **does not link to the node**. On this machine (PipeWire
   1.6.8, pactl 17) it silently records the *default source* instead, which is a muted
   microphone, so every capture is digital silence with a zero exit status. Tapping the
   game's own `Stream/Output/Audio` node that way reads zero while the game is audible, and
   so does a tone played back in the same command.
2. The PulseAudio-compatible `parecord --device=<name>` **does** link, and reads a real
   level.
3. Monitor sources exist for the PulseAudio API but `pw-dump` lists none of them, which is
   why they look absent. `pactl list short sinks` is the honest list.

So capture is: a null sink named `clank-game` that only the game feeds, a loopback from it
back to the operator's real sink so the game stays audible, and `parecord` on
`clank-game.monitor`. The desktop mix never enters the capture, and a notification or the
operator talking cannot be mistaken for JobBot.

Recording is gated on speech: 50 ms RMS blocks, the floor is the 20th percentile of every
block before the gate opened, and the utterance ends after `hang_s` (2.0 s) of quiet.
Digital silence between lines is the normal state of this game, so the gate keys on the floor
being *known*, not on it being non-zero.

The floor covers the whole pre-start period and is never recomputed from a rolling window: a
capture that opens on the game already talking otherwise takes the speech for the floor and
gates itself deaf to that same speech. MEASURED on the confirm window of run
190131-679de2 - the first half second's median was 2445, so the gate sat at 9781 while every
block of an audible line peaked at 4415.

**The percentile has a limit worth knowing:** it only rescues a loud opening when at least a
fifth of the pre-start blocks are quiet, because the 20th percentile of mostly-loud blocks is
loud. Real speech has word gaps, so it usually does; a capture that is loud from its first block
to its last still never opens, and `test_continuous_loud_audio_is_not_an_utterance` pins that.
That question - "what did the game just say" - is what `record()` is for, and it has no gate.
"""

from __future__ import annotations

import array
import collections
import contextlib
import io
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import wave
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

#: 16 kHz mono is what a speech model wants and all the game needs.
RATE = 16000
#: `application.name` of the game's playback stream, matched case-insensitively.
DEFAULT_PATTERN = "JobSimulator"
#: The null sink only the game feeds. Its monitor is a real capture device.
SINK_NAME = "clank-game"
#: The node class that carries an application's output.
PLAYBACK_CLASS = "Stream/Output/Audio"
BLOCK_S = 0.05
#: Legal-looking speech gate for a quiet mix: four times the floor, never below this.
GATE_MIN_RMS = 250.0
#: `pactl` and `parecord` are both PulseAudio-compatible front ends; the native
#: `pw-record --target` does not link on this machine, measured above.
PACTL = "pactl"
PARECORD = "parecord"
CAPTURE_TIMEOUT_S = 10.0
#: How long a capture may produce nothing before it is treated as gone, once it has
#: started. Opening a Bluetooth sink or a loopback can take a second or two on the first
#: read, so startup gets its own, longer allowance.
#: `parecord` flushes its WAV in bursts of a second or more, so a short gap is not death.
EOF_GRACE_S = 4.0
STARTUP_GRACE_S = 6.0


class EarsError(RuntimeError):
    """No game stream, no tool, or the capture died before it produced audio."""


@dataclass(frozen=True)
class Hearing:
    """One captured utterance and the numbers that say whether it was one.

    `speech_s` is the time the gate called speech. A fixed-length `record()` has no gate,
    so it equals `seconds` there.

    `ended_by` names the rule that stopped `listen`: `hang` (the quiet after the utterance),
    `deadline` (`max_s` ran out), `eof` (the capture ended), or `never-started` (the gate
    never opened, so there was no utterance to end). `record()` is a fixed window - it stops
    when the window is full - and leaves `ended_by` empty.
    """

    wav: bytes = b""
    seconds: float = 0.0
    speech_s: float = 0.0
    peak: int = 0
    rms: float = 0.0
    node: int = 0
    node_name: str = ""
    device: str = ""
    floor: float = 0.0
    gate: float = 0.0
    ended_by: str = ""
    error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.error and self.speech_s > 0.0 and bool(self.wav)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "seconds": round(self.seconds, 2),
            "speech_s": round(self.speech_s, 2),
            "peak": self.peak,
            "rms": round(self.rms, 1),
            "node": self.node,
            "node_name": self.node_name,
            "device": self.device,
            "floor": round(self.floor, 1),
            "gate": round(self.gate, 1),
            "ended_by": self.ended_by,
            "bytes": len(self.wav),
            "error": self.error,
        }


def _state_dir() -> pathlib.Path:
    base = os.environ.get("CLANK_EARS_DIR", "")
    if base:
        return pathlib.Path(base)
    return pathlib.Path.home() / ".local" / "state" / "Clank" / "ears"


# --------------------------------------------------------------------------- #
# Talking to PipeWire through pactl
# --------------------------------------------------------------------------- #


def _pactl(*args: str, timeout: float = CAPTURE_TIMEOUT_S) -> str:
    try:
        proc = subprocess.run([PACTL, *args], capture_output=True, timeout=timeout,
                              check=False)
    except FileNotFoundError as exc:
        raise EarsError("pactl is not installed; the PulseAudio tools are required") from exc
    except subprocess.TimeoutExpired as exc:
        raise EarsError(f"pactl {' '.join(args)} timed out") from exc
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()[:200]
        raise EarsError(f"pactl {' '.join(args)} failed: {detail or proc.returncode}")
    return proc.stdout.decode("utf-8", "replace")


def _pactl_json(*args: str) -> Any:
    raw = _pactl("-f", "json", *args)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EarsError(f"pactl {' '.join(args)} did not return json: {exc}") from exc


def sinks() -> list[dict[str, Any]]:
    """Every sink, as pactl describes it."""
    parsed = _pactl_json("list", "sinks")
    return [s for s in parsed if isinstance(s, dict)] if isinstance(parsed, list) else []


def sink_names() -> list[str]:
    return [str(s.get("name", "")) for s in sinks() if s.get("name")]


def sink_inputs() -> list[dict[str, Any]]:
    """Every application output stream, as pactl describes it."""
    parsed = _pactl_json("list", "sink-inputs")
    return [s for s in parsed if isinstance(s, dict)] if isinstance(parsed, list) else []


def game_streams(pattern: str = DEFAULT_PATTERN) -> list[dict[str, Any]]:
    """The sink-inputs whose application name matches the game."""
    needle = pattern.strip().lower()
    if not needle:
        return []
    found = []
    for stream in sink_inputs():
        props = stream.get("properties") or {}
        name = " ".join(str(props.get(key, "")) for key in
                        ("application.name", "application.process.binary", "media.name"))
        if needle in name.lower():
            found.append(stream)
    return found


def ensure_sink(name: str = SINK_NAME) -> bool:
    """Create the capture sink if it is missing. True when it was created."""
    if name in sink_names():
        return False
    _pactl("load-module", "module-null-sink", f"sink_name={name}",
           f"sink_properties=device.description={name}")
    return True


def _loopback_loaded(sink: str) -> bool:
    """Whether a loopback already carries `sink`'s monitor somewhere."""
    for line in _pactl("list", "short", "modules").splitlines():
        if "module-loopback" in line and f"source={sink}.monitor" in line:
            return True
    return False


def ensure_loopback(sink: str = SINK_NAME, target: str = "") -> bool:
    """Make the capture sink audible again, so moving the game does not silence it."""
    if _loopback_loaded(sink):
        return False
    destination = target or _pactl("get-default-sink").strip()
    if not destination or destination == sink:
        return False
    _pactl("load-module", "module-loopback", f"source={sink}.monitor",
           f"sink={destination}", "latency_msec=20")
    return True


def move_game_streams(pattern: str = DEFAULT_PATTERN, sink: str = SINK_NAME) -> list[int]:
    """Route the game's streams into the capture sink. Returns the indices it moved."""
    moved: list[int] = []
    for stream in game_streams(pattern):
        index = stream.get("index")
        if not isinstance(index, int):
            continue
        if str(stream.get("sink")) == _sink_id(sink):
            continue
        _pactl("move-sink-input", str(index), sink)
        moved.append(index)
    return moved


def unmute_game_streams(pattern: str = DEFAULT_PATTERN) -> list[int]:
    """Clear a remembered mute on the game's streams. Returns the indices it unmuted.

    MEASURED, twice: `module-stream-restore` remembers `Mute: yes` for `JobSimulator.exe`, and
    it reapplies it to every NEW sink-input the game opens - so a restarted game is silent while
    the recorder works perfectly, the route is correct, the loopback is running and the monitor
    reads peak 0 for the whole session. Nothing in the pipeline can tell that from a quiet game.

    The mute is per stream, so this has to run after the game has opened one, and it has to run
    again after every restart. `route()` calls it for that reason.
    """
    cleared: list[int] = []
    for stream in game_streams(pattern):
        index = stream.get("index")
        if not isinstance(index, int):
            continue
        if not stream.get("mute"):
            continue
        _pactl("set-sink-input-mute", str(index), "0")
        cleared.append(index)
    return cleared


def _sink_id(name: str) -> str:
    for entry in sinks():
        if entry.get("name") == name:
            return str(entry.get("index", ""))
    return ""


def capture_device(pattern: str = DEFAULT_PATTERN, sink: str = SINK_NAME) -> str:
    """The device to record: the capture sink's monitor once the game is behind it."""
    return f"{sink}.monitor"


def route(pattern: str = DEFAULT_PATTERN, sink: str = SINK_NAME) -> dict[str, Any]:
    """Put the game behind the capture sink and report what had to change.

    Idempotent: a second call creates nothing and moves nothing, which is what makes it
    safe to call before every utterance.
    """
    if not game_streams(pattern):
        raise EarsError(f"no playback stream matching {pattern!r} "
                        f"(playing: {_playing()})")
    created = ensure_sink(sink)
    looped = ensure_loopback(sink)
    moved = move_game_streams(pattern, sink)
    # A muted stream is silence that looks like a quiet game, so it is cleared here: measured,
    # `module-stream-restore` re-mutes every new sink-input the game opens.
    unmuted = unmute_game_streams(pattern)
    return {"sink": sink, "created": created, "loopback": looped,
            "moved": moved, "unmuted": unmuted, "device": capture_device(pattern, sink)}


def _playing() -> str:
    names = []
    for stream in sink_inputs():
        props = stream.get("properties") or {}
        name = props.get("application.name")
        if name:
            names.append(str(name))
    return ", ".join(sorted(set(names))) or "nothing"


# --------------------------------------------------------------------------- #
# pw-dump, kept for diagnosis: which nodes exist at all
# --------------------------------------------------------------------------- #


def _node_props(obj: dict[str, Any]) -> dict[str, Any]:
    info = obj.get("info")
    if not isinstance(info, dict):
        return {}
    props = info.get("props")
    return props if isinstance(props, dict) else {}


def _node_label(props: dict[str, Any]) -> str:
    for key in ("application.name", "node.name", "application.process.binary",
                "node.description"):
        value = props.get(key)
        if isinstance(value, str) and value:
            return value
    return "?"


def playback_nodes(dump: Iterable[Any]) -> list[tuple[int, str, dict[str, Any]]]:
    """Every application output stream in a `pw-dump` document, in id order."""
    found: list[tuple[int, str, dict[str, Any]]] = []
    for obj in dump:
        if not isinstance(obj, dict) or obj.get("type") != "PipeWire:Interface:Node":
            continue
        props = _node_props(obj)
        if props.get("media.class") != PLAYBACK_CLASS:
            continue
        try:
            node_id = int(obj["id"])
        except (KeyError, TypeError, ValueError):
            continue
        found.append((node_id, _node_label(props), props))
    return sorted(found)


def audio_dump() -> list[Any]:
    """`pw-dump` as parsed JSON. Raises EarsError if the tool is missing or unhappy."""
    try:
        proc = subprocess.run(["pw-dump"], capture_output=True, timeout=CAPTURE_TIMEOUT_S,
                              check=False)
    except FileNotFoundError as exc:
        raise EarsError("pw-dump is not installed; PipeWire tools are required") from exc
    except subprocess.TimeoutExpired as exc:
        raise EarsError("pw-dump timed out") from exc
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()[:200]
        raise EarsError(f"pw-dump failed: {detail or proc.returncode}")
    try:
        parsed = json.loads(proc.stdout.decode("utf-8", "replace"))
    except json.JSONDecodeError as exc:
        raise EarsError(f"pw-dump did not return json: {exc}") from exc
    if not isinstance(parsed, list):
        raise EarsError("pw-dump returned no object list")
    return parsed


# --------------------------------------------------------------------------- #
# PCM helpers
# --------------------------------------------------------------------------- #


def level(pcm: bytes) -> tuple[int, float]:
    """Peak and RMS of s16le mono PCM."""
    if len(pcm) < 2:
        return 0, 0.0
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return 0, 0.0
    peak = max(max(samples), -min(samples))
    total = 0.0
    for value in samples:
        total += float(value) * float(value)
    return int(peak), math.sqrt(total / len(samples))


def wav_bytes(pcm: bytes, rate: int = RATE, channels: int = 1) -> bytes:
    """Wrap raw s16le PCM in a WAV container the model accepts."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm)
    return buf.getvalue()


def wav_pcm(data: bytes) -> tuple[bytes, int, int]:
    """Read a WAV back into (pcm, rate, channels)."""
    with wave.open(io.BytesIO(data), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        frames = handle.getnframes()
        return handle.readframes(frames), rate, channels


def pcm_offset(head: bytes) -> int:
    """Where the samples start in a WAV stream `parecord` is still writing.

    The writer leaves the `data` chunk length as a placeholder, so the header is found by
    searching for the chunk id rather than trusting its sizes.
    """
    marker = head.find(b"data")
    if marker < 0:
        return -1
    return marker + 8


class TailReader:
    """Reads a WAV file while `parecord` is still appending to it.

    `parecord` writes a WAV header and then samples; there is no portable way to make it
    stream to a pipe here, so the file is tailed instead. Header discovery is deferred
    until enough bytes exist, and a file with no `data` chunk is treated as raw PCM.
    """

    def __init__(self, path: pathlib.Path, wait_s: float = 0.5):
        self.path = path
        self.wait_s = wait_s
        self.offset = 0
        self.started = False

    def _locate(self, size: int) -> bool:
        if self.started:
            return True
        try:
            head = self.path.read_bytes()[:4096]
        except OSError:
            return False
        found = pcm_offset(head)
        if found >= 0:
            self.offset = found
            self.started = True
            return True
        if size >= 4096:
            self.offset = 0
            self.started = True
            return True
        return False

    def read(self, size: int) -> bytes:
        deadline = time.monotonic() + self.wait_s
        while True:
            try:
                total = self.path.stat().st_size
            except OSError:
                total = 0
            if self._locate(total) and total > self.offset:
                try:
                    with self.path.open("rb") as handle:
                        handle.seek(self.offset)
                        chunk = handle.read(min(size, total - self.offset))
                except OSError:
                    chunk = b""
                if chunk:
                    self.offset += len(chunk)
                    return chunk
            if time.monotonic() >= deadline:
                return b""
            time.sleep(0.01)


# --------------------------------------------------------------------------- #
# A ring: one recorder for the whole run
# --------------------------------------------------------------------------- #


class RingReader:
    """Reads a growing raw PCM file from a byte offset, whole blocks at a time.

    `TailReader` tails a WAV from the end; this reads a *raw* file from anywhere, which is what
    makes the ring useful: the same file can be replayed from the start (the greeting that was
    spoken while the body was resetting) and read live from the end, and both go through the
    same gate in `listen`.

    Only whole blocks come back, so RMS blocks never straddle a read boundary and a replay's
    block boundaries line up with the live capture's.
    """

    def __init__(self, path: pathlib.Path, start: int = 0, block: int = 0):
        self.path = path
        self.offset = start
        self.block = block

    def available(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    def read(self, size: int) -> bytes:
        total = self.available()
        usable = total - (total % self.block) if self.block else total
        if usable <= self.offset:
            return b""
        want = min(size, usable - self.offset)
        try:
            with self.path.open("rb") as handle:
                handle.seek(self.offset)
                chunk = handle.read(want)
        except OSError:
            return b""
        self.offset += len(chunk)
        return chunk


class Ring:
    """One `parecord` for the whole run, readable backwards and forwards.

    MEASURED 2026-11-13, run 181738-ee1d6d: the recording's own continuous capture held 12.8 s
    of JobBot's greeting and 14.9 s of his next instruction, and the brain heard one line of
    it. The greeting played while the body was resetting; the instruction played during a
    generated reach. `listen` records only while it is listening and `confirm` opens its window
    after the act, so both were deaf windows, and no gate tuning can recover speech that was
    never recorded.

    So the run records once, continuously, and every question is asked of that one file:
    `listen(start=0)` replays from the beginning, `listen()` reads live, and `read_from(mark)`
    hands the act's own seconds to `confirm`.
    """

    def __init__(
        self,
        path: pathlib.Path | str,
        pattern: str = DEFAULT_PATTERN,
        sink: str = SINK_NAME,
        rate: int = RATE,
        device: str = "",
    ) -> None:
        self.path = pathlib.Path(path)
        self.pattern = pattern
        self.sink = sink
        self.rate = rate
        self.device = device or capture_device(pattern, sink)
        self.block = int(BLOCK_S * rate) * 2
        self._proc: subprocess.Popen[bytes] | None = None
        self.error = ""
        #: Where the last `listen` stopped reading. The next one resumes here, so a turn never
        #: replays the utterance it just heard.
        self.cursor = 0

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> bool:
        """Begin recording. False means the run has no ring and must fall back per window."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        args = [
            PARECORD,
            f"--device={self.device}",
            f"--rate={self.rate}",
            "--channels=1",
            "--format=s16le",
            "--no-remix",
            "--file-format=raw",
            str(self.path),
        ]
        try:
            self._proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        except FileNotFoundError:
            self.error = "parecord is not installed"
            return False
        except OSError as exc:
            self.error = str(exc)
            return False
        return True

    def stop(self) -> str:
        if self._proc is None:
            return ""
        detail = _stop(self._proc)
        self._proc = None
        return detail

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # -- reading ------------------------------------------------------------ #

    def bytes_available(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    def available_s(self) -> float:
        return self.bytes_available() / (self.rate * 2)

    def mark(self) -> int:
        """A block-aligned offset at the current end: where "now" is in the ring."""
        total = self.bytes_available()
        return total - (total % self.block)

    def reader(self, start: int | None = None) -> RingReader:
        return RingReader(self.path, self.mark() if start is None else start, self.block)

    def read_from(self, start: int, max_s: float = 0.0) -> bytes:
        """Bytes from `start` to now, optionally capped. No waiting: this reads the past."""
        total = self.bytes_available()
        if total <= start:
            return b""
        want = total - start
        if max_s > 0:
            want = min(want, int(max_s * self.rate) * 2)
        try:
            with self.path.open("rb") as handle:
                handle.seek(start)
                return handle.read(want)
        except OSError:
            return b""

    def read_back(self, seconds: float) -> bytes:
        start = max(0, self.mark() - int(seconds * self.rate) * 2)
        return self.read_from(start)

    def listen(self, start: int | None = None, **kwargs: Any) -> Hearing:
        """`listen` over the ring, and leave `cursor` where it stopped.

        `start=0` replays the run from its beginning: that is how the greeting spoken while the
        body was resetting still arrives, through the same gate, a minute late and unhurried.
        """
        kwargs.setdefault("rate", self.rate)
        reader = self.reader(self.cursor if start is None else start)
        hearing = listen(reader=reader, **kwargs)
        self.cursor = reader.offset
        return hearing


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #


def _spawn(device: str, rate: int, path: pathlib.Path) -> subprocess.Popen[bytes]:
    args = [
        PARECORD,
        f"--device={device}",
        f"--rate={rate}",
        "--channels=1",
        "--format=s16le",
        "--no-remix",
        str(path),
    ]
    try:
        return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise EarsError("parecord is not installed; the PulseAudio tools are required") from exc


def _stop(proc: subprocess.Popen[bytes]) -> str:
    detail = ""
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
    try:
        if proc.stderr is not None:
            detail = proc.stderr.read().decode("utf-8", "replace").strip()[:200]
    except (OSError, ValueError):
        detail = ""
    return detail


@contextlib.contextmanager
def capture(
    pattern: str = DEFAULT_PATTERN,
    sink: str = SINK_NAME,
    rate: int = RATE,
    device: str = "",
    route_first: bool = True,
) -> Any:
    """Yield a reader over the game's audio, and clean up the recorder afterwards."""
    info: dict[str, Any] = {"device": device}
    if not device:
        if route_first:
            info = route(pattern, sink)
        device = capture_device(pattern, sink)
        info.setdefault("device", device)
    # One file per capture: a shared name lets two recorders (a listener and a probe)
    # truncate each other's file, which reads as "no audio" with everything working.
    state = _state_dir()
    state.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="capture-", suffix=".wav", dir=state)
    os.close(fd)
    path = pathlib.Path(name)
    proc = _spawn(device, rate, path)
    reader = TailReader(path)
    try:
        yield reader, proc, info
    finally:
        detail = _stop(proc)
        with contextlib.suppress(OSError):
            path.unlink()
        if detail and proc.poll() not in (0, None):
            info["stderr"] = detail


def record(
    seconds: float,
    pattern: str = DEFAULT_PATTERN,
    sink: str = SINK_NAME,
    rate: int = RATE,
    device: str = "",
) -> Hearing:
    """Fixed-length capture, no gate. For tests and for grabbing a known cue."""
    chunks: list[bytes] = []
    error = ""
    tolerance = max(seconds, 1.0) + STARTUP_GRACE_S
    deadline = time.monotonic() + tolerance
    with capture(pattern, sink, rate, device) as (reader, proc, yield_info):
        want = int(seconds * rate) * 2
        got = 0
        while got < want and time.monotonic() < deadline:
            block = reader.read(min(int(BLOCK_S * rate) * 2, want - got))
            if not block:
                if proc.poll() is not None:
                    break
                continue
            chunks.append(block)
            got += len(block)
    if not chunks:
        error = "no audio received"
    if yield_info.get("stderr"):
        error = error or str(yield_info["stderr"])
    pcm = b"".join(chunks)[: int(seconds * rate) * 2]
    peak, rms = level(pcm)
    seconds_got = len(pcm) / (rate * 2)
    return Hearing(
        wav=wav_bytes(pcm, rate) if pcm else b"",
        seconds=seconds_got,
        speech_s=seconds_got,
        peak=peak,
        rms=rms,
        device=str(yield_info.get("device", "")),
        error=error,
    )


def listen(
    pattern: str = DEFAULT_PATTERN,
    sink: str = SINK_NAME,
    rate: int = RATE,
    max_s: float = 25.0,
    pre_roll_s: float = 0.45,
    floor_s: float = 0.5,
    hang_s: float = 2.0,
    tail_s: float = 0.2,
    gate_ratio: float = 4.0,
    gate_min: float = GATE_MIN_RMS,
    device: str = "",
    reader: Any = None,
) -> Hearing:
    """Record one utterance from the game, gated on speech.

    `hang_s` is the quiet that ends an utterance. MEASURED 2026-11-13 on the audio of run
    190131-679de2: the pause *inside* JobBot's opening is 0.60 s ("...fine job museum." to
    "Here's my card."), while the gap between two separate turns is 6.3-6.9 s (the opening to
    his nag, the card line to the cartridge line). 2.0 s sits between the two with a 3x margin
    on the turn gap. Note that 0.60 s is *below* the old 0.8 s, so a split at 0.8 s means the
    capture dropped out rather than that the pause was long - `Hearing.ended_by` now says which.

    `reader` replaces the capture entirely and is how the tests drive the gate without an
    audio device.
    """
    block = int(BLOCK_S * rate) * 2
    pre: collections.deque[bytes] = collections.deque(maxlen=max(1, int(pre_roll_s / BLOCK_S)))
    keep: list[bytes] = []
    floor_samples: list[float] = []
    floor = 0.0
    gate = gate_min
    floor_known = False
    speech_blocks = 0
    quiet_blocks = 0
    started = False
    ended_by = "deadline"
    error = ""
    info: dict[str, Any] = {}
    manager = None
    proc: subprocess.Popen[bytes] | None = None
    if reader is not None:
        source = reader
    else:
        manager = capture(pattern, sink, rate, device)
        source, proc, info = manager.__enter__()

    deadline = time.monotonic() + max_s
    last_data = time.monotonic()
    seen_audio = False
    try:
        while time.monotonic() < deadline:
            chunk = source.read(block)
            if chunk:
                last_data = time.monotonic()
                seen_audio = True
            if not chunk:
                if proc is not None and proc.poll() is not None:
                    ended_by = "eof"
                    if not keep:
                        error = "capture ended"
                    break
                grace = EOF_GRACE_S if seen_audio else STARTUP_GRACE_S
                if time.monotonic() - last_data > grace:
                    # Nothing for a second from a capture that has not ended: it is not
                    # coming. Without this the gate spins to `max_s` on a dead reader. A
                    # stream that ends after speech began is not an error, it is the end
                    # of the utterance.
                    ended_by = "eof"
                    if not keep:
                        error = "capture went quiet"
                    break
                continue
            _, rms = level(chunk)
            if not started:
                pre.append(chunk)
                # Every block before the gate opens, kept and never cleared. The floor is the
                # 20th percentile of all of them, so a loud opening - the game already
                # talking, or a bang - cannot set the floor for the rest of the window and
                # gate the utterance out. A median of the first half second did exactly that.
                floor_samples.append(rms)
                if len(floor_samples) * BLOCK_S >= floor_s:
                    ordered = sorted(floor_samples)
                    floor = ordered[max(0, len(ordered) // 5)]
                    gate = max(gate_ratio * floor, gate_min)
                    floor_known = True
                # `floor_known`, not `floor > 0`: the game is digitally silent between
                # lines, so a floor of exactly zero is the normal case and gating on it
                # would make it deaf.
                if floor_known and rms > gate:
                    started = True
                    keep.extend(pre)
                    pre.clear()
                    speech_blocks = 1
                continue
            keep.append(chunk)
            if rms > gate:
                speech_blocks += 1
                quiet_blocks = 0
            else:
                quiet_blocks += 1
                if quiet_blocks * BLOCK_S >= hang_s:
                    ended_by = "hang"
                    break
    finally:
        if manager is not None:
            manager.__exit__(None, None, None)
        if info.get("stderr") and not error:
            error = str(info["stderr"])

    if not started:
        reason = "no speech heard"
        if error:
            reason = f"{reason} ({error})"
        return Hearing(node_name="", device=str(info.get("device", "")),
                       floor=floor, gate=gate, ended_by="never-started", error=reason)

    # Trim the trailing quiet, but only when there really is trailing quiet. A `hang` ending
    # means `hang_s` of silence was observed, so `hang_s - tail_s` of it can go. A `deadline` or
    # `eof` ending has no quiet to trim - the line was cut mid-speech - and removing
    # `hang_s - tail_s` there deletes 1.8 s of what the game actually said. At hang_s = 0.8 the
    # error was 0.6 s and invisible; at 2.0 it is most of a sentence.
    drop_s = max(0.0, hang_s - tail_s) if ended_by == "hang" else 0.0
    drop = int(drop_s / BLOCK_S) * block
    pcm = b"".join(keep)
    if drop and len(pcm) > drop + block * 4:
        pcm = pcm[:-drop]
    peak, rms = level(pcm)
    seconds = len(pcm) / (rate * 2)
    return Hearing(
        wav=wav_bytes(pcm, rate),
        seconds=seconds,
        speech_s=min(seconds, speech_blocks * BLOCK_S),
        peak=peak,
        rms=rms,
        device=str(info.get("device", "")),
        floor=floor,
        gate=gate,
        ended_by=ended_by,
        error=error,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Record the game's own audio.")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN,
                    help="substring of the game's application name")
    ap.add_argument("--sink", default=SINK_NAME, help="capture sink to route the game into")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="fixed length instead of waiting for speech")
    ap.add_argument("--max-s", type=float, default=25.0, dest="max_s")
    ap.add_argument("--list", action="store_true",
                    help="print sinks, game streams and the capture device")
    ap.add_argument("--route-only", action="store_true",
                    help="put the game behind the capture sink and exit")
    ap.add_argument("--no-route", action="store_true",
                    help="record without moving the game's stream first")
    ap.add_argument("--save", default="", help="write the capture here")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--transcribe", action="store_true", help="send it to the model as well")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        try:
            for name in sink_names():
                print(f"sink  {name}")
            for stream in game_streams(args.pattern):
                props = stream.get("properties") or {}
                print(f"game  {stream.get('index')} {props.get('application.name')} "
                      f"-> sink {stream.get('sink')}")
            print(f"device {capture_device(args.pattern, args.sink)}")
        except EarsError as exc:
            print(f"ears: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.route_only:
        try:
            print(json.dumps(route(args.pattern, args.sink)) if args.json
                  else f"ears: routed ({route(args.pattern, args.sink)})")
        except EarsError as exc:
            print(f"ears: {exc}", file=sys.stderr)
            return 2
        return 0

    kwargs = {"pattern": args.pattern, "sink": args.sink}
    if args.no_route:
        kwargs["device"] = f"{args.sink}.monitor"
    try:
        hearing = (record(args.seconds, **kwargs) if args.seconds > 0
                   else listen(max_s=args.max_s, **kwargs))
    except EarsError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}))
        else:
            print(f"ears: {exc}", file=sys.stderr)
        return 2

    saved = ""
    if not args.no_save and hearing.wav:
        target = pathlib.Path(args.save) if args.save else _state_dir() / "last.wav"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(hearing.wav)
        saved = str(target)

    if args.json:
        print(json.dumps({**hearing.to_dict(), "saved": saved}))
    else:
        state = "heard" if hearing.ok else "nothing"
        print(f"ears: {state} {hearing.seconds:.2f}s from {hearing.device or 'default'} "
              f"peak {hearing.peak} rms {hearing.rms:.0f} "
              f"(floor {hearing.floor:.0f} gate {hearing.gate:.0f})")
        if hearing.error:
            print(f"ears: {hearing.error}", file=sys.stderr)
        if saved:
            print(f"ears: saved {saved}")

    if args.transcribe and hearing.wav:
        from . import brain

        result = brain.transcribe(brain.BrainConfig(), hearing.wav)
        if args.json:
            print(json.dumps({"transcript": result.to_dict()}))
        else:
            print(f"heard: {result.text or '(' + (result.error or 'nothing') + ')'} "
                  f"[{result.decode_ms:.0f} ms]")
    return 0 if hearing.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
