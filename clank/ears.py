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

Recording is gated on speech: 50 ms RMS blocks, floor from the first half second, and the
utterance ends after `hang_s` of quiet. Digital silence between lines is the normal state of
this game, so the gate keys on the floor being *known*, not on it being non-zero.
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
#: How long a capture may produce nothing before it is treated as gone.
EOF_GRACE_S = 1.0


class EarsError(RuntimeError):
    """No game stream, no tool, or the capture died before it produced audio."""


@dataclass(frozen=True)
class Hearing:
    """One captured utterance and the numbers that say whether it was one.

    `speech_s` is the time the gate called speech. A fixed-length `record()` has no gate,
    so it equals `seconds` there.
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
    return {"sink": sink, "created": created, "loopback": looped,
            "moved": moved, "device": capture_device(pattern, sink)}


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
    path = _state_dir() / "capture.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
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
    with capture(pattern, sink, rate, device) as (reader, proc, yield_info):
        want = int(seconds * rate) * 2
        got = 0
        while got < want:
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
    hang_s: float = 0.8,
    tail_s: float = 0.2,
    gate_ratio: float = 4.0,
    gate_min: float = GATE_MIN_RMS,
    device: str = "",
    reader: Any = None,
) -> Hearing:
    """Record one utterance from the game, gated on speech.

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
    try:
        while time.monotonic() < deadline:
            chunk = source.read(block)
            if chunk:
                last_data = time.monotonic()
            if not chunk:
                if proc is not None and proc.poll() is not None:
                    if not keep:
                        error = "capture ended"
                    break
                if time.monotonic() - last_data > EOF_GRACE_S:
                    # Nothing for a second from a capture that has not ended: it is not
                    # coming. Without this the gate spins to `max_s` on a dead reader. A
                    # stream that ends after speech began is not an error, it is the end
                    # of the utterance.
                    if not keep:
                        error = "capture went quiet"
                    break
                continue
            _, rms = level(chunk)
            if not started:
                pre.append(chunk)
                floor_samples.append(rms)
                if len(floor_samples) * BLOCK_S >= floor_s:
                    ordered = sorted(floor_samples)
                    floor = ordered[len(ordered) // 2]
                    gate = max(gate_ratio * floor, gate_min)
                    floor_known = True
                    del floor_samples[:]
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
                       floor=floor, gate=gate, error=reason)

    drop_s = max(0.0, hang_s - tail_s)
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
