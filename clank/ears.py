"""Hear the game, and nothing else.

The game is a PipeWire client like any other, so its output is a node we can record
directly: `Stream/Output/Audio` with `application.name = JobSimulator.exe`. Recording that
node captures the game and only the game - a notification, a video on the desktop or the
operator talking on the mic never reaches it. Measured 2026-11-08: `pw-dump` lists the game
as node 161 while a session runs, and `pw-record --target 161 --raw ... -` streams its PCM.

The node id changes on every game start, so it is resolved per capture by matching the
application name in `pw-dump`, never hardcoded.

Capture is gated on speech, because the game is silent between lines and a fixed-length
recording is nine parts silence. Blocks of 50 ms are scored by RMS against a noise floor
taken from the first half second; when a block crosses it the pre-roll is kept, and the
recording ends after `hang_s` of quiet. What comes back is one utterance, ready for the
model.
"""

from __future__ import annotations

import array
import collections
import contextlib
import io
import json
import math
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
#: `application.name` of the game's playback stream, matched case-insensitively. Steam and
#: Proton change the process, but the audio client keeps the executable's name.
DEFAULT_PATTERN = "JobSimulator"
#: The node class that carries an application's output. A microphone is `Audio/Source` and
#: must never match.
PLAYBACK_CLASS = "Stream/Output/Audio"
BLOCK_S = 0.05
#: Legal-looking speech gate for a quiet mix: four times the floor, never below this.
GATE_MIN_RMS = 250.0


class EarsError(RuntimeError):
    """No node, no pw-record, or the stream died before it produced audio."""


@dataclass(frozen=True)
class Hearing:
    """One captured utterance and the numbers that say whether it was one."""

    wav: bytes = b""
    seconds: float = 0.0
    speech_s: float = 0.0
    peak: int = 0
    rms: float = 0.0
    node: int = 0
    node_name: str = ""
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
            "floor": round(self.floor, 1),
            "gate": round(self.gate, 1),
            "bytes": len(self.wav),
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# Which node is the game
# --------------------------------------------------------------------------- #


def _node_props(obj: dict[str, Any]) -> dict[str, Any]:
    info = obj.get("info")
    if not isinstance(info, dict):
        return {}
    props = info.get("props")
    return props if isinstance(props, dict) else {}


def _node_label(props: dict[str, Any]) -> str:
    for key in ("application.name", "node.name", "application.process.binary", "node.description"):
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


def pick_node(
    dump: Iterable[Any],
    pattern: str = DEFAULT_PATTERN,
) -> tuple[int, str] | None:
    """The playback node whose application name contains `pattern`.

    Case-insensitive substring, because the observable name is
    `JobSimulator.exe` and nobody should have to spell the extension. Returns None rather
    than guessing: recording the wrong client is worse than not recording.
    """
    needle = pattern.strip().lower()
    if not needle:
        return None
    for node_id, label, props in playback_nodes(dump):
        haystack = " ".join(
            str(props.get(key, "")) for key in (
                "application.name",
                "node.name",
                "application.process.binary",
                "node.description",
            )
        ).lower()
        if needle in haystack:
            return node_id, label
    return None


def audio_dump() -> list[Any]:
    """`pw-dump` as parsed JSON. Raises EarsError if the tool is missing or unhappy."""
    try:
        proc = subprocess.run(
            ["pw-dump"], capture_output=True, timeout=10.0, check=False
        )
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


def game_node(pattern: str = DEFAULT_PATTERN) -> tuple[int, str]:
    """Resolve the game's playback node, or say what was playing instead."""
    node = pick_node(audio_dump(), pattern)
    if node is None:
        seen = ", ".join(label for _, label, _ in playback_nodes(audio_dump())) or "nothing"
        raise EarsError(f"no playback stream matching {pattern!r} (playing: {seen})")
    return node


# --------------------------------------------------------------------------- #
# Recording
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
    """Read a WAV back into (pcm, rate, channels). Used by tests and by callers that
    only have a file on disk."""
    with wave.open(io.BytesIO(data), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        frames = handle.getnframes()
        return handle.readframes(frames), rate, channels


def _spawn(node_id: int, rate: int) -> subprocess.Popen[bytes]:
    args = [
        "pw-record", "--raw", "-",
        "--target", str(node_id),
        "--rate", str(rate),
        "--channels", "1",
        "--format", "s16",
    ]
    try:
        return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise EarsError("pw-record is not installed; PipeWire tools are required") from exc


def _stop(proc: subprocess.Popen[bytes]) -> str:
    detail = ""
    try:
        if proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=2.0)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
    try:
        if proc.stderr is not None:
            detail = proc.stderr.read().decode("utf-8", "replace").strip()[:200]
    except (OSError, ValueError):
        detail = ""
    return detail


def record(
    seconds: float,
    node: int | None = None,
    pattern: str = DEFAULT_PATTERN,
    rate: int = RATE,
) -> Hearing:
    """Fixed-length capture, no gate. For tests and for grabbing a known cue."""
    node_id, name = (node, "") if node is not None else game_node(pattern)
    proc = _spawn(node_id, rate)
    want = int(seconds * rate) * 2
    chunks: list[bytes] = []
    try:
        if proc.stdout is None:
            raise EarsError("pw-record produced no stdout")
        while sum(len(c) for c in chunks) < want:
            block = proc.stdout.read(int(BLOCK_S * rate) * 2)
            if not block:
                break
            chunks.append(block)
    finally:
        detail = _stop(proc)
    pcm = b"".join(chunks)[:want]
    peak, rms = level(pcm)
    if not pcm:
        return Hearing(node=node_id, node_name=name, error=detail or "no audio received")
    return Hearing(
        wav=wav_bytes(pcm, rate),
        seconds=len(pcm) / (rate * 2),
        speech_s=len(pcm) / (rate * 2),
        peak=peak,
        rms=rms,
        node=node_id,
        node_name=name,
    )


def listen(
    node: int | None = None,
    pattern: str = DEFAULT_PATTERN,
    rate: int = RATE,
    max_s: float = 25.0,
    pre_roll_s: float = 0.45,
    floor_s: float = 0.5,
    hang_s: float = 0.8,
    tail_s: float = 0.2,
    gate_ratio: float = 4.0,
    gate_min: float = GATE_MIN_RMS,
    reader: Any = None,
) -> Hearing:
    """Record one utterance from the game, gated on speech.

    `reader` replaces the pw-record stream with any object that has `.read(n)` and is used
    by the tests; production passes None and the game's node is recorded.
    """
    proc = None
    node_id, name = 0, ""
    if reader is None:
        node_id, name = (node, "") if node is not None else game_node(pattern)
        proc = _spawn(node_id, rate)
        source = proc.stdout
        if source is None:
            raise EarsError("pw-record produced no stdout")
    else:
        source = reader

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
    deadline = time.monotonic() + max_s
    try:
        while time.monotonic() < deadline:
            chunk = source.read(block)
            if not chunk:
                error = "stream ended" if not keep else ""
                break
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
        if proc is not None:
            detail = _stop(proc)
            if detail and not error:
                error = detail

    if not started:
        reason = "no speech heard"
        if error:
            reason = f"{reason} ({error})"
        return Hearing(node=node_id, node_name=name, floor=floor, gate=gate, error=reason)

    # Trim the hang that ended the recording, keeping a little of it: cutting at the last
    # loud block clips the final consonant, and the model hears a truncated word.
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
        node=node_id,
        node_name=name,
        floor=floor,
        gate=gate,
        error=error,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _default_state() -> pathlib.Path:
    base = pathlib.Path.home() / ".local" / "state" / "Clank"
    return base / "ears"


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Record the game's own audio.")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN,
                    help="substring of the game's PipeWire application name")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="fixed length instead of waiting for speech")
    ap.add_argument("--max-s", type=float, default=25.0, dest="max_s")
    ap.add_argument("--node", type=int, default=0, help="use this node id instead of matching")
    ap.add_argument("--list", action="store_true", help="print candidate playback nodes")
    ap.add_argument("--save", default="", help="write the capture here (default: state dir)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--transcribe", action="store_true", help="send it to the model as well")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        for node_id, label, _props in playback_nodes(audio_dump()):
            print(f"{node_id:6d}  {label}")
        return 0

    node = args.node or None
    try:
        hearing = (
            record(args.seconds, node, args.pattern)
            if args.seconds > 0
            else listen(node, args.pattern, max_s=args.max_s)
        )
    except EarsError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}))
        else:
            print(f"ears: {exc}", file=sys.stderr)
        return 2

    saved = ""
    if not args.no_save and hearing.wav:
        target = pathlib.Path(args.save) if args.save else _default_state() / "last.wav"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(hearing.wav)
        saved = str(target)

    if args.json:
        print(json.dumps({**hearing.to_dict(), "saved": saved}))
    else:
        state = "heard" if hearing.ok else "nothing"
        print(f"ears: {state} {hearing.seconds:.2f}s from {hearing.node_name or '?'} "
              f"node {hearing.node} peak {hearing.peak} rms {hearing.rms:.0f} "
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
