#!/usr/bin/env python3
"""Feed the body's *achieved* poses to the game, replacing the scripted client.

The chain is:

    SONIC body_view (one sim) -> body_state.json -> this relay -> TCP 4242
        -> Monado `remote` driver -> Superhot

This is the join. `spike/c0/c0_client.py` sends scripted constants; this sends
whatever the simulated humanoid actually did, so what the body tile shows and
what the game sees are the same body rather than two lookalikes.

Stdlib only, on purpose: it imports neither mujoco nor numpy, so it runs under
the same interpreter as the dashboard and the test suite can cover it.

Staleness is explicit. If the state file stops advancing the relay keeps sending
the last pose and marks the CSV `stale`, because Monado freezes the far end on
disconnect and a frozen-but-connected client is easier to diagnose than a game
that silently stops responding. Before any state has ever arrived it sends the
measured standing pose, so the game sees a humanoid at 1.221 m, not the origin.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import struct
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:  # both shapes work: `-m clank.pose_relay` and a direct file run
    from . import player_scale as player_scale_file
    from . import wire
except ImportError:  # pragma: no cover - only when executed as a script
    import player_scale as player_scale_file  # type: ignore[no-redef]
    import wire  # type: ignore[no-redef]

#: How old body_state.json may be before the relay reports it stale, seconds.
STALE_AFTER_SEC = 0.30

#: Ways to force a button without the body cooperating, for the hub A/B.
FORCE_MODES = ("none", "left", "right", "both")

#: Which wire field a hand closure drives. The game decides which one it reads, so this
#: is a switch - but the default is now `both`, measured: Job Simulator grabs on squeeze
#: and *draws the fist* from `hand_curl`, so squeeze alone registers a grab with the
#: glove still open. `spike/c0/c0_manual.py:365-374`, which is the path where a click
#: visibly fisted the hand, sets squeeze_value, squeeze_force and all five hand_curl
#: slots together. Measured 2026-11-08: with `both` the glove closes into a fist;
#: with `squeeze` the open and closed frames are pixel-identical.
GRAB_MODES = ("squeeze", "curl", "both")

#: The driver closes the connection if it is not reading; a few seconds of retry
#: covers Monado still coming up when the session starts.
CONNECT_TIMEOUT_SEC = 20.0

#: How long one handshake attempt may wait for the driver's two reply packets.
#: The driver serves one client at a time, so a mode switch can connect while the
#: previous client still owns it; a short attempt keeps the retry responsive.
HANDSHAKE_TIMEOUT_SEC = 1.0


def forced_pair(mode: str) -> tuple[float, float]:
    """(left, right) for a CLI force mode. 'none' releases both."""
    return {
        "none": (0.0, 0.0),
        "left": (1.0, 0.0),
        "right": (0.0, 1.0),
        "both": (1.0, 1.0),
    }[mode]


def buttons_from_state(
    state: Mapping[str, Any],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Read (closure, trigger) out of a state dict, as (left, right).

    Tolerates a body that predates the fields, and a malformed pair: a bad button
    field releases the button rather than killing the pipe.
    """

    def pair(key: str) -> tuple[float, float]:
        raw = state.get(key)
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            return (0.0, 0.0)
        try:
            return (float(raw[0]), float(raw[1]))
        except (TypeError, ValueError):
            return (0.0, 0.0)

    return pair("closure"), pair("trigger")


class ClickEdge:
    """Turns a held trigger value into one click per press, per hand.

    The wire format carries a click as well as a value, and a click is an edge.
    Sending click=true on every packet while the trigger is down is what the old
    fill() did; Unity reads that as a fresh press every frame.

    `hold_packets` keeps the click high for a couple of packets. One packet is
    about 20 ms, and Monado's hub latches only the newest packet, so a single
    packet can be overwritten before a slow game frame ever samples it - measured
    at 1 click in 30 packets, which is exactly the intermittent fire that makes a
    null result unattributable.
    """

    def __init__(self, hold_packets: int = 2) -> None:
        self._prev: tuple[float, float] = (0.0, 0.0)
        self._left = [0, 0]
        self.hold_packets = max(hold_packets, 1)

    def update(self, values: Sequence[float]) -> tuple[bool, bool]:
        # Clamp rather than raise: this runs inside the 50 Hz loop, where losing a
        # packet beats dying mid-session.
        pair = (float(values[0]), float(values[1])) if len(values) >= 2 else (0.0, 0.0)
        clicks: list[bool] = []
        for i, value in enumerate(pair):
            if value > 0.0 and self._prev[i] <= 0.0:
                self._left[i] = self.hold_packets
            elif value <= 0.0:
                self._left[i] = 0
            clicks.append(self._left[i] > 0)
            if self._left[i] > 0:
                self._left[i] -= 1
        self._prev = pair
        return (clicks[0], clicks[1])


class StateSource:
    """Reads a BodyState JSON file, re-parsing only when its mtime changes."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._mtime_ns: int | None = None
        self._doc: dict[str, Any] | None = None
        self._seen_at: float | None = None

    def read(self, now: float) -> tuple[dict[str, Any] | None, float | None]:
        """Return (state, age_seconds); age is None when nothing was ever read."""
        try:
            mtime_ns = self.path.stat().st_mtime_ns
        except OSError:
            return self._doc, (now - self._seen_at if self._seen_at is not None else None)
        if mtime_ns != self._mtime_ns:
            try:
                self._doc = json.loads(self.path.read_text(encoding="utf-8"))
                self._mtime_ns = mtime_ns
                self._seen_at = now
            except (OSError, ValueError):
                pass
        age = now - self._seen_at if self._seen_at is not None else None
        return self._doc, age


#: A file the brain can write to re-aim the robot in the virtual room. The body cannot be
#: turned - a commanded yaw rate measured +0.2 deg where 69 was asked - so the heading is
#: the only channel for facing something, and it has to be settable while the session runs.
#: The orchestrator's `--yaw-offset-deg` stays the default; this overrides it when present.
DEFAULT_YAW_FILE = "~/.local/state/Clank/relay-yaw.json"


class YawFile:
    """Re-reads the heading whenever the file changes. Silent when the file is absent."""

    def __init__(self, path: str | os.PathLike[str] | None) -> None:
        self.path = Path(os.path.expanduser(path)) if path else None
        self._seen: tuple[int, int] | None = None
        self.rad: float | None = None

    def poll(self) -> float | None:
        """The heading in radians, or None when the file says nothing useful."""
        if self.path is None:
            return None
        try:
            stamp = self.path.stat()
        except OSError:
            return None
        key = (stamp.st_mtime_ns, stamp.st_size)
        if key == self._seen:
            return self.rad
        self._seen = key
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return self.rad
        deg = raw.get("yaw_offset_deg") if isinstance(raw, dict) else None
        if not isinstance(deg, (int, float, str)) or isinstance(deg, bool):
            return self.rad
        try:
            value = float(deg)
        except (TypeError, ValueError):
            return self.rad
        if not math.isfinite(value):
            return self.rad
        self.rad = math.radians(value)
        return self.rad


#: How long a requested press stays down. Long enough for the game to sample it at 60-90 Hz
#: from a relay sending ~50 Hz packets, short enough that it reads as a click and not a hold.
CLICK_HOLD_S = 0.15

#: Where a press request lives. Someone else writes it - the dashboard button, or the brain
#: choosing the `menu` skill - and the relay turns it into packets.
DEFAULT_CLICK_FILE = "~/.local/state/Clank/relay-click.json"


def request_click(hand: str = "right", path: str | os.PathLike[str] | None = None,
                  hold_s: float = CLICK_HOLD_S) -> dict:
    """Ask the relay to press the controller's menu button once.

    The menu button is on the controller, not on the body: no joint of the simulated G1 can press
    it, and JobBot asks for it early ("hit the menu button"). So a press is a request in a file
    that the relay polls, and the relay holds the bit down for `hold_s` so a 60-90 Hz game sees it.
    """
    target = Path(os.path.expanduser(path or DEFAULT_CLICK_FILE))
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seq": time.time_ns(),
        "hand": hand if hand in ("left", "right") else "right",
        "hold_s": float(hold_s),
    }
    raw = json.dumps(payload).encode("utf-8")
    tmp = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, target)
    return payload


class ClickFile:
    """Turns a new press request into a short click the game can see.

    It is deliberately not a level: a file left behind must not hold the menu button down, or the
    game would sit in a menu nobody can leave. Only the change of a request's sequence number
    starts a new press, so the relay can poll as fast as it writes packets.
    """

    def __init__(self, path: str | os.PathLike[str] | None,
                 hold_s: float = CLICK_HOLD_S) -> None:
        self.path = Path(os.path.expanduser(path)) if path else None
        self.hold_s = hold_s
        self._seen: int | None = None
        self._until = 0.0
        self._hand = "right"

    def pressed(self, now: float) -> tuple[bool, bool]:
        """(left, right) for this packet."""
        if self.path is not None:
            try:
                raw = json.loads(self.path.read_text())
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                raw = None
            if isinstance(raw, dict):
                seq = raw.get("seq")
                if isinstance(seq, int) and not isinstance(seq, bool) and seq != self._seen:
                    self._seen = seq
                    hand = raw.get("hand")
                    self._hand = hand if hand in ("left", "right") else "right"
                    hold = raw.get("hold_s")
                    hold = float(hold) if isinstance(hold, (int, float)) and not isinstance(
                        hold, bool) else self.hold_s
                    self._until = now + (hold if hold > 0 else self.hold_s)
        if now < self._until:
            return (self._hand == "left", self._hand == "right")
        return (False, False)


class PacketSource:
    """Turns body states into XR poses, holding through gaps."""

    def __init__(
        self,
        source: StateSource,
        stale_after: float = STALE_AFTER_SEC,
        yaw_offset_rad: float = 0.0,
        player_scale: float = 1.0,
        player_pivot: tuple[float, float, float] = (0.0, 0.0, 0.0),
        player_scale_head: bool = True,
        yaw_file: str | os.PathLike[str] | None = None,
    ) -> None:
        self.source = source
        self.yaw_file = YawFile(yaw_file)
        self._yaw_note = 0.0
        self.stale_after = stale_after
        # The robot's heading in the virtual room. Rejected here rather than
        # silently producing NaN poses downstream: a nan position would take the
        # compositor somewhere undefined and look like a body fault.
        if not math.isfinite(yaw_offset_rad):
            raise ValueError("yaw_offset_rad must be finite")
        self.yaw_offset_rad = yaw_offset_rad
        # How big the emulated player is against the room. Rejected here for the
        # same reason as the yaw: a zero or negative scale collapses or mirrors
        # every pose, and the symptom downstream would look like lost tracking
        # rather than the argument error it is.
        if not math.isfinite(player_scale) or player_scale <= 0.0:
            raise ValueError("player_scale must be finite and positive")
        self.player_scale = player_scale
        self.player_pivot = tuple(float(c) for c in player_pivot)
        self.player_scale_head = bool(player_scale_head)
        self._last = wire.REST_POSES
        self._last_note = "rest"
        #: The most recent state that parsed, so buttons can be forwarded even
        #: when the pose is held or stale. A dead body must not release a grab.
        self.last_state: dict[str, Any] | None = None

    def poses(self, now: float) -> tuple[dict[str, Any], str, float | None]:
        state, age = self.source.read(now)
        if state is not None:
            self.last_state = state
        if state is None:
            return self._last, self._last_note if self._last_note != "live" else "stale", age
        live_yaw = self.yaw_file.poll()
        if live_yaw is not None and live_yaw != self.yaw_offset_rad:
            self.yaw_offset_rad = live_yaw
            if abs(live_yaw - self._yaw_note) > math.radians(0.5):
                self._yaw_note = live_yaw
                print(f"pose_relay: heading now {math.degrees(live_yaw):+.1f} deg "
                      f"(set by {self.yaw_file.path})", flush=True)
        try:
            poses = wire.body_state_to_xr(
                state,
                yaw_offset_rad=self.yaw_offset_rad,
                scale=self.player_scale,
                pivot=self.player_pivot,
                scale_head=self.player_scale_head,
            )
        except (KeyError, TypeError, ValueError):
            # A half-written or shape-changed file must not kill the pipe.
            return self._last, "malformed", age
        if age is not None and age > self.stale_after:
            self._last_note = "stale"
            return self._last, "stale", age
        self._last = poses
        self._last_note = "live"
        return poses, "live", age


def read_handshake(sock: socket.socket, size: int = wire.SIZE) -> None:
    """The driver replies with two packets on connect; both must carry our header."""
    for name in ("reset", "latest"):
        buf = bytearray()
        while len(buf) < size:
            chunk = sock.recv(size - len(buf))
            if not chunk:
                raise ConnectionError(f"peer closed during the {name} handshake")
            buf.extend(chunk)
        header = struct.unpack_from("<Q", bytes(buf))[0]
        if header != wire.HEADER:
            raise ValueError(f"handshake {name} header 0x{header:016x} != expected")


def connect(
    host: str,
    port: int,
    timeout: float = CONNECT_TIMEOUT_SEC,
    handshake_timeout: float = HANDSHAKE_TIMEOUT_SEC,
) -> socket.socket:
    """Connect and complete the handshake, retrying the whole attempt.

    The remote driver serves one client at a time. A mode switch can therefore
    accept our TCP connection while the previous client still owns the driver,
    and no handshake arrives until that client is gone. Retrying (rather than
    failing the session on the first quiet socket) lets the switch wait it out.
    """
    deadline = time.monotonic() + timeout
    while True:
        sock = None
        remaining = deadline - time.monotonic()
        try:
            sock = socket.create_connection(
                (host, port), timeout=max(0.05, min(5.0, remaining))
            )
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(max(0.05, min(handshake_timeout, remaining)))
            read_handshake(sock)
            return sock
        except (OSError, ValueError):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
            if time.monotonic() >= deadline:
                raise
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))


def log_row(writer: Any, t: float, poses: dict[str, Any], note: str, age: float | None,
            grip: tuple[float, float] = (0.0, 0.0),
            fire: tuple[float, float] = (0.0, 0.0),
            curl: tuple[float, float] = (0.0, 0.0),
            click: tuple[bool, bool] = (False, False)) -> None:
    """One row per emitted packet. `grip`/`curl`/`fire` are what the wire carried.

    They are logged after the --grab-mode mask on purpose: logging the request
    instead would claim a squeeze the game never saw under --grab-mode curl.
    """
    hp, hq = poses["head"]
    lp, _ = poses["left"]
    rp, _ = poses["right"]
    writer.writerow([
        f"{t:.4f}",
        f"{hp[0]:.4f}", f"{hp[1]:.4f}", f"{hp[2]:.4f}",
        f"{hq[0]:.5f}", f"{hq[1]:.5f}", f"{hq[2]:.5f}", f"{hq[3]:.5f}",
        f"{lp[0]:.4f}", f"{lp[1]:.4f}", f"{lp[2]:.4f}",
        f"{rp[0]:.4f}", f"{rp[1]:.4f}", f"{rp[2]:.4f}",
        f"{grip[0]:.1f}", f"{grip[1]:.1f}",
        f"{curl[0]:.1f}", f"{curl[1]:.1f}",
        f"{fire[0]:.1f}", f"{fire[1]:.1f}",
        f"{int(click[0])}", f"{int(click[1])}",
        "" if age is None else f"{age:.3f}",
        note,
    ])


LOG_HEADER = ["t", "hx", "hy", "hz", "hqx", "hqy", "hqz", "hqw",
              "lx", "ly", "lz", "rx", "ry", "rz",
              "grip_l", "grip_r", "curl_l", "curl_r",
              "fire_l", "fire_r", "click_l", "click_r",
              "state_age", "note"]


#: Seconds before the end of a bounded run at which the relay starts warning.
EXPIRY_WARN_S = 60.0

#: Printed when a bounded run reaches its duration.
#:
#: A relay that exits quietly leaves the game holding the last pose indefinitely,
#: and nothing else in the stack can tell: the process succeeded at everything it
#: was asked to do. Measured 2026-11-01 - the game held a frozen pose for 17
#: minutes while the dashboard reported `state: running, last_error: None`, and the
#: only visible trace was a cheery "sent 180000 packets in 3600.00 s" line. So the
#: exit says what it costs.
EXPIRY_NOTE = (
    "pose_relay: duration reached - STOPPING. The game will hold the last pose and "
    "will not respond until a relay is started again."
)


def unlimited(duration: float) -> bool:
    """`duration <= 0` means "run until killed". NaN counts as unlimited.

    NaN matters: `now - started >= nan` is False, so a NaN duration would loop
    forever, but `not (nan > 0)` is True, so it is classified rather than hoped
    about.
    """
    return not (duration > 0.0)


def deadline_state(elapsed: float, duration: float) -> str:
    """`unlimited` | `running` | `warning` | `expired`, for a bounded run."""
    if unlimited(duration):
        return "unlimited"
    remaining = duration - elapsed
    if remaining <= 0.0:
        return "expired"
    if remaining <= EXPIRY_WARN_S:
        return "warning"
    return "running"


def run(host: str, port: int, state_path: Path, log_path: Path | None,
        duration: float, hz: float, force_grab: str = "none",
        force_trigger: str = "none", grab_mode: str = "both",
        yaw_offset_rad: float = 0.0,
        fov_rad: tuple[float, float] | None = None,
        player_scale: float = 1.0,
        player_pivot: tuple[float, float, float] = (0.0, 0.0, 0.0),
        player_scale_head: bool = True,
        yaw_file: str | None = DEFAULT_YAW_FILE,
        click_file: str | None = DEFAULT_CLICK_FILE) -> int:
    wire.assert_layout()
    source = PacketSource(
        StateSource(state_path), yaw_offset_rad=yaw_offset_rad,
        player_scale=player_scale, player_pivot=player_pivot,
        player_scale_head=player_scale_head, yaw_file=yaw_file or None,
    )
    clicks = ClickFile(click_file or None)
    sock = connect(host, port)
    print(f"pose_relay: connected {host}:{port}", flush=True)
    print("pose_relay: handshake ok", flush=True)
    print(f"pose_relay: state {state_path} at {hz:g} Hz", flush=True)
    # Declared on disk as well as to the game. Every point derived from the depth stream
    # is in game space, so anything that unprojects a pixel has to know how big the
    # player was when the frame was taken; assuming 1.0 aims 1.43x too far and reads as a
    # grounding failure. Written even when unscaled, because a file left by an earlier
    # session would otherwise lie about this one.
    declared = player_scale_file.write(
        player_scale_file.PlayerScale(
            scale=player_scale,
            pivot=player_pivot,
            head_scaled=player_scale_head,
            yaw_offset_deg=math.degrees(yaw_offset_rad),
        ),
        state_path.parent,
    )
    print(f"pose_relay: player scale declared to {declared}", flush=True)
    if yaw_offset_rad:
        # Printed because a capture is only interpretable if you know which heading
        # the poses were emitted under.
            print(
                f"pose_relay: robot heading offset {math.degrees(yaw_offset_rad):+.1f} deg "
                "(declared, not walked)",
                flush=True,
            )
    if player_scale != 1.0:
        # Printed for the same reason as the heading: a capture is only interpretable if
        # you know how big the emulated player was when it was taken. This one changes
        # every distance in the room, so an unlabelled capture is actively misleading.
        print(
            f"pose_relay: emulated player scaled {player_scale:.3f}x about "
            f"({player_pivot[0]:+.3f},{player_pivot[1]:+.3f},{player_pivot[2]:+.3f}) "
            f"(declared: the robot's own reach is unchanged, the room is measured "
            f"{player_scale:.3f}x smaller for it)",
            flush=True,
        )
    if fov_rad is not None:
        # Printed because the field of view is the thing under test: a capture is only
        # interpretable if you know which one was declared when it was taken.
        half_h, half_v = fov_rad
        print(
            f"pose_relay: declaring fov {2 * math.degrees(half_h):.1f} x "
            f"{2 * math.degrees(half_v):.1f} deg full "
            f"({math.degrees(half_h):.1f}/{math.degrees(half_v):.1f} half)",
            flush=True,
        )

    period = 1.0 / max(hz, 1.0)
    started = time.monotonic()
    sent = 0
    notes: dict[str, int] = {}
    edge = ClickEdge()
    grip_force = forced_pair(force_grab)
    fire_force = forced_pair(force_trigger)
    handle = None
    writer = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # Line buffered: a 3600 s relay only closes the file at the end, so an
            # unflushed CSV is empty exactly while someone is trying to read it.
            handle = log_path.open("w", newline="", buffering=1)
            writer = csv.writer(handle)
            writer.writerow(LOG_HEADER)
        stream = sock.makefile("wb")
        next_tick = started
        warned = False
        while True:
            now = time.monotonic()
            state = deadline_state(now - started, duration)
            if state == "expired":
                print(EXPIRY_NOTE, flush=True)
                break
            if state == "warning" and not warned:
                warned = True
                print(
                    f"pose_relay: {duration - (now - started):.0f} s left; "
                    "the game freezes when this relay stops",
                    flush=True,
                )
            poses, note, age = source.poses(now)
            # Buttons: a forced mode wins, otherwise the body's own state. Both
            # are per hand, and the two channels stay separate so a gun can be
            # held while the trigger is pulled.
            grip_state, fire_state = (
                buttons_from_state(source.last_state)
                if isinstance(source.last_state, Mapping)
                else ((0.0, 0.0), (0.0, 0.0))
            )
            grip = grip_force if grip_force != (0.0, 0.0) else grip_state
            fire = fire_force if fire_force != (0.0, 0.0) else fire_state
            grip_field = grip if grab_mode in ("squeeze", "both") else (0.0, 0.0)
            curl_field = grip if grab_mode in ("curl", "both") else (0.0, 0.0)
            click = edge.update(fire)
            # The menu button: a request in a file becomes a short press here, because no joint
            # of the simulated body can reach a controller button and JobBot asks for it. The
            # game reads this as the A (right) / X (left) button, not the system button.
            menu = clicks.pressed(now)
            packet = wire.fill(
                wire.blank_packet(),
                poses,
                grip=grip_field,
                curl=curl_field,
                trigger=fire,
                trigger_click=click,
                a_click=menu,
                fov=fov_rad,
            )
            stream.write(bytes(packet))
            stream.flush()
            sent += 1
            notes[note] = notes.get(note, 0) + 1
            if writer is not None:
                log_row(writer, now - started, poses, note, age,
                        grip_field, fire, curl_field, click)
            next_tick += period
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()
    except (BrokenPipeError, ConnectionResetError) as exc:
        print(f"pose_relay: driver closed the connection: {exc}", flush=True)
        return 1
    except KeyboardInterrupt:
        print("\npose_relay: interrupted", flush=True)
    finally:
        if handle is not None:
            handle.close()
    elapsed = time.monotonic() - started
    print(f"pose_relay: sent {sent} packets in {elapsed:.2f} s "
          f"({sent / elapsed:.1f} Hz), notes {notes}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pose_relay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4242)
    ap.add_argument("--state", required=True, help="body_state.json written by the body view")
    ap.add_argument("--log", default=None, help="CSV of every emitted pose")
    ap.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="seconds to run; 0 or negative means until killed. The default is "
        "unlimited because a relay that expires freezes the game silently, while "
        "a relay that overstays is killed in a second",
    )
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--force-grab", choices=FORCE_MODES, default="none",
                    help="hold one hand closed regardless of the body, for the hub A/B")
    ap.add_argument("--force-trigger", choices=FORCE_MODES, default="none",
                    help="hold one trigger pressed regardless of the body")
    ap.add_argument("--grab-mode", choices=GRAB_MODES, default="both",
                    help="which wire field a closure drives. `squeeze` is the controller "
                         "button and registers the grab; `curl` is hand_curl, which is what "
                         "the game draws the fist from. Default `both`: squeeze alone grabs "
                         "with a visibly open glove")
    ap.add_argument("--yaw-offset-deg", type=float, default=0.0, dest="yaw_offset_deg",
                    help="the robot's heading in the virtual room, degrees, positive "
                         "toward its left; chooses the facing rather than walking it")
    ap.add_argument("--click-file", default=DEFAULT_CLICK_FILE, dest="click_file",
                    help="menu-button press requests (empty string disables)")
    ap.add_argument("--yaw-file", default=DEFAULT_YAW_FILE, dest="yaw_file",
                    help="a JSON file whose yaw_offset_deg overrides the flag while the "
                         "session runs, so the brain can aim the robot without a restart; "
                         "pass an empty string to ignore it")
    ap.add_argument("--fov-h-deg", type=float, default=None, dest="fov_h_deg",
                    help="full horizontal field of view to declare, degrees")
    ap.add_argument("--fov-v-deg", type=float, default=None, dest="fov_v_deg",
                    help="full vertical field of view to declare, degrees")
    ap.add_argument(
        "--player-scale",
        type=float,
        default=float(os.environ.get("RELAY_PLAYER_SCALE", "1.0")),
        dest="player_scale",
        help="how big the emulated player is against the room; the G1 is 1.32 m with "
             "~0.45 m of arm travel and Job Simulator is laid out for an adult, so 1.43 "
             "puts the emulated eye at 1.75 m and shrinks the reach a given object "
             "demands by the same factor. Defaults to $RELAY_PLAYER_SCALE, else 1.0. NOT "
             "$CLANK_*: clank/config.py treats any CLANK_<section>_<key> as a config "
             "override and refuses to start the stack on an unknown one - measured, the "
             "orchestrator exited with 'unknown environment override CLANK_PLAYER_SCALE'. "
             "The robot's own dimensions are unchanged",
    )
    ap.add_argument(
        "--player-pivot",
        default=os.environ.get("RELAY_PLAYER_PIVOT", "0,0,0"),
        dest="player_pivot",
        help="x,y,z the player scale is measured from, in the body's world frame "
             "(the root sits at z=0.787). The floor origin raises the hand faster than it "
             "lengthens the arm, so it makes a low, near object harder; the resting wrist "
             "is the pivot that delivers the promised reach",
    )
    ap.add_argument(
        "--player-scale-head",
        choices=("on", "off"),
        default=os.environ.get("RELAY_PLAYER_SCALE_HEAD", "on"),
        dest="player_scale_head",
        help="whether the head is scaled with the hands. OFF is the setting that "
             "actually shortens a reach: scaling the head lifts the camera, which changes "
             "the depth it reports and cancels most of the longer arm (measured: 1.43 "
             "bought 0.500 -> 0.472 m with the head scaled, against 0.35 m without). The "
             "camera then stays where every measurement in research/ was taken",
    )
    args = ap.parse_args(argv)
    log_path = Path(args.log) if args.log else None
    if not math.isfinite(args.yaw_offset_deg):
        # Refused here so the operator gets a named flag rather than a traceback out
        # of the packet builder.
        ap.error("--yaw-offset-deg must be a finite number")
    if not math.isfinite(args.player_scale) or args.player_scale <= 0.0:
        # Same reasoning as the yaw: name the flag rather than let a traceback out of
        # the packet builder.
        ap.error(
            f"--player-scale must be finite and positive, got {args.player_scale}"
        )
    try:
        pivot = tuple(float(c) for c in str(args.player_pivot).split(","))
    except ValueError:
        ap.error(f"--player-pivot must be x,y,z, got {args.player_pivot!r}")
    if len(pivot) != 3:
        ap.error(f"--player-pivot must be x,y,z, got {args.player_pivot!r}")
    fov_rad = None
    if args.fov_h_deg is not None or args.fov_v_deg is not None:
        if args.fov_h_deg is None or args.fov_v_deg is None:
            # One angle alone would need an invented aspect ratio.
            ap.error("give both --fov-h-deg and --fov-v-deg, or neither")
        angles = (("--fov-h-deg", args.fov_h_deg), ("--fov-v-deg", args.fov_v_deg))
        for flag, value in angles:
            if not 0.0 < value < 180.0:
                ap.error(
                    f"{flag} must be between 0 and 180 degrees (exclusive), got {value}"
                )
        # Halved exactly once, here: `wire.fill` takes half-angles, operators speak
        # in full ones.
        fov_rad = (math.radians(args.fov_h_deg / 2.0), math.radians(args.fov_v_deg / 2.0))
    return run(args.host, args.port, Path(args.state), log_path, args.duration, args.hz,
               force_grab=args.force_grab, force_trigger=args.force_trigger,
               grab_mode=args.grab_mode,
               yaw_offset_rad=math.radians(args.yaw_offset_deg),
               fov_rad=fov_rad, player_scale=args.player_scale,
               player_pivot=pivot,
               player_scale_head=(args.player_scale_head == "on"),
               # Both were parsed and then dropped until 2026-11-10: `--yaw-file` silently did
               # nothing, which is why an operator could set a heading and see no change.
               yaw_file=(args.yaw_file or None),
               click_file=(args.click_file or None))


if __name__ == "__main__":
    raise SystemExit(main())
