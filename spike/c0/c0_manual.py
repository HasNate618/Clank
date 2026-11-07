#!/usr/bin/env python3
"""Manual keyboard/mouse control for the C0 Superhot session.

Receives browser input from the clank dashboard over a Unix datagram socket,
turns it into an HMD pose plus two controller poses, and streams the same
376-byte Monado `remote` packets as c0_client.py.

Controls
  mouse             look (yaw/pitch)
  W A S D           move
  Space / Z         up / down
  Shift / Ctrl      fast / slow
  arrows            move the active hand
  wheel             reach the active hand forward/back
  F                 one-shot punch with the active hand
  left mouse / T    trigger (fire)
  right mouse / G   grip (grab/hold)
  Tab               switch active hand
  1 / 2             active hand = right / left
  R                 reset body
  H                 reset hand offset

The browser must be pointer-locked on the preview to send input. Esc releases
the lock; the client keeps streaming the last pose while disarmed.

The default hand offset is (0.000, 0.056, -0.702) m in HMD space: that is the
pistol's measured position in The Fall, so looking at it and gripping should
pick it up.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import signal
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c0_client as c0

INPUT_LEASE_SECONDS = 0.5
PUNCH_OUT_SECONDS = 0.12
PUNCH_HOLD_SECONDS = 0.04
PUNCH_BACK_SECONDS = 0.18
PUNCH_TOTAL_SECONDS = PUNCH_OUT_SECONDS + PUNCH_HOLD_SECONDS + PUNCH_BACK_SECONDS
PUNCH_REACH_M = 0.5
INPUT_KEYS = frozenset(
    {
        "1", "2", "3", "4", "5", "tab", "q", "w", "e", "r", "t", "u", "i",
        "o", "p", "enter", "ctrl", "a", "s", "d", "f", "g", "h", "j", "k",
        "l", "shift", "z", "x", "c", "v", "b", "n", "m", "space", "up",
        "down", "left", "right",
    }
)

BTN_LEFT, BTN_RIGHT = 0, 2

KEY = {
    "esc": 1, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6,
    "tab": 15, "q": 16, "w": 17, "e": 18, "r": 19, "t": 20,
    "u": 22, "i": 23, "o": 24, "p": 25, "enter": 28, "ctrl": 29,
    "a": 30, "s": 31, "d": 32, "f": 33, "g": 34, "h": 35,
    "j": 36, "k": 37, "l": 38, "shift": 42, "z": 44, "x": 45,
    "c": 46, "v": 47, "b": 48, "n": 49, "m": 50, "space": 57,
    "up": 103, "down": 108, "left": 105, "right": 106,
}


def _number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    if not math.isfinite(number) or abs(number) > 10_000:
        return 0.0
    return number


class BrowserInput:
    def __init__(self, path: Path, clock=time.monotonic):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            self._sock.bind(str(self.path))
            os.chmod(self.path, 0o600)
            self._sock.setblocking(False)
        except BaseException:
            self._sock.close()
            raise
        self._clock = clock
        self.armed = False
        self.last_batch: float | None = None
        self.keys: set[str] = set()
        self.buttons: set[int] = set()
        self.dx = self.dy = self.wheel = 0.0

    def readable(self):
        return [self._sock]

    def read(self) -> None:
        while True:
            try:
                data, _ = self._sock.recvfrom(65536)
            except BlockingIOError:
                break
            except OSError:
                break
            self._consume(data)
        self._expire()

    def _consume(self, data: bytes) -> None:
        try:
            payload = json.loads(data.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        armed = payload.get("armed")
        if armed is False:
            self._clear()
            return
        if armed is not True:
            return
        keys = payload.get("keys")
        self.keys = (
            {item for item in keys if isinstance(item, str) and item in INPUT_KEYS}
            if isinstance(keys, list)
            else set()
        )
        buttons = payload.get("buttons")
        self.buttons = (
            {
                item
                for item in buttons
                if isinstance(item, int) and not isinstance(item, bool) and item in (0, 1, 2)
            }
            if isinstance(buttons, list)
            else set()
        )
        self.dx += _number(payload.get("dx", 0))
        self.dy += _number(payload.get("dy", 0))
        self.wheel += _number(payload.get("wheel", 0))
        self.armed = True
        self.last_batch = self._clock()

    def _expire(self) -> None:
        if (
            self.armed
            and self.last_batch is not None
            and self._clock() - self.last_batch > INPUT_LEASE_SECONDS
        ):
            self._clear()

    def _clear(self) -> None:
        self.armed = False
        self.last_batch = None
        self.keys.clear()
        self.buttons.clear()
        self.dx = self.dy = self.wheel = 0.0

    def take_mouse(self):
        if not self.armed:
            self.dx = self.dy = self.wheel = 0.0
            return 0.0, 0.0, 0.0
        dx, dy, wheel = self.dx, self.dy, self.wheel
        self.dx = self.dy = self.wheel = 0.0
        return dx, dy, wheel

    def down(self, name: str) -> bool:
        return self.armed and name in self.keys

    def close(self):
        self._sock.close()
        self.path.unlink(missing_ok=True)


class Manual:
    def __init__(self, args):
        self.args = args
        # Declared once: the frustum must not change mid-session, or the game's
        # swapchains go stale (see orchestrator._start_pose).
        self.fov = c0.fov_from_args(args)
        self.pos = [0.0, 1.221, 0.0]
        self.yaw = 0.0
        self.pitch = 0.0
        self.hand = list(args.hand)
        self.active = "R"
        self.trigger = False
        self.grip = False
        self.prev_trigger = False
        self.prev_squeeze = False
        self.prev_keys: set[str] = set()
        self.crouching = False
        self.punch_elapsed: float | None = None
        self.punch_side = "R"
        self._last_dt = 0.0
        self._last_packet_positions: dict[str, tuple[float, float, float]] = {}
        self.reset()

    def reset(self):
        self.pos = [0.0, 1.221, 0.0]
        self.yaw = self.pitch = 0.0
        self.hand = list(self.args.hand)
        self.trigger = self.grip = False
        self.prev_trigger = False
        self.crouching = False
        self.cancel_punch()
        self._last_packet_positions = {}

    @property
    def punch_active(self) -> bool:
        return self.punch_elapsed is not None

    def cancel_punch(self) -> None:
        self.punch_elapsed = None

    def punch_extension(self) -> float:
        t = self.punch_elapsed
        if t is None:
            return 0.0
        if t < PUNCH_OUT_SECONDS:
            return PUNCH_REACH_M * c0.smoothstep(t / PUNCH_OUT_SECONDS)
        if t < PUNCH_OUT_SECONDS + PUNCH_HOLD_SECONDS:
            return PUNCH_REACH_M
        remaining = PUNCH_TOTAL_SECONDS - t
        if remaining <= 0.0:
            return 0.0
        return PUNCH_REACH_M * c0.smoothstep(remaining / PUNCH_BACK_SECONDS)

    def quat(self):
        cy, sy = math.cos(self.yaw / 2), math.sin(self.yaw / 2)
        cp, sp = math.cos(self.pitch / 2), math.sin(self.pitch / 2)
        return (cy * sp, sy * cp, -sy * sp, cy * cp)

    @staticmethod
    def rot(q, v):
        x, y, z, w = q
        vx, vy, vz = v
        tx = 2 * (y * vz - z * vy)
        ty = 2 * (z * vx - x * vz)
        tz = 2 * (x * vy - y * vx)
        return (vx + w * tx + (y * tz - z * ty),
                vy + w * ty + (z * tx - x * tz),
                vz + w * tz + (x * ty - y * tx))

    def update(self, inp: BrowserInput, dt: float):
        def pressed(name):
            return inp.down(name) and name not in self.prev_keys

        if pressed("r"):
            self.reset()
        if pressed("h"):
            self.hand = list(self.args.hand)
        if pressed("tab"):
            self.cancel_punch()
            self.active = "L" if self.active == "R" else "R"
        if pressed("1"):
            self.cancel_punch()
            self.active = "R"
        if pressed("2"):
            self.cancel_punch()
            self.active = "L"
        if pressed("f") and not self.punch_active:
            self.punch_elapsed = 0.0
            self.punch_side = self.active

        dx, dy, wheel = inp.take_mouse()
        self.yaw -= dx * math.radians(self.args.sens)
        sign = 1.0 if self.args.invert_y else -1.0
        self.pitch += sign * dy * math.radians(self.args.sens)
        self.pitch = max(-math.radians(85), min(math.radians(85), self.pitch))
        if self.yaw > math.pi:
            self.yaw -= 2 * math.pi
        elif self.yaw < -math.pi:
            self.yaw += 2 * math.pi

        speed = self.args.speed
        if inp.down("shift"):
            speed *= 2.5
        if inp.down("ctrl"):
            speed *= 0.3
        fwd = (-math.sin(self.yaw), 0.0, -math.cos(self.yaw))
        right = (math.cos(self.yaw), 0.0, -math.sin(self.yaw))
        if inp.down("w"):
            self.pos = [self.pos[i] + fwd[i] * speed * dt for i in range(3)]
        if inp.down("s"):
            self.pos = [self.pos[i] - fwd[i] * speed * dt for i in range(3)]
        if inp.down("d"):
            self.pos = [self.pos[i] + right[i] * speed * dt for i in range(3)]
        if inp.down("a"):
            self.pos = [self.pos[i] - right[i] * speed * dt for i in range(3)]
        if inp.down("space"):
            self.pos[1] += speed * 0.5 * dt
        if inp.down("z"):
            self.pos[1] -= speed * 0.5 * dt
        if inp.down("c"):
            if not self.crouching:
                self.pos[1] = 0.9
                self.crouching = True
        elif self.crouching:
            self.pos[1] = 1.221
            self.crouching = False
        self.pos[0] = max(-20.0, min(20.0, self.pos[0]))
        self.pos[1] = max(0.3, min(2.5, self.pos[1]))
        self.pos[2] = max(-20.0, min(20.0, self.pos[2]))

        hspeed = self.args.hand_speed
        if inp.down("up"):
            self.hand[1] += hspeed * dt
        if inp.down("down"):
            self.hand[1] -= hspeed * dt
        if inp.down("left"):
            self.hand[0] -= hspeed * dt
        if inp.down("right"):
            self.hand[0] += hspeed * dt
        self.hand[2] += wheel * 0.03
        self.hand[0] = max(-0.8, min(0.8, self.hand[0]))
        self.hand[1] = max(-0.8, min(0.8, self.hand[1]))
        self.hand[2] = max(-1.2, min(-0.15, self.hand[2]))

        if self.punch_active:
            self.punch_elapsed = (self.punch_elapsed or 0.0) + dt
            if self.punch_elapsed >= PUNCH_TOTAL_SECONDS:
                self.punch_elapsed = None

        self.trigger = inp.down("t") or BTN_LEFT in inp.buttons
        self.grip = inp.down("g") or BTN_RIGHT in inp.buttons or self.punch_active
        self.prev_keys = {name for name in KEY if inp.down(name)}
        self._last_dt = dt

    def packet(self):
        q = self.quat()
        hp = self.pos
        active_off = list(self.hand)
        if self.punch_active and self.punch_side == self.active:
            active_off[2] -= self.punch_extension()
            active_off[0] = max(-0.8, min(0.8, active_off[0]))
            active_off[1] = max(-0.8, min(0.8, active_off[1]))
            active_off[2] = max(-1.2, min(-0.15, active_off[2]))
        rest_l = (-0.222, -0.5, 0.017)
        rest_r = (0.222, -0.5, 0.017)
        hand = {
            "L": ((hp[0] + rest_l[0], hp[1] + rest_l[1], hp[2] + rest_l[2]),
                  (0.0, 0.0, 0.0, 1.0)),
            "R": ((hp[0] + rest_r[0], hp[1] + rest_r[1], hp[2] + rest_r[2]),
                  (0.0, 0.0, 0.0, 1.0)),
        }
        off = self.rot(q, active_off)
        hand[self.active] = ((hp[0] + off[0], hp[1] + off[1], hp[2] + off[2]), q)

        st = c0.State()
        st.head_p = (hp[0], hp[1], hp[2])
        st.head_q = q
        st.l_p, st.l_q = hand["L"]
        st.r_p, st.r_q = hand["R"]
        st.trigger = 1.0 if self.trigger else 0.0
        st.trigger_click = self.trigger and not self.prev_trigger
        st.squeeze = 1.0 if self.grip else 0.0
        d = c0.fill(c0.blank_packet(), st, fov=self.fov)
        for side in (d.left, d.right):
            side.active = True
            side.hand_tracking_active = False
            side.thumbstick.x = side.thumbstick.y = 0.0
            side.squeeze_force.x = 1.0 if self.grip else 0.0
            side.trigger_touch = self.trigger
            for i in range(5):
                side.hand_curl[i] = 1.0 if self.grip else 0.0
        inactive = d.left if self.active == "R" else d.right
        inactive.trigger_value.x = 0.0
        inactive.squeeze_value.x = 0.0
        inactive.squeeze_force.x = 0.0
        inactive.trigger_click = False
        inactive.trigger_touch = False
        inactive.thumbstick.x = inactive.thumbstick.y = 0.0
        for i in range(5):
            inactive.hand_curl[i] = 0.0

        current = {"L": st.l_p, "R": st.r_p}
        dt = self._last_dt
        for side, controller in (("L", d.left), ("R", d.right)):
            previous = self._last_packet_positions.get(side)
            if previous is not None and dt > 0.0:
                controller.linear_velocity.x = (current[side][0] - previous[0]) / dt
                controller.linear_velocity.y = (current[side][1] - previous[1]) / dt
                controller.linear_velocity.z = (current[side][2] - previous[2]) / dt
            else:
                controller.linear_velocity.x = 0.0
                controller.linear_velocity.y = 0.0
                controller.linear_velocity.z = 0.0
        self._last_packet_positions = current

        self.prev_trigger = self.trigger
        self.prev_squeeze = self.grip
        return d

    def status(self):
        return (f"yaw={math.degrees(self.yaw):+6.1f} pitch={math.degrees(self.pitch):+5.1f} "
                f"pos=({self.pos[0]:+.2f},{self.pos[1]:.2f},{self.pos[2]:+.2f}) "
                f"hand={self.active} off=({self.hand[0]:+.2f},{self.hand[1]:+.2f},{self.hand[2]:+.2f}) "
                f"trig={int(self.trigger)} grip={int(self.grip)} punch={int(self.punch_active)}")


def read_handshake(sock) -> None:
    """Read the driver's two reply packets, then report that the client is ready.

    The marker comes after both packets on purpose: the orchestrator uses it to
    decide the mode switch worked. Printing it at TCP connect would report
    success while the driver was still busy with the client it just replaced.
    """
    c0.read_handshake(sock, tag="[manual]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4242)
    ap.add_argument("--hz", type=float, default=90.0)
    ap.add_argument("--sens", type=float, default=0.072, help="mouse degrees per count")
    ap.add_argument("--speed", type=float, default=0.8, help="move speed, m/s")
    ap.add_argument("--hand-speed", type=float, default=0.5, help="hand speed, m/s")
    ap.add_argument("--hand", type=float, nargs=3, default=(0.0, 0.056, -0.702),
                    metavar=("X", "Y", "Z"), help="hand offset in HMD space")
    ap.add_argument("--invert-y", action="store_true")
    ap.add_argument("--input-socket", required=True, help="Unix datagram socket for browser input")
    ap.add_argument("--duration", type=float, default=0.0, help="exit after N seconds (0 = forever)")
    ap.add_argument("--fov-h-deg", type=float, default=0.0, dest="fov_h_deg",
                    help="declared horizontal field of view, degrees (0 = undeclared)")
    ap.add_argument("--fov-v-deg", type=float, default=0.0, dest="fov_v_deg",
                    help="declared vertical field of view, degrees (0 = undeclared)")
    args = ap.parse_args()

    inp = BrowserInput(Path(args.input_socket))
    print(f"[manual] input socket: {args.input_socket}", flush=True)
    print("[manual] controls: mouse look | WASD move | Space/Z up/down | arrows hand | wheel reach")
    print("[manual]           F punch | LMB/T fire | RMB/G grip | Tab/1/2 hand | R reset | H hand | Esc release")

    c0.assert_layout()
    # Retries, because the hub serves one client at a time and a switch can be
    # waiting for the previous pose client to be let go. A one-shot attempt here
    # took the whole session down on 2026-11-06 (5 s timeout, no handshake).
    sock = c0.connect_with_handshake(args.host, args.port, tag="[manual]")
    out = sock.makefile("wb")

    m = Manual(args)
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    dt = 1.0 / args.hz
    next_t = time.monotonic()
    t0 = time.monotonic()
    last_status = 0.0
    n = 0
    tty = sys.stdout.isatty()
    try:
        while running:
            now = time.monotonic()
            if args.duration and now - t0 >= args.duration:
                break
            wait = next_t - now
            if wait > 0:
                r, _, _ = select.select(inp.readable(), [], [], wait)
                if r:
                    inp.read()
                continue
            inp.read()
            m.update(inp, dt)
            out.write(bytes(m.packet()))
            out.flush()
            n += 1
            next_t += dt
            if next_t < now:
                next_t = now + dt
            if now - last_status >= 0.2:
                last_status = now
                line = "[manual] " + m.status()
                if tty:
                    print("\x1b[2K\r" + line, end="", flush=True)
                else:
                    print(line, flush=True)
    finally:
        inp.close()
        if tty:
            print("\x1b[2K\r", end="")
        print(f"[manual] sent {n} packets in {time.monotonic() - t0:.2f} s")
        sock.close()


if __name__ == "__main__":
    main()
