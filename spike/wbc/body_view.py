#!/usr/bin/env python3
"""Third-person body view: SONIC in MuJoCo, rendered to JPEG, state to JSON.

Runs beside a session and writes two files that a dashboard tile can read:

    <out>/body_latest.jpg    third-person render, atomically replaced
    <out>/body_state.json    the achieved BodyState, same cadence

Why a separate process: the renderer and the policy both want the CPU, and the
orchestrator must not block on EGL. If this process dies the session survives,
which is the same rule the preview capture follows.

Camera is deliberately a fixed free camera, not a tracked one: a moving camera
makes it impossible to judge whether the *body* moved or the view did.

Usage:
    spike/e3/.venv/bin/python spike/wbc/body_view.py --out /tmp/body --seconds 10
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import socket
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
# MUJOCO_GL is not enough on its own: PyOpenGL picks its backend separately.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# gl_boot is imported for its side effect: it re-execs with LD_LIBRARY_PATH so
# libEGL resolves, which must happen before mujoco is imported. Sorting this block
# would move the local import after the third-party ones and break the renderer.
import gl_boot  # pyright: ignore[reportMissingImports]  # noqa: E402,F401,I001

# mujoco / PIL live in spike/e3/.venv only; `clank/` stays stdlib-only.
import mujoco  # pyright: ignore[reportMissingImports]  # noqa: E402
from PIL import Image  # pyright: ignore[reportMissingImports]  # noqa: E402

from sonic_executor import SonicExecutor  # pyright: ignore[reportMissingImports]  # noqa: E402
from clank.body import (  # noqa: E402
    AUTH_CLOSURE,
    AUTH_HEAD,
    AUTH_L_HAND,
    AUTH_R_HAND,
    AUTH_TRIGGER,
    BodyCommand,
    hold,
    rest,
)
from clank.bus import CommandBus  # noqa: E402
from clank.camera import PRESETS, FollowCamera, yaw_from_quat  # noqa: E402
from clank.intent import Intent, validate_intent, verb_to_intent  # noqa: E402


class BodyRenderer:
    """Fixed free camera on the robot, so motion in frame is the body's."""

    def __init__(
        self, model: mujoco.MjModel, width: int = 640, height: int = 480,  # pyright: ignore[reportAttributeAccessIssue]
                 follow: FollowCamera | None = None,
                 extra_views: tuple[str, ...] = ()) -> None:
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = mujoco.MjvCamera()  # pyright: ignore[reportAttributeAccessIssue]
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE  # pyright: ignore[reportAttributeAccessIssue]
        self.camera.lookat[:] = [0.0, 0.0, 0.85]
        self.camera.distance = 2.2
        self.camera.azimuth = 140.0
        self.camera.elevation = 0.0
        self.follow = follow
        self.views: dict[str, FollowCamera | None] = {}
        if follow is not None:
            self.views[follow.preset] = follow
        for name in extra_views:
            if name not in PRESETS or name in self.views:
                continue
            self.views[name] = FollowCamera(
                preset=name, distance=follow.distance if follow else 2.2,
                elevation=follow.elevation if follow else 0.0,
                yaw_follow=follow.yaw_follow if follow else 0.0)
        self.primary = follow.preset if follow is not None else next(iter(self.views), "three_quarter")
        self._last_t = time.monotonic()

    def _aim(
        self, data: mujoco.MjData, cam: FollowCamera | None, dt: float  # pyright: ignore[reportAttributeAccessIssue]
    ) -> None:
        """Point one camera at the body, with damping. Skips when follow is off."""
        if cam is None:
            return
        root = (float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2]))
        q = data.qpos[3:7]
        yaw = yaw_from_quat((float(q[0]), float(q[1]), float(q[2]), float(q[3])))
        lookat = cam.update(root, yaw, dt)
        self.camera.lookat[:] = list(lookat)
        self.camera.distance = cam.distance
        self.camera.azimuth = cam.azimuth(yaw)
        self.camera.elevation = cam.elevation

    # mujoco ships no type stubs, so attribute access on its classes is unchecked.
    # The pyrightconfig override for spike/wbc says the same thing; this comment
    # covers checkers that have not reloaded it yet.
    def frames(self, data: mujoco.MjData) -> dict[str, Any]:  # pyright: ignore[reportAttributeAccessIssue]
        """Render every configured angle. Each keeps its own damping state,
        so all tiles follow smoothly instead of sharing one camera."""
        out: dict[str, object] = {}
        now = time.monotonic()
        dt = now - self._last_t
        self._last_t = now
        for name, cam in self.views.items():
            self._aim(data, cam, dt)
            self.renderer.update_scene(data, camera=self.camera)
            out[name] = self._render_frame()
        return out

    def frame(self, data: mujoco.MjData):  # pyright: ignore[reportAttributeAccessIssue]
        return self.frames(data)[self.primary]

    def _render_frame(self):
        # The policy's model has nlight == 0, so a free camera renders the scene
        # nearly black (measured mean RGB 18.7, 95% near-black). MuJoCo 3.17 does
        # not let us write model.nlight, but the MjvScene is writable, and a
        # headlight follows the camera for free. Measured after: mean RGB 56.7.
        scene = self.renderer.scene
        scene.nlight = 1
        light = scene.lights[0]
        light.headlight = 1
        light.diffuse[:] = [1.0, 1.0, 1.0]
        light.ambient[:] = [0.35, 0.35, 0.35]
        light.specular[:] = [0.3, 0.3, 0.3]
        light.castshadow = 0
        return self.renderer.render()


def write_atomic(path: Path, payload: bytes) -> None:
    # The temp name carries the pid: two viewers rendering into the same --out used
    # to share one .tmp path, so each one's replace() could find the other's tmp
    # already renamed away (20 FileNotFoundError tracebacks in 103 s).
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)


def reel_plan(
    executor: SonicExecutor,
) -> list[tuple[str, float, Callable[[], BodyCommand] | None]]:
    """The verb reel, as (name, seconds, command factory) triples.

    `hold` is the empty authority mask, so it is `None` rather than a command -
    there is no hold verb to send. Everything else reuses the executor's own
    target builders, so the reel cannot drift from what the executor supports.
    """
    return [
        ("hold", 1.5, None),
        ("look left 35", 3.0, lambda: BodyCommand(
            seq=0, authority=AUTH_HEAD, head=executor.look_target(math.radians(35.0)))),
        ("hold", 1.5, None),
        ("look right 35", 3.0, lambda: BodyCommand(
            seq=0, authority=AUTH_HEAD, head=executor.look_target(math.radians(-35.0)))),
        ("hold", 1.5, None),
        ("aim right", 2.5, lambda: BodyCommand(
            seq=0, authority=AUTH_R_HAND,
            right_hand=executor.aim_target("right", forward_m=0.22))),
        ("punch right", 2.0, lambda: BodyCommand(
            seq=0, authority=AUTH_R_HAND, right_hand=executor.punch_target("right"))),
        ("dodge left", 1.5, lambda: BodyCommand(
            seq=0, authority=AUTH_HEAD, head=executor.dodge_target(0.14))),
        ("hold", 2.0, None),
    ]


def _hand_command(pose, side: str, frame: str = "root") -> BodyCommand:
    """One hand's target on the matching authority bit.

    `frame="head"` lets the executor carry the target with the torso twist, so a
    reached hand stays where the camera is looking instead of swinging out of view.
    """
    if side == "left":
        return BodyCommand(seq=0, authority=AUTH_L_HAND, left_hand=pose, frame=frame)
    return BodyCommand(seq=0, authority=AUTH_R_HAND, right_hand=pose, frame=frame)


def _command_from_intent(
    executor: SonicExecutor, intent: Intent
) -> BodyCommand | None:
    """Compile an intent to the `BodyCommand` it implies, via executor builders.

    Only the parameterised skills route through here. Buttons stay in
    `verb_command`'s own table because they travel with a hand pose, and that pose
    is what makes a null result attributable to the reach rather than the button.
    """
    if intent.skill == "hold":
        # Freeze-hold at what the body achieved. `executor.last_state` is the same
        # BodyState the bus publishes, so the adapter and the bus agree.
        return hold(state=executor.last_state)
    if intent.skill == "rest":
        return rest()
    if intent.skill == "look":
        return BodyCommand(
            seq=0,
            authority=AUTH_HEAD,
            head=executor.look_target(float(intent.yaw_rad or 0.0)),
        )
    if intent.skill == "dodge":
        # The lean is on the pelvis-frame lateral axis. A writer that names a
        # shallower lean than the default gets exactly that.
        _, _, lateral = intent.offset or (0.0, 0.0, 0.14)
        return BodyCommand(
            seq=0, authority=AUTH_HEAD, head=executor.dodge_target(lateral)
        )

    forward, up, lateral = intent.offset or (0.0, 0.0, 0.0)
    if intent.skill == "reach":
        return _hand_command(
            executor.hand_target(
                intent.side, forward_m=forward, up_m=up, lateral_m=lateral
            ),
            intent.side,
            frame=intent.frame,
        )
    if intent.skill == "strike":
        return _hand_command(
            executor.hand_target(intent.side, forward_m=forward, up_m=up),
            intent.side,
            frame=intent.frame,
        )
    return None


def verb_command(
    executor: SonicExecutor, name: str, intent: Intent | None = None
) -> BodyCommand | None:
    """Map a pushed button to a `BodyCommand`, using the executor's own builders.

    This is an adapter, exactly like the future game adapters: the contract stays
    `BodyCommand`, and a verb is only a shortcut for the pose it implies. Returns
    None for an unknown verb so the caller can report it rather than guess.

    `intent` is the already-parsed, already-validated form of a parameterised verb
    (`look` with a yaw, `reach` with an offset). Parsing happens once in the caller
    so that rejection has one owner; this function only compiles.
    """
    if intent is not None:
        return _command_from_intent(executor, intent)

    # Yaw is about +Z in this Z-up frame, and a positive rotation turns toward
    # +Y, which is the robot's own left. These two signs were swapped: "look
    # left" yawed -35 and turned the body to its right, measured in the world
    # forward vector (facing -Y at rest, so its left is +X).
    def head_left() -> BodyCommand:
        return BodyCommand(
            seq=0, authority=AUTH_HEAD, head=executor.look_target(math.radians(35.0)))

    def head_right() -> BodyCommand:
        return BodyCommand(
            seq=0, authority=AUTH_HEAD, head=executor.look_target(math.radians(-35.0)))

    def aim() -> BodyCommand:
        return BodyCommand(
            seq=0, authority=AUTH_R_HAND,
            right_hand=executor.aim_target("right", forward_m=0.22))

    def punch() -> BodyCommand:
        return BodyCommand(
            seq=0, authority=AUTH_R_HAND, right_hand=executor.punch_target("right"))

    def dodge_left() -> BodyCommand:
        return BodyCommand(
            seq=0, authority=AUTH_HEAD, head=executor.dodge_target(0.14))

    def dodge_right() -> BodyCommand:
        return BodyCommand(
            seq=0, authority=AUTH_HEAD, head=executor.dodge_target(-0.14))

    def hand_button(side: str, closed: bool) -> BodyCommand:
        """Close or open one hand, at the reach the aim verb already uses.

        The closure travels with a hand pose on purpose: a bare button press fires
        at whatever the arm happens to be doing, which makes a null result
        impossible to attribute to a reach or to a button.
        """
        target = executor.aim_target(side, forward_m=0.22)
        closed_pair = (1.0, 0.0) if side == "left" else (0.0, 1.0)
        closure = closed_pair if closed else (0.0, 0.0)
        if side == "right":
            return BodyCommand(
                seq=0,
                authority=AUTH_R_HAND | AUTH_CLOSURE,
                right_hand=target,
                closure=closure,
            )
        return BodyCommand(
            seq=0,
            authority=AUTH_L_HAND | AUTH_CLOSURE,
            left_hand=target,
            closure=closure,
        )

    def fire(side: str) -> BodyCommand:
        trigger = (0.0, 1.0) if side == "right" else (1.0, 0.0)
        return BodyCommand(seq=0, authority=AUTH_TRIGGER, trigger=trigger)

    verbs = {
        "hold": hold,
        "look_left": head_left,
        "look_right": head_right,
        "aim": aim,
        "punch": punch,
        "dodge_left": dodge_left,
        "dodge_right": dodge_right,
        "grab_left": lambda: hand_button("left", True),
        "grab_right": lambda: hand_button("right", True),
        "release_left": lambda: hand_button("left", False),
        "release_right": lambda: hand_button("right", False),
        "fire_left": lambda: fire("left"),
        "fire_right": lambda: fire("right"),
    }
    make = verbs.get(name)
    if make is None:
        return None
    return make()


class VerbServer:
    """Serve newline-JSON on a Unix socket: a verb name, or a full BodyCommand.

    Verbs are stamped here with a sequence number and a timestamp, so a UI never
    has to reason about clocks, and a stale press cannot rewind the body.
    """

    def __init__(self, path: Path, bus, executor: SonicExecutor,
                 set_view: Callable[[str], None] | None = None) -> None:
        self.path = Path(path)
        self.bus = bus
        self.executor = executor
        self.set_view = set_view
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._seq = 0
        self._stop = False

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            # A live owner means a working control channel. Unlinking it anyway
            # leaves that process listening on a path nobody can reach, while the
            # new one silently answers the dashboard: the dashboard then drives a
            # body that is not the one on screen. Refuse, and let the caller say so.
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            probe.settimeout(0.5)
            try:
                probe.connect(str(self.path))
            except OSError:
                pass
            else:
                raise RuntimeError(
                    f"another body view is already listening on {self.path}"
                )
            finally:
                probe.close()
            with contextlib.suppress(FileNotFoundError):
                self.path.unlink()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(str(self.path))
        self._sock.listen(4)
        self._sock.settimeout(0.5)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _handle(self, line: str) -> dict:
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"bad json: {exc}"}
        if not isinstance(raw, dict):
            return {"ok": False, "error": "expected an object"}

        if "view" in raw and "verb" not in raw:
            name = str(raw["view"])
            if name not in PRESETS:
                return {"ok": False, "error": f"unknown view: {name}"}
            if self.set_view is None:
                return {"ok": False, "error": "view switching unavailable"}
            self.set_view(name)
            return {"ok": True, "view": name}

        if "verb" not in raw:
            result = self.bus.push_json(line)       # a full BodyCommand
            return {"ok": bool(result.ok), "error": result.reason or ""}

        name = str(raw["verb"])
        # Parameterised verbs (`look` with a yaw, `reach` with an offset) are parsed
        # and validated here, so rejection has exactly one owner and the error can
        # name the problem instead of claiming an unknown verb.
        #
        # The parse is guarded because it raises on non-numeric input: unguarded, a
        # single `{"verb": "reach", "forward_m": "far"}` would propagate out of
        # `_serve` and kill the socket thread, leaving the dashboard driving a body
        # that no longer answers. Measured before the guard: ValueError escaped.
        try:
            parsed = verb_to_intent(name, raw)
        except (TypeError, ValueError) as exc:
            return {"ok": False, "error": f"bad verb parameters: {exc}"}
        if parsed is not None:
            problem = validate_intent(parsed)
            if problem:
                return {"ok": False, "error": problem}
        cmd = verb_command(self.executor, name, parsed)
        if cmd is None:
            return {"ok": False, "error": f"unknown verb: {raw['verb']}"}
        self._seq += 1
        try:
            ttl_ms = int(raw.get("ttl_ms", 2000))
        except (TypeError, ValueError) as exc:
            return {"ok": False, "error": f"bad ttl_ms: {exc}"}
        stamped = BodyCommand(**{
            **cmd.__dict__,
            "seq": self._seq,
            "t_mono_ns": time.monotonic_ns(),
            "ttl_ms": ttl_ms,
        })
        self.bus.push(stamped)
        return {"ok": True, "verb": raw["verb"], "seq": self._seq,
                "authority": stamped.authority}

    def _serve(self) -> None:
        sock = self._sock
        if sock is None:
            return
        while not self._stop:
            try:
                conn, _ = sock.accept()
            except (TimeoutError, OSError):
                continue
            with conn:
                buf = b""
                while not self._stop:
                    try:
                        chunk = conn.recv(65536)
                    except OSError:
                        break
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if not line.strip():
                            continue
                        reply = self._handle(line.decode("utf-8", "replace"))
                        with contextlib.suppress(OSError):
                            conn.sendall((json.dumps(reply) + "\n").encode())

    def stop(self) -> None:
        self._stop = True
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
            self._sock = None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="body_view")
    ap.add_argument("--out", required=True, help="directory for body_latest.jpg and body_state.json")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = run forever")
    ap.add_argument("--fps", type=float, default=10.0, help="render cadence (policy still runs at 50 Hz)")
    ap.add_argument("--clip", default=None, help="motion clip (default: the hold clip)")
    ap.add_argument("--control", default=None,
                    help="Unix socket that accepts {\"verb\": ...} or a full BodyCommand")
    ap.add_argument("--no-render", action="store_true",
                    help="physics and state only; needs no GPU")
    ap.add_argument("--reel", action="store_true",
                    help="cycle the verb reel (hold, look, aim, punch, dodge) instead of holding")
    ap.add_argument("--look-deg", type=float, default=0.0, help="hold a fixed head yaw, for testing")
    ap.add_argument("--views", default="three_quarter",
                    help="comma-separated camera angles rendered together; "
                    "the first is the primary tile (choices: three_quarter,front,side,back)")
    ap.add_argument("--yaw-follow", type=float, default=0.0,
                    help="0 = camera keeps its world angle (default); 1 = ride the body's yaw")
    ap.add_argument("--no-follow", action="store_true",
                    help="static camera at the origin instead of following the body")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    executor = SonicExecutor(clip=args.clip) if args.clip else SonicExecutor()
    wanted = [w.strip() for w in str(args.views).split(",") if w.strip()]
    for w in wanted:
        if w not in PRESETS:
            print(f"body_view: unknown view {w!r}", file=sys.stderr, flush=True)
            return 2
    if not wanted:
        wanted = ["three_quarter"]
    renderer = None
    if not args.no_render:
        follow = None if args.no_follow else FollowCamera(
            preset=wanted[0], yaw_follow=args.yaw_follow)
        renderer = BodyRenderer(executor.runner.model, width=args.width,
                                height=args.height, follow=follow,
                                extra_views=tuple(wanted[1:4]))
        print(f"body_view: views {sorted(renderer.views)} "
              f"(primary {renderer.primary}, follow {follow is not None})", flush=True)

    bus = None
    server = None
    if args.control:
        bus = CommandBus()
        def set_view(name: str) -> None:
            if renderer is not None:
                if name in renderer.views:
                    renderer.primary = name
                elif len(renderer.views) < 4 and not args.no_render:
                    base = renderer.follow
                    renderer.views[name] = FollowCamera(
                        preset=name,
                        distance=base.distance if base else 2.2,
                        elevation=base.elevation if base else 0.0,
                        yaw_follow=base.yaw_follow if base else 0.0)
                    renderer.primary = name
                else:
                    return
            print(f"body_view: view {name}", flush=True)

        server = VerbServer(Path(args.control), bus, executor, set_view=set_view)
        try:
            server.start()
        except RuntimeError as exc:
            print(f"body_view: {exc}", file=sys.stderr, flush=True)
            return 1
        print(f"body_view: control socket {args.control}", flush=True)

    if args.look_deg:
        cmd = BodyCommand(seq=0, authority=AUTH_HEAD,
                          head=executor.look_target(math.radians(args.look_deg)))
    else:
        cmd = hold()

    period = 1.0 / max(args.fps, 0.1)
    # The policy was designed for 50 Hz (see the three-loop design). Without a
    # sleep the loop free-ran at ~90 Hz and pinned a core for no benefit.
    control_period = 1.0 / 50.0
    started = time.monotonic()
    next_render = started
    renders = 0

    plan = reel_plan(executor) if args.reel else None
    phase_name = ""
    print(
        f"body_view: {'reel' if plan else 'hold'} clip {executor.clip} -> {out}",
        flush=True,
    )
    if plan:
        print("body_view: reel = " + " -> ".join(p[0] for p in plan), flush=True)
        phase_total = sum(p[1] for p in plan)
    try:
        while True:
            loop_start = time.monotonic()
            if bus is not None:
                cmd = bus.take(now_ns=time.monotonic_ns())
            elif plan:
                t = (time.monotonic() - started) % phase_total
                acc = 0.0
                for name, secs, factory in plan:
                    acc += secs
                    if t < acc:
                        cmd = factory() if factory else hold()
                        if name != phase_name:
                            phase_name = name
                            print(f"body_view: phase {name}", flush=True)
                        break
            state = executor.step(cmd, now_ns=time.monotonic_ns())
            now = time.monotonic()
            if bus is not None:
                bus.publish_state(state)
            # Written every control tick, NOT inside the render branch. This file
            # is what the pose relay reads, so gating it on the render cadence made
            # the game's pose rate a side effect of the tile's frame rate: at
            # --fps 10 the body ran at 45 Hz but the game got 100 ms pose steps.
            # ~2 KB at 45 Hz is cheap next to a 22 ms control tick.
            write_atomic(
                out / "body_state.json",
                json.dumps(state.to_dict()).encode("utf-8"),
            )
            if now >= next_render:
                if renderer is not None:
                    for name, img in renderer.frames(executor.runner.data).items():
                        buf = io.BytesIO()
                        Image.fromarray(img).save(buf, format="JPEG", quality=85)
                        write_atomic(out / f"body_{name}.jpg", buf.getvalue())
                        if name == renderer.primary:
                            write_atomic(out / "body_latest.jpg", buf.getvalue())
                    write_atomic(
                        out / "body_camera.json",
                        json.dumps({
                            "views": sorted(renderer.views),
                            "primary": renderer.primary,
                            "follow": renderer.follow is not None,
                            "lookat": {
                                n: [round(v, 4) for v in c.lookat]
                                for n, c in renderer.views.items()
                                if c is not None
                            },
                        }).encode("utf-8"),
                    )
                next_render = now + period
                renders += 1
            if executor.fell:
                print("body_view: FALL detected, stopping", file=sys.stderr, flush=True)
                return 3
            if args.seconds and (now - started) >= args.seconds:
                break
            spare = control_period - (time.monotonic() - loop_start)
            if spare > 0:
                time.sleep(spare)
    except KeyboardInterrupt:
        pass

    if server is not None:
        server.stop()
    elapsed = time.monotonic() - started
    print(f"body_view: {renders} frames in {elapsed:.1f}s "
          f"({renders / max(elapsed, 1e-9):.1f} fps), ticks={executor.ticks}, "
          f"fell={executor.fell}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
