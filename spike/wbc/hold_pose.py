#!/usr/bin/env python3
"""Hold a head+hand pose against the 2 s verb TTL, for calibration.

Dashboard verbs are momentary by design: the bus stamps them with a short TTL and
returns to `hold` when it expires, so `aim` raises the hand and then lets it fall.
That is correct for play and useless for calibration, where the operator needs the
pose to *stay* while they look at the game view.

This script re-sends one combined `BodyCommand` (head yaw + one hand) every
`--interval` seconds, so the pose holds until Ctrl-C or `--seconds` runs out. On a
clean exit it sends one `rest` to give the body back its standing clip - `hold`
is freeze-hold and would leave the arm where the sweep ended.

Two modes:

- hold:   `--yaw-deg -40 --side right --forward-m 0.20`
- sweep:   `--sweep-from -60 --sweep-to 60 --sweep-step 10 --sweep-dwell 4`
  steps the head through yaw angles, printing each one, so the operator watching
  the preview can note where the desk appears. The hand holds the reach the whole
  time.

The pose builders come from the executor, so this cannot drift from what the body
supports. Only the target *values* are new; the stepping loop is untouched.

Usage (venv python, because the executor needs MuJoCo and ONNX Runtime):
    spike/e3/.venv/bin/python spike/wbc/hold_pose.py \\
        --control /home/nate/.local/state/Clank/sessions/body/control.sock \\
        --yaw-deg -40 --seconds 30
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from clank.body import (  # noqa: E402
    AUTH_HEAD,
    AUTH_L_HAND,
    AUTH_R_HAND,
    BodyCommand,
    rest,
)

HAND_BITS = {"left": AUTH_L_HAND, "right": AUTH_R_HAND}


def send_command(path: Path, cmd: BodyCommand) -> dict:
    """Push one full BodyCommand and return the server's reply."""
    payload = (json.dumps(cmd.to_dict()) + "\n").encode("utf-8")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(2.0)
        sock.connect(str(path))
        sock.sendall(payload)
        reply = b""
        while b"\n" not in reply:
            chunk = sock.recv(4096)
            if not chunk:
                break
            reply += chunk
    line, _, _ = bytes(reply).partition(b"\n")
    try:
        # A truncated reply must not kill the holder mid-sweep; the next tick
        # re-sends anyway, so a missed acknowledgement is free.
        return json.loads(line.decode("utf-8") or '{"ok": false}')
    except (ValueError, UnicodeDecodeError):
        return {"ok": False, "error": "unparseable reply"}


def build_command(
    executor, yaw_deg: float, side: str, forward_m: float, up_m: float,
    lateral_m: float, seq: int,
) -> BodyCommand:
    """One command carrying both channels, so latest-wins keeps both."""
    pose = executor.hand_target(
        side, forward_m=forward_m, up_m=up_m, lateral_m=lateral_m
    )
    kwargs = {"left_hand": pose} if side == "left" else {"right_hand": pose}
    return BodyCommand(
        seq=seq,
        t_mono_ns=time.monotonic_ns(),
        ttl_ms=2000,
        source="script",
        authority=AUTH_HEAD | HAND_BITS[side],
        head=executor.look_target(math.radians(yaw_deg)),
        **kwargs,  # type: ignore[arg-type]
    )


def run(args) -> int:
    # Imported here so --help works without the venv.
    from sonic_executor import SonicExecutor  # noqa: E402

    control = Path(args.control)
    executor = SonicExecutor()
    interval = max(0.2, args.interval)
    deadline = None if args.seconds <= 0 else time.monotonic() + args.seconds
    seq = 0

    if args.sweep_from is not None:
        yaws: list[float] = []
        yaw = args.sweep_from
        while (yaw <= args.sweep_to) if args.sweep_step > 0 else (yaw >= args.sweep_to):
            yaws.append(yaw)
            yaw += args.sweep_step
    else:
        yaws = [args.yaw_deg]

    print(
        f"hold_pose: {len(yaws)} yaw(s) side={args.side} "
        f"reach=({args.forward_m}, {args.up_m}, {args.lateral_m}) "
        f"interval={interval}s" + ("" if deadline is None else f" for {args.seconds}s"),
        flush=True,
    )
    try:
        for yaw in yaws:
            dwell_until = None
            if len(yaws) > 1:
                dwell_until = time.monotonic() + args.sweep_dwell
                print(f"hold_pose: yaw {yaw:+.0f} deg", flush=True)
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    return 0
                seq += 1
                cmd = build_command(
                    executor, yaw, args.side,
                    args.forward_m, args.up_m, args.lateral_m, seq,
                )
                try:
                    reply = send_command(control, cmd)
                except OSError as exc:
                    print(f"hold_pose: control unreachable: {exc}", flush=True)
                    return 1
                if not reply.get("ok", False):
                    print(f"hold_pose: rejected: {reply}", flush=True)
                    return 1
                if dwell_until is not None and time.monotonic() >= dwell_until:
                    break
                if dwell_until is None and deadline is None:
                    pass  # single hold runs until Ctrl-C or --seconds
                time.sleep(interval)
                if dwell_until is None and deadline is not None and time.monotonic() >= deadline:
                    return 0
    except KeyboardInterrupt:
        print("hold_pose: interrupted", flush=True)
    finally:
        try:
            send_command(control, rest(seq=seq + 1, now_ns=time.monotonic_ns()))
            print("hold_pose: released to standing clip", flush=True)
        except OSError:
            pass
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hold_pose")
    ap.add_argument("--control", required=True, help="body control socket path")
    ap.add_argument("--yaw-deg", type=float, default=0.0, help="head yaw to hold")
    ap.add_argument("--side", default="right", choices=("left", "right"))
    ap.add_argument("--forward-m", type=float, default=0.20)
    ap.add_argument("--up-m", type=float, default=0.12)
    ap.add_argument("--lateral-m", type=float, default=0.0)
    ap.add_argument("--interval", type=float, default=0.5, help="re-send cadence (s)")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = until Ctrl-C")
    ap.add_argument("--sweep-from", type=float, default=None)
    ap.add_argument("--sweep-to", type=float, default=60.0)
    ap.add_argument("--sweep-step", type=float, default=10.0)
    ap.add_argument("--sweep-dwell", type=float, default=4.0)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
