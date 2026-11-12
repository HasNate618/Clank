#!/usr/bin/env python3
"""Gate 5: does SONIC track a generated reach? Game off, body standing, GPU to itself.

This is the measurement that decides whether the whole ARDY reach survives. Generation is
already measured (`probe_ardy_reach.py`: ~46 mm of hand adherence, 0.94 GB); the open
question is whether the tracker follows the clip it is handed. NVIDIA pairs a kinematic
generator with this policy; this deploy has not measured that pairing.

The probe runs the 50 Hz body process itself (`body_view.py` under `spike/e3/.venv`, no
renderer, no game, no Monado), waits for it to settle on the standing clip, then does what
`reach_ardy` does: generate a constrained reach from the body's own achieved ring, deposit
it through the session mailbox, and watch.

Reported, with the gate from design §11:

* **residual** - the achieved hand against the constrained point, both through the same FK,
  gate **< 8 cm**. Above **15 cm** the probe stops with exit 3: that is a
  stop-the-line result for the design, not a number to tune or a reason to fall back to the
  `reach_vr` cruise.
* **root path** - how far the root actually travelled, and how far it ended from where it
  started. A planted reach should show a small shuffle, not a walk.
* **fall** - the body's own `FLAG_FALL_RISK` (base height and world-up threshold) at every
  sampled tick, plus the minimum root height and up-component for the record. The body
  process exits 3 when it reports a fall, which is also caught here.
* **look** - the achieved torso's world yaw at the end against the clip's own look, gate
  **< 15 deg**. G1 has no neck and `body_state.head` is the torso relative to the root, so
  the world look is the root's yaw plus the torso's own; the clip's look is its last-frame
  root yaw plus its `waist_yaw` hinge, which is the whole chain above the pelvis on this
  body. The achieved look is also compared with the bearing to the object and printed, but
  measured and printed against its own gate, not gated here: the torso is the camera on this
  neckless body, and the number is the generator's own waist twist plus however far off the
  object sat. Turning the root to face it was measured both as a heading on the reach's root
  goal and as a turn-only clip before it: 93-97 mm of hand residual for 15-35 deg of torso
  look, against 65 mm and 45 deg untouched. Facing swings the right hand away from the object
  and the waist twist is not a root angle, so the look is a separate problem.
  Gate 5 asks whether the tracker followed the clip.

The constrained point is placed **in front of the body**, along its own heading, from the
achieved right hand plus `--forward` and `--up`. The first attempt added those along ARDY
world +z instead, which put the target ~90 deg to the right of a body facing -89 deg and
measured the tracker on a target nobody asked it to reach.

    LD_LIBRARY_PATH=/run/opengl-driver/lib spike/ardy/.venv/bin/python \\
        spike/wbc_game/probe_ardy_track.py

Run it with the ARDY venv (torch, ARDY) - the body it launches uses `spike/e3/.venv` for
MuJoCo and ONNX. Without `LD_LIBRARY_PATH`, torch reports `cuda False` on a working driver
and `ardy_gen` silently falls back to the CPU; the probe prints which happened.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_WBC = Path(__file__).resolve().parents[1] / "wbc"
if str(_WBC) not in sys.path:
    sys.path.insert(0, str(_WBC))
_GAME = Path(__file__).resolve().parent
if str(_GAME) not in sys.path:
    sys.path.insert(0, str(_GAME))

import ardy_gen  # noqa: E402  # pyright: ignore[reportMissingImports]
import end_pose  # noqa: E402  # pyright: ignore[reportMissingImports]
import mb_mailbox  # noqa: E402  # pyright: ignore[reportMissingImports]
import reach_ardy  # noqa: E402  # pyright: ignore[reportMissingImports]

from clank.body import FLAG_FALL_RISK  # noqa: E402
from clank.camera import yaw_from_quat  # noqa: E402

E3_PYTHON = REPO / "spike" / "e3" / ".venv" / "bin" / "python"
BODY_VIEW = REPO / "spike" / "wbc" / "body_view.py"
OUT_DIR = Path.home() / ".local" / "state" / "Clank" / "ardy" / "track"

#: The design's gate, and the number that stops the line instead of failing a run.
TRACK_SUCCESS_M = end_pose.TRACK_SUCCESS_M
STOP_M = 0.15
LOOK_GATE_DEG = 15.0
#: The body's own fall thresholds, reported for the record. `FLAG_FALL_RISK` already
#: applies them inside the executor; these only say how close the run came.
FALL_Z_M = 0.25
FALL_UP_Z = 0.30

STARTUP_S = 30.0
SETTLE_S = 2.5
CLIP_TIMEOUT_S = 8.0
SAMPLE_S = 0.05


def wrap(angle: float) -> float:
    """Signed angle difference, so a heading either side of pi does not read as a jump."""
    return math.atan2(math.sin(angle), math.cos(angle))


def up_z(root_q_wxyz: tuple[float, float, float, float]) -> float:
    """The world-up component of the root's z axis; 1.0 upright, 0 on its side."""
    w, x, y, z = (float(v) for v in root_q_wxyz)
    del w, z
    return 1.0 - 2.0 * (x * x + y * y)


def place_at_bearing(
    root: tuple[float, float, float],
    target: tuple[float, float, float],
    bearing_deg: float,
) -> tuple[float, float, float]:
    """`target` swung about the vertical through `root` onto a chosen bearing, in ARDY world.

    ``bearing_deg`` is measured the way `ardy_gen.heading_angle` is - 0 faces ARDY +z, which
    is the body's forward at a zero heading - so the object's own bearing off the body is
    read from one number. The distance from the root is kept and the arm's work is not: the
    point of this knob is to move the object around the body without moving it further away.
    """
    dx, dz = target[0] - root[0], target[2] - root[2]
    delta = math.radians(bearing_deg) - math.atan2(dx, dz)
    c, s = math.cos(delta), math.sin(delta)
    return (root[0] + dx * c + dz * s, target[1], root[2] - dx * s + dz * c)


def place_at_radius(
    root: tuple[float, float, float],
    target: tuple[float, float, float],
    radius_m: float,
) -> tuple[float, float, float]:
    """`target` pulled onto a circle of `radius_m` about the root, same bearing and height.

    The default placement is `--forward` beyond the *hand* along the body's heading, which is
    0.75 m from the root - a short reach for a body that reaches straight ahead, but at the
    arm's limit for a body that has turned to face the object. With that placement a
    facing-against-not-facing comparison measures the radius rather than the bearing, so this
    knob holds the radius and leaves the bearing as the variable.
    """
    dx, dz = target[0] - root[0], target[2] - root[2]
    distance = math.hypot(dx, dz)
    if distance <= 0.0:
        return target
    scale = float(radius_m) / distance
    return (root[0] + dx * scale, target[1], root[2] + dz * scale)


def frame0_continuity(
    live: tuple[float, ...], generated: tuple[float, ...]
) -> dict:
    """How far a generated clip's first frame sits from the body it was placed on.

    The placement (`init_global_translation`, `init_first_heading_angle`) exists so the clip
    starts where the body is; anything that changes the clip's frame 0 spends that continuity
    and SONIC has to absorb the jump. Two numbers, both against their gate.
    """
    distance, heading = ardy_gen.frame0_delta(live, generated)
    return {"frame0_root_m": distance, "frame0_yaw_deg": math.degrees(heading)}


def head_yaw_deg(q_xyzw: tuple[float, float, float, float]) -> float:
    """A root-relative torso heading about Z, from the contract's `xyzw` quaternion."""
    x, y, z, w = (float(v) for v in q_xyzw)
    return math.degrees(yaw_from_quat((w, x, y, z)))


def world_head_yaw_deg(root_q_wxyz: tuple[float, float, float, float],
                       head_q_xyzw: tuple[float, float, float, float]) -> float:
    """The torso's world heading: the root's yaw plus the torso's yaw relative to it.

    `body_state.head` is root-relative (the same frame `reach_vr` unprojects the camera
    through), and both rotations are about the vertical, so the yaws add.
    """
    return math.degrees(yaw_from_quat(root_q_wxyz)) + head_yaw_deg(head_q_xyzw)


def read_state(session_dir: Path) -> dict | None:
    """`body_state.json` as written, or `None` when it is missing or torn."""
    try:
        data = json.loads((session_dir / "body_state.json").read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def start_body(session_dir: Path, out_dir: Path, seconds: float):
    """Launch the 50 Hz body process on the standing clip, logging beside the report."""
    session_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "body_view.log"
    log = log_path.open("wb")
    command = [
        str(E3_PYTHON), str(BODY_VIEW), "--out", str(session_dir), "--no-render",
        "--seconds", str(seconds),
    ]
    print(f"body: {' '.join(command)}  (log {log_path})")
    process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        command, stdout=log, stderr=subprocess.STDOUT
    )
    return process, log


def wait_for_body(session_dir: Path, process, timeout: float = STARTUP_S) -> str:
    """Wait until the pose ring is ticking; return "" or why it is not."""
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if process is not None and process.poll() is not None:
            return f"body process exited with {process.returncode}"
        ring = mb_mailbox.read_pose36(session_dir)
        if ring is not None and ring["seq"] > 50:
            return ""
        time.sleep(0.1)
    return f"no ticking pose36 ring in {timeout:.0f} s"


def play_and_sample(session_dir: Path, seq: int) -> tuple[str, list[dict], dict | None]:
    """Wait for clip `seq`, sampling the body state at every poll; return its outcome."""
    samples: list[dict] = []
    ring: dict | None = None
    started = time.monotonic()
    while True:
        status = mb_mailbox.read_status(session_dir)
        state = read_state(session_dir)
        if state is not None:
            samples.append(state)
        ring = mb_mailbox.read_pose36(session_dir) or ring
        if status is not None and status["seq_ack"] == seq:
            if status["status"] == "failed":
                return "failed", samples, ring
            if status["status"] == "done":
                return "done", samples, ring
        if time.monotonic() - started > CLIP_TIMEOUT_S:
            return "timeout", samples, ring
        time.sleep(SAMPLE_S)


def analyse(
    samples: list[dict],
    final_pose: tuple[float, ...],
    target: tuple[float, float, float],
    clip_look_deg: float,
) -> dict:
    """The four numbers gate 5 turns on, from the sampled body states."""
    residual = reach_ardy.hand_residual(final_pose, target)
    roots = [tuple(float(v) for v in st["root_p"]) for st in samples if "root_p" in st]
    first, last = samples[0], samples[-1]
    start = tuple(float(v) for v in first["root_p"])
    end = tuple(float(v) for v in last["root_p"])
    path = sum(math.dist(a, b) for a, b in itertools.pairwise(roots))
    fell = any(int(st.get("flags") or 0) & FLAG_FALL_RISK for st in samples)
    target_world = reach_ardy.world_point(target)
    bearing = math.degrees(
        math.atan2(target_world[1] - end[1], target_world[0] - end[0])
    )
    head_start = world_head_yaw_deg(
        tuple(first["root_q"]), tuple(first["head"]["q_xyzw"])
    )
    head_end = world_head_yaw_deg(tuple(last["root_q"]), tuple(last["head"]["q_xyzw"]))
    return {
        "residual_m": residual,
        "root_start": list(start),
        "root_end": list(end),
        "root_net_m": math.dist(start, end),
        "root_path_m": path,
        "min_root_z_m": min(p[2] for p in roots),
        "min_up_z": min(up_z(tuple(st["root_q"])) for st in samples),
        "fell": fell,
        "head_yaw_start_deg": head_start,
        "head_yaw_end_deg": head_end,
        "head_turn_deg": math.degrees(wrap(math.radians(head_end - head_start))),
        "clip_look_deg": clip_look_deg,
        "look_track_error_deg": math.degrees(
            wrap(math.radians(head_end - clip_look_deg))
        ),
        "look_at_object_error_deg": math.degrees(
            wrap(math.radians(head_end) - math.radians(bearing))
        ),
        "samples": len(samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="probe_ardy_track")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--session-dir", type=Path, default=None,
        help="body mailbox directory (default: <out>/body)",
    )
    parser.add_argument(
        "--no-launch", action="store_true",
        help="attach to a body already running against --session-dir",
    )
    parser.add_argument("--body-seconds", type=float, default=60.0)
    parser.add_argument("--steps", type=int, default=reach_ardy.STEPS)
    parser.add_argument("--forward", type=float, default=ardy_gen.HAND_DELTA_FWD)
    parser.add_argument(
        "--radius-m", type=float, default=None,
        help="put the object at this distance from the root instead of --forward beyond "
             "the hand (about 0.75 m); use it to compare bearings at one reach distance",
    )
    parser.add_argument("--up", type=float, default=ardy_gen.HAND_DELTA_UP)
    parser.add_argument(
        "--bearing-deg", type=float, default=None,
        help="swing the object to this bearing off the body's heading (0 = straight ahead, "
             "negative = to the body's right) instead of leaving it on the hand's forward "
             "line (about -29 deg)",
    )
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    session_dir = Path(args.session_dir) if args.session_dir else out_dir / "body"

    why = reach_ardy.preflight()
    if why is not None:
        print(f"cannot generate: {why}")
        return 3

    process = None
    log = None
    try:
        if not args.no_launch:
            process, log = start_body(session_dir, out_dir, args.body_seconds)
        else:
            print(f"body: attaching to {session_dir}")
        reason = wait_for_body(session_dir, process)
        if reason:
            print(f"body not ready: {reason}")
            return 3
        # The body reset to the standing clip at construction; give it the same settle a
        # session gives it before a command can latch a transient pose.
        time.sleep(SETTLE_S)

        ring = mb_mailbox.read_pose36(session_dir)
        if ring is None:
            print("the pose ring vanished before generation")
            return 3
        latest = ring["poses"][-1]
        hand = ardy_gen.hand_position(latest)
        heading = ardy_gen.heading_angle(latest)
        # The body's forward in ARDY world, where 0 heading faces +z (`root_waypoints`).
        forward = (math.sin(heading), 0.0, math.cos(heading))
        target = (
            hand[0] + forward[0] * args.forward,
            hand[1] + args.up,
            hand[2] + forward[2] * args.forward,
        )
        root_ardy = ardy_gen.world_root(latest)
        if args.bearing_deg is not None:
            target = place_at_bearing(
                root_ardy, target, math.degrees(heading) + args.bearing_deg
            )
        if args.radius_m is not None:
            target = place_at_radius(root_ardy, target, args.radius_m)
        off_deg = math.degrees(ardy_gen.off_bearing(latest, target))
        look_start_deg = math.degrees(ardy_gen.look_error(latest, target))
        torso_start_deg = math.degrees(ardy_gen.torso_yaw(latest))
        print(
            f"body root {tuple(round(v, 3) for v in latest[:3])} "
            f"facing {math.degrees(heading):.1f} deg"
        )
        print(
            f"right hand {tuple(round(v, 3) for v in hand)} -> constrained "
            f"{tuple(round(v, 3) for v in target)} "
            f"({args.forward:+.2f} m forward, {args.up:+.2f} m up, ARDY world metres)"
        )
        print(
            f"object {off_deg:+.1f} deg off the body's heading; the torso starts "
            f"{look_start_deg:+.1f} deg off it (gate "
            f"{math.degrees(ardy_gen.LOOK_AT_OBJECT_RAD):.0f} deg)"
        )

        frames, metrics = ardy_gen.generate(
            ring["poses"],
            hand_delta=(target[0] - hand[0], target[1] - hand[1], target[2] - hand[2]),
            root_forward_m=None,
            steps=args.steps,
        )
        if not reach_ardy.finite(frames):
            print("generated clip carries a non-finite value; nothing deposited")
            return 3
        clip_look_deg = math.degrees(ardy_gen.torso_yaw(frames[-1]))
        print(
            f"generated {len(frames)} frames in {metrics['seconds']:.3f} s, "
            f"generator residual {metrics['hand_residual_m'] * 1000:.1f} mm, "
            f"vram {metrics['peak_vram_gb']:.2f} GB"
        )
        seq = reach_ardy.deposit(session_dir, frames)
        print(f"deposited clip seq {seq}; watching the tracked body")
        outcome, samples, final_ring = play_and_sample(session_dir, seq)
        if outcome != "done":
            print(f"clip seq {seq} {outcome}")
            return 2
        if not samples or final_ring is None:
            print("no body state sampled during the clip")
            return 2

        report = {
            "generator": metrics,
            "constrained_point": list(target),
            "hand_at_start": list(hand),
            "forward_m": args.forward,
            "up_m": args.up,
            "object_off_bearing_deg": off_deg,
            "torso_at_start_deg": torso_start_deg,
            **frame0_continuity(latest, frames[0]),
            **analyse(samples, final_ring["poses"][-1], target, clip_look_deg),
        }
        residual = float(report["residual_m"])
        print(
            f"residual {residual * 1000:.1f} mm  "
            f"(gate {TRACK_SUCCESS_M * 1000:.0f} mm, stop {STOP_M * 1000:.0f} mm)"
        )
        print(
            f"root: start {tuple(round(v, 3) for v in report['root_start'])} -> "
            f"end {tuple(round(v, 3) for v in report['root_end'])}, "
            f"net {report['root_net_m']:.3f} m, path {report['root_path_m']:.3f} m"
        )
        print(
            f"frame 0: {report['frame0_root_m'] * 1000:.0f} mm, "
            f"{report['frame0_yaw_deg']:+.1f} deg from the body it was placed on"
        )
        print(
            f"fall: {'YES' if report['fell'] else 'no'}  "
            f"min root z {report['min_root_z_m']:.3f} m  "
            f"min up z {report['min_up_z']:.3f}  ({report['samples']} samples)"
        )
        print(
            f"look: achieved torso {report['head_yaw_start_deg']:.1f} -> "
            f"{report['head_yaw_end_deg']:.1f} deg world "
            f"(turn {report['head_turn_deg']:+.1f}), clip {report['clip_look_deg']:.1f}, "
            f"tracking error {report['look_track_error_deg']:+.1f} deg "
            f"(gate {LOOK_GATE_DEG:.0f}); object bearing "
            f"{report['look_at_object_error_deg']:+.1f} deg off "
            f"(gate {math.degrees(ardy_gen.LOOK_AT_OBJECT_RAD):.0f} deg)"
        )

        failures: list[str] = []
        if residual >= TRACK_SUCCESS_M:
            failures.append(
                f"residual {residual * 1000:.1f} mm >= {TRACK_SUCCESS_M * 1000:.0f} mm"
            )
        if report["fell"] or report["min_root_z_m"] < FALL_Z_M or report["min_up_z"] < FALL_UP_Z:
            failures.append("the body fell or came within its own fall thresholds")
        if abs(report["look_track_error_deg"]) > LOOK_GATE_DEG:
            failures.append(
                f"torso {report['look_track_error_deg']:+.1f} deg off the clip's look "
                f"(gate {LOOK_GATE_DEG:.0f})"
            )
        report["failures"] = failures
        path = out_dir / "probe_ardy_track.json"
        path.write_text(json.dumps(report, indent=1) + "\n")
        print(f"report: {path}")

        if residual > STOP_M:
            # No retry, no threshold change, no fallback: the tracker cannot follow a
            # generated reach, which is the one thing this design could not survive.
            print(
                f"STOP: residual {residual * 1000:.0f} mm > {STOP_M * 1000:.0f} mm. "
                f"SONIC is not tracking the generated clip; do not fall back to the cruise."
            )
            return 3
        if failures:
            print("FAIL\n  " + "\n  ".join(failures))
            return 1
        print("PASS")
        return 0
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        if log is not None:
            log.close()


if __name__ == "__main__":
    sys.exit(main())
