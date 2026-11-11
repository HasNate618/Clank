#!/usr/bin/env python3
"""Put the hand on a point with an ARDY-generated clip, tracked by SONIC mode 0.

This is the executor the design replaces the Cartesian cruise with: state the hand target,
let a motion prior choose the reach, let the tracker put it on the body. Nothing here
streams a reference, pins the head, or clamps a wrist box - those belong to `reach_vr.py`,
which stays in the tree as the A/B until the live gate passes.

The loop, each step from a measured failure:

1. `pose36.json` is the only pose source the generator can use: `BodyState.q` is
   policy-order and cannot be unpacked back into the 36-float layout. A missing ring means
   no body process is ticking, so there is nothing to reach with.
2. The object is a robot-world point from the declared player scale
   (`clank.player_scale`), or `--world`. The declaration is not optional: assuming 1.0 aims
   1.43x too far in a scaled session. The relay's yaw offset is *not* applied - it cancels,
   and applying it mirrored every left-hand reach (`reach_vr`'s docstring has the numbers).
3. The hand constraint is `right hand at that point`, with the wrist orientation inherited
   from the achieved pose. A missing torch, CUDA or checkpoint prints and exits 3 without
   touching the session: a generator that is unavailable is unsupported, not a tear-down.
4. Generate one 52-frame window around the body's own root and heading.
5. Resample 25 -> 50 Hz and deposit it as `mb_clip.npz` plus a new `mb_play.json` seq. The
   body reads the clip, not this process.
6. Wait for the body's `done`, with two failure clocks: the 8 s play timeout, and a frozen
   `pose36.json` seq - a body whose session directory vanished keeps running and goes
   silent, so a missing status is not the only way to be stood up. With `--grab`, close the
   hand the moment the achieved hand is within `CLASP_START_M` while the clip is still
   playing; a 2 s gap between reach and close once cost 0.09 m of drift.
7. `end_pose.decide_next` decides once: `escalate` regenerates with a root goal, `stop`
   reports, `saturated` reports the residual it could not beat and exits 2.
8. Exit 0 when the achieved hand is within `--tol` (default 8 cm).

Usage:
    reach_ardy.py <px> <py> [--side right] [--world X Y Z] [--grab] [--tol 0.08]
                  [--session-dir DIR]

NixOS: run it with the ARDY venv and `LD_LIBRARY_PATH=/run/opengl-driver/lib`, e.g.

    LD_LIBRARY_PATH=/run/opengl-driver/lib spike/ardy/.venv/bin/python \\
        spike/wbc_game/reach_ardy.py 652 299 --grab
"""

from __future__ import annotations

import argparse
import io
import math
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_WBC = Path(__file__).resolve().parents[1] / "wbc"
if str(_WBC) not in sys.path:
    sys.path.insert(0, str(_WBC))

import ardy_gen  # noqa: E402  # pyright: ignore[reportMissingImports]
import clip_from_gen  # noqa: E402  # pyright: ignore[reportMissingImports]
import end_pose  # noqa: E402  # pyright: ignore[reportMissingImports]
import mb_mailbox  # noqa: E402  # pyright: ignore[reportMissingImports]
import reach_vr  # noqa: E402  # pyright: ignore[reportMissingImports]

from clank.body import AUTH_CLOSURE, Pose  # noqa: E402
from clank.depth_stream import Stream  # noqa: E402
from clank.player_scale import read as read_player_scale  # noqa: E402

#: Where the body process keeps its mailbox by default; `--session-dir` overrides it.
SESSION_DIR = reach_vr.SOCK.parent

#: How long one deposited clip may take to play. The clip is 2.08 s of motion; the margin is
#: for a body that is ticking slowly, not for a body that has stopped.
PLAY_TIMEOUT_S = 8.0
#: A pose ring that has not advanced in this long means the body is not reporting. This is
#: the second clock: a session directory removed under a running body makes its status
#: writes fail silently, and the ring's `seq` is then the only sign of life.
BODY_STALL_S = 1.0
POLL_S = 0.05

#: Depth samples per pixel. One read occasionally comes back torn, so three are taken and
#: the median used (`reach_vr` does the same for the same reason).
PIXEL_SAMPLES = 3
PIXEL_SAMPLE_S = 0.15

#: Denoising steps: the checkpoint's own schedule, which `ardy_gen.check_steps` enforces.
#: More trips a CUDA device-side assert.
STEPS = 10


def ardy_point(world: tuple[float, float, float]) -> tuple[float, float, float]:
    """A MuJoCo-world point (x forward, y left, z up) in ARDY world (x left, y up, z forward)."""
    return (float(world[1]), float(world[2]), float(world[0]))


def world_point(ardy: tuple[float, float, float]) -> tuple[float, float, float]:
    """The inverse of `ardy_point`: ARDY world metres back to MuJoCo world metres."""
    return (float(ardy[2]), float(ardy[0]), float(ardy[1]))


def hand_delta_for(
    latest: tuple[float, ...], target: tuple[float, float, float], side: str = "right"
) -> tuple[float, float, float]:
    """The displacement from the achieved hand to `target`, both in ARDY world metres.

    ARDY pins the hand *on* `hand_position + delta`, so this is the whole intent: where the
    hand has to go, stated relative to where it is.
    """
    hand = ardy_gen.hand_position(latest, hand=side)
    return (target[0] - hand[0], target[1] - hand[1], target[2] - hand[2])


def hand_residual(
    latest: tuple[float, ...], target: tuple[float, float, float], side: str = "right"
) -> float:
    """How far the achieved hand is from `target`, in metres, in ARDY world.

    Measured through the same FK the constraint was built with, so a mapping error cancels
    out of the comparison instead of turning into a false residual.
    """
    return math.dist(ardy_gen.hand_position(latest, hand=side), target)


def preflight() -> str | None:
    """Why this reach cannot be generated here, or `None`.

    A missing generator is unsupported, not a reason to touch the session: the body, the
    relay and the game are doing fine without it. NixOS reports `cuda False` on a working
    driver unless `LD_LIBRARY_PATH` carries `libcuda.so.1`, so the message names that.
    """
    try:
        import torch  # pyright: ignore[reportMissingImports]
    except (ImportError, OSError) as exc:
        return f"torch is not importable: {exc}"
    if not torch.cuda.is_available():
        return (
            "cuda is unavailable; every ARDY run needs "
            "LD_LIBRARY_PATH=/run/opengl-driver/lib"
        )
    model_dir = ardy_gen.ARDY_CKPT_DIR / ardy_gen.ARDY_MODEL
    if not model_dir.is_dir():
        return f"ARDY checkpoint missing at {model_dir}"
    return None


def object_world(
    args: argparse.Namespace, session_dir: Path, st: dict
) -> tuple[float, float, float] | None:
    """The object in MuJoCo world metres, from `--world` or the marked pixel.

    The pixel goes through `PlayerScale.pixel_point`, which applies the relay's declaration
    in one call: the camera half and the hands half are separate numbers when the head is
    not scaled, and applying one without the other is the 1.43x miss.
    """
    if args.world is not None:
        return (float(args.world[0]), float(args.world[1]), float(args.world[2]))
    declared = read_player_scale(session_dir)
    head = Pose(p=tuple(st["head"]["p"]), q_xyzw=tuple(st["head"]["q_xyzw"]))
    samples: list[tuple[float, float, float]] = []
    with Stream() as stream:
        for _ in range(PIXEL_SAMPLES):
            frame = stream.read(timeout=2.0)
            if frame is not None:
                cw, ch = int(frame["color_width"]), int(frame["color_height"])
                point = declared.pixel_point(
                    frame, args.px / cw, args.py / ch, head, reach_vr.FOV,
                    nearest=True, min_m=reach_vr.MIN_SURFACE_M,
                )
                if point is not None:
                    samples.append(point)
            time.sleep(PIXEL_SAMPLE_S)
    if not samples:
        return None
    rel = tuple(statistics.median(p[k] for p in samples) for k in range(3))
    print(
        f"depth samples {len(samples)}  spread "
        f"{max(math.dist(p, rel) for p in samples) * 1000:.0f} mm"
    )
    return reach_vr.to_world(rel, st)


def deposit(session_dir: Path, frames: list[tuple[float, ...]]) -> int:
    """Resample `frames` to SONIC's 50 Hz, write the clip, and ask the body to play it.

    The npz is replaced atomically under its final name before the play request is written:
    the body may read it on the very next tick, and a torn zip container is a `failed`
    status that costs a whole attempt.
    """
    import numpy as np  # pyright: ignore[reportMissingImports]

    motion = clip_from_gen.sonic_dict(clip_from_gen.resample_pose36(frames))
    buffer = io.BytesIO()
    np.savez(buffer, jp=motion["jp"], jv=motion["jv"], bp=motion["bp"], bq=motion["bq"])
    mb_mailbox.write_atomic(session_dir / mb_mailbox.CLIP_NAME, buffer.getvalue())
    play = mb_mailbox.read_play(session_dir)
    seq = (int(play["seq"]) if play is not None else 0) + 1
    mb_mailbox.write_play(session_dir, seq)
    return seq


def closure_for(side: str) -> tuple[float, float]:
    """The grip tuple with only `side`'s hand closed."""
    return (1.0, 0.0) if side == "left" else (0.0, 1.0)


def wait_for_play(
    session_dir: Path,
    seq: int,
    target_world: tuple[float, float, float],
    side: str,
    grab: bool,
    timeout: float = PLAY_TIMEOUT_S,
) -> tuple[str, str]:
    """Wait for the deposited clip `seq` to finish; clasp on approach while it plays.

    Returns `(outcome, why)` with outcome one of `done`, `failed`, `timeout` or `stalled`.
    `stalled` is the pose ring's `seq` not advancing: the body's status writes are
    best-effort, so a body whose session directory vanished keeps running and goes silent,
    and waiting out the full timeout on it would look like a slow body.

    The clasp compares the drawn hand, in world metres, with the object: `body_state`'s hand
    points are root-relative, which is the frame `reach_vr` commands in and the wrong one to
    measure a world target in.
    """
    started = time.monotonic()
    last_seq: int | None = None
    last_change = started
    clasped = False
    while True:
        status = mb_mailbox.read_status(session_dir)
        ring = mb_mailbox.read_pose36(session_dir)
        now = time.monotonic()
        if ring is not None and ring["seq"] != last_seq:
            last_seq = int(ring["seq"])
            last_change = now
        if grab and not clasped:
            st = reach_vr.state()
            achieved = reach_vr.to_world(tuple(st[f"{side}_hand"]["p"]), st)
            distance = math.dist(achieved, target_world)
            if distance <= end_pose.CLASP_START_M:
                reach_vr.send(authority=AUTH_CLOSURE, closure=closure_for(side))
                clasped = True
                print(
                    f"clasp  {side} hand {distance * 1000:.0f} mm from the object "
                    f"(<= {end_pose.CLASP_START_M * 1000:.0f} mm), closed mid-clip"
                )
        if status is not None and status["seq_ack"] == seq:
            if status["status"] == "failed":
                return "failed", "the body rejected the deposited clip"
            if status["status"] == "done":
                return "done", ""
        if now - started > timeout:
            return "timeout", f"no done for seq {seq} in {timeout:.1f} s"
        if now - last_change > BODY_STALL_S:
            return "stalled", f"pose36 seq frozen for {BODY_STALL_S:.1f} s"
        time.sleep(POLL_S)


def generate_clip(
    ring: dict, target: tuple[float, float, float], root_forward_m: float | None
) -> tuple[list[tuple[float, ...]], dict, tuple[float, float, float]]:
    """One generated window for `target`, from the achieved ring the body just wrote."""
    latest = ring["poses"][-1]
    delta = hand_delta_for(latest, target)
    frames, metrics = ardy_gen.generate(
        ring["poses"], hand_delta=delta, root_forward_m=root_forward_m, steps=STEPS
    )
    return frames, metrics, delta


def finite(frames: list[tuple[float, ...]]) -> bool:
    """Whether every generated value is a number. A NaN clip is not a bad reach, it is no clip."""
    return all(math.isfinite(value) for row in frames for value in row)


def main() -> int:
    parser = argparse.ArgumentParser(prog="reach_ardy")
    parser.add_argument("px", type=float, nargs="?", default=None, metavar="PX")
    parser.add_argument("py", type=float, nargs="?", default=None, metavar="PY")
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--world", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
        help="object in MuJoCo world metres instead of a pixel; the honest way to repeat a "
             "reach, because a world point does not go stale when the body leans",
    )
    parser.add_argument(
        "--grab", action="store_true",
        help="close the hand on approach, mid-clip, not after it",
    )
    parser.add_argument(
        "--tol", type=float, default=end_pose.TRACK_SUCCESS_M,
        help="success tolerance on the achieved hand residual, in metres",
    )
    parser.add_argument(
        "--session-dir", type=Path, default=SESSION_DIR,
        help="the body's mailbox directory (default: the live session body directory)",
    )
    args = parser.parse_args()
    if args.world is None and (args.px is None or args.py is None):
        parser.error("give a pixel pair or --world X Y Z")

    session_dir = Path(args.session_dir)
    # `reach_vr.send`/`state` are the measured readers of this socket and file; point them
    # at the session under test instead of re-deriving both here.
    reach_vr.SOCK = session_dir / "control.sock"
    reach_vr.STATE = session_dir / "body_state.json"

    ring = mb_mailbox.read_pose36(session_dir)
    if ring is None:
        print(
            f"no pose36 ring at {session_dir / mb_mailbox.POSE36_NAME} "
            f"(is the body running?)"
        )
        return 3
    if args.side != "right":
        print(
            "ardy_gen pins the right hand only (RightHandConstraintSet in "
            f"spike/wbc/ardy_gen.py); --side {args.side} is not implemented",
            file=sys.stderr,
        )
        return 3
    why = preflight()
    if why is not None:
        print(f"cannot generate: {why}")
        return 3

    st = reach_vr.state()
    target_world = object_world(args, session_dir, st)
    if target_world is None:
        print(f"no surface at ({args.px:.0f}, {args.py:.0f})")
        return 2
    target = ardy_point(target_world)
    p_root = reach_vr.to_rel(target_world, st)
    # There is no mink planted solve under the ARDY adapter: the generator is the reach. So
    # the only classification input is the direction gate, and `classify` says whether the
    # object is outside the planted envelope (`step`) or merely unresolved
    # (`need_planted_solve`, which `decide_next` treats as a planted reach).
    kind = end_pose.classify(p_root, None)
    print(f"object world {tuple(round(v, 3) for v in target_world)}  pelvis "
          f"{tuple(round(v, 3) for v in p_root)}  classified {kind}")
    print(f"session {session_dir}")

    started = time.monotonic()
    escalated = False
    generated = 0
    best = math.inf
    root_forward: float | None = None
    while True:
        ring = mb_mailbox.read_pose36(session_dir)
        if ring is None:
            print("the pose ring disappeared mid-reach; the body stopped")
            return 2
        generated += 1
        root_note = "" if root_forward is None else f", root {root_forward:+.2f} m forward"
        try:
            frames, metrics, _ = generate_clip(ring, target, root_forward)
        except (ImportError, FileNotFoundError) as exc:
            print(f"cannot generate: {exc}")
            return 3
        if not finite(frames):
            print("generated clip carries a non-finite value; nothing deposited")
            return 2
        print(
            f"gen {generated}: {len(frames)} frames in {metrics['seconds']:.3f} s, "
            f"generator residual {metrics['hand_residual_m'] * 1000:.1f} mm, "
            f"vram {metrics['peak_vram_gb']:.2f} GB{root_note}"
        )
        seq = deposit(session_dir, frames)
        outcome, detail = wait_for_play(
            session_dir, seq, target_world, args.side, args.grab
        )
        if outcome != "done":
            print(f"play {seq} {outcome}: {detail}")
            return 2
        final_ring = mb_mailbox.read_pose36(session_dir)
        if final_ring is None:
            print("no pose ring after the clip; the body stopped")
            return 2
        residual = hand_residual(final_ring["poses"][-1], target)
        best = min(best, residual)
        print(f"gen {generated}: achieved hand residual {residual * 1000:.1f} mm")
        decision = end_pose.decide_next(
            kind=kind,
            residual_m=residual,
            escalated=escalated,
            gen=generated,
            elapsed_s=time.monotonic() - started,
            generated=generated,
        )
        if decision == "escalate":
            # One root goal: put the root where the preferred hand point lands on the
            # object. Only the forward component is commanded - `generate`'s root path runs
            # along the body's own heading, and the hand constraint carries the facing.
            placed, _ = end_pose.step_root(
                (p_root[0], p_root[1]), args.side, (0.0, 0.0)
            )
            if end_pose.step_too_far(placed, (0.0, 0.0)):
                print(
                    f"refusing a root goal {math.dist(placed, (0.0, 0.0)):.2f} m from here: "
                    f"more than MAX_STEP_M, so the object point is a bad measurement"
                )
                return 2
            root_forward = placed[0]
            escalated = True
            print(f"escalate: one root goal, {root_forward:+.2f} m forward")
            continue
        if decision == "saturated":
            print(
                f"residual {best * 1000:.1f} mm best of {generated} generations: "
                f"no budget left for another root goal"
            )
            return 0 if best <= args.tol else 2
        break

    print(f"achieved hand residual {best * 1000:.1f} mm after {generated} generation(s)")
    return 0 if best <= args.tol else 2


if __name__ == "__main__":
    sys.exit(main())
