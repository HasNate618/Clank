#!/usr/bin/env python3
"""Does a constrained reach generate, land on its target, and land in the body's own frame?

The generator is measured in `research/motion-generators.md` §9: 0.94 GB peak, 0.07-0.20 s per
52-frame window, 60.3 mm of hand adherence. This probe exists so a regression is visible, and
so the two things a whole-body reach depends on are checked with the game closed and the GPU to
itself:

* the residual between the requested right-hand point and where the hand actually lands,
* the world frame of the exported clip: frame 0 must be at the body's own root and heading, or
  the 50 Hz body jumps when the clip is deposited, and SONIC spends the first second unwinding
  it (`reach_ardy` deposits clips into the live `mjData`),
* the round trip of a generated frame through the module's own `_motion_pose` FK, in position
  and in rotation. The residual is measured against a target `_motion_pose` itself built, so a
  joint-mapping error cancels out of it and biases every reach while the residual still looks
  small; this is the only number that sees such an error. Its gate is 30 mm rather than the 5
  mm a mapping error would deserve, because ARDY's exporter collapses each joint to one atan2:
  its own `posed_joints` sit up to 16 mm from MuJoCo's FK of the qpos it wrote itself (3.8 deg
  of off-axis rotation in the decoded local rotations), and `spike/ardy/verify_fk.py` is the
  proof that none of that gap is ours - it is 0.00 mm. The rotation term is there because a
  joint's *anchor* does not move when a leaf joint turns about its own axis, so a wrong axis or
  sign on `*_wrist_yaw` / `*_ankle_roll` would be invisible above; `hand_constraint` pins the
  hand's global rotation, so such a bug would bias every reach.

Plus the out-of-reach case, where a `root2d` path has to move the body *and* keep the hand on
its target - two constraints that silently fight if the hand's own root pin disagrees with the
path.

    LD_LIBRARY_PATH=/run/opengl-driver/lib spike/ardy/.venv/bin/python \\
      spike/wbc_game/probe_ardy_reach.py

NixOS: without that variable torch reports `cuda False` on a working driver and the whole run
falls back to the CPU. The probe prints which happened.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_WBC = _HERE.parent / "wbc"
if str(_WBC) not in sys.path:
    sys.path.insert(0, str(_WBC))

import ardy_gen  # noqa: E402  # pyright: ignore[reportMissingImports]

#: A standing pelvis: the deployed `g1_29dof_with_hand.xml` puts it 0.793 m above the floor,
#: and ARDY's sink is the same skeleton, so a zero-hinge pose at that height is its rest stand.
STAND_ROOT = (0.6, 0.0, 0.793)
STAND_HEADING = 0.8
#: Where the stand is in the game's world, as MuJoCo (x forward, y left, z up) metres.
ROOT_FRAME_GATE_M = 0.25
#: The clip's frame-0 root height against the stand's 0.793 m: measured 5-11 mm over 18
#: probe runs, so the planar tolerance was over 20x too loose for it.
HEIGHT_FRAME_GATE_M = 0.05
HEADING_FRAME_GATE_RAD = 0.25
RESIDUAL_GATE_M = 0.10
#: `_motion_pose` is checked against ARDY's own `posed_joints`: 19.4 mm on a measured clip, of
#: which ~16 mm is ARDY's export loss (its decoded local rotations are up to 3.8 deg off the
#: canonical hinge axes, and `to_qpos` keeps one atan2 per joint), and 0 mm is ours - MuJoCo's
#: FK of the same qpos agrees with `_motion_pose` to 0.00 mm (`spike/ardy/verify_fk.py`). So
#: this catches a gross mapping error - wrong joint order, axis or offset transpose, which is
#: hundreds of mm - and not a small bias.
#:
#: The gate is 50 mm, not the 30 mm this started at, because the metric is a max over 29 joints
#: and its own floor moves with diffusion noise: 36 windows over 18 runs spanned 6.7-40.3 mm,
#: so a 30 mm gate failed one run in eighteen for no reason. A gate that fires on the
#: measurement's own spread teaches the next reader to ignore it. 50 mm still sits far below
#: the hundreds of mm a real mapping error produces.
FK_ROUNDTRIP_GATE_M = 0.050
#: The same export loss in rotation: measured 2.0-10.5 deg max joint over 36 generated windows
#: (18 probe runs, typically 2.5-5, the root goal's wider motion being the high end). A leaf
#: joint on the wrong axis or sign moves its global rotation by roughly its own angle or twice
#: it - the clips' right wrist alone travels 26-74 deg of roll and 14-25 deg of yaw - so the bug
#: this exists for is tens of degrees and 20 is 1.9x the noise floor seen, not a threshold that
#: flakes.
FK_ROUNDTRIP_GATE_DEG = 20.0
SECONDS_GATE = 1.0
VRAM_GATE_GB = 2.0
#: Of a 0.6 m root goal: enough to show the path is followed, far below it if ignored (0.03).
ROOT_GATE_M = 0.40
ROOT_GOAL_M = 0.60
OUT_DIR = Path.home() / ".local/state/Clank/ardy/probe"


def yaw_quat(angle: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0))


def stand_pose36(root=STAND_ROOT, heading: float = STAND_HEADING) -> tuple[float, ...]:
    """A neutral stand at `root`, facing `heading` - what a live body ring looks like."""
    return tuple(root) + yaw_quat(heading) + (0.0,) * 29


def wrap(angle: float) -> float:
    """Signed angle difference, so a heading either side of pi does not read as a jump."""
    return math.atan2(math.sin(angle), math.cos(angle))


def frame_checks(frames, stand) -> dict:
    """Where the clip's first frame sits relative to the body it was generated for.

    The height is checked as well as the plane: `pose36[2]` carries it, and SONIC absorbs a
    vertical mismatch at deposit exactly as it absorbs a planar one.
    """
    first = frames[0]
    body_root = ardy_gen.world_root(stand)
    clip_root = ardy_gen.world_root(first)
    offset = math.dist((body_root[0], body_root[2]), (clip_root[0], clip_root[2]))
    height = abs(float(first[2]) - float(stand[2]))
    heading = wrap(ardy_gen.heading_angle(first) - ardy_gen.heading_angle(stand))
    return {"root_offset_m": offset, "height_offset_m": height, "heading_offset_rad": heading}


def write_fk_clip(out_dir: Path, frames) -> Path:
    """The clip and `_motion_pose`'s own world joints for it, for `spike/ardy/verify_fk.py`.

    MuJoCo has to read the qpos in another interpreter (the ARDY venv has no mujoco), so the
    comparison leaves here as data: `qpos`, `posed_joints`, the skeleton's joint names and the
    model XML the exporter used.
    """
    import numpy as np  # pyright: ignore[reportMissingImports]

    skeleton = ardy_gen._skeleton()
    converter = ardy_gen._converter(skeleton)
    posed = [
        ardy_gen._motion_pose(row, skeleton, converter)["posed_joints"][0, 0].tolist()
        for row in frames
    ]
    names = [""] * skeleton.nbjoints
    for name, index in skeleton.bone_index.items():
        if isinstance(index, int):
            names[index] = name
    path = out_dir / "fk_clip.npz"
    np.savez(
        path,
        qpos=np.array(frames),
        posed_joints=np.array(posed),
        joint_names=np.array(names),
        xml_path=str(converter.xml_path),
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    # Before any generation: the FK clip is written straight after the first GPU call, and
    # `np.savez` does not create parents - a fresh `--out` would otherwise raise
    # FileNotFoundError after the GPU work and before a single number is printed.
    args.out.mkdir(parents=True, exist_ok=True)

    import torch  # pyright: ignore[reportMissingImports]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"device={device} ({name}) torch={torch.__version__}")
    if device == "cpu":
        print("warning: cuda is unavailable - set LD_LIBRARY_PATH=/run/opengl-driver/lib")

    stand = stand_pose36()
    context = [stand] * 4
    hand = ardy_gen.hand_position(stand)
    delta = ardy_gen.hand_delta(ardy_gen.HAND_DELTA_FWD, ardy_gen.HAND_DELTA_UP)
    target = (hand[0] + delta[0], hand[1] + delta[1], hand[2] + delta[2])
    print(f"stand at {STAND_ROOT} facing {STAND_HEADING:.2f} rad")
    print(f"right hand {tuple(round(v, 3) for v in hand)} -> target {tuple(round(v, 3) for v in target)}")

    report: dict = {"device": device, "torch": torch.__version__, "steps": args.steps,
                    "stand": list(STAND_ROOT), "stand_heading": STAND_HEADING,
                    "cfg_weight": list(ardy_gen.CFG_WEIGHT)}

    failures: list[str] = []

    frames, metrics = ardy_gen.generate(
        context, hand_delta=delta, root_forward_m=None, steps=args.steps
    )
    checks = frame_checks(frames, stand)
    fk_clip = write_fk_clip(args.out, frames)
    print(f"hand only: {len(frames)} frames, {metrics['seconds']:.3f} s, "
          f"hand residual {metrics['hand_residual_m'] * 1000:.1f} mm")
    print(f"  peak VRAM {metrics['peak_vram_gb']:.2f} GB"
          if metrics["peak_vram_gb"] is not None else "  peak VRAM n/a (cpu)")
    print(f"  clip frame 0: root offset {checks['root_offset_m'] * 1000:.0f} mm, "
          f"height offset {checks['height_offset_m'] * 1000:.0f} mm, "
          f"heading offset {math.degrees(checks['heading_offset_rad']):.1f} deg")
    print(f"  fk roundtrip {metrics['fk_roundtrip_m'] * 1000:.1f} mm max joint, "
          f"{metrics['fk_roundtrip_deg']:.1f} deg max rotation "
          f"(gates {FK_ROUNDTRIP_GATE_M * 1000:.0f} mm / {FK_ROUNDTRIP_GATE_DEG:.0f} deg; "
          f"ours is 0.00 mm - verify: spike/e3/.venv/bin/python spike/ardy/verify_fk.py "
          f"{fk_clip})")
    report["hand_only"] = {**metrics, **checks}

    if len(frames) != ardy_gen.ARDY_HORIZON:
        failures.append(f"expected {ardy_gen.ARDY_HORIZON} frames, got {len(frames)}")
    if any(len(row) != ardy_gen.POSE_LEN for row in frames):
        failures.append("a frame is not 36 floats")
    if any(not all(math.isfinite(v) for v in row) for row in frames):
        failures.append("a frame carries a non-finite value")
    if metrics["hand_residual_m"] >= RESIDUAL_GATE_M:
        failures.append(f"hand residual {metrics['hand_residual_m'] * 1000:.1f} mm >= "
                        f"{RESIDUAL_GATE_M * 1000:.0f} mm")
    if checks["root_offset_m"] >= ROOT_FRAME_GATE_M:
        failures.append(f"clip starts {checks['root_offset_m']:.2f} m from the body")
    if checks["height_offset_m"] >= HEIGHT_FRAME_GATE_M:
        failures.append(f"clip starts {checks['height_offset_m'] * 1000:.0f} mm above/below the "
                        f"body")
    if abs(checks["heading_offset_rad"]) >= HEADING_FRAME_GATE_RAD:
        failures.append(f"clip starts {math.degrees(checks['heading_offset_rad']):.1f} deg off heading")
    if metrics["fk_roundtrip_m"] >= FK_ROUNDTRIP_GATE_M:
        failures.append(f"fk roundtrip {metrics['fk_roundtrip_m'] * 1000:.1f} mm >= "
                        f"{FK_ROUNDTRIP_GATE_M * 1000:.0f} mm")
    if metrics["fk_roundtrip_deg"] >= FK_ROUNDTRIP_GATE_DEG:
        failures.append(f"fk roundtrip {metrics['fk_roundtrip_deg']:.1f} deg >= "
                        f"{FK_ROUNDTRIP_GATE_DEG:.0f} deg")
    if metrics["seconds"] >= SECONDS_GATE:
        failures.append(f"generation took {metrics['seconds']:.2f} s >= {SECONDS_GATE:.1f} s")
    if metrics["peak_vram_gb"] is not None and metrics["peak_vram_gb"] >= VRAM_GATE_GB:
        failures.append(f"peak VRAM {metrics['peak_vram_gb']:.2f} GB >= {VRAM_GATE_GB:.1f} GB")

    goal_frames, goal_metrics = ardy_gen.generate(
        context, hand_delta=delta, root_forward_m=ROOT_GOAL_M, steps=args.steps
    )
    start, end = goal_frames[0], goal_frames[-1]
    from_root = ardy_gen.world_root(start)
    to_root = ardy_gen.world_root(end)
    heading = ardy_gen.heading_angle(stand)
    travelled = (
        (to_root[0] - from_root[0]) * math.sin(heading)
        + (to_root[2] - from_root[2]) * math.cos(heading)
    )
    print(f"root goal {ROOT_GOAL_M:.2f} m: {goal_metrics['seconds']:.3f} s, "
          f"hand residual {goal_metrics['hand_residual_m'] * 1000:.1f} mm, "
          f"root travelled {travelled:+.3f} m")
    print(f"  fk roundtrip {goal_metrics['fk_roundtrip_m'] * 1000:.1f} mm max joint, "
          f"{goal_metrics['fk_roundtrip_deg']:.1f} deg max rotation")
    report["root_goal"] = {**goal_metrics, "forward_m": ROOT_GOAL_M, "travelled_m": travelled}

    if travelled < ROOT_GATE_M:
        failures.append(f"root travelled {travelled:.2f} m of {ROOT_GOAL_M:.2f} m asked")
    if goal_metrics["hand_residual_m"] >= RESIDUAL_GATE_M:
        failures.append(f"root goal hand residual {goal_metrics['hand_residual_m'] * 1000:.1f} mm "
                        f">= {RESIDUAL_GATE_M * 1000:.0f} mm")
    if goal_metrics["fk_roundtrip_m"] >= FK_ROUNDTRIP_GATE_M:
        failures.append(f"root goal fk roundtrip {goal_metrics['fk_roundtrip_m'] * 1000:.1f} mm "
                        f">= {FK_ROUNDTRIP_GATE_M * 1000:.0f} mm")
    if goal_metrics["fk_roundtrip_deg"] >= FK_ROUNDTRIP_GATE_DEG:
        failures.append(f"root goal fk roundtrip {goal_metrics['fk_roundtrip_deg']:.1f} deg "
                        f">= {FK_ROUNDTRIP_GATE_DEG:.0f} deg")

    report["failures"] = failures
    path = args.out / "probe_ardy_reach.json"
    path.write_text(json.dumps(report, indent=1) + "\n")
    print(f"report: {path}")
    if failures:
        print("FAIL\n  " + "\n  ".join(failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
