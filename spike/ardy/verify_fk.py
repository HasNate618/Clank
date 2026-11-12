#!/usr/bin/env python3
"""Is a generated clip's qpos what its own FK said it was? MuJoCo is the judge, not ARDY.

The hand residual cannot answer this: it is measured against a target `_motion_pose` itself
built, so a mapping error cancels out of it. This compares the two things that do not cancel -
the clip's qpos rows, read by MuJoCo, and the `posed_joints` the generator computed from them.
ARDY's own `posed_joints` are not the reference: its exporter collapses each joint to a single
atan2, so they sit up to 16 mm away from MuJoCo's FK of the qpos ARDY itself wrote.

    spike/wbc_game/probe_ardy_reach.py            # writes probe/fk_clip.npz beside its report
    spike/e3/.venv/bin/python spike/ardy/verify_fk.py

Needs only mujoco + numpy, so it runs under any interpreter that has those (the ARDY venv has
torch but no mujoco; this needs no torch at all). Measured 0.00 mm at every joint on a clip
the probe generated, which is what lets the probe's `fk_roundtrip_m` gate sit at 30 mm: that
gate is for a gross mapping error, and the ~16 mm it still reads is ARDY's export loss.

The axis swap is imported from `ardy_gen`, not copied, and every hinge in the model has to be
matched by a skeleton joint: a partial match would quietly compare fewer joints and still pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco  # pyright: ignore[reportMissingImports]

_WBC = Path(__file__).resolve().parents[1] / "wbc"
if str(_WBC) not in sys.path:
    sys.path.insert(0, str(_WBC))

from ardy_gen import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    MUJOCO_TO_ARDY as _MUJOCO_TO_ARDY,
)

MUJOCO_TO_ARDY = np.array(_MUJOCO_TO_ARDY)
GATE_M = 0.005
DEFAULT_NPZ = Path.home() / ".local/state/Clank/ardy/probe/fk_clip.npz"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path, nargs="?", default=DEFAULT_NPZ)
    args = parser.parse_args()

    clip = np.load(args.npz, allow_pickle=False)
    qpos, posed = clip["qpos"], clip["posed_joints"]
    joint = {}
    for skeleton_index, name in enumerate(str(n) for n in clip["joint_names"]):
        joint[name.replace("_skel", "_joint")] = skeleton_index

    model = mujoco.MjModel.from_xml_path(str(clip["xml_path"]))  # pyright: ignore[reportAttributeAccessIssue]
    data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    targets = []  # (skeleton index, joint id or None for the pelvis body)
    hinges = 0
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)  # pyright: ignore[reportAttributeAccessIssue]
        if name in joint:
            targets.append((joint[name], j))
            hinges += 1
    if hinges != model.njnt - 1:
        raise SystemExit(
            f"matched {hinges} of the model's {model.njnt - 1} hinges (the other is the free "
            f"joint): a partial match would compare fewer joints than the clip has and pass"
        )
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")  # pyright: ignore[reportAttributeAccessIssue]
    targets.append((joint["pelvis_joint"], None))

    worst = (0.0, "", 0)
    for frame, row in enumerate(qpos):
        data.qpos[:] = row
        data.qpos[3:7] = row[3:7][[3, 0, 1, 2]]  # the clip is xyzw, MuJoCo wants wxyz
        mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]
        for skeleton_index, hinge in targets:
            here = data.xpos[pelvis] if hinge is None else data.xanchor[hinge]
            error = float(np.linalg.norm(posed[frame, skeleton_index] - MUJOCO_TO_ARDY @ here))
            if error > worst[0]:
                worst = (error, str(clip["joint_names"][skeleton_index]), frame)

    print(f"{len(qpos)} frames x {hinges} hinges + pelvis of {args.npz}: "
          f"max {worst[0] * 1000:.2f} mm at {worst[1]} (frame {worst[2]})")
    print("PASS" if worst[0] < GATE_M else f"FAIL at >= {GATE_M * 1000:.0f} mm")
    return 0 if worst[0] < GATE_M else 1


if __name__ == "__main__":
    sys.exit(main())
