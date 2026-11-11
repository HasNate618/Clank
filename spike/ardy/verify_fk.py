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
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco  # pyright: ignore[reportMissingImports]

MUJOCO_TO_ARDY = np.array(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)))
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

    model = mujoco.MjModel.from_xml_path(str(clip["xml_path"]))
    data = mujoco.MjData(model)
    targets = []  # (skeleton index, joint id or None for the pelvis body)
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name in joint:
            targets.append((joint[name], j))
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    targets.append((joint["pelvis_joint"], None))

    worst = (0.0, "", 0)
    for frame, row in enumerate(qpos):
        data.qpos[:] = row
        data.qpos[3:7] = row[3:7][[3, 0, 1, 2]]  # the clip is xyzw, MuJoCo wants wxyz
        mujoco.mj_forward(model, data)
        for skeleton_index, hinge in targets:
            here = data.xpos[pelvis] if hinge is None else data.xanchor[hinge]
            error = float(np.linalg.norm(posed[frame, skeleton_index] - MUJOCO_TO_ARDY @ here))
            if error > worst[0]:
                worst = (error, str(clip["joint_names"][skeleton_index]), frame)

    print(f"{len(qpos)} frames x {len(targets)} joints of {args.npz}: max {worst[0] * 1000:.2f} mm "
          f"at {worst[1]} (frame {worst[2]})")
    print("PASS" if worst[0] < GATE_M else f"FAIL at >= {GATE_M * 1000:.0f} mm")
    return 0 if worst[0] < GATE_M else 1


if __name__ == "__main__":
    sys.exit(main())
