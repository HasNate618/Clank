#!/usr/bin/env python3
"""First ARDY G1 measurements: does the denoiser run on this card, and can it be told a hand target.

Answers three of the open questions in `research/motion-generators.md` §7:

1. peak VRAM of the G1 denoiser at batch 1 / horizon 52, with the text tower absent
2. wall time per window at the default step count
3. whether end-effector constraints are accepted, and how far the hand lands from the request

Text is a single pooled 4096-vector per sample in this checkpoint (`llm_shape: [1, 4096]` in
`config.yaml`), so the tower is not needed to run: `load_model(text_encoder=False)` plus
`text_feat=zeros(1, 1, 4096)`. That is the path the model already takes for a blank prompt
(`ardy_model.py:325`, "handle empty string (set to zero)").

The hand test is two-stage because a hand constraint needs a full pose to start from
(`EndEffectorConstraintSet` takes whole-body joint arrays and uses only the named joints). So:
generate once unconstrained, take that pose, move the right hand forward and up, constrain the
last frame to it, regenerate, and measure where the hand actually landed.

Run with the GPU to itself:
    LD_LIBRARY_PATH=/run/opengl-driver/lib \\
      spike/ardy/.venv/bin/python spike/ardy/probe_g1.py [--constraints none|root2d|hand]

NixOS note: `libcuda.so.1` lives under `/run/opengl-driver/lib`, which is not on the loader path,
so without LD_LIBRARY_PATH torch reports cuda False even with a working driver.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

CKPT_DIR = Path.home() / ".local/state/Clank/ardy"
MODEL = "ARDY-G1-RP-25FPS-Horizon52"
FPS = 25
HORIZON = 52
TEXT_DIM = 4096
OUT_DIR = Path.home() / ".local/state/Clank/ardy/probe"

#: Where to put the right hand, in ARDY model space (y up, z forward), relative to where the
#: unconstrained generation left it. 0.35 m forward and 0.15 m up is a reach in front of the chest.
HAND_DELTA = (0.0, 0.15, 0.35)


def text_feat_zero(device: str) -> torch.Tensor:
    """Zero text: one pooled 4096-d feature, which is what an empty prompt becomes."""
    return torch.zeros(1, 1, TEXT_DIM, device=device)


def root2d_constraint(skeleton, device: str, num_frames: int) -> list:
    """Walk a short way forward: the cheapest constraint that exercises the conditioning path.

    Frame indices must stay inside the generated window: the constraint filler indexes
    `motion_mask` with them (`ardy_motionrep.py:_fill_root_2d_constraints`), and an index equal to
    `num_frames` trips a device-side assert rather than a Python IndexError.
    """
    from ardy.constraints import Root2DConstraintSet

    last = num_frames - 1
    mid = num_frames // 2
    return [
        Root2DConstraintSet(
            skeleton=skeleton,
            frame_indices=torch.tensor([0, mid, last]),
            root_2d=torch.tensor([[0.0, 0.0], [0.0, -0.4], [0.0, -0.8]]),
        )
    ]


def hand_constraint(skeleton, source: dict, device: str) -> tuple[list, tuple[float, float, float]]:
    """Right hand on a displaced point at the last frame, hips carried from the same pose."""
    from ardy.constraints import RightHandConstraintSet

    frame = source["posed_joints"].shape[1] - 1
    # Constraints are assembled on the CPU: `create_pairs` (constraints.py:40) stacks the
    # constraint's frame indices with its joint indices, so both must be on one device.
    positions = source["posed_joints"][:, frame : frame + 1].cpu().clone()  # [B,1,J,3]
    rots = source["global_rot_mats"][:, frame : frame + 1].cpu().clone()  # [B,1,J,3,3]

    idx = skeleton.bone_index
    hand_joints = skeleton.right_hand_joint_names
    delta = torch.tensor(HAND_DELTA, device=positions.device, dtype=positions.dtype)
    for name in hand_joints:
        positions[:, :, idx[name]] = positions[:, :, idx[name]] + delta
    target = positions[0, 0, idx[hand_joints[0]]].tolist()

    return (
        [
            RightHandConstraintSet(
                skeleton=skeleton,
                frame_indices=torch.tensor([frame]),
                global_joints_positions=positions.squeeze(0),
                global_joints_rots=rots.squeeze(0),
                root_2d=None,
            )
        ],
        tuple(target),
    )


def generate(model, num_frames: int, steps: int, constraint_lst: list, device: str,
             cfg_weight=(2.0, 0.0)) -> dict:
    from ardy.motion_rep.tools import length_to_mask

    lengths = torch.tensor([num_frames], device=device)
    pad_mask = length_to_mask(lengths)

    observed_motion, motion_mask = None, None
    if constraint_lst:
        observed_motion, motion_mask = model.motion_rep.create_conditions_from_constraints_batched(
            constraint_lst, lengths, to_normalize=True, device=device
        )

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    with torch.no_grad():
        motion = model(
            texts=[""],
            num_frames=num_frames,
            num_denoising_steps=steps,
            pad_mask=pad_mask,
            first_heading_angle=torch.zeros(1, device=device),
            motion_mask=motion_mask,
            observed_motion=observed_motion,
            text_feat=text_feat_zero(device),
            text_pad_mask=torch.ones(1, 1, dtype=torch.bool, device=device),
            cfg_weight=cfg_weight,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        output = model.motion_rep.inverse(motion, is_normalized=True)

    return {
        "output": output,
        "seconds": elapsed,
        "peak_alloc_gb": (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else None,
        "peak_reserved_gb": (torch.cuda.max_memory_reserved() / 1e9) if device == "cuda" else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--constraints", choices=["none", "root2d", "hand"], default="none")
    ap.add_argument("--steps", type=int, default=None, help="denoising steps; default is the model's")
    ap.add_argument("--seconds", type=float, default=HORIZON / FPS)
    ap.add_argument("--cfg", nargs="+", type=float, default=[2.0, 0.0],
                    help="classifier-free guidance: [text] or [text, constraint]. The scalar form "
                         "sets the constraint weight to 0, which silently ignores constraints.")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    from ardy.model import load_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'})")

    t0 = time.time()
    model = load_model(modelname=MODEL, device=device, text_encoder=False, checkpoints_dir=str(CKPT_DIR))
    model.eval()
    print(f"loaded in {time.time() - t0:.1f} s; text_encoder={model.text_encoder}")
    if device == "cuda":
        print(f"weights resident: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    num_frames = int(round(args.seconds * FPS))
    num_frames = (num_frames // model.num_frames_per_token) * model.num_frames_per_token
    steps = args.steps if args.steps is not None else model.diffusion.num_base_steps
    if steps > model.diffusion.num_base_steps:
        raise SystemExit(f"--steps must be <= {model.diffusion.num_base_steps} (the model's schedule)")
    cfg_weight = tuple(args.cfg) if len(args.cfg) > 1 else args.cfg[0]

    report: dict = {"model": MODEL, "device": device, "frames": num_frames, "steps": steps,
                    "torch": torch.__version__, "text": "zero (no encoder)"}

    constraint_lst: list = []
    target_hand = None
    if args.constraints == "root2d":
        constraint_lst = root2d_constraint(model.skeleton, device, num_frames)
    elif args.constraints == "hand":
        print("stage 1: unconstrained pose to constrain from")
        base = generate(model, num_frames, steps, [], device, cfg_weight)
        constraint_lst, target_hand = hand_constraint(model.skeleton, base["output"], device)
        print(f"stage 1: {base['seconds']:.2f} s, hand target {target_hand}")

    print(f"stage {'2' if target_hand else '1'}: constraints "
          f"{[type(c).__name__ for c in constraint_lst] or 'none'}")
    run = generate(model, num_frames, steps, constraint_lst, device, cfg_weight)
    print(f"generate: {run['seconds']:.2f} s for {num_frames} frames "
          f"({num_frames / FPS:.2f} s of motion)")
    if run["peak_alloc_gb"] is not None:
        print(f"peak VRAM: allocated {run['peak_alloc_gb']:.2f} GB, reserved {run['peak_reserved_gb']:.2f} GB")

    if args.constraints == "root2d":
        root = run["output"]["root_positions"][0]
        travelled = float(root[-1][2] - root[0][2])
        report["root_travelled_z"] = round(travelled, 3)
        report["root_target_z"] = -0.8
        print(f"root travelled {travelled:.3f} m in z (target -0.8)")

    report.update(cfg=str(cfg_weight), seconds=round(run["seconds"], 3), peak_alloc_gb=run["peak_alloc_gb"],
                  peak_reserved_gb=run["peak_reserved_gb"], constraints=args.constraints)

    if target_hand is not None:
        idx = model.skeleton.bone_index
        hand = model.skeleton.right_hand_joint_names[0]
        landed = run["output"]["posed_joints"][0, -1, idx[hand]]
        err = float(torch.linalg.norm(landed - torch.tensor(target_hand, device=landed.device)))
        report["hand_target"] = [round(v, 4) for v in target_hand]
        report["hand_landed"] = [round(float(v), 4) for v in landed]
        report["hand_residual_m"] = round(err, 4)
        print(f"hand residual: {err * 1000:.1f} mm  (landed {report['hand_landed']})")

    args.out.mkdir(parents=True, exist_ok=True)
    from ardy.exports.mujoco import MujocoQposConverter

    conv = MujocoQposConverter(model.skeleton).to(device)
    as_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in run["output"].items()}
    qpos = conv.dict_to_qpos(as_dev, device)
    csv_path = args.out / f"probe_{args.constraints}.csv"
    conv.save_csv(qpos, str(csv_path))
    report["qpos_csv"] = str(csv_path)
    report["qpos_shape"] = list(tuple(qpos.shape))
    print(f"qpos csv: {csv_path}  shape={tuple(qpos.shape)}")

    (args.out / f"probe_{args.constraints}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
