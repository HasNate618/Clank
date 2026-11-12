#!/usr/bin/env python3
"""Is the spread across generations sampling noise or a bias? One pose, many identical calls.

Residuals for the same reach differ by 40+ mm across unseeded runs, and the design's own number -
45.5 mm adherence with a 45-80 mm spread - mixes three things that need different fixes:

* **noise** - different samples from the same distribution, fixable by best-of-N;
* **bias** - the same signed miss every time, fixable by aiming off by it;
* **tracking** - what the body does with the clip after it is generated, measured by the gate-5
  probe instead.

Every previous figure came from a *different* target each run (`probe_ardy_track` re-derives the
object point from the run's own achieved hand), so run-to-run variation conflated all three. Here
the context, the delta and the target are frozen and only ARDY's sampling varies.

Prints, per run, the signed glove error in ARDY axes; then mean, standard deviation and the
mean of best-of-k for k = 1..n, so "best of three" is a measured improvement rather than an
assumed one.

    LD_LIBRARY_PATH=/run/opengl-driver/lib spike/ardy/.venv/bin/python \\
        spike/ardy/probe_noise.py --runs 16
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
for extra in (REPO / "spike" / "wbc", REPO / "spike" / "wbc_game"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import ardy_gen  # noqa: E402  # pyright: ignore[reportMissingImports]
from probe_ardy_reach import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    STAND_HEADING,
    STAND_ROOT,
    stand_pose36,
)

OUT_DIR = Path.home() / ".local" / "state" / "Clank" / "ardy" / "noise"


def main() -> int:
    parser = argparse.ArgumentParser(prog="probe_noise")
    parser.add_argument("--runs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--forward", type=float, default=ardy_gen.HAND_DELTA_FWD)
    parser.add_argument("--up", type=float, default=ardy_gen.HAND_DELTA_UP)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    import torch  # pyright: ignore[reportMissingImports]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__}")
    if device == "cpu":
        print("warning: cuda is unavailable - set LD_LIBRARY_PATH=/run/opengl-driver/lib")

    stand = stand_pose36(STAND_ROOT, STAND_HEADING)
    context = [stand] * 4
    glove = ardy_gen.glove_position(stand)
    wrist = ardy_gen.hand_position(stand)
    delta = ardy_gen.hand_delta(args.forward, args.up)
    target = tuple(a + b for a, b in zip(glove, delta, strict=True))
    wrist_target = tuple(a + b for a, b in zip(wrist, delta, strict=True))
    print(
        f"frozen target at {tuple(round(v, 3) for v in target)} ARDY\n"
        f"  from glove {tuple(round(v, 3) for v in glove)}, wrist "
        f"{tuple(round(v, 3) for v in wrist)} (lever 0.18 m), delta {delta}"
    )

    runs: list[dict] = []
    for run in range(1, args.runs + 1):
        started = time.monotonic()
        frames, metrics = ardy_gen.generate(
            context, hand_delta=delta, root_forward_m=None, steps=args.steps
        )
        landed = ardy_gen.glove_position(frames[-1])
        error = tuple(a - b for a, b in zip(landed, target, strict=True))
        our_wrist = math.dist(ardy_gen.hand_position(frames[-1]), wrist_target)
        runs.append(
            {
                "run": run,
                "residual_m": metrics["hand_residual_m"],
                "wrist_residual_m": metrics["wrist_residual_m"],
                "our_wrist_residual_m": our_wrist,
                "our_glove_residual_m": math.dist(landed, target),
                "fk_roundtrip_m": metrics["fk_roundtrip_m"],
                "fk_roundtrip_deg": metrics["fk_roundtrip_deg"],
                "error_ardy_m": [round(v, 4) for v in error],
                "seconds": metrics["seconds"],
                "wall_s": round(time.monotonic() - started, 3),
            }
        )
        print(
            f"{run:3d}: glove {metrics['hand_residual_m'] * 1000:6.1f} mm "
            f"(wrist {metrics['wrist_residual_m'] * 1000:6.1f}), "
            f"error {tuple(round(v * 1000) for v in error)} mm ARDY, "
            f"ours {math.dist(landed, target) * 1000:.1f} glove / {our_wrist * 1000:.1f} wrist, "
            f"roundtrip {metrics['fk_roundtrip_m'] * 1000:.1f} mm "
            f"{metrics['fk_roundtrip_deg']:.1f} deg, "
            f"{metrics['seconds']:.2f} s"
        )

    residuals = [run["residual_m"] for run in runs]
    components = list(zip(*(run["error_ardy_m"] for run in runs), strict=True))
    mean = tuple(statistics.fmean(values) for values in components)
    spread = tuple(statistics.stdev(values) for values in components) if len(runs) > 1 else (0,) * 3
    mean_magnitude = math.dist(mean, (0.0, 0.0, 0.0))
    noise_magnitude = math.sqrt(sum(value * value for value in spread))
    best_curve = {
        f"k={k}": round(sorted(residuals)[:k][-1], 4) for k in range(1, len(residuals) + 1)
    }

    report = {
        "device": device,
        "torch": torch.__version__,
        "runs": args.runs,
        "steps": args.steps,
        "forward_m": args.forward,
        "up_m": args.up,
        "target_ardy": [round(v, 4) for v in target],
        "glove_ardy": [round(v, 4) for v in glove],
        "residual_m": {
            "median": round(statistics.median(residuals), 4),
            "min": round(min(residuals), 4),
            "max": round(max(residuals), 4),
            "stdev": round(statistics.stdev(residuals), 4) if len(runs) > 1 else 0.0,
        },
        "signed_error_ardy_m": {
            "mean": [round(v, 4) for v in mean],
            "stdev": [round(v, 4) for v in spread],
            "mean_magnitude_m": round(mean_magnitude, 4),
            "noise_magnitude_m": round(noise_magnitude, 4),
        },
        "best_of_k_m": best_curve,
        "runs_detail": runs,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "noise.json").write_text(json.dumps(report, indent=2) + "\n")

    print(
        f"\nmedian {statistics.median(residuals) * 1000:.1f} mm, "
        f"min {min(residuals) * 1000:.1f}, max {max(residuals) * 1000:.1f}, "
        f"stdev {report['residual_m']['stdev'] * 1000:.1f}"
    )
    print(
        "signed error mean "
        f"{tuple(round(v * 1000) for v in mean)} mm ARDY "
        f"(magnitude {mean_magnitude * 1000:.1f} mm = bias), "
        f"per-axis stdev {tuple(round(v * 1000) for v in spread)} "
        f"(magnitude {noise_magnitude * 1000:.1f} mm = noise)"
    )
    print(
        "which dominates: "
        + ("bias" if mean_magnitude > noise_magnitude else "noise")
        + f" ({mean_magnitude * 1000:.1f} vs {noise_magnitude * 1000:.1f} mm)"
    )
    print("best of k: " + ", ".join(f"{k} {v * 1000:.1f} mm" for k, v in best_curve.items()))
    print(f"written {args.out / 'noise.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
