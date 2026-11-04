"""Fade the game's background by depth, so a model looking at one crop sees one object.

Job Simulator is flat and bright: a white badge on a white console in a room full of white
signage. Asking the model to find one thing in that frame measured 340-360 px of error on
the full frame against 39-65 px on a crop, and it is the background that the crop removes.
This does the same thing with the depth the renderer already gives us, so the crop can be
generated instead of hand-drawn - which is also what refusal needs, since the model only
says "not there" when the crop is tight.

## The curve, and why it is a curve

The depth stream is not clean: it returns torn frames (one read of the card gave 0.29 m and
another 1.20 m a minute later), and the game has thin geometry that the 480x270 buffer
resolves poorly. A hard cut at "the object's depth" would therefore cut into the object
itself on a bad frame.

So there is a **dead zone**: everything within `near` times the object's depth is untouched,
and only geometry past `far` times it is fully faded, with a smoothstep between. Measured on
the card at 0.58 m, the counter it stands on is 0.55-0.65 m and the room behind is 3-11 m,
so the useful band is wide and the default 1.8/3.5 lands in the empty space between them.

Invalid depth (0 from a missing sample, or anything past `_FAR_SENTINEL`) is treated as far.
That is the safe direction: fading something that was close loses information the model can
recover from another frame, while keeping something that is far is what confuses it.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from PIL import Image

#: Depth at or above this is a missing sample, not a surface. The stream reported
#: `far_m 0.0` on torn frames, and single reads have produced 11.8 m inside a room.
FAR_SENTINEL = 12.0

#: Multiples of the object's depth: below `NEAR_K` nothing changes, past `FAR_K` only the
#: faded colour remains.
NEAR_K = 1.8
FAR_K = 3.5

#: What the background fades toward. Dark and neutral: the game is white-on-white, so
#: blending toward white would leave the object as flat as everything else.
BACKDROP = (18, 22, 26)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def fade(
    rgb: np.ndarray,
    depth: np.ndarray,
    target_m: float,
    near: float = NEAR_K,
    far: float = FAR_K,
    backdrop: tuple[int, int, int] = BACKDROP,
) -> np.ndarray:
    """`rgb` with everything further than `far * target_m` replaced by `backdrop`.

    `depth` may be a different resolution from `rgb`; it is upsampled nearest-neighbour,
    which is right here because the fade is a mask, not a picture - interpolating across a
    depth edge would smear the object's own outline.
    """
    if target_m <= 0:
        raise ValueError(f"target_m must be positive, got {target_m}")
    if far <= near:
        raise ValueError(f"far ({far}) must exceed near ({near})")
    height, width = rgb.shape[:2]
    dh, dw = depth.shape[:2]
    if (dh, dw) != (height, width):
        scale_y = dh / height
        scale_x = dw / width
        rows = np.clip((np.arange(height) * scale_y).astype(int), 0, dh - 1)
        cols = np.clip((np.arange(width) * scale_x).astype(int), 0, dw - 1)
        depth = depth[rows][:, cols]
    d = np.asarray(depth, dtype=np.float64)
    d = np.where(np.isfinite(d) & (d > 0.0) & (d < FAR_SENTINEL), d, FAR_SENTINEL)
    # 0 at the near edge, 1 at the far edge, smoothed so a bad depth sample on the object's
    # own surface only dims it slightly rather than deleting it.
    t = _smoothstep((d - near * target_m) / (far * target_m - near * target_m))
    alpha = (1.0 - t)[..., None]
    return (rgb.astype(np.float64) * alpha + np.asarray(backdrop, float) * (1.0 - alpha)).astype(
        np.uint8
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("colour", type=pathlib.Path)
    ap.add_argument("depth", type=pathlib.Path, help=".npy of the 480x270 float32 depth")
    ap.add_argument("--target", type=float, required=True, help="object depth in metres")
    ap.add_argument("--near", type=float, default=NEAR_K)
    ap.add_argument("--far", type=float, default=FAR_K)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/tmp/faded.png"))
    args = ap.parse_args()
    rgb = np.asarray(Image.open(args.colour).convert("RGB"))
    depth = np.load(args.depth)
    out = fade(rgb, depth, args.target, args.near, args.far)
    Image.fromarray(out).save(args.out)
    kept = float((out.sum(axis=2) > sum(BACKDROP) + 6).mean())
    print(f"wrote {args.out}  ({100 * kept:.1f}% of pixels kept)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
