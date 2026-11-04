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


def near_mask(
    depth: np.ndarray,
    target_m: float,
    near: float = 0.75,
    far: float = 2.4,
) -> np.ndarray:
    """Boolean mask of surfaces plausibly at the object's own distance.

    Used to build a crop without an operator click. The band *starts below* the object
    (0.75x) rather than above it: the object is the nearest thing in its own neighbourhood,
    so the useful question is "what is behind it and still part of the scene", and the band
    stops at 2.4x - the fade's 3.5x is for a backdrop, this is for a crop, and everything a
    crop shows the model will try to interpret. Invalid depth is excluded, not
    treated as far: unlike the fade, a forgotten sample here would put the room back in the
    crop.
    """
    d = np.asarray(depth, dtype=np.float64)
    ok = np.isfinite(d) & (d > 0.0) & (d < FAR_SENTINEL)
    return ok & (d >= near * target_m) & (d <= far * target_m)


def component_at(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    """The 4-connected component of `mask` containing `seed`.

    A flood fill, not a library call: the mask is one object plus whatever it touches, and
    pulling in scipy for that would be a heavier dependency than the eighteen lines below.
    """
    height, width = mask.shape
    sx, sy = seed
    if not (0 <= sx < width and 0 <= sy < height) or not mask[sy, sx]:
        return np.zeros_like(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    stack = [(sx, sy)]
    while stack:
        x, y = stack.pop()
        if out[y, x] or not mask[y, x]:
            continue
        out[y, x] = True
        if x > 0:
            stack.append((x - 1, y))
        if x + 1 < width:
            stack.append((x + 1, y))
        if y > 0:
            stack.append((x, y - 1))
        if y + 1 < height:
            stack.append((x, y + 1))
    return out


def nearest_surface(
    depth: np.ndarray, seed_px: tuple[float, float], radius: int = 8
) -> tuple[float, tuple[int, int]] | None:
    """The nearest valid depth near a seed, and where it is.

    A single sample is not enough and this is measured, not defensive: the JobBot card's
    own pixel read **1.31 m** while the card is at **0.38 m**, because the card is thin and
    the 480x270 raster misses it at that pixel - the background behind it wins. The object
    is the nearest thing in its own neighbourhood, so take the minimum.
    """
    dh, dw = depth.shape[:2]
    sx = int(seed_px[0] * dw / (seed_px[2] if len(seed_px) > 2 else dw))
    sy = int(seed_px[1] * dh / (seed_px[3] if len(seed_px) > 3 else dh))
    x0, x1 = max(0, sx - radius), min(dw, sx + radius + 1)
    y0, y1 = max(0, sy - radius), min(dh, sy + radius + 1)
    patch = np.asarray(depth[y0:y1, x0:x1], dtype=np.float64)
    valid = np.isfinite(patch) & (patch > 0.0) & (patch < FAR_SENTINEL)
    if not valid.any():
        return None
    best = np.argmin(np.where(valid, patch, np.inf))
    by, bx = divmod(int(best), patch.shape[1])
    return float(patch[by, bx]), (x0 + bx, y0 + by)


def auto_crop(
    rgb: np.ndarray,
    depth: np.ndarray,
    target_m: float,
    seed_px: tuple[float, float],
    pad: float = 0.15,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """A crop around the object's own connected near surface, as `(image, box)`.

    `seed_px` is in **colour** pixels; the depth mask is at the buffer's resolution and is
    indexed with the scaled coordinate. The box is padded by `pad` of its own size, because
    a crop that cuts the object's edge teaches the model that the object ends there.
    """
    height, width = rgb.shape[:2]
    dh, dw = depth.shape[:2]
    sx = int(seed_px[0] * dw / width)
    sy = int(seed_px[1] * dh / height)
    mask = near_mask(depth, target_m)
    if not mask[sy, sx]:
        # The seed's own sample is the background behind a thin object; take the nearest
        # surface in its neighbourhood, which is the object.
        found = nearest_surface(depth, (seed_px[0], seed_px[1], width, height))
        if found is not None:
            target_m, (sx, sy) = found
            mask = near_mask(depth, target_m)
    blob = component_at(mask, (sx, sy))
    if not blob.any():
        raise ValueError(f"seed ({sx}, {sy}) is not on a near surface at {target_m:.2f} m")
    ys, xs = np.nonzero(blob)
    x0, x1 = xs.min() * width / dw, (xs.max() + 1) * width / dw
    y0, y1 = ys.min() * height / dh, (ys.max() + 1) * height / dh
    w, h = x1 - x0, y1 - y0
    x0, x1 = max(0, int(x0 - pad * w)), min(width, int(x1 + pad * w))
    y0, y1 = max(0, int(y0 - pad * h)), min(height, int(y1 + pad * h))
    return rgb[y0:y1, x0:x1], (x0, y0, x1, y1)
