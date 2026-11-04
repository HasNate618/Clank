"""Follow a marked object in the game view, and say whether it is glowing.

The reach kept losing its target because the head will not hold a commanded pose: held
against one target for 10 s, the achieved head wobbled **11.7 deg of yaw** and sat ~16 deg
off the command, so a pixel marked once names something else 300 ms later (measured: the
card swung 1009 -> 1122 -> 865 -> 1007 px inside a single four-command reach). Fighting
that with a stronger pin does not work - the pin is a request the policy answers with
something else. So the pixel is re-derived every tick instead.

Three stages, and the expensive one is not in the loop:

* **seed** - the brain names the noun, `clank.brain.ground` says which quadrant of a crop,
  `compose_pixel` turns that into a coarse pixel. Runs once per instruction (44/44 on the
  tray; the same prompt on the whole frame scored 9/12, so it needs the crop).
* **refine** - a coarse quadrant is ~1/4 of a crop, far wider than a local search can
  cover, so the seed is refined once by searching for the object's own signature.
* **follow** - the object's world point is fixed and the camera pose is measured every
  tick, so the pixel is *predicted* rather than searched for.

## Why following is prediction and not matching

The object's appearance changes underneath the tracker: a grabbable card goes from white
`(153, 166, 169)` to emissive cyan `(130, 219, 219)` between one frame and the next. There
is no template that survives that, and the prediction is accurate - the hand's pixel,
predicted from its own 3D position, landed **18 px** from where the game drew it.

## The glow is not a validated instrument, and that is the finding

The glow is the game's own "you may grab this", and on hand-picked patches it separates
cleanly: `min(G - R, B - R)` of 97-108 glowing against 5.9-40.4 not. Every one of those
numbers came from a box drawn by hand, and when the patch is placed automatically the
separation does not survive:

| sample | reading |
| --- | --- |
| fixed 48x48 patch at the predicted pixel, 60 px off the glow | 59-93 |
| fixed 48x48 patch on the glow, same frame | 144-168 |
| max over a +-45 px window, card glowing, hand 10 cm away | 164-176 |
| **same window max, on a saved frame with the hand away** | **153.5** |
| **same window max, on a saved frame with the hand at the card** | **124.8** |

The last two are the point: the window max is not selective, because the room is painted in
the same hue - walls, counter strips, the JOB CART sign - and the brightest cyan in a +-45 px
window is often scenery. A single frame cannot tell a glowing card from a glowing wall.

So the detector is **not** trustworthy enough to gate an action, and it should not gate one.
What is trustworthy is the reach's own measured residual: the policy puts the hand where it
is told to 2-3 cm inside the envelope, and that number is measured every command against the
same geometry the command was built from.

Use the glow for what it is good for - an operator-visible confirmation that the game
considers the object grabbable - and calibrate the grab radius by *trying the grab* (close the
hand and see whether the object moves) rather than by watching for a glow. `glow_near()` is
kept as the reporting helper, not as a decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from clank.projection import Fov, project_point

#: Half-size of the patch the glow statistic is taken over, in colour pixels. The card is
#: ~90 px wide at the counter, so a 24 px half-window stays on it while the arm passes.
PATCH_HALF = 24

#: Half-size of the window `glow_near` searches. A prediction's error is ~60 px, so this
#: has to cover it; it is still small enough that the room's cyan signage outside it cannot
#: win the max.
SEARCH_PX = 45

#: `mean(min(G - R, B - R))` above which a patch *is* glowing, when there is no baseline
#: to compare against: 89-97 measured glowing, 5.9-40.4 not.
GLOW_ON = 60.0

#: Change in that score at a fixed patch that counts as the glow switching state.
GLOW_STEP = 25.0


@dataclass(frozen=True)
class Patch:
    """One patch's colour statistics. `glowing` is the game's own "grabbable" answer."""

    pixel: tuple[float, float]
    red: float
    green: float
    blue: float
    glow: float

    @property
    def glowing(self) -> bool:
        return self.glow >= GLOW_ON

    def to_dict(self) -> dict[str, object]:
        return {
            "pixel": (round(self.pixel[0], 1), round(self.pixel[1], 1)),
            "red": round(self.red, 1),
            "green": round(self.green, 1),
            "blue": round(self.blue, 1),
            "glow": round(self.glow, 1),
            "glowing": self.glowing,
        }


def _clip(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def glow_near(
    rgb: np.ndarray, x: float, y: float, radius: int = SEARCH_PX, half: int = PATCH_HALF
) -> tuple[float, tuple[float, float]]:
    """The strongest glow within `radius` of `(x, y)`, and where it is.

    A prediction can be 60 px off - that is the measured spread of the projection against
    the game's own drawing - and a 48 px patch missing the glow by that much reads 59-93
    where the glow itself reads 164-176. Taking the max over the window removes that
    dependency without letting the search wander into the room's own cyan signage.
    """
    height, width = rgb.shape[:2]
    step = max(3, half // 3)
    best: tuple[float, float, float] = (-1.0, x, y)
    for dy in range(-radius, radius + 1, step):
        for dx in range(-radius, radius + 1, step):
            cx, cy = x + dx, y + dy
            if not (0 <= cx < width and 0 <= cy < height):
                continue
            score = patch_at(rgb, cx, cy, half).glow
            if score > best[0]:
                best = (score, cx, cy)
    return best[0], (best[1], best[2])


def patch_at(rgb: np.ndarray, x: float, y: float, half: int = PATCH_HALF) -> Patch:
    """Colour statistics of the patch centred on `(x, y)`.

    `rgb` is an `(h, w, 3)` array. The window is clipped to the frame rather than rejected,
    so a patch on the edge still answers with the pixels it has: the card sat against the
    right edge in every frame this was measured on.
    """
    height, width = rgb.shape[:2]
    x0, x1 = _clip(int(x) - half, 0, width), _clip(int(x) + half + 1, 0, width)
    y0, y1 = _clip(int(y) - half, 0, height), _clip(int(y) + half + 1, 0, height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"patch at ({x:.0f}, {y:.0f}) is outside the {width}x{height} frame")
    window = rgb[y0:y1, x0:x1].astype(np.float64)
    red = float(window[..., 0].mean())
    green = float(window[..., 1].mean())
    blue = float(window[..., 2].mean())
    # Both channels must clear red, and the weaker of the two decides: the glow is cyan, so
    # a blue-only or green-only patch (the wall, a floor strip) must not score.
    glow = float(np.minimum(green - red, blue - red).mean())
    return Patch(pixel=(float(x), float(y)), red=red, green=green, blue=blue, glow=glow)


@dataclass
class GlowWatch:
    """Tracks the glow state of one patch, and records when it changes.

    The state is the **absolute** `G - R` against `GLOW_ON`, not a change from a baseline,
    and that is a measurement rather than a preference: the same card, not glowing, scored
    5.9 seen one way and 40.4 seen another, while glowing it scored 96.7 and 108.0. A
    baseline-plus-delta rule fired "on" on the 40.4 frame; the absolute threshold separates
    every sample with room on both sides. `baseline` is kept for the report - the jump from
    it is what a human reads as "the glow lit" - but it does not decide the state.

    `occluded` is a real state, not an error: the hand that is reaching for the object
    covers it, and calling that "not glowing" would report a failed grab at the exact
    moment the grab is happening. The caller knows where the hand is, so it can say so
    rather than guess from the pixels.
    """

    baseline: float | None = None
    state: str = "unknown"
    log: list[tuple[int, str, float]] = field(default_factory=list)
    _ticks: int = 0

    def update(self, patch: Patch, occluded: bool = False) -> str:
        self._ticks += 1
        if self.baseline is None:
            self.baseline = patch.glow
        if occluded:
            new = "occluded"
        else:
            new = "on" if patch.glowing else "off"
        if new != self.state:
            self.log.append((self._ticks, new, patch.glow))
            self.state = new
        return self.state

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": None if self.baseline is None else round(self.baseline, 1),
            "state": self.state,
            "transitions": [
                {"tick": t, "state": s, "glow": round(g, 1)} for t, s, g in self.log
            ],
        }


@dataclass(frozen=True)
class Track:
    """Where the object is this tick, and whether the game says it is grabbable."""

    predicted: tuple[float, float] | None
    patch: Patch
    moved_px: float

    def to_dict(self) -> dict[str, object]:
        return {
            "predicted": None
            if self.predicted is None
            else (round(self.predicted[0], 1), round(self.predicted[1], 1)),
            "moved_px": round(self.moved_px, 1),
            **self.patch.to_dict(),
        }


def predict(
    head_p: tuple[float, float, float],
    head_q: tuple[float, float, float, float],
    point_rel: tuple[float, float, float],
    fov: Fov,
    size: tuple[int, int],
) -> tuple[float, float] | None:
    """The pixel the object *should* be at, from the measured head pose.

    This is what makes following possible without a model: the object does not move, the
    camera does, and the camera is measured every tick. Returns None when the point is
    behind the image plane, which callers must treat as "no answer", not as the centre.
    """
    from clank.body import Pose

    uv = project_point(Pose(p=head_p, q_xyzw=head_q), point_rel, fov)
    if uv is None:
        return None
    width, height = size
    return (uv[0] * width, uv[1] * height)


def follow(
    rgb: np.ndarray,
    point_rel: tuple[float, float, float],
    head_p: tuple[float, float, float],
    head_q: tuple[float, float, float, float],
    fov: Fov,
    previous: tuple[float, float] | None = None,
) -> Track:
    """Sample the object's patch at the predicted pixel for this tick.

    When the prediction fails - the head has swung so far the object leaves the frame -
    `previous` stands, so a reach that has already locked on keeps a pixel rather than
    losing the object entirely.
    """
    size = (rgb.shape[1], rgb.shape[0])
    predicted = predict(head_p, head_q, point_rel, fov, size)
    anchor = predicted if predicted is not None else previous
    if anchor is None:
        raise ValueError("no prediction and no previous pixel: nothing to follow")
    patch = patch_at(rgb, anchor[0], anchor[1])
    moved = 0.0 if previous is None else math.dist(previous, patch.pixel)
    return Track(predicted=predicted, patch=patch, moved_px=moved)
