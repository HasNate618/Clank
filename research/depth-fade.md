# Fading the background by depth

**Question.** The game is flat and bright - white badge on a white console in a room of white
signage - and the model gets confused by depth cues it cannot use. Does fading everything
substantially further than the target make grounding work on the full frame, so a crop is not
needed?

**Answer: partly, and not enough.** Full-frame grounding moved from **631 px to 363 px** of
error, against 60 px for the same call on a crop. The fade is worth keeping as a preprocessing
step and is not a crop substitute.

## The curve

`spike/wbc_game/fade.py`. Everything within `1.8x` the object's depth is untouched; past
`3.5x` only the backdrop remains; smoothstep between. A hard cut at the object's depth is
wrong because the depth buffer is torn often enough to bite into the object: one read of the
card returned 0.29 m and another 1.20 m a minute apart, and the layer's `far_m` has come back
as 0.0 on bad frames.

Invalid depth (0, non-finite, or past 12 m) is treated as far. That is the safe direction:
fading something close loses information the next frame returns, while keeping something far
is what confuses the model.

## The trap that cost a run

The object's depth must be the **nearest** surface in a patch, not the median. Measured at the
JobBot card: the 9x9 patch around its pixel had a **median of 1.31 m** and a **minimum of
0.38 m** - the card is a thin badge standing in front of a background that fills most of the
patch. Fading at 1.31 m keeps everything within 2.4 m, i.e. the entire room, and the fade does
nothing. Fading at 0.38 m keeps 38.7% of the frame and produces the image in
`research/evidence/fade-result.png`: the card, the monitor it leans on, and the console
survive; the room is gone.

## Numbers

| call | original | faded (target 0.378 m) |
| --- | --- | --- |
| `one-call` on the crop | 60 px | 60 px |
| `ground-full` | 630.9 px | **363.4 px** |

The cropped call is unaffected, which says the model was already resolving the object in a
crop and the background was not what limited it there. On the full frame the fade helps but
351 px of error remains - still a third of the frame.

## Why it is still worth building on

It is the only *automatic* crop-like step available: the depth already isolates the object
from everything behind it, and the crop that refusal needs (see the brain spec, §10) can come
from that mask rather than an operator click. What is missing is a mask that segments *near*
geometry from *the object*: the console and the monitor are as close as the card.

## Automatic crop from the depth mask: measured, and it does not work

The obvious next step was to use the same mask to generate the crop that refusal needs.
`fade.auto_crop()` builds a near-surface mask, takes the connected component containing the
object, and returns its bounding box.

**It returns the whole console.** The component containing the JobBot card is 910x350 px of a
960x540 frame - the card, the green monitor it leans on, the counter under both, and the
till to the left. They are all at 0.25-0.60 m, so depth cannot separate the object from the
furniture it is standing on. See `research/evidence/fade-result.png` and the box
`(50, 190, 960, 540)`.

So the crop still has to come from something that knows *which object*: the model's own
quadrant within a container, a template, or an operator click. Depth tells you what is near,
not what is interesting.

## The near-surface bug, twice

Both defects in this file are the same mistake at different scales: **a point sample of this
depth buffer is not evidence.**

- The fade's target taken as the *median* of a patch read 1.31 m and kept the whole room.
- The auto-crop's seed pixel read 1.31 m when the card is at 0.25-0.38 m.

The card is thin and the 480x270 raster misses it at its own pixel; the background behind it
wins. `nearest_surface()` now scans a small patch and takes the minimum valid depth, which is
the right statistic because the object of interest is the nearest thing in its own
neighbourhood.

## `spike/wbc_game/see.py`

The integration point. Captures the frame, finds the nearest surface at the object's pixel,
fades, and writes a JPEG: what the robot sees, with the room it will never touch removed.
Measured on the JobBot card: 0.249 m, 38.5% of pixels kept, and the card, monitor and console
survive with the signage gone.

That is a distraction fix, not an accuracy fix - full-frame grounding still sits at 236-363 px
against 39-65 px for a crop. But it costs one array pass and it removes the thing the model
keeps mistaking for the target.
