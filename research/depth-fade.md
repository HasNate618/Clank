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
