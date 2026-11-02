# The glow: confirmed on a real grab, and why it cannot be found by statistics

`[OPERATOR 2026-11-01]`: in `Job Simulator VR FULL WALKTHROUGH ... .mp4`, at
**36.783 s** the player reaches for a cartridge and it takes on the blue glow.
`[MEASURED 2026-11-01]` from the frames, by extracting them and looking.

Frames kept in `research/evidence/` (the 24 MB video is not tracked):

| t | state |
| --- | --- |
| 36.500 s | hand approaching, **no glow**, cartridge plain |
| 36.783 s | hand on the cartridge, **glow on** - a large saturated cyan region |
| 36.950 s | cartridge grabbed, hand and glow both gone because it is now held |

So the cue is real, and it is **pre-grab only**: it marks "in range, not yet
grabbed", and it disappears the moment the object is held. That matters for
`reach_outcome`: glow present means *arrived*, never *grabbed*.

## The glow, as pixels

The cartridge box at its 36.783 location, `(1060, 550)-(1240, 700)`:

| frame | cyan px | bright px | cyan mean RGB |
| --- | ---: | ---: | --- |
| 36.500 off | 3904 (14.5%) | 18.6% | 41, 198, 210 |
| 36.783 **on** | 3969 (14.7%) | **54.0%** | 55, 213, 217 |
| 36.950 held | 1697 (6.3%) | 13.4% | 57, 209, 220 |

Two things are already visible here. The **cyan fraction does not move** - 14.5% to
14.7% - because the crate's own label is cyan whether or not anything glows. And the
**bright fraction that does move is contaminated by the hand**: the hand is white, so
it scores as "bright" exactly when it arrives. Neither number is a detector.

## No whole-frame statistic separates it

The obvious next try is a contiguous cyan region, since the glow is a solid area rather
than thin text. Total cyan-dominant pixels in each frame, and the largest connected
blobs (downsampled by 4, area reported at full resolution):

| frame | total cyan px | largest blob |
| --- | ---: | --- |
| 36.500 off | 8346 | 48336 px, x 1052-1916 y 508-988 |
| 36.783 on | 8567 | 29680 px, x 1452-1916 y 748-1076 |
| 36.950 held | 8646 | 25840 px, x 1544-1916 y 776-1076 |

The totals are **within 4% of each other across all three frames**, and the largest
blob in every one of them is floor strip or signage, not the glow. The glow object is
not the biggest cyan thing in its own frame.

**Conclusion: blue is this game's decorative palette.** Pedestals, painted slots,
"Calibrate / Options" signs, stanchion lights and floor strips are all the same hue as
the cue. Any detector built on cyan statistics over a frame is measuring the room.

## What this means for the design

The whole-frame approach is dead, and the module was already half right about why:
`clank/glow.py` insists on a **baseline diff** rather than a colour gate. But this pair
now shows the baseline diff's own weakness - **the glow and the hand arrive together**,
so a brightness delta around the object is dominated by the white hand. Of 1144 pixels
at delta 255 between the off and on frames, **1130 were in one blob**: the hand.

The signal that survives is *location*. The glow is cyan, and it is near the hand. We
know where the hand is - `body_state` gives the wrist and `project_point` gives its
pixel, which is validated to within a hand width on a real render
(`research/camera-equals-hmd-pose.md` section 8). The walkthrough does not have that
ground truth; our capture does.

So the detector is **cyan-fraction in a patch around the wrist's projected pixel**, not
a fixed-pixel baseline over the frame. And its threshold has to be calibrated in *our*
renderer, where the wrist pixel is known - lifting a number out of a video with a
different camera, different resolution and unknown hand pose would be a number that
happens to fit one frame.

## Caveat

The camera moves between these frames, so even the in-box colour comparison above is
not a clean threshold: the box holds different scenery at 36.500 and 36.783. What the
pair establishes is the *existence, appearance and timing* of the cue. It does not
establish a working threshold, and no threshold should be quoted from it.
