# Locating an object with no human in the loop

**STATUS: MEASURED, 2026-11-04.** `spike/wbc_game/locate.py`. The grounding table in
`grounding-omni.md` was measured against **hand-made** crops: a human looked at the frame and
chose the box. That is not what the artifact claims. This is the attempt to remove the human.

## The pipeline

Three stages, each of which was already measured on its own:

1. **Quadrant.** One question about the whole frame: which quadrant holds the noun? (4-way was
   the only full-frame question measured correct: 44/44 on a crop, 9/12 on the frame.)
2. **Pixel.** Cut that quadrant (grown 15% so an object on the seam is not sliced), resize to
   960 wide, ask once for an exact pixel. This is the 39 px one-call measured earlier.
3. **Verify.** Cut a tight `+/-9%` box around stage 2's answer and ask `YES`/`NO`. Refusal was
   measured to be a property of the *tight* crop, so this is where "not there" is decided.

Cost: ~1.1-1.6 s, three calls, no grid, no offset table, no game layout.

## Results

Truth pixel marked by a human once, for scoring only.

| frame | noun | quadrant | answer px | error | verify | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| seed-view 960x540 | JobBot ID card | BOTTOM-RIGHT | 707,437 | **17.4** | YES | **good** |
| mark 960x540 | JobBot ID card | BOTTOM-RIGHT | 742,413 | **3.0** | YES | **good** |
| grab_before 1920x1080 | JobBot ID card | BOTTOM-LEFT | 826,773 | 996 | YES | **confident error** |
| grab_before, resized to 960 | JobBot ID card | BOTTOM-RIGHT | 632,304 | 326 | NO | refused |
| seed-view | red fire extinguisher | BOTTOM-RIGHT | 759,404 | - | NO | **refused** |
| mark | red fire extinguisher | (none) | - | - | - | **refused at stage 1** |
| live, card not in frame | JobBot ID card | (none) | - | - | - | **refused at stage 1** |
| mark | GOURMET CHEF cartridge | BOTTOM-RIGHT | 891,437 | 66.5 | YES | **wrong by 66 px** |
| mark | AUTO MECHANIC cartridge | TOP-RIGHT | 594,39 | 369 | YES | **confident error** |
| live capture, card fallen out of frame | JobBot ID card | BOTTOM-LEFT | 356,341 | - | YES | **confident error** |

**7 correct-or-safe, 3 confident errors.** The card is located to 3-17 px with nothing
hand-marked. The cartridge rack defeats it: small, stacked, near-identical objects, and stage 3
verifies the wrong one because a cartridge *is* in the crop it looked at - it just is not the
cartridge asked for.

## Two things learned about the model, by measurement

**The crop is normalised to a square, width-based grid.** A quadrant of a 960x540 frame becomes
960x483, so dividing y by 483 looked obviously right. Measured against three truth pixels:
dividing both axes by 960 gave 4.7 and 17.4 px; dividing y by the crop's own height gave 136 and
163 px. The implied divisor from known points is 968 and 1089 - about 960. The model ignores
aspect ratio; do not "correct" it.

**Stage 3 catches pixel errors but not identity errors.** When stage 2's pixel was wrong, the
tight crop missed the object and stage 3 said NO - a fail-safe. When the *wrong object* was in
the crop, stage 3 said YES. So the guard covers position, not naming. Naming is what PLAN is for.

## The live test, which it fails

`find_and_reach.py --dry-run` on a live capture, with the card fallen out of the frame, returned
`BOTTOM-LEFT (356, 341)`, `present: true`. There is no card there. Some white rectangle in the
console satisfied stage 3, so the pipeline would have reached at nothing.

That is the whole difficulty in one case: **stage 3 answers "is *a* thing here", not "is *the*
thing here"**. On the seeded frame the noun names something unmistakable (a badge with "JobBot"
printed on it), and it works to 3-17 px. On a scene it was not measured on, it invents.

So: **the hand-marked pixel is replaced only for the seeded frame.** That is enough to run the
demo without me framing it by hand, and it is not enough to call the problem solved. Two honest
ways forward, in order of how much they buy per unit of work:

1. **Restore the scene** before a run (level restart), so the frame is the one that was measured.
   Cheap, and it is what a demo does anyway.
2. **A real presence check.** Not another model question - a remembered background. The game's
   own cue was tried and invalidated (`glow-calibration.md`), but a still-life subtraction is
   deterministic, and the card is the thing that is not there when it is not there.

## The verifier was wrong in both directions

`VERIFY_HALF` was 0.09 of the frame. Measured 2026-11-04, live:

- a pixel **dead centre on the GOURMET CHEF cartridge label** was **refused** - a crop that tight
  holds one word of a label and reads as a fragment, not as a cartridge;
- with verification switched off, the next run put "AUTO MECHANIC cartridge" at (367, 326), on the
  robot's *left*, 0.83 m away, and nothing objected;
- with `VERIFY_HALF = 0.16` the same "GOURMET CHEF cartridge" question returned **(364, 341)** -
  still the wrong side of the frame - and the verifier said **YES**.

So the crop size fixed the false refusal and changed nothing about the confident error: on a rack
of near-identical cartridges the model lands on the wrong one and confirms it. Verification checks
*presence*, and a cartridge-shaped thing is present wherever it looks.

**The rack needs a different kind of answer than another model question** - the objects differ by a
few words of text and by position within a known tower. That is a template problem, not a
grounding problem.

## What this changes

The card path no longer needs a human to frame it: `locate.py` produces the pixel that
`reach_vr.py` consumes, at 3-17 px, and refuses rather than guessing when the card is not
visible. That is enough for the seeded demo and not enough to be called a detector.
