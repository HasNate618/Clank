# Locating an object with no human in the loop

**STATUS: MEASURED, 2026-11-04.** `spike/wbc_game/locate.py`. The grounding table in
`grounding-omni.md` was measured against **hand-made** crops: a human looked at the frame and
chose the box. That is not what the artifact claims. This is the attempt to remove the human.

**UPDATE 2026-11-09.** The pipeline below is the version measured on 2026-11-04, and its numbers
are left as they were recorded. Stage 2 has since become a **box ensemble read five times at
jittered framings**, and a background fade derived from the depth map runs before stage 1. Both
changes are in "Stage 2 since 2026-11-09" and in `locate.py`'s constants. Latency is now measured
by the tool itself - see "Latency".

## The pipeline

Three stages, each of which was already measured on its own:

1. **Quadrant.** One question about the whole frame: which quadrant holds the noun? (4-way was
   the only full-frame question measured correct: 44/44 on a crop, 9/12 on the frame.)
2. **Pixel.** Cut that quadrant (grown 15% so an object on the seam is not sliced), resize to
   960 wide, ask once for an exact pixel. This is the 39 px one-call measured earlier.
3. **Verify.** Cut a tight `+/-9%` box around stage 2's answer and ask `YES`/`NO`. Refusal was
   measured to be a property of the *tight* crop, so this is where "not there" is decided.

Cost: ~1.1-1.6 s, three calls, no grid, no offset table, no game layout. **That cost was measured
before the box ensemble and is now wrong by about 3x**: three calls at 0.25-0.4 s each plus a
0.4 s verify is closer to 1.5 s, and the shipped path makes seven calls. See "Latency".

### Stage 2 since 2026-11-09: a box ensemble, on a faded frame

A box ask beats a point ask **3.4x** on the same crop for the same single call (12.7-25.8 px vs
35.9-65.8 px), because "where is its centre" is ambiguous for anything with a printed face - the
barcode or the text? So stage 2 asks for a box, and `BOX_READS` jitters the framing five times
(send width 640-960 px, crop padding 0-4%) so the reads are independent.

The model is **bimodal**: the same question at a different send width flipped between 9 px and
384 px. One read is therefore a coin flip, and the answer is the **median of each edge** - per
edge, not per centre, so one wild read cannot drag it. On the lever, two of five reads landed on
the console and the spread was 181 px; the median still came out 5.9 px from truth.

Verification now crops to the model's **own box**, floored at `VERIFY_HALF` (0.16). Without the
floor a thin box (the lever's was 33x7 px) holds a grey stick with no knob in it and refuses an
answer that was 5.9 px from truth.

Fading first is what made the badge reliable: `fade.foreground_depth()` takes the fifth percentile
of valid depth with a 0.28 m floor (anything nearer is the robot's own arm, and during a reach the
percentile would otherwise follow the hand), and everything past **1.06-1.45x** that seed is faded.
The band is tight on purpose - the 1.8-3.5x default kept the console signage the model was
actually pointing at. On the badge: 235 px to 35 px.

Measured on a live frame with the fade: **card 21.0 px** (21 mm at 0.332 m), **lever 5.9 px**
(6 mm), and **32.3 px** on a 1705x961 still the pipeline was never tuned on. The honest bar is the
37 mm reach residual and the 90 mm grab radius, not an arbitrary 20 px.

### Stage 3 since 2026-11-09: ask what is in the box, then judge in code

The old stage 3 asked "is there a {noun} in this image?" and the file already recorded what
that costs: the answer is yes to *a* thing, not *the* thing. Asked with the robot brain's own
words, taken from JobBot's .srt lines, it was worse than that table suggests:

| JobBot said | brain named | old stage 3 | what was actually in the box |
| --- | --- | --- | --- |
| "take my card" | `card` | YES, 22.5 px | JobBot's ID badge - **right** |
| "grab one of the job cartridges" | `job cartridge` | **YES**, 22.2 px | JobBot's ID badge - **wrong, confirmed** |
| "grab one of the job cartridges" | `JobBot cartridge` | YES | ID badge - wrong, confirmed |
| (absent) | `banana` | **YES** | the brightest thing in frame |

Every one of those is within 2 px of every other, because the faded frame holds one near
object and the model answers whichever question it is asked about it. The spread did not move
either: 39.9 px for the right noun, 39.7 px for the wrong one. **The pipeline's own confidence
signals could not tell success from the confident-error class.**

Three prompt shapes were measured against that, on the same frame and two crops:

| change | measured effect |
| --- | --- |
| offer a refusal: `{"none": true}` when absent | **works**: banana 5/5 reads refuse. Without it the model never refused once |
| add a negative ("the white barcode badge is NOT a {noun}") | does not move badge/cartridge: 1/5 refusals |
| add the object's appearance ("a small plastic cartridge case with a printed label") | does not move it either |
| ask a forced choice, "(A) badge or (B) cartridge?" | useless: answered "A" for a lever |
| **ask what is in the crop, and judge the answer in code** | **works**: "Job Bot card", "JobBot Museum Curator card", "Blue mushroom" for the lever knob, on two frames |

The model can *say* what it sees and cannot *judge* whether that is what was asked for. So
stage 3 now asks `DESCRIBE_PROMPT` (six words, no JSON) and `describe_matches()` compares the
answer to the noun in this file: a stoplist for the summoner's own name, a job vocabulary of
synonyms (`card`/`badge`/`id`, `cartridge`/`cart`/`case`, `lever`/`knob`/`sphere`/`mushroom`),
and **no colour-only match**. That last rule came from the generalisation frame: a box
described as "green robot computer" was accepted as a weak match for `job cartridge` because
it shared a colour word, and pointed the robot at a green monitor. A colour may appear in a
description; it may not carry a verdict.

Results with the brain's real nouns, live 960x540 frame with the fade, `describe` mode:

| case | verdict | pixel error |
| --- | --- | --- |
| `card`, faded | accept | 22.5 px |
| `card`, unfaded | accept | 17.2 px |
| `job cartridge` | **refuse** | (box was the badge; described as a card) |
| `blue lever knob` | accept | 5.8 px |
| `banana` | **refuse** at stage 2, 5/5 | - |
| 1705x961 frame: `card` | accept | 23.5-33.1 px |
| 1705x961 frame: `job cartridge` | **refuse** | - |
| 1705x961 frame: `red fire extinguisher` | **refuse** at stage 1 | - |

**The conjunction now refuses instead of confidently pointing at the wrong object**, for the
cost of no extra calls: the describe replaces the yes/no, so a locate is still seven calls and
2.3-3.4 s. What it cannot do is find an object the model does not see: on this frame the job
cartridges on the desk are outside the fade band, so "not visible" is the correct answer and
is now the answer given. Whether the band should follow the target's depth rather than the
frame's nearest surface is the next question, not a solved one.

## Latency, measured 2026-11-09

Wall clock from `locate()` against the live server, on the 960x540 live frame with the fade, two
runs. `timings` is now part of the result, so this is reproducible with `--jsonl` instead of a
bespoke script.

| step | ms |
| --- | --- |
| fade (read still, numpy, encode) | 60-141 |
| stage 1, quadrant | 182-265 |
| stage 2, five box reads | 2000-2462 (398-522 each) |
| stage 3, verify | 184-214 |
| **total, five reads + verify** | **2.32-2.94 s** |
| three reads + verify | 1.51 s |
| five reads, no verify | 2.16 s |
| first call after idle (cold) | 0.30 s |

**A crop call is 0.40-0.52 s, not the 0.72 s this file used to imply.** That figure was carried
from the earlier grounding measurements and is wrong for this server and these crops; the code now
reports its own clock so the number cannot silently rot again.

Stage 2 is **86% of the cost**, and it is the part that buys the accuracy on bimodal replies.
Three reads cost 1.51 s and landed 20.1 px on the measured frame against 22.4 px for five - inside
the reach residual either way - but the lever is why five exists: two of its five reads were wild,
and with three reads two wild ones would be the majority. Depth already replaces some of what
stage 1 does for free; using it to cut stage 1 outright is the next honest saving, worth 0.2 s and
a tighter crop.

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
