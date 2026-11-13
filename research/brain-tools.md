# Can the model choose a sense tool?

**STATUS: MEASURED, 2026-11-10.** LFM3-Omni-2B on `:11434`, temperature 0, one 960x540 still per
call, ~210 ms prompt + ~203 ms decode. Harness: `spike/wbc_game/eval_plan.py` (not collected by
`python3 -m unittest`; it talks to the model server).

The question the loop rewrite turns on: when JobBot says "take my card", does the model *choose*
`locate`, or does the choreography have to live in Python?

## The gate, and it fails

| Constitution | Seed | Still | Reply |
| --- | --- | --- | --- |
| base (`locate` offered) | the default choreography seed | frame-173133 | `release right "JobBot MacroSimulator cartridge"` |
| base | choreography | frame-172830 | `release right left` |
| base | choreography | grounding-seed-view | `release right "PULL LEVER TO JOB"` |
| base | none | frame-173133 | `hold right target=card` |
| base | neutral | frame-173133 | `grab target=card` |
| `locate`→`find` | neutral | frame-173133 | **`find target=card`** |
| `locate`→`find` | neutral | frame-172830 | `grab target=card` |
| `locate`→`find` | neutral | grounding-seed-view | `grab target=card` |

So the sense tool is chosen **1 of 3 stills**, and `locate` as a word is never chosen at all.

## What the failure actually is

**1. The seed dominates the speech.** `DEFAULT_SEED` still carries the walkthrough choreography
("use the store clerk cartridge (green label) ... pull the lever down to start the job"). With that
seed the reply is `release ... cartridge` on every still and every wording; with no seed it becomes
`hold`/`grab` of the *card* the voice actually named. Removing that seed is worth more than any
constitution edit tested here.

**2. The model copies the constitution instead of selecting from it.** Put `release` first in the
enum and it answers `"skill":"release"|` — the list's own text, pipe included. Replace the enum
with prose (a placeholder instead of the member list) and it answers
`"skill": "release open one hand"`, a verbatim list line. Two of the replies in the table above
(`"skill":"reach"|`) are the same artifact. This is the same collapse recorded on 2026-11-04 in
`research/next-step.md`, now visible as copying.

**3. A `not_located` last line does not correct it.** The mechanism the design leans on - refuse,
tell the model why, let it choose differently - measures backwards:

| last line | still | reply |
| --- | --- | --- |
| `(none)` | frame-173133 (find constitution) | `find target=card` |
| `reach right target=card -> not_located` | frame-173133 | `reach right card` |
| `reach right target=card -> not_located` | frame-172830 | `reach right card` |
| `reach right target=card -> not_located` | grounding-seed-view | `reach right card` |

3/3 it re-issues the refused action. Adding the policy to the constitution in words ("if your last
action was refused because you had not found the thing, your next action is find it") changes
nothing: 3/3 `reach` again, still with the corruption.

## What the model does do

- It reads the **object** off JobBot's sentence correctly every time it names a motor: `card`, on
  all six motor replies.
- It follows direct instructions about which body part and which sense: "look to the left" →
  `{"look":"left"}`; "hold still and wait" → `hold`.
- Its prompt is honoured: a hostile constitution ("always answer hold") got `hold`.

## Consequence

The plan's assumption - *the model chooses `locate`, and learns from `not_located`* - is false on
this model. Two ways forward, and the choice is a design decision, not a bug fix:

- **A. The model chooses the object and the motor; code resolves where.** `grab(card)` is the
  model's decision; `locate` is how the body finds out where the card is. This is the 2026-10-31
  spec's shape (PLAN names a noun, GROUND resolves it) and is still a chooser - the model picks the
  noun from the audio and the still, which is what `task_intent` replaced with a regex.
- **B. Keep the sense tool a choice and accept 1/3.** Then the loop needs a rule for the other 2/3,
  which is A wearing a longer coat.

Both need the choreography seed deleted first: it is the single largest cause of the wrong verb.

## The question shape wins, and it is the whole result

Same four live stills, same instruction, same endpoint, two question shapes. **The model is
competent at saying what it wants and incompetent at picking from a list.**

| asked | replies |
| --- | --- |
| "pick one of hold/look/find/reach/grab/release" | `hold`, `release`, `release`, `look` — **0 grabs in 4** |
| "what are you trying to do next?" | `"take the JobBot card"`, `"take the JobBot card"`, `"take the JobBot card"`, `"take the card"` — **4 of 4 correct** |

So the live chooser asks for the goal in words and reads the body's vocabulary out of *the model's
own sentence* (`skill_from_goal`). That is not the old parser: `task_intent` read **JobBot's**
instruction, which made the model decoration. Here the model reads the instruction *and the still*
and states its decision; the mapping only decodes it into a skill the adapter can compile.

Two smaller findings from the same run:

- **The `object` field is not the noun.** The goal was "take the JobBot card" while `object` said
  `"JobBot"`, and locate went looking for the summoner. The noun now comes from the goal sentence,
  which gives `jobbot card` - and locate's stoplist strips `jobbot`, so it matches the card.
- **"menu" in the action list costs the card task its answer.** With the menu action offered, "take
  my card" came back `menu(card)`; without it, `look`/`grab`. One extra action captures an unrelated
  instruction on this model, so it is not in the list at all. The menu button is reached by the
  goal phrase ("press the menu button" -> `menu`) and by the dashboard button.

## Live, 2026-11-10

The whole chain, with the line typed so the game's own menu could stay where it was:

```text
heard: Hey, Hillman, take my card.
think: plan: grab(jobbot card) (side=right)
act:   locate(jobbot card)   -> present, 0.37 m
result: grab(card)           -> reach saturated 107 mm short, hand closed
operator: the card was grabbed and JobBot spoke
```

The reach still ends ~100 mm short and the game grabs anyway: the hand closes inside the stream at
the nearest surface the depth buffer reports, and the grab radius is the game's, not the arm's. The
diary called it "failed (code=2)" because the exit code is the arm's and the confirmation was
skipped under `--say`; confirmation now runs every time, because JobBot reacts to the game and not
to how the instruction arrived.

## The five box reads on the server's four slots, 2026-11-13

`read_box` walked its five stage-2 reads in a `for` loop while the llama.cpp server was already
serving four requests at once (`/props` reports `total_slots: 4`, and the server log cycles slots
0-3). The reads are independent - five different crops of the same quadrant - so they belong on
the slots.

Measured on `frames/frame-190251.png`, noun `card`, real server, three runs each, one warm-up:

| path | `timings.total_ms` | `reads_ms` | `pixel_px` |
| --- | --- | --- | --- |
| serial | 2395.8 / 2367.4 / 2379.1 | 1977.8 | 797.4, 450.5 (once) and 788.9, 450.5 |
| parallel | 1540.5 / 1569.3 / 1557.8 | 1136.3 | 797.4, 450.5 (3/3) |

**-34.5 % of locate's wall clock, -42.5 % of the reads stage.** Two waves of four-plus-one, as
predicted. Per-read wall time goes *up* under contention (672-724 ms against 363-424 ms) because
the reads share slots, which is why `locate()` keeps timing the stage separately from the reads.

Two honest limits:

- **Temperature 0 does not make the server reproducible across slots.** Both paths flip between
  x 788.9 and 797.4 and between y 443.1 and 450.5 from run to run, which is the same jitter the
  ensemble already reports as a 42-49 px read spread - an order of magnitude larger than the
  centre difference. So "the pixel is unchanged" means *congruent within the model's own jitter*,
  not identical.
- **This buys latency, not compute.** The same tokens are generated. It is a win only while the
  slots are otherwise idle; a second client on `:11434` (a dashboard, a probe) halves it.

Not a win elsewhere: quadrant -> box -> describe is sequential by construction, and so is the
planner chain now that session memory feeds each call the previous reply.

The whole pipeline on the parallel path, measured the same day, three runs: **1579-1593 ms** total
(quadrant ~200 ms, reads 1186-1198 ms, describe check 192-205 ms), so the reads-plus-check figure
the README quotes is **1.38-1.40 s** rather than the 2.32-2.94 s it used to be.

## 2026-11-13 — run `190131-679de2`: what the diary could not explain

Nine model calls: two planners and seven for one `locate("card")`. The card was grabbed and JobBot
spoke. Three things went wrong in ways the diary of the time could not show, and each is now a row
that makes it a one-line diagnosis.

**The instruction was spoken while the loop was not listening.** Transcribed from the recording's own
audio: video 12.4-18.4 s is **"Here's my card."** `hang_s = 0.8` ended the introduction at "…fine job
museum.", and `ready()` then spent 5.8 s standing the body up. Measured gaps in that audio: **0.60 s
inside the opening** and **6.3-6.9 s between turns** — so a 0.8 s pause was never going to split it,
and the line was lost to a capture dropout rather than to the pause rule. The fix is not a longer
pause: JobBot's reply after the grab is now captured by the confirmation and handed to the next turn
as the instruction.

**The confirmation was gated off its own question.** `ears.listen` measures its floor as the median of
the first 0.5 s; on that capture the floor was **2445**, so the gate came out at **9781** against a
**peak of 4415** and the line read as silence. Transcribed directly, the same clip gives the words in
**183 ms**. `confirm()` now records a fixed window and transcribes it, with no gate.

**A greeting became a chore.** The prompt listed the props and said the console is where they go, so
"welcome to the job simulator, my name is JobBot" produced
`place(job cartridge into console)`. `assemble` refused it (`place needs a side`) and the loop held,
but the plan was the prompt's invention. With the prop list gone and an utterance that names no task
gated before the model, the same still makes no call at all.

Replayed at temperature 0 against the saved stills: greeting -> no call, `hold`; "Hey, Gilman, take my
card." -> goal `take the card`, object `JobBot card`, skill `grab`; "press the menu button" -> goal
`open the menu`, skill `menu`. Deleting the prop list did not cost the model its nouns.

## 2026-11-13 — the session, measured against the live endpoint

Two planner calls, the same saved still, one `SessionMemory`, `LFM3-Omni-2B-Q8_0` on `:11434`,
temperature 0:

| call | heard | answer | history sent |
| --- | --- | --- | --- |
| 1 | "Hey, Gilman, take my card." | `grab` / `card` | 0 earlier messages |
| 2 | "press the menu button" | `menu` / `menu button` | 2 earlier messages, ~49 tokens |

What the second call sent, verbatim: the first turn's situation, the first raw reply, then the new
situation — no instruction boilerplate, because the history stores the situation and the template
stays on the newest message only. After two turns the history is **~96 estimated tokens**, so the
~8k budget is roughly 160 turns of real use rather than the 65 the pair count would have allowed.

Both answers are the right act for the sentence, which is the point: the model was told "press the
menu button" *after* a card grab was in its history, and it did not repeat the grab.

## 2026-11-13 — what the grounder is shown, and what the brain's own description is worth

Run `113127-dfcacc` could not find JobBot's card: quadrant `BOTTOM-LEFT` (the robot's own glove),
stage 3 refusing `OFF button`, `Control panel`, `Sign`. Three experiments on that frame, three reads
each, temperature 0.

**1. The fade band is the whole difference, and the module's default band is wrong for this view.**

| still given to the grounder | quadrant | pixel | stage 3 | result |
| --- | --- | --- | --- | --- |
| raw | BOTTOM-LEFT x3 | 346 | `Sign`, `Floppy disk` | refused |
| faded, module default (1.06-1.45 x nearest clutter = 0.35-0.48 m) | BOTTOM-LEFT x3 | 339 | `Switch` | refused |
| faded, the depth gate's own band (0.28-1.00 m) | **BOTTOM-RIGHT x3** | 888 | `Job Bot sign` | refused |

The default band blackens everything past **0.48 m**; the card sits at **0.50 m** from the camera and
JobBot's body nearer still, so the fade deleted the object and kept its distractor. Measured depth:
card region nearest 0.25 m / median 0.30 m, glove region 0.31-0.37 m, p5 reference 0.33 m. A fade
whose near edge is the nearest clutter is a fade that removes whatever is *behind* that clutter, and
in this game the thing you want is often behind it.

So the grounder now gets the gate's band, and the band is one constant pair rather than four numbers
in two modules (`GROUNDING_BAND_M` in `auto.py`).

**2. That fixes stage 1 only.** Stage 2 then puts the box on **JobBot's body** ~100 px from the card,
and stage 3 refuses at every crop size: 225 px wide -> `Job Bot sign`, 167 -> `JobBot`, 134 -> `JobBot`,
96 -> `Job sign`, 57 -> `Label`. A tighter crop is not the fix; the box is in the wrong place.

**3. The brain's own description is not a reliable search phrase.** Asked to describe the thing it
must pick up, it described **JobBot** ("a green rectangular screen with black circular eyes"), and the
grounder went to JobBot. Asked to describe "the card", it described **the JOB CARTRIDGES sign** - and
the grounder then *accepted* the rack at x = 652 with `present: True`. A confident wrong description
produces a false accept, which is worse than a refusal. A human-written phrase ("white card with a
photo of JobBot") scored 3/3, so the mechanism works; the model's choice of subject does not.
