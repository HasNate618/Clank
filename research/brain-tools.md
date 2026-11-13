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
