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

## 2026-11-13 — consistent grounding: the identity check was the flapper

With the gate-band fade, stage 1 and stage 2 became stable (BOTTOM-RIGHT, pixel on the card, 5/5).
Stage 3 then refused the card, and the reason took three measurements to find.

**The card is a sign.** It is a white rectangle printed with "JobBot / Museum Curator / 101 Job
Street". Asked to describe that crop, the model reads the card's own text back at us, and every
reading is a refusal:

| crop half | width | description (5/5 identical) | `describe_matches("card", ...)` |
| --- | --- | --- | --- |
| 0.16 | 307 px | `Job Bot sign` | refuse |
| 0.08 | 153 px | `JobBot Museum Curator` | accept (curator is a synonym) |
| 0.05 | 96 px | `Job 30` | refuse |
| 0.03 | 57 px | `A label` | refuse |

The accept window is one crop size wide, and it moves when the median box moves: the same nominal
0.16 crop accepted 3/3 on one box and refused 5/5 on a box 6 px away. That is not a tuning problem,
it is a question that cannot be answered about a text-covered object.

**A closed question answers it.** Asking what *kind* of object it is, with a plausible alternative in
the prompt, is right on every case measured, 3/3 each:

| crop | `...a {noun}, a sign, or something else?` |
| --- | --- |
| the card | `Card` 3/3 |
| the console's OFF button | `Sign` 3/3 |
| the robot's own glove | `Something else` 3/3 |
| the console | `Sign` 3/3 |

And the two-way form - "is it a {noun}, or something else?" - is not enough: the OFF button answered
`card` 3/3, because a binary question leaves the model no plausible way to say no. Same lesson as
`{"none": true}` in `BOX_PROMPT`: refusal is offered, not hoped for. "A sign" is a generic visual
category, not a list of props.

**The vote is now three reads.** One call was measured to flip between `Job Bot card` (accept) and
`Job Bot sign` (refuse) on the same still, because temperature 0 is not reproducible across the
server's four slots. Three reads on the four-slot server cost one call's wall clock.

End to end on the run's own frame, five consecutive `locate("card")` on the faded still:

```text
BOTTOM-RIGHT px=(884,473) PRESENT [True, True, True] 'Card | Card | Card' 2167ms
BOTTOM-RIGHT px=(888,466) PRESENT [True, True, True] 'Card | Card | Card' 2060ms
BOTTOM-RIGHT px=(888,466) PRESENT [True, True, True] 'Card | Card | Card' 2108ms
BOTTOM-RIGHT px=(888,466) PRESENT [True, True, True] 'Card | Card | Card' 2146ms
BOTTOM-RIGHT px=(884,473) PRESENT [True, True, True] 'Card | Card | Card' 2005ms
```

and a noun that is not in the scene refuses, 2/2 (`sign`, `Something else`).

## 2026-11-13 — tool calling, measured: the brain does not use it, and it would not help

`clank/brain.py` sends `{messages, max_tokens, temperature, stream}` and nothing else: no
`tools`, no `functions`, no `response_format`. The model is asked for JSON **in prose** and
`extract_json_object()` brace-matches the reply, tolerating fences and chatter. The skills in
`clank.intent.SKILLS` are the adapter's, never the model's: the model writes a goal sentence and
`_SKILL_WORDS` / `_TASK_VERBS` / `skill_from_goal` pick the skill.

Since the model was trained on OpenAI-style tool calls and the server runs `--jinja`, the
obvious question is whether a real `tools=` array beats the prose contract. Four arms, the same
six utterances, the still from run `145444-56e4fc`, three samples each, `temperature 0.0`:

| arm | input |
| --- | --- |
| A | the live `NEXT_STEP_PROMPT` (goal in words, JSON in prose) |
| C | the enum-in-prose that measured 0/4 grabs on 2026-11-04 |
| B1 | `tools=` with grab / place / reach / look / release / hold |
| B2 | B1 plus `move(direction, metres)` — the tool a generic pull needs |

**Tool calls work.** `tools` + `tool_choice: "auto"` produced clean typed arguments
(`move({"direction": "back", "metres": 0.5})`); with `tool_choice: "required"` emission was
**18/18 tool calls** and no prose at all. With `auto` roughly a third of replies came back as a
scene description instead of a call — the model chatted about the Job Simulator room.

**It is not more accurate.** Act choice, `tool_choice: "required"`, n=3 each:

| case | wanted | A prose | C enum | B2 tools+move |
| --- | --- | --- | --- | --- |
| greeting, "...Here's my card" | grab | reach 0/3 | grab 3/3 | look 0/3 |
| "take my card" | grab | grab 3/3 | grab 3/3 | grab 3/3 |
| "grab one of the job cartridges" | grab | grab 3/3 | grab 3/3 | grab 3/3 |
| "insert the cartridge into the slot" | place | **place 3/3** | place 3/3 | grab 0/3 |
| "pull the lever to job" | pull | reach 0/3 | grab 0/3 | grab 0/3 |
| "a large machine is running with a steady hum" | hold | **reach 0/3** | **hold 3/3** | look 0/3 |

A 3/6, C 4/6, B2 2/6.

Three things this actually measured, in order of how much they matter:

1. **Asked a bare prompt, the model invents a chore from a non-order.** A steady hum became
   `{"goal": "operate the JobBot machine"}` and a `reach`, 3/3, in two separate runs, and the
   enum arm answered `hold` 3/3 for the same input. But this is **not** a live defect, and the
   first draft of this section said it was: the loop never asks that question. `auto.py` calls
   `brain.think()`, which returns `hold` before the model is paid whenever
   `task_intent(heard) == ("", "")` — the gate added in `07eea01` — and the diary shows it
   working (run `144738-015c22`, 14:53:45: hum heard, `plan: hold(-)`, zero model calls). So the
   prose contract's inability to say "nothing" is real and already answered in code, which is
   the better place for it: the enum arm reaches the same verdict by paying for a call.
2. **A goal sentence carries a relation that a single tool call cannot.** "insert the cartridge
   into the slot" has two nouns and a preposition; `skill_from_goal` reads them and A answers
   `place` 3/3. A tool call must commit to one name and one destination in one breath, and
   `grab({"object": "Job Cartridge"})` is what it commits to. Two-argument tools are not
   free — the model has to already know the second argument.
3. **"Pull the lever to job" is destroyed by the prose arm** (`goal: "job simulator"`,
   `reach`, 0/3): the noun never survives, so no amount of direction parsing would help. The
   tools arm at least kept `grab({"object": "lever"})` 3/3 — the first half of the pull.

And the one result that survived as a positive: **a direction is expressible to this model.**
With `move` in the tool list, "pull the lever to job" answered
`move({"direction": "back", "metres": 0.5})` 3/3 on one run — an exact, typed pull — and
`grab({"object": "lever"})` 3/3 on the next. Unreliable choice, correct vocabulary. That is the
same conclusion `docs/superpowers/plans/2026-11-13-directional-motion.md` reached from the code
side, and it is why that plan keeps the goal sentence and maps direction words in code rather
than moving the brain onto tool calls.

Caveat on all of it: `temperature 0.0` is not reproducible across the server's four slots, so a
3/3 is a congruent run, not a guarantee — case 5's B2 arm flipped between two runs of the same
script. n=6 cases. This is enough to say tool calling is not a free win and to name the two
defects worth fixing; it is not enough to declare A or C the winner.

## 2026-11-14 — the cartridge, and why the label is not in the prompt

Run `193516-ca1fde` grabbed the card and then made **seven** attempts at the job cartridge. All
seven pointed at the same object, and it was not a cartridge:

| # | quadrant | box centre (full frame) | stage 3 | depth | outcome |
| --- | --- | --- | --- | --- | --- |
| 1 | BOTTOM-RIGHT | x 597-613, y 312-468 | `Card` 3/3 | 0.19 m | refused |
| 2 | BOTTOM-LEFT | x 437-570, y 250-429 | `Job cartridge` 3/3 | 0.19 m | identity accepted, depth refused |
| 3 | BOTTOM-LEFT | x 456-519, y 270-401 | `Job cartridge` 3/3 | 0.21 m | identity accepted, depth refused |
| 4-7 | BOTTOM-LEFT | x ~450-570, y ~275-405 | `Sign` 3/3 | 0.20-0.21 m | refused |

Mapping the boxes back to the frame shows what they are on: the **terminal's card slot**, whose
front panel reads **INSERT CARTRIDGE** in 40 px letters, 20 cm from the robot's hand. The rack of
three cartridges - `AUTO MECHANIC`, `STORE CLERK`, `GOURMET CHEF` - is on the other side of the
frame. "job cartridge" names both, and the grounder is text-driven, so it reads the nearest big
text.

**Why they repeated.** `one_turn` retries `sense` up to `SENSE_BUDGET + 1` times, and each retry
asked the identical question of an almost identical frame at temperature 0: same quadrant, same
box, same answer. Then the next turn re-planned the same goal, because `memory.find(target)` was
`present: False` and that was read as "not found yet" rather than "I looked and it is not there".
Nothing carried the refusal forward: `sense()` took a noun and nothing else.

**Fixed three ways.** A refusal now ends the turn (the `continue` that re-asked is gone, both call
sites). The refused box goes into `auto.REFUSED` and the next search passes it as `locate(avoid=)`,
which sends stage 2 to the quadrant *beside* the refused one - mirroring left/right keeps the row
the model chose and changes the one thing it got wrong. And a candidate nearer than
`MIN_SURFACE_M` is now `too_close`, not `not_visible`: it was visible, it was the console the robot
was standing at, and stage 3 had named it correctly off its own panel.

**The label, in code.** The operator asked for the green `STORE CLERK` cartridge to be named in
the planner prompt. Tried twice:

- unconditional - "when you choose a job cartridge, choose the one with the green label that reads
  STORE CLERK": JobBot said "Here's my card." and the model answered
  `{"goal": "pick up the Job Cartridge with the green 'STORE CLERK' label"}`
- conditional - "if the newest thing heard asks for a job cartridge, ...": still
  `object="Job Cartridge"` to "Here's my card." and to "take my card", 3/3 each

A prop named in the prompt outranks the sentence this model was told to obey - spec 3.2's lesson,
repeating. So `brain.refine_target` does it instead: `"job cartridge"` -> `"STORE CLERK cartridge"`
on the way to the grounder, where `APPROACH_M` already lives. Measured after the change, 3/3 each:
`"Here's my card."` -> `card`, `"take my card"` -> `Job Bot card`,
`"grab one of the job cartridges"` -> `STORE CLERK cartridge`.

**And the offer rule, which the planner path never asked.** The same frame produced
`"Job Bot card"` in one run and `"Job Simulator game console (center foreground, labeled ...)"`
3/3 in the next for the same sentence. `task_intent` has read offers since the first day
("JobBot does not only give orders - he holds the thing out and says what it is") but `next_step`
took the noun from the model's `object` field. An offer now names the noun in code.

**[OPERATOR] The cartridges are not on the table until the card is taken.** A cartridge search
before the card grab is a search of an empty table, and its refusal says nothing about grounding.
