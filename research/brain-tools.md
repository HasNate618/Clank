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
