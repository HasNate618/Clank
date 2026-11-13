# Brain tools — Design (autonomous chooser)

> **Archived 2026-11-13.** Superseded for the live chooser by
> [`2026-11-13-session-brain.md`](../../superpowers/specs/2026-11-13-session-brain.md): the prompt no
> longer lists props or cues, the run is one conversation rather than one message per turn, and every
> model call is recorded in `llm.jsonl`. This document is kept for the measurements it contains — the
> four-still enum comparison, the locate pipeline, and why the sense tool is not trusted to be chosen.
> The parent [`2026-10-31-brain.md`](../../superpowers/specs/2026-10-31-brain.md) is still current.

- Date: 2026-11-10
- Status: draft for review; no loop rewrite until this document is accepted
- Parent: `docs/superpowers/specs/2026-10-31-brain.md` (instruction following). This spec does not replace it. It restores the claim that spec already made, and names the one tool the live loop currently hides.
- Implements against: `clank/brain.py` (`plan`, `assemble`, `run_turn`), `spike/wbc_game/auto.py`, `spike/wbc_game/locate.py`
- Does not implement: targeting adapter, reach envelope, Job Simulator glow detectors, model serving

Claim tags: `[MEASURED]` / `[OPERATOR]` / `[CITED]` / `[ASSUMPTION]` as in the parent.

## 1. Problem

The parent spec's claim is: a G1 hears speech, looks at one still, emits one legal action, sees the outcome, and tries again. Sequences are not compiled in Python. "If card → cartridge → slot → lever is a `HubFSM` in code, the brain is fake" `[CITED: 2026-10-31-brain.md §2]`.

The live loop in `spike/wbc_game/auto.py` is that HubFSM:

1. Regex `task_intent("take my card")` → `("grab", "card")` `[MEASURED: tests/test_brain.py::TaskIntent]`.
2. `sequence()` then always locates that noun and always `reach_vr --grab`.
3. Any other skill is logged as skipped.

The model never chooses `locate`. A parser that already knows the choreography does. That is the opposite of the premise.

Two later measurements are why the loop drifted, and they stay true:

- Putting `grab|place|reach|release` on a *goal* prompt collapsed every reply to `grab` `[MEASURED: research/next-step.md]`.
- Asked "what should you do?", Omni skipped to a later step: `"take my card"` + a still → `{"goal": "insert the card into the machine"}` `[MEASURED: 2026-11-08, task_intent docstring]`.

Those measurements condemn that prompt shape. They do not condemn a constitution that lists actions. The constitution path already bound speech on replay: audio+image, card visible → `{"skill":"reach","side":"right","target":"card"}` `[MEASURED 2026-11-01: parent §4]`.

## 2. Claim

The brain chooses the next tool from the still and the transcript. Code runs the tool, writes the outcome into working memory, and asks again. When JobBot says "take my card", the first tool the brain is expected to choose is `locate` with a target that names the card.

`locate` is a sense tool. It is not a motor skill, not an `Intent`, and not a hidden step the loop always runs after a parser decides `grab`.

## 3. What is a tool

Two kinds. The planner emits one JSON object naming exactly one.

| Kind | Names | Becomes | Who runs it |
| --- | --- | --- | --- |
| **Sense** | `locate` | a `Sighting` in working memory | `spike/wbc_game/locate.py` (already measured) |
| **Motor** | `hold` `look` `reach` `grab` `release` | `clank.intent/1` | adapter / `reach_vr` |

`locate` is **absent** from `clank.intent.SKILLS`. A planner that names it is legal. An `Intent` that names it is not.

`rest` stays off the constitution. The live loop may still send `rest` as a body-safety reset before the first think `[MEASURED 2026-11-08: a restart without rest returns the T-pose]`. That is not a planner output.

## 4. Architecture

```text
speech ends
        │
        v
TRANSCRIBE   Omni        words. Separate call. `[MEASURED: transcribe first, decide second]`
RESET        code        rest pose, once per turn, before the first still
        │
        v
     freeze-hold
        │
        +-- think (max SENSE_BUDGET sense tools, then one motor) ----+
        |                                                           |
        v                                                           |
SENSE      code        still + BodyState + live block + working memory
PLAN       Omni        → {skill, side, target}     constitution, not next_step
        |                                                           |
        +-- skill == locate ----------------------------------------+
        |     run locate.py; memory.remember(noun, sighting)        |
        |     last: locate none target=<noun> → present|not_visible |
        |     ask PLAN again on a new still (head may have moved)   |
        |                                                           |
        +-- skill in hold|look|reach|grab|release ------------------+
              ASSEMBLE binds a Sighting pixel; validates; emits Intent
              motor turn ends the freeze
```

Rules:

- **The model chooses `locate`.** Code never calls `locate.py` unless PLAN just named it.
- **The model chooses `look`.** The `LOOK_DEG` sweep in `auto.py` is retired. A `not_visible` or `too_far` outcome is how the model learns to look, not a hardcoded yaw table.
- **Pixels never enter a prompt.** Working memory holds `(pixel, camera, present)`. The live block only says `present` / `not_visible` / `too_far`. Parent §10 forbids layout tables; this is the same rule applied to a just-measured point.
- **A motor that needs a place, without a present Sighting, is refused.** `reach`/`grab`/`look` with a target and no `memory.find(target)` → freeze-hold, reason `not_located`, last line tells PLAN. The next think is expected to `locate`. Do not auto-locate to "help".
- **`task_intent`, `next_step`, and `sequence` leave the live loop.** They remain in `clank/brain.py` as measured history until a later deletion. `auto.py` and the demo path must not call them.
- **One motor per freeze.** Sense tools may chain up to `SENSE_BUDGET = 3` (locate, then look, then locate again). A fourth sense request becomes `hold`.
- **Grab is still never automatic on glow.** PLAN must emit `grab`. The motor implementation of `grab` given a Sighting is the already-measured `reach_vr --grab` (close inside the same process `[MEASURED: 2 s gap drifted 0.09 m]`). That is execution, not a hidden skill.

## 5. Constitution (replaces parent §7.1 for this writer)

Keep it this short. The added lines are `locate` in the list, `locate` in the schema, and "if you have not located the thing, locate it". Nothing else.

```text
You are a robot with two hands. You see one image from your eyes and
the speech since your last action.

You may do exactly one action:
- hold    keep still
- look    turn to look; name a target to look toward it, or omit to
          look a step left or right
- locate  find a named thing in what you see; do this before you
          reach or grab it
- reach   move one hand toward a named thing you have located
- grab    close one hand
- release open one hand

While holding something, reach moves it with that hand.

After locate you are told present or not_visible. After a hand
action you are told done, saturated, or timeout, and you see a
new image.

Reply with one JSON object only:
{"skill":"hold"|"look"|"locate"|"reach"|"grab"|"release",
 "side":"none"|"left"|"right",
 "target": <noun string or null>}

side is none for hold, look, and locate, and left or right for
reach, grab, and release.

Do what the voice asked. Do not invent extra chores. If you have
not located the thing, locate it. If locate said it is not there,
look. If you are unsure, hold.

Grab only when glow is yes. A yellow ghost means the held thing
will go into the slot if you release. A full hand cannot grab;
use the other hand or release.
```

`fire` / `strike` / `dodge` / `twitch` / `rest` / `insert` / `pull` stay absent.

Seed is unchanged (parent §7.2). Identity is allowed. Positions are not.

## 6. Live block and working memory

Live block, assembled every think, still has no task list and no coordinates:

```text
heard: <verbatim new speech, or "(none)">
self:  left=<empty|holding> right=<empty|holding> yaw=<rad>
       glow=<yes|no|unknown> ghost=<yes|no|unknown>
last:  <skill> <side> target=<noun|null> → <outcome>
```

`outcome` for `locate` is `present` | `not_visible` | `too_far` | `error`. `too_far` is the depth gate already measured: locate answered the JOB SIMULATOR sign at 4.9 m `[MEASURED 2026-11-08]`. Code sets that word; the model is not told the metre value.

Working memory (code only):

```text
Sighting(noun, present, pixel=(u,v)|None, pixel_px=(x,y)|None, camera, metres)
WorkingMemory.sightings: dict[str, Sighting]
WorkingMemory.find(noun) -> Sighting | None
WorkingMemory.remember(noun, Sighting) -> None
```

`find` matches if the asked noun is a substring of a stored key or the reverse, case-insensitive. `"card"` finds `"JobBot card"`. A miss is `None`, not a guess.

## 7. Planner output

Same three keys as the parent. `locate` is a new allowed `skill`. Parser rules:

- `action` is still a synonym for `skill` (parent: audio path used `action`).
- Skill-as-key (`{"locate":"none","target":"card"}`) still folds.
- `locate` requires `side="none"` and a non-empty `target`. Missing target → plan error, freeze-hold.
- `insert` / `pull` / `rest` / `place` remain illegal planner skills. `place` is an adapter skill, not a constitution action; a cartridge going into a slot is `reach` then `release`.

ASSEMBLE additions:

1. `skill == "locate"` → do not build a motor `Intent`. Caller runs locate. The `Decision` on that think is freeze-hold with note `locate`.
2. `skill in {reach, grab}` and `target` set and `memory.find(target)` is missing or `present is False` → freeze-hold, reason `not_located`.
3. `skill in {reach, grab}` and a present Sighting → `Intent.pixel` from `Sighting.pixel` (normalised). Generator JSON still has no `pixel` field.
4. Existing occupancy and glow/ghost gates stay.

## 8. The loop (`auto.py`)

Per spoken line, after transcribe and the rest-pose reset:

```text
memory = WorkingMemory()
for _ in range(SENSE_BUDGET + 1):
    still = capture()
    thought = brain.think(...)          # PLAN + ASSEMBLE, no locate import in clank/
    log think: skill(target)
    if thought.sense == "locate":
        found = locate.locate(config, frame, target, size)
        gate depth; memory.remember(...)
        log act: locate(...)
        continue
    if thought.intent.skill == "look":
        pin head; continue
    run motor; confirm() with JobBot; break
```

`clank/` stays stdlib-only and does not import `spike/wbc_game/locate.py`. `think()` returns a `Thought`; the spike loop is the only caller of `locate()`.

JobBot's next sentence remains the oracle for whether a grab worked (parent, and `auto.confirm`).

## 9. What this does not change

- Transcribe is still a separate Omni call. Do not pass the wav into PLAN on the live path (parent: one call for words and a decision ignored the words).
- GROUND (`brain.ground` quadrant crop) stays for the replay writer (`run_turn`). The live chooser uses `locate.py`, which already replaced that grounder for the game view.
- `run_turn` stays for replay tests. Do not silently change its "always GROUND after PLAN" behaviour in this work; the live path is `think`, not `run_turn`.
- Closed motor vocabulary on the socket. Models still never emit a 3D point or an OpenXR pose.
- Operator click stays a writer on the same socket.

## 10. Eval

Three arms. The first two are unittest with `_chat` patched and must not open `:11434`. The third is a spike script that *does* talk to Omni, and is not part of `python3 -m unittest`.

| Arm | Input | Bar | Gates the loop? |
| --- | --- | --- | --- |
| **Legal** | constitution + fixtures, no image | JSON parses; skill ∈ the six; `locate` has side none and a target; `rest`/`insert` rejected | **yes** |
| **Bind** | patched replies | "take my card" + scripted `{"skill":"locate","side":"none","target":"card"}` is accepted; a scripted `reach` without a Sighting is `not_located`; a scripted `reach` after `memory.remember("card", present)` fills `Intent.pixel` | **yes** |
| **Speech-bind (Omni)** | frozen still of the card in view + heard `Hey, Hillman, take my card.` | PLAN skill is `locate` and target names the card (substring `card`). Not `grab` of JobBot, not `insert`, not `reach` with no locate. | **yes** for wiring `auto.py`. If this arm fails, shorten the constitution; do not put the choreography back into Python |

Outcome-use, same Omni script, stills only:

- `last: locate none target=card → present` → next skill is `reach` or `grab`, target names the card, not a second locate of a different noun.
- `last: locate none target=card → not_visible` → next skill is `look` (or locate again once). Not `reach`.
- `heard: (none)`, seed only → `look` or `hold`, not `grab`.

Live (operator watches the diary, not an exit code):

1. Session start, JobBot intro, then "take my card".
2. Diary shows `think: locate(card)` (or a target containing `card`) **before** any `act: locate` and **before** any reach.
3. No diary line of the form `the task says 'card', the model said ... - the task wins`.
4. After `locate → present`, diary shows `think: reach` or `think: grab`.
5. `confirm()` still uses JobBot's next line as the grab oracle.

## 11. Review focus

1. **Parser loop.** `auto.py` must not call `task_intent`, `next_step`, or `sequence`. A grep test gates this.
2. **Hidden locate.** `locate.locate` is only called when `thought.sense == "locate"`. The `LOOK_DEG` for-loop that locates on every yaw is gone.
3. **Pixel in the prompt.** `live_block` / constitution / seed contain no `u=`, no `px`, no `TOP-LEFT`.
4. **`locate` on the socket.** `validate_intent` still rejects `skill="locate"`. Only `Thought.sense` carries it.
5. **Walkthrough in the constitution.** No `take then grab then insert` sequence. The new sentences name the *locate* tool, not the museum choreography.
6. **Goal-enum prompt.** Do not revive `NEXT_STEP_PROMPT`'s skill enum. Constitution list ≠ that prompt `[MEASURED: research/next-step.md]`.
7. **Auto-locate on refuse.** `not_located` is a hold + a last line, not a code path that then calls `locate.py`.

## 12. Work order

| Step | Pass | Allowed fail |
| --- | --- | --- |
| **T0** Spec accepted | this document | — |
| **T1** Parser + constitution | Legal arm | — |
| **T2** Working memory + ASSEMBLE bind/refuse | Bind arm | — |
| **T3** `think()` | Bind arm through `think` | — |
| **T4** Omni speech-bind on a frozen still | Speech-bind arm | fail → iterate constitution, **do not** wire `auto.py` |
| **T5** `auto.py` uses `think`; grep-clean of the parser | diary shape on a `--say` dry run | live JobBot still needed |
| **T6** Live JobBot | §10 live diary | reach residual is a body problem; a diary that never locates is a brain fail |
