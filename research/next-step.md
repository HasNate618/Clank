# Deciding the next step when nobody says what it is

**STATUS: MEASURED, 2026-11-04.** LFM3-Omni-2B, temperature 0, one 960x540 still, ~0.4-0.5 s.

JobBot says "take a cartridge" and then stops. The player is meant to understand that the
cartridge goes in the slot. So the brain cannot be a parser of instructions - for the second half
of every task there is no instruction to parse. `brain.next_step()` asks for a goal instead.

## The prompt that works

```
You are a robot playing a VR game. The game told you to take a cartridge, and you are
holding one. No further instruction was given. Decide what you are trying to do next, and
where, from what you can see. Reply with JSON only, no explanation:
{"goal": "the outcome you want", "object": "the thing you act on or are holding",
 "destination": "the name of the visible thing it goes into, or none", "side": "left|right"}
```

Replies, on two stills:

| still | goal | destination |
| --- | --- | --- |
| faded console | `insert the cartridge into the game console` | `the game console` |
| live frame | `insert cartridge into the game console` | `the game console` |

On a fuller prompt the destination came back sharper: `INSERT CARTRIDGE screen on the central
console`. That is a name `locate` can resolve, which is all the adapter needs.

## The prompt that does not

Adding the skill vocabulary to the prompt - `"skill": "grab|place|reach|release"` - **broke it**:

| still | reply with the enum | reply with the goal |
| --- | --- | --- |
| card in hand | `{"skill": "grab", "target": "cartridge"}` | `insert the cartridge into the game console` |
| faded console | `{"skill": "place", "target": "cartridge"}` | destination named |

With `grab` on the menu it picks `grab` every time, and with `place` it puts the *object* in the
target slot rather than the destination - "place cartridge" tells the body nothing about where.
Telling it there is nothing to grab (the hand is already closed) did not help; measured, it
believes the image over the transcript.

So the division is: **the model says what it wants in words, and `skill_from_goal()` picks from
the closed vocabulary in code.** Same shape as the rest of the system - PLAN names a noun, GROUND
resolves it - applied to the choice of skill instead of the choice of point.

## What this does not solve

The reply is a *plan*, not a fact. It said "insert the cartridge into the game console" while the
hand in the image was holding a JobBot card. Nothing here checks that the premise is true, which
is why the grasp check has to exist before a `place` is allowed to open the hand.

## The live run, 2026-11-04

Scene restarted, body reset, camera held at yaw 4.

**The card step works end to end.** `find_and_reach.py --noun "JobBot card" --grab`: locate
returned (780, 426), the reach converged to 44 mm, the hand closed, and **the cartridge tray
appeared** - the game's own progression, unlocked with no human in the loop. That is the first
task of the level completed by the pipeline: one model call to decide, one to locate, one reach.

**Then it stalls, on two things.**

### The model plans the goal, never the step

with the tray visible and the hands empty, every phrasing produced the same shape:

| situation | reply |
| --- | --- |
| "The cartridge tray is on the counter. Your hands are empty." | `insert cartridge` → `INSERT CARTRIDGE` |
| "...decide the immediate next action only - one step" | `insert cartridge` → `INSERT CARTRIDGE` |
| "Transcript: ... you are holding nothing." | `insert the cartridge into the machine` → `the machine` |

A `place` with an empty hand. So `brain.sequence()` orders the steps in code: place + empty hand
= grab(object), then place(destination). Five tests pin it.

### It cannot name an object it is not holding

Asked what it must pick up first, with empty hands:

| field | reply |
| --- | --- |
| `pick_up` | `"empty hands"` |
| `object` | `"the thing you must be holding to do it"` (echoed the description) |
| `object`, second try | `"the white cartridge pusher/insert tool"` — invented |

So the object noun comes from the **task line** ("take a cartridge"), not from the model. This is
the same rule as everywhere else: the thing that is named is named by whoever knows it.

### Where the cartridge step actually fails

| attempt | locate | distance from rest | reach |
| --- | --- | --- | --- |
| GOURMET CHEF | (888, 448) **dead on the label** | 0.52 m | diverged, 145 mm, body drifted 105 mm |
| GOURMET CHEF, verify on | (888, 448) | - | **verifier refused a correct pixel** |
| AUTO MECHANIC | (367, 326) **wrong object** | 0.83 m | diverged, 97 mm |

Three distinct faults, no overlap:

1. **The tower of cartridges is at 0.5-0.8 m.** Past the 0.40 m envelope. The reach does not fail
   gracefully - it diverges past the envelope, as it did on the card, and the root drifts with it.
2. **The verifier gave a false refusal on a correct pixel.** A crop tight enough to be trustworthy
   reads a label as a fragment, not as a cartridge. For a grab, the grasp itself is the stronger
   post-condition, so verification is now off for grabs (`locate --no-verify`).
3. **Without verification, the rack produces confident errors** - AUTO MECHANIC landed on
   something on the left side of the frame, 0.83 m away, and nothing objected.

### What this means

The card task is done and the machinery around it works. The cartridge task needs the body
**closer** (0.3 m of step) and a rack locator that survives near-identical neighbours. Neither is
a controller problem; both were already on the list before this run, and this run put numbers on
them instead of opinions.
