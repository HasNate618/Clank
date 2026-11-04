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
