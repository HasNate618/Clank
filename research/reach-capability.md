# Reach capability: does the arm get to JobBot's card?

`[MEASURED 2026-11-01]`, live session `20261102T222649Z-52ab67`, arm commands over
`control.sock`, frames grabbed from Xwayland (`spike/wbc_game/x11_grab.py`).

The earlier note `research/sonic-reach-envelope.md` measured the envelope along +X and
said the card was "probably reachable, depth unknown". Two claims have since been made
in conversation that this note corrects: that the card is ~0.6-0.9 m away, and that the
robot probably cannot reach it. **Both were guesses, and the second is contradicted by
the frames.**

## The arm's limit, re-measured in this session

| command | achieved wrist | note |
| --- | --- | --- |
| clip target (zero offset) | `(0.111, -0.170, -0.238)` | the offset origin |
| `forward 0.55, lateral 0.10` | `(0.450, -0.060, -0.031)` | **0.45 m forward achieved, 0.10 m short** |

So ~0.45 m of forward reach is what this controller delivers reliably from this pose,
consistent with the envelope note's 0.45-0.59 m band with 0.10-0.18 m residual.

## Where the hand gets to, relative to JobBot

Commanded at the card's pixel, from the pose that frames JobBot:

| command | achieved wrist | wrist pixel | distance to card pixel |
| --- | --- | ---: | ---: |
| card ray @ 0.35 m | `(0.338, -0.193, 0.239)` | (0.530, 0.647) | **208 px** |
| card ray @ 0.45 m | `(0.358, -0.212, 0.332)` | (0.447, 0.526) | 486 px |
| card ray @ 0.55 m | `(0.419, -0.215, 0.266)` | (0.388, 0.548) | 540 px |
| max forward | `(0.450, -0.060, -0.031)` | (0.716, 1.103) | 771 px |

At the 0.35 m command the hand lands within **0.19 m** of the commanded point, and the
frame shows it **beside JobBot's face, at his card**. That is not the picture of an
unreachable object.

## What this does and does not establish

Established: the arm reaches into JobBot's personal space, at his card's height, with a
tracking residual of ~0.19 m. So **"the robot cannot reach the card" is withdrawn.** It
was an inference from an unmeasured distance and it does not survive contact with a
frame.

Not established: whether a **grab registers**. Image proximity is not contact, and the
card's true distance is still unmeasured. The 0.19 m residual plus a plausible
0.2-0.3 m grab radius means it could go either way.

## The obstacle to testing a grab

The verb table closes the hand **by commanding a pose with it**
(`spike/wbc/body_view.py`, `hand_button`): `grab_right` sets the wrist to
`aim_target` - `forward 0.22 m, up 0.12 m` - and closure together, deliberately, so a
button press is attributable to a pose. There is no verb that closes the hand **where
it is**.

That is the wrong shape for a grab: the hand has to be at the card when it closes, and
`grab_right` would pull it back toward the body first. `fire_right` is the only
pose-free command, and it drives the trigger, not the grip.

So the next test needs either a closure-only command (an authority mask of
`AUTH_CLOSURE` alone, which `BodyCommand` already supports) or an adapter that sends
the card pose *with* closure set. Both are small; neither exists yet.

Until then the reachability question is **open but no longer doubtful**: the arm gets
there. Whether the game agrees is the thing to measure, and the test is whether the
card **moves with the hand** - not whether anything glows, because the cyan spiral under
JobBot's card is decorative and is present with the arm away.
