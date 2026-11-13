# Grab Buttons Through the Join Implementation Plan

> **Archived 2026-11-13.** Implementation plan, carried out. Kept for the task breakdown and the gates it defined; the code is the record now.
> Current behaviour: [`README.md`](../../../README.md) · agent rules: [`AGENTS.md`](../../../AGENTS.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the simulated humanoid's hand closure and trigger reach the game as two independent, per-hand controller channels, so a glowing floppy in Superhot's hub can actually be picked up.

**Architecture:** The contract gains one bit and two fields; the executor echoes what it was told because the G1 has no fingers and therefore no achieved closure; `wire.fill` grows per-hand button tuples; the relay forwards them and can force either button for an A/B. Nothing about pushing pixels, walking, or thinking is in scope.

**Tech Stack:** Python 3.12 standard library for `clank/`; the executor stays in `spike/wbc/` and keeps needing MuJoCo and ONNX Runtime.

**Spec:** `research/superhot-play.md` (operator briefing: hub sequence, glow-to-grab, two buttons) and `docs/2026-10-23-clank-design.md` §9 (the brain names skills; the pretty pipe must work scripted first).

## Not in this plan

Sequenced out on purpose, because none of them can be tested until a floppy lifts:

- **Headset-over-head.** Needs its own reach measurement — the G1 has no neck joint and head pose is derived, so "above you" may sit outside the 0.20 m envelope. `research/superhot-play.md` §5 item 3.
- **A bounded 0.3–0.7 m step** (`AUTH_ROOT_VEL`). Only if a pickup is out of arm's reach, and it must stay inside the ~1 m spawn radius.
- **Time-advance micro-motion.** Task 6's empty-click result on a held trigger will inform it, but it is a policy question, not a button.
- **Two-hand overlap.** Allowed by the game, not required for the hub.

## Global Constraints

- `clank/` is **standard-library only**. No numpy, no mujoco, no onnxruntime, no new third-party packages anywhere in the test suite.
- Current suite is **323 tests**, all passing, no network and no GPU. Every task leaves it green.
- `AUTH_CLOSURE` already exists at `1 << 5` and is already in the executor's `SUPPORTED_AUTH`. Do not renumber existing bits — `AUTH_GAIT`/`AUTH_POSTURE` are at `1 << 7`/`1 << 8` and are already flagged `UNSUPPORTED_IN_V1`.
- Measured geometry that constrains the demo: standing reach **~0.20 m**, The Fall's pistol **0.70 m**, spawn radius **~1 m** `[OPERATOR]`. Grab must work from a stand.
- Claim discipline: a step that produces a number records where it came from. Never write a residual or a rate that was not printed by a command in this repo.
- Commit only files this plan touches. Commit after each task.

## Review Focus

- **A grab that misses by 15 cm is indistinguishable from a broken button.** Both produce "nothing happened". The relay's CSV must therefore record what buttons it sent on every row, so a null result can be attributed to the button rather than to the reach.
- **`squeeze_value` may be the wrong field; the game may read `hand_curl`.** The manual client sets both, so its success would not say which mattered. The A/B must be able to send one and withhold the other.
- **A held trigger is not a click.** Unity's OpenXR action for select distinguishes value from click; a level-triggered `trigger_click` re-fires every frame. Clicks must be edge-triggered, once per press.
- **Buttons must survive a dead body.** If the state file stops advancing, the relay holds the last pose; it must still send the forced button rather than falling back to a rest pose with no buttons.
- **Left/right must not be transposed.** A per-hand force test with asymmetric values is the only thing that catches a swapped tuple.

---

### Task 1: Contract — a trigger bit, and buttons on the state

**Files:**
- Modify: `clank/body.py` (`AUTH_TRIGGER`, `KNOWN_AUTH`, `BodyCommand.trigger`, `BodyState.closure`, `BodyState.trigger`, `commanded_buttons()`)
- Modify: `clank/validate.py` (trigger range, gated on its own bit)
- Test: `tests/test_body.py`, `tests/test_validate.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `AUTH_TRIGGER = 1 << 9`; `KNOWN_AUTH = (1 << 10) - 1`; `BodyCommand.trigger: tuple[float, float] | None = None`; `BodyState.closure: tuple[float, float] = (0.0, 0.0)`; `BodyState.trigger: tuple[float, float] = (0.0, 0.0)`; `commanded_buttons(cmd: BodyCommand) -> tuple[tuple[float, float], tuple[float, float]]` returning `(closure_lr, trigger_lr)`, zeroes where the authority bit is unset or the field is `None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_body.py
def test_trigger_is_its_own_authority_bit(self) -> None:
    self.assertEqual(AUTH_TRIGGER, 1 << 9)
    self.assertEqual(AUTH_TRIGGER & AUTH_CLOSURE, 0)
    self.assertNotEqual(AUTH_TRIGGER & ~KNOWN_AUTH, 0, "trigger must be inside KNOWN_AUTH")

def test_commanded_buttons_are_per_hand_and_authority_gated(self) -> None:
    grip_and_fire = BodyCommand(
        authority=AUTH_CLOSURE | AUTH_TRIGGER,
        closure=(0.0, 1.0),
        trigger=(1.0, 0.0),
    )
    self.assertEqual(commanded_buttons(grip_and_fire), ((0.0, 1.0), (1.0, 0.0)))

    no_authority = BodyCommand(closure=(1.0, 1.0), trigger=(1.0, 1.0))
    self.assertEqual(commanded_buttons(no_authority), ((0.0, 0.0), (0.0, 0.0)))

    absent = BodyCommand(authority=AUTH_CLOSURE | AUTH_TRIGGER)
    self.assertEqual(commanded_buttons(absent), ((0.0, 0.0), (0.0, 0.0)))
```

```python
# tests/test_validate.py
def test_trigger_outside_unit_range_is_rejected(self) -> None:
    cmd = BodyCommand(authority=AUTH_TRIGGER, trigger=(1.5, 0.0))
    result = validate(cmd)
    self.assertIsNone(result.command)
    self.assertIn("trigger", result.problem)

def test_closure_and_trigger_are_independent(self) -> None:
    grip_only = BodyCommand(authority=AUTH_CLOSURE, closure=(1.0, 0.0), trigger=(9.0, 9.0))
    self.assertIsNotNone(validate(grip_only).command, "an unset bit must not be validated")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_body tests.test_validate -v`
Expected: ImportError or `NameError: AUTH_TRIGGER`, and the validation tests fail with a missing attribute.

- [ ] **Step 3: Implement the contract change**

In `clank/body.py`, next to the other bits:

```python
AUTH_CLOSURE = 1 << 5  # 0 open .. 1 closed; the adapter maps it to grip/squeeze
AUTH_WAIST = 1 << 6
AUTH_GAIT = 1 << 7
AUTH_POSTURE = 1 << 8
AUTH_TRIGGER = 1 << 9  # 0 released .. 1 pressed; a separate channel from closure

KNOWN_AUTH = (1 << 10) - 1
```

Add `trigger: tuple[float, float] | None = None` directly after `closure` in `BodyCommand`, add `closure` and `trigger` to `BodyCommand.to_dict()`, and add them to `BodyState` with `(0.0, 0.0)` defaults plus both keys in `BodyState.to_dict()`. Then:

```python
def commanded_buttons(
    cmd: BodyCommand,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """(closure, trigger) as (left, right), zero where the writer asked for nothing.

    The G1 has no fingers, so there is no achieved closure to report — see
    `BodyState.closure`. This is the commanded value, and callers must not read it
    as evidence that the hand closed.
    """
    grip = (
        cmd.closure
        if (cmd.closure is not None and (cmd.authority & AUTH_CLOSURE) != 0)
        else (0.0, 0.0)
    )
    fire = (
        cmd.trigger
        if (cmd.trigger is not None and (cmd.authority & AUTH_TRIGGER) != 0)
        else (0.0, 0.0)
    )
    return (float(grip[0]), float(grip[1])), (float(fire[0]), float(fire[1]))
```

In `clank/validate.py`, mirror the existing closure block:

```python
if cmd.trigger is not None and (cmd.authority & AUTH_TRIGGER):
    for value in cmd.trigger:
        if not _finite(value) or not (0.0 <= value <= 1.0):
            return ValidationResult(None, "trigger outside [0, 1]")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_body tests.test_validate -v`
Expected: PASS. Then `python3 -m unittest discover -s tests -t . -q` → OK.

- [ ] **Step 5: Commit**

```bash
git add clank/body.py clank/validate.py tests/test_body.py tests/test_validate.py
git commit -m "feat(contract): trigger is its own channel, not a meaning for closure"
```

---

### Task 2: Wire — per-hand grip, curl, trigger and edge-triggered clicks

**Files:**
- Modify: `clank/wire.py` (`fill`)
- Test: `tests/test_wire.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `fill(d, poses, *, grip=(0.0, 0.0), curl=(0.0, 0.0), trigger=(0.0, 0.0), trigger_click=(False, False)) -> RemoteData`, every tuple `(left, right)`. Sets `squeeze_value`, `squeeze_force`, `hand_curl[0..4]` from `grip`/`curl`, `trigger_value` and `trigger_touch` from `trigger`, and `trigger_click` from the boolean.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wire.py
def test_fill_sends_buttons_per_hand(self) -> None:
    packet = wire.fill(
        wire.blank_packet(),
        wire.REST_POSES,
        grip=(0.0, 1.0),
        trigger=(1.0, 0.0),
        trigger_click=(True, False),
    )
    self.assertAlmostEqual(packet.left.squeeze_value.x, 0.0)
    self.assertAlmostEqual(packet.right.squeeze_value.x, 1.0)
    self.assertAlmostEqual(packet.left.trigger_value.x, 1.0)
    self.assertAlmostEqual(packet.right.trigger_value.x, 0.0)
    self.assertTrue(packet.left.trigger_click)
    self.assertFalse(packet.right.trigger_click)

def test_curl_and_squeeze_are_separately_controllable(self) -> None:
    squeeze_only = wire.fill(wire.blank_packet(), wire.REST_POSES, grip=(1.0, 1.0), curl=(0.0, 0.0))
    self.assertAlmostEqual(squeeze_only.left.squeeze_value.x, 1.0)
    self.assertAlmostEqual(squeeze_only.left.hand_curl[0], 0.0)

def test_held_trigger_does_not_repeat_the_click(self) -> None:
    held = wire.fill(wire.blank_packet(), wire.REST_POSES, trigger=(1.0, 1.0))
    self.assertFalse(held.left.trigger_click, "a click is an edge, not a level")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_wire -v`
Expected: `TypeError: fill() got an unexpected keyword argument 'grip'`.

- [ ] **Step 3: Implement `fill`**

Replace the `trigger`/`squeeze` parameters with the four keyword tuples and write, per side:

```python
        c.squeeze_value.x = grip[i]
        c.squeeze_force.x = grip[i]
        for k in range(5):
            c.hand_curl[k] = curl[i]
        c.trigger_value.x = trigger[i]
        c.trigger_touch = trigger[i] > 0.0
        c.trigger_click = bool(trigger_click[i])
        c.thumbstick.x = 0.0
        c.thumbstick.y = 0.0
```

The old body set `trigger_click = trigger > 0.0`, which re-clicks every frame a button is held. The caller now owns the edge.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_wire -v`
Expected: PASS. Existing `test_fill_writes_poses_and_buttons` must be updated for the new signature, not deleted.

- [ ] **Step 5: Commit**

```bash
git add clank/wire.py tests/test_wire.py
git commit -m "feat(wire): per-hand grip, curl, trigger, and edge-triggered clicks"
```

---

### Task 3: Executor — honour both bits and report them as achieved-is-commanded

**Files:**
- Modify: `spike/wbc/sonic_executor.py` (`SUPPORTED_AUTH`, `step`)
- Modify: `spike/wbc/body_view.py` (the `hold` constructor and any `BodyState` assembly that would drop the new fields)
- Test: none can cover this — the executor imports MuJoCo. Verified by the run in Task 5 Step 4 and recorded in `research/superhot-play.md`.

**Interfaces:**
- Consumes: `commanded_buttons()` and `AUTH_TRIGGER` from Task 1.
- Produces: `BodyState.closure` / `BodyState.trigger` carrying the commanded values every tick, and `FLAG_UNSUPPORTED` no longer raised by `AUTH_TRIGGER`.

- [ ] **Step 1: Record the expected behaviour as a failing run**

Run:
```bash
spike/e3/.venv/bin/python spike/wbc/body_view.py --out /tmp/btn --no-render --seconds 2 \
  --control /tmp/btn/control.sock
```
then, in another shell, send a command with `{"authority": 544, "closure": [0.0, 1.0], "trigger": [1.0, 0.0]}` (544 = `AUTH_CLOSURE | AUTH_TRIGGER`).
Expected: `/tmp/btn/body_state.json` has no `closure`/`trigger` keys, or they are all zero — proving the bits do nothing yet.

- [ ] **Step 2: Add the bit and the echo**

`SUPPORTED_AUTH` gains `AUTH_TRIGGER`. In `step()`, after the residuals block and before constructing `BodyState`, call `closure, trigger = commanded_buttons(cmd)` and pass both to `BodyState(...)`. Do not cache them on `self`: nothing reads them between ticks, and an empty-mask hold must report `(0.0, 0.0)` rather than the last grip.

- [ ] **Step 3: Re-run the failing check to see it pass**

Run the same body view and control message.
Expected: `body_state.json` shows `"closure": [0.0, 1.0], "trigger": [1.0, 0.0]`, and `flags` does not include `FLAG_UNSUPPORTED` (8).

- [ ] **Step 4: Run the standard suite**

Run: `python3 -m unittest discover -s tests -t . -q`
Expected: OK — the executor is not imported by the suite, so this only proves nothing else broke.

- [ ] **Step 5: Commit**

```bash
git add spike/wbc/sonic_executor.py spike/wbc/body_view.py
git commit -m "feat(executor): echo commanded grip and trigger into BodyState"
```

---

### Task 4: Relay — forward the buttons, and force them for an A/B

**Files:**
- Modify: `clank/pose_relay.py` (`main`, the packet build, `log_row`, `LOG_HEADER`)
- Test: `tests/test_pose_relay.py`

**Interfaces:**
- Consumes: `wire.fill(..., grip, curl, trigger, trigger_click)` from Task 2; `closure`/`trigger` keys in the state dict from Task 1.
- Produces: CLI `--force-grab {none,left,right,both}` and `--force-trigger {none,left,right,both}` (default `none`), and `--grab-mode {squeeze,curl,both}` (default `both`). New CSV columns `grip_l, grip_r, fire_l, fire_r, click`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pose_relay.py
def test_forced_grab_is_per_hand(self) -> None:
    self.assertEqual(pose_relay.forced_pair("left"), (1.0, 0.0))
    self.assertEqual(pose_relay.forced_pair("both"), (1.0, 1.0))
    self.assertEqual(pose_relay.forced_pair("none"), (0.0, 0.0))

def test_click_is_the_rising_edge_only(self) -> None:
    edge = pose_relay.ClickEdge()
    self.assertEqual(edge.update((1.0, 0.0)), (True, False), "first press clicks")
    self.assertEqual(edge.update((1.0, 0.0)), (False, False), "holding does not re-click")
    self.assertEqual(edge.update((0.0, 0.0)), (False, False))
    self.assertEqual(edge.update((1.0, 0.0)), (True, False), "re-press clicks again")

def test_state_without_buttons_still_sends_zero(self) -> None:
    grip, fire = pose_relay.buttons_from_state(sample_state())
    self.assertEqual((grip, fire), ((0.0, 0.0), (0.0, 0.0)))

def test_forced_grab_survives_a_missing_state_file(self) -> None:
    """A dead body must not take the button with it."""
    grip, fire = pose_relay.buttons_from_state({})
    grip = pose_relay.forced_pair("both") or grip
    self.assertEqual(grip, (1.0, 1.0))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_pose_relay -v`
Expected: `AttributeError: module 'clank.pose_relay' has no attribute 'forced_pair'`.

- [ ] **Step 3: Implement the three helpers and thread them through**

```python
def forced_pair(mode: str) -> tuple[float, float]:
    """(left, right) for a CLI force mode. 'none' releases both."""
    return {
        "none": (0.0, 0.0),
        "left": (1.0, 0.0),
        "right": (0.0, 1.0),
        "both": (1.0, 1.0),
    }[mode]


def buttons_from_state(state: Mapping[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Read closure/trigger out of a state dict, tolerating a body that predates them."""
    def pair(key: str) -> tuple[float, float]:
        raw = state.get(key)
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            return (0.0, 0.0)
        return (float(raw[0]), float(raw[1]))
    return pair("closure"), pair("trigger")


class ClickEdge:
    """Turns a held trigger value into one click per press, per hand."""

    def __init__(self) -> None:
        self._prev = (0.0, 0.0)

    def update(self, values: Sequence[float]) -> tuple[bool, bool]:
        clicks = tuple(v > 0.0 and p <= 0.0 for v, p in zip(values, self._prev, strict=True))
        self._prev = (float(values[0]), float(values[1]))
        return (clicks[0], clicks[1])
```

In the tick loop, the force options win over state, and `--grab-mode squeeze` withholds the curl. `ClickEdge` is an attribute of the relay object created in `main()` (`self._edge = ClickEdge()`), not a module global and not re-created per tick:

```python
        grip_state, fire_state = buttons_from_state(state) if state else ((0.0, 0.0), (0.0, 0.0))
        grip_force, fire_force = forced_pair(args.force_grab), forced_pair(args.force_trigger)
        grip = grip_force if any(grip_force) else grip_state
        fire = fire_force if any(fire_force) else fire_state
        curl = grip if args.grab_mode in ("curl", "both") else (0.0, 0.0)
        grip_field = grip if args.grab_mode in ("squeeze", "both") else (0.0, 0.0)
        click = self._edge.update(fire)
        packet = wire.fill(base_packet, poses, grip=grip_field, curl=curl,
                           trigger=fire, trigger_click=click)
```

Keep the `ClickEdge` on the relay object created in `main()`, not a module global. Extend `LOG_HEADER` and `log_row` with the five button columns so every emitted frame records what was sent.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_pose_relay -v`
Expected: PASS, including the existing handshake, staleness and malformed-state tests.

- [ ] **Step 5: Commit**

```bash
git add clank/pose_relay.py tests/test_pose_relay.py
git commit -m "feat(relay): forward grip and trigger, and force either for an A/B"
```

---

### Task 5: Verbs — `grab`, `release`, `fire` from the dashboard

**Files:**
- Modify: `spike/wbc/body_view.py` (`verb_command`)
- Modify: `clank/body_preview.py` (`BODY_VERBS`)
- Modify: `clank/static/app.js`, `clank/static/index.html` (buttons)
- Test: `tests/test_body_preview.py`

**Interfaces:**
- Consumes: `AUTH_CLOSURE`, `AUTH_TRIGGER` from Task 1.
- Produces: verbs `grab_right`, `release_right`, `fire_right` and their `_left` twins, mapping to `BodyCommand(authority=AUTH_R_HAND | AUTH_CLOSURE, right_hand=<current aim target>, closure=(0.0, 1.0))` so the grab carries a reach with it — a closure sent with no hand pose is a button press at whatever the arm happens to be doing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_body_preview.py
def test_button_verbs_include_grab_and_fire(self) -> None:
    for verb in ("grab_right", "release_right", "fire_right", "grab_left", "release_left"):
        self.assertIn(verb, body_preview.BODY_VERBS)
    self.assertNotIn("grab", body_preview.BODY_VERBS, "a grab without a side cannot be bound")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m unittest tests.test_body_preview -v`
Expected: FAIL, `'grab_right' not found in ...`.

- [ ] **Step 3: Implement the verbs**

In `verb_command`, add two factories so the per-hand pairs stay one shape:

```python
    def hand_button(side: str, closed: bool) -> BodyCommand:
        """Close or open one hand, at the reach the aim verb already uses.

        The closure travels with a hand pose on purpose: a bare button press fires
        at whatever the arm happens to be doing, which makes a null result
        impossible to attribute.
        """
        target = executor.aim_target(side, forward_m=0.22)
        closed_pair = (1.0, 0.0) if side == "left" else (0.0, 1.0)
        if side == "right":
            return BodyCommand(
                seq=0,
                authority=AUTH_R_HAND | AUTH_CLOSURE,
                right_hand=target,
                closure=closed_pair if closed else (0.0, 0.0),
            )
        return BodyCommand(
            seq=0,
            authority=AUTH_L_HAND | AUTH_CLOSURE,
            left_hand=target,
            closure=closed_pair if closed else (0.0, 0.0),
        )

    def fire(side: str) -> BodyCommand:
        trigger = (0.0, 1.0) if side == "right" else (1.0, 0.0)
        return BodyCommand(seq=0, authority=AUTH_TRIGGER, trigger=trigger)
```

Register `grab_right`, `release_right`, `grab_left`, `release_left`, `fire_right`, `fire_left` in the `verbs` dict, and extend `BODY_VERBS`. Add one button per verb to the dashboard's body panel, following the existing button markup.

- [ ] **Step 4: Verify by running it, since no stdlib test can import the viewer**

Run:
```bash
spike/e3/.venv/bin/python spike/wbc/body_view.py --out /tmp/btn --no-render --seconds 6 \
  --control /tmp/btn/control.sock
python3 - <<'PY'
import json, socket
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect("/tmp/btn/control.sock")
for verb in ("grab_right", "release_right"):
    s.sendall((json.dumps({"verb": verb}) + "\n").encode())
    print(verb, s.recv(4096).decode().strip())
PY
```
Expected: both verbs reply accepted, and `/tmp/btn/body_state.json` shows `closure` going `[0.0, 1.0]` then back to `[0.0, 0.0]`.

- [ ] **Step 5: Commit**

```bash
git add spike/wbc/body_view.py clank/body_preview.py clank/static/app.js clank/static/index.html tests/test_body_preview.py
git commit -m "feat(verbs): grab, release and fire as per-hand dashboard buttons"
```

---

### Task 6: Hub A/B — which button picks up a floppy

**Files:**
- Create: `research/grab-ab-results.md`
- Modify: `README.md` (status table rows for trigger and grab)

**Interfaces:**
- Consumes: everything above.
- Produces: a recorded answer to "which wire field does Superhot read for grab", or a recorded failure with the four combinations tried.

- [ ] **Step 1: Run the session and park the body at the hub**

This task needs the operator: it is the one step in the plan where the machine cannot supply the answer. Start the session from the dashboard, get the view to the hub, then start the body viewer with the relay already connected. Confirm the floppy is on screen and the body's hand is within ~0.20 m of it — screenshot before the attempt, because a grab at 0.5 m is a reach failure, not a button failure.

- [ ] **Step 2: Run the four combinations, one at a time**

For each of `--grab-mode squeeze`, `curl`, `both`, and `--force-trigger both`, restart the relay with that flag, hold it for ~3 s, and capture a screenshot. Change one flag between runs.

Expected: at most one combination makes the disk glow and lift. Record which, and record the neutral result for the others.

- [ ] **Step 3: Write `research/grab-ab-results.md`**

One row per combination: flag, screenshot path, what changed on screen, and what it proves. If nothing worked, the document says so and lists the next hypothesis (reach error first, then `squeeze_force`, then two-hand).

- [ ] **Step 4: Update the README status table**

Replace "Trigger reaches the game — not measured" with whichever is true after the A/B, and add a grab row. Do not upgrade a row on a partial result.

- [ ] **Step 5: Commit**

```bash
git add research/grab-ab-results.md README.md
git commit -m "docs: record which button Superhot reads for grab"
```
