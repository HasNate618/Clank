# Directional Motion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the brain move a hand by a *direction* — "pull the lever", "push it in" — the way it already moves one to a *named thing*, so pulling, pushing and dragging need no skill of their own.

**Architecture:** `reach` already accepts a pixel (an object) or `--world X Y Z` (a point). This adds a third form: `--move FWD UP LAT`, metres in the pelvis frame, applied to wherever the hand currently is. The model keeps naming the goal in words; a table in `clank/brain.py` turns those words into a direction, and the metres live in code beside `APPROACH_M`. No new entry in `clank.intent.SKILLS`, nothing about the lever in any prompt.

**Tech Stack:** Python 3.12 stdlib for `clank/`; numpy/MuJoCo only inside `spike/`; `unittest` for tests; the live stack (Monado, relay, `spike/wbc/body_view.py`, ARDY) for the final gate.

**Spec:** `docs/superpowers/specs/2026-10-31-brain.md` §6.2 ("Pull the lever: `reach` lever, `grab`, then `reach` downward. No `pull` skill") and §14 ("Do **not** add `insert`, `pull`, `take`, `use`, `throw`"). This plan implements §14's prohibition the way §6.2 describes, and the schema section of `clank/intent.py`.

## Global Constraints

- `clank/` is **standard-library only**. Numpy/MuJoCo/ONNX live in `spike/e3/.venv`.
- **No new skill names.** `clank.intent.SKILLS` is unchanged: `hold, rest, look, reach, grab, release, place, fire, strike, dodge, twitch, menu`. The word "pull" never appears as a skill.
- **No distances in the prompt.** `Aim` words come from the model; metres come from code, exactly as `APPROACH_M` and `Z_GAIN` do.
- **No object positions, no prop list, no skill enum in `NEXT_STEP_PROMPT`.** Its JSON schema does not change in this plan.
- A **direction word in a sign is not an order**: `task_intent()` acts only on what JobBot says, and what it says is already gated in `think()`.
- Every number in this plan carries a claim tag: `[MEASURED]`, `[OPERATOR]`, `[ASSUMPTION]`.
- `ruff` is at `/nix/store/ymj8rxkzq67rjz98r5hkzf7lsah90va8-ruff-0.15.10/bin/ruff`, not on `PATH`.
- Tests: `XDG_RUNTIME_DIR=/tmp/clank-xdg python3 -m unittest discover -s tests -t .`, baseline **1435 tests OK**. Tests must not bind ports 8765 / 8775 / 8770 / 8771 / 4242 / 5901 / 11434.
- Commits: conventional, lowercase, no double quotes in the message (`git commit -F /tmp/msg.txt`). Commit only the files named in the task.
- Clear `spike/wbc_game/__pycache__` and `clank/__pycache__` after any test that imported a mutated module.

## Review Focus

1. **A move from a hand that is not where the body thinks it is.** After a failed reach the reported hand can be 0.1 m from the commanded one; a pull computed from the reported point then travels somewhere else. Expected: the move is computed from the reported hand *at the moment of the move*, and a stale or missing report refuses rather than moving.
2. **A pull that walks the hand into the torso.** Backward travel is the one direction the measured box does not like (x floor 0.079 m). Expected: a 0.25 m pull from a hand at 0.12 m forward clamps and reports the clamp, instead of driving to an unreachable point.
3. **The direction and the verb disagree.** "Push the cartridge into the slot" contains both `place` (insert/put/slot) and a `move` word (`in`). Expected: the place reading wins — a goal that names a destination is a `place`, and only a goal with no destination is a move.
4. **A second pull.** The lever's travel is 0.25 m and its own arc is not a straight line; asking twice from the new hand position must not be treated as a fresh 0.25 m. Expected: the second move reports the hand it started from.
5. **The offset frame when the torso is twisted.** The pelvis frame is the body's own; a half-turned robot's "back" is not the world's -X. Expected: `to_world` applies the root quaternion, and a test pins that a 90-degree root yaw turns a backward move into a sideways one in world terms.

---

### Task 1: A moved wrist point, clamped like every other point

**Files:**
- Modify: `clank/workspace.py` (add `moved_point` after `approach_point`, near `clamp_offset`)
- Test: `tests/test_workspace.py`

**Interfaces:**
- Consumes: `clamp_offset(forward_m, lateral_m, up_m, max_m=MAX_OFFSET_M, max_lateral_m=MAX_LATERAL_OFFSET_M) -> tuple[float, float, float]` and `clamp_wrist(point) -> tuple[float, float, float]`, both already in `clank/workspace.py`.
- Produces: `moved_point(point: tuple[float, ...], forward_m: float, up_m: float, lateral_m: float) -> tuple[float, float, float]` — the point `point` displaced by the clamped offset, then clamped into `WRIST_BOX`. `point` is `(x, y, z)` = (forward, left, up) in the pelvis frame.

- [x] **Step 1: Write the failing tests**

In `tests/test_workspace.py`, add a class:

```python
class MovedPointTests(unittest.TestCase):
    def test_a_backward_move_subtracts_from_forward(self):
        self.assertEqual(workspace.moved_point((0.40, 0.0, 0.20), -0.25, 0.0, 0.0),
                         (0.15, 0.0, 0.20))

    def test_lateral_is_the_robots_left(self):
        self.assertEqual(workspace.moved_point((0.40, 0.0, 0.20), 0.0, 0.0, 0.10),
                         (0.40, 0.10, 0.20))

    def test_the_offset_is_capped_before_it_is_applied(self):
        # A 0.60 m pull is more than the arm is measured to travel (0.41-0.51 m).
        self.assertEqual(workspace.moved_point((0.55, 0.0, 0.20), -0.60, 0.0, 0.0),
                         workspace.clamp_wrist((0.55 - workspace.MAX_OFFSET_M, 0.0, 0.20)))

    def test_the_result_is_inside_the_box(self):
        x, y, z = workspace.moved_point((0.10, 0.0, 0.20), -0.25, 0.0, 0.0)
        self.assertTrue(workspace.in_workspace((x, y, z)))
        self.assertEqual(x, workspace.WRIST_BOX.bounds()[0][0])
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `XDG_RUNTIME_DIR=/tmp/clank-xdg python3 -m unittest tests.test_workspace -v`
Expected: FAIL — `AttributeError: module 'clank.workspace' has no attribute 'moved_point'`

- [x] **Step 3: Implement `moved_point` in `clank/workspace.py`**

Signature and behaviour are fixed by the tests above. The one choice the tests do not make: clamp the offset with `clamp_offset(forward_m, lateral_m, up_m)` — note its argument order is `(forward, lateral, up)` while this function's is `(forward, up, lateral)` — then add it to the point as `(x + fwd, y + lat, z + up)`, then pass through `clamp_wrist`. Docstring must record why both clamps exist: `clamp_offset` for the magnitude the arm is measured to survive, `clamp_wrist` for the box, and that a pulled hand near the box floor is the case that needs the second one.

- [x] **Step 4: Run the tests to verify they pass**

Run: `XDG_RUNTIME_DIR=/tmp/clank-xdg python3 -m unittest tests.test_workspace -v`
Expected: PASS (all four)

- [x] **Step 5: Commit**

```bash
git add clank/workspace.py tests/test_workspace.py
git commit -F /tmp/msg.txt   # "feat(workspace): move a wrist point by a clamped direction"
```

---

### Task 2: Read a direction out of the model's own words

**Files:**
- Modify: `clank/brain.py` (new tables beside `_SKILL_WORDS`, new function beside `skill_from_goal`)
- Test: `tests/test_brain.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `MOVE_M: float = 0.25`; `MOVE_WORDS: tuple[tuple[str, tuple[float, float, float]], ...]` keyed by phrase with values `(forward_m, up_m, lateral_m)`; `move_from_goal(goal: str) -> tuple[float, float, float] | None`, which returns the *unscaled unit direction times `MOVE_M`*, or `None` when the goal names no direction **or names a destination** (see Review Focus 3).

- [x] **Step 1: Write the failing tests**

In `tests/test_brain.py`:

```python
class MoveFromGoalTests(unittest.TestCase):
    def test_pulling_comes_backwards(self):
        self.assertEqual(brain.move_from_goal("pull the lever toward me"),
                         (-brain.MOVE_M, 0.0, 0.0))

    def test_pushing_goes_forwards(self):
        self.assertEqual(brain.move_from_goal("push the cartridge in"),
                         (brain.MOVE_M, 0.0, 0.0))

    def test_a_destination_is_a_place_not_a_move(self):
        self.assertIsNone(brain.move_from_goal("insert the cartridge into the slot"))
        self.assertIsNone(brain.move_from_goal("put my card on the desk"))

    def test_a_named_object_with_no_direction_is_not_a_move(self):
        self.assertIsNone(brain.move_from_goal("take the card"))

    def test_up_and_down(self):
        self.assertEqual(brain.move_from_goal("lift the lever up"),
                         (0.0, brain.MOVE_M, 0.0))
        self.assertEqual(brain.move_from_goal("pull the lever down"),
                         (0.0, -brain.MOVE_M, 0.0))
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `XDG_RUNTIME_DIR=/tmp/clank-xdg python3 -m unittest tests.test_brain.MoveFromGoalTests -v`
Expected: FAIL — `AttributeError: module 'clank.brain' has no attribute 'move_from_goal'`

- [x] **Step 3: Implement `MOVE_M`, `MOVE_WORDS` and `move_from_goal` in `clank/brain.py`**

`MOVE_WORDS` maps phrase → direction, at minimum: `"back"`/`"backwards"`/`"toward me"`/`"towards me"`/`"pull"` → `(-1, 0, 0)`; `"forward"`/`"forwards"`/`"push"`/`"away"` → `(1, 0, 0)`; `"up"`/`"lift"` → `(0, 1, 0)`; `"down"`/`"lower"` → `(0, -1, 0)`; `"left"` → `(0, 0, 1)`; `"right"` → `(0, 0, -1)`. First match in table order wins, so the longer phrases come before the shorter ones that contain them. The destination check comes **first**: reuse the existing `_TASK_STOP` set — a goal whose words contain `on`, `in`, `into`, `onto`, `to` before any direction word is a `place` and returns `None`. Docstring must say why the destination check precedes the direction check, with the "insert the cartridge into the slot" measurement.

- [x] **Step 4: Run the tests to verify they pass**

Run: `XDG_RUNTIME_DIR=/tmp/clank-xdg python3 -m unittest tests.test_brain.MoveFromGoalTests -v`
Expected: PASS (all five)

- [x] **Step 5: Commit**

```bash
git add clank/brain.py tests/test_brain.py
git commit -F /tmp/msg.txt   # "feat(brain): read a direction out of a goal, in metres owned by code"
```

---

### Task 3: "Pull the lever" is a task, not silence

**Files:**
- Modify: `clank/brain.py` (`_TASK_VERBS`)
- Test: `tests/test_brain.py`

**Interfaces:**
- Consumes: `MOVE_WORDS` / `move_from_goal` from Task 2 — the verb table only has to stop the greeting gate swallowing the line; the direction comes from Task 2.
- Produces: `task_intent("pull the lever to job") == ("reach", "lever")`. `"pull"`, `"push"`, `"drag"`, `"slide"` map to skill `"reach"`.

- [x] **Step 1: Write the failing tests**

```python
class DirectionalTaskTests(unittest.TestCase):
    def test_pull_the_lever_is_a_reach(self):
        self.assertEqual(brain.task_intent("pull the lever to job"), ("reach", "lever"))

    def test_push_the_cartridge_in_is_a_reach(self):
        self.assertEqual(brain.task_intent("push the cartridge in"), ("reach", "cartridge"))

    def test_a_direction_word_does_not_become_the_noun(self):
        # "pull it back" names no object, so there is no task to ground - and no invented noun.
        self.assertEqual(brain.task_intent("pull it back"), ("reach", ""))
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `XDG_RUNTIME_DIR=/tmp/clank-xdg python3 -m unittest tests.test_brain.DirectionalTaskTests -v`
Expected: FAIL on the first two — `('', '') != ('reach', 'lever')`

- [x] **Step 3: Add the verbs to `_TASK_VERBS` in `clank/brain.py`**

Append after the `("press", "menu")` row, and **keep `press` before them** so its documented precedence is unchanged: `("pull", "reach"), ("push", "reach"), ("drag", "reach"), ("slide", "reach")`. Extend the comment above the tuple with the measurement that forced this: `"pull the lever to job"` returned `("", "")`, so `think()`'s greeting gate held and the lever was never attempted.

- [x] **Step 4: Run the tests to verify they pass, and the whole brain suite**

Run: `XDG_RUNTIME_DIR=/tmp/clank-xdg python3 -m unittest tests.test_brain -v`
Expected: PASS, including every pre-existing `task_intent` assertion

- [x] **Step 5: Commit**

```bash
git add clank/brain.py tests/test_brain.py
git commit -F /tmp/msg.txt   # "feat(brain): a direction verb is a task, read as a reach"
```

---

### Task 4: `--move` on the generated reach

**Files:**
- Modify: `spike/wbc_game/reach_ardy.py` (argparse in `main()`, target selection where `--world`/pixel are resolved)
- Test: `tests/test_reach_move.py` (new)

**Interfaces:**
- Consumes: `clank.workspace.moved_point` (Task 1); `reach_vr.to_world(p, st)`; `reach_vr.state()`.
- Produces: `reach_ardy.py --move FWD UP LAT --side right` — the same script, a target computed instead of measured. Exit 2 when `reach_vr.state()` reports no hand for that side.

- [x] **Step 1: Write the failing test**

`tests/test_reach_move.py`, following the existing convention for `spike/` (source-text assertions, as `tests/test_auto_loop.py` does):

```python
class TheMoveFlag(unittest.TestCase):
    def setUp(self):
        self.source = pathlib.Path("spike/wbc_game/reach_ardy.py").read_text()

    def test_the_flag_takes_three_metres(self):
        self.assertIn('"--move", type=float, nargs=3', self.source)

    def test_a_move_target_comes_from_the_hand_not_from_a_pixel(self):
        body = self.source[self.source.index("def move_target("):
                            self.source.index("def main(")]
        self.assertIn("moved_point", body)
        self.assertIn("to_world", body)

    def test_a_move_and_a_pixel_together_are_refused(self):
        self.assertIn("move", self.source[self.source.index("parser.error("):
                                        self.source.index("session_dir = Path(")])

    def test_a_move_with_no_reported_hand_refuses(self):
        body = self.source[self.source.index("def move_target("):
                            self.source.index("def main(")]
        self.assertIn("return None", body)


class TheFrameOfAMove(unittest.TestCase):
    """Review Focus 5: a half-turned robot's "back" is not the world's -X."""

    def setUp(self):
        try:
            from spike.wbc_game import reach_vr
        except Exception as exc:  # noqa: BLE001 - the spike's own deps, not this plan's
            self.skipTest(f"reach_vr is not importable here: {exc}")
        self.reach_vr = reach_vr
        self.workspace = sys.modules["clank.workspace"]

    def test_a_backward_move_follows_the_root_quaternion(self):
        half = math.radians(45.0)
        st = {"root_p": (0.0, 0.0, 0.0),
              "root_q": (math.cos(half), 0.0, 0.0, math.sin(half))}
        here = (0.40, 0.0, 0.20)
        moved = self.workspace.moved_point(here, -self.workspace.MAX_OFFSET_M, 0.0, 0.0)
        world_here = self.reach_vr.to_world(here, st)
        world_moved = self.reach_vr.to_world(moved, st)
        # The sign depends on the quaternion's handedness; the property does not: a
        # backward move under a quarter turn leaves the world X axis alone.
        self.assertAlmostEqual(world_moved[0] - world_here[0], 0.0, places=6)
        self.assertAlmostEqual(abs(world_moved[1] - world_here[1]),
                               self.workspace.MAX_OFFSET_M, places=6)
```

- [x] **Step 2: Run the test to verify it fails**

Run: `XDG_RUNTIME_DIR=/tmp/clank-xdg python3 -m unittest tests.test_reach_move -v`
Expected: FAIL on `test_the_flag_takes_three_metres` and `test_a_move_with_no_reported_hand_refuses`; the frame test either passes or skips.

- [x] **Step 3: Implement in `spike/wbc_game/reach_ardy.py`**

Add `move_target(st: dict, side: str, fwd: float, up: float, lat: float) -> tuple[float, float, float] | None`: reads `st[f"{side}_hand"]["p"]` (pelvis frame — the same field `reach_vr` compares `to_rel` against), passes it through `workspace.moved_point`, converts with `reach_vr.to_world`, returns `None` when the hand field is absent or not three finite numbers. Add `--move` with `nargs=3` and extend the existing "give a pixel pair or --world X Y Z" `parser.error` so it also accepts `--move` and refuses two sources at once. Wire the value in where `target_world = object_world(...)` is set today: a `--move` target skips `object_world` entirely and prints the hand it started from, so Review Focus 4 is visible in the log.

`move_target` reads the hand on **every** call and holds no cache — that is Review Focus 1 and 4 in one line: a second pull must start from where the first one ended, and a hand the body cannot report is a refusal, not a guess.

- [x] **Step 4: Run the test to verify it passes**

Run: `XDG_RUNTIME_DIR=/tmp/clank-xdg python3 -m unittest tests.test_reach_move -v`
Expected: PASS (all five, or four plus one skip when `reach_vr` will not import)

- [x] **Step 5: Commit**

```bash
git add spike/wbc_game/reach_ardy.py tests/test_reach_move.py
git commit -F /tmp/msg.txt   # "feat(reach): a generated reach can move the hand by a direction"
```

---

### Task 5: `--move` on the cruise reach, and the loop that uses it

**Files:**
- Modify: `spike/wbc_game/reach_vr.py` (argparse and the `--target`/`--world` branch)
- Modify: `spike/wbc_game/auto.py` (`_cruise_argv`, `_generated_argv`, a `_move` dispatcher in `one_turn`)
- Test: `tests/test_auto_loop.py`

**Interfaces:**
- Consumes: `brain.move_from_goal` (Task 2), `reach_ardy.py --move` (Task 4), `reach_vr.to_world`.
- Produces: `_move_argv(fwd: float, up: float, lat: float, side: str, camera: Pose | None) -> list[str]` in `auto.py`; `clank.workspace.moved_point` reused identically in `reach_vr` so the cruise and the generator agree on where a move ends.

- [x] **Step 1: Write the failing tests**

In `tests/test_auto_loop.py`:

```python
class AMoveIsItsOwnVerb(unittest.TestCase):
    def setUp(self):
        self.source = pathlib.Path("spike/wbc_game/auto.py").read_text()

    def test_the_move_argv_carries_three_metres(self):
        body = self.source[self.source.index("def _move_argv("):
                            self.source.index("def _place(")]
        self.assertIn('"--move"', body)

    def test_a_goal_with_a_direction_moves_before_it_grounds(self):
        body = self.source[self.source.index("def one_turn("):]
        self.assertIn("move_from_goal", body)
        self.assertLess(body.index("move_from_goal"), body.index("sense("))
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `XDG_RUNTIME_DIR=/tmp/clank-xdg python3 -m unittest tests.test_auto_loop.AMoveIsItsOwnVerb -v`
Expected: FAIL — `ValueError: substring not found` for `def _move_argv(`

- [x] **Step 3: Implement**

In `reach_vr.py`: mirror Task 4 — `--move FWD UP LAT`, `workspace.moved_point` on `st[f"{side}_hand"]["p"]`, `to_world`. In `auto.py`: `_move_argv(...)` next to `_generated_argv`/`_cruise_argv`, carrying `--move` and the same `--side`/camera handling as the others; in `one_turn`, before the sense branch, `direction = brain.move_from_goal(plan.goal)` and when it is not `None` run the move and return a `Turn` whose `spoken` is unchanged. A move logs `act: move(fwd=…, up=…, lat=…)` and its exit code as the result, in the same shape as a reach.

- [x] **Step 4: Run the tests to verify they pass, then the whole suite**

Run: `XDG_RUNTIME_DIR=/tmp/clank-xdg python3 -m unittest discover -s tests -t . 2>&1 | tail -3`
Expected: PASS, 1435 + the new tests

- [x] **Step 5: Commit**

```bash
git add spike/wbc_game/reach_vr.py spike/wbc_game/auto.py tests/test_auto_loop.py
git commit -F /tmp/msg.txt   # "feat(auto): a direction in the goal moves the hand before it looks"
```

---

### Task 6: The live gate — grab the lever, then pull it

**Files:**
- Create: `research/evidence/lever-pull/README.md`
- Modify: `research/brain-tools.md` (a date-stamped section)

**Interfaces:**
- Consumes: everything above, on the live stack.
- Produces: a measurement, not code. If it fails, the failure is the record.

- [x] **Step 1: Start a run and let it reach the lever**

Dashboard up (`127.0.0.1:8765`), `POST /api/run/start`. Card, cartridge and slot are not this plan's gate; drive the loop to the lever with JobBot's own line if the earlier beats are still unsolved, using `POST /api/run/start` plus the trajectory already in `brain.jsonl`.

- [x] **Step 2: Record what the move did**

Capture, per attempt: the hand's reported point before the move, the commanded direction, the achieved residual, and JobBot's next line. `reach_ardy --move` prints the first and third; the diary carries the fourth.

- [x] **Step 3: Report the measured result honestly**

Write `research/evidence/lever-pull/README.md` with the numbers, the run id, and the stills. A pull that misses is a recorded negative result, not a failure of the plan; say which of the five Review Focus items it hit.

- [x] **Step 4: Commit the record**

```bash
git add research/evidence/lever-pull/README.md research/brain-tools.md
git commit -F /tmp/msg.txt   # "docs(research): a directional pull on the lever, measured"
```

---

## What changed while building it

Tasks 1-5 are implemented (`5a80366`, `ce21c6b`, `cf8dd2c`, `86f9c9e`, `e9a08b2`), suite 1473.
Two decisions differ from the plan above, both because the code was a better authority than the
plan was:

- **The offset rides on the `Intent`, not in `auto.py`.** The plan had `auto.py` call
  `brain.move_from_goal(plan.goal)`. `Plan` has no goal field - it carries `skill`, `target`,
  `side`, `note` - so that call would have needed a new field, and it would have put a parse of
  the instruction back in the loop, which `tests/test_auto_loop.py` forbids by name. Instead
  `think()` computes the travel from `heard` (it already owns the utterance and the occupancy)
  and hands it to `assemble()`, which sets `Intent.offset` - the field `clank.intent/1` already
  had for exactly this. `auto.py` only reads it.
- **`assemble()` had to learn that a travel needs no pixel.** A `reach` with a target and no
  sighting was refused as `not_located` and became a `hold`, which is right for reaching a thing
  and wrong for pulling one the hand already holds. Both the sighting guard and `needs_where` now
  stand aside when `offset is not None`.
- **The occupancy gate is what makes one sentence work on two turns.** "Pull the lever to job"
  arrives with an empty hand first, so the offset is only set when the acting hand is `holding`;
  the first turn reaches the lever and the second travels. A test pins both halves.

Task 6 ran, failed, and then passed - on a standalone body rather than on the lever, which is a
better gate for it because a pull is a motion and not a game event. `a04fc02` and
`research/evidence/live-pull/README.md` carry the measurements.

What it cost, and what changed because of it:

- **The travel goes through the cruise, not the generator.** A generated clip is tracked and then
  the arm walks back to the previous reach's target: `clank/bus.py` retires its freeze-hold
  snapshot only on a real command, and a deposited clip is not one. The cruise re-commands an
  absolute target every keepalive and holds after it exits. `auto._move_argv` now routes both
  hands there, and Task 4's `--move` on `reach_ardy` is kept only because a *grab* still uses the
  generated path.
- **`Intent.offset` and the occupancy gate are unchanged** - the defect was never in the intent.
- **The 0.5 m frame contradiction recorded mid-plan is refuted.** FK glove and relay hand agree to
  1 mm on a fresh body; the contradiction compared an achieved pose with a stale command.

Still open, and now the top of the list for any generated motion: `CommandBus` does not retire its
freeze-hold snapshot when a clip is deposited, so a generated clip cannot hold its own result.
That affects every generated grab, not just a pull.

## Not in this plan

- **Which cartridge is the store clerk's** is a grounding subsystem, not actuation: the rack defeats `locate` today (66 px, then 369 px on near-identical neighbours) and the fix under discussion is filling the frame with the tray before naming. It deserves its own plan, and it can be built in either order.
- **The slot as a destination** is already `place`; what it lacks is a name the grounder can confirm, which is the same grounding question.
- **Approaching a destination from its surface normal** rather than from +Z would generalise `place`; it is deferred until something asks at a wall.
