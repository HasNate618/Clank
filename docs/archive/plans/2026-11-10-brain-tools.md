# Brain Tools Implementation Plan

> **Archived 2026-11-13.** Implementation plan, carried out. Kept for the task breakdown and the gates it defined; the code is the record now.
> Current behaviour: [`README.md`](../../../README.md) · agent rules: [`AGENTS.md`](../../../AGENTS.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the live loop a chooser: Omni picks `locate` / `look` / `reach` / `grab` / `release` / `hold` from the still and the transcript, and `auto.py` only runs `locate.py` when the model just named it.

**Architecture:** Restore constitution `plan()` as the live decider. Add `locate` as a planner-only sense skill (not an `Intent`). `WorkingMemory` holds pixels in code; the live block only says `present` / `not_visible` / `too_far`. New `think()` returns a `Thought`. `auto.py` becomes a small tool runner. `task_intent` / `next_step` / `sequence` stay in the tree but leave the live path.

**Tech Stack:** Python 3.14 stdlib in `clank/` (`unittest`, `unittest.mock`). `python3 -m unittest` (pytest is not installed). Live locate still runs under `spike/e3/.venv`. LFM3-Omni on `:11434` for Task 4 and Task 6 only.

**Spec:** `docs/superpowers/specs/2026-11-10-brain-tools.md`. Parent: `docs/superpowers/specs/2026-10-31-brain.md` §§2, 7, 9, 10. Intent: `clank/intent.py`. Locate: `spike/wbc_game/locate.py`.

## Global Constraints

- `clank/` stays **standard-library only**. Do not import `spike/wbc_game/locate.py` from `clank/`.
- Unittest **must not** open `127.0.0.1:11434`, `:4242`, `:5901`, `:8765`, `:8770`, or `:8771`. Patch `clank.brain._chat`.
- Do **not** edit `run.sh`, `config.toml`, `config.example.toml`, `flake.nix`, `clank/processes.py`, `clank/portowner.py`, `clank/orchestrator.py`, `clank/config.py`, `clank/depth_stream.py`, or `spike/c0/launch_monado.sh`.
- Prompt may **never** enumerate object positions. Generator JSON has no `pixel` field.
- Do **not** add `insert`, `pull`, `place`, or `rest` to the constitution. Do **not** put a `grab|place|reach|release` enum on `NEXT_STEP_PROMPT`.
- Do **not** auto-call `locate.py` on a `not_located` refuse.
- Grab is never automatic on `glow=yes`.
- `temperature: 0`. Commit only paths this plan names. Conventional commits, lowercase, no embedded double quotes; `git commit -F /tmp/msg.txt`.
- Existing `run_turn` behaviour (always GROUND after PLAN) stays; the live path is `think`, not `run_turn`.
- After any source mutation test, `find . -type d -name __pycache__ -path './tests/*' -o -path './clank/*' | xargs rm -rf` is enough; do not hunt the whole tree.

## Out of scope

- Reach residual / wrist-yaw / heading (body). A diary that locates then reaches and saturates is a body fail, not a brain fail.
- Wiring `place` / cartridge rack locator.
- Deleting `task_intent` / `next_step` / `sequence` from `clank/brain.py`.
- Rewriting `spike/wbc_game/do_task.py` (manual helper; later).
- Glow/ghost detectors, freeze-hold rewrite, walking.

## Review Focus

1. **Parser loop.** `auto.py` calling `task_intent` / `next_step` / `sequence` ships the fake brain again. Task 5 grep test.
2. **Hidden locate.** A `LOOK_DEG` loop that locates on every yaw, ignoring PLAN. Task 5 grep test.
3. **Pixel in the prompt.** `live_block` interpolating `pixel` or metres. Task 2.
4. **`locate` on the socket.** `Intent(skill="locate")` passing `validate_intent`. Task 1.
5. **Auto-locate on refuse.** `not_located` then calling `locate.py` in `think` or `assemble`. Task 3.

---

### Task 1: `locate` is a planner skill, not an Intent

**Files:**
- Modify: `clank/brain.py` (`PLANNER_SKILLS`, `PLANNER_SIDE_SKILLS`, `CONSTITUTION`, `plan()`)
- Test: `tests/test_brain.py` (`PlannerSkills`, `PromptText`, `PlanParser`)

**Interfaces:**
- Consumes: existing `plan()`, `extract_json_object`, `CONSTITUTION`.
- Produces: `PLANNER_SKILLS = ("hold", "look", "locate", "reach", "grab", "release")`, `SENSE_SKILLS = ("locate",)`, `MOTOR_SKILLS = ("hold", "look", "reach", "grab", "release")`, `PLANNER_SIDE_SKILLS` unchanged (`reach`, `grab`, `release`). Constitution text exactly as spec §5.

- [ ] **Step 1: Write the failing tests** in `tests/test_brain.py`

Replace `PlannerSkills.test_the_planner_skills_are_a_subset_of_the_intent_schema` (it will go false once `locate` is a planner skill). Add:

```python
def test_sense_tools_are_not_intent_skills(self) -> None:
    self.assertEqual(set(brain.SENSE_SKILLS), {"locate"})
    self.assertTrue(set(brain.SENSE_SKILLS).isdisjoint(SKILLS))
    self.assertEqual(set(brain.PLANNER_SKILLS),
                     set(brain.MOTOR_SKILLS) | set(brain.SENSE_SKILLS))
    self.assertLessEqual(set(brain.MOTOR_SKILLS), set(SKILLS))

def test_locate_is_accepted_with_side_none(self) -> None:
    # ChatPatchMixin reply '{"skill":"locate","side":"none","target":"card"}'
    # plan(...).ok and plan.skill == "locate" and plan.target == "card"

def test_locate_with_a_side_is_rejected(self) -> None:
    # '{"skill":"locate","side":"right","target":"card"}' -> not ok

def test_locate_without_a_target_is_rejected(self) -> None:
    # '{"skill":"locate","side":"none","target":null}' -> not ok
```

In `PromptText.test_constitution_schema_has_no_pixel_key`, keep the no-pixel and banned-skill asserts; add `self.assertIn("locate", brain.CONSTITUTION.lower())`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_brain.PlannerSkills tests.test_brain.PromptText -v`

Expected: FAIL on `SENSE_SKILLS` / `locate` not in constitution / subset assert still expecting five skills.

- [ ] **Step 3: Implement in `clank/brain.py`**

Set the three tuples. Replace `CONSTITUTION` with spec §5 verbatim. In `plan()`, treat `locate` like `look` for the side rule; after the existing target-type check, if `skill == "locate"` and not `target`, set `error` to `locate needs a target`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_brain.PlannerSkills tests.test_brain.PromptText tests.test_brain.PlanParser -v`

Expected: PASS. Then `python3 -m unittest tests.test_brain tests.test_next_step -v` still PASS (`run_turn` unchanged).

- [ ] **Step 5: Commit**

```bash
git add clank/brain.py tests/test_brain.py
git commit -F /tmp/msg.txt
```

Message: `feat(brain): locate is a planner sense tool, not an intent`

---

### Task 2: Working memory binds pixels in code

**Files:**
- Modify: `clank/brain.py` (new types + `assemble` + `live_block` last-line outcomes)
- Test: `tests/test_brain.py` (new class `WorkingMemoryTests`, extend `Occupancy` / assemble tests)

**Interfaces:**
- Consumes: Task 1 `SENSE_SKILLS`, existing `assemble`, `last_line`.
- Produces:

```python
@dataclass(frozen=True)
class Sighting:
    noun: str
    present: bool
    pixel: tuple[float, float] | None = None      # full-frame [0, 1]
    pixel_px: tuple[float, float] | None = None
    metres: float | None = None

@dataclass
class WorkingMemory:
    sightings: dict[str, Sighting] = field(default_factory=dict)
    def remember(self, noun: str, sighting: Sighting) -> None: ...
    def find(self, noun: str) -> Sighting | None: ...
```

`find`: case-insensitive; exact key, else substring either way (`"card"` finds `"JobBot card"`). `remember` keys on `noun.strip().lower()`.

`assemble(..., memory: WorkingMemory | None = None)`:
- `skill == "locate"` → `Decision` of `hold` / `side=none` / `source=vlm`, `note="locate"`, `reason=""` (accepted freeze while the caller senses). Do not call `validate_intent` on a locate Intent.
- `skill in {"reach", "grab"}` and `target` set: if `memory` is None or `memory.find(target)` is None or `present is False` → hold, `reason` starts with `not_located`.
- Else `Intent.pixel = sighting.pixel`.
- `look` with a target does **not** require a Sighting (look is how you search). Occupancy / glow / ghost gates unchanged.

- [ ] **Step 1: Write the failing tests**

```python
class WorkingMemoryTests(unittest.TestCase):
    def test_card_finds_jobbot_card(self) -> None:
        mem = brain.WorkingMemory()
        mem.remember("JobBot card", brain.Sighting(
            noun="JobBot card", present=True, pixel=(0.8, 0.4)))
        found = mem.find("card")
        self.assertIsNotNone(found)
        self.assertEqual(found.pixel, (0.8, 0.4))

    def test_missing_noun_is_none(self) -> None:
        self.assertIsNone(brain.WorkingMemory().find("card"))

class AssembleLocate(unittest.TestCase):
    def test_locate_does_not_emit_a_locate_intent(self) -> None:
        d = brain.assemble(Plan(skill="locate", side="none", target="card"),
                           None, seq=0, t_mono_ns=0)
        self.assertEqual(d.intent.skill, "hold")
        self.assertEqual(d.note, "locate")
        self.assertTrue(d.accepted)

    def test_reach_without_a_sighting_is_not_located(self) -> None:
        d = brain.assemble(Plan(skill="reach", side="right", target="card"),
                           None, seq=0, t_mono_ns=0, memory=brain.WorkingMemory())
        self.assertFalse(d.accepted)
        self.assertIn("not_located", d.reason)
        self.assertEqual(d.intent.skill, "hold")

    def test_reach_with_a_sighting_fills_the_pixel(self) -> None:
        mem = brain.WorkingMemory()
        mem.remember("card", brain.Sighting(
            noun="card", present=True, pixel=(0.81, 0.45)))
        d = brain.assemble(Plan(skill="reach", side="right", target="card"),
                           None, seq=0, t_mono_ns=0, memory=mem)
        self.assertTrue(d.accepted)
        self.assertEqual(d.intent.skill, "reach")
        self.assertEqual(d.intent.pixel, (0.81, 0.45))

    def test_live_block_does_not_carry_coordinates(self) -> None:
        text = brain.live_block(
            "take my card", "empty", "empty", 0.0, "unknown",
            "locate none target=card → present")
        self.assertNotIn("0.8", text)
        self.assertNotIn("px", text.lower())
        self.assertNotIn("TOP-LEFT", text)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_brain.WorkingMemoryTests tests.test_brain.AssembleLocate -v`

Expected: FAIL with `WorkingMemory` / `Sighting` not defined, `assemble` unexpected kwarg `memory`.

- [ ] **Step 3: Implement `Sighting`, `WorkingMemory`, and the `assemble` cases in `clank/brain.py`**

Default `memory=None` so every existing `assemble(...)` call still means "no sightings". When `memory is None` and skill is `reach`/`grab` with a target, keep today's GROUND-pixel path (`ground_result.pixel`) so `run_turn` tests stay green. The `not_located` refuse is only when `memory` is passed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_brain tests.test_next_step -v`

Expected: PASS, including old `CueGates` / `RunTurnCrop`.

- [ ] **Step 5: Commit**

Message: `feat(brain): bind locate pixels in working memory, never in the prompt`

---

### Task 3: `think()` is the live chooser

**Files:**
- Modify: `clank/brain.py` (new `Thought`, `think()`)
- Test: `tests/test_brain.py` (new class `ThinkTests` using `ChatPatchMixin` / `ScriptedChat`)

**Interfaces:**
- Consumes: `plan()`, `assemble()`, `WorkingMemory`, `live_block`.
- Produces:

```python
@dataclass
class Thought:
    plan: Plan
    decision: Decision
    sense: str | None = None   # "locate" or None

def think(
    config: BrainConfig,
    seed: str,
    frame: Frame,
    heard: str,
    history: list[str],
    memory: WorkingMemory,
    *,
    seq: int = 0,
    left: str = "empty",
    right: str = "empty",
    yaw_rad: float = 0.0,
    glow: str = "unknown",
    ghost: str = "unknown",
    last: str = "(none)",
    default_side: str = "right",
) -> Thought:
```

Body: `live = live_block(...)`; `plan_result = plan(config, seed, frame, live, transcript=history)` with no `wav` (transcribe stays a separate call); `decision = assemble(plan_result, None, ..., memory=memory)`; `sense = "locate" if plan_result.ok and plan_result.skill == "locate" else None`. Do not import or call `locate.py`. Do not call `ground()`.

- [ ] **Step 1: Write the failing tests**

```python
class ThinkTests(ChatPatchMixin, unittest.TestCase):
    def test_take_my_card_chooses_locate(self) -> None:
        self.reply = '{"skill":"locate","side":"none","target":"card"}'
        thought = brain.think(
            brain.BrainConfig(), "", fake_frame(),
            "Hey, Hillman, take my card.", [], brain.WorkingMemory())
        self.assertEqual(thought.sense, "locate")
        self.assertEqual(thought.plan.target, "card")
        self.assertEqual(thought.decision.intent.skill, "hold")

    def test_reach_without_memory_is_not_located(self) -> None:
        self.reply = '{"skill":"reach","side":"right","target":"card"}'
        thought = brain.think(
            brain.BrainConfig(), "", fake_frame(),
            "take my card", [], brain.WorkingMemory())
        self.assertIsNone(thought.sense)
        self.assertIn("not_located", thought.decision.reason)

    def test_reach_after_remembering_emits_reach(self) -> None:
        self.reply = '{"skill":"reach","side":"right","target":"card"}'
        mem = brain.WorkingMemory()
        mem.remember("card", brain.Sighting(
            noun="card", present=True, pixel=(0.8, 0.4)))
        thought = brain.think(
            brain.BrainConfig(), "", fake_frame(),
            "take my card", [], mem)
        self.assertEqual(thought.decision.intent.skill, "reach")
        self.assertEqual(thought.decision.intent.pixel, (0.8, 0.4))

    def test_think_does_not_send_audio(self) -> None:
        self.reply = '{"skill":"hold","side":"none","target":null}'
        brain.think(brain.BrainConfig(), "", fake_frame(),
                    "take my card", [], brain.WorkingMemory())
        content = self.chat_calls[0]["messages"][-1]["content"]
        kinds = [p["type"] for p in content]
        self.assertNotIn("input_audio", kinds)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_brain.ThinkTests -v`

Expected: FAIL `think` is not defined.

- [ ] **Step 3: Implement `Thought` and `think()` in `clank/brain.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_brain tests.test_next_step -v`

Expected: PASS.

- [ ] **Step 5: Commit**

Message: `feat(brain): think() chooses locate or a motor, and never calls locate itself`

---

### Task 4: Measure Omni speech-bind before rewriting the loop

**Files:**
- Create: `spike/wbc_game/eval_plan.py`
- Do not wire `auto.py` in this task.

**Interfaces:**
- Consumes: `brain.plan`, `brain.BrainConfig`, `brain.server_status`, `clank.frames.Frame`.
- Produces: a CLI that talks to `:11434` and is **not** collected by `python3 -m unittest`.

```text
spike/e3/.venv/bin/python spike/wbc_game/eval_plan.py \
  --image PATH --heard TEXT [--expect SKILL] [--contains SUBSTR]
```

Exit 0 iff `server_status().up` and `plan.ok` and (`--expect` absent or `plan.skill == expect`) and (`--contains` absent or `--contains` in `(plan.target or "").lower()`). Print one JSON object: `skill`, `side`, `target`, `raw`, `error`, `prompt_ms`, `decode_ms`. If the server is down, exit 2 with `server down` on stderr.

Still: prefer a live auto frame `~/.local/state/Clank/auto/frame-*.png` (newest) if it shows the card; otherwise `research/evidence/grounding-seed-view.png`. Heard text: `Hey, Hillman, take my card.`

- [ ] **Step 1: Write `eval_plan.py`** (no unittest; the script *is* the test harness)

Stdlib + `clank.brain` / `clank.frames` only. Read the image as JPEG bytes (`ffmpeg` if the file is PNG, same pattern `clank.frames` already uses, or pass a `.jpg`). Call `brain.plan(config, brain.DEFAULT_SEED, frame, brain.live_block(heard, "empty", "empty", 0.0, "unknown", "(none)"))`.

- [ ] **Step 2: Run the speech-bind arm against Omni**

Run:

```bash
spike/e3/.venv/bin/python spike/wbc_game/eval_plan.py \
  --image <still with the card> \
  --heard 'Hey, Hillman, take my card.' \
  --expect locate --contains card
```

Expected: exit 0, `skill=locate`, target contains `card`.

If it emits `reach` + `card` instead: shorten CONSTITUTION, do **not** add a museum sequence, re-run once. If it still will not emit `locate`, **stop and report the raw JSON** — do not implement Task 5 with a hidden locate.

Also run, same still:

```bash
spike/e3/.venv/bin/python spike/wbc_game/eval_plan.py \
  --image <still> --heard '(none)' --expect hold
```

`look` is an allowed alternate if `--expect` is relaxed by the implementer; `grab` is not. Record both raw replies in `research/brain-tools.md` (short, measured table).

- [ ] **Step 3: Commit the script and the measurement note**

Message: `test(brain): frozen-still speech-bind for locate`

Do not start Task 5 unless the locate expect passed.

---

### Task 5: `auto.py` is a tool runner

**Files:**
- Modify: `spike/wbc_game/auto.py` (`one_turn`)
- Test: `tests/test_auto_loop.py` (new; grep-style, no game)

**Interfaces:**
- Consumes: `brain.think`, `brain.WorkingMemory`, `brain.Sighting`, `brain.last_line`, existing `ready`, `frame_and_jpeg`, `locate.locate`, `grab_pixel`, `look`, `confirm`.
- Produces: `SENSE_BUDGET = 3`. `one_turn` does not call `brain.next_step`, `brain.task_intent`, or `brain.sequence`. `LOOK_DEG` is not used as an automatic locate sweep. `look()` is only called when `thought.decision.intent.skill == "look"` (use `intent.yaw_rad` if set, else 0).

Loop body per spec §8. Depth gate stays: after `locate.locate`, if metres > `MAX_OBJ_M` remember `present=True` but outcome `too_far` (so PLAN can look); if metres < `reach_vr.MIN_SURFACE_M`, treat as not the object (`not_visible`). Pixel stored only when the gate passes. `grab` with a Sighting still calls `grab_pixel` (the measured close-inside-reach). `reach` with a Sighting calls the same `reach_vr.py` **without** `--grab`.

- [ ] **Step 1: Write the failing grep tests** in `tests/test_auto_loop.py`

```python
import pathlib
import unittest

SRC = pathlib.Path("spike/wbc_game/auto.py").read_text()

class AutoLoop(unittest.TestCase):
    def test_does_not_parse_the_instruction(self) -> None:
        self.assertNotIn("task_intent", SRC)
        self.assertNotIn("next_step", SRC)
        self.assertNotIn("sequence(", SRC)

    def test_thinks_before_it_locates(self) -> None:
        self.assertIn("brain.think", SRC)
        self.assertIn("SENSE_BUDGET", SRC)
        self.assertIn('thought.sense == "locate"', SRC)

    def test_does_not_sweep_look_deg_to_locate(self) -> None:
        self.assertNotIn("for attempt, dyaw in enumerate(LOOK_DEG", SRC)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_auto_loop -v`

Expected: FAIL (`task_intent` still in `auto.py`).

- [ ] **Step 3: Rewrite `one_turn` in `spike/wbc_game/auto.py`**

Keep `ready()`, transcribe-first, `confirm()`, `grab_pixel`, diary logging. Drop the parser and the `LOOK_DEG` locate sweep. `LOOK_DEG` may remain as a named constant unused, or be deleted; the grep test only bans the enumerate loop.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_auto_loop tests.test_brain tests.test_next_step -v`

Expected: PASS.

Dry run (game may be up; no JobBot required):

```bash
spike/e3/.venv/bin/python spike/wbc_game/auto.py --for 0 --say "take my card" --plan-only
```

`--plan-only` must still capture a still and call `think`, logging `think: locate(...)` (or whatever Omni returns) **without** calling `locate.py`. If `--plan-only` currently returns before `think`, extend it so it runs `think` once and returns.

- [ ] **Step 5: Commit**

Message: `feat(auto): run locate only when the brain chooses it`

---

### Task 6: Live JobBot diary

**Files:**
- None required. Measurement only. Append the diary excerpt to `research/brain-tools.md`.

**Interfaces:**
- Consumes: Task 5 `auto.py`, live session on `:8765`, JobBot intro.

- [ ] **Step 1: Restart the session and the loop**

Body reset happens inside `ready()`. Heading stays at the config default (90) via `~/.local/state/Clank/relay-yaw.json`. Do not edit `config.toml`.

```bash
spike/e3/.venv/bin/python spike/wbc_game/auto.py --for 180
```

- [ ] **Step 2: Watch the diary, not the exit code**

Run: `python3 -m clank.brainlog --tail 30`

Pass (spec §10 live):

1. After "take my card", a `think` line names `locate` and a target containing `card` **before** `act: locate`.
2. No line `the task says ... the task wins`.
3. After `locate` present, a `think` names `reach` or `grab`.
4. Body was standing (rest) before the first still.

Fail the *brain* if (1) or (2) fail. Fail the *body* if (3) happens and the hand saturates short — record the residual, do not put the parser back.

- [ ] **Step 3: Commit the measurement note** if the diary passed or if it failed in a way that changes the spec (raw PLAN JSON).

Message: `docs(brain): live diary of the locate chooser`

---

## Self-review

1. **Spec coverage:** §3 tuples → T1. §6 memory → T2. §4/`think` → T3. §10 Omni arm → T4. §8 loop + review 1–2 → T5. §10 live → T6. Parent `run_turn` left alone. `do_task.py` explicitly out of scope.
2. **Step scan:** each step is one test, one fail run, one signature, one pass run, or one commit. Omni raw JSON on Task 4 fail is a stop, not a hidden locate.
3. **Types:** `Sighting`, `WorkingMemory`, `Thought`, `think()` names match across T2–T5. `assemble(..., memory=)` is optional.
4. **Review Focus:** all five have a named test in T1, T2, T3, or T5.
5. **Proportion:** signatures and assertions, not function bodies, except the constitution which the spec already pins.
