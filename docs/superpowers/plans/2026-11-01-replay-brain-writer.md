# Replay Brain Writer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the replay brain writer match `docs/superpowers/specs/2026-10-31-brain.md` §6.2 / §7 / §10 / §11 on frozen stills, with tests that never talk to the model server or launch the game.

**Architecture:** Keep PLAN / GROUND / ASSEMBLE as three functions in `clank/brain.py`. Tests patch `_chat` so every path is a value. GROUND becomes one name→quadrant question on a caller-supplied container crop; a pure `compose_pixel` maps that quadrant through the crop box into a full-frame pixel. ASSEMBLE gains glow/ghost gates. VERIFY is a separate namer on a wrist crop. Nothing posts to the intent socket.

**Tech Stack:** Python 3.14 stdlib only in `clank/` (`unittest`, `unittest.mock`, `urllib` already in `brain.py`). `python3 -m unittest` (pytest is not installed). `ffmpeg` already used by `ReplaySource`; used again only to crop JPEGs.

**Spec:** `docs/superpowers/specs/2026-10-31-brain.md` §§6.2, 7, 9, 10, 11, 14, 19 (this slice is P1 / B0 / GROUND swap / B2 unit, not B1). Parent: `docs/2026-10-23-clank-design.md` §9. Intent: `clank/intent.py`.

## Global Constraints

- `clank/` stays **standard-library only**. No NumPy, no PIL, no `requests`.
- Tests **must not** open `127.0.0.1:11434`, `:4242`, `:5901`, `:8765`, or `:8770`. Patch `clank.brain._chat`.
- Do **not** launch Steam, Proton, Monado, llama-server, or the game.
- Do **not** edit `run.sh`, `config.toml`, `config.example.toml`, `flake.nix`, `clank/processes.py`, `clank/portowner.py`, `clank/orchestrator.py`, `clank/config.py`, or `spike/c0/launch_monado.sh`.
- Do **not** add `insert`, `pull`, or `take_cartridge` to `PLANNER_SKILLS`. Do **not** import `clank.glow` from `clank.brain`.
- The prompt may **never** enumerate object positions. Identity (green STORE CLERK, slot, lever, ghost) is allowed. A layout table is not.
- Generator JSON has no `pixel` field. Only GROUND fills `Intent.pixel`.
- Grab is never automatic on `glow=yes`. Release is never automatic on `ghost=yes`.
- `temperature: 0`. Do not add majority vote.
- Commit only paths this plan names. Do not add the walkthrough `.mp4` / `.mp3`.
- Existing occupancy guards stay: `release` on empty and `grab` on holding still refuse.

## Out of scope (later plans)

- Freeze-hold / `rest` in `spike/wbc/sonic_executor.py` (P2).
- Wiring PLAN onto the intent socket (B1).
- Job Simulator blue/yellow detectors in `clank/glow.py` (P2c).
- Live card envelope (P2d). `VncSource`. Live `pw-record`. Container detection. Fine-tune.

## Review Focus

1. **Positions in the prompt.** `CONSTITUTION` and `DEFAULT_SEED` must not contain `TOP-LEFT`, `TOP-RIGHT`, `BOTTOM-LEFT`, or `BOTTOM-RIGHT`. Task 2 pins this.
2. **Pixel from the generator.** Constitution schema keys are exactly `skill`, `side`, `target`. Task 2 pins this.
3. **Glow is not an auto-grab.** `glow="yes"` plus PLAN `hold` stays `hold`. Task 3 pins this.
4. **Ground prompt is not a layout table.** The GROUND user text names only the asked noun, not the other three cartridges, unless that noun is the one asked. Task 4 pins this.
5. **Ghost gate is insert-only.** `release` of a card (`target="card"`) with `ghost="no"` is allowed when the hand is holding. `release` with `target="slot"` and `ghost="no"` is refused. Task 3 pins this.

---

### Task 1: Parser, occupancy, and a fake Frame

**Files:**
- Create: `tests/test_brain.py`
- Modify: none (land the untracked modules as they are, except do not add tests that lock the thirds grounder)
- Also land: `clank/brain.py`, `clank/frames.py` (already written, untracked)

**Interfaces:**
- Consumes: existing `clank.brain.plan`, `assemble`, `extract_json_object`, `Frame`.
- Produces: `tests/test_brain.py` helpers `fake_frame() -> Frame` and a `_chat` patcher that later tasks reuse.

- [ ] **Step 1: Write the failing tests** in `tests/test_brain.py`

```python
def fake_frame(index: int = 0) -> Frame:
    return Frame(index=index, t_s=float(index), jpeg=b"\xff\xd8\xff\xd9", source="test")

class ExtractJson(unittest.TestCase):
    def test_extracts_a_fenced_object(self) -> None:
        text = 'sure\n```json\n{"skill":"reach","side":"right","target":"card"}\n```'
        self.assertEqual(brain.extract_json_object(text)["skill"], "reach")

class PlanParser(unittest.TestCase):
    def test_action_is_accepted_as_skill(self) -> None:
        # patch _chat to return '{"action":"reach","target":"card"}'
        # plan(...).skill == "reach" and "side missing" in plan.note

    def test_skill_as_a_key_is_folded(self) -> None:
        # '{"reach":"right","target":"card"}' -> skill "reach", side "right"

    def test_insert_is_rejected(self) -> None:
        # '{"skill":"insert","side":"right","target":"slot"}' -> not ok

class Occupancy(unittest.TestCase):
    def test_release_on_an_empty_hand_is_held(self) -> None:
        d = brain.assemble(Plan(skill="release", side="right"), None, seq=0, t_mono_ns=0, right="empty")
        self.assertFalse(d.accepted)
        self.assertEqual(d.intent.skill, "hold")

    def test_grab_on_a_holding_hand_is_held(self) -> None:
        d = brain.assemble(Plan(skill="grab", side="right", target="card"), None, seq=0, t_mono_ns=0, right="holding")
        self.assertFalse(d.accepted)

    def test_not_visible_reach_becomes_a_look_scan(self) -> None:
        d = brain.assemble(
            Plan(skill="reach", side="right", target="card"),
            Ground(noun="card", error="not visible"),
            seq=0, t_mono_ns=0,
        )
        self.assertEqual(d.intent.skill, "look")
        self.assertEqual(d.note, "downgraded to scan")
```

Patch `_chat` by replacing `clank.brain._chat` in `setUp` with a function that returns `ModelReply(text=self.reply)`. Do not call `urlopen`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/nate/Projects/Clank && python3 -m unittest tests.test_brain -v`

Expected: `ModuleNotFoundError` if `clank/brain.py` is missing; otherwise some of these already pass. If they all pass against the untracked file, that is the land-as-is result — do not rewrite the parser. If `clank/brain.py` is absent, Step 3 copies it from git status untracked (it is already in the tree).

- [ ] **Step 3: Keep `plan()` / `assemble()` occupancy behaviour in `clank/brain.py`.** Do not change GROUND in this task.

- [ ] **Step 4: Re-run**

Run: `python3 -m unittest tests.test_brain -v` — all pass.

- [ ] **Step 5: Commit**

```bash
git add clank/brain.py clank/frames.py tests/test_brain.py
git commit -m "feat(brain): land replay writer with parse and occupancy tests"
```

---

### Task 2: Constitution, seed, live block

**Files:**
- Modify: `clank/brain.py` (`CONSTITUTION`, `DEFAULT_SEED`, `live_block`)
- Test: `tests/test_brain.py`

**Interfaces:**
- Consumes: Task 1 helpers.
- Produces: `CONSTITUTION: str`, `DEFAULT_SEED: str`, `live_block(heard: str, left: str, right: str, yaw_rad: float, glow: str, last: str, ghost: str = "unknown") -> str`.

Exact `DEFAULT_SEED` (verbatim, including newlines):

```
use the store clerk cartridge (green label).
a cartridge goes in the glowing slot; a yellow ghost means you may release.
pull the lever down to start the job.
```

Exact `live_block` body:

```
heard: {heard or "(none)"}
self:  left={left} right={right} yaw={yaw_rad:.2f} glow={glow} ghost={ghost}
last:  {last}
```

`CONSTITUTION` is spec §7.1 plus the measured line `All three keys are always present:` (keep it; omitting it made the model drop `side`) plus the three cue/occupancy sentences from §7.1 as amended. It must still list only `hold|look|reach|grab|release`. Keep `fire` / `strike` / `dodge` / `twitch` / `rest` absent.

- [ ] **Step 1: Write the failing tests**

```python
class PromptText(unittest.TestCase):
    def test_seed_names_the_green_cartridge_the_slot_and_the_lever(self) -> None:
        s = brain.DEFAULT_SEED.lower()
        self.assertIn("store clerk", s)
        self.assertIn("green", s)
        self.assertIn("ghost", s)
        self.assertIn("lever", s)

    def test_seed_and_constitution_do_not_list_positions(self) -> None:
        blob = brain.CONSTITUTION + "\n" + brain.DEFAULT_SEED
        for word in ("TOP-LEFT", "TOP-RIGHT", "BOTTOM-LEFT", "BOTTOM-RIGHT"):
            self.assertNotIn(word, blob)

    def test_constitution_schema_has_no_pixel_key(self) -> None:
        self.assertIn('"skill"', brain.CONSTITUTION)
        self.assertNotIn("pixel", brain.CONSTITUTION.lower())
        for banned in ("fire", "strike", "dodge", "twitch", "insert", "pull"):
            self.assertNotIn(banned, brain.CONSTITUTION.lower())

    def test_live_block_carries_glow_and_ghost(self) -> None:
        text = brain.live_block("take my card", "empty", "holding", 0.0, "yes", "reach right → done", ghost="no")
        self.assertIn("glow=yes", text)
        self.assertIn("ghost=no", text)
        self.assertIn("right=holding", text)
        self.assertNotIn("current goal", text.lower())
```

- [ ] **Step 2: Run** `python3 -m unittest tests.test_brain.PromptText -v` — fail on seed / ghost.

- [ ] **Step 3: Replace `CONSTITUTION`, `DEFAULT_SEED`, and `live_block` in `clank/brain.py`.** Update `run_turn` only enough to pass `ghost` through `live_block` (default `"unknown"`). `run_turn` still does not refuse on ghost; that is Task 3.

- [ ] **Step 4: Run** `python3 -m unittest tests.test_brain -v` — all pass.

- [ ] **Step 5: Commit** `feat(brain): seed and live block for glow and ghost`

---

### Task 3: Glow and ghost gates in `assemble`

**Files:**
- Modify: `clank/brain.py` `assemble`
- Test: `tests/test_brain.py`

**Interfaces:**
- Consumes: `assemble` from Task 1, `live_block` from Task 2.
- Produces: `assemble(..., glow: str = "unknown", ghost: str = "unknown") -> Decision`.

Cue rules (after occupancy, before `not_visible` look-downgrade):

| skill | extra | glow/ghost | result |
| --- | --- | --- | --- |
| `grab` | any | `glow=="no"` | hold; reason contains `glow is no` |
| `grab` | any | `glow` in `{yes, unknown}` | not refused for glow |
| `release` | `"slot"` in `target.lower()` | `ghost=="no"` | hold; reason contains `ghost is no` |
| `release` | `target=="card"` | `ghost=="no"`, hand holding | accepted (drop, not insert) |
| `hold` | — | `glow=="yes"` | stays `hold` |

`unknown` never refuses. Do not rewrite a refused grab into a release.

- [ ] **Step 1: Write the failing tests**

```python
class CueGates(unittest.TestCase):
    def test_grab_with_glow_no_is_held(self) -> None:
        d = brain.assemble(Plan(skill="grab", side="right", target="card"),
                           Ground(noun="card", pixel=(0.5, 0.5)),
                           seq=0, t_mono_ns=0, glow="no")
        self.assertFalse(d.accepted)
        self.assertEqual(d.intent.skill, "hold")
        self.assertIn("glow is no", d.reason)

    def test_unknown_glow_does_not_refuse_grab(self) -> None:
        d = brain.assemble(Plan(skill="grab", side="right", target="card"),
                           Ground(noun="card", pixel=(0.5, 0.5)),
                           seq=0, t_mono_ns=0, glow="unknown")
        self.assertTrue(d.accepted)
        self.assertEqual(d.intent.skill, "grab")

    def test_glow_yes_does_not_emit_grab_without_a_plan(self) -> None:
        d = brain.assemble(Plan(skill="hold", side="none"), None,
                           seq=0, t_mono_ns=0, glow="yes")
        self.assertEqual(d.intent.skill, "hold")

    def test_release_into_slot_without_ghost_is_held(self) -> None:
        d = brain.assemble(Plan(skill="release", side="right", target="slot"),
                           None, seq=0, t_mono_ns=0, right="holding", ghost="no")
        self.assertFalse(d.accepted)
        self.assertIn("ghost is no", d.reason)

    def test_release_of_a_card_does_not_need_ghost(self) -> None:
        d = brain.assemble(Plan(skill="release", side="right", target="card"),
                           None, seq=0, t_mono_ns=0, right="holding", ghost="no")
        self.assertTrue(d.accepted)
        self.assertEqual(d.intent.skill, "release")
```

- [ ] **Step 2: Run** `python3 -m unittest tests.test_brain.CueGates -v` — fail (`assemble` has no `glow`).

- [ ] **Step 3: Add `glow` and `ghost` parameters to `assemble` in `clank/brain.py`.** Thread them from `run_turn`. Slot test is substring `"slot"` in `(target or "").lower()`.

- [ ] **Step 4: Run** `python3 -m unittest tests.test_brain -v` — all pass.

- [ ] **Step 5: Commit** `feat(brain): refuse grab without glow and insert without ghost`

---

### Task 4: GROUND is a container quadrant

**Files:**
- Modify: `clank/brain.py` (`Ground`, `ground`, drop `THIRDS_ROWS` / `THIRDS_COLS` / `_CELL_LETTERS` if unused)
- Test: `tests/test_brain.py`

**Interfaces:**
- Consumes: `ModelReply`, `_chat`, `Frame`.
- Produces:
  - `QUADRANTS = ("TOP-LEFT", "TOP-RIGHT", "BOTTOM-LEFT", "BOTTOM-RIGHT")`
  - `compose_pixel(box: tuple[float, float, float, float], quadrant: str) -> tuple[float, float]`
  - `ground(config, frame, noun, box=(0.0, 0.0, 1.0, 1.0), container="tray", transcript=None) -> Ground`
  - `Ground.quadrant: str | None`, `Ground.pixel`, `Ground.raw`, `Ground.methods == "quad"`, `Ground.box`

`compose_pixel` uses the geometric centre of the quadrant inside `box`:

```
u = u0 + (u1 - u0) * ((1 if RIGHT else 0) + 0.5) / 2
v = v0 + (v1 - v0) * ((1 if BOTTOM else 0) + 0.5) / 2
```

Do **not** add per-cartridge calibration offsets. VERIFY is the precision path.

GROUND user text, exactly this shape (noun and container interpolated):

```
Which position in this {container} does the {noun} occupy? Answer with one of: TOP-LEFT, TOP-RIGHT, BOTTOM-LEFT, BOTTOM-RIGHT.
```

`max_tokens=60`. One `_chat` call, not two. Parse the first matching `QUADRANTS` token in the reply (same normalisation as `_match_word`: strip non-letters, allow `TOP RIGHT`). Unparseable or `NONE` / `NOT VISIBLE` → `error="not visible"`, `pixel=None`.

`Ground.to_dict()` reports `quadrant` (and may keep `cell` as an alias of `quadrant`). It must not require `row`/`col` for brainview.

- [ ] **Step 1: Write the failing tests**

```python
class ComposePixel(unittest.TestCase):
    def test_top_right_of_the_measured_tray(self) -> None:
        box = (0.270, 0.325, 0.548, 0.566)
        u, v = brain.compose_pixel(box, "TOP-RIGHT")
        self.assertAlmostEqual(u, 0.270 + (0.548 - 0.270) * 0.75)
        self.assertAlmostEqual(v, 0.325 + (0.566 - 0.325) * 0.25)

class GroundQuad(unittest.TestCase):
    def test_store_clerk_top_right_composes_into_the_box(self) -> None:
        # _chat returns "TOP-RIGHT"
        g = brain.ground(BrainConfig(), fake_frame(), "STORE CLERK",
                         box=(0.270, 0.325, 0.548, 0.566))
        self.assertEqual(g.quadrant, "TOP-RIGHT")
        self.assertEqual(g.methods, "quad")
        self.assertIsNotNone(g.pixel)
        self.assertAlmostEqual(g.pixel[0], 0.270 + (0.548 - 0.270) * 0.75)

    def test_prompt_names_only_the_asked_noun(self) -> None:
        captured: list[str] = []
        # wrap _chat to append the user text part to captured, return "TOP-LEFT"
        brain.ground(BrainConfig(), fake_frame(), "STORE CLERK")
        blob = captured[0]
        self.assertIn("STORE CLERK", blob)
        self.assertNotIn("AUTO MECHANIC", blob)
        self.assertNotIn("GOURMET CHEF", blob)
        self.assertNotIn("OFFICE WORKER", blob)
        self.assertIn("TOP-LEFT", blob)  # answer vocabulary, not a layout table

    def test_none_is_not_visible(self) -> None:
        # _chat returns "NONE"
        g = brain.ground(BrainConfig(), fake_frame(), "card")
        self.assertEqual(g.error, "not visible")
        self.assertIsNone(g.pixel)
```

- [ ] **Step 2: Run** `python3 -m unittest tests.test_brain.ComposePixel tests.test_brain.GroundQuad -v` — fail (`compose_pixel` missing; current `ground` asks thirds).

- [ ] **Step 3: Implement `compose_pixel` and replace `ground()` in `clank/brain.py`.** Delete the two-call thirds path. Update the `ground()` docstring: the 3×3 method is retired; cite spec §10.

- [ ] **Step 4: Run** `python3 -m unittest tests.test_brain -v` — all pass.

- [ ] **Step 5: Commit** `feat(brain): ground a noun by container quadrant`

---

### Task 5: VERIFY names the crop

**Files:**
- Modify: `clank/brain.py`
- Test: `tests/test_brain.py`

**Interfaces:**
- Consumes: `_chat`, `Frame`.
- Produces: `Verify` frozen dataclass `noun: str, heard: str, match: bool, raw: str, error: str` with `to_dict()`; `verify(config: BrainConfig, frame: Frame, noun: str) -> Verify`.

User text, exactly:

```
This is a close-up of one object. What is it? Reply with the name, or NONE.
```

`max_tokens=40`. `match` is true when every alphabetic word in `noun.lower()` appears in `heard.lower()` (so `"STORE CLERK"` matches `"the store clerk cartridge"`). `NONE` / empty / `_chat` error → `match=False`.

`run_turn` does **not** call `verify` yet (Task 6). Tests call `verify` directly.

- [ ] **Step 1: Write the failing tests**

```python
class VerifyCrop(unittest.TestCase):
    def test_store_clerk_matches_a_paraphrase(self) -> None:
        # _chat returns "the store clerk cartridge"
        v = brain.verify(BrainConfig(), fake_frame(), "STORE CLERK")
        self.assertTrue(v.match)

    def test_a_neighbour_does_not_match(self) -> None:
        # _chat returns "GOURMET CHEF"
        v = brain.verify(BrainConfig(), fake_frame(), "STORE CLERK")
        self.assertFalse(v.match)

    def test_none_does_not_match(self) -> None:
        # _chat returns "NONE"
        v = brain.verify(BrainConfig(), fake_frame(), "card")
        self.assertFalse(v.match)
```

- [ ] **Step 2: Run** `python3 -m unittest tests.test_brain.VerifyCrop -v` — fail (`verify` missing).

- [ ] **Step 3: Add `Verify` and `verify()` in `clank/brain.py`. Export both in `__all__`.

- [ ] **Step 4: Run** `python3 -m unittest tests.test_brain -v` — all pass.

- [ ] **Step 5: Commit** `feat(brain): verify a wrist crop by name`

---

### Task 6: `run_turn` crop path and brainview overlay

**Files:**
- Modify: `clank/brain.py` (`run_turn`, `TurnRecord.self_state`)
- Modify: `clank/frames.py` (add `crop_jpeg`)
- Modify: `spike/brainview/app.py` (pass box, draw quadrant on the box, show ghost)
- Modify: `.gitignore` (walkthrough media)
- Test: `tests/test_brain.py`, `tests/test_frames.py`

**Interfaces:**
- Consumes: `ground(..., box=)`, `assemble(..., glow=, ghost=)`, `verify`.
- Produces:
  - `crop_jpeg(jpeg: bytes, box: tuple[float, float, float, float], width: int = 1024) -> bytes` in `clank/frames.py`
  - `run_turn(..., ghost: str = "unknown", box: tuple[float, float, float, float] | None = None, container: str = "tray") -> TurnRecord`
  - brainview `TRAY_BOX = (0.270, 0.325, 0.548, 0.566)` passed as `box`

`crop_jpeg` shells ffmpeg the same way `ReplaySource.prepare` does, stdin JPEG / stdout JPEG:

```
ffmpeg -y -loglevel error -i pipe:0 -vf crop=iw*{u1-u0}:ih*{v1-v0}:iw*{u0}:ih*{v0},scale={width}:-1 -f mjpeg pipe:1
```

On `OSError` / non-zero exit, an empty stdout, or a degenerate box (`u0 >= u1` or `v0 >= v1`), raise `RuntimeError`. On a crop failure `run_turn` falls back to grounding the whole frame and records `self_state["crop_error"]`, rather than skipping GROUND — losing accuracy beats losing the turn, and the degradation stays visible in the trace. (As built; see the final-review fix wave.)

`run_turn` behaviour:

1. `live = live_block(..., glow, last, ghost=ghost)`
2. `plan(...)`
3. If the plan needs a location: if `box` is set, `crop_jpeg` then `ground(cropped_frame, ..., box=box, container=container)`; else `ground(frame, ..., box=(0,0,1,1))`.
4. If `plan.skill == "grab"` and `glow == "yes"` and `plan.target`: `verify(config, cropped-or-full, plan.target)`. On `not match`, take the **existing** `Ground` and `dataclasses.replace` it to `pixel=None` with `error=f"verify mismatch: {check.error}"` when the verify call itself failed, else `error="verify mismatch"`. Pass that into `assemble` so grab is refused (existing `grab` + not visible path). Do not invent a pixel, and **do not fabricate a fresh `Ground`**: the grounder's own quadrant and raw reply must survive, or a dead model server becomes indistinguishable from a wrong cartridge in the trace. Record the `Verify` on `TurnRecord.verify` for **both** match and mismatch, and emit it as `verify.to_dict()`.
5. `assemble(..., glow=glow, ghost=ghost)`
6. `self_state` includes `ghost` and `box`.

brainview: `Session.step` passes `box=TRAY_BOX`. Overlay: if `ground.box` and `ground.quadrant`, draw the 2×2 cell **inside that box** in viewBox `0 0 100 100` (box is already normalised). Show `ghost` on the self panel. Keep `/api/step` off the intent socket.

`.gitignore` add:

```
*.mp4
*.mp3
```

Do not add the walkthrough files.

- [ ] **Step 1: Write the failing tests**

`tests/test_frames.py`:

```python
class CropJpeg(unittest.TestCase):
    def test_crop_jpeg_passes_box_fractions_to_ffmpeg(self) -> None:
        # mock subprocess.run; assert "crop=iw*0.278:" appears (0.548-0.270)
        # return a fake CompletedProcess(stdout=b"\xff\xd8\xff\xd9")
        out = frames.crop_jpeg(b"\xff\xd8\xff\xd9", (0.270, 0.325, 0.548, 0.566))
        self.assertTrue(out.startswith(b"\xff\xd8"))
```

`tests/test_brain.py`:

```python
class RunTurn(unittest.TestCase):
    def test_self_state_includes_ghost(self) -> None:
        # _chat returns a hold JSON; run_turn(..., ghost="yes")
        rec = brain.run_turn(BrainConfig(), brain.DEFAULT_SEED, fake_frame(),
                             heard="(none)", history=[], ghost="yes")
        self.assertEqual(rec.self_state["ghost"], "yes")
        self.assertIn("ghost=yes", rec.live)

    def test_verify_mismatch_blocks_grab_when_glow_is_yes(self) -> None:
        # first _chat (plan) returns grab/right/STORE CLERK
        # second _chat (ground) returns TOP-RIGHT
        # third _chat (verify) returns GOURMET CHEF
        rec = brain.run_turn(..., glow="yes", box=(0.27, 0.32, 0.55, 0.57))
        self.assertEqual(rec.decision.intent.skill, "hold")
```

For the verify test, stub `crop_jpeg` to return the input jpeg so ffmpeg is not required.

- [ ] **Step 2: Run** `python3 -m unittest tests.test_frames tests.test_brain.RunTurn -v` — fail.

- [ ] **Step 3: Implement `crop_jpeg`, extend `run_turn`, update brainview overlay and `.gitignore`.**

- [ ] **Step 4: Run** `python3 -m unittest tests.test_brain tests.test_frames -v` then `python3 -m unittest discover -s tests -t .` — all pass.

- [ ] **Step 5: Commit** `feat(brain): crop the container in run_turn and show it in brainview`

---

## Self-review

**Spec coverage (this slice):** §7.1–7.3 prompt (Task 2), §9 occupancy + cue gates (Tasks 1, 3), §10 crop-then-ask GROUND (Task 4), §11 wrist-crop VERIFY (Tasks 5–6), §14 no new skills (Task 1), §20 items 1, 3, 10, 11 (Tasks 2–4). Not this slice: P2 freeze-hold, P2c detectors, P2d envelope, B1 socket, live audio, VncSource.

**Step scan:** each test names the assertion; each code step names the signature and the spec string it must emit.

**Types:** `ghost: str`, `box: tuple[float, float, float, float] | None`, `Verify.match: bool` used the same way in Tasks 3–6.

**Review Focus:** each of the five lines has a named test in Tasks 2–4.

**Proportion:** six tasks on two modules plus a throwaway overlay. No HubFSM, no detector, no socket.

---

## Appendix: signatures verified in the tree (2026-11-01)

Re-checked against `clank/brain.py` at HEAD so the test assertions above are literal, not guessed.

```
Plan      skill, side, target, raw, error, note, prompt_ms, decode_ms     (note exists)
Ground    noun, row, col, pixel, raw_row, raw_col, error, methods="thirds" (Task 4 replaces row/col)
Decision  intent, reason, note          accepted == not reason
TurnRecord turn, frame_index, t_s, heard, seed, live, plan, ground, decision,
          self_state, outcome, elapsed_ms
BrainConfig, ModelReply                                 (test-only, never constructed over HTTP)

live_block(heard, left, right, yaw_rad, glow, last)      -> Task 2 appends ghost="unknown"
assemble(plan_result, ground_result, seq, t_mono_ns, look_step=1, left="empty", right="empty")
                                                          -> Task 3 appends glow/ghost
ground(config, frame, noun, transcript=None)              -> Task 4 rewrites
run_turn(config, seed, frame, heard, history, *, seq, left, right, yaw_rad,
         glow, last, wav, turn, look_step, elapsed_ms)     -> Task 6 appends ghost/box/container
```

The exact downgrade note already in the tree is `note="downgraded to scan"` (`clank/brain.py:556`).
The grab-not-visible reason is already `f"grab {target!r} but it is not visible"` (`:563`).
`tests/__init__.py` exists, so `python3 -m unittest tests.test_brain -v` resolves.
