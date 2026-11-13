# Session Brain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The live chooser hears JobBot's paused offer, remembers the run, stops inventing chores from a prop list, and writes enough diary that the next silent-miss is a one-line diagnosis.

**Architecture:** `ears.listen` stops splitting a 1.5 s pause; `confirm` records a fixed window and hands the transcript to the next turn. `SessionMemory` is the run's chat history (text only; current still is the only image). The planner prompt names no props and no unimplemented cues. `ready()` resets once per run, then only on failure.

**Tech Stack:** stdlib `clank/` (brain, ears, brainlog), `spike/wbc_game/auto.py` as the tool runner, unittest, llama-server on `:11434` for the live replay only.

**Spec:** `docs/superpowers/specs/2026-11-13-session-brain.md`

## Global Constraints

- `clank/` stays standard-library only; numpy/PIL/torch stay in `spike/`.
- Tests never open `11434 / 4242 / 5901 / 8765 / 8770`; patch `clank.brain._chat` via `ChatPatchMixin`.
- The planner prompt never lists object positions, a skill enum, a grabable-prop menu, glow, or ghost.
- `locate._MATCH_SYNONYMS` is not copied into the planner prompt.
- Conventional commits, `type(scope): lowercase`, message via `git commit -F /tmp/msg.txt`, no embedded double quotes; commit only files this plan names (do not stage `config.py`, `orchestrator.py`, `pixel3d.py`, `reach_vr.py`, `config.example.toml`).
- After source mutations: `python3 -m unittest discover -s tests -t .` with `XDG_RUNTIME_DIR` set; clear `__pycache__` if a mutation-test wrote one.
- Docs that describe this behaviour change in the same commit as the code (`docs/README.md` map, archived spec banner).

## Review Focus

- A listen that **opens on JobBot already talking** must still return that utterance (the confirm window of run `190131-679de2`); covered in Task 1 `test_speech_that_starts_the_capture_is_still_heard`.
- A greeting with no verb and no offer must **not** call the model, even with a cartridge visible in the still; covered in Task 3 `test_a_greeting_does_not_call_the_model`.
- A prior assistant reply `place(console)` must not override a new utterance `take my card`; covered in Task 4 `test_the_newest_heard_line_wins_over_an_old_goal`.
- Heard utterances survive the 80-pair cap; covered in Task 4 `test_heard_lines_survive_the_cap`.
- A successful grab must not call `ready()` (hands would open); covered in Task 5 `test_a_successful_grab_does_not_reset_the_body`.
- Parallel box reads must give the same median as serial ones on the same five replies; covered in Task 3b `test_the_median_does_not_depend_on_read_order`.

---

### Task 1: Hear the pause, and confirm without the gate

**Files:**
- Modify: `clank/ears.py` (`Hearing`, `listen`, default `hang_s`)
- Modify: `spike/wbc_game/auto.py` (`confirm`)
- Test: `tests/test_ears.py`
- Test: `tests/test_auto_loop.py` (only if `confirm` is grepped; otherwise leave)

**Interfaces:**
- Consumes: existing `ears.listen(..., hang_s=..., floor_s=..., reader=)` and `ears.record`
- Produces: `Hearing.ended_by: str` (`hang` | `deadline` | `eof` | `never-started`); `listen` default `hang_s=2.0`; floor = 20th percentile of **all** pre-start RMS blocks (do not `del floor_samples`); `confirm(pattern, max_s=8.0)` uses `ears.record` then `brain.transcribe`, not `listen`

- [ ] **Step 1: Write the failing tests in `tests/test_ears.py`**

```python
def test_a_1s_pause_is_the_same_utterance(self):
    # JobBot: "...museum." + ~1.2 s + "Here's my card."
    gap = int(1.2 / ears.BLOCK_S)
    stream = FakeStream(
        [silence(ears.BLOCK_S)] * 12
        + [pcm(8000, ears.BLOCK_S)] * 8
        + [silence(ears.BLOCK_S)] * gap
        + [pcm(8000, ears.BLOCK_S)] * 8
        + [silence(ears.BLOCK_S)] * 80
    )
    hearing = ears.listen(reader=stream, max_s=10.0)
    self.assertTrue(hearing.ok, hearing.error)
    self.assertGreater(hearing.speech_s, 0.6)
    self.assertEqual(hearing.ended_by, "hang")

def test_a_2_5s_pause_is_two_utterances(self):
    gap = int(2.5 / ears.BLOCK_S)
    stream = FakeStream(
        [silence(ears.BLOCK_S)] * 12
        + [pcm(8000, ears.BLOCK_S)] * 8
        + [silence(ears.BLOCK_S)] * gap
        + [pcm(8000, ears.BLOCK_S)] * 8
    )
    hearing = ears.listen(reader=stream, max_s=10.0)
    self.assertTrue(hearing.ok)
    self.assertLess(hearing.speech_s, 0.6)

def test_speech_that_starts_the_capture_is_still_heard(self):
    # confirm() opened on JobBot; 0.5 s median floor was the speech itself.
    stream = FakeStream(
        [pcm(8000, ears.BLOCK_S)] * 20
        + [silence(ears.BLOCK_S)] * 10
        + [pcm(8000, ears.BLOCK_S)] * 12
        + [silence(ears.BLOCK_S)] * 80
    )
    hearing = ears.listen(reader=stream, max_s=10.0)
    self.assertTrue(hearing.ok, hearing.error)
```

Keep `test_continuous_loud_audio_is_not_an_utterance` passing (constant 9000 RMS, no contrast → `never-started`).
`test_a_loud_room_raises_the_gate` may need its numbers retuned if the 20th percentile of a 400-RMS room is no longer 400; keep the claim (a 1200-RMS blip under 4× room tone is not speech).

- [ ] **Step 2: Run the new tests, expect FAIL**

Run: `python3 -m unittest tests.test_ears.ListenTest.test_a_1s_pause_is_the_same_utterance tests.test_ears.ListenTest.test_speech_that_starts_the_capture_is_still_heard`
Expected: FAIL (`hang_s` still 0.8, and/or `ended_by` missing)

- [ ] **Step 3: Implement in `clank/ears.py`**

- `hang_s: float = 2.0` on `listen`
- `Hearing.ended_by: str = ""`
- Floor: keep `floor_samples` for the whole pre-start period; `floor = ordered[max(0, len(ordered)//5)]`; `gate = max(gate_ratio * floor, gate_min)`
- Set `ended_by` on every return path
- `confirm` in `auto.py`: `hearing = ears.record(pattern=pattern, seconds=max_s)` (8 s default), then transcribe; empty transcript returns `""` as today

- [ ] **Step 4: Run `python3 -m unittest tests.test_ears tests.test_transcribe`**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add clank/ears.py spike/wbc_game/auto.py tests/test_ears.py
git commit -F /tmp/msg.txt
```

Message: `fix(ears): keep a 1s pause in one utterance and confirm without the speech gate`

---

### Task 2: Write every model call to `llm.jsonl`

**Files:**
- Modify: `clank/brain.py` (`_chat`)
- Modify: `clank/run.py` (`brain_env` — add `CLANK_LLM_LOG`)
- Test: `tests/test_brain.py` (new class `LlmTrace`)

**Interfaces:**
- Consumes: `_chat(config, messages, max_tokens=None) -> ModelReply`
- Produces: if `CLANK_LLM_LOG` is set, one JSON line per call at that path with keys `t`, `prompt_text`, `image_sha256` (sha256 of the JPEG bytes if an `image_url` part is present, else `""`), `raw_reply`, `prompt_ms`, `decode_ms`, `error`; never write `data:image` bytes; a missing path or `OSError` must not raise

- [ ] **Step 1: Write `LlmTrace` in `tests/test_brain.py`**

```python
class LlmTrace(ChatPatchMixin, unittest.TestCase):
    def test_a_chat_writes_the_prompt_and_not_the_image(self) -> None:
        # Call the real _chat with a patched urlopen that returns a canned body,
        # with CLANK_LLM_LOG pointing at a temp file.
        ...
        row = json.loads(path.read_text())
        self.assertIn("take my card", row["prompt_text"])
        self.assertNotIn("base64", row["prompt_text"])
        self.assertEqual(len(row["image_sha256"]), 64)
        self.assertEqual(row["raw_reply"], '{"goal":"take the card"}')
```

Also: `test_a_broken_log_path_does_not_fail_the_call` (parent dir unwritable → `ModelReply` still has the text).

- [ ] **Step 2: Run it, expect FAIL** (`CLANK_LLM_LOG` unknown)

- [ ] **Step 3: Implement the writer inside `_chat` after the reply is parsed.** `run.py` `brain_env`: `env["CLANK_LLM_LOG"] = str(run_dir / "llm.jsonl")`. Tests that replace `_chat` (the mixin) do not write, which is correct.

- [ ] **Step 4: `python3 -m unittest tests.test_brain.LlmTrace tests.test_run`** — PASS

- [ ] **Step 5: Commit** `feat(brain): record every model call next to the diary`

---

### Task 3: Prompt without a menu, and no think on a greeting

**Files:**
- Modify: `clank/brain.py` (`NEXT_STEP_PROMPT`, `CONSTITUTION`, `DEFAULT_SEED`, `live_block`, `think`, delete live use of `PROP_NOUNS`)
- Modify: `tests/test_brain.py` (`PromptText`, `ThinkTests`, glow/ghost live-block tests)
- Modify: `tests/test_prop_vocabulary.py` (stop requiring the prompt to name props; keep the console-is-not-locatable tests)

**Interfaces:**
- Consumes: `task_intent(heard) -> (skill, noun)`
- Produces: `NEXT_STEP_PROMPT` as spec §3.2 (exact copy). `live_block(heard, left, right, yaw_rad, last)` — no `glow`/`ghost` parameters. `think(...)` does not take `seed` or `glow`/`ghost`; if `task_intent(heard) == ("", "")` it returns `Thought` with `plan.skill="hold"`, `plan.note="nothing asked yet"`, **zero** `_chat` calls. `CONSTITUTION` loses the glow/ghost paragraph. `DEFAULT_SEED` is identity only: `The game is Job Simulator. The voice is a job instructor.`

- [ ] **Step 1: Write / rewrite tests**

`ThinkTests.test_a_greeting_does_not_call_the_model` — heard `"Fellow human, welcome to the job simulator."` → `chat_calls == []`, `plan.note` contains `nothing asked yet`.
`ThinkTests.test_heres_my_card_does_call_the_model` — heard `"Here's my card."` → one chat call, and `user_text` contains `This is what was heard`.
`PromptText.test_the_planner_prompt_names_no_props` — `NEXT_STEP_PROMPT` does not contain `cartridge`, `lever`, `knob`, `console`, `glow`, `ghost`, `JobBot said`.
`PromptText.test_live_block_has_no_cues` — replaces `test_live_block_carries_glow_and_ghost`.
Keep assemble's `glow=="no"` tests: those are the replay path, not the live prompt. Drop the ghost parameter from `live_block` only; `assemble(..., ghost=)` may stay so `run_turn` tests compile, or go in this same task if the signature is already open — prefer **delete `ghost` from `live_block` and from the `self:` line**, leave `assemble`'s ghost gate (dead, but replay tests pin it) unless deleting it is fewer lines.

`test_prop_vocabulary.py`: delete `test_the_seed_names_every_prop_and_says_what_the_console_is` and `test_the_step_prompt_names_them_too` and `test_every_recognised_noun_is_offered_to_the_plan`. Keep `TheNounThatFailedIsStillNotLocatable`. Keep `test_every_prop_the_prompt_names_is_one_the_locator_recognises` only if `PROP_NOUNS` remains as a locator concern; if `PROP_NOUNS` has no callers after this task, delete the constant and that test.

- [ ] **Step 2: Run `python3 -m unittest tests.test_brain.ThinkTests tests.test_brain.PromptText tests.test_prop_vocabulary` — expect FAIL**

- [ ] **Step 3: Edit `NEXT_STEP_PROMPT`, `think()`, `live_block()`, `CONSTITUTION`, `DEFAULT_SEED` to the spec. `said = f'This is what was heard: "{heard.strip()}"'` (or `(none)`). Do not send `live`'s duplicate `heard:` line.**

- [ ] **Step 4: `python3 -m unittest tests.test_brain tests.test_prop_vocabulary tests.test_next_step tests.test_auto_loop` — PASS**

- [ ] **Step 5: Commit** `fix(brain): stop listing props and stop planning a greeting`

---

### Task 3b: Read the five boxes at once (independent, may land any time)

**Why:** the server already serves four requests concurrently — `/props` reports `total_slots: 4` and `~/AI/logs/omni_*.log` shows slots 0–3 cycling — and `read_box` still walks its five reads in a `for` loop. Measured in the 2026-11-13 replay: 7 locate calls in 2351 ms, so ~0.34 s each, so five reads cost ~1.7 s of the ~2.4 s. Two waves of four-plus-one slots should cost ~0.8 s. This buys latency only: the same tokens are computed, and only while the slots are otherwise free.

**Not a win:** the quadrant → box → describe pipeline is sequential, and so is the planner chain; `n` is useless here because each read needs a *different* crop image.

**Files:**
- Modify: `spike/wbc_game/locate.py` (`read_box`)
- Test: `tests/test_locate_grounding.py`

**Interfaces:**
- Consumes: `ask_box(config, frame, noun, box, width, pad) -> (box|None, raw)`
- Produces: `read_box(..., parallel: int = 4)` returning the **same** tuple shape; per-read durations still appended to `times` (now wall time per task, which is what a caller deciding affordability wants); median still per edge; refusal still a strict majority

- [ ] **Step 1: Failing test** `test_the_median_does_not_depend_on_read_order` — five canned replies, assert the median box and refusal flag are identical to the serial path, and that `times` has five entries.

- [ ] **Step 2: Run it, expect FAIL** (no `parallel` parameter)

- [ ] **Step 3: Implement with `concurrent.futures.ThreadPoolExecutor(max_workers=parallel)`**; `_chat` is blocking urllib, so threads are the right tool and `clank/` stays stdlib. Preserve reply order (`executor.map`), keep the majority-refusal rule.

- [ ] **Step 4: Measure on the saved frame** `~/.local/state/Clank/sessions/20261113T000018Z-30a1b5/runs/190131-679de2/frames/frame-190251.png`, noun `card`: record `timings.total_ms` serial vs parallel, three runs each, and the pixel. Keep the change only if parallel is faster and the pixel is unchanged; write both numbers in `research/brain-tools.md`.

- [ ] **Step 5: `python3 -m unittest tests.test_locate_grounding` — PASS, then commit** `perf(locate): read the five boxes on the server's four slots`

---

### Task 4: SessionMemory — the run is one conversation

**Files:**
- Modify: `clank/brain.py` (`SessionMemory`, `think`, `_chat` callers)
- Test: `tests/test_brain.py` (new class `SessionMemoryTests`)

**Interfaces:**
- Consumes: Task 3's `think()` and `_chat(messages=...)`
- Produces:

```python
@dataclass
class SessionMemory:
    messages: list[dict]          # OpenAI chat messages, text only
    heard: list[str]

    def note_heard(self, text: str) -> None: ...
    def note_result(self, text: str) -> None: ...
    def remember_turn(self, user_text: str, assistant_text: str) -> None: ...
    def for_chat(self, frame: Frame, prompt_text: str) -> list[dict]: ...
    def cap(self, max_pairs: int = 80) -> None: ...
```

`note_heard` appends a user message `This is what was heard: "..."` and appends to `self.heard`.
`note_result` appends a user message `result: ...`.
`remember_turn` appends the planner user text (no image) and the assistant raw reply.
`for_chat` returns `self.messages + [{role:user, content:[_image_part(frame), {type:text, text:prompt_text}]}]`.
`cap` drops oldest `result:` user messages first, then oldest planner user+assistant pairs; **never** drops a `This is what was heard:` line.
`think(..., session: SessionMemory | None = None)`: if `session` is given, `_chat` gets `session.for_chat(...)` and `remember_turn` runs after the reply. If `None`, today's one-shot (tests that do not care).

- [ ] **Step 1: Write `SessionMemoryTests`**

```python
def test_the_second_think_sends_the_first_reply(self) -> None: ...
def test_prior_messages_have_no_image(self) -> None: ...
def test_the_newest_heard_line_wins_over_an_old_goal(self) -> None:
    # session already has assistant {"goal": "place the job cartridge into the console"}
    # new heard is "take my card"; the LAST user text contains take my card and
    # "Obey the newest thing that was heard"
def test_heard_lines_survive_the_cap(self) -> None:
    session = SessionMemory()
    session.note_heard("here's my card")
    for i in range(100):
        session.remember_turn(f"user {i}", '{"goal":"hold"}')
        session.note_result(f"hold {i}")
    session.cap(80)
    blob = json.dumps(session.messages)
    self.assertIn("here's my card", blob)
```

- [ ] **Step 2: Run, expect FAIL** (`SessionMemory` missing)

- [ ] **Step 3: Implement `SessionMemory` and wire it in `think()`. Keep locate on `_chat` with a one-element messages list (do not pass `session` into locate).**

- [ ] **Step 4: `python3 -m unittest tests.test_brain` — PASS**

- [ ] **Step 5: Commit** `feat(brain): carry the run into every planner call`

---

### Task 5: The loop owns one session, resets once, and tells the truth

**Files:**
- Modify: `spike/wbc_game/auto.py` (`main`, `one_turn`, `ready`, diary lines)
- Test: `tests/test_auto_loop.py`

**Interfaces:**
- Consumes: `SessionMemory` (Task 4), `confirm` returning a transcript (Task 1), `ready()`
- Produces: `main()` constructs one `SessionMemory` and one `pending_heard: str` and passes both into `one_turn`. `one_turn` calls `session.note_heard(spoken)` before `think`, `session.note_result(...)` after locate/grab/refuse/confirm. `think(..., session=session)`. `ready()` is called from `main()` once; `one_turn` calls it only when `_body_needs_reset(last_code)` is true. `_body_needs_reset` is true when `last_code not in (None, 0)` or the control socket is missing. Diary: refusals → `brainlog.log("result", f"refused {skill}: {reason}", ok=False)`; `ready` → `brainlog.log("act", "reset body", seconds=...)`. Confirm transcript, if any, is assigned to `pending_heard` for the next `one_turn` (used instead of a fresh `listen` when non-empty). Listen rows always include `floor`, `gate`, `peak`, `ended_by` even on quiet (quiet + numbers is allowed).

- [ ] **Step 1: Extend `tests/test_auto_loop.py` as grep tests**

```python
def test_one_session_memory_lives_in_main(self) -> None:
    self.assertIn("SessionMemory()", self.source[self.source.index("def main("):])
    self.assertNotIn("WorkingMemory()", self.source[self.source.index("def main("):])
    # WorkingMemory stays inside one_turn

def test_think_receives_the_session(self) -> None:
    turn = self.source[self.source.index("def one_turn("):]
    self.assertIn("session=session", turn)

def test_confirm_feeds_the_next_turn(self) -> None:
    self.assertIn("pending_heard", self.source)

def test_a_successful_grab_does_not_reset_the_body(self) -> None:
    turn = self.source[self.source.index("def one_turn("):]
    ready_at = turn.find("ready(")
    grab_at = turn.index("grab_pixel(")
    self.assertTrue(ready_at == -1 or ready_at < grab_at)
    # the success path after grab_pixel must not call ready()
    after = turn[grab_at:]
    self.assertNotIn("ready(", after)

def test_refusals_are_results(self) -> None:
    self.assertIn('log("result"', self.source)
    self.assertNotIn('hold ({thought.decision.reason})', self.source)
```

Also assert `one_turn` still does not import `task_intent` (the greeting gate lives in `think`).

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement the loop changes. `one_turn(..., session, pending_heard="") -> tuple[str, str]` returning `(spoken, new_pending)`. `main` holds both. Do not reset inside the success path.**

- [ ] **Step 4: `python3 -m unittest tests.test_auto_loop tests.test_brain` — PASS**

- [ ] **Step 5: Commit** `fix(auto): one session, one reset, confirm is the next line`

---

### Task 6: Wire `place`

**Files:**
- Modify: `spike/wbc_game/auto.py` (the `skill not in ("reach", "grab")` branch)
- Test: `tests/test_auto_loop.py`

**Interfaces:**
- Consumes: existing `spike/wbc_game/place.py` approach (`APPROACH_M`, locate destination, reach, open). Prefer calling into functions already in `auto.py` (`sense`, `grab_pixel(..., grab=False)`, `reach_vr.send(verb=f"open_{side}")`) rather than importing `place.py` if that import pulls a second argparse main.
- Produces: `skill == "place"` locates `thought.plan.target`, reaches the approach point, opens. Empty hands → `result: place refused: hands empty`, no motion. Do not call `brain.sequence`.

- [ ] **Step 1: Grep test** `test_place_is_wired` — `one_turn` source contains `skill == "place"` and does not contain `is not wired yet` for place. `test_place_does_not_call_sequence` already covered by the existing `sequence(` ban.

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement the branch. Reuse `sense(target, ...)` for the destination pixel; convert to a point with the existing depth helper in `sense`/`grab_pixel`; if `place.py` has a function you can import without running `main`, use it — otherwise copy the three-line approach (`approach_point` then `grab_pixel(..., grab=False)` then `open_{side}`) into `auto.py` rather than a new module.**

- [ ] **Step 4: `python3 -m unittest tests.test_auto_loop tests.test_workspace` — PASS**

- [ ] **Step 5: Commit** `feat(auto): place is a reach to the destination then open`

---

### Task 7: Docs match the code

**Files:**
- Modify: `README.md` — it quotes the live prompt verbatim today (the `seed:` line with the prop list, `glow=unknown ghost=unknown`, and the "Grab when glow is yes…" paragraph), and it claims the seed names the props and that the model may only use locatable nouns. All of that becomes false in Task 3. Replace the quoted prompt with the §3.2 text, state that a greeting does not call the model, that the run is one conversation (`SessionMemory`), and that every model call is written to `llm.jsonl` beside the diary. Refresh the test count in the "tests" line rather than leaving a stale number.
- Modify: `docs/superpowers/specs/2026-11-13-session-brain.md` only if a signature drifted during tasks
- Create (move): `docs/archive/specs/2026-11-10-brain-tools.md` from `docs/superpowers/specs/2026-11-10-brain-tools.md`, with banner: archived 2026-11-13, replaced by `docs/superpowers/specs/2026-11-13-session-brain.md` for the live chooser; parent `2026-10-31-brain.md` stays current
- Modify: `docs/README.md` current/archived tables
- Modify: `docs/superpowers/specs/2026-10-31-brain.md` §6 MEMORY / glow-ghost: one short note that glow/ghost are unwired and the live transcript is `SessionMemory`, pointing at the new spec
- Modify: `research/brain-tools.md` — add a dated section for run `190131-679de2` (the missed "Here's my card", the confirm gate, the prompt-invented place)
- Modify: `AGENTS.md` only the brain/hearing bullets that this slice makes false (hang_s, diary is write-only, ready every turn)

- [ ] **Step 1: Move and banner the 2026-11-10 spec; update the three maps; add the research section with the numbers from the spec (floor 2445 / gate 9781 / peak 4415; hang split; replay JSON).**
- [ ] **Step 1b: Rewrite the README prompt block and the two claims it makes about the seed and the noun list; keep the prose in the README's own voice (first person, no marketing).**

- [ ] **Step 2: Grep for stale promises**

Run: `rg -n "standing up before I look|JobBot said:|glow=unknown|hang_s = 0.8|history=\\[\\]" --glob '!docs/archive/**' --glob '!research/**'`
Expected: no hits in `clank/` or `spike/wbc_game/auto.py`. Hits in `research/` are dated measurements and stay.

- [ ] **Step 3: `python3 -m unittest discover -s tests -t .` — PASS**

- [ ] **Step 4: Commit** `docs(brain): the live chooser is a session, and the 11-10 spec is archived`

---

## Self-review

1. Spec coverage: §3.1 hearing → Task 1; §3.4 `llm.jsonl` → Task 2; §3.2 prompt / greeting gate / no props → Task 3; parallel box reads → Task 3b (perf, independent); §3.3 SessionMemory → Task 4; §3.4 diary honesty, §3.5 reset, confirm-as-next-line → Task 5; §3.6 place → Task 6; maps → Task 7.
2. Signatures named: `Hearing.ended_by`, `SessionMemory.{note_heard,note_result,remember_turn,for_chat,cap}`, `think(..., session=)`, `confirm` via `record`, `one_turn(..., session, pending_heard)`.
3. `ghost` left on `assemble` on purpose so Task 3 does not rewrite `run_turn` tests; live prompt does not mention it.
4. Live replay of the four card stills (temperature 0, no prop list) is **not** a unittest; do it once after Task 3 against `:11434` if the operator is watching, and record the four goals in `research/brain-tools.md`. Failure there (model names "console" as the object of "take my card") is a stop, not a skip.
