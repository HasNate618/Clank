# Freeze-Hold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make adapter `hold` latch last-achieved head and both wrists, and make `rest` the only path that returns the standing-clip VR points.

**Architecture:** One pure function in `clank/body.py` decides the pose for each of head / left / right. Empty authority is `rest` (clip). Any other command fills unmentioned pose channels from the last `BodyState`. `hold()` sets `FREEZE_HOLD_AUTH` and does not touch grip. The executor overlays those poses; it does not invent a second policy. C3m's standing clip + anchor-index freeze stays — that is the legs, not the wrists.

**Tech Stack:** Python 3.14 stdlib in `clank/` (`unittest`). `python3 -m unittest` (pytest is not installed). Numpy/MuJoCo stay in `spike/wbc/sonic_executor.py`; this plan does not add ONNX tests.

**Spec:** `docs/superpowers/specs/2026-10-31-brain.md` §6.1, §14 (`rest`), §19 P2, §20 item 5. Parent: `docs/2026-10-23-clank-design.md` §8 `hold` row. Adapter: `docs/superpowers/specs/2026-10-30-targeting-adapter.md` §6 hold row (currently wrong: "empty authority (executor standing clip)").

## Global Constraints

- `clank/` stays **standard-library only**. No NumPy, no ONNX, no MuJoCo in `clank/` or `tests/`.
- Tests **must not** open `:11434`, `:4242`, `:5901`, `:8765`, or `:8770`. Tests **must not** construct `SonicExecutor`.
- Do **not** launch Steam, Proton, Monado, llama-server, or the game.
- Do **not** edit `run.sh`, `config.toml`, `config.example.toml`, `flake.nix`, `clank/processes.py`, `clank/portowner.py`, `clank/orchestrator.py`, `clank/config.py`, or `spike/c0/launch_monado.sh`.
- Do **not** add `rest` to `clank.brain.PLANNER_SKILLS` or the constitution. Planner v1 does not emit it.
- Do **not** put `AUTH_CLOSURE` on freeze-hold. Grip already latches; freeze-hold must not reopen a hand.
- Do **not** put `AUTH_ROOT_VEL` or `AUTH_WAIST` on freeze-hold. Those bits are unsupported in a hold session (`FLAG_UNSUPPORTED`); brain spec §6.1 is head + both hands.
- Do **not** add `insert` / `pull` / `take_cartridge`.
- Branch: `feat/freeze-hold` from `main`. Independent of `feat/replay-brain-writer`.
- Commit only paths this plan names.

## Out of scope (later plans)

- P2b (`FLAG_SATURATED` unread by PLAN).
- P2c Job Simulator blue/yellow detectors.
- P2d live card envelope.
- A 20 s live freeze-diff in Job Simulator (genre assumption, not this rewrite).
- A C3m re-soak of RMS velocity. Clip-hold RMS is already measured; this plan changes the *VR overlay*, not the standing-clip freeze.

## Review Focus

1. **Look after a reach keeps the hands.** `channel_pose` on an `AUTH_HEAD`-only command with a last state returns the last left and right wrists, not clip. Task 1 pins this. Without it, looking at JobBot drops the card.
2. **TTL expiry is freeze-hold, not clip.** `CommandBus.take` after TTL returns `FREEZE_HOLD_AUTH` and, when a state has been published, those three poses. Task 2 pins this. This is the production path (`body_view` loop).
3. **`rest` still snaps to clip.** Empty authority plus a last state still yields `None` (clip) on every channel. Task 1 pins this. `hold_pose.py` clean-exit and the socket "clear" test use `rest()`, not `hold()`.
4. **Grip is not in the freeze-hold mask.** `hold().authority & AUTH_CLOSURE == 0`. Task 1 pins this.
5. **Planner cannot emit `rest`.** `rest` is in `intent.SKILLS` and not in `brain.PLANNER_SKILLS`. Task 3 pins this.

---

### Task 1: `hold` / `rest` / `channel_pose`

**Files:**
- Modify: `clank/body.py` (`FREEZE_HOLD_AUTH`, `hold`, new `rest`, new `channel_pose`)
- Test: `tests/test_body.py`

**Interfaces:**
- Consumes: existing `BodyCommand`, `BodyState`, `Pose`, `AUTH_HEAD`, `AUTH_L_HAND`, `AUTH_R_HAND`, `AUTH_CLOSURE`.
- Produces: `FREEZE_HOLD_AUTH: int = AUTH_HEAD | AUTH_L_HAND | AUTH_R_HAND`; `hold(seq: int = 0, now_ns: int = 0, ttl_ms: int = DEFAULT_TTL_MS, state: BodyState | None = None) -> BodyCommand`; `rest(seq: int = 0, now_ns: int = 0, ttl_ms: int = DEFAULT_TTL_MS) -> BodyCommand`; `channel_pose(cmd: BodyCommand, last: BodyState | None, name: str, bit: int) -> Pose | None` (`None` means standing clip for that channel).

- [ ] **Step 1: Write the failing tests** in `tests/test_body.py`

Reuse `sample_state()`. Reached wrists that are not the clip rest:

```python
REACHED_RIGHT = Pose((0.18, -0.20, 0.40), (0.0, 0.0, 0.0, 1.0))
REACHED_LEFT = Pose((0.18, 0.20, 0.40), (0.0, 0.0, 0.0, 1.0))
LOOKED_HEAD = Pose((0.0, 0.0, 0.40), (0.0, 0.0, 0.1, 0.995))
```

Build `reached = BodyState(**{**sample_state().__dict__, "head": LOOKED_HEAD, "left_hand": REACHED_LEFT, "right_hand": REACHED_RIGHT})`.

```python
class FreezeHold(unittest.TestCase):
    def test_hold_requests_head_and_both_hands(self) -> None:
        cmd = hold(seq=5, now_ns=10)
        self.assertEqual(cmd.authority, FREEZE_HOLD_AUTH)
        self.assertEqual(cmd.authority & AUTH_CLOSURE, 0)
        self.assertIsNone(cmd.head)
        self.assertFalse(cmd.is_ballistic())

    def test_hold_copies_poses_from_state(self) -> None:
        cmd = hold(state=reached)
        self.assertEqual(cmd.head, LOOKED_HEAD)
        self.assertEqual(cmd.left_hand, REACHED_LEFT)
        self.assertEqual(cmd.right_hand, REACHED_RIGHT)
        self.assertEqual(cmd.authority, FREEZE_HOLD_AUTH)

    def test_rest_requests_no_authority(self) -> None:
        cmd = rest(seq=5, now_ns=10)
        self.assertEqual(cmd.authority, 0)
        self.assertIsNone(cmd.head)

    def test_rest_ignores_last_state(self) -> None:
        cmd = rest()
        self.assertIsNone(channel_pose(cmd, reached, "right_hand", AUTH_R_HAND))
        self.assertIsNone(channel_pose(cmd, reached, "head", AUTH_HEAD))

    def test_look_keeps_the_hands(self) -> None:
        cmd = BodyCommand(authority=AUTH_HEAD, head=LOOKED_HEAD)
        self.assertEqual(channel_pose(cmd, reached, "left_hand", AUTH_L_HAND), REACHED_LEFT)
        self.assertEqual(channel_pose(cmd, reached, "right_hand", AUTH_R_HAND), REACHED_RIGHT)
        self.assertEqual(channel_pose(cmd, reached, "head", AUTH_HEAD), LOOKED_HEAD)

    def test_reach_keeps_the_other_hand_and_head(self) -> None:
        cmd = BodyCommand(authority=AUTH_R_HAND, right_hand=REACHED_RIGHT)
        self.assertEqual(channel_pose(cmd, reached, "left_hand", AUTH_L_HAND), REACHED_LEFT)
        self.assertEqual(channel_pose(cmd, reached, "head", AUTH_HEAD), LOOKED_HEAD)
        self.assertEqual(channel_pose(cmd, reached, "right_hand", AUTH_R_HAND), REACHED_RIGHT)

    def test_no_last_state_is_clip_on_unset_channels(self) -> None:
        cmd = BodyCommand(authority=AUTH_HEAD, head=LOOKED_HEAD)
        self.assertIsNone(channel_pose(cmd, None, "right_hand", AUTH_R_HAND))
        self.assertEqual(channel_pose(cmd, None, "head", AUTH_HEAD), LOOKED_HEAD)

    def test_freeze_hold_bits_with_no_poses_use_last(self) -> None:
        cmd = hold()  # bits set, poses None
        self.assertEqual(channel_pose(cmd, reached, "right_hand", AUTH_R_HAND), REACHED_RIGHT)
```

Flip the existing `test_hold_requests_no_authority` to the freeze-hold assertion (or delete it; `test_hold_requests_head_and_both_hands` replaces it). `test_accepts_minimal_hold` in `tests/test_validate.py` stays: empty `BodyCommand()` is `rest` and must still validate.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_body.FreezeHold -v`

Expected: FAIL (`FREEZE_HOLD_AUTH` / `rest` / `channel_pose` not defined, and `hold()` still has `authority == 0`).

- [ ] **Step 3: Implement** in `clank/body.py`

```python
FREEZE_HOLD_AUTH = AUTH_HEAD | AUTH_L_HAND | AUTH_R_HAND
```

`hold(...)`: `authority=FREEZE_HOLD_AUTH`; copy `state.head` / `left_hand` / `right_hand` when `state` is not None; otherwise those fields stay `None`. Docstring: freeze-hold, not clip return. Grip is not in the mask.

`rest(...)`: the current `hold()` body (`authority=0`). Docstring: standing-clip VR points.

`channel_pose(cmd, last, name, bit) -> Pose | None`:

1. If `cmd.authority == 0`: return `None` (rest → clip), even when `last` is set.
2. If `(cmd.authority & bit)` and `getattr(cmd, name) is not None`: return that pose.
3. If `last is not None`: return `getattr(last, name)`.
4. Else: return `None` (clip).

`name` is one of `"head"`, `"left_hand"`, `"right_hand"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_body tests.test_validate -v`

Expected: PASS. Then the full suite will be red on bus tests that still assert `hold().authority == 0` — that is Task 2.

- [ ] **Step 5: Commit**

```bash
git add clank/body.py tests/test_body.py
git commit -m "$(cat <<'EOF'
feat(body): freeze-hold latches last poses; rest is clip return

EOF
)"
```

---

### Task 2: Bus TTL expiry fills freeze-hold

**Files:**
- Modify: `clank/bus.py` (`take`)
- Test: `tests/test_bus.py`

**Interfaces:**
- Consumes: Task 1 `hold(..., state=)`, `rest()`, `FREEZE_HOLD_AUTH`.
- Produces: `CommandBus.take` returns freeze-hold (not empty-authority) when idle or TTL-expired; copies poses from `publish_state` when present.

- [ ] **Step 1: Write the failing tests** in `tests/test_bus.py`

Import `FREEZE_HOLD_AUTH`, `rest`, and a small `sample_state` (copy from `tests/test_body.py` or import if you lift it to a helper — do not import the test module). Keep assertions on `seq` as they are.

```python
def test_expired_ttl_returns_hold(self) -> None:
    # existing setup
    out = bus.take(now_ns=10 * 1_000_000_000)
    self.assertEqual(out.authority, FREEZE_HOLD_AUTH)
    self.assertEqual(out.seq, 3)

def test_empty_bus_is_hold(self) -> None:
    self.assertEqual(bus.take(now_ns=0).authority, FREEZE_HOLD_AUTH)

def test_rejected_command_is_not_pushed(self) -> None:
    # last line today: take().authority == 0
    self.assertEqual(bus.take(now_ns=0).authority, FREEZE_HOLD_AUTH)

def test_expired_ttl_copies_published_poses(self) -> None:
    bus = CommandBus(clock=FakeClock())
    bus.push(BodyCommand(seq=3, t_mono_ns=0, ttl_ms=200, authority=AUTH_ROOT_VEL, v_xy=(0.5, 0.0)))
    bus.publish_state(sample_state())
    out = bus.take(now_ns=10 * 1_000_000_000)
    self.assertEqual(out.authority, FREEZE_HOLD_AUTH)
    self.assertEqual(out.head, sample_state().head)
    self.assertEqual(out.right_hand, sample_state().right_hand)

def test_hold_sent_over_socket_replaces_with_freeze_hold(self) -> None:
    # was test_hold_sent_over_socket_clears_authority
    # after sending hold(seq=2), take().authority == FREEZE_HOLD_AUTH
    # not 0, and not AUTH_ROOT_VEL

def test_rest_sent_over_socket_clears_authority(self) -> None:
    # same shape as the old clear test, but send rest(seq=2, ...)
    # take().authority == 0
```

`test_socket_roundtrip` waits on `bus.take().authority == 0` as "not yet received". After this task empty-bus hold has bits, so that wait is instantly false. Change the wait to `bus.take().authority != AUTH_ROOT_VEL` (or wait until `seq == 1`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_bus -v`

Expected: FAIL on the new pose-copy test and/or the socket wait; existing `authority == 0` assertions fail because Task 1 already flipped `hold()`.

- [ ] **Step 3: Implement** in `clank/bus.py`

In `take`, both `hold(now_ns=now)` sites become `hold(seq=..., now_ns=now, state=self._state)`. Idle (`cmd is None`) uses `seq=0`. Expired uses `seq=cmd.seq`. Do not call `rest()` here.

`test_valid_json_is_accepted` pushes `{"seq": 1, "authority": 0}` — that is a writer-sent rest and must stay accepted.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_bus tests.test_body tests.test_validate -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add clank/bus.py tests/test_bus.py
git commit -m "$(cat <<'EOF'
fix(bus): TTL expiry freeze-holds last achieved poses

EOF
)"
```

---

### Task 3: Intent `rest` and the adapter compile

**Files:**
- Modify: `clank/intent.py` (`SKILLS`, `verb_to_intent`)
- Modify: `spike/wbc/body_view.py` (`_command_from_intent`)
- Modify: `spike/wbc/hold_pose.py` (clean-exit)
- Test: `tests/test_intent.py`
- Test: `tests/test_brain.py` only if `SKILLS` vs `PLANNER_SKILLS` needs a pin — add one assertion, do not retune PLAN.

**Interfaces:**
- Consumes: Task 1 `hold`, `rest`.
- Produces: `"rest"` in `intent.SKILLS`, not in `SIDE_SKILLS`, not in `brain.PLANNER_SKILLS`. `verb_to_intent("rest", {}) -> Intent(skill="rest")`. `_command_from_intent(..., skill="hold")` → `hold(state=executor.last_state)`; `skill="rest"` → `rest()`.

- [ ] **Step 1: Write the failing tests** in `tests/test_intent.py`

```python
def test_rest_needs_no_side(self) -> None:
    self.assertEqual(intent.validate_intent(intent.Intent(skill="rest")), "")

def test_rest_rejects_a_side(self) -> None:
    self.assertTrue(intent.validate_intent(intent.Intent(skill="rest", side="right")))

def test_rest_verb_maps(self) -> None:
    i = intent.verb_to_intent("rest", {})
    assert i is not None
    self.assertEqual(i.skill, "rest")
```

In `tests/test_brain.py` (existing `test_planner_skills_are_a_subset` or adjacent):

```python
def test_planner_does_not_emit_rest(self) -> None:
    self.assertNotIn("rest", brain.PLANNER_SKILLS)
    self.assertIn("rest", SKILLS)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_intent tests.test_brain -v`

Expected: FAIL (`unknown skill 'rest'`).

- [ ] **Step 3: Implement**

`clank/intent.py`: append `"rest"` to `SKILLS` after `"hold"`. Do not add it to `SIDE_SKILLS`. In `verb_to_intent`, `if name == "rest": return make("rest")` next to `hold`.

`spike/wbc/body_view.py` `_command_from_intent`:

- `hold` → `hold(state=executor.last_state)` (not bare `hold()`).
- `rest` → `rest()`.

`spike/wbc/hold_pose.py`: the clean-exit send that "gives the body back its standing clip" calls `rest(...)`, not `hold(...)`. Update that comment.

`spike/wbc/reel.py`: leave the `"hold"` phases as `hold(seq=0)`. After this rewrite those phases freeze the last pose, which is the correct reel (look, then stay looked). Do not switch them to `rest`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_intent tests.test_brain tests.test_body tests.test_bus -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add clank/intent.py tests/test_intent.py tests/test_brain.py spike/wbc/body_view.py spike/wbc/hold_pose.py
git commit -m "$(cat <<'EOF'
feat(intent): rest is clip return; hold compiles to freeze-hold

EOF
)"
```

---

### Task 4: Executor overlays `channel_pose`

**Files:**
- Modify: `spike/wbc/sonic_executor.py` (`_targets`)
- Modify: `docs/superpowers/specs/2026-10-30-targeting-adapter.md` §6 hold row
- Modify: `docs/superpowers/specs/2026-10-31-brain.md` §19 P2 row only if you can mark it implemented; otherwise leave P2 as the gate until a live soak
- Test: none new. Policy is Task 1. Do not import `SonicExecutor` from `tests/`.

**Interfaces:**
- Consumes: Task 1 `channel_pose`, `AUTH_HEAD` / `AUTH_L_HAND` / `AUTH_R_HAND`.
- Produces: `_targets` returns clip (`None, None`) iff every channel is clip; otherwise overlays last-achieved / commanded poses onto the standing-clip arrays.

- [ ] **Step 1: There is no failing ONNX test.** Re-run Task 1 — those assertions *are* the overlay contract. If any FreezeHold test is skipped or weakened, stop and restore it.

Run: `python3 -m unittest tests.test_body.FreezeHold -v`

Expected: PASS (already green from Task 1).

- [ ] **Step 2: Implement `_targets`** in `spike/wbc/sonic_executor.py`

Keep the standing-clip fetch and the `frame="head"` yaw carry. Replace the "skip if bit unset" loop with:

```python
from clank.body import channel_pose  # add to the existing body import

for bit, index, name in (
    (AUTH_HEAD, HEAD, "head"),
    (AUTH_L_HAND, L_WRIST, "left_hand"),
    (AUTH_R_HAND, R_WRIST, "right_hand"),
):
    pose = channel_pose(cmd, self.last_state, name, bit)
    if pose is None:
        continue
    # existing overlay + yaw-carry for hands; `touched = True`
if not touched:
    return None, None
return pts, orns
```

Do not change `HOLD_CLIP` / `HOLD_FRAME` / `runner.reset(..., start=self.hold_frame)`. Do not set `AUTH_CLOSURE` here. Do not honour `AUTH_ROOT_VEL`.

Update the file docstring: empty-mask is `rest` (clip VR points); any other command holds unmentioned wrists/head at last achieved.

- [ ] **Step 3: Docs**

Adapter §6 table, `hold` row: BodyCommand is `FREEZE_HOLD_AUTH` with last achieved poses, not "empty authority (executor standing clip)". Add a `rest` row: empty authority, clip VR points, not a planner output.

Brain spec §14 already says add `rest` when this rewrite lands — that sentence is now true. Do not retarget §19 P2 to PASS; a unit overlay is not a live latch measurement.

- [ ] **Step 4: Run the full suite**

Run: `python3 -m unittest discover -s tests -t .`

Expected: OK, same count as before plus the new FreezeHold / rest / bus tests (expect ~10 more than the previous 594 on `feat/replay-brain-writer`; this branch is from `main`, so count against `main`'s suite).

- [ ] **Step 5: Commit**

```bash
git add spike/wbc/sonic_executor.py docs/superpowers/specs/2026-10-30-targeting-adapter.md docs/superpowers/specs/2026-10-31-brain.md
git commit -m "$(cat <<'EOF'
fix(executor): unmentioned pose channels hold last achieved

EOF
)"
```

---

## Self-review

**Spec coverage:** §6.1 freeze-hold vs rest (Tasks 1, 4). Grip must not reopen (Task 1, no `AUTH_CLOSURE`). §14 add `rest`, planner does not emit it (Task 3). §19 P2 "adapter hold latches last achieved; rest is clip return" (Tasks 1–4). §20 item 5 (Task 1 look-keeps-hands + Task 2 TTL). Parent §8 `AUTH_ROOT_VEL` on hold is **not** implemented; brain spec §6.1 wins because the executor cannot honour that bit in a hold session. P2b/P2c/P2d out of scope.

**Step scan:** `channel_pose` is the only algorithm written out; executor overlay is the existing loop with a different pose source.

**Type consistency:** `hold(..., state: BodyState | None)`, `rest(...)`, `channel_pose(...) -> Pose | None`, `FREEZE_HOLD_AUTH` used in body, bus, and tests.

**Review Focus:** all five have a named test.

**Proportion:** four tasks, one new function, two builders, a bus fill, a skill name, an executor loop change.
