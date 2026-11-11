# ARDY Reach Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the VR-cruise reach/grab with constraints, a generated reference, and SONIC mode-0 tracking, so an in-front grab plants and looks, and an off-axis grab turns or steps because the model chose to.

**Architecture:** Adapter process (`reach_ardy.py`) states what to achieve — right hand at the object, feet planted, or a root waypoint when the object is out of reach — and ARDY-G1 generates the motion. The 50 Hz body process only writes a pose ring and plays deposited clips. CUDA and torch stay out of SONIC. `hold` / `look` / `rest` stay on VR mode 1. `reach_vr.py` stays until the live gate passes.

**Tech Stack:** Python 3 stdlib in packing, mailbox, geometry, chain policy, and `tests/`. NumPy / MuJoCo / torch / ARDY in `spike/ardy/.venv`. ARDY `ARDY-G1-RP-25FPS-Horizon52`, text tower absent (zero text vector). `python3 -m unittest` (pytest is not installed).

**Re-scope 2026-11-11 — read this before executing any task.** `research/motion-generators.md` compares MotionBricks, Kimodo and ARDY, and §9 records ARDY measured on this machine: 0.94 GB peak VRAM, 0.07-0.20 s per 52-frame window, and 60.3 mm hand-constraint adherence. Decision: pursue ARDY. Consequences for this document, which was written MotionBricks-first:

| Task | Status |
| --- | --- |
| 3 mailbox, 4 resample, 7 executor playback | unchanged, generator-agnostic |
| 1 packing | unchanged; ARDY's MuJoCo export is the same 36-float layout |
| 2 envelope geometry | **demoted**: `classify` decides only whether a root goal is needed; `step_root` and `MAX_STEP_*` are no longer how a step is produced |
| 5 (was: mink still frames) | **replaced** by the ARDY adapter |
| 6 (was: MotionBricks generate) | **replaced** by constraint authoring plus the reachability escalation |
| 8 adapter | same loop minus the step chain |

The spec itself was rewritten the same day: it now argues ARDY and carries no superseded sections. MotionBricks appears only as the documented fallback (spec §14, plan §"Fallback"), and no task should build it without a new decision.

**Spec:** `docs/superpowers/specs/2026-11-11-ardy-reach-design.md`.

## Global Constraints

- `clank/` stays **standard-library only**. No generator, torch, ARDY, mink, or ONNX import in `clank/`.
- `tests/` stays stdlib: no NumPy, no MuJoCo, no ONNX, no constructing `SonicExecutor`.
- Unittest **must not** open `:4242`, `:5901`, `:8765`, `:8770`, `:8771`, or `:11434`.
- Do **not** edit `run.sh`, `config.toml`, `flake.nix`, `clank/orchestrator.py`, `clank/config.py`, `clank/body.py` (`FRAMES`, `BodyCommand`), or `reach_vr.py` except a later one-line pointer after §11.5.
- Do **not** put object XYZ or a clip on `BodyCommand`. Do **not** feed mink joints to SONIC as `upper_body` or `ctrl`.
- Do **not** enable TF32. Do **not** build a TensorRT plan while the game is up.
- **ARDY's torch needs `LD_LIBRARY_PATH=/run/opengl-driver/lib`.** NixOS keeps `libcuda.so.1` off the loader path, so torch reports `cuda False` with a working driver. Every ARDY run uses `spike/ardy/.venv/bin/python` with that variable set.
- **A scalar `cfg_weight` disables constraints.** With `cfg_type="separated"`, ARDY sets `w_text, w_cstr = cfg_weight, 0.0`, so a constrained run under the default ignores its constraints (measured: 870 mm instead of 60 mm). Always pass `(text, constraint)`.
- **`num_denoising_steps` must be <= `num_base_steps`** (10 for this checkpoint). Larger values trip a device-side assert, not a clean error.
- Do **not** inherit `STEP_GUARD_M` from `reach_vr.py` onto stepped reaches.
- Default executor mode stays **1**. Mode 0 is only while a generated clip is playing.
- Commit only paths this plan names. Conventional commits, lowercase, no embedded double quotes; `git commit -F /tmp/msg.txt`.

## Out of scope

- Brain, locate, glow, neck, HumanML3D, MotionBricks/SONIC retraining.
- Switching the live grab path off `reach_vr.py` before Task 8's gate.
- Vendoring Warp/Newton/Scene Preview. Sample MotionBricks BVH/CSV clips stay out of the session.

## Review Focus

1. **Hands in the 29.** Slicing `qpos[7:36]` copies the left Dex3 joints into the body vector. Task 1 packing test leaves `qpos[29:36]` and `qpos[43:50]` unchanged.
2. **`BodyState.q` as a pose.** That tuple is policy-order minus `DEFAULT`. Task 1 refuses to round-trip it; Task 3 context comes from `pose36.json`.
3. **Lookahead walks after the clip.** Native freeze at `n-1` still looks 0.9 s into empty/walking frames. Task 4 flatten-hold test.
4. **Far grab cranes instead of stepping.** Yaw > 35° or planted residual ≥ 5 cm must classify `step`. Task 2.
5. **Clasp after the clip, from a second Python.** That is the 0.09 m drift. Task 7 sends `AUTH_CLOSURE` while `mb_status` is `playing`.

---

### Task 1: 36-float packing

**Files:**
- Create: `spike/wbc/g1_pack.py`
- Test: `tests/test_mb_packing.py`

**Interfaces:**
- Consumes: `CSV2MJ` from spec §7.3 / `sonic_runner.py:76-78`; qpos layout of `g1_29dof_with_hand.xml`.
- Produces:

```python
JOINT_NAMES: tuple[str, ...]  # 29, hardware order, spec §6.5
CSV2MJ: tuple[int, ...]       # (0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28)
QPOS_QADR: tuple[int, ...]    # (7..28) + (36..42)  — 29 hinge qpos addresses; skips hands
HAND_QPOS = (slice(29, 36), slice(43, 50))

def wxyz_to_xyzw(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]
def xyzw_to_wxyz(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]
def pose36_from_qpos(qpos: Sequence[float]) -> tuple[float, ...]  # len 36: xyz + xyzw + 29 hinges
def apply_pose36(qpos: list[float], pose: Sequence[float]) -> list[float]  # copy; hands preserved
def jp_from_pose36(pose: Sequence[float]) -> tuple[float, ...]   # len 29, policy/CSV order
def pose36_from_jp(root_xyz, root_xyzw, jp: Sequence[float]) -> tuple[float, ...]
def context4_from_ring(ring: Sequence[Sequence[float]], hz: float = 50.0) -> tuple[tuple[float, ...], ...]
```

`pose36_from_qpos` reads `qpos[0:3]`, `wxyz_to_xyzw(qpos[3:7])`, then `qpos[a]` for `a in QPOS_QADR`. `jp_from_pose36`: `jp[csv_i] = pose[7 + CSV2MJ[csv_i]]`. `apply_pose36` writes the inverse into a copy. `context4_from_ring`: four poses oldest-first, last is `ring[-1]`, stride `round(hz / 30)` (50 Hz → 2). If `len(ring) < 4`, repeat the last pose to fill.

- [ ] **Step 1: Write the failing tests** in `tests/test_mb_packing.py`

```python
def test_qadr_skips_the_hands(self) -> None:
    self.assertEqual(g1_pack.QPOS_QADR[:22], tuple(range(7, 29)))
    self.assertEqual(g1_pack.QPOS_QADR[22:], tuple(range(36, 43)))
    self.assertEqual(len(g1_pack.JOINT_NAMES), 29)
    self.assertEqual(g1_pack.JOINT_NAMES[15], "left_shoulder_pitch_joint")
    self.assertEqual(g1_pack.JOINT_NAMES[22], "right_shoulder_pitch_joint")

def test_round_trip_leaves_hands_untouched(self) -> None:
    qpos = [float(i) for i in range(50)]
    pose = g1_pack.pose36_from_qpos(qpos)
    self.assertEqual(len(pose), 36)
    self.assertEqual(pose[3:7], g1_pack.wxyz_to_xyzw(tuple(qpos[3:7])))
    out = g1_pack.apply_pose36(qpos, pose)
    self.assertEqual(out[29:36], qpos[29:36])
    self.assertEqual(out[43:50], qpos[43:50])
    self.assertEqual(g1_pack.pose36_from_qpos(out), pose)

def test_jp_uses_csv_order_not_hardware(self) -> None:
    pose = tuple(float(i) for i in range(36))
    jp = g1_pack.jp_from_pose36(pose)
    self.assertEqual(jp[0], pose[7 + 0])   # left_hip_pitch
    self.assertEqual(jp[1], pose[7 + 6])   # right_hip_pitch (CSV2MJ[1] == 6)
    self.assertEqual(jp[2], pose[7 + 12])  # waist_yaw

def test_policy_q_is_not_a_pose36(self) -> None:
    # 29-vector (policy q) must not be accepted as pose36
    with self.assertRaises(ValueError):
        g1_pack.pose36_from_qpos([0.0] * 29)

def test_context4_repeats_when_short(self) -> None:
    one = tuple(float(i) for i in range(36))
    ctx = g1_pack.context4_from_ring([one])
    self.assertEqual(ctx, (one, one, one, one))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_mb_packing -v`

Expected: FAIL, `g1_pack` not found.

- [ ] **Step 3: Implement `spike/wbc/g1_pack.py`**

Stdlib only (`math` is fine; no NumPy). Tests insert `spike/wbc` on `sys.path` the same way `reach_vr.py` inserts the repo root.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_mb_packing -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spike/wbc/g1_pack.py tests/test_mb_packing.py
git commit -F /tmp/msg.txt
```

Message: `add g1 pose36 packing that skips the interleaved hands`

---

### Task 2: Envelope and stepped placement

**Files:**
- Create: `spike/wbc/end_pose.py` (stdlib at import time; mink stays inside `solve_*`)
- Test: `tests/test_end_pose.py`

**Interfaces:**
- Consumes: Task 1 packing only if a later solve writes pose36; geometry does not need it.
- Produces:

```python
PLANTED_YAW_RAD = math.radians(35)
PLANTED_RESIDUAL_M = 0.05
TRACK_SUCCESS_M = 0.08
CLASP_START_M = 0.08
MAX_STEP_M = 1.5
MAX_STEP_GEN = 4
STEP_CHAIN_S = 8.0
HAND_PREF_RIGHT = (0.35, -0.28, 0.15)
HAND_PREF_LEFT = (0.35, 0.28, 0.15)

def yaw_error(p_root: tuple[float, float, float]) -> float  # atan2(y, x)
def classify(p_root: tuple[float, float, float], planted_residual_m: float | None) -> str
    # "planted" iff abs(yaw_error) <= PLANTED_YAW_RAD and residual is not None and residual < PLANTED_RESIDUAL_M
    # else "step". residual None (not yet solved) with yaw inside the gate returns "need_planted_solve"
def step_root(obj_xy: tuple[float, float], side: str, root_xy_guess: tuple[float, float]) -> tuple[tuple[float, float], float]
    # yaw and root_xy from spec §6.4 steps 1–2. Caller checks distance.
def step_too_far(root_xy, guess_xy) -> bool  # hypot > MAX_STEP_M
```

Do **not** implement mink in this task.

- [ ] **Step 1: Write the failing tests** in `tests/test_end_pose.py`

```python
def test_in_front_short_reach_is_planted(self) -> None:
    self.assertEqual(end_pose.classify((0.35, -0.05, 0.15), 0.02), "planted")

def test_ninety_degrees_steps_even_if_arm_can_crane(self) -> None:
    self.assertEqual(end_pose.classify((0.05, -0.40, 0.15), 0.01), "step")

def test_in_front_but_short_arm_steps(self) -> None:
    self.assertEqual(end_pose.classify((0.80, 0.0, 0.15), 0.12), "step")

def test_step_root_puts_preferred_hand_on_the_object(self) -> None:
    obj = (1.0, 0.0)
    root_xy, yaw = end_pose.step_root(obj, "right", (0.0, 0.0))
    # Rz(yaw) @ HAND_PREF_RIGHT + root_xy ≈ obj, within 1 mm
    self.assertFalse(end_pose.step_too_far(root_xy, (0.0, 0.0)))

def test_three_metre_unprojection_is_too_far(self) -> None:
    root_xy, yaw = end_pose.step_root((3.0, 0.0), "right", (0.0, 0.0))
    self.assertTrue(end_pose.step_too_far(root_xy, (0.0, 0.0)))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_end_pose -v`

Expected: FAIL, `end_pose` not found.

- [ ] **Step 3: Implement the geometry in `spike/wbc/end_pose.py`**

No NumPy. `solve_planted` / `solve_stepped` can exist as `raise NotImplementedError` stubs for Task 5.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_end_pose -v`

Expected: PASS.

- [ ] **Step 5: Commit**

Message: `classify planted vs step from yaw gate and arm residual`

---

### Task 3: Session mailbox

**Files:**
- Create: `spike/wbc/mb_mailbox.py`
- Test: `tests/test_mb_mailbox.py`

**Interfaces:**
- Consumes: Task 1 `pose36` tuples; `pathlib` + json.
- Produces: files under a caller-supplied `session_dir` (live: `~/.local/state/Clank/sessions/body/`):

```python
POSE36_NAME = "pose36.json"
CLIP_NAME = "mb_clip.npz"       # written by adapter (venv); mailbox only stores the path
PLAY_NAME = "mb_play.json"
STATUS_NAME = "mb_status.json"
RING = 32

def write_atomic(path: Path, payload: bytes) -> None  # pid-suffixed tmp + replace, same as body_view
def write_pose36(session_dir: Path, pose: Sequence[float], seq: int, t_mono_ns: int, prev: list | None) -> list
    # prepend? append pose, keep last RING, atomic json:
    # {"seq": int, "t_mono_ns": int, "hz": 50, "poses": [[36 floats], ...]}  oldest first
def read_pose36(session_dir: Path) -> dict | None
def write_play(session_dir: Path, seq: int, path: str = CLIP_NAME) -> None
    # {"seq": seq, "path": path}
def read_play(session_dir: Path) -> dict | None
def write_status(session_dir: Path, seq_ack: int, status: str, frame: int, n: int) -> None
    # status in {"playing", "done", "failed"}
def read_status(session_dir: Path) -> dict | None
```

Latest-wins: each write replaces the file. `write_pose36` is called every 50 Hz tick; keep it a small json dump + rename, no flock.

- [ ] **Step 1: Write the failing tests** in `tests/test_mb_mailbox.py`

Use `tempfile.TemporaryDirectory`. Assert ring caps at 32; `read_pose36` after two writes has `seq` of the second; torn/missing file returns `None` not an exception; `write_play` seq 2 is what `read_play` returns after seq 1; status round-trip.

Also: `test_play_payload_has_no_xyz_or_clip_bytes` — `mb_play.json` keys are exactly `seq` and `path`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_mb_mailbox -v`

Expected: FAIL, `mb_mailbox` not found.

- [ ] **Step 3: Implement `spike/wbc/mb_mailbox.py`**

Stdlib. Duplicate the pid-tmp `write_atomic` here (do not import `body_view`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_mb_mailbox -v`

Expected: PASS.

- [ ] **Step 5: Commit**

Message: `add pose36 and clip-play mailbox as session json files`

---

### Task 4: Resample and flatten-hold

**Files:**
- Create: `spike/wbc/clip_from_gen.py`
- Test: `tests/test_mb_clip.py`

**Interfaces:**
- Consumes: Task 1 `jp_from_pose36`, `xyzw_to_wxyz`.
- Produces:

```python
SRC_HZ = 25          # ARDY-G1 runs at 25 fps; 30 was MotionBricks
DST_HZ = 50
LOOKAHEAD = 10
LOOKAHEAD_STEP = 5  # 0.9 s at 50 Hz

def lerp(a: float, b: float, t: float) -> float
def slerp_xyzw(q0, q1, t) -> tuple[float, float, float, float]
def resample_pose36(frames: Sequence[Sequence[float]], src_hz: float = SRC_HZ, dst_hz: float = DST_HZ) -> list[tuple[float, ...]]
def sonic_dict(frames50: Sequence[Sequence[float]]) -> dict
    # keys jp, jv, bp, bq, n. jp (n,29) CSV order; jv finite-diff, last row copied;
    # bp (n,14,3) zeros except [:,0] = root xyz; bq (n,14,4) identity wxyz except [:,0] = xyzw_to_wxyz(root)
    # Values are lists-of-lists so tests stay stdlib. Adapter wraps with numpy when writing npz.
def lookahead_idx(frame: int, n: int, flattened: bool) -> tuple[int, ...]
    # flattened True -> (n-1,)*10; else motion_frames: min(frame + 5*i, n-1) for i in 0..9
def clip_tick(frame: int, n: int) -> tuple[int, bool]
    # if frame >= n-1: (n-1, True)  # stay, flatten
    # else: (frame+1, False)
```

- [ ] **Step 1: Write the failing tests** in `tests/test_mb_clip.py`

```python
def test_resample_25_to_50_length(self) -> None:
    # 25 frames @ 25 Hz (ARDY) = 1.0 s -> 50 frames. Unit xyzw (0,0,0,1).
    frames = [(i / 25, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0) + (0.0,) * 29 for i in range(25)]
    out = clip_from_gen.resample_pose36(frames)
    self.assertEqual(len(out), 50)

def test_resample_30_to_50_still_supported(self) -> None:
    frames = [(i / 30, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0) + (0.0,) * 29 for i in range(30)]
    self.assertEqual(len(clip_from_gen.resample_pose36(frames, src_hz=30.0)), 50)
    self.assertAlmostEqual(out[0][0], frames[0][0])
    self.assertAlmostEqual(out[-1][0], frames[-1][0], places=5)

def test_lookahead_flat_at_end(self) -> None:
    self.assertEqual(clip_from_gen.lookahead_idx(99, 100, True), (99,) * 10)
    live = clip_from_gen.lookahead_idx(0, 100, False)
    self.assertEqual(live[0], 0)
    self.assertEqual(live[1], 5)
    self.assertEqual(live[-1], 45)

def test_clip_tick_flattens_on_last_frame(self) -> None:
    self.assertEqual(clip_from_gen.clip_tick(0, 10), (1, False))
    self.assertEqual(clip_from_gen.clip_tick(9, 10), (9, True))

def test_sonic_jp_is_csv_order(self) -> None:
    pose = tuple(float(i) for i in range(36))
    d = clip_from_gen.sonic_dict([pose, pose])
    self.assertEqual(d["n"], 2)
    self.assertEqual(d["jp"][0], list(g1_pack.jp_from_pose36(pose)))
    self.assertEqual(d["bq"][0][0], list(g1_pack.xyzw_to_wxyz(pose[3:7])))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_mb_clip -v`

Expected: FAIL, `clip_from_gen` not found.

- [ ] **Step 3: Implement `spike/wbc/clip_from_gen.py`**

Stdlib. Slerp: if dot < 0, negate `q1`; then lerp + normalize.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_mb_clip -v`

Expected: PASS.

- [ ] **Step 5: Commit**

Message: `resample motionbricks clips to sonic 50 hz with a flat hold`

---

### Task 5: ARDY generator adapter

**Files:**
- Create: `spike/wbc/ardy_gen.py` (torch + ARDY; the only module that imports them)
- Create: `spike/wbc_game/probe_ardy_reach.py` (game off; prints residuals and timing)
- Test: `tests/test_ardy_gen.py` (stdlib: constraint geometry and the wrapper's pure parts only)

**Interfaces:**
- Consumes: Task 1 `pose36_from_qpos` / `jp_from_pose36`; Task 2 `classify` (root-goal decision only); measured facts in `research/motion-generators.md` §9.
- Produces:

```python
ARDY_CKPT_DIR = Path.home() / ".local/state/Clank/ardy"
ARDY_MODEL = "ARDY-G1-RP-25FPS-Horizon52"
ARDY_FPS = 25
ARDY_HORIZON = 52
ARDY_TEXT_DIM = 4096
CFG_WEIGHT = (1.0, 2.0)          # (text, constraint) -- the scalar form zeroes the constraint
HAND_DELTA_FWD = 0.35            # m, ARDY model space (y up, z forward)
HAND_DELTA_UP = 0.15

def load_ardy(device: str = "cuda") -> object            # text_encoder=False
def zero_text(device: str) -> Tensor                     # zeros(1, 1, 4096)
def hand_constraint(skeleton, pose: dict, hand_delta: tuple[float, float, float]) -> list
def root_goal_constraint(skeleton, num_frames: int, forward_m: float) -> list
def generate(pose36_context: Sequence[Sequence[float]], *, hand_delta, root_forward_m: float | None,
             num_frames: int = ARDY_HORIZON, steps: int = 10,
             history_features=None) -> tuple[list[tuple[float, ...]], dict]
    # returns (25 Hz pose36 frames, metrics dict with seconds/peak_vram/hand_residual_m)
```

`generate` must: build the constraint from the **latest achieved pose** in the context; pass `cfg_weight=CFG_WEIGHT`; reject `steps > model.diffusion.num_base_steps` with a message; run `model.motion_rep.inverse(...)` then `MujocoQposConverter` (which already returns our 36 floats, `(1, T, 36)`); and return the achieved right-hand residual against the requested point.

**Continuity, and the one thing to leave out at first.** ARDY continues from `init_history_sequence`, which the demo fills from its own previously generated *feature* tensor (`scripts/interactive_demo/generation.py:150-176, 275`). Our body state is MuJoCo `qpos`, and there is no `motion_rep.encode(pose36)` — feature assembly lives in `ardy/data_processing`. So v1 does **not** pass history: it generates from a canonical standing pose translated to the body's current root and heading (`init_global_translation`, `init_first_heading_angle`), and lets SONIC absorb the small pose offset because it starts from the live `mjData` regardless. Chained generation inside one reach reuses the previous window's features, which is the case the demo already exercises. A pose36 -> features encoder is a known follow-on, not a Task 5 requirement.

- [ ] **Step 1: Write the failing stdlib tests** in `tests/test_ardy_gen.py`

```python
def test_cfg_weight_is_a_pair_not_a_scalar(self) -> None:
    self.assertEqual(ardy_gen.CFG_WEIGHT, (1.0, 2.0))
    self.assertIsInstance(ardy_gen.CFG_WEIGHT[1], float)
    self.assertGreater(ardy_gen.CFG_WEIGHT[1], 0.0)

def test_steps_above_the_schedule_are_refused(self) -> None:
    with self.assertRaises(ValueError):
        ardy_gen.check_steps(25, num_base_steps=10)

def test_hand_delta_is_forward_and_up_in_ardy_space(self) -> None:
    d = ardy_gen.hand_delta(0.35, 0.15)
    self.assertEqual(d, (0.0, 0.15, 0.35))   # y up, z forward
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -m unittest tests.test_ardy_gen -v`
Expected: FAIL, `ardy_gen` not found.

- [ ] **Step 3: Implement `spike/wbc/ardy_gen.py`**

Import torch/ARDY lazily inside `generate`/`load_ardy` so `import ardy_gen` stays cheap and the stdlib tests can import the module. `check_steps(steps, num_base_steps)` raises `ValueError` with the model's limit in the message.

- [ ] **Step 4: Run stdlib tests**

Run: `python3 -m unittest tests.test_ardy_gen -v`
Expected: PASS.

- [ ] **Step 5: Run the probe, game off, GPU free**

Run: `LD_LIBRARY_PATH=/run/opengl-driver/lib spike/ardy/.venv/bin/python spike/wbc_game/probe_ardy_reach.py`

Expected: prints peak VRAM **< 2.0 GB**, generation **< 1.0 s** per window, and hand residual **< 0.10 m** at `CFG_WEIGHT`. All three are already measured (0.94 GB / 0.07-0.20 s / 60 mm) — the probe exists so a regression is visible, not to discover them again.

- [ ] **Step 6: Commit**

Message: `generate g1 reaches from hand and root constraints with ardy`

---

### Task 6: Reachability escalation

**Files:**
- Modify: `spike/wbc/end_pose.py` (`decide_next`, and demote `step_root`'s docstring to "root-goal placement")
- Test: `tests/test_end_pose.py` (extend)

**Interfaces:**
- Consumes: Task 2 `classify`, `yaw_error`, `step_too_far`, `MAX_STEP_M`; Task 5 `generate`'s `hand_residual_m`.
- Produces:

```python
ROOT_GOAL_RESIDUAL_M = 0.10      # above this, retry once with a root goal
def decide_next(*, kind: str, residual_m: float, escalated: bool, gen: int, elapsed_s: float,
                generated: int) -> str      # "generate" | "escalate" | "stop" | "saturated"
```

Rules: `residual_m <= TRACK_SUCCESS_M` -> `"stop"`; else if not `escalated` and `residual_m > ROOT_GOAL_RESIDUAL_M` and `generated < MAX_STEP_GEN` and `elapsed_s < STEP_CHAIN_S` -> `"escalate"`; else `"stop"` with the best-so-far. **No mink solve, no constructed stepped pose** — the root goal is a constraint given to the generator, which is what ARDY is for. mink is optional as a pre-flight feasibility check and is not required by any test in this plan.

- [ ] **Step 1: Write the failing tests** in `tests/test_end_pose.py`

```python
def test_success_stops(self) -> None:
    self.assertEqual(end_pose.decide_next(kind="planted", residual_m=0.06, escalated=False,
                                          gen=1, elapsed_s=1.0, generated=1), "stop")

def test_short_hand_escalates_once(self) -> None:
    self.assertEqual(end_pose.decide_next(kind="planted", residual_m=0.25, escalated=False,
                                          gen=1, elapsed_s=1.0, generated=1), "escalate")

def test_already_escalated_stops(self) -> None:
    self.assertEqual(end_pose.decide_next(kind="planted", residual_m=0.25, escalated=True,
                                          gen=2, elapsed_s=2.0, generated=2), "stop")

def test_generation_budget_is_bounded(self) -> None:
    self.assertEqual(end_pose.decide_next(kind="planted", residual_m=0.25, escalated=False,
                                          gen=1, elapsed_s=1.0, generated=end_pose.MAX_STEP_GEN), "saturated")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -m unittest tests.test_end_pose -v`
Expected: FAIL, `decide_next` not defined.

- [ ] **Step 3: Implement `decide_next` in `spike/wbc/end_pose.py`**

Pure stdlib arithmetic over the constants; no imports added beyond `math`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_end_pose -v`
Expected: PASS (all previous Task 2 tests still green).

- [ ] **Step 5: Commit**

Message: `escalate to a root goal once instead of authoring a stepped pose`

---

### Task 7: Executor plays a deposited clip

**Files:**
- Modify: `spike/wbc/sonic_executor.py` (`__init__`, `step`)
- Modify: `spike/wbc/body_view.py` (pass `session_dir=out` into `SonicExecutor`)
- Test: `tests/test_mb_play.py` (stdlib; does **not** construct `SonicExecutor`)

**Interfaces:**
- Consumes: Task 3 mailbox, Task 4 `clip_tick` / `lookahead_idx` / `sonic_dict`.
- Produces: `SonicExecutor(session_dir: Path | None = None)`. When `session_dir` is set, each `step`:

  1. Pack achieved qpos with Task 1 and `write_pose36`.
  2. `read_play`; if `seq != self._mb_seq`, load `npz`/`sonic_dict` from `path`, set `runner.mode = 0`, `self.motion = loaded`, `self._clip_frame = 0`, `self._flattened = False`, **do not** `reset()`. Keep current `mjData`.
  3. If playing: `build_encoder_obs(self.motion, self._clip_frame, ...)` with no VR overlay (mode 0 ignores it). Then `clip_tick`; on flatten: `lookahead` is flat (override `motion_frames` by repeating the last row in a copy, or pass a motion whose remaining frames equal the last pose — **pick: replace `jp/jv/bp/bq` rows `[frame:]` with the last played row** so `motion_frames` cannot see walking). `write_status(playing|done)`.
  4. When flattened and status `done`: `runner.mode = 1`, restore `HOLD_CLIP` / `HOLD_FRAME` freeze, freeze-hold via the incoming `BodyCommand` (adapter sends `hold()`). Do not teleport.

Default `session_dir=None` keeps today's behaviour bit-identical (no files, mode 1).

`body_view.py`: `SonicExecutor(clip=..., session_dir=out)`.

- [ ] **Step 1: Write the failing tests** in `tests/test_mb_play.py`

Do not import `sonic_executor` (it pulls mujoco). Test a helper extracted to `clip_from_gen` or a tiny `mb_play.py`:

```python
# spike/wbc/mb_play.py  (stdlib)
def on_play_seq_change(old_seq: int | None, play: dict | None) -> bool
def after_clip(frame: int, n: int) -> str  # "playing" | "done"
```

`on_play_seq_change(None, {"seq": 1, "path": "mb_clip.npz"})` is True; same seq False. `after_clip(9, 10) == "done"`.

Grep test in the same file:

```python
def test_executor_does_not_import_the_generator(self) -> None:
    text = Path("spike/wbc/sonic_executor.py").read_text()
    self.assertNotIn("ardy_gen", text)
    self.assertNotIn("end_pose", text)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_mb_play -v`

Expected: FAIL until the helper exists / grep is red on current executor the grep should already PASS for the generator names; that is fine — the seq helper fails).

- [ ] **Step 3: Implement session_dir playback in `sonic_executor.py` and pass `out` from `body_view.py`**

Mode 0 obs path already writes 10×29 `jp`/`jv`. Use it. Flatten by rewriting trailing clip rows, not by calling `reset()`.

- [ ] **Step 4: Run stdlib tests**

Run: `python3 -m unittest tests.test_mb_play tests.test_mb_packing tests.test_mb_mailbox tests.test_end_pose tests.test_mb_clip tests.test_mb_gen -v`

Expected: PASS. Then `python3 -m unittest discover -s tests -t .` still PASS (no SonicExecutor constructed).

- [ ] **Step 5: Commit**

Message: `play deposited motionbricks clips in sonic mode 0 without reset`

---

### Task 8: `reach_ardy` adapter

**Files:**
- Create: `spike/wbc_game/reach_ardy.py`
- Create: `spike/wbc_game/probe_ardy_track.py`
- Test: `tests/test_reach_ardy.py` (stdlib: chain policy, clasp timing, no STEP_GUARD)

**Interfaces:**
- Consumes: Tasks 1–7, `clank.player_scale.read`, `clank.pixel3d` / the same unproject `reach_vr.py` uses, `control.sock` + `body_state.json` like `reach_vr.send` / `state`.
- Produces: CLI

```text
reach_ardy.py <px> <py> [--side left|right] [--world X Y Z] [--grab] [--tol 0.08]
```

Loop:

1. Read `pose36.json`; abort if missing (`body` not running).
2. Object in robot world metres (`PlayerScale.pixel_point` or `--world`). Never apply yaw offset to the camera (same bug as `reach_vr` docstring).
3. `end_pose.classify(...)` decides only whether a root goal is plausible; there is no mink solve. Missing checkpoint / torch / CUDA → print, exit 3, **do not** kill the session.
4. `ardy_gen.generate(context4_from_ring(...), hand_delta=..., root_forward_m=None)`.
5. `resample` → write `mb_clip.npz` (numpy, venv) → `write_play` new seq.
6. Poll `mb_status` until `done` or 8 s. While playing, if `--grab` and `dist(hand, obj) <= CLASP_START_M`, `send(authority=AUTH_CLOSURE, closure=...)` on `control.sock`. Do **not** wait for clip end. Do **not** send a 20 Hz head pin.
7. `end_pose.decide_next(...)` with the measured residual. `"escalate"` regenerates **once** with a root goal constraint; anything else stops with best-so-far. `"saturated"` prints the residual and exits 2.
8. Print achieved residual (world hand vs object). Exit 0 if ≤ `TRACK_SUCCESS_M` (8 cm).

`tests/test_reach_ardy.py`:

```python
def test_reach_ardy_does_not_use_step_guard(self) -> None:
    text = Path("spike/wbc_game/reach_ardy.py").read_text()
    self.assertNotIn("STEP_GUARD", text)
    self.assertNotIn("WRIST_BOX", text)

def test_chain_stops_at_four(self) -> None:
    # import a pure function decide_next(kind, gen, elapsed, residual, prev_residual) -> "step"|"planted_once"|"stop"
```

`decide_next(...)` lives in `end_pose.py` (Task 6), so the chain policy is unit-tested without a session.

`probe_ardy_track.py` (game off, body_view running against the standing clip, **no** Job Simulator): generate a constrained reach, deposit it, report the achieved hand residual against the constrained point, the root path, and any fall. Gate: residual **< 8 cm**, no fall, look error **< 15°**. If residual **> 15 cm**, stop — do not fall back to the `reach_vr` cruise.

- [ ] **Step 1: Write the failing adapter tests**

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_reach_ardy -v`

- [ ] **Step 3: Implement `reach_ardy.py`**

- [ ] **Step 4: Run stdlib tests and the track probe**

Run: `python3 -m unittest tests.test_reach_ardy -v` and `LD_LIBRARY_PATH=/run/opengl-driver/lib spike/ardy/.venv/bin/python spike/wbc_game/probe_ardy_track.py`

Expected: unittest PASS; probe residual < 0.08 m or a printed kill at > 0.15 m.

- [ ] **Step 5: Commit**

Message: `reach with ardy constraints and a sonic tracked clip`

---

### Task 9: Live grab gate

**Files:** none new. Operator run.

**Interfaces:** consumes Task 8 `reach_ardy.py --grab`.

- [ ] **Step 1: Warm the ARDY checkpoint with the game down.** `nvidia-smi` before and after; record peak VRAM (measured 0.94 GB).

- [ ] **Step 2: Session up.** Reset body (`POST /api/body/stop` then `/start`). One in-front grab (card or operator pixel). One off-axis or out-of-reach grab. `--grab` both.

Expected: in-front feet planted, chest toward object, residual ≤ 8 cm, HMD looks at it, clasp while approaching. Off-axis: the model turns or steps, and the player moves in Job Simulator. `body_pose.csv` `curl_*` columns change on grab. `hold` / `look` / `rest` still work afterward.

- [ ] **Step 3: If Step 2 passes, add a one-line note at the top of `reach_vr.py`** pointing at `reach_ardy.py` as the live path; do not delete the cruise.

- [ ] **Step 4: Commit that note only if Step 2 passed**

Message: `point live grab at reach_ardy after the in-game gate`

---

## Spec coverage

| Spec | Task |
| --- | --- |
| §6 still frame, 35° / 5 cm, HAND_PREF, MAX_STEP_M | 2, 5 |
| §7 generate, tokens, trim, TF32 off, no IKPin | 6 |
| §7.3 packing, xyzw, CSV2MJ, hands | 1 |
| §8.1 mode 0, no reset, flatten-hold | 4, 7 |
| §8.2 clasp at 8 cm via AUTH_CLOSURE | 8 |
| §8.3 step chain ≤ 4 / 8 s, one planted correction | 8 |
| §8.4 no 20 Hz head pin | 8 |
| §9 mailbox, adapter generates, executor plays | 3, 7, 8 |
| §11.1–11.4 game-off gates | 1, 5, 6, 8 probe |
| §11.5 live grab | 9 |
| Anti-goal: no session tear-down on missing ONNX | 8 exit 3 |
| `hold`/`look`/`rest` stay VR mode 1 | 7 default |

## Execution notes

Mink and ONNX probes (Tasks 5, 6, 8 probe) run under `spike/e3/.venv/bin/python`. Flake `unittest discover` must stay green after every task that touches `tests/`.

---

## Fallback (only on a new decision)

If ARDY fails a gate on this machine, the documented fallback is **MotionBricks live + Kimodo-authored constraints**, not a return to the old Task 5/6 text: Kimodo generates the still poses once from `right-hand` + foot constraints (so they are in-distribution), MotionBricks in-betweens them at 2.67 ms with no torch in the loop. Rationale and measurements: `research/motion-generators.md` §6. That path needs its own plan revision; do not build it from this document's superseded tasks.
