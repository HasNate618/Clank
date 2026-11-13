# MotionBricks Reach Implementation Plan

> **Archived 2026-11-13.** Implementation plan, carried out. Kept for the task breakdown and the gates it defined; the code is the record now.
> Current behaviour: [`README.md`](../../../README.md) · agent rules: [`AGENTS.md`](../../../AGENTS.md).

> **Superseded** by `2026-11-11-ardy-reach.md`. The work it describes was carried out against ARDY
> instead; keep this for the task breakdown and the gates, not as a sequence to follow.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the VR-cruise reach/grab with an IK still frame, a MotionBricks inbetween, and SONIC mode-0 tracking, so an in-front grab plants and looks, and an off-axis grab steps or pivots.

**Architecture:** Adapter process (`reach_mb.py`) authors a 36-float G1 pose and generates a clip. The 50 Hz body process only writes a pose ring and plays deposited clips. CUDA stays out of SONIC. `hold` / `look` / `rest` stay on VR mode 1. `reach_vr.py` stays until the live gate passes.

**Tech Stack:** Python 3 stdlib in packing, mailbox, geometry, and `tests/`. NumPy / MuJoCo / mink / ONNX in `spike/e3/.venv`. MotionBricks G1 ONNX (FP16) via ONNX Runtime CUDA, TensorRT engine if it loads, CPU ORT last. `python3 -m unittest` (pytest is not installed).

**Spec:** `docs/archive/specs/2026-11-11-motionbricks-reach-design.md`.

## Global Constraints

- `clank/` stays **standard-library only**. No generator, mink, ONNX, or MotionBricks import in `clank/`.
- `tests/` stays stdlib: no NumPy, no MuJoCo, no ONNX, no constructing `SonicExecutor`.
- Unittest **must not** open `:4242`, `:5901`, `:8765`, `:8770`, `:8771`, or `:11434`.
- Do **not** edit `run.sh`, `config.toml`, `flake.nix`, `clank/orchestrator.py`, `clank/config.py`, `clank/body.py` (`FRAMES`, `BodyCommand`), or `reach_vr.py` except a later one-line pointer after §11.5.
- Do **not** put object XYZ or a clip on `BodyCommand`. Do **not** feed mink joints to SONIC as `upper_body` or `ctrl`.
- Do **not** enable TF32. Do **not** build a TensorRT plan while the game is up.
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
SRC_HZ = 30
DST_HZ = 50
LOOKAHEAD = 10
LOOKAHEAD_STEP = 5  # 0.9 s at 50 Hz

def lerp(a: float, b: float, t: float) -> float
def slerp_xyzw(q0, q1, t) -> tuple[float, float, float, float]
def resample_pose36(frames: Sequence[Sequence[float]], src_hz: float = 30.0, dst_hz: float = 50.0) -> list[tuple[float, ...]]
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
def test_resample_30_to_50_length(self) -> None:
    # 30 frames @ 30 Hz = 1.0 s -> 50 frames. Unit xyzw (0,0,0,1).
    frames = [(i / 30, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0) + (0.0,) * 29 for i in range(30)]
    out = clip_from_gen.resample_pose36(frames)
    self.assertEqual(len(out), 50)
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

### Task 5: Mink still frames

**Files:**
- Modify: `spike/wbc/end_pose.py` (`solve_planted`, `solve_stepped`, `end_pose`)
- Create: `spike/wbc_game/probe_end_pose.py` (game off; prints residuals)
- Test: geometry already in Task 2. This task's check is the probe, not flake unittest.

**Interfaces:**
- Consumes: Task 1 `pose36_from_qpos` / `apply_pose36`; Task 2 `classify` / `step_root`; `scene_sonic_floor.xml` via `sonic_runner.XML`; mink `FrameTask` / `PostureTask` / `ComTask` as in `spike/wbc_classical/wbc_spike.py` (`DEFAULT_WEIGHTS`, head ori cost **500** for look).
- Produces:

```python
@dataclass(frozen=True)
class EndPose:
    kind: str                    # "planted" | "step" | "saturated"
    pose36: tuple[float, ...]    # zeros if saturated
    residual_m: float
    yaw: float

def solve_planted(qpos: Sequence[float], obj_world: tuple[float, float, float], side: str) -> EndPose
def solve_stepped(qpos: Sequence[float], obj_world: tuple[float, float, float], side: str) -> EndPose
def end_pose(qpos, obj_world, side) -> EndPose
    # classify with a planted solve when yaw is inside the gate; fall through to stepped if residual >= 5 cm
    # stepped: if step_too_far -> kind saturated, do not solve
```

Reported hand is `wrist_yaw_link` + `sonic_runner.VR_OFFSETS[side]`, same as `arm_ik.vr_point` but in **world** metres (mink target is world). Scratch `MjModel` of `scene_sonic_floor.xml`; do not touch the live `mjData`. Seed planted from `qpos`; seed stepped from the stand keyframe with `step_root` yaw/xy and stand `root_z`. Clamp hinges to `jnt_range ± 0.02`. Wrist ori cost 100, roll free (LookAt or skip ori). Other wrist holds achieved.

`probe_end_pose.py`: load stand qpos from a scratch model, object `(rest_hand + (0.35, 0, 0.15))` world → planted residual **< 15 mm**, `|Δroot_xy| < 1 cm`, head yaw toward object **< 2°**. Second object 1.0 m forward → `kind=="step"`, hand residual **< 15 mm** on the keyframe.

- [ ] **Step 1: Implement `solve_planted` / `solve_stepped` / `end_pose` in `spike/wbc/end_pose.py`**

Lazy-import mink/mujoco/numpy inside those functions so `import end_pose` stays flake-safe.

- [ ] **Step 2: Run the probe, game off**

Run: `spike/e3/.venv/bin/python spike/wbc_game/probe_end_pose.py`

Expected: printed planted residual < 0.015 m, stepped residual < 0.015 m, both kinds as above. If mink cannot aim the VR lever, stop and fix the site/offset; do not proceed.

- [ ] **Step 3: Commit**

Message: `author planted and stepped g1 still frames with mink`

---

### Task 6: MotionBricks generate

**Files:**
- Create: `spike/wbc/motionbricks_gen.py`
- Create: `spike/wbc_game/probe_mb_gen.py`
- Test: `tests/test_mb_gen.py` (no GPU; mocks the ONNX session)

**Interfaces:**
- Consumes: Task 1 pose36; spec §7.1 call shape.
- Produces:

```python
PLANTED_TOKENS = tuple(range(6, 11))   # 6..10
STEPPED_TOKENS = tuple(range(10, 17))  # 10..16
ONNX_DIR = Path.home() / ".local/state/Clank/motionbricks"
TRT_CACHE = Path.home() / ".local/state/Clank/motionbricks-trt"

def allowed_tokens(kind: str) -> tuple[int, ...]
def valid_frame_count(pred_num_tokens: int) -> int  # 4*n - 6, clamp 18..58
def generate(context4: Sequence[Sequence[float]], target: Sequence[float], *, kind: str) -> list[tuple[float, ...]]
    # returns trimmed, endpoint-corrected 30 Hz poses. Raises FileNotFoundError if ONNX missing.
```

Fallback order inside `generate`: TensorRT G1 engine if `MotionBricksUnitreeG1Engine` imports; else `onnxruntime` CUDA; else CPU. `enable_tf32=False`. `num_poses_to_use=1`. `correct_generated_diff=True`. Do not ship raw `pred_qpos`.

Weights: download FP16 `mmm_v6_g1_fp16.onnx` from Hugging Face `nvidia/MotionBricks` (G1 folder) into `ONNX_DIR` on first use if absent. Do not git-add the ONNX. Do not `git lfs pull` a 134-byte stub and call it a model.

`tests/test_mb_gen.py`: `allowed_tokens("planted") == PLANTED_TOKENS`, stepped 10–16, `valid_frame_count(8) == 26`, `valid_frame_count(6) == 18`, `valid_frame_count(16) == 58`. Patch `generate`'s session to return a linear lerp of context[-1] → target over 24 raw frames and assert the trimmed clip starts at context[-1] and ends at target (endpoint correction). No real ONNX.

- [ ] **Step 1: Write the failing token/trim tests** in `tests/test_mb_gen.py`

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_mb_gen -v`

Expected: FAIL, `motionbricks_gen` not found.

- [ ] **Step 3: Implement `spike/wbc/motionbricks_gen.py`**

Keep Warp/Newton optional. If the NVIDIA engine is not importable, copy only heading alignment, 3-frame boundary trim, and endpoint correction (root lerp over the clip, joint fade) so ORT output matches spec §7.1. Document the copy as such in the module docstring.

- [ ] **Step 4: Run stdlib tests**

Run: `python3 -m unittest tests.test_mb_gen -v`

Expected: PASS.

- [ ] **Step 5: Probe generation, game off, GPU allowed**

Run: `spike/e3/.venv/bin/python spike/wbc_game/probe_mb_gen.py`

Uses stand × 4 as context and Task 5 still frames as targets. Assert duration 0.6–1.9 s, no NaNs, first frame ≈ stand, last ≈ target. Print whether backend was TRT / CUDA / CPU. If OOM, drop to CPU and record that; do not build TRT next to a session.

- [ ] **Step 6: Commit** (code and probes only, no ONNX weights)

Message: `generate g1 inbetweens from a still frame with motionbricks`

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
def test_executor_does_not_import_motionbricks(self) -> None:
    text = Path("spike/wbc/sonic_executor.py").read_text()
    self.assertNotIn("motionbricks_gen", text)
    self.assertNotIn("end_pose", text)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_mb_play -v`

Expected: FAIL until the helper exists / grep is red on current executor (grep should already PASS for motionbricks_gen; that's fine — the seq helper fails).

- [ ] **Step 3: Implement session_dir playback in `sonic_executor.py` and pass `out` from `body_view.py`**

Mode 0 obs path already writes 10×29 `jp`/`jv`. Use it. Flatten by rewriting trailing clip rows, not by calling `reset()`.

- [ ] **Step 4: Run stdlib tests**

Run: `python3 -m unittest tests.test_mb_play tests.test_mb_packing tests.test_mb_mailbox tests.test_end_pose tests.test_mb_clip tests.test_mb_gen -v`

Expected: PASS. Then `python3 -m unittest discover -s tests -t .` still PASS (no SonicExecutor constructed).

- [ ] **Step 5: Commit**

Message: `play deposited motionbricks clips in sonic mode 0 without reset`

---

### Task 8: `reach_mb` adapter

**Files:**
- Create: `spike/wbc_game/reach_mb.py`
- Create: `spike/wbc_game/probe_mb_track.py`
- Test: `tests/test_reach_mb.py` (stdlib: classify loop, clasp timing, no STEP_GUARD)

**Interfaces:**
- Consumes: Tasks 1–7, `clank.player_scale.read`, `clank.pixel3d` / the same unproject `reach_vr.py` uses, `control.sock` + `body_state.json` like `reach_vr.send` / `state`.
- Produces: CLI

```text
reach_mb.py <px> <py> [--side left|right] [--world X Y Z] [--grab] [--tol 0.08]
```

Loop:

1. Read `pose36.json`; abort if missing (`body` not running).
2. Object in robot world metres (`PlayerScale.pixel_point` or `--world`). Never apply yaw offset to the camera (same bug as `reach_vr` docstring).
3. `end_pose(qpos_from_pose36, obj, side)`. `saturated` → print residual, exit 2. Missing ONNX / CUDA → print, exit 3, **do not** kill the session.
4. `generate(context4_from_ring(...), pose36, kind=end.kind)`.
5. `resample` → write `mb_clip.npz` (numpy, venv) → `write_play` new seq.
6. Poll `mb_status` until `done` or 8 s. While playing, if `--grab` and `dist(hand, obj) <= CLASP_START_M`, `send(authority=AUTH_CLOSURE, closure=...)` on `control.sock`. Do **not** wait for clip end. Do **not** send a 20 Hz head pin.
7. Re-run envelope on latest pose36. If still `step` and gen < `MAX_STEP_GEN` and wall < `STEP_CHAIN_S` and not `step_too_far`: goto 4 with new stepped still frame. Elif planted residual > 5 cm: **one** planted generate, then stop. If residual grew: `saturated`, best-so-far.
8. Print achieved residual (world hand vs object). Exit 0 if ≤ `TRACK_SUCCESS_M` (8 cm).

`tests/test_reach_mb.py`:

```python
def test_reach_mb_does_not_use_step_guard(self) -> None:
    text = Path("spike/wbc_game/reach_mb.py").read_text()
    self.assertNotIn("STEP_GUARD", text)
    self.assertNotIn("WRIST_BOX", text)

def test_chain_stops_at_four(self) -> None:
    # import a pure function decide_next(kind, gen, elapsed, residual, prev_residual) -> "step"|"planted_once"|"stop"
```

Extract `decide_next(...)` into `end_pose.py` or `reach_mb.py` so the chain policy is unit-tested without a session.

`probe_mb_track.py` (game off, body_view running against the standing clip, **no** Job Simulator): generate a planted clip, deposit it, report achieved hand residual vs still-frame hand, root path, falls. Gate: planted residual **< 8 cm**, no fall, look error **< 15°**. If residual **> 15 cm**, stop the project — do not fall back to `reach_vr` cruise.

- [ ] **Step 1: Write the failing adapter tests**

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_reach_mb -v`

- [ ] **Step 3: Implement `reach_mb.py` + `decide_next`**

- [ ] **Step 4: Run stdlib tests and the track probe**

Run: `python3 -m unittest tests.test_reach_mb -v` and `spike/e3/.venv/bin/python spike/wbc_game/probe_mb_track.py`

Expected: unittest PASS; probe residual < 0.08 m or a printed kill at > 0.15 m.

- [ ] **Step 5: Commit**

Message: `reach with an ik still frame and a motionbricks clip`

---

### Task 9: Live grab gate

**Files:** none new. Operator run.

**Interfaces:** consumes Task 8 `reach_mb.py --grab`.

- [ ] **Step 1: Build the TensorRT/ORT plan with the game down.** `nvidia-smi` before and after. Record backend.

- [ ] **Step 2: Session up.** Reset body (`POST /api/body/stop` then `/start`). One in-front grab (card or operator pixel). One off-axis or out-of-reach grab. `--grab` both.

Expected: in-front feet planted, chest toward object, residual ≤ 8 cm, HMD looks at it, clasp while approaching. Off-axis: pivot/step, player moves in Job Simulator. `body_pose.csv` `curl_*` columns change on grab. `hold` / `look` / `rest` still work afterward.

- [ ] **Step 3: If Step 2 passes, add a one-line note at the top of `reach_vr.py`** pointing at `reach_mb.py` as the live path; do not delete the cruise.

- [ ] **Step 4: Commit that note only if Step 2 passed**

Message: `point live grab at reach_mb after the in-game gate`

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
