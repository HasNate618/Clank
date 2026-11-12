# Per-run Recordings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every run leaves a playable `recording.mp4` of the game view beside the robot view, with the game's audio, and the dashboard gets a page that lists past runs and plays them with the run's diary.

**Architecture:** `clank/recording.py` holds one `Recorder` per run: it starts three ffmpeg encoders at the run's `running` phase (game from a rawvideo pipe fed by the depth-stream reader, robot from an `image2pipe` JPEG feed, audio from `ears.capture()`), closes them when the run ends, and hands a detached job the compositing pass that `hstack`s the two pictures and muxes the audio into `recording.mp4`. `RunManager` owns the recorder; the dashboard serves a listing, a Range-capable video route, and a new page.

**Tech Stack:** Python 3 standard library only in `clank/` (`subprocess`, `threading`, `mmap` via `clank.depth_stream`), ffmpeg on `PATH` for encoding, `unittest` for tests, vanilla JS/CSS for the page.

**Spec:** `docs/superpowers/specs/2026-11-12-recordings-design.md`

## Global Constraints

- `clank/` is **standard-library only**. No numpy, PIL, requests, pytest. Tests use `unittest`.
- A missing or failing recorder **never** fails, delays or blocks a run: every recorder failure becomes a manifest with `state: "failed"` and a reason.
- Geometry is fixed: game 960×540 left, robot tile scaled to 540 high (720 wide) right, total **1680×540**, black padding for a missing source.
- Frame rate default 15 fps (`robot_fps` 10), so one frame is 67 ms.
- `recording.json` states, in order of life: `recording` → `composing` → `ready` | `failed`.
- `recording.json`'s `started_at` is the **recorder's** start, never `run.started_at` (which is ~25 s earlier: gate, game restart, body reset).
- Retention: `max_total_mb = 20480`; prune oldest-first after a composite finishes, never the run just recorded, log each deletion to `session.log`.
- New config keys go in all four places: `config.example.toml`, `DEFAULTS`, the section dataclass, `_validate`. Unknown `CLANK_*` overrides stay rejected. No new `CLANK_*` env vars.
- Commit messages: lowercase prose, no conventional-commit prefix.
- Test command: `mkdir -p "${XDG_RUNTIME_DIR:-/tmp/clank-xdg}" && XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/clank-xdg}" python3 -m unittest discover -s tests -t .`
- Before any `nix build`, `git add` new files (flakes see only tracked files).

## Review Focus

1. **ffmpeg missing from `PATH`.** The run must reach `running`, play its brain loop, keep its diary, and leave `recording.json` with `state: "failed"` and ffmpeg named — the operator's page says "recording failed", not an empty list and not a crashed run.
2. **No depth stream** (layer disabled, game not rendering): the game source fails, the composite is the robot tile plus black padding, and the page still plays it.
3. **Dashboard killed mid-run**: its encoders outlive it. The next dashboard start must delete the orphaned parts, mark the run `failed: dashboard exited`, and never leave unbounded bytes on disk.
4. **Range edges** (`bytes=0-`, `bytes=-500`, past EOF, malformed, multi-range): 206 for the first three, 416 for the malformed and past-EOF, 200 for multi-range and for a plain GET.
5. **Diary events outside the file** (before the recording started, or after `duration_s`): the page clamps a seek to `[0, duration_s]` and never produces a negative or `NaN` `currentTime`.
6. **A muted game** (`module-stream-restore` once remembered `Mute: yes` for JobSimulator): the recording is `ready` with `audio_peak: 0`, and the page says the track was silent rather than pretending all is well.

## File Structure

- `clank/recording.py` (new): `SIZE` constants, `composite_argv()`, `parse_range()`, `Recorder`, `run_composite()`, `sweep_stale()`, `prune()`, `list_recordings()`. One responsibility: turning a live run into a file and knowing what is on disk.
- `clank/config.py`: `RecordingConfig` + `[recording]` defaults + validation.
- `clank/run.py`: own a `Recorder`, start it at `running`, stop it in `_finish`, expose its state in `snapshot()`.
- `clank/dashboard.py`: `/api/recordings`, `/api/recordings/<id>/video`, `/recordings` page, startup sweep + prune hook.
- `clank/static/recordings.html` + `recordings.js`: the page. `clank/static/index.html` gains a link.
- Tests: `tests/test_recording.py` (new), plus additions in `tests/test_config.py`, `tests/test_run.py`, `tests/test_dashboard.py`.

---

### Task 1: The `[recording]` config section

**Files:**
- Modify: `clank/config.py` (dataclass near `RunConfig`, `DEFAULTS`, `Config`, `_validate`)
- Modify: `config.example.toml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Config.recording: RecordingConfig` with fields `enabled: bool`, `fps: int`, `robot_fps: int`, `crf: int`, `preset: str`, `ffmpeg: str`, `keep_parts: bool`, `max_total_mb: int`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_config.py`:

```python
class RecordingConfigTests(RunConfigTests):  # inherits write_toml
    def test_defaults_describe_a_small_watchable_recording(self):
        cfg = load_config(None, env={})
        self.assertTrue(cfg.recording.enabled)
        self.assertEqual(cfg.recording.fps, 15)
        self.assertEqual(cfg.recording.robot_fps, 10)
        self.assertEqual(cfg.recording.crf, 28)
        self.assertEqual(cfg.recording.preset, "veryfast")
        self.assertEqual(cfg.recording.ffmpeg, "ffmpeg")
        self.assertFalse(cfg.recording.keep_parts)
        self.assertEqual(cfg.recording.max_total_mb, 20480)

    def test_a_run_can_record_off(self):
        cfg = load_config(self.write_toml("[recording]\nenabled = false\n"), env={})
        self.assertFalse(cfg.recording.enabled)

    def test_zero_or_negative_rates_are_refused(self):
        for bad in ("fps = 0", "robot_fps = -1", "crf = 60", "max_total_mb = -1"):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                load_config(self.write_toml(f"[recording]\n{bad}\n"), env={})
```

- [ ] **Step 2: Run it, expect failure** — `XDG_RUNTIME_DIR=... python3 -m unittest tests.test_config.RecordingConfigTests -v`; expect `AttributeError: 'Config' object has no attribute 'recording'`.

- [ ] **Step 3: Implement** — `RecordingConfig` frozen dataclass with the eight fields above, in the field order listed; `DEFAULTS["recording"]` with the eight default values (`enabled` True, `fps` 15, `robot_fps` 10, `crf` 28, `preset` "veryfast", `ffmpeg` "ffmpeg", `keep_parts` False, `max_total_mb` 20480); `Config.recording`; validation: `fps` and `robot_fps` > 0, `crf` in `1..51`, `max_total_mb` >= 0, `preset` and `ffmpeg` non-empty. Add the commented section to `config.example.toml` beside `[run]`, with the measured sizes in the comments (24.1 MB/min at 30 fps measured; 15 fps default).

- [ ] **Step 4: Run it, expect pass** — same command; expect `OK`.

- [ ] **Step 5: Commit** — `git add clank/config.py config.example.toml tests/test_config.py && git commit -m "give a recording its own settings"`

---

### Task 2: `composite_argv()` and `parse_range()`, as pure functions

**Files:**
- Create: `clank/recording.py`
- Test: `tests/test_recording.py`

**Interfaces:**
- Consumes: `Config.recording` (Task 1).
- Produces:
  - `GAME_WIDTH = 960`, `GAME_HEIGHT = 540`, `ROBOT_WIDTH = 720`, `WIDTH = 1680`, `HEIGHT = 540`
  - `composite_argv(*, game: Path, robot: Path | None, audio: Path | None, out: Path, fps: int, crf: int, preset: str, ffmpeg: str = "ffmpeg") -> list[str]`
  - `parse_range(value: str, size: int) -> tuple[int, int] | None` — `None` means "serve the whole thing"

- [ ] **Step 1: Write the failing test** — `tests/test_recording.py`:

```python
"""Turning a live run into a watchable file: pure parts first."""

from __future__ import annotations

import unittest
from pathlib import Path

from clank import recording
from clank.recording import composite_argv, parse_range


class CompositeArgvTests(unittest.TestCase):
    def argv(self, **kw):
        base = dict(game=Path("/r/game.mp4"), robot=Path("/r/robot.mp4"),
                    audio=Path("/r/audio.m4a"), out=Path("/r/recording.mp4"),
                    fps=15, crf=28, preset="veryfast")
        base.update(kw)
        return composite_argv(**base)

    def test_both_pictures_are_stacked_at_a_fixed_size(self):
        argv = self.argv()
        graph = argv[argv.index("-filter_complex") + 1]
        self.assertIn("hstack", graph)
        self.assertIn("scale=960:540", graph)
        self.assertIn("scale=720:540", graph)
        self.assertEqual(argv[argv.index("-movflags") + 1], "+faststart")

    def test_audio_is_muxed_and_the_shorter_stream_wins(self):
        argv = self.argv()
        self.assertIn("-c:a", argv)
        self.assertIn("aac", argv)
        self.assertIn("-shortest", argv)

    def test_a_missing_source_is_black_padding(self):
        argv = self.argv(robot=None, audio=None)
        graph = argv[argv.index("-filter_complex") + 1]
        self.assertIn("color=black", graph)
        self.assertNotIn("-c:a", argv)

    def test_the_encoder_is_named_in_the_argv(self):
        self.assertEqual(self.argv(ffmpeg="/opt/ff")[0], "/opt/ff")


class ParseRangeTests(unittest.TestCase):
    def test_the_three_shapes_a_video_player_sends(self):
        self.assertEqual(parse_range("bytes=0-1", 1000), (0, 1))
        self.assertEqual(parse_range("bytes=100-", 1000), (100, 999))
        self.assertEqual(parse_range("bytes=-500", 1000), (500, 999))

    def test_a_plain_request_is_not_a_range(self):
        self.assertIsNone(parse_range("", 1000))

    def test_multi_range_is_served_whole_rather_than_guessed(self):
        self.assertIsNone(parse_range("bytes=0-1,5-6", 1000))

    def test_unsatisfiable_and_malformed_are_refused(self):
        with self.assertRaises(ValueError):
            parse_range("bytes=5000-", 1000)
        with self.assertRaises(ValueError):
            parse_range("bytes=abc", 1000)
```

- [ ] **Step 2: Run it, expect failure** — `python3 -m unittest tests.test_recording -v`; expect `ModuleNotFoundError: No module named 'clank.recording'`.

- [ ] **Step 3: Implement** — `composite_argv` returns
  `[ffmpeg, "-y", "-i", str(game), "-i", str(robot), "-i", str(audio), "-filter_complex", graph, "-map", "[v]", ("-map", "2:a"), "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", "-r", str(fps), "-c:a", "aac", "-b:a", "96k", "-shortest", "-movflags", "+faststart", str(out)]`
  with `graph` = `[0:v]scale=960:540[g];[1:v]scale=720:540[r];[g][r]hstack=inputs=2[v]` for two sources, and for a missing one a `color=black:s=960x540:d=1[g]` (or the robot side `color=black:s=720x540:d=1[r]`) plus `loop`/`shortest` so the pad matches duration; omit the audio `-map`/`-c:a`/`-b:a` when `audio is None`. `parse_range` returns inclusive `(start, end)`, `None` for no header or a comma, and raises `ValueError` on a malformed or unsatisfiable range (start >= size, or an unparsable number).

- [ ] **Step 4: Run it, expect pass** — same command; expect `OK`.

- [ ] **Step 5: Commit** — `git add clank/recording.py tests/test_recording.py && git commit -m "compose the two pictures as a pure argv and parse a byte range"`

---

### Task 3: `Recorder` — three encoders, one manifest

**Files:**
- Modify: `clank/recording.py`
- Test: `tests/test_recording.py`

**Interfaces:**
- Consumes: `composite_argv` (Task 2), `Config.recording` (Task 1), `clank.depth_stream.Stream`, `clank.ears.capture`.
- Produces: `Recorder(cfg: Config, run_dir: Path, *, stream_factory=None, capture=None, spawn=None, clock=time.monotonic)` with `start() -> None`, `pump(seconds: float) -> None` (the tick: the threads' body, and what a test drives directly), `stop() -> dict` (the manifest it wrote), `manifest() -> Path`, and the module function `read_manifest(run_dir: Path) -> dict | None`. The manifest carries `parts: {"game": "parts/game.mp4", "robot": ..., "audio": ...}` (only the sources that started) — Task 4 composites from it and never guesses a filename. Each `sources` value is one of `ok`, `missing`, `failed`.

- [ ] **Step 1: Write the failing test** — add to `tests/test_recording.py` a harness with a **fake encoder**: a shell script `fake-ffmpeg` that appends its stdin to the last path in its argv, standing in for both the pipes and the composite:

```python
FAKE_FFMPEG = """#!/bin/sh
# last argument is the output file; copy stdin there so the test can see the bytes
for a in "$@"; do out="$a"; done
cat > "$out"
echo "$@" > "$out.argv"
"""


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.run_dir = self.base / "runs" / "101010-a1b2c3"
        self.run_dir.mkdir(parents=True)
        self.ffmpeg = self.base / "fake-ffmpeg"
        self.ffmpeg.write_text(FAKE_FFMPEG)
        self.ffmpeg.chmod(0o755)
        cfg = load_config(None, env={})
        self.cfg = replace(cfg, recording=replace(cfg.recording, ffmpeg=str(self.ffmpeg)))

    def test_start_records_and_stop_composes_a_manifest(self):
        rec = Recorder(self.cfg, self.run_dir, stream_factory=FakeStream, capture=FakeCapture)
        rec.start()
        self.assertEqual(read_manifest(self.run_dir)["state"], "recording")
        rec.pump(0.4)
        manifest = rec.stop()
        self.assertEqual(manifest["state"], "composing")
        self.assertEqual(manifest["width"], 1680)
        self.assertEqual(manifest["sources"], {"game": "ok", "robot": "ok", "audio": "ok"})
        self.assertEqual(manifest["audio_peak"], 9000)

    def test_a_missing_encoder_is_a_manifest_not_a_crash(self):
        cfg = replace(self.cfg, recording=replace(self.cfg.recording, ffmpeg=str(self.base / "nope")))
        rec = Recorder(cfg, self.run_dir)
        rec.start()  # must not raise
        manifest = rec.stop()
        self.assertEqual(manifest["state"], "failed")
        self.assertIn("ffmpeg", manifest["error"])

    def test_no_depth_stream_records_the_robot_side_only(self):
        rec = Recorder(self.cfg, self.run_dir, stream_factory=None, capture=FakeCapture)
        rec.start()
        manifest = rec.stop()
        self.assertEqual(manifest["sources"]["game"], "missing")
        self.assertEqual(manifest["state"], "composing")
```

with `FakeStream` yielding dicts shaped like a real frame (`{"color": bytes(960*540*4), "seq": n}`) and `FakeCapture` a context manager yielding PCM blocks of a known peak. `pump(seconds)` is the documented seam: the threads call it in a loop, and a test drives it directly so no sleep-based race enters the suite.

- [ ] **Step 2: Run it, expect failure** — `python3 -m unittest tests.test_recording.RecorderTests -v`; expect `ImportError: cannot import name 'Recorder'`.

- [ ] **Step 3: Implement** — `Recorder.start()`: probe ffmpeg (`-version`, keep the first line for the manifest), then spawn the game encoder (`-f rawvideo -pix_fmt rgba -s 960x540 -r fps -i -` + `-vf vflip`), the robot encoder (`-f image2pipe -vcodec mjpeg -framerate robot_fps -i -` + `-vf scale=720:540`), and the audio encoder (`-f s16le -ar 16000 -ac 1 -i -` → `audio.m4a`, fed from `ears.capture()`), each writing `<run_dir>/parts/*.mp4` with `start_new_session=True`; write the manifest with `state: "recording"`, `started_at` from a wall clock taken once, and each source's availability. Threads: one pulling `Stream.read(flip=False)` and **repeating the last frame** when the clock says a tick is due with no new frame, one copying the newest robot JPEG when its mtime moves, one draining `capture()` into the audio pipe and tracking peak. `stop()`: close stdin, `wait(timeout=10)`, SIGKILL on timeout, sum the parts' bytes, write `state: "composing"` and return it. Every OSError/subprocess failure sets `state: "failed"` with a one-line reason and returns; nothing raises into the caller.

- [ ] **Step 4: Run it, expect pass** — same command; expect `OK`.

- [ ] **Step 5: Commit** — `git add clank/recording.py tests/test_recording.py && git commit -m "record the game, the robot and the game's voice while a run happens"`

---

### Task 4: The detached composite, the stale sweep and pruning

**Files:**
- Modify: `clank/recording.py`
- Modify: `clank/dashboard.py` (call the sweep at startup)
- Test: `tests/test_recording.py`

**Interfaces:**
- Consumes: `read_manifest` (Task 3) for `parts`, `sources`, `started_at`, `fps`; `composite_argv` (Task 2).
- Produces: `run_composite(run_dir: Path, cfg: Config) -> dict` (synchronous; the caller detaches it), `sweep_stale(sessions_root: Path) -> list[Path]`, `prune(sessions_root: Path, max_total_mb: int, keep: Path | None = None) -> list[Path]`, `list_recordings(sessions_root: Path) -> list[dict]`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_recording.py`:

```python
class CompositeTests(unittest.TestCase):
    def test_the_composite_turns_composing_into_ready(self):
        ...  # a run dir with parts/ and a manifest in "composing"
        manifest = run_composite(self.run_dir, self.cfg)
        self.assertEqual(manifest["state"], "ready")
        self.assertEqual(manifest["file"], "recording.mp4")
        self.assertTrue((self.run_dir / "recording.mp4").is_file())
        self.assertFalse((self.run_dir / "parts").exists())  # keep_parts false

    def test_a_failed_composite_keeps_the_parts_and_says_why(self):
        cfg = replace(self.cfg, recording=replace(self.cfg.recording, ffmpeg=str(self.base / "nope")))
        manifest = run_composite(self.run_dir, cfg)
        self.assertEqual(manifest["state"], "failed")
        self.assertTrue((self.run_dir / "parts" / "game.mp4").is_file())


class SweepAndPruneTests(unittest.TestCase):
    def test_a_run_whose_dashboard_died_is_marked_and_its_parts_are_deleted(self):
        ...  # manifest in "recording", parts on disk
        removed = sweep_stale(self.sessions)
        self.assertEqual(len(removed), 1)
        self.assertEqual(read_manifest(self.run_dir)["state"], "failed")
        self.assertIn("dashboard exited", read_manifest(self.run_dir)["error"])

    def test_pruning_takes_the_oldest_and_never_the_run_just_recorded(self):
        ...  # three ready runs of 1 MB with max_total_mb = 2
        removed = prune(self.sessions, 2, keep=self.newest)
        self.assertEqual([p.parent.parent.name for p in removed], ["<oldest run id>"])
        self.assertTrue((self.newest / "recording.mp4").is_file())

    def test_the_listing_is_newest_first_and_says_what_is_missing(self):
        rows = list_recordings(self.sessions)
        self.assertEqual(rows[0]["run_id"], "<newest>")
        self.assertEqual(rows[-1]["state"], "failed")  # the swept one
```

- [ ] **Step 2: Run it, expect failure** — `python3 -m unittest tests.test_recording.CompositeTests tests.test_recording.SweepAndPruneTests -v`; expect `AttributeError: module 'clank.recording' has no attribute 'run_composite'`.

- [ ] **Step 3: Implement** — `run_composite` builds `composite_argv` from the manifest, runs ffmpeg with a 30-minute timeout, and rewrites the manifest: `ready` with `bytes`, `width`, `height`, `fps`, `duration_s` from the parts' probe (or the recorded frame count / fps), else `failed` with the last stderr line. `sweep_stale` walks `sessions/*/runs/*`, and for any manifest whose `state` is `recording`/`composing` (a dashboard died) or any `parts/` with no manifest: delete the parts and mark `failed: dashboard exited`. `prune` sums `bytes` of every `ready` recording, deletes oldest-first until under the cap, never touching `keep`, and appends one line per deletion to `<session>/session.log`. `list_recordings` returns one dict per run with a manifest: `run_id`, `session_id`, `dir`, `state`, `started_at`, `ended_at`, `duration_s`, `bytes`, `width`, `height`, `fps`, `audio_peak`, `has_diary`, sorted by `started_at` descending. In `dashboard.py`'s `run_dashboard`, call `sweep_stale(cfg.paths.session_root)` once before serving and log the count.

- [ ] **Step 4: Run it, expect pass** — same command; expect `OK`.

- [ ] **Step 5: Commit** — `git add clank/recording.py clank/dashboard.py tests/test_recording.py && git commit -m "finish the recording after the run, and clean up after a dashboard that died"`

---

### Task 5: `RunManager` owns a recorder

**Files:**
- Modify: `clank/run.py` (`__init__`, `_worker`, `_finish`, `snapshot`)
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: `Recorder`, `run_composite` (Tasks 3–4).
- Produces: `RunManager(..., recorder=None)`; `snapshot()["recording"]` is the manifest dict or `None`.

- [ ] **Step 1: Write the failing test** — add to `tests/test_run.py`:

```python
class FakeRecorder:
    def __init__(self):
        self.started = 0
        self.stopped = 0
        self.manifest = {"state": "ready", "file": "recording.mp4"}

    def start(self) -> None:
        self.started += 1

    def stop(self) -> dict:
        self.stopped += 1
        return self.manifest


class RecorderWiringTests(RunHarness):
    def test_the_recorder_starts_with_the_brain_and_stops_with_the_run(self):
        recorder = FakeRecorder()
        run = self.make(recorder=recorder)
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        self.assertEqual(recorder.started, 1)
        self.child.alive_flag = False
        self.assertTrue(run.wait_idle(5))
        self.assertEqual(recorder.stopped, 1)

    def test_a_run_refused_at_the_gate_never_starts_a_recorder(self):
        recorder = FakeRecorder()
        run = self.make(recorder=recorder, probe=lambda: brain.ServerStatus(up=False, status="down", model=""))
        run.start()
        self.assertTrue(run.wait_idle(5))
        self.assertEqual(run.snapshot()["state"], "failed")
        self.assertEqual(recorder.started, 0)

    def test_the_composite_does_not_hold_the_run_open(self):
        """stop() must return before the composite: wait_idle is 15 s elsewhere."""
        recorder = SlowCompositeRecorder()
        run = self.make(recorder=recorder)
        run.start()
        self.assertEqual(self.wait_state("running"), "running")
        self.child.alive_flag = False
        self.assertTrue(run.wait_idle(5))
        self.assertTrue(recorder.composite_started)  # detached, not awaited
        self.assertLess(recorder.stop_seconds, 0.5)
```

- [ ] **Step 2: Run it, expect failure** — `python3 -m unittest tests.test_run.RecorderWiringTests -v`; expect `TypeError: make() got an unexpected keyword argument 'recorder'`.

- [ ] **Step 3: Implement** — `__init__` accepts `recorder=None` (meaning: build a real `Recorder` lazily at `running`, only when `cfg.recording.enabled`). In `_worker`, immediately after `self._set_phase(RunState.RUNNING)` and before `_monitor`: `self._start_recording(run_dir)` which wraps `start()` in `try/except Exception` and records a detail on failure. In `_finish`, before the state flip: `self._stop_recording()` → `recorder.stop()` (bounded, no composite), then `threading.Thread(target=run_composite, args=(run_dir, self._cfg), daemon=True).start()`. `snapshot()` gains `"recording": read_manifest(self._dir) if self._dir else None`.

- [ ] **Step 4: Run it, expect pass** — same command, then the whole file: `python3 -m unittest tests.test_run`; expect `OK`.

- [ ] **Step 5: Commit** — `git add clank/run.py tests/test_run.py && git commit -m "let a run record itself"`

---

### Task 6: The dashboard routes

**Files:**
- Modify: `clank/dashboard.py` (`do_GET`)
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `list_recordings`, `parse_range` (Tasks 2, 4).
- Produces: three routes — `GET /api/recordings` → `{"recordings": [...], "recording_available": bool}`; `GET /api/recordings/<run_id>/video` → the file, 206 + `Content-Range` for a usable range, 416 for an unsatisfiable one, 200 otherwise, 404 for an unknown id or a run that is not `ready`; `GET /recordings` → the page's HTML.

- [ ] **Step 1: Write the failing test** — add to `tests/test_dashboard.py`:

```python
class RecordingRouteTests(DashboardHarness):
    def test_the_listing_is_newest_first(self):
        status, raw = self.request("GET", "/api/recordings")
        self.assertEqual(status, 200)
        body = json.loads(raw)
        self.assertEqual([r["run_id"] for r in body["recordings"]], ["110000-bbbbbb", "100000-aaaaaa"])

    def test_a_range_request_gets_a_206_slice(self):
        status, headers, body = self.request_with_headers(
            "/api/recordings/100000-aaaaaa/video", headers={"Range": "bytes=0-1"})
        self.assertEqual(status, 206)
        self.assertEqual(headers["Content-Range"], f"bytes 0-1/{self.size}")
        self.assertEqual(len(body), 2)

    def test_a_suffix_range_and_a_multi_range(self):
        status, headers, _ = self.request_with_headers(
            "/api/recordings/100000-aaaaaa/video", headers={"Range": "bytes=-500"})
        self.assertEqual(status, 206)
        self.assertEqual(headers["Content-Range"], f"bytes {self.size-500}-{self.size-1}/{self.size}")
        # A multi-range is served whole rather than guessed at.
        status, _, body = self.request_with_headers(
            "/api/recordings/100000-aaaaaa/video", headers={"Range": "bytes=0-1,5-6"})
        self.assertEqual(status, 200)
        self.assertEqual(len(body), self.size)

    def test_an_impossible_range_is_a_416(self):
        status, _, _ = self.get_raw("/api/recordings/100000-aaaaaa/video", headers={"Range": "bytes=999999-"})
        self.assertEqual(status, 416)
        self.assertEqual(headers.get("Content-Range"), f"bytes */{self.size}")

    def test_an_unknown_or_unfinished_run_is_a_404(self):
        self.assertEqual(self.request("GET", "/api/recordings/nope/video")[0], 404)
        self.assertEqual(self.request("GET", "/api/recordings/110000-bbbbbb/video")[0], 404)  # composing

    def test_the_page_is_served(self):
        status, body = self.get_raw("/recordings")
        self.assertEqual(status, 200)
        self.assertIn(b"recordings", body.lower())
```

- [ ] **Step 2: Run it, expect failure** — `python3 -m unittest tests.test_dashboard.RecordingRouteTests -v`; expect 404s where 200s belong.

- [ ] **Step 3: Extend the harness, then implement** — `DashboardHarness.request_with_headers(path, headers=None) -> (status, headers, body)` (today it takes only a path); it is what the Range tests read. Then add the two API branches and the page branch to `do_GET`, reusing `RUN_ID_RE` for the id so a traversing id is a 404 like `/api/brain/messages`. The video branch resolves the run's `recording.json` through `list_recordings` (never by joining the URL onto a path), requires `state == "ready"`, opens the file, applies `parse_range`, and writes `206` + `Content-Range` + `Content-Length` (and `Accept-Ranges: bytes`) or `416` + `Content-Range: bytes */size`. Keep HTTP/1.0. `/recordings` serves `static/recordings.html` through the existing static path resolution.

- [ ] **Step 4: Run it, expect pass** — same command; expect `OK`.

- [ ] **Step 5: Commit** — `git add clank/dashboard.py tests/test_dashboard.py && git commit -m "serve recordings, in slices a video player can seek"`

---

### Task 7: The page

**Files:**
- Create: `clank/static/recordings.html`, `clank/static/recordings.js`
- Modify: `clank/static/index.html` (a "Recordings" link), `clank/static/style.css`
- Test: `tests/test_dashboard.py` (`ShippedStaticAssetsTests`)

**Interfaces:**
- Consumes: `/api/recordings`, `/api/recordings/<id>/video`, `/api/brain/messages?run=<id>` (Tasks 6, and the existing diary route).
- Produces: shipped assets only.

- [ ] **Step 1: Write the failing test** — extend `ShippedStaticAssetsTests`:

```python
    def test_the_page_is_served(self):
        status, _, body = self.request_with_headers("/recordings")
        self.assertEqual(status, 200)
        self.assertIn(b"recording-list", body)

    def test_the_recordings_page_ships_list_video_and_diary(self):
        html = (STATIC / "recordings.html").read_text()
        js = (STATIC / "recordings.js").read_text()
        self.assertIn('id="recording-list"', html)
        self.assertIn('id="recording-video"', html)
        self.assertIn('id="run-diary"', html)
        self.assertIn('controls', html)
        self.assertIn("/api/recordings", js)
        self.assertIn("Math.min", js)       # a seek is clamped to the file, both ways
        self.assertIn("Math.max", js)
        self.assertIn("currentTime", js)
        self.assertIn("silent", js)         # audio_peak 0 is said out loud
        self.assertIn("recording_available", js)  # "recording off" is not an empty list
```

- [ ] **Step 2: Run it, expect failure** — `python3 -m unittest tests.test_dashboard.ShippedStaticAssetsTests -v`.

- [ ] **Step 3: Implement** — `recordings.html`: one left column listing `recording-list` rows (run id, started, duration, MB, a `silent track` chip when `audio_peak == 0`, `composing…`/`failed` otherwise), a `<video id="recording-video" controls preload="metadata">`, and a `run-diary` column. `recordings.js`: fetch the listing, render rows, play the selected one from `/api/recordings/<id>/video`, fetch `/api/brain/messages?run=<id>`, and seek on a diary row click with `video.currentTime = Math.min(Math.max(0, eventTime - startedAt), video.duration || 0)`. Reuse `style.css` (grid). When `recording_available` is false, replace the list with one line saying recording is off on this machine. Add the link to `index.html`'s topbar beside the other controls **and** a "watch this run" link in the Brain panel that opens `/recordings?run=<id>`.

- [ ] **Step 4: Run it, expect pass** — same command; then a headless screenshot:
  `chromium --headless=new --no-sandbox --disable-gpu --user-data-dir=/tmp/clank-chromium --window-size=1600,1000 --virtual-time-budget=4000 --screenshot=/tmp/clank-recordings.png http://127.0.0.1:8765/recordings`
  Read the PNG: the list, the player and the diary must all be visible with no overlap.

- [ ] **Step 5: Commit** — `git add clank/static tests/test_dashboard.py && git commit -m "a page to watch what a run did"`

---

### Task 8: Write down how a recording works

**Files:**
- Modify: `AGENTS.md` (the "Stack and runs" section + a short "Recordings" subsection), `README.md` (the controls/UI paragraph)

- [ ] **Step 1: Document** — in `AGENTS.md` after the run section: the four states of `recording.json`; that the composite is detached and why (`wait_idle`, next Start run); the fixed 1680×540 geometry; that `started_at` is the recorder's, not the run's; the 20 GB cap and its pruning rule; the sweep for a dead dashboard; the traps (no ffmpeg → recorded as failed; no depth stream → robot-only; muted game → `audio_peak: 0`); the page's URL. In `README.md`, add the recordings page to the UI description.

- [ ] **Step 2: Verify docs render and the suite is green** — the full test command, then:
  `nix build .#checks.x86_64-linux.default` → exit 0.

- [ ] **Step 3: Commit** — `git add AGENTS.md README.md && git commit -m "write down what a recording is and where it lives"`

---

## Live verification (after the plan, with the stack up)

1. `./run.sh --replace`, `POST /api/session/start`, wait for `running`.
2. `POST /api/run/start`; while the run is live check `GET /api/state` → `run.recording.state == "recording"`, and `~/…/sessions/<id>/runs/<run>/parts/` growing.
3. `POST /api/run/stop`; then within a minute `GET /api/recordings` → that run reaches `ready` with `bytes > 0` and `audio_peak > 0` (JobBot talks; a peak of 0 means the game is muted — check `module-stream-restore`).
4. `ffprobe` the file: 1680×540, 15 fps, one video stream and one AAC stream, duration within a frame of `duration_s`.
5. Open `/recordings`, play it, click a diary row and confirm the picture matches the event.
6. With the model server stopped: `POST /api/run/start` → `failed` at the gate, and `/api/recordings` gains no new row.
