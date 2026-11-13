# One button, one panel, one picture — implementation plan

> **Archived 2026-11-13.** Implementation plan, carried out. Kept for the task breakdown and the gates it defined; the code is the record now.
> Current behaviour: [`README.md`](../../../README.md) · agent rules: [`AGENTS.md`](../../../AGENTS.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a run start with one press, move every game/body control into one panel, and turn the recordings page into a video-first presentation with a portrait robot tile.

**Architecture:** The run state machine learns to boot the stack itself (`RunState.STACK`), so `POST /api/run/start` is the only start the page offers. The console's chrome shrinks to status, a `Console|Recordings` tab pair, and the run buttons; everything that steers the body or the game moves to a Controls panel in the rail. The recordings page gives the video the full width with a wide synced diary beneath it. The composite becomes 1264×540 by rendering the body view at 405×720 instead of 560×420.

**Tech Stack:** Python 3.14 stdlib (`http.server`, `threading`), vanilla JS, CSS grid, ffmpeg for the composite, MuJoCo body view (already present).

**Spec:** `docs/superpowers/specs/2026-11-13-console-and-recordings-ui-design.md`

## Global Constraints

- `clank/` is standard-library only. No numpy, torch, PIL, requests, pytest.
- Tests are `unittest`: `python3 -m unittest discover -s tests -t .`
- Config keys live in four places: `config.example.toml`, `DEFAULTS`, the section dataclass, and `_validate`. Unknown `CLANK_*` env overrides stay rejected.
- No new `CLANK_*` env vars.
- No hardcoded `/nix/store`, compatdata id, or save paths.
- Commit messages are lowercase prose, no conventional-commit prefixes.
- A recording never fails, delays, or blocks a run. A refusal costs nothing.
- `run.recording` in `/api/state` is the only recording source the page uses.
- Existing 1680×540 recordings must keep playing.

## Review Focus

- **A run cancelled during `stack`** must not leave a half-booted stack and must not shut down one that was already up. Test in Task 1.
- **A run started when the stack is already up** must not bounce it (that is the whole reason Stop run exists). Test in Task 1.
- **The stack failing mid-boot** (monado refuses, Steam dies) must be a failed run with the orchestrator's reason, not a stuck spinner. Test in Task 1.
- **A page opened while the stack is down** must still offer Start run, and pressing it must be the thing that boots the stack. Test in Task 2.
- **A run whose video failed or was pruned** must say so where the saved indicator goes, not silently show nothing. Static assertion in Task 4.

---

### Task 1: A run boots the stack

**Files:**
- Modify: `clank/config.py` (`RunConfig` ~line 121, `DEFAULTS["run"]` ~line 267, `_validate` ~line 380)
- Modify: `config.example.toml` (`[run]`)
- Modify: `clank/run.py` (`RunState` ~line 57, `PHASE_DETAILS` ~line 46, `start()` ~line 246, `_worker` ~line 323, `_check_cancel` ~line 356)
- Test: `tests/test_run.py`, `tests/test_config.py`
- Modify: `AGENTS.md` (`## Stack and runs` — the flow paragraph)

**Interfaces:**
- Consumes: `SessionManager.snapshot() -> dict` (keys `state`, `session_id`), `SessionManager.start() -> None` (non-blocking; raises `ConflictError` unless stopped/failed).
- Produces: `RunState.STACK = "stack"`; `PHASE_DETAILS["stack"] = "starting the stack…"`; `RunConfig.session_start_sec: float` (default `180.0`); `RunManager._check_cancelled(generation: int) -> None` (raises `_Cancelled`; does **not** consult the session); `RunManager._stack(generation: int) -> None`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_run.py`, using the existing `RunManager` fixture and its fake manager (the fake exposes `snapshot()` and records `start()` calls):

```python
def test_a_run_starts_the_stack_when_it_is_down(self):
    self.manager.state = "stopped"
    self.run.start()
    self.assertTrue(self.run.wait_idle(20))
    self.assertEqual(self.manager.started, 1)          # the stack was booted
    self.assertIn("stack", self.phases_seen())         # and it said so
    self.assertEqual(self.snapshot()["state"], "running")

def test_a_run_does_not_touch_a_running_stack(self):
    self.manager.state = "running"
    self.run.start()
    self.assertTrue(self.run.wait_idle(20))
    self.assertEqual(self.manager.started, 0)
    self.assertNotIn("stack", self.phases_seen())

def test_a_stack_that_fails_to_start_fails_the_run_with_its_reason(self):
    self.manager.state, self.manager.start_result = "stopped", "failed"
    self.run.start()
    self.assertTrue(self.run.wait_idle(20))
    info = self.snapshot()
    self.assertEqual(info["state"], "failed")
    self.assertIn("stack", info["error"].lower())

def test_a_run_cancelled_while_the_stack_starts_leaves_it_alone(self):
    self.manager.state, self.manager.start_result = "stopped", "never"
    self.run.start()
    self.run.stop()                                    # cancel during `stack`
    self.assertTrue(self.run.wait_idle(20))
    self.assertEqual(self.manager.stopped, 0)          # nothing was shut down
    self.assertEqual(self.snapshot()["state"], "idle")
```

In `tests/test_config.py`:

```python
def test_session_start_sec_must_be_positive(self):
    with self.assertRaises(ConfigError):
        load_config(None, env={"CLANK_RUN_SESSION_START_SEC": "0"})
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python3 -m unittest tests.test_run tests.test_config -v`
Expected: `AttributeError: 'RunState' has no attribute 'STACK'` / `ConfigError` not raised.

- [ ] **Step 3: Add the config key**

`RunConfig.session_start_sec: float` after `game_restart_timeout_sec`; `DEFAULTS["run"]["session_start_sec"] = 180.0` with the one-line reason "a cold boot is monado, Weston, Steam and a first frame of the game"; `_validate` refuses `<= 0`; `config.example.toml` gains the key under `[run]`.

- [ ] **Step 4: Give the state machine the phase**

Add `STACK = "stack"` to `RunState` (before `GAME`) and `"stack": "starting the stack…"` to `PHASE_DETAILS`.

Split the cancel check, keeping the old meaning for the phases after it:

```python
def _check_cancelled(self, generation: int) -> None:
    """Cancel and generation only: a stack that is starting is not a reason."""

def _check_cancel(self, generation: int) -> None:
    self._check_cancelled(generation)
    if self._manager.snapshot().get("state") != "running":
        raise _Cancelled()
```

- [ ] **Step 5: Implement `_stack` and call it first in `_worker`**

`_worker` becomes `self._stack(generation)` then `_check_cancel(generation)` before the existing `RunState.PREPARING` line.

`_stack` reads `self._manager.snapshot()["state"]`: if `"running"`, return without touching it; otherwise `_set_phase(RunState.STACK)`, call `self._manager.start()` inside `contextlib.suppress(ConflictError)`, then poll `snapshot()["state"]` every `0.2` s until `"running"` (return), `"failed"`/`"stopped"` (raise `RunError(f"the stack did not start: {snapshot.get('last_error') or state}")`), the deadline `time.monotonic() + cfg.run.session_start_sec` passes (`raise RunError("the stack did not come up in N s")`), or `_check_cancelled(generation)` raises `_Cancelled`.

Delete the `session is not running` guard from `start()` (the `info.get("state") != "running"` check at ~line 252) — keep `session_id` for the run directory, taken *after* `_stack` returns, which means the run's directory is created later than today. Move `_ensure_run_dir(run_dir)` accordingly: `run_dir` is computed in `_stack`'s success path, and `self._dir`/`self._session_id` are set there under the lock.

- [ ] **Step 6: Run the tests**

Run: `python3 -m unittest tests.test_run tests.test_config -v`
Expected: PASS, plus the existing `test_run` suite still green.

- [ ] **Step 7: Update `AGENTS.md`**

In `## Stack and runs, 2026-11-12`, replace the "Start boots the stack / Start run is the thing you repeat" paragraph with: one press of Start run boots the stack when it is down and never touches it when it is up; Stop run leaves the stack up; Shut down is the old Stop.

- [ ] **Step 8: Commit**

```bash
git add clank/run.py clank/config.py config.example.toml tests/test_run.py tests/test_config.py AGENTS.md
git commit -m "let a run start the stack it needs"
```

---

### Task 2: One start button, tabs, a demoted shutdown

**Files:**
- Modify: `clank/static/index.html` (topbar ~line 46-90)
- Modify: `clank/static/recordings.html` (topbar ~line 20-45)
- Modify: `clank/static/app.js` (`renderActions` ~line 736, handlers ~line 1070)
- Modify: `clank/static/recordings.js` (theme handler only, if it looks for `#recordings-link`)
- Modify: `clank/static/style.css` (`.top-actions`, new `.page-tabs`)
- Test: `tests/test_dashboard.py` (`ShippedStaticAssetsTests`)

**Interfaces:**
- Consumes: `run.start()` now boots the stack (Task 1), so the page needs no `Start`.
- Produces: markup contract other tasks and tests rely on — `#start-run`, `#stop-run`, `#stop` (inside `#shutdown-menu`), `.page-tabs` with `a[data-page="console"]` and `a[data-page="recordings"]`, `aria-current="page"` on the active tab.

- [ ] **Step 1: Write the failing test**

In `tests/test_dashboard.py` (`ShippedStaticAssetsTests`):

```python
def test_the_console_offers_one_start_and_a_tab_pair(self):
    _, body = self.request("/")
    page = body.decode()
    self.assertIn('id="start-run"', page)
    self.assertNotIn('id="start"', page)          # no second start
    self.assertIn('data-page="recordings"', page)
    self.assertIn('data-page="console"', page)
    self.assertIn('aria-current="page"', page)

def test_the_recordings_page_carries_the_same_tabs(self):
    _, body = self.request("/recordings")
    page = body.decode()
    self.assertIn('data-page="console"', page)
    self.assertIn('data-page="recordings"', page)
```

- [ ] **Step 2: Run it, watch it fail**

Run: `python3 -m unittest tests.test_dashboard.ShippedStaticAssetsTests -v`
Expected: FAIL on `assertNotIn('id="start"')` first.

- [ ] **Step 3: Rewrite the console topbar**

Order: brand · status cluster (`#session-id`, `#run-state`, spinner, `#uptime`) · `.page-tabs` · `#start-run` · `#stop-run` · `#shutdown-menu`. `#shutdown-menu` is `<details>`/`<summary>▾</summary>` holding `#stop` plus the theme toggle, so the bar ends in one control. Mute and `#menu-press` leave this bar entirely (Task 3 owns where they land) — for this task, delete them from the bar and leave their handlers intact.

- [ ] **Step 4: Point `app.js` at the new elements**

`renderActions` becomes: `#start-run` disabled only while `actionInFlight`, a run is active, or the model is down (it no longer requires `running`); `#stop-run` disabled while no run is active; `#stop` (in the menu) disabled while `actionInFlight`. Handlers for `#start`/`#recordings-link` are deleted; any `$("start")`-style lookups that remain must go, since a missing element throws on `.onclick`.

- [ ] **Step 5: Give the recordings page the same tabs**

Replace its `Console` link with the identical `.page-tabs` pair, `aria-current="page"` on Recordings.

- [ ] **Step 6: Style the tab pair**

`.page-tabs` is a segmented control: two anchors sharing one rounded border, the active one filled, `:focus-visible` ring preserved, 32px tall so it matches `.btn`. `#shutdown-menu > summary` looks like `.btn` with a chevron.

- [ ] **Step 7: Run the tests**

Run: `python3 -m unittest tests.test_dashboard -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add clank/static/index.html clank/static/recordings.html clank/static/app.js clank/static/recordings.js clank/static/style.css tests/test_dashboard.py
git commit -m "give the console one start and a way to cross between pages"
```

---

### Task 3: A Controls panel

**Files:**
- Modify: `clank/static/index.html` (rail ~line 160-190; remove `.body-controls` from the Robot panel)
- Modify: `clank/static/app.js` (`renderActions` ~line 780-790, verb/mode/body handlers ~line 1080-1130)
- Modify: `clank/static/style.css` (`.panel-controls`, `.body-controls` moves)
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `/api/menu`, `/api/body/command` (`look_left`/`look_right`), `/api/session/mode`, `/api/body/start`, `/api/body/stop`, `/api/audio/mute` — all unchanged.
- Produces: `#controls-panel` containing `#menu-press`, `#mute-audio`, `[data-verb]` look pads, `#mode-toggle`, `#body-start`, `#body-stop`, `#body-state` (text, not a button).

- [ ] **Step 1: Write the failing test**

```python
def test_the_rail_holds_the_controls_panel(self):
    _, body = self.request("/")
    page = body.decode()
    self.assertIn('id="controls-panel"', page)
    panel = page.split('id="controls-panel"', 1)[1].split("</section>", 1)[0]
    for control in ("menu-press", "mute-audio", "mode-toggle", "body-start", "body-stop", "body-state"):
        self.assertIn(f'id="{control}"', panel, control)
    # The robot panel shows the views and nothing that steers them.
    robot = page.split('id="body-box"', 1)[1].split("</aside>", 1)[0]
    self.assertNotIn('data-verb=', robot)
```

- [ ] **Step 2: Run it, watch it fail**

Run: `python3 -m unittest tests.test_dashboard.ShippedStaticAssetsTests.test_the_rail_holds_the_controls_panel -v`

- [ ] **Step 3: Move the controls**

Cut `.body-controls` out of `#body-box` and paste it into a new `<section class="panel panel-controls" id="controls-panel">` in `.col-rail` between Brain and Logs, with `<h2>Controls</h2>` and a `<p id="body-state">` line. Rail order becomes Brain, Controls, Logs, Telemetry.

- [ ] **Step 4: Make the body buttons honest**

`#body-start`/`#body-stop` disable only on `actionInFlight` or when the body is not configured (`!body || !body.enabled`). `#body-state` reads `body.running ? "running" : "stopped"` plus `body.error` when present, so the panel says what is true instead of showing two grey buttons that look broken.

- [ ] **Step 5: Label the menu press for what it does**

`#menu-press` text "Menu press" with `title="Presses the controller's menu button in the game"`; `#body-state` and the look pads keep their existing handlers untouched.

- [ ] **Step 6: Run the dashboard tests**

Run: `python3 -m unittest tests.test_dashboard -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add clank/static/index.html clank/static/app.js clank/static/style.css tests/test_dashboard.py
git commit -m "put what steers the body in one panel"
```

---

### Task 4: Say what happened to the video

**Files:**
- Modify: `clank/static/index.html` (status cluster)
- Modify: `clank/static/app.js` (`renderState`/`renderActions`)
- Modify: `clank/static/style.css` (`.rec-status`, `.rec-status.is-ready`, `.is-bad`)
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `run.recording` from `/api/state` — `null` or a manifest dict with `state` (`recording`/`composing`/`ready`/`failed`/`pruned`), `error`, `notes`, `duration_s`, `bytes`; and the run id at `state.run.id`.
- Produces: `#run-recording` element and `recordingStatus(recording, runId) -> {text, href, tone}` in `app.js`, where `tone` is `"busy" | "ok" | "bad"`.

- [ ] **Step 1: Write the failing test**

```python
def test_the_topbar_can_say_what_happened_to_the_video(self):
    _, body = self.request("/")
    self.assertIn('id="run-recording"', body.decode())
```

- [ ] **Step 2: Run it, watch it fail**

Run: `python3 -m unittest tests.test_dashboard.ShippedStaticAssetsTests.test_the_topbar_can_say_what_happened_to_the_video -v`

- [ ] **Step 3: Add the element and the mapping**

`<a id="run-recording" class="rec-status" hidden></a>` in the status cluster. `recordingStatus` returns: hidden for `null`; `"saving video…"` (tone `busy`) for `recording`/`composing`; `"video saved · watch"` with `href = "/recordings?run=" + runId` (tone `ok`) for `ready`; `"no video: " + (error or notes)` (tone `bad`) for `failed`; `"video pruned to make room"` (tone `bad`) for `pruned`. It is rendered on every state poll, so the transition `saving → saved` needs no extra polling.

- [ ] **Step 4: Style it**

Hidden when empty; `busy` muted with a pulsing dot that stops under `prefers-reduced-motion`; `ok` green; `bad` amber. It sits after `#run-state` and never wraps the bar: `white-space: nowrap; overflow: hidden; text-overflow: ellipsis`.

- [ ] **Step 5: Run the tests**

Run: `python3 -m unittest tests.test_dashboard -v`
Expected: PASS.

- [ ] **Step 6: Verify by eye**

`chromium --headless=new --no-sandbox --disable-gpu --window-size=1600,1000 --virtual-time-budget=6000 --screenshot=/tmp/clank-indicator.png http://127.0.0.1:8765/` against a live dashboard, and read the element's `hidden`/text in the DOM dump rather than trusting the screenshot.

- [ ] **Step 7: Commit**

```bash
git add clank/static/index.html clank/static/app.js clank/static/style.css tests/test_dashboard.py
git commit -m "say when the video is saved"
```

---

### Task 5: A video-first recordings page

**Files:**
- Modify: `clank/static/recordings.html` (`main.rec-deck` ~line 47-92)
- Modify: `clank/static/recordings.js` (list rendering, reveal, seek)
- Modify: `clank/static/style.css` (`.rec-deck` ~line 1057-1140)
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `/api/recordings` list rows (`run_id`, `started_at`, `duration_s`, `state`, `bytes`), the manifest's `width`/`height`, and the existing synced-diary functions.
- Produces: page structure `#run-strip` (chips) · `#rec-stage` (video + transport) · `#run-diary` (wide log). `?run=`/`?t=` behaviour unchanged.

- [ ] **Step 1: Write the failing test**

```python
def test_the_recordings_page_puts_the_video_first(self):
    _, body = self.request("/recordings")
    page = body.decode()
    self.assertIn('id="run-strip"', page)
    # The video comes before the diary in document order: the picture is the page.
    self.assertLess(page.index('id="recording-video"'), page.index('id="run-diary"'))
    self.assertNotIn('class="panel rec-list-panel"', page)
```

- [ ] **Step 2: Run it, watch it fail**

Run: `python3 -m unittest tests.test_dashboard.ShippedStaticAssetsTests.test_the_recordings_page_puts_the_video_first -v`

- [ ] **Step 3: Restructure the markup**

`main.rec-deck` becomes a single column: `#run-strip` (a horizontal, horizontally-scrolling row of chips: time, duration, state badge), then `section.rec-stage` holding `#recording-title`, `.rec-player` (video + `.rec-controls` unchanged), `#recording-note`, `#recording-meta`, then `section.panel.rec-diary-panel` with `#run-diary`.

- [ ] **Step 4: Restyle**

`.rec-deck` is `grid-template-rows: auto 1fr` with `max-width: 1680px`; `.rec-player video` is `width: 100%; aspect-ratio: var(--rec-aspect, 1264 / 540); height: auto; background: #000`. `recordings.js` sets `--rec-aspect` from the manifest's `width`/`height` on load, so 1680×540 recordings stay correct. `.rec-diary-panel` shows rows in `columns: 2` at ≥1200px and 3 at ≥1600px, each row `time · kind · text`, scrolling with the transport as it does now.

- [ ] **Step 5: Keep the synced diary intact**

Only the container changes: `revealUpTo`, `REVEAL_LEAD_S`, `rec-current`, row-click seek, `pendingSeek`, and the composing poll stay byte-identical. If a selector referenced the old grid, prefer `#run-diary` over a structural selector.

- [ ] **Step 6: Run the tests**

Run: `python3 -m unittest tests.test_dashboard -v`
Expected: PASS.

- [ ] **Step 7: Look at it**

Take a headless screenshot of `/recordings?run=<a real run>` and confirm: video spans the width, no dead space under it, diary rows readable in columns, ticks still on the rail.

- [ ] **Step 8: Commit**

```bash
git add clank/static/recordings.html clank/static/recordings.js clank/static/style.css tests/test_dashboard.py
git commit -m "make the picture the page on recordings"
```

---

### Task 6: A portrait robot tile

**Files:**
- Modify: `clank/recording.py` (`GAME_WIDTH`/`WIDTH` constants ~line 40-60, `composite_argv`, `Recorder` geometry note)
- Modify: `clank/config.py` (`DEFAULTS["body"]` ~line 215, `_validate` body bounds)
- Modify: `config.example.toml` (`[body]`)
- Modify: `AGENTS.md` (`## Recordings` — geometry line)
- Test: `tests/test_recording.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: the body view's `--width`/`--height` and `body_latest.jpg`; `composite_argv(...)`.
- Produces: `ROBOT_WIDTH = 304`, `WIDTH = 1264` (unchanged names); `[body] width = 405`, `height = 720` (exactly 9:16).

- [ ] **Step 1: Write the failing tests**

In `tests/test_recording.py`:

```python
def test_the_composite_is_a_portrait_tile_beside_the_game(self):
    self.assertEqual(recording.WIDTH, 1264)
    self.assertEqual(recording.HEIGHT, 540)
    self.assertEqual(recording.ROBOT_WIDTH, 304)
    self.assertEqual(recording.GAME_WIDTH, 960)

def test_the_robot_part_scales_into_the_portrait_tile(self):
    argv = composite_argv(game=Path("g.mp4"), robot=Path("r.mp4"), audio=None,
                          out=Path("o.mp4"), fps=15, crf=28, preset="veryfast")
    graph = argv[argv.index("-filter_complex") + 1]
    self.assertIn("scale=304:540", graph)
    self.assertIn("scale=960:540", graph)
```

In `tests/test_config.py`:

```python
def test_the_default_body_render_is_portrait(self):
    cfg = load_config(None, env={})
    self.assertEqual((cfg.body.width, cfg.body.height), (405, 720))
    self.assertEqual(cfg.body.width * 16, cfg.body.height * 9)   # exactly 9:16
```

- [ ] **Step 2: Run them, watch them fail**

Run: `python3 -m unittest tests.test_recording tests.test_config -v`
Expected: FAIL on `1264 != 1680`.

- [ ] **Step 3: Change the geometry**

`ROBOT_WIDTH = 304`; `WIDTH = GAME_WIDTH + ROBOT_WIDTH` (1264); `HEIGHT` stays 540. `DEFAULTS["body"]["width"] = 405`, `"height"] = 720`. `config.example.toml` `[body]` gains the pair with the reason "portrait, so the recording's robot tile is 9:16 instead of a 4:3 letterbox".

- [ ] **Step 4: Update the comments that cited the old numbers**

Any docstring/comment in `clank/recording.py` naming 720×540 or 1680×540 becomes 304×540 and 1264×540; the manifest still records its own `width`/`height`, which is what keeps old recordings playable (Task 5 reads it).

- [ ] **Step 5: Run the suite**

Run: `python3 -m unittest discover -s tests -t .`
Expected: PASS. Tests that hardcoded 1680/720 were already reading the constants; fix any that pinned the literal.

- [ ] **Step 6: Verify the composite offline**

Build three parts with ffmpeg (game 960×540, robot 405×720, audio), run `run_composite` against a seeded manifest, and probe the result: `width=1264 height=540`, duration within 0.1 s of the manifest. This is the check the real-ffmpeg test already does; do it once by hand with the game off.

- [ ] **Step 7: Note the live measurement still owed**

In `AGENTS.md`, state that 405×720 is the render and 304×540 the tile, and that the follow camera's framing in 9:16 is **unverified live** until a real session is recorded — if the camera clips a raised arm, the fallback is the same portrait render padded, never a 4:3 crop.

- [ ] **Step 8: Commit**

```bash
git add clank/recording.py clank/config.py config.example.toml tests/test_recording.py tests/test_config.py AGENTS.md
git commit -m "render the body portrait so its tile is not a letterbox"
```

---

## Amendment, same day

Task 3 is withdrawn as a *panel*: the operator saw the Controls panel and asked for the
controls back in the Robot panel, mute back in the topbar, and no overflow menu. See the
spec's Amendment section. What shipped from Task 3 is the honest body state: buttons that
are off only when they cannot act, and a `#body-state` line that says `running` or
`stopped`. Task 2 shipped the tabs and the single Start run, and its `#shutdown-menu` was
replaced by a plain Shut down button in the bar.

## Self-review

**Spec coverage:** one button (Task 1 + 2), Controls panel (Task 3), tabs (Task 2), saved indicator (Task 4), video-first page (Task 5), portrait geometry (Task 6), Shut down demoted (Task 2), body buttons honest (Task 3), old recordings playable (Task 5 Step 4 + Task 6 Step 4). No gap found.

**Step scan:** every code step names a signature or an exact value; no "handle edge cases".

**Type consistency:** `RunState.STACK`, `PHASE_DETAILS["stack"]`, `RunConfig.session_start_sec`, `_check_cancelled`/`_check_cancel`, `#run-recording`/`recordingStatus`, `ROBOT_WIDTH`, `WIDTH`, `[body] 405×720` are used with the same names and values throughout.

**Review Focus:** five items, each with its test named in Task 1/2/4.
