# Runs — Design (an attempt you can repeat without a stack boot)

- Date: 2026-11-12
- Status: draft for review; no implementation until this document is accepted
- Scope: what a **run** is, what it resets, who owns it, and how the dashboard exposes it. Does
  not change the depth layer, the pose relay's wire, how the brain decides, or how a reach is
  planned.
- Related: `AGENTS.md` (process topology, the reach reset, the diary), `docs/superpowers/specs/2026-10-31-brain.md`
  §6.2 (the turn), `clank/orchestrator.py` (`SessionManager`), `clank/brainlog.py`,
  `spike/wbc_game/auto.py`.
- Target machine: the existing NixOS host with Job Simulator running through Monado.

Claim tags: `[MEASURED]` local file or named run; `[OPERATOR]` told by the person who owns the
game; `[ASSUMPTION]` inference, untested.

## 1. Problem

Three facts about the tree, 2026-11-12.

**1. Start/Stop is the whole stack, not one attempt.** `SessionManager.start()` runs preflight →
`monado-service` → Weston VNC → websockify → pose relay → Steam → `steam://rungameid` → client
wait → fullscreen; `stop()` reverses it, including `steam -shutdown` and killing the game by
argv0 `[MEASURED: clank/orchestrator.py §_startup, §_cleanup]`. A person testing the brain pays a
cold boot for every attempt, and nothing in the tree knows an attempt happened.

**2. The brain diary has no run boundary.** `clank/brainlog.py` appends to one global
`~/.local/state/Clank/brain/messages.jsonl`; `/api/brain/messages` and the log stream tail its
last 50 rows `[MEASURED: clank/brainlog.py:35, clank/dashboard.py:205]`. The panel therefore opens
on the previous attempt's sentences.

**3. Per-attempt state survives.** The simulated body keeps whatever pose it reached, the overlay
keeps whatever boxes the last grounding drew, and the game keeps its job state.

## 2. Claim

The **stack** is what costs a cold boot and can be reused. The **run** is what a person repeats.
They are different lifetimes and the dashboard conflates them.

A run is worth having when starting one resets everything a new attempt needs — and nothing a boot
needs — so the next attempt is one button, not thirty seconds plus a pose reset.

## 3. Vocabulary

| | today | after |
| --- | --- | --- |
| stack | `SessionManager`, `start()`/`stop()` | unchanged; "Start" / "Shut down" |
| run | does not exist | `RunManager`, `start()`/`stop()`; "Start run" / "Stop run" |
| turn | `auto.one_turn()` | unchanged |

A run requires a `running` stack. A run's end never stops the stack. Shutting the stack down stops
any active run first.

## 4. What a run resets

Resets, in this order:

1. **A new directory**: `sessions/<session-id>/runs/<run-id>/` (`run.json`, `brain.jsonl`,
   `brain.log`, `frames/`), with `run-id = <HHMMSS>-<6 hex>` like a session id.
2. **The overlay**: every grounding box from the previous run is dropped.
3. **The game's save data**: wiped while the game is dead, then the game is killed and relaunched
   through Steam, so the job is back at its very first frame (§5.1).
4. **The body**: stopped, started, and sent `rest` until it stands — the sequence `auto.ready()`
   measured (`AGENTS.md`: resetting first is the difference between 49-79 mm and 37 mm).
5. **The mode**: `body`, so the run does not inherit `manual` from a previous operator.

The stack is checked *before* any of this: if the planner's llama.cpp server is not up serving
LFM3-Omni, the run is refused and no game restart is paid for (§5.2).

Does **not** touch: monado-service, Weston, websockify, the pose relay, Steam, the session's own
`session.log` / `body_pose.csv`, the audio mute, the depth layer (it lives in the game process and
re-initialises itself).

**Manual shell runs are not runs.** `spike/e3/.venv/bin/python spike/wbc_game/auto.py` still works
and still writes the global diary; it just does not appear in the run panel. Only the dashboard
creates runs.

## 5. Architecture

**`clank/run.py` — `RunManager(manager, body, cfg, overlay)`.**

- `RunState(StrEnum)`: `idle, checking, preparing, game, body, brain, running, stopping, failed`.
- One worker thread per run, a generation counter cleared on stop — the same shape as
  `SessionManager`, because the same two hazards exist (a stop racing a start; a phase that cannot
  be interrupted).
- `start()` returns immediately (202) and raises `ConflictError` when the stack is not `running` or
  a run is active. `stop()` is idempotent. `snapshot()` is lock-free-safe and never touches a
  process.
- `body_reset()` lives here, not in `SessionManager`: `BodyPreview` is deliberately outside the
  session manager ("if it dies the session must not notice") `[MEASURED: clank/body_preview.py:1-10]`.
  It calls `body.stop()`, `body.start()`, then `send_control(..., {"verb": "rest"})` up to
  `REST_TICKS` times. With the body disabled the phase is skipped and logged; a run still works.

**`SessionManager` gains `restart_game()` and `wipe_game_saves()`.**

- `restart_game()` kills the tracked game pids, waits until `pids_by_argv0(game_exe)` is empty
  (bounded), wipes the save data, then calls the existing `_launch_game()`. That method already does
  rungameid → `wait_pids` → audio mute → pointer update → client-log wait → fullscreen; none of it
  is re-implemented in `RunManager`.
- **`_reset_openxr()` is called first.** `_launch_game` waits for `application_name:
  '<openxr_client_name>'` in `monado.log`, and the parser state is only reset in `start()`; with the
  previous game still in it, the wait returns immediately and the run starts against a game that is
  not up. This is the trap that would make the feature silently wrong.
- Refuses unless the stack is `running` and the stack's `_stop_event` is clear, under the
  lifecycle lock. A failure raises `StartupError` with the `steam.log`/`monado.log` tail, like
  `_step` does today.

### 5.1 Save data wipe (`wipe_game_saves`)

Measured on this install, 2026-11-12, prefix `compatdata/3415720001/pfx`:

| path under the prefix | size | what |
| --- | --- | --- |
| `drive_c/users/steamuser/AppData/LocalLow/Owlchemy Labs/Job Simulator/User/` | 5.1 KB | the save: `JobSimulatorSave.save.bak-002050` is the *only* file, so the game names its backup with a timestamp and a wipe must clear the directory, not a `.save` glob |
| `drive_c/users/steamuser/AppData/LocalLow/Unity/Owlchemy Labs_Job Simulator/` | 22 MB | Unity's per-product store, rewritten as the game is played; unclear whether it carries progress, so it goes too |
| `drive_c/users/steamuser/AppData/LocalLow/Owlchemy Labs/Job Simulator/output_log.txt` | 19 KB | the Unity log; **kept** — it is what a failed run is diagnosed from, and the dashboard already links it |

There is no `userdata/<account>/<appid>/remote/` for this title: it is a non-Steam shortcut
(`shortcut_id 14670405696621641728`, Proton prefix `3415720001`), so there is no Steam Cloud copy to
restore a wiped save `[MEASURED 2026-11-12]`. If the title is ever installed from the store, the
cloud question returns.

Rules:

- The paths come from `[paths] game_save_dirs`, each **relative to the Proton prefix**. A value that
  is absolute or resolves outside the prefix (after `realpath`) is rejected at config load; a wipe
  is never allowed to delete something the operator did not name.
- Wiped after the game's pids are gone and before `rungameid`, so no file is open and the game
  cannot rewrite them on exit.
- Each entry is removed rather than emptied in place, then recreated empty, so a leftover file with
  an unexpected name cannot survive. File and byte counts go to `session.log`.
- A missing path is not an error: a first run has nothing to wipe.

### 5.2 The model gate

`brain.server_status()` already distinguishes what matters `[MEASURED: clank/brain.py:227-284]`:
`up` + `status == "ok"` (weights loaded), `up` + `loading` (llama.cpp answers 503), `down` (nothing
listening), and `model`, the short name from `/v1/models`. The run's first phase asks it directly —
not `SessionManager.model_status()`, whose 5 s cache is for a one-second dashboard poll, not for a
start gate.

- Pass requires: `up`, `status == "ok"`, and `cfg.run.required_model` (default `LFM3-Omni`) as a
  case-insensitive substring of the reported model name.
- `down` or the wrong model → the run is **refused** (`failed`, `last_error`) with the answer in
  words ("nothing is listening on :11434", "serving LFM3-VL-2B-Q8_0, need LFM3-Omni") plus the
  command from `cfg.run.model_start_cmd`. Refusing *before* the game restart is the whole point: a
  missing model must not cost 40 s and a wiped save.
- `loading` → wait up to `cfg.run.model_timeout_sec`, re-probing every 2 s, with the phase text
  saying so. A 2B Q8_0 load is tens of seconds; it is a wait, not a failure.
- `cfg.run.model_autostart` (default **false**) runs `model_start_cmd` as a child when the server is
  down, then waits the same way. Off by default because the model, arguments and GPU flags are the
  operator's choice; on only when they have written the command down.

**The brain child.** `RunManager` spawns `cfg.run.brain_python` running `cfg.run.brain_script` with
`--for <cfg.run.brain_duration_sec>`, cwd = repo root, env = the stack's environment plus:

- `CLANK_BRAIN_LOG=<run>/brain.jsonl` — already honoured by `brainlog.default_path()`; no writer
  changes.
- `CLANK_AUTO_STATE=<run>/frames` — one new line in `auto.py`, replacing the hard-coded
  `~/.local/state/Clank/auto`, so run N's frames do not mix with run N+1's.

stdout and stderr are appended to `<run>/brain.log`. The child's exit ends the run.

## 6. Phases

```
idle --start--> checking --> preparing --> game --> body --> brain --> running
                     \__________ failure ___________/--> failed (stack untouched)
running/stopping: stop() -> stopping -> idle
```

1. **checking** (0.4 s, or up to `model_timeout_sec` when loading): the model gate (§5.2). First
   because it is the cheapest phase and refuses the run without spending anything.
2. **preparing** (instant): touch `brain.jsonl`, clear the overlay, rewrite `run.json`
   (`run_id`, `session_id`, `started_at`, `mode`, `brain_argv`, `diary`, `frames`). The run
   directory itself is created by the `start` call, before this thread runs, so a run refused by
   the model gate still has a `run.json` to record why.
3. **game** (20-40 s `[ASSUMPTION]`): `manager.restart_game()` — kill, wipe the saves, relaunch.
   This dominates wall time.
4. **body** (~5 s): `set_mode("body")`, then `body_reset()`.
5. **brain** (instant): spawn the child.
6. **running**: monitor the child every second; on exit write `ended_at`, `exit_code`,
   `duration_s` to `run.json` and go `idle`.
7. **stopping**: SIGTERM the child, SIGKILL after `cfg.run.stop_timeout_sec`; `rest` the body;
   clear the overlay; write the end fields.

**A failed run never touches the stack.** `failed` + `last_error`, run directory kept as evidence,
monado/Weston/Steam/game left running. That is the point of the change.

**Stop during `game` takes effect after the restart completes.** The restart runs inside the
stack's lifecycle lock and is not interruptible mid-way; the run sets a cancel flag that every
later phase checks. This is stated rather than hidden: the alternative is a game half-killed with
Steam believing it is alive.

## 7. Diary per run

- Writer: unchanged code, different path (`CLANK_BRAIN_LOG`).
- Reader: `/api/brain/messages` and the brain rows of `/api/logs` read
  `RunManager.snapshot()["dir"] + "/brain.jsonl"` explicitly — the dashboard's own `os.environ` has
  no override, so relying on `default_path()` would show the global file.
- `/api/brain/messages?run=<id>` reads a past run from disk (path validated: the id must match the
  run-id pattern and no separators).
- The panel keeps the last run's rows visible when the state is `idle`, labelled with that run's id
  — a panel that blanks between attempts is the complaint this change exists to answer.
- The global `brain/messages.jsonl` stops being appended to by dashboard runs. Manual shell runs
  still write it. `python3 -m clank.brainlog --tail` keeps reading it; pass `--path` for a run.

## 8. Dashboard

`/api/state` gains `run: {state, id, dir, phase, started_at, ended_at, error, exit_code,
duration_s}`.

| control | endpoint | enabled when |
| --- | --- | --- |
| Start | `POST /api/session/start` | stack `stopped`/`failed` |
| Start run | `POST /api/run/start` | stack `running`, run not active |
| Stop run | `POST /api/run/stop` | run `preparing`/`game`/`body`/`brain`/`running` |
| Shut down | `POST /api/session/stop` (unchanged) | stack not `stopped` |

Both run endpoints answer 202 and a 409 with the reason when refused. Shutting the stack down
stops the run first and waits for the run thread (bounded) — otherwise Steam is shut down
underneath a phase that expects a game.

The topbar shows the run id and phase beside the session id; the Brain panel header shows
`run <id> · <phase>`. `app.js` polls `/api/state` already, so no new polling loop. A refused model
gate writes its reason into `run.last_error`, which the page already renders in the error line, and
the telemetry panel's existing model row shows `model`/`status`/`up` from `brain.server_status()`.
The Start run button is also greyed while the model is `down`, with the reason as its title.

### 8.1 Loading is visible

Five waits exist and each one looks like a frozen page unless it is drawn as a wait. Every one
gets a spinner and a phrase; a spinner without a phrase is an anxiety, a phrase without a spinner
looks like a stuck page.

| wait | where it is shown | phrase |
| --- | --- | --- |
| session boot (~40-60 s) | topbar, beside the state badge | the existing badge text (`starting…`, `stopping…`) |
| run phase (`checking` → `brain`) | topbar run slot | `run <id> · <phase>` plus `run.detail` |
| model loading (up to `model_timeout_sec`) | same slot | `waiting for the model to load…` |
| preview transport | the picture's own overlay | the existing `waiting for VNC transport…` text |
| body view starting | the Robot panel | the existing `body view off` slot gains a spinner while starting |

`RunManager.snapshot()` therefore carries `detail`: one short human phrase for the current phase
(empty when `idle` or `failed`). It is display text only — nothing parses it.

The spinner is a CSS ring (`@keyframes clank-spin`), `hidden` by default, and is not animated under
`prefers-reduced-motion: reduce` (a static ring, so the state is still visible).

## 9. Configuration

```toml
[paths]
# Relative to the Proton prefix (<compatdata>/<appid>/pfx). Absolute paths and
# entries that resolve outside it are rejected at load. The Unity log is not
# listed: it is kept.
game_save_dirs = [
  "drive_c/users/steamuser/AppData/LocalLow/Owlchemy Labs/Job Simulator/User",
  "drive_c/users/steamuser/AppData/LocalLow/Unity/Owlchemy Labs_Job Simulator",
]

[run]
brain_python = "spike/e3/.venv/bin/python"   # must exist; the loop needs torch
brain_script = "spike/wbc_game/auto.py"
brain_duration_sec = 900                     # what ends a run on its own
game_restart_timeout_sec = 180
stop_timeout_sec = 10
rest_ticks = 8
required_model = "LFM3-Omni"                 # substring of /v1/models' name
model_timeout_sec = 180                      # a loading 2B model is a wait
model_autostart = false                      # off: the command is the operator's
model_start_cmd = ""                         # e.g. llama-server -m ~/AI/models/LFM3-Omni-2B-Q8_0.gguf ...
```

Validated like the rest of `config.py`: a missing `brain_python` fails loudly at load, not at run
start. Overrides are `CLANK_RUN_*`; unknown `CLANK_*` keys are still rejected.

## 10. Tests (stdlib, no GPU, no game)

Fakes for `SessionManager`, `BodyPreview`, and the child process; a temporary `session_root`.

1. `start()` creates the run dir, `run.json`, and `brain.jsonl`, and calls the fake manager's
   `restart_game()` exactly once, before the body reset and the spawn.
2. Phase order is game → body → brain; the body phase is skipped cleanly when `body is None`.
3. `start()` twice → `ConflictError`; `start()` with the stack stopped → `ConflictError`.
4. Child exit → `idle`, `run.json` end fields, exit code recorded.
5. `restart_game()` raising → `failed`, `last_error` set, the fake manager's `stop()` never called,
   the run dir kept.
6. `stop()` during `preparing` cancels before the spawn; during `running` terminates the child.
7. Dashboard: `/api/run/start` 409 when the stack is stopped; `run` appears in `/api/state`;
   `/api/brain/messages` reads the run's diary, not `$CLANK_BRAIN_LOG`; `?run=<id>` reads a past
   run and rejects `..`.
8. `auto.py`'s state dir is `$CLANK_AUTO_STATE` when set (a two-line unit test, no model).
9. `restart_game()` calls `_reset_openxr()` before `_launch_game` (a stub runner that records the
   order) — the trap in §5.
10. `wipe_game_saves()` removes files inside a temporary fake prefix, leaves a sibling
    `output_log.txt` alone, tolerates an absent directory, and rejects a configured entry that
    resolves outside the prefix (a symlink out of it, and `../..`).
11. `restart_game()` wipes only after the game pids are gone (a stub runner whose `kill_pid`
    records the timestamp order).
12. Model gate: down → refused with the command in `last_error`; `loading` → waits and passes when
    the probe turns `ok`; wrong model → refused; `model_autostart` runs the configured command
    once. A refused gate must leave the fake manager's `restart_game()` uncalled.

## 11. Non-goals

- Changing how the brain decides, or `auto.one_turn`.
- Restarting monado/Weston/Steam between runs.
- A queue of runs, scheduling, or automatic retry after a failed run.
- Per-run `body_pose.csv`: the relay is per-stack and restarting it races Monado's one-client hub.
  `run.json`'s `started_at`/`ended_at` slice the session CSV instead.
- In-game job reset. The save wipe plus the game restart *is* the reset; see §12.
- Starting, tuning or choosing the model server. The gate refuses a run; it does not own the
  process except through the opt-in `model_autostart` command.
- Steam Cloud save restoration. Measured absent for this install (§5.1); it comes back with a
  store-purchased copy.

## 12. Open risks

1. **Steam may refuse `rungameid` while it still believes the game is running.** Mitigation: kill
   by argv0 and wait for pids to disappear before relaunching; surface `steam.log`'s tail on
   failure. Measure on the first restart.
2. **The save wipe is the reset, and that is an assumption until measured.** The wipe removes the
   only save file present, so a relaunch should open at the game's first frame. What is *not* known
   is whether Job Simulator keeps any progress outside the two wiped directories (a registry value
   in `user.reg`, a file named differently next time). Measure on the first three runs: is the
   intro card back at its spawn, and is the job clock at the start? Record the answer in
   `AGENTS.md`.
3. **The gate can be right and the run still fail.** A server that answers `/health` and
   `/v1/models` can still refuse an image call. The gate buys a fast, clear failure for the common
   case (server not started), not a guarantee about the first turn.
4. **20-40 s per run may be too slow to iterate on the brain.** If the first three measured
   restarts say so, a `[run] game_restart = "never"` switch is the cheap escape: keep the game,
   reset only the body, diary and overlay.
