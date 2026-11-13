# C0 Hidden Session Orchestrator and Local Dashboard — Design

> **Archived 2026-11-13.** Designed the PNG-preview dashboard. The preview is the live depth layer (`spike/e5`) now, with VNC as the fallback.
> Current behaviour: [`README.md`](../../../README.md) · agent rules: [`AGENTS.md`](../../../AGENTS.md).

- Date: 2026-10-25
- Status: approved for implementation planning
- Scope: one local C0 session at a time, manually operated from a local web dashboard
- Target machine: the existing NixOS/Hyprland host, 1280x720 preview, Superhot VR under Proton

## 1. Problem

The C0 harness currently needs several manual terminals: Monado, Steam, the game, the pose client, and a separate capture step. The game also opens a visible window on the Hyprland desktop, which the user does not want.

The design must:

- launch the whole C0 session programmatically from one place;
- keep Steam and the game out of the user's Hyprland session;
- provide a local dashboard with manual lifecycle controls and a live preview;
- leave no background processes behind when the session stops.

## 2. Proven basis

This design is based on a successful attended spike on 2026-10-25:

- **Weston headless works as the hidden display.**  
  `weston --backend=headless --renderer=gl --width=1280 --height=720 --socket=<name> --no-config --xwayland --debug` creates a private compositor and Xwayland display. No window appears in Hyprland.
- **Steam can run inside that Weston session.**  
  Starting Steam with `DISPLAY=<weston-xwayland>` and `WAYLAND_DISPLAY=<weston-socket>` from a working directory under `$HOME` starts the client without touching the host desktop.
- **The shortcut launches with `steam://`.**  
  `steam steam://rungameid/18251483899710406656` launched the non-Steam shortcut. Steam resolved it to AppId `4249504744` and started Proton. This is the launch path the current C0 README did not yet have.
- **Monado's null compositor accepts the game.**  
  Starting `monado-service` with `XRT_COMPOSITOR_NULL=1`, `P_OVERRIDE_ACTIVE_CONFIG=remote`, `IPC_EXIT_ON_DISCONNECT=off`, and `XRT_NO_STDIN=1` works. The game connected as OpenXR client `Unity Application`; `c0_client.py` connected at the same time.
- **The game's mirror view is capturable.**  
  `weston-screenshooter` captured the real in-game view (hand/gun and red outlines) at 1280x720. A proof frame from the spike is kept at `/tmp/opencode/game-proof-weston-headless.png` (not committed).
- **Rejected alternatives:**
  - `gamescope --backend headless` with PipeWire capture produced black frames.
  - A Hyprland headless output produced black frames in `grim`.
  - These are not part of the design.

## 3. Goals and non-goals

### Goals

1. Start and stop one complete C0 session from a local web page.
2. Run Monado (null compositor + remote driver), Weston headless, Steam, Superhot, and the C0 pose client as one managed lifecycle.
3. Keep all Steam/game windows inside Weston's private display.
4. Show a live preview of the game at 1280x720, 5–10 fps, with the Weston panel cropped.
5. Expose manual C0 pose modes: `hold`, `yaw`, `trigger`, `dodge`, `walk`, `clock`.
6. Report process and OpenXR-client health, show log tails, and fail safely.
7. Stop cleanly with no leftover game, Steam, Weston, or Monado processes.

### Non-goals

- autonomous control, VLM integration, or decision-making;
- headset hardware or direct display output;
- audio routing or muting;
- video recording, streaming to other machines, or remote access;
- multiple simultaneous sessions;
- changing NixOS, Steam, the Proton prefix, or the game files;
- Windows or non-NixOS deployment.

## 4. Architecture

```text
Browser (localhost:8765)
    |
    | HTTP: HTML, /api/state, /api/session/*, /preview.mjpg
    v
clank.dashboard  (stdlib ThreadingHTTPServer)
    |
    | calls SessionManager methods
    v
clank.orchestrator  (single state machine, one lock)
    |
    +-- monado-service            TCP 127.0.0.1:4242, null compositor, remote driver
    +-- weston (headless, gl)     private Wayland socket + Xwayland display :N
    |     +-- steam               inside Weston, cwd=$HOME
    |     |     +-- Proton
    |     |           +-- SUPERHOT_VR.exe
    |     +-- xwayland            all game/Steam windows live here
    +-- c0_client.py              TCP 127.0.0.1:4242, pose/input stream
    +-- preview loop              weston-screenshooter -> crop -> JPEG -> latest.jpg
```

The dashboard process is the only long-lived process. It owns the session state and all child process handles. It binds to `127.0.0.1` only.

### 4.1 Components

| Unit | Purpose | Depends on |
| --- | --- | --- |
| `clank.config` | Typed configuration loaded from `config.toml` plus environment overrides; validates paths and numeric bounds. | Python stdlib |
| `clank.orchestrator.SessionManager` | State machine and process lifecycle; start, stop, mode switch, health probes, cleanup. | config, process helpers, preview |
| `clank.processes` | Small wrappers for process groups, `pgrep` probes, waiting with timeouts, log redirection. | stdlib subprocess/os/signal |
| `clank.preview.PreviewCapture` | Weston screenshot loop, crop, JPEG encode, atomic `latest.jpg` publication. | `weston-screenshooter`, `magick` |
| `clank.dashboard` | HTTP routes, JSON state, MJPEG stream, static files, request validation. | stdlib `http.server` |
| `clank/static/index.html` | Dashboard UI: preview, state, controls, logs. | no build step |
| `spike/c0/c0_client.py` | Existing pose client, reused unchanged via its CLI. | existing |

No FastAPI, Flask, Pillow, or other Python package is required. The only non-stdlib tools are Weston's own `weston-screenshooter`, ImageMagick `magick`, Steam, Proton, and Monado, all already present on the target machine.

### 4.2 Session state machine

States: `stopped`, `starting`, `running`, `stopping`, `failed`.

- `stopped -> starting -> running`
- `starting` failure -> cleanup -> `failed` (with `last_error`)
- `running -> stopping -> stopped`
- `failed -> stopping -> stopped`
- A second start request while not `stopped`/`failed` returns HTTP 409.
- Stop is idempotent from `stopped` and `failed`.
- Mode changes are allowed only in `running`; the manager stops the old pose client, starts the new one, and updates `mode`.

The manager holds one `threading.RLock` for lifecycle transitions. Health probes and preview capture run outside the lock and only read atomic status fields.

### 4.3 Startup sequence

Each step has a timeout and writes to the session log. `startup_timeout_sec` is an outer bound on the whole sequence; if it expires, the current step fails and the shutdown sequence runs. Any failure triggers the shutdown sequence and enters `failed`.

1. **Preflight**
   - Read `~/.local/state/Clank/current-session.json` if present. For each recorded process-group ID, check that it is alive and that its command line matches the recorded command name. If any match, fail with the PIDs and a targeted stop instruction. If none match, remove the pointer. Never kill a process only because its name matches `steam`, `weston`, or `monado`.
   - Host Steam is not running: `pgrep -x steam` must find nothing. If found, fail with: `Close the host Steam client before starting a hidden C0 session.`
   - `monado.service` and `monado.socket` are not active (`systemctl --user is-active --quiet ...`). If active, fail with a stop instruction.
   - TCP 4242 is free. If `/run/user/1000/monado_comp_ipc` exists and no process is listening on 4242, remove it as stale.
   - Required binaries/files exist and are executable/readable: Monado, Weston, `weston-screenshooter`, `weston-debug`, `magick`, Steam, game `.exe`, `~/.config/openxr/1/active_runtime.json`, `spike/c0/c0_client.py`.
   - Create the session directory under `~/.local/state/Clank/sessions/<UTC timestamp>-<short id>/` and write `~/.local/state/Clank/current-session.json` with the session path. Update it with each process-group ID as that child starts.
2. **Monado**
   - Start `monado-service` in its own process group with:
     - `XDG_RUNTIME_DIR`
     - `XRT_COMPOSITOR_NULL=1`
     - `P_OVERRIDE_ACTIVE_CONFIG=remote`
     - `IPC_EXIT_ON_DISCONNECT=off`
     - `XRT_NO_STDIN=1`
   - stdout/stderr -> `session/monado.log`.
   - Wait until TCP 127.0.0.1:4242 accepts connections (poll, 20 s timeout).
3. **Weston**
   - Start Weston in its own process group with:
     - `--backend=headless --renderer=gl`
     - `--width=1280 --height=720`
     - `--socket=clank-<short id>`
     - `--no-config --xwayland --debug`
     - `--log=<session>/weston.log`
   - stdout/stderr -> `session/weston-stdout.log`.
   - Parse `xserver listening on display :N` from the Weston log. The resulting display is `:N`.
   - Wait for both the Weston process and the log line (20 s timeout).
4. **Pose client (default `hold`)**
   - Start before the game so Monado always has a remote client.
   - Command:
     `python3 spike/c0/c0_client.py --mode hold --duration 3600 --log <session>/c0_hold.csv`
   - stdout/stderr -> `session/pose.log`.
   - Wait for `handshake ok` in the pose log (10 s timeout).
5. **Steam**
   - Start Steam in its own process group with:
     - `DISPLAY=:N`, `WAYLAND_DISPLAY=<weston socket>`, `XDG_RUNTIME_DIR`
     - cwd = `$HOME` (never `/tmp`; Steam's pressure-vessel container cannot chdir there)
     - `steam -cef-disable-gpu`
   - stdout/stderr -> `session/steam.log`.
   - Wait for Steam readiness:
     - Poll `weston-debug scene-graph` every 2 s for up to `steam_ready_timeout_sec` (120 s) until a window title contains `Steam`.
     - If `weston-debug` is unavailable, wait `steam_settle_sec` (60 s) and require the Steam process to be alive.
     - If the timeout expires while the Steam process is alive, log a warning and continue after the settle period. If the process died, fail.
6. **Game launch**
   - Run `steam steam://rungameid/18251483899710406656` with the same env and cwd as Steam. This command returns quickly; it only sends the URL to the running client.
   - Wait for the game process `SUPERHOT_VR.exe` (60 s timeout). Match the Wine game process by its `argv[0]` (`Z:\home\nate\Games\SUPERHOT VR\SUPERHOT_VR.exe`), not the `steam.exe` wrapper that carries the path as an argument.
   - Wait for a Monado client named `Unity Application` in `session/monado.log` (60 s timeout). This is the authoritative signal that the game bound to the runtime.
7. **Preview**
   - Start the capture loop as soon as Weston is ready (step 3). It shows the Steam UI during startup and the game when it appears. The loop is independent of the game and keeps the last frame on failure.
8. `state = running`, `started_at` set.

The shortcut ID is `18251483899710406656`. The Proton AppId is `4249504744`. Do not launch with `-applaunch 4249504744` or `rungameid/4249504744`; those resolve to a store AppID and do not launch this non-Steam shortcut.

### 4.4 Shutdown sequence

Executed on stop, on any startup failure, and on dashboard process exit. `stop_sec` is an outer bound; the individual waits below still apply.

1. Stop the pose client: `SIGTERM`, wait 2 s, `SIGKILL` if needed.
2. Stop Steam and the game:
   - Run `steam -shutdown` with the session env and cwd `$HOME`; wait up to 20 s.
   - If `SUPERHOT_VR.exe` or Steam's process group remains, send `SIGTERM` to the Steam process group, wait 5 s, then `SIGKILL` the group.
   - Probe for a remaining `SUPERHOT_VR.exe`; if present, `SIGKILL` by exact process match. Do not use a blanket Wine kill that could affect other prefixes.
3. Stop Weston: `SIGTERM` the process group, wait 5 s, `SIGKILL` if needed.
4. Stop Monado: `SIGTERM`, wait 5 s, `SIGKILL` if needed. Remove `/run/user/1000/monado_comp_ipc` only if no Monado process remains and the socket is stale.
5. Stop the preview loop thread and flush the session log.
6. Remove `~/.local/state/Clank/current-session.json`. Set `state = stopped`; preserve `last_error` only for the failed case.

Cleanup is best-effort but verified: after the sequence, the manager probes for `SUPERHOT_VR.exe`, `steam` (ours), the Weston process, and Monado. Any survivor is logged as a warning and reported in `/api/state`.

### 4.5 Preview pipeline

`PreviewCapture` is a daemon thread owned by `SessionManager`.

- Uses Weston's `weston-screenshooter` binary from the same package as `weston_bin`.
- Environment: `XDG_RUNTIME_DIR`, `WAYLAND_DISPLAY=<weston socket>`.
- Each capture runs in a fresh empty directory under `session/preview-capture/`, so the timestamped PNG cannot be confused with an older frame. Only one capture is in flight.
- After the command exits, take the newest `*.png` in that directory and require it to have been created after the capture started. If no new file appears, skip the frame.
- Crop the Weston desktop panel: `magick <png> -crop 1280x688+0+32 +repage -quality 80 jpg:<session>/latest.jpg.tmp`, then atomically rename to `session/latest.jpg`. The explicit `jpg:` prefix is required because the temporary file does not end in `.jpg`.
  - Width and height come from config; the crop height is `height - panel_height`.
  - The panel is 32 px in the tested Weston desktop shell. Cropping is harmless if a future Weston has no panel.
- Target rate: 8 fps. Each iteration sleeps for the remainder of `1 / preview_fps`; if a capture plus encode exceeds the budget, start the next iteration immediately and record a dropped frame.
- `latest.jpg` is replaced atomically, so HTTP readers never see a partial file.
- On capture failure, keep the previous frame, increment `preview.errors`, and log at most one warning per 10 s.
- The loop stops when the session stops or the Weston process exits.

### 4.6 Dashboard HTTP API

Server: Python stdlib `ThreadingHTTPServer`, bound to `127.0.0.1:8765` by default.

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/` | Serve `clank/static/index.html` and its CSS/JS. |
| GET | `/api/state` | JSON: state, session id, timestamps, mode, process health, preview status, last error, log paths. |
| POST | `/api/session/start` | Start the session; 202 if accepted, 409 if not stopped/failed, 400 on preflight failure with the error message, 500 on an unexpected server error. |
| POST | `/api/session/stop` | Stop the session; idempotent. |
| POST | `/api/session/mode` | Body `{"mode": "hold"|"yaw"|"trigger"|"dodge"|"walk"|"clock"}`; 400 for unknown mode, 409 if not running. |
| POST | `/api/session/kill` | Body `{"target": "game"|"steam"}`; force-stop that target, then refresh health; 400 for unknown target. |
| GET | `/preview.mjpg` | `multipart/x-mixed-replace` stream of `latest.jpg` at the capture rate. |
| GET | `/api/frame.jpg` | Current `latest.jpg`, or 404 before the first frame. |
| GET | `/api/logs/<name>` | Last 200 lines of `monado`, `weston`, `steam`, `pose`, or `session` log; 404 for unknown name. |

`/api/state` shape (stable keys):

```json
{
  "state": "running",
  "session_id": "20261025T223900Z-a1b2c3",
  "started_at": "2026-10-25T22:39:00Z",
  "mode": "hold",
  "processes": {
    "monado": {"alive": true, "pid": 1234},
    "weston": {"alive": true, "pid": 1235, "display": ":2"},
    "steam": {"alive": true, "pid": 1236},
    "game": {"alive": true, "pid": 1237, "exe": "SUPERHOT_VR.exe"},
    "pose": {"alive": true, "pid": 1238}
  },
  "openxr_client": "Unity Application",
  "preview": {"width": 1280, "height": 688, "fps_target": 8, "fps_actual": 7.4, "last_frame_at": "...", "errors": 0},
  "last_error": null,
  "logs": {"session": "...", "monado": "...", "weston": "...", "steam": "...", "pose": "..."}
}
```

The OpenXR client is derived from the last `application_name` line in the Monado log after the current session started. If the last event is `client_disconnected`, report `null`.

### 4.7 Dashboard UI

Single page, no build step:

- Header: state badge, session id, uptime, Start/Stop.
- Preview: `<img src="/preview.mjpg">` with a waiting overlay until the first frame.
- Pose modes: six buttons; the active mode is highlighted; disabled unless `running`.
- Health: cards for Monado, Weston, Steam, game, pose client, OpenXR client, preview fps.
- Danger zone: `Kill game`, `Shutdown Steam` (both require a confirm dialog).
- Logs: a select box for log name and a monospaced tail pane refreshed on demand.
- Errors: `last_error` shown prominently; preflight failures instruct the user to close host Steam or stop Monado units.

State is polled every 1 s. The MJPEG image is independent of polling.

## 5. Configuration

`config.toml` at the repository root, loaded with `tomllib`. Environment variables prefixed `CLANK_` override individual keys using uppercase `CLANK_<SECTION>_<KEY>` names (for example `CLANK_PATHS_WESTON_BIN`). Unknown keys are rejected at startup.

Defaults:

```toml
[paths]
monado_bin = "/run/wrappers/bin/monado-service"
weston_bin = "weston"
weston_screenshooter_bin = "weston-screenshooter"
weston_debug_bin = "weston-debug"
magick_bin = "magick"
steam_bin = "/home/nate/.local/bin/steam"
game_exe = "/home/nate/Games/SUPERHOT VR/SUPERHOT_VR.exe"
pose_client = "spike/c0/c0_client.py"
openxr_active_runtime = "/home/nate/.config/openxr/1/active_runtime.json"
session_root = "/home/nate/.local/state/Clank/sessions"

[launch]
shortcut_id = "18251483899710406656"
proton_app_id = "4249504744"
steam_extra_args = ["-cef-disable-gpu"]
steam_settle_sec = 60
steam_ready_timeout_sec = 120
game_process_timeout_sec = 60
openxr_client_timeout_sec = 60
startup_timeout_sec = 300

[preview]
width = 1280
height = 720
panel_height = 32
fps = 8
jpeg_quality = 80

[dashboard]
host = "127.0.0.1"
port = 8765

[timeouts]
monado_startup_sec = 20
weston_startup_sec = 20
pose_handshake_sec = 10
stop_sec = 60
```

`weston_bin`, `weston_screenshooter_bin`, and `weston_debug_bin` must resolve to the same Weston package. If `weston_bin` is an absolute path and the other two are left at their defaults, derive them as siblings in the same directory. If `weston` is not on `PATH` (the current NixOS host has it in the store but not in the profile), `run.sh` puts it on `PATH` via `nix shell nixpkgs#weston` or the user sets `CLANK_PATHS_WESTON_BIN` (and the two sibling keys if needed). The application must not hard-code store paths. `proton_app_id` is used to build the Player log path and is never used as a launch target.

## 6. File layout

```text
config.toml
run.sh
clank/
  __init__.py
  __main__.py          # argparse entry: serve dashboard
  config.py
  processes.py
  orchestrator.py
  preview.py
  dashboard.py
  static/
    index.html
    style.css
    app.js
tests/
  test_config.py
  test_orchestrator.py
  test_preview.py
  test_dashboard.py
docs/archive/specs/2026-10-25-c0-hidden-session-dashboard-design.md
```

`spike/c0/c0_client.py` stays where it is and is invoked by path.

## 7. Error handling and recovery

- Every wait is bounded; no unbounded `readline` or `Popen.wait`.
- Startup failures include the failing step and the tail of the relevant log in `last_error`.
- The manager never leaves the state at `starting` or `stopping`; a `finally` block always converges to `failed`/`stopped`.
- If the dashboard is killed (SIGINT/SIGTERM), it runs the shutdown sequence. If it is SIGKILLed, the next start's preflight detects stale Monado/Weston/Steam and instructs the user; it does not silently kill unrelated processes.
- If the game exits while `running`, the session stays `running` but `processes.game.alive` becomes `false`; the dashboard shows "game exited" and offers Stop. Start is enabled again once the session reaches `stopped`. It does not auto-relaunch.
- If Monado dies, the session transitions to `failed` and stops the remaining children.
- If Weston dies, preview stops and the session transitions to `failed` (the game's X display is gone).
- If the preview loop fails repeatedly, the session remains `running`; only preview health degrades.

## 8. Security and privacy

- Bind to `127.0.0.1` only. Do not add authentication or expose on the LAN.
- The dashboard can stop Steam and kill the game; the danger-zone actions require confirmation.
- The preflight refuses to run when the host Steam is active, preventing accidental control of a user's normal Steam session.
- No secrets are read, written, or logged. Log files may contain Steam IDs from Steam's own output; they remain local.

## 9. Testing

Tests use the Python stdlib `unittest` runner: `python -m unittest discover -s tests -v`.

### Automated, no game required

- `test_config.py`: defaults, env overrides, unknown-key rejection, path validation.
- `test_orchestrator.py`: state transitions with a fake process runner; start failure triggers cleanup; concurrent start returns conflict; host-Steam preflight; mode switch restarts the pose process; stop is idempotent.
- `test_preview.py`: crop math for 1280x720 with a 32 px panel; stale PNG rejection; atomic replace; frame rate accounting.
- `test_dashboard.py`: route validation, 409/400 responses, `/api/state` shape, `/api/logs` allowlist.
- `python -m clank --preview-smoke`: starts Weston headless with `mpv` (lavfi test pattern) on the private display, waits for a non-black frame, checks `/api/state` and `/api/frame.jpg`, then stops and asserts cleanup. Skipped if Weston/mpv are unavailable. This test must not start Monado, Steam, or the game.

### Manual acceptance with the real session

1. Start with host Steam closed. Click Start.
2. Confirm no Steam or Superhot window appears in Hyprland.
3. Confirm the dashboard reaches `running`, shows `SUPERHOT_VR.exe`, and `openxr_client = Unity Application`.
4. Confirm the preview shows the in-game view (not the Weston desktop panel).
5. Select `yaw` and confirm the in-game view rotates; select `hold` and confirm it stops.
6. Select each other mode once and confirm the pose client restarts without killing the game.
7. Click Stop. Confirm the session returns to `stopped` and `pgrep` finds no `SUPERHOT_VR.exe`, hidden Steam, Weston, or `monado-service` from the session.

### Acceptance criteria

- One click starts or stops the whole session.
- The game and Steam never appear in Hyprland.
- The preview is 1280x720 (minus the 32 px panel) at 5–10 fps on the target laptop.
- The game binds to Monado (`Unity Application`) without manual terminal commands.
- Stopping leaves no managed processes behind.
- Starting with host Steam running fails before launching anything.

## 10. Implementation notes

- Use process groups (`start_new_session=True`) and signal the group on stop.
- Do not run Steam from `/tmp`; use `$HOME`. The spike's `bwrap: Can't chdir to /tmp/...` failure is a known trap.
- `weston-screenshooter` writes a timestamped PNG into its current directory and has no output-path option; use a fresh directory per frame.
- The Weston desktop shell adds a 32 px panel. Do not try to use the kiosk shell for the game: in the spike, the kiosk shell plus an Xwayland client captured black frames.
- The game's Player log is at:
  `~/.local/share/Steam/steamapps/compatdata/4249504744/pfx/drive_c/users/steamuser/AppData/LocalLow/SUPERHOT_Team/SUPERHOT_VR/Player.log`
  The dashboard may show this path; it is not part of session cleanup.
- The OpenXR runtime pointer the game uses is `~/.config/openxr/1/active_runtime.json`. The orchestrator verifies it exists but does not rewrite it.
- The `spike/c0/launch_monado.sh` foreground workflow remains a fallback for debugging; the dashboard is the primary C0 path.
