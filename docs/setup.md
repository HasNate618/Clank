# Setup, ports and controls

The material that does not belong on the front page: what to install, what to put in `config.toml`,
what listens where, and what each control does. The README has the five commands that get you a run.

## What has to be on the machine

| | |
| --- | --- |
| NixOS with Steam | Tested on a Hyprland/Wayland host. Nothing else is claimed. |
| *Job Simulator* | Your own copy, added to Steam as a non-Steam shortcut and running under Proton. Not distributed here. |
| Monado with the `remote` driver | The service comes from `CLANK_PATHS_MONADO_BIN` in the devshell; the *client* comes from your host's `~/.config/openxr/1/active_runtime.json`, which may be a different build. |
| NVIDIA GPU, 8 GB | The game, the body and the model are co-resident. |
| llama.cpp on `:11434` | Serving the 2B model named in `run.required_model`. The flags are pinned in `AGENTS.md`; defaults answer differently. |
| `weston`, `websockify`, `novnc`, `weston-debug`, `xdotool` | Supplied by `flake.nix`; `run.sh` re-enters `nix develop` when they are not on `PATH`. |
| The PAM service `weston-remote-access` | Authenticates the VNC viewer. Host configuration, not something this repo installs. |

## config.toml

`config.toml` is git-ignored because it holds absolute paths. `config.example.toml` is the template and
every value is validated at startup.

| Key | What to put there |
| --- | --- |
| `paths.steam_bin` | Your Steam client launcher. |
| `paths.game_exe` | Path to `JobSimulator.exe`. It is launched through Steam, not directly. |
| `paths.openxr_active_runtime` | Your Monado runtime pointer, below. |
| `paths.session_root` | Where session logs, runs and recordings go. |
| `launch.shortcut_id`, `launch.proton_app_id` | Your Steam shortcut's AppId and the Proton compatibility id. |
| `run.required_model` | The model name the gate requires on `:11434` before a run may spend anything. |

**The OpenXR runtime pointer.** Under Wine the game resolves its runtime through the Proton prefix
registry to `wineopenxr.dll` and then the host loader, which reads
`~/.config/openxr/1/active_runtime.json`. `/etc/xdg` is the *container's* inside pressure-vessel, and
`~/.local/share/openxr/1/` is not a searched path at all. The loader also enumerates manifests once per
process, so fixing the pointer requires a game restart.

```bash
mkdir -p ~/.config/openxr/1
cp /etc/xdg/openxr/1/active_runtime.json ~/.config/openxr/1/active_runtime.json
```

## Ports and processes

Session startup order is Monado, Weston, websockify, the pose client, Steam, then the game.

| Port | Process |
| --- | --- |
| 8765 | dashboard HTTP |
| 5901 | Weston VNC |
| 6080 | websockify |
| 4242 | Monado remote driver |

Stale Weston, Monado or websockify processes block a restart and leave a black preview; kill them and
free the ports. State, logs, the live stream and the injected layer manifests live under
`~/.local/state/Clank/`.

## Controls

The dashboard is one local page. **Start run** is the only start the page offers, and it is repeatable.

| Control | Effect |
| --- | --- |
| **Start run** | Boots the stack if it is down, never touches one that is up, checks the model, then wipes the game's saved job, relaunches the game, resets the body and starts the loop. Refuses before spending anything when the model server is not up. |
| **Stop run** | Ends the attempt and leaves the stack, the game and the body running. |
| **Shut down** | Stops the run, then the session, then reports anything that survived. |
| **mode** | Switches the pose client: `hold`, `yaw`, `trigger`, `dodge`, `walk`, `clock`, `manual`, `body`. `body` is the join. |
| **kill** | Force-stops the game or Steam. These hit *your* Steam. |
| **preview** | The live depth-layer stream, or VNC when the layer is absent. View-only. |
| **Recordings** | `GET /recordings` lists past runs and plays one beside that run's diary. |

## Safety

- This launches Steam and the game in a nested Weston session under a null-compositor Monado, so it does
  not draw on your desktop.
- The preview is loopback-only and the control API has no authentication. Keep `dashboard.host` on
  loopback.
- The kill buttons are force kills: anything unsaved in another Steam game does not survive one.
- One session at a time, guarded.
- It does not modify your OpenXR configuration. It validates the pointer and fails preflight if it is
  missing.
