# C0 — does Superhot accept our synthetic HMD?

C0 has been the unrun go/no-go since the start of the project. Everything measured in `spike/e1`–`spike/wbc_classical` and `spike/wbc` is worthless to the artifact if the game will not take a fabricated pose stream. This directory is the harness to answer it, plus the game-side numbers `research/wbc-game-proposal.md` could only specify.

**This is a deliberate, attended session.** Nothing here should run in a background pane: it opens a Proton game, injects an OpenXR runtime, and the results only mean anything if someone watches the screen.

## What makes this cheaper than expected

Two facts, both measured, both reduce the risk:

1. **The build speaks OpenXR natively.** `SUPERHOT_VR_Data/Plugins/x86_64/` contains `UnityOpenXR.dll` and `openxr_loader.dll`. There is no OpenVR shim in the path, so **xrizer is not needed**. Monado's `remote` driver is directly the runtime the game will bind to.
2. **The clock floor is `0.005`, not zero.** `TimeControl.minTimescale` is serialized as `0.005` in levels 0–5, so a 20 s think costs ≈0.1 s of game time. C0 confirms this empirically rather than trusting the constant.

## The archive

`/home/nate/Downloads/Superhot-VR.rar` is **actually a ZIP** (compression method=store) and it is a **Windows** build, so it needs Proton. `extract_game.sh` unpacks it without touching the archive.

`GE-Proton10-34` is installed at `~/.local/share/Steam/compatibilitytools.d/GE-Proton10-34`. `umu-run` is not installed, so the path is Steam → add as non-Steam game → force a Proton build.

**Measured 2026-10-24 — what actually worked differs from the above.** The game was extracted to `/home/nate/Games/SUPERHOT VR/` and launched through **`proton-cachyos-slr`**, not GE-Proton10-34. It reached a focused session on Monado with **no `XR_RUNTIME_JSON` launch option and no `PRESSURE_VESSEL_IMPORT_OPENXR_1_RUNTIMES`**, because under Wine the *Windows* loader reads the prefix registry (`C:\openxr\wineopenxr64.json` → `wineopenxr.dll`) and that shim forwards to the host Linux loader. What it needs instead is a runtime pointer at **`~/.config/openxr/1/active_runtime.json`** — `/etc/xdg` is not visible inside the container, and `~/.local/share/openxr/1/` is not a searched path. The loader caches its runtime list per process, so **a game restart is mandatory** after adding it. **The launch must come from the Steam client UI:** `steam -applaunch` resolves the shortcut to a store AppID and fails `AppError_9`; `rungameid` reports `Unknown GameID type`; running the Proton entry point by hand fails `bwrap: execvp true`. Full results: `research/c0-results.md`.

## Order of operations

```bash
# 0. one time
spike/c0/extract_game.sh

# 1. terminal A — the runtime. Ctrl-C to stop.
spike/c0/launch_monado.sh

# 2. terminal B — prove the pipe BEFORE the game, with hello_xr (E1/E2 recipe)
sleep 300 | /nix/store/4z8ibn91k9r6yismm2nrcnl19b1v9rp7-openxr-loader-1.1.54/bin/hello_xr -g Vulkan2

# 3. terminal B — the client, in the order below, capturing as you go
python3 spike/c0/c0_client.py --mode yaw     --duration 20
python3 spike/c0/c0_client.py --mode trigger --duration 20
```

The client speaks the E1 wire format (376-byte `r_remote_data`, header `mndrmt3\0`) and logs every tick to CSV. Geometry is the **real G1**, not E1's placeholder: HMD at the measured **1.221 m**, wrists at the measured rest pose `(∓0.222, 0.721, 0.017)` converted through the E3 frame map. E1 used `(0.2, 1.3, -0.5)`, which is not a body we have — do not reuse it, because the game's apparent scale is part of the honesty claim.

Modes: `hold` · `yaw` · `trigger` · `dodge` · `walk` · `clock`.

## The six measurements, in dependency order

Run them in this order. Stop at the first hard fail.

| # | Question | How | Pass |
| --- | --- | --- | --- |
| 1 | Does the game bind to our runtime at all? | launch it, watch the headset view | non-black stereo view that responds to `--mode yaw` |
| 2 | Does the view follow TCP poses? | `--mode yaw --yaw-deg 35 --duration 20` | the in-game view **rotates ±35°** and **keeps following**; it must not freeze when packets keep arriving |
| 3 | Does input reach the game? | `--mode trigger --duration 20`, watch the pistol / UI | trigger registers (shot fired, or the UI reacts) |
| 4 | Does grim get the *game*? | `spike/c0/capture_game.sh` while the game is up | a frame of the game, not the desktop; geometry sidecar written |
| 5 | What is the clock floor in practice? | `--mode hold --duration 20` then `--mode walk --duration 10`, capture both | enemy/scene pixel motion is ~**0.5%** of the walk case during the hold |
| 6 | Is the pistol handed over, and must the exit be touched? | play/observe The Fall to its end | documented: gun in hand or pickup distance; level ends without a step or not |

### Clock protocol (measurement 5)

The serialized `0.005` is a constant; the mapping from *player motion* to *timescale* is not decompiled. Measure it with observable pixels only — **never** game memory, which is an anti-goal:

1. Get into The Fall with the enemy active (it approaches you).
2. `--mode hold --duration 20`, capturing at ~2 Hz. The scene should be nearly static.
3. `--mode walk --walk-speed 0.64 --duration 10`, capturing identically. This is SONIC's measured walk speed, so this case is what the body will actually look like when moving.
4. Compare the enemy/scene pixel displacement per real second between the two captures. The ratio is the timescale ratio.
5. Sanity-check against `0.005`: if the hold case moves ~0.5% as fast, the constant and the behaviour agree.

Record the frames. The number goes in the doc as the freeze budget, and it decides whether a 20 s VLM think is safe.

## What counts as a pass for C0 as a whole

- The view follows scripted poses, and keeps following.
- Trigger reaches the game.
- grim captures the game window.
- The clock floor measured in-game is consistent with `0.005` (within an order of magnitude).
- The Fall's pistol/exit question is answered either way — "the gun must be picked up" is a *result*, not a failure of C0.

## Hard fails, and what they mean

- **No non-black view / the game never binds** → Monado `remote` is not acceptable to this build. This is the one case where switching title is allowed (`../../docs/2026-10-23-clank-design.md` §12).
- **The view follows once then freezes while packets still arrive** → the driver adopted a pose and stopped reading; check `IPC_EXIT_ON_DISCONNECT` and that only one client is connected (the hub accepts exactly one).
- **Trigger does nothing** → first suspect controller `active`/`trigger_click` bits in the packet rather than the game; then the game's input mapping.
- **grim returns the desktop** → window class/title matching; `capture_game.sh` prints every open window when it fails.

## Artifacts to keep

- `spike/c0/logs/c0_<mode>.csv` — the poses we actually commanded, per tick. Written locally and **git-ignored**: per-tick logs are not committed, the numbers live in `research/c0-results.md`.
- `spike/c0/logs/shots/game_*.png` + `game_*.txt` — the game frames and their geometry.
- A short note in `research/` with the six answers and the clock ratio.

Do not modify the game files, the Proton prefix, Steam, Monado's unit, or NixOS. Monado runs as a foreground process here for exactly that reason.
