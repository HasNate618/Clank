# AGENTS.md

Orientation for coding agents. Read this first, then `README.md` for the design
and `research/` for the measured evidence. Some README/RESEARCH prose is
historical: it says Superhot VR, while the current `config.toml` and evidence
target **Job Simulator**. The architecture is the same.

## What this is

Clank runs a simulated Unitree G1 in a VR game with no headset:

```
body view (MuJoCo) -> BodyCommand/BodyState -> pose_relay -> Monado remote driver :4242 -> game
game (Proton/OpenXR) -> Vulkan depth layer (spike/e5) -> 480x270 mono depth + colour -> dashboard / model
```

Two live inputs matter:

- **Depth/colour stream** from `spike/e5`: the current, fast source. 480x270,
  left eye, float32 metres, ~13 ms layer latency, ~0.5-1 ms transport.
- **RenderDoc capture** from `spike/e4`: full-resolution stereo fallback, but
  ~16 s per frame. Kept for ground truth and debugging.

The dashboard at `http://127.0.0.1:8765` starts/stops the session, shows the live
stream, drives pose modes, and exposes the logs.

## Fast commands

```bash
# Dashboard WITH the depth layer (writes the layer manifest, sets DEPTH_STREAM_*).
# run.sh now runs this by default; launch.py is the same thing spelled out.
./run.sh --replace

# Plain dashboard, no depth layer. The preview is then VNC only, which is a
# frozen black window for Job Simulator (it renders only to the OpenXR views).
./run.sh --no-depth --replace

# Stack control: once per boot. The UI calls the same endpoints.
curl -s -X POST -H 'Content-Type: application/json' -d '{}' \
  http://127.0.0.1:8765/api/session/start
curl -s -X POST -H 'Content-Type: application/json' -d '{}' \
  http://127.0.0.1:8765/api/session/stop        # "Shut down" in the UI

# Run control: what you repeat. The stack stays up between runs.
curl -s -X POST -H 'Content-Type: application/json' -d '{}' \
  http://127.0.0.1:8765/api/run/start
curl -s -X POST -H 'Content-Type: application/json' -d '{}' \
  http://127.0.0.1:8765/api/run/stop
curl -s http://127.0.0.1:8765/api/state          # state.run walks the phases

# One capture. The reader is stdlib-only: no nix develop, ~0.4 s total.
python3 spike/e5/depth_stream.py --once --verify --out research/evidence/depth-stream

# Latency samples.
python3 spike/e5/depth_stream.py --watch --seconds 10

# Tests. XDG_RUNTIME_DIR must exist; the flake check makes one.
mkdir -p "${XDG_RUNTIME_DIR:-/tmp/clank-xdg}"
XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/clank-xdg}" \
  python3 -m unittest discover -s tests -t .
nix build .#checks.x86_64-linux.default   # sandboxed copy of the suite
# prefer this over `nix flake check`, which also evaluates the aarch64 outputs

# Rebuild the Vulkan layer (devshell supplies vulkan-headers; or set VULKAN_INCLUDE).
nix develop --command spike/e5/build.sh

# After editing the shader: rebuild SPIR-V, then the embedded uint32 header.
nix develop --command bash -c \
  'glslangValidator -V spike/e5/depth_stream.comp -o spike/e5/depth_stream.comp.spv &&
   python3 spike/e5/gen_header.py'
```

`nix build` leaves a `result` symlink; it is ignored. Flakes only see **tracked**
files, so `git add` a new patch/helper before building.

## Process topology and ports

Session startup order (`clank/orchestrator.py`): monado-service -> Weston VNC ->
websockify -> pose client/relay -> Steam -> game. Ports:

| Port | Process |
| --- | --- |
| 8765 | dashboard HTTP |
| 5901 | Weston VNC |
| 6080 | websockify |
| 4242 | Monado remote driver |

Stale Weston/monado/websockify processes block a restart and leave a black
preview. Kill them and free the ports before retrying. State, logs, the stream
and the injected layer manifests live under `~/.local/state/Clank/`:

```text
session-root/sessions/<timestamp>-<id>/   # per-session logs, pose CSV
depth-stream.bin / depth-stream.log       # live stream (spike/e5)
depth-layer/vulkan/implicit_layer.d/      # e5 manifest
renderdoc-layer/vulkan/implicit_layer.d/  # e4 manifest
```

## Stack and runs, 2026-11-12

**Start** boots the stack (Monado → Weston → websockify → pose relay → Steam → game)
once. **Start run** is the thing you repeat, and it never touches the stack: a new
directory, a new diary, the overlay cleared, the game's saved job wiped and the game
relaunched, the body stood up, then `spike/wbc_game/auto.py` under `spike/e3/.venv`.
**Stop run** ends the attempt and leaves the stack up; **Shut down** is the old Stop.
A failed run keeps its directory as evidence and never stops the stack.

```text
sessions/<session-id>/runs/<HHMMSS>-<6 hex>/
  run.json      # id, argv, started/ended, exit code, error
  brain.jsonl   # this run's diary; the Brain panel reads it
  brain.log     # the loop's stdout/stderr
  frames/       # CLANK_AUTO_STATE: the stills the brain looked at
```

- **The model gate is first, and it is cheapest.** `GET /health` + `/v1/models` on
  `:11434` must answer `ok` with `run.required_model` (`LFM3-Omni`) in the name, or the
  run is refused with the command to start it — before a game restart or a save wipe is
  paid for. `loading` is a wait of up to `run.model_timeout_sec`, not a failure. The
  topbar greys Start run only when nothing is listening.
- **The save wipe is what makes the game restart a reset.** `paths.game_save_dirs` are
  relative to the Proton prefix and are wiped while the game is dead; absolute paths,
  `..` and symlinks out of the prefix are refused. Measured 2026-11-12: the job's save
  is `LocalLow/Owlchemy Labs/Job Simulator/User/JobSimulatorSave.save.bak-<time>` (the
  only file there), and `LocalLow/Unity/Owlchemy Labs_Job Simulator/` is 22 MB of
  per-play state. `output_log.txt` beside them is kept — a failed run is diagnosed from
  it. There is no Steam Cloud copy: the game is a non-Steam shortcut. **Verified live
  the same day:** after a wipe the relaunched game spoke JobBot's opening line again
  ("Hello human, welcome to the job simulator... Here's my card"), so the wipe really
  does put the job back at its first frame.
- **A game restart costs 5.2 s, not the 20-40 s it was guessed at** (measured: wipe at
  `15:31:57.827`, game restarted at `15:32:03.008`). A whole run reaches `running` in
  ~25 s of which most is the body reset.
- **A refusal must be free.** With nothing on `:11434` the run failed in **0.03 s**, the
  game's pids were unchanged, and `run.json` held the reason. That is the gate's whole
  point: the message names the llama.cpp server and goes to the error line, not a
  tooltip.
- **`restart_game()` resets the OpenXR log parser first.** `_launch_game` waits for the
  new client's `application_name` line in `monado.log`; with the killed game still in
  the parser, that wait returns instantly and the run starts against a dead window.
- **What is loading is on screen.** A spinner plus a phrase in the topbar for the stack
  and the run phases (`run.detail`: "restarting the game…", "waiting for the model to
  load…"), on the preview while the transport comes up, and in the Robot panel. Under
  `prefers-reduced-motion` the ring stops turning but stays visible.
- The body reset is `BodyPreview.stop/start` + `rest` × `run.rest_ticks`, not a session
  concern: the body view is deliberately outside the session manager. A run works with
  the body disabled.
- **The body view's control socket is the fragile part of a run, and it cost two live
  runs.** `BodyPreview.start()` returns as soon as the process spawns; MuJoCo and the
  weights load after that. Worse, a spawn that finds the *dying* viewer still holding
  `sessions/body/control.sock` prints "another body view is already listening" and
  exits at once — so the old process answers the first `rest` and then unlinks the
  socket as it goes. Three defences, all measured:
  - the run waits for the socket, then **restarts a child that died** (bounded at four
    starts) rather than calling a reset that the old process acknowledged;
  - `run.body_timeout_sec` (120 s) bounds the wait;
  - `reach_vr.send` retries a refused or missing socket for `SOCK_WAIT_S` (20 s), the
    same doctrine as the Monado hub's "a quiet socket is not a failure, it is a queue".
    Without that retry the brain loop died 0.2 s into the run, twice, on `ready()`.
  The sidecar is stable once up: 1178/1178 connects over 60 s, one inode, no gaps.
- `python3 -m clank.brainlog --tail` still reads the global diary, which runs no longer
  write; pass `--path sessions/<id>/runs/<run>/brain.jsonl` for a run, or use
  `GET /api/brain/messages?run=<id>`.

## Depth stream (spike/e5)

- `depth_stream_layer.c` is an implicit Vulkan layer. It finds the game's
  `3840x1080` 4xMSAA `D32S8` depth (`fmt=130`) and the `fmt=37` colour target,
  and injects a compute dispatch after the final eye pass. The target is written
  twice per frame (a `CLEAR` scene pass and a `LOAD` pass that adds the other
  eye/glow); inject only the `LOAD` pass, otherwise the preview alternates
  between dim and bright frames.
- `depth_stream.comp` samples the left eye (`eye=0`) into `480x270`, converts
  reversed-Z with `z = near/d`, and packs RGBA8 colour at
  `480*COLOR_SCALE`. The default scale is 2 (`960x540`); depth stays `480x270`.
- The layer publishes a seqlock-protected mmap file: header `<12I3Q2f16I`
  (144 B, magic `0x434c4b44`, 3 slots), then float32 depth and packed RGBA8
  colour per slot. The layer's rows are **bottom-up**.
- `clank/depth_stream.py` is the stdlib reader. `Stream.read()` and `latest()`
  flip rows by default so PNG/npy/CLI and the dashboard are upright. The
  dashboard fetches `GET /api/preview/live.rgba` (raw RGBA plus
  `X-Frame-Width/Height`) and draws it on a canvas; `.bmp` remains for
  compatibility.
- Layer env vars (all `DEPTH_STREAM_*`, set by `spike/e5/launch.py`): `PATH`,
  `LOG`, `WIDTH`, `HEIGHT`, `EYE`, `NEAR`, `COLOR`, `COLOR_SCALE`, `FPS`,
  `LAYER`. `FPS` defaults to 30; the layer skips eye passes to hit it, and the
  dashboard canvas polls at the same 30 fps. Do **not** name them `CLANK_*`: the
  config loader rejects unknown `CLANK_*` overrides.
- The manifest is gated on `STEAM_COMPAT_DATA_PATH` +
  `GNUTLS_SYSTEM_PRIORITY_FILE=/dev/null` (present only in the game container)
  and disabled by `DISABLE_DEPTH_STREAM_PATH=1`. `run.sh` starts the dashboard
  through `spike/e5/launch.py`; when the layer library or the compatdata dir is
  missing it warns and starts the plain dashboard instead of failing.
- Near plane defaults to `research/evidence/depth-live/depth-calibration.json`
  (`near_m = 0.009403795`).

## RenderDoc fallback (spike/e4)

```bash
nix develop --command python3 spike/e4/depth_capture.py launch --replace
python3 spike/e4/depth_capture.py status
python3 spike/e4/depth_capture.py capture --out research/evidence/depth-live
python3 spike/e4/depth_capture.py extract --capture /tmp/game.rdc --out /tmp/depth
```

The flake pins RenderDoc 1.45 from the compat nixpkgs because Steam Linux
Runtime 4 has glibc 2.42 and RenderDoc 1.46 wants 2.43. Replay needs the RTX
4060 (lavapipe lacks `VK_EXT_conservative_rasterization`).

**RenderDoc RAM-flood trap:** the host OpenXR runtime still contains Monado's
RenderDoc integration. Loading `librenderdoc.so` into the game makes the OpenXR
client start/end a capture around every `xrBeginFrame`/`xrEndFrame`, filling the
container's RAM-backed `/tmp`. The repo Monado is built with
`-DXRT_FEATURE_RENDERDOC:BOOL=OFF`; the e4 launcher forces `XR_RUNTIME_JSON` to
the repo build for exactly this reason. Do not drop that forcing.

## Monado patches (flake.nix)

`monadoFor` applies `spike/e5/monado-no-visibility-mask.patch` and `postPatch`
edits the null compositor and remote HMD:

- null compositor: 60 Hz (was 20 Hz), recommends 1920x1080 per eye, cap raised.
- remote HMD: 3840x1080 display, 140 deg horizontal FOV, 16:9 per eye.
- visibility mask: empty hidden mesh + full visible quad. Without this the game
  stencils the lens corners black and the eye render/depth is a rounded
  rectangle.

The **service** comes from `CLANK_PATHS_MONADO_BIN` (devshell sets it). The
**client** comes from the host `~/.config/openxr/1/active_runtime.json`, which
may be a different (unpatched) Monado. Visibility masks are computed in the
service, so the repo service is enough for full corners. If you edit the patch,
rebuild (`nix build .#monado`), restart the dashboard/session, and use the new
`CLANK_PATHS_MONADO_BIN`.

## Body control: constraints into a generator, 2026-11-11

The reach/grab path has been replaced. **It is live**: the intro card was grabbed by the generated path
with the game running on 2026-11-12 (`research/evidence/ardy-live-grab/`), and `reach_vr.py` is now the
fallback rather than the session's path.
Design: `docs/superpowers/specs/2026-11-11-ardy-reach-design.md`. Plan: `docs/superpowers/plans/2026-11-11-ardy-reach.md`.
Critique and the open defects it found: `research/reach-architecture-critique.md`.
Comparison and measurements: `research/motion-generators.md`.

The old shape was "hand-write a reference, then ask a policy to follow it". The new shape is "state what to
achieve, let a motion prior choose the motion, let SONIC do the physics":

```
right hand at the object  +  feet planted  (or a root waypoint when out of reach)
  -> ARDY-G1-RP-25FPS-Horizon52  ->  36-float qpos at 25 Hz
  -> resample 25 -> 50 Hz        ->  SONIC mode 0 (10-frame look-ahead, flatten-hold at the end)
  -> mj_step 500 Hz              ->  body_state.json -> pose_relay -> game
```

Measured on the 4060 `[MEASURED: research/motion-generators.md §9]`: 0.94 GB peak VRAM, 0.07–0.20 s per
52-frame window, **45.5 mm** hand-constraint residual, 0.566 m travelled of a 0.60 m root goal, and clip
frame 0 within 1 mm / 0.8° of the live body. Tracking was then measured (gate 5, game off: 66 mm wrist
median) and finally live (`research/evidence/ardy-live-grab/`: 15.9 mm generator, 24.7 mm achieved on the **glove**).

### What the live run took, and what it still does not do, 2026-11-12

- **The constraint must pin the glove, not the wrist link.** The relay reports the hand 0.18 m out from
  `right_wrist_yaw_link` (`VR_LEVER`), so a wrist-pinned constraint leaves the game's hand 0.18 m past the
  object and makes the clasp's own measurement (0.18 m) larger than `CLASP_START_M` — it can never fire.
  `ardy_gen.glove_position` is the point to aim, measure and clasp on.
- **A bare closure command loses the arm.** `body_view.close_hand` records it: with no hand authority the
  controller falls back. Send the `close_<side>` verb, which re-sends the achieved wrist pose with the
  closure attached — three times, as the cruise's own grab does.
- **The freeze-hold must be re-latched at clip end.** `_fallback` keeps its latch keyed on *commanded*
  poses, and "nothing commanded" is the key it held before the reach, so the arms otherwise spring back to
  the pre-reach pose. `_finish_clip` re-latches onto `last_state`.
- **Scale decides whether the object is inside the arm.** Uniform scaling about the floor is the only
  variant that keeps the VR hands consistent with the head, but at 1.43 the card maps **0.566 m** from the
  sim's shoulder against a 0.41–0.51 m useful arm. Live runs use **1.2** (player 1.58 m, card inside the
  arm). A wrist pivot fits the arm (0.339 m) but stops scaling the camera, and the player looks short; two
  pivots break the hand-to-head relation. See `research/g1-scale.md`.
- **Still open:** the look (torso starts 60–64° off the object and the reach does not turn it — §8.5 of the
  design), the out-of-reach case (never converted), and `CLASP_START_M = 0.08`, which is **uncalibrated** and
  was the margin on the successful run (the clasp fired at 78 mm).

Four traps, all of which cost real time:

- **`LD_LIBRARY_PATH=/run/opengl-driver/lib`.** On NixOS `libcuda.so.1` is off the loader path, so torch
  reports `cuda False` with a perfectly good driver. Every ARDY run needs it.
- **A scalar `cfg_weight` silently disables constraints.** With `cfg_type="separated"`, ARDY sets
  `w_text, w_cstr = cfg_weight, 0.0` (`ardy_model.py:274-279`): 870 mm instead of 60 mm. Pass
  `(text, constraint)`.
- **`num_denoising_steps` must be <= `num_base_steps`** (10). Larger trips a CUDA device-side assert.
- **No text encoder is needed.** The checkpoint conditions on one pooled 4096-vector
  (`llm_shape: [1, 4096]`); pass zeros. Do not chase the gated Llama licence for a runtime dependency.

The workspace: `spike/ardy/.venv` (Python 3.12, torch cu128, ARDY editable from a clone) and the checkpoint at
`~/.local/state/Clank/ardy/ARDY-G1-RP-25FPS-Horizon52`. `spike/ardy/probe_g1.py` is the smallest working call.
The shared transport (`spike/wbc/g1_pack.py`, `mb_mailbox.py`, `clip_from_gen.py`, `end_pose.py`) is stdlib and
tested without a GPU.

## Gotchas

- `clank/` is **standard-library only**. Numpy/MuJoCo/ONNX live in
  `spike/e3/.venv`; do not import them from `clank/`.
- `config.toml` is machine-specific and git-ignored; `config.example.toml` is
  the template. Env overrides use `CLANK_...` and unknown keys are rejected.
  Read `config.toml` for the live paths (game exe, Proton app id, Monado/runtime
  pointer); do not hardcode `/nix/store` paths or the compatdata id in code.
- The pose relay runs with `--duration 3600`. When it ends the game holds the
  last pose and stops responding. Restart it with
  `POST /api/session/mode {"mode":"body"}` (or restart the session).
- The game's flat Xwayland window is static in VR, so Weston/VNC shows one
  frozen frame. The live picture is `spike/e5` colour via
  `/api/preview/live.bmp`; VNC is only a fallback when the layer is absent,
  and its login is the host PAM service `weston-remote-access`.
- `nix develop` costs ~7 s. The depth reader is stdlib and runs directly.
- `nix build` creates `./result`; it is ignored. Intermediates `*.ppm`/`*.pgm`
  beside the evidence PNGs are ignored and safe to delete.
- The session is one at a time. Steam/Proton leftovers and busy ports are the
  usual cause of a "stuck" start.

## Verification checklist

After a change:

1. `python3 -m unittest discover -s tests -t .` and
   `nix build .#checks.x86_64-linux.default`.
2. With the game running, `python3 spike/e5/depth_stream.py --once --verify`.
   Check `color.png` reads upright (text not mirrored), corners are non-black,
   and `frame.json` has `valid_frac=1.0`, `min_m` near 0.0094, median a few
   metres. Corner depth must not be the near plane.
3. `layer`/`age` medians should be ~13-17 ms, not hundreds, and `--watch`
   should report ~30 fps.
4. For the dashboard, a headless screenshot is enough:

```bash
chromium --headless=new --no-sandbox --disable-gpu \
  --user-data-dir=/tmp/clank-chromium --window-size=1600,1000 \
  --virtual-time-budget=6000 --screenshot=/tmp/clank-dashboard.png \
  http://127.0.0.1:8765/
```

5. Watch `free -h` for a minute. The layer should not grow memory; a flood means
   RenderDoc got loaded without the patched client runtime.

## Where to look

| Path | What |
| --- | --- |
| `clank/` | App: config, orchestrator, `run.py` (run state machine), dashboard/HTTP, VNC, command bus, pose relay, `depth_stream.py` reader |
| `spike/e5/` | Live Vulkan depth layer, shader (+`gen_header.py`), launcher, reader CLI, Monado patch |
| `spike/e4/` | RenderDoc trigger/replay/extract pipeline and metric post-processing |
| `spike/c0/` | Scripted OpenXR pose clients (`hold`, `yaw`, `manual`, ...) |
| `spike/e3/`, `spike/wbc*` | MuJoCo/SONIC body controller, third-person view, gates |
| `tests/` | Unit tests; `test_depth_stream.py` covers the mmap reader and encoders |
| `research/` | Measured findings and `evidence/` artifacts (`depth-stream/`, `depth-live/`) |
| `docs/` | Design docs; some are historical (the PNG preview predates VNC) |
| `README.md`, `RESEARCH.md` | Broad design and original feasibility study |

## Hearing the game, 2026-11-08

`clank/ears.py` records the game's own audio and nothing else; `clank/brain.transcribe()` turns a
clip into words; `clank/brainlog.py` writes the brain's sentences as it produces them.

```bash
python3 -m clank.ears --list                 # candidate playback streams
python3 -m clank.ears --transcribe           # one utterance, then the words
spike/e3/.venv/bin/python spike/wbc_game/listen.py --for 120
spike/e3/.venv/bin/python spike/wbc_game/do_task.py --listen --dry-run
python3 -m clank.brainlog --tail 20
```

- **Capture is a null sink the game alone feeds, recorded with `parecord`.** `pw-record
  --target <node>` **silently records the muted default source** on this machine - peak 0,
  exit 0, no stderr - which cost an hour of blaming the game. Proved with a positive control:
  a 440 Hz tone to a sink reads peak 0 through `pw-record` and peak 9000 through
  `parecord --device=<sink>.monitor`. Monitor sources also only exist for the PulseAudio API;
  `pw-dump` lists none. The route is
  `module-null-sink sink_name=clank-game` + `module-loopback` from its monitor to the
  operator's sink (so the game stays audible) + `move-sink-input`, then `parecord`. `ears.route()`
  is idempotent and `python3 -m clank.ears --route-only` applies it by hand.
- **Two ways the node goes silent, both seen.** `module-stream-restore` had remembered
  `Mute: yes` for `JobSimulator.exe`, so every sample was zero with the recorder working
  perfectly - check `pactl list sink-inputs | grep -A2 JobSimulator` first. And the game is
  silent at the `PULL LEVER TO JOB` menu: JobBot's first lines come after the job starts, so a
  restart alone does not produce speech.
- **Digital silence is the normal state between lines.** The gate is RMS on 50 ms blocks,
  floor = median of the first 0.5 s, gate = `max(4 x floor, 250)`, and it keys on `floor_known`
  - gating on `floor > 0` made the ear deaf to exactly this game.
- **Transcribe first, decide second.** One call asking for both words and a decision produced a
  goal that ignored what was said. `transcribe()` returns the line and the caller feeds it to
  `next_step()` as the situation, so the calibrated prompts keep working unchanged. Measured on
  JobBot's real audio: `Hello human, welcome to the job simulator. My name is JobBot...`,
  238 ms prefill + 310 ms decode. Silence parses to an empty transcript, not an error.
- **The diary logs events, not polling.** A quiet window is counted and reported once per run,
  never written: `listen.py` records a line only when words were heard, transcription failed, or
  the game was not up. An episode of "nothing happened" rows is a bug in the logger, not news.
- **The diary is the demo.** Every `heard`/`think`/`act`/`result` line goes to
  `~/.local/state/Clank/brain/messages.jsonl` (JSON per line, `CLANK_BRAIN_LOG` overrides) and to
  stderr as `[brain] 11:27:33 think: I will grab the job cartridge`. Nothing reads it back to make
  a decision - the dashboard is being rebuilt, so it is a diary, not state.

## Locating without a human, 2026-11-04

`spike/wbc_game/locate.py` - quadrant, then one pixel call on that crop, then a YES/NO on a tight
crop. Measured in `research/locate.md`. **Since 2026-11-09 stage 2 is a five-read box ensemble and
stage 3 is a description match; the three bullets below are the 2026-11-04 measurements.**

- **Stage 3 does not ask the model to judge its own answer any more.** It asks what object is in the
  box in six words and `describe_matches()` decides in code. Fed the brain's own nouns from JobBot's
  .srt - "card", "job cartridge" - the old YES/NO returned JobBot's ID badge for both, 2 px apart,
  with `present: true` and the same spread (39.9 vs 39.7 px). Now "card" accepts at 22.5 px and
  "job cartridge" **refuses**, on the untuned 1705x961 frame too.
- **Refusal is offered, not hoped for.** `{"none": true}` in the box prompt makes the model refuse
  5/5 reads for an absent noun; without that clause it never refused once.
- **Colour is not identity.** A box described as "green robot computer" matched "job cartridge" on the
  shared colour word and pointed at a monitor. No colour-only matches. Synonyms are this job's
  vocabulary (card/cartridge/lever), so an unknown noun refuses even when the box was right -
  JobBot's head was found to 42.8 px and refused because the model called it a "computer monitor".
- **`auto.py` gates on `present: False` now.** It used to log the field and reach at the pixel anyway.
- **Two optimisations measured and rejected**: the spread does not separate right from wrong (198 px
  was a correct answer, 455 px was 85 px off), and clustering the reads beats the per-edge median on
  nothing (1-4 px, winning two cases and losing two).

- **Card: 3.0 px and 17.4 px**, nothing hand-marked. Refuses when the card is genuinely absent.
- **Cartridge rack defeats it** (66 px, then 369 px), and on a live frame with the card fallen out
  of view it returned a confident pixel at a white rectangle. **Stage 3 checks that *a* thing is
  there, not that *the* thing is.**
- **The model normalises a crop to a square, width-based grid.** Both axes divide by the resized
  width. Dividing y by the crop's own height turned a 4.7 px answer into 163 px. Do not "correct"
  the aspect ratio.
- **locate on a 960-wide still.** The one confident error in the fixture set was on an un-resized
  1920x1080 frame; resized, it refuses instead.
- `find_and_reach.py` chains capture -> locate -> `reach_vr`, and stops with exit 2 on a refusal.

## Which model, 2026-11-04

`research/grounding-vl-vs-omni.md`. Same image, same prompts, both models:

| | vl3 | omni |
| --- | --- | --- |
| full frame | **22 px** | 360 px |
| one call on a crop | **30 px** / 19.3 s | 39 px / **0.73 s** |
| absent object | hallucinated 4/4 | hallucinated 3/4 |

Reasoning buys accuracy and costs 5-36x the wall time and the only audio channel. **Omni stays the
planner**; vl3 is a one-shot locator when 5.7 s per pixel is acceptable. Neither model can be
trusted to refuse, so absence is the caller's job.

## The grab, 2026-11-04

**It works.** Reset the body, mark the card, reach, close, lift, carry 0.29 m, open, drop.
Details and images in `research/reach-placement.md`.

Two things had to be right, and neither was tuning:

1. **Reset the body first** (`POST /api/body/stop`, then `/start`). Every failed attempt started
   from wherever the policy had wandered; the 49-79 mm residuals were that, not the reach.
2. **Close the hand inside the reach loop, with no gap.** `close_hand` already pins the hand to
   its achieved pose, so the pose was never wrong - but closing from a separate process left ~2 s
   in which the hand drifted 0.09 m, which is more than the grab radius. The fingers shut on air.

Reach + close is now one command: `reach_vr.py --grab`.

## Pointing at things, 2026-11-06

The agent cannot see the game, so every "find the lever" attempt guessed a pixel and every guess was
wrong: a depth local-minimum scan found the console's corner, the vision model found a card 0.81 m
away, and projecting candidates onto the handle found a queue stanchion. Ask the operator instead.

```bash
spike/e3/.venv/bin/python spike/wbc_game/pick_pixel.py        # resets the body, serves one frame
# operator opens http://127.0.0.1:8771/ and clicks the object
cat ~/.local/state/Clank/pick-pixel/pixel.json                # {"x":652,"y":299,...}
```

The page draws a labelled 100 px grid and a centre crosshair. Unproject the pixel with
`clank.pixel3d.point_from_pixel`, then reach it.

`spike/wbc_game/measure_reach.py X Y Z` does **one** reach and one measurement - no correction
passes, no head pin - and prints the gap, plus how far the head and root moved on their own.

- **The hand ends ~0.25 m too high, and it is a gain, not a bias.** Ask for z = 0.10 and the hand goes
  to 0.33; ask 0.357 and it goes to 0.43-0.60. `reach_vr.Z_GAIN = 1.58` divides the climb about the
  resting hand: one shot went 143/197/229 mm -> **60/79/78 mm** at three heights, with dz collapsing
  to ~0 at all three. End to end with one correction, the operator's clicked lever point lands
  **37 mm** in 4.3 s.
- **`WRIST_BOX` is a steering limit, not the arm's reach.** It caps y at -0.31; unclamped the arm holds
  y = -0.389. `reach_vr.py --no-clamp` bypasses it, and that is what let the lever be reached at all.
- **A reach moves the body even when you ask it not to**: measured 25-55 mm of head and 62-87 mm of
  root drift inside a single 3.8 s reach. A pixel marked once goes stale, and re-pinning the head at
  20 Hz is what twisted the torso earlier.

## The fingers close, 2026-11-06

The body used to grab with an open hand: the game saw the squeeze and the 14 hand motors were never
written. `SonicExecutor._clasp` now ramps the fingers to `g * GRASP_Q` (GR00T's `grasp_qpos` for the
deployed Dex3 hand) at 0.25 rad/tick. Measured 0.13 rad from target, and photographed in
`research/evidence/hand-clasp-closed.png` / `-released.png`. Details: `research/hand-clasp.md`.

- **The hand motors are torque units**, `<motor>` with `ctrlrange +-0.7 N m` (`+-2.45` on `thumb_0`),
  so they take PD (`KP 3.0`, `KD 0.10`), not an angle. `SonicRunner.step_policy` writes `ctrl[:29]`;
  the hand motors are 29..42.
- **A clasp test must raise the arm first.** In the rest pose the hand rests against
  `right_hip_pitch_link` and the thumb is pinned at its limit with full torque applied - measured
  `thumb_1` at +0.730 while its motor pushed -0.700. That looks exactly like a broken hand.
- **`BodyState.closure` stays the commanded value.** The game's squeeze button must fire the instant
  the body is told to grab; making the field measured would delay the button by the 0.1 s clasp.
- **The game draws the fist from `hand_curl`, not from squeeze.** `clank/pose_relay.py` therefore
  defaults to `--grab-mode both`: squeeze alone registers the grab with a visibly open glove. If a
  grab works but the glove stays open, check `sessions/<id>/body_pose.csv` - the `curl_l,curl_r`
  columns say what was actually sent, which is not always what you think is running.

## Seeing the object, 2026-11-04

Three measurements change how the reach and the grounding are built. Read them before
changing anything in `spike/wbc_game/reach_vr.py`, `track.py` or `clank/brain.py`.

**The head will not hold a commanded pose.** Held against one target for 10 s, the achieved
head wobbled **11.7 deg of yaw**, sat ~16 deg off the command, and its position drifted
**56 mm**. So a pixel marked once is stale within a tick - measured, the card moved
`1009 -> 1122 -> 865 -> 1007 px` inside a single four-command reach. Do not try to solve
this by pinning harder; the pin is a request the policy answers with something else.

**Re-derive the pixel instead.** The object is fixed and the camera is measured every tick,
so projecting the object's own point through the measured head is accurate: the *hand's*
pixel, predicted from its 3D position, landed **18 px** from where the game drew it.
`spike/wbc_game/track.py` does that; `watch_glow.py` runs beside a reach and logs the
object pixel, the hand pixel, their separation, and the glow state.

**Grounding is one call on a crop.** Measured against the JobBot card (truth (705, 420)):

| mode | error |
| --- | --- |
| full frame, pixel | 360 px |
| full frame, 3x3 cell | 565 px (wrong cell) |
| crop, pixel | 65 px |
| crop, quadrant | correct |
| **one call including the crop** | **39 px, 0.72 s** |

Two calls cost more and are *less* accurate (65 px plus 0.78 s for PLAN). PLAN alone invents
targets when it has no image, so always pass the frame - `clank.brain.plan()` does.

**The glow is a cliff, not a judgement.** The same card patch measures `min(G-R, B-R)` of
97-108 glowing and 5.9-40.4 not. It cannot be *found* by scanning: within 120 px of the card,
the brightest cyan patch in a non-glowing frame scored 163 against the card's 5.9, because the
room is painted the same hue. Only the change at a known patch means anything.

**Depth fading helps a little and is not a crop substitute.** `spike/wbc_game/fade.py` blends
past `1.8x` the object's depth to a dark backdrop by `3.5x`, with a smoothstep because the
depth buffer is torn often enough that a hard cut bites into the object. Measured: full-frame
grounding 631 -> 363 px. Still unusable, so the crop stays. Sample the object's depth as the
**nearest** surface in a patch, not the median - the card is 0.38 m inside a neighbourhood
that is mostly background at 1.31 m.

**The emulated hands are visible in the game view.** Hold one up in front of the face and it
appears where the geometry says. An earlier note in this file said otherwise; it was wrong.

## Pose clients and the hub, 2026-11-06

Switching to `manual` used to kill the session: `c0_manual` raised `TimeoutError` in
`read_handshake`, the orchestrator saw the pose process die, and the whole session (game,
Weston, monado) was torn down. Three measurements explain it, and two fixes stop it.

- **The remote hub serves one client at a time.** `src/xrt/drivers/remote/r_hub.c` accepts,
  writes the two handshake packets, then reads that client until EOF. A second client's TCP
  connect still succeeds (kernel backlog) and it receives **0 bytes for as long as the first
  client lives** - measured against a live session. So a switch is a race against the previous
  pose client being let go.
- **A quiet socket is not a failure, it is a queue.** Every pose client now retries the whole
  handshake (`c0_client.connect_with_handshake`, 20 s total, 1 s attempts) and prints
  `driver busy (...); retrying`. `pose_relay.py` always did this; `c0_manual.py` and
  `c0_client.py` did not. Measured before the fix: a real switch needed ~10 attempts.
- **The hub's handshake write could kill monado-service.** It used a plain `write()`/`send()`
  with no `MSG_NOSIGNAL`, so when the hub finally served a socket whose client had already
  given up, SIGPIPE terminated the service - `monado.log` ends at `Connection received! N` with
  no `Server exiting`, and every later connect is refused. `flake.nix` now substitutes
  `send(..., MSG_NOSIGNAL)` in `r_hub.c` (the IPC channel next door already did this). Verified
  by starving the hub on purpose: the switch retried, handshook in 4.9 s, and both monado and
  the session stayed up.

`timeouts.pose_handshake_sec` must outlast the client's 20 s retry window; it is 25 now. A
`state: failed` right after a mode switch, with `monado.log` truncated mid-hub and later
`Connection refused` in `pose.log`, is this bug's signature.
