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

# Session control (the UI calls the same endpoints).
curl -s -X POST -H 'Content-Type: application/json' -d '{}' \
  http://127.0.0.1:8765/api/session/start
curl -s -X POST -H 'Content-Type: application/json' -d '{}' \
  http://127.0.0.1:8765/api/session/stop
curl -s http://127.0.0.1:8765/api/state

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

```
session-root/sessions/<timestamp>-<id>/   # per-session logs, pose CSV
depth-stream.bin / depth-stream.log       # live stream (spike/e5)
depth-layer/vulkan/implicit_layer.d/      # e5 manifest
renderdoc-layer/vulkan/implicit_layer.d/  # e4 manifest
```

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
| `clank/` | App: config, orchestrator, dashboard/HTTP, VNC, command bus, pose relay, `depth_stream.py` reader |
| `spike/e5/` | Live Vulkan depth layer, shader (+`gen_header.py`), launcher, reader CLI, Monado patch |
| `spike/e4/` | RenderDoc trigger/replay/extract pipeline and metric post-processing |
| `spike/c0/` | Scripted OpenXR pose clients (`hold`, `yaw`, `manual`, ...) |
| `spike/e3/`, `spike/wbc*` | MuJoCo/SONIC body controller, third-person view, gates |
| `tests/` | Unit tests; `test_depth_stream.py` covers the mmap reader and encoders |
| `research/` | Measured findings and `evidence/` artifacts (`depth-stream/`, `depth-live/`) |
| `docs/` | Design docs; some are historical (the PNG preview predates VNC) |
| `README.md`, `RESEARCH.md` | Broad design and original feasibility study |

## Locating without a human, 2026-11-04

`spike/wbc_game/locate.py` - quadrant, then one pixel call on that crop, then a YES/NO on a tight
crop. Measured in `research/locate.md`.

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
