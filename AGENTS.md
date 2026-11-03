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
# Use this, not run.sh, when you need depth.
nix develop --command python3 spike/e5/launch.py --replace

# Plain dashboard, no depth layer.
./run.sh --replace

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
  `LOG`, `WIDTH`, `HEIGHT`, `EYE`, `NEAR`, `COLOR`, `COLOR_SCALE`, `LAYER`. Do
  **not** name them `CLANK_*`: the config loader rejects unknown `CLANK_*`
  overrides.
- The manifest is gated on `STEAM_COMPAT_DATA_PATH` +
  `GNUTLS_SYSTEM_PRIORITY_FILE=/dev/null` (present only in the game container)
  and disabled by `DISABLE_DEPTH_STREAM_PATH=1`.
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
3. `layer`/`age` medians should be ~13-16 ms, not hundreds.
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
