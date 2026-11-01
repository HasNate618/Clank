# Capturing the game view with no human in the loop

- Date: 2026-11-01, session `20261101T223603Z-27abb1` (Job Simulator under Proton +
  OpenComposite, Weston 16 VNC backend, Xwayland `:1`).
- Status: **solved.** Two transports were measured, neither needs a password.
  `[MEASURED]`

The premise this started from — "every path needs the operator's VNC password" —
is wrong, and the stale claim that `weston-screenshooter` writes nothing is also
wrong. What it does is worse-behaved than either: it ignores its output argument,
writes a timestamped PNG into the current directory, returns **0** even when the
frame is blank, and **blocks** when the VNC output has been painted but no
authenticated VNC client is attached. All of that was measured below.

## 1. The two working transports

### (a) Compositor screenshot — production path, no password

```bash
XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=clank-27abb1 \
  nix develop . -c weston-screenshooter
```

`weston-screenshooter` is a native Wayland client: it asks the compositor for the
output, so there is no RFB, no noVNC, and no credential anywhere. Measured on the
live session, five consecutive shots: **1920x1080 RGB PNG, 1.41-1.42 MB, 1.27-1.31 s
through `nix develop`, 0.39-0.45 s with the binary invoked directly.** `[MEASURED]`

The production wrapper for this is `clank/capture.py::capture_frame`, which pins the
two traps: the tool ignores argv and writes into cwd, so it runs in a fresh
directory; and it may exit 0 with a non-PNG/blank file, so the result is checked.
`clank.preview.PreviewCapture.capture_once` now uses it.

```python
python3 -c 'from clank.capture import capture_frame; print(capture_frame(
  screenshooter="/nix/store/.../bin/weston-screenshooter",
  runtime_dir="/run/user/1000", wayland_display="clank-27abb1",
  work_dir="/tmp/libcap/work", destination="/tmp/libcap/game-raw.png"))'
```

Measured with the plain system `python3` (no PIL, no venv): 1,415,993-byte PNG,
1920x1080, scratch directory empty afterwards. `[MEASURED]`

**Caveat, measured:** on this VNC backend the capture only completes while an
authenticated VNC client is attached. The live session has the operator's browser
connected through websockify, which is why the five shots above work. A throwaway
Weston on `:5911-5915` with an animating `weston-flower` and no VNC client blocked
`weston-screenshooter` for the full 10 s timeout in every attempt (6 timeouts
across five probe configurations); with no painted surface at all it exited 0 in
28 ms with a 1,289-byte blank PNG and
`Output capture error: unauthorized`. `[MEASURED]` The auth itself is against a
local account (`VNC: wrong user ''`, and `spike/vnc_grab.py` confirms RSA-AES256 /
PAM), so an unattended session cannot create its own client without the password.

### (b) X11 grab of the game window — no VNC at all, upright

The game is an X client on the rootless Xwayland display (`:1`). `XGetImage` on the
`Job Simulator` window returns the frame directly from Xwayland:

```bash
spike/e3/.venv/bin/python spike/wbc_game/x11_grab.py \
  --display :1 --out /tmp/game.png --check
```

Measured: window `0x4a00001 "Job Simulator"`, 1920x1080, 32 bpp BGRX
(`0xff0000/0xff00/0xff`, LSBFirst), ~1.0-1.5 s including the pure-Python decode.
Result: 1,399,745-byte PNG, luma std 58.6, 136,356 distinct colours, horizontal /
vertical neighbour correlation 0.994 / 0.989. `[MEASURED]`

The grab is **upright** and contains only the game window, not Steam's UI and not
whatever the compositor has on top. That last part is useful on its own: the
screenshooter captures the compositor's z-order, so a Steam window forward means a
Steam frame. `[MEASURED]`

It stays a spike, not a production callable, for two honest reasons: NixOS does not
put `libX11` on the default loader path (the script globs the store, a production
version wants a config key), and Xlib's default error handler calls `exit(1)` on a
`BadMatch` (the spike installs a counting handler plus `XSync`; a long-lived
orchestrator needs that to raise, not exit). The decode and PPM writing are the
easy part and are unit-testable; the ctypes transport is not tested by the suite.

## 2. Evidence the frames are real scenes

| capture | size | mean RGB | luma std | distinct colours | corr h / v |
| --- | ---: | --- | ---: | ---: | --- |
| screenshooter, flipped upright | 1,395,869 B | 94.2 / 149.2 / 159.3 | 58.5 | 135,944 | 0.9941 / 0.9895 |
| X11 window grab | 1,399,745 B | 94.0 / 148.6 / 158.7 | 58.6 | 136,356 | 0.9941 / 0.9894 |
| blank VNC frame (no client) | 1,289 B | — | — | — | — |

A cleared window is a few hundred bytes to a few kB with almost no colour variety;
these are 1.4 MB, ~136k colours, and neighbouring pixels are 99% correlated - the
signature of a shaded 3D scene, not noise or a flat fill. `[MEASURED]`

Two captures 1 s apart differ on 7.2% of pixels (mean 2.4, max 194), from the hub's
CRT static and animated props - so these are live frames, not a cached still.
`[MEASURED]`

The two transports agree with each other: mean absolute difference **1.45 levels**
(X11 grab vs flipped screenshooter), while the same screenshooter frame in its raw
mirrored orientation differs by **71.2 levels**. That is a direct check that the
flip, not the scene, is what differs. `[MEASURED]`

## 3. Orientation: the compositor screenshot is vertically mirrored

`weston-screenshooter` output from this session is mirrored across the horizontal
axis: the "JOB SIMULATOR" signage is upside down but still reads left-to-right, and
the robot's right hand is on the right of the frame. A 180° camera roll would reverse
the reading direction, so this is a capture flip, not a game pose. `[MEASURED]`

The compositor says why, in its own log:

```text
GL version: OpenGL ES 3.2 NVIDIA 610.57.04
               glReadPixels supports y-flip: no
...
               screen capture uses y-flip: yes
```

The VNC/noVNC canvas the operator sees is upright, and the X11 grab is upright. So:
**flip a screenshooter frame before comparing it with a dashboard frame.** The
`spike/wbc/frames.py` helpers now say this, and `load_upright()` is the explicit
decode for absolute screen coordinates. `clank/` cannot decode PNG, so
`capture_frame` returns the raw file and documents the orientation instead of
silently guessing.

## 4. What was tried before, and why it failed

| path | outcome |
| --- | --- |
| `weston-screenshooter /tmp/x.png` | exits 0, writes `wayland-screenshot-<ts>.png` in **cwd**; the argument is ignored. The two stray `wayland-screenshot-2026-11-01_17-39-*.png` in the repo root are from this: the tool worked, the file was looked for in the wrong place. |
| `spike/vnc_grab.py --port 5901` | server offers RSA-AES256 (types 129/5/30), auth is PAM against a **local account**: `VNC: wrong user ''`. Not pursued, by constraint. |
| `ffmpeg -f x11grab` | this build has no `x11grab` input. |
| ImageMagick `import -display :1 -window root` | the earlier "cannot load libX11" failure did **not** reproduce: `import -window 0x4a00001` works from a plain shell and writes the same upright 1920x1080 scene. Use the game window, not root (Xwayland is `-rootless`). |
| `weston-debug -l` | scopes are log / scene-graph / timeline / libseat-debug / proto / vnc-backend / gl-renderer-* / xwm-wm-x11. **No pixel or screenshot scope**, so lead (a) is a dead end. |
| `spike/c0/capture_game.sh` | Superhot-era: finds a Superhot window via `hyprctl`, grabs the host desktop with `grim`. Wrong stack. |
| Weston VNC `--password` | **does not exist** in Weston 16.0.0. `weston --help` lists only `--vnc-tls-cert`, `--vnc-tls-key`, `--disable-transport-layer-security`. Auth is neatvnc + PAM, so "pass a password from config" is not a patch that can be made here. |

## 5. VNC-client dependency, measured

Each probe is a throwaway Weston VNC backend on its own port (never 5901), a
`weston-flower` client to force painting, and `weston-screenshooter` with a 10 s
timeout. The live session was not touched.

| probe | `--debug` | painted client | VNC client | result |
| --- | :---: | :---: | :---: | --- |
| live session | yes | game | yes (browser) | 5/5 captured, ~0.4 s |
| 1 | no | none | no | exit 0, 28 ms, 1,289 B blank, `Output capture error: unauthorized` |
| 2 | yes | none | no | blocked, timeout 10 s |
| 3 | yes | `weston-flower` | no | blocked, timeout 10 s |
| 4 | no | `weston-flower` | no | blocked, timeout 10 s (2/2) |
| 5 | no | `weston-flower` + raw TCP read of the RFB banner | no (unauthenticated) | blocked, timeout 10 s |
| 6 | yes | `weston-flower`, `--xwayland` like the live session | no | blocked, timeout 10 s - Xwayland is not the unblocker |

So an unauthenticated TCP connection is not enough; an authenticated VNC client
keeps the VNC output capture completing. This is the one thing standing between the
screenshooter path and a fresh unattended session.

## 6. Recommendation

1. **For measurements now:** use the X11 grab (`spike/wbc_game/x11_grab.py`). It is
   upright, window-only, and has no VNC dependency. It needs the `:1` display, which
   the orchestrator already records (`processes.weston.display`, from
   `_wait_weston_display`).
2. **For a production callable:** `clank.capture.capture_frame` is the
   screenshooter path, stdlib-only and unit-tested. It is honest about needing a VNC
   client on this backend.
3. **To promote X11 into `clank/`:** add a `paths.libx11` config key (owned elsewhere)
   and a `capture_x11_frame(display=..., destination=..., window_title=...)` that
   installs an Xlib error handler, calls `XSync`, and raises `CaptureError` on
   `BadMatch`. Keep the mask/byte-order decode as a pure function and unit-test it
   with synthetic bytes. No orchestrator patch is needed beyond reading the display
   it already stores.

## 7. Unmeasured / residual risk

- The live session was not tested with the operator's VNC client detached; that
  would disturb it. The probe table is the evidence for the dependency.
- X11 grab of an unmapped or fully obscured window was not tested; the spike's error
  handler is there so that case fails instead of exiting.
- `[ASSUMPTION]` Xwayland keeps updating the game window's pixmap when no VNC client
  consumes the compositor output. The X11 grab is current while the browser is
  attached (it matches the screenshooter frame), but "no VNC client at all" was only
  tested on a throwaway compositor with a different client.
- One game, one resolution (1920x1080), one session. Window title is the selector;
  a localized title would need a class/`steam_app_` match instead.
- The flip is tied to this renderer (`screen capture uses y-flip: yes`). A pixman or
  headless renderer may not flip; `spike/wbc_game/capture_game.py` reports whether
  it flipped, and `--no-flip` keeps the raw orientation.
