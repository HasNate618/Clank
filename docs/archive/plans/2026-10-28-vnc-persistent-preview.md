# VNC Persistent Preview Implementation Plan

> **Archived 2026-11-13.** Implementation plan, carried out. Kept for the task breakdown and the gates it defined; the code is the record now.
> Current behaviour: [`README.md`](../../../README.md) · agent rules: [`AGENTS.md`](../../../AGENTS.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the one-shot PNG preview loop with a persistent 1280x720, 30 fps Weston VNC stream viewed through noVNC, and raise the game cadence from 20 Hz to 60 Hz.

**Architecture:** Weston runs with its VNC backend on `127.0.0.1:5901`; `websockify` proxies that to `127.0.0.1:6080`; the dashboard serves noVNC core assets and the browser connects view-only. PAM authenticates the viewer. The Monado null compositor is patched to 60 Hz. Runtime screenshot stills are removed because Weston powers the VNC output off when the last client disconnects.

**Tech Stack:** Python 3.12 stdlib, Weston 16 VNC backend, Neat VNC, noVNC 1.7.0, websockify 0.13.0, Nix flakes, NixOS PAM, Monado.

---

## Current tree notes

- `flake.nix` is staged/new in the main checkout; `README.md` and `run.sh` are modified but unstaged.
- The package directory is `clank/`, not `vrbot/`; earlier notes using `vrbot` are stale.
- Do not discard the existing `flake.nix`, `run.sh`, or `README.md` changes.
- Do not commit unless explicitly requested.

---

### Task 0: Baseline

**Files:** none.

- [ ] Record `git status --short` and confirm the existing changes.
- [ ] Run `python3 -m unittest discover -s tests -t .`.
- [ ] Confirm the live tree has no preview refactor yet with `rg -n "PreviewCapture|fps_actual|/api/frame.png|/preview.png" clank tests`.

---

### Task 1: Config fields and validation

**Files:**
- Modify: `clank/config.py`
- Modify: `config.example.toml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

Add to `ConfigTests`:

```python
def test_vnc_defaults(self) -> None:
    cfg = load_config(None, env={})
    self.assertEqual(cfg.preview.fps, 30.0)
    self.assertEqual(cfg.preview.vnc_port, 5901)
    self.assertEqual(cfg.preview.proxy_port, 6080)
    self.assertEqual(cfg.paths.novnc_bin, "novnc")
    self.assertEqual(cfg.paths.websockify_bin, "websockify")

def test_vnc_ports_must_differ(self) -> None:
    cfg = load_config(None, env={})
    with self.assertRaises(ConfigError):
        validate_config(
            replace(cfg, preview=replace(cfg.preview, proxy_port=cfg.preview.vnc_port))
        )

def test_vnc_ports_are_bounded(self) -> None:
    cfg = load_config(None, env={})
    for key in ("vnc_port", "proxy_port"):
        with self.subTest(key=key):
            with self.assertRaises(ConfigError):
                validate_config(replace(cfg, preview=replace(cfg.preview, **{key: 0})))
            with self.assertRaises(ConfigError):
                validate_config(replace(cfg, preview=replace(cfg.preview, **{key: 70000})))

def test_novnc_web_root_is_derived(self) -> None:
    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw)
        bin_dir = base / "novnc" / "bin"
        web = base / "novnc" / "share" / "webapps" / "novnc"
        (web / "core").mkdir(parents=True)
        (web / "core" / "rfb.js").write_text("")
        bin_dir.mkdir(parents=True)
        tool = bin_dir / "novnc"
        tool.write_text("")
        tool.chmod(0o755)
        cfg = load_config(None, env={"CLANK_PATHS_NOVNC_BIN": str(tool)})
        self.assertEqual(cfg.paths.novnc_web_root, str(web))
```

- [ ] **Step 2: Run to verify failure**

```bash
python3 -m unittest tests.test_config -v
```

- [ ] **Step 3: Implement**

Add `novnc_bin`, `websockify_bin`, `novnc_web_root` to `PathsConfig`; add `vnc_port`, `proxy_port` to `PreviewConfig`.

Add defaults:

```python
"novnc_bin": "novnc",
"websockify_bin": "websockify",
"novnc_web_root": "",
```

and under `preview`:

```python
"fps": 30.0,
"vnc_port": 5901,
"proxy_port": 6080,
```

Validate ports in `_validate`; derive `novnc_web_root` from the resolved `novnc` executable; add noVNC and websockify to `missing_requirements`.

Update `config.example.toml` with the new paths and preview ports.

- [ ] **Step 4: Verify**

```bash
python3 -m unittest tests.test_config -v
```

---

### Task 2: Weston VNC and websockify lifecycle

**Files:**
- Modify: `clank/orchestrator.py`
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: Write failing tests**

Remove `FakePreview`; update `FakeRunner` to record the Weston config file and support per-port conflicts. Add tests for VNC args, config refresh rate, proxy args, snapshot transport fields, and port conflict failure.

- [ ] **Step 2: Verify failure**

```bash
python3 -m unittest tests.test_orchestrator -v
```

- [ ] **Step 3: Implement**

Add `proxy` to `LOG_FILES`. Remove `preview_factory`, `_preview`, `_start_preview`, and `preview_path`.

`_start_weston` writes a session-local `weston.ini` containing `[vnc]` and `refresh-rate=<preview.fps>`, checks the VNC port, starts:

```text
--backend=vnc --renderer=gl --width --height --socket --config
--xwayland --debug --log --address=127.0.0.1 --port
--disable-transport-layer-security
```

then waits for the Xwayland display and the VNC port.

Add `_start_proxy` starting `websockify 127.0.0.1:<proxy_port> 127.0.0.1:<vnc_port>`, waiting for the proxy port.

Startup order: monado, weston, proxy, pose, steam, game.

Snapshot preview becomes:

```python
{
    "mode": "vnc",
    "width": cfg.preview.width,
    "height": cfg.preview.height - cfg.preview.panel_height,
    "fps_target": cfg.preview.fps,
    "vnc_port": cfg.preview.vnc_port,
    "proxy_port": cfg.preview.proxy_port,
    "vnc_alive": bool(weston_handle and weston_handle.alive()),
    "proxy_alive": bool(proxy_handle and proxy_handle.alive()),
}
```

Add proxy to processes, logs, survivors, cleanup (before Weston), and monitor checks.

- [ ] **Step 4: Verify**

```bash
python3 -m unittest tests.test_orchestrator -v
```

---

### Task 3: Dashboard noVNC routes

**Files:**
- Modify: `clank/dashboard.py`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write failing tests**

Update `FakeManager` for the new preview fields and remove `preview_path`; add `/api/preview/config`, `/novnc/` asset serving, and traversal rejection tests.

- [ ] **Step 2: Verify failure**

```bash
python3 -m unittest tests.test_dashboard -v
```

- [ ] **Step 3: Implement**

Add `/api/preview/config` returning mode, ports, username, availability. Add `/novnc/<path>` serving resolved files under `cfg.paths.novnc_web_root` with containment checks and MIME types. Remove the multipart `/preview.png` stream, `/api/frame.png`, `_preview_stream`, `_peer_closed`, and unused imports.

- [ ] **Step 4: Verify**

```bash
python3 -m unittest tests.test_dashboard -v
```

---

### Task 4: Browser noVNC client

**Files:**
- Modify: `clank/static/index.html`
- Modify: `clank/static/app.js`
- Modify: `clank/static/style.css`

- [ ] **Step 1: HTML**

Replace `<img id="preview">` with `<canvas id="preview" tabindex="-1"></canvas>`. Add a visible auth form with username/password and a Connect button. Add `proxy` to the log dropdown.

- [ ] **Step 2: JS**

Add preview state (`previewRfb`, `previewConnecting`, `previewCreds`, `previewConfig`, `previewError`), dynamic import of `/novnc/core/rfb.js`, RFB setup with `viewOnly`, no viewport scaling, no session resize, no focus-on-click, connect/disconnect handlers, auth submit, and preview-box click filtering so auth-form clicks do not request pointer lock. Remove the old image preview retry/error logic.

- [ ] **Step 3: CSS**

Remove the vertical image flip. Add canvas positioning and auth-form styles.

- [ ] **Step 4: Verify syntax**

```bash
node --check clank/static/app.js
```

---

### Task 5: Flake and documentation

**Files:**
- Modify: `flake.nix`
- Modify: `README.md`

- [ ] **Step 1: Add packages**

Add `pkgs.novnc` and `pkgs.python3Packages.websockify` to the devshell.

- [ ] **Step 2: Update README**

Document VNC preview, PAM requirement, flake-provided packages, removed still endpoints, loopback-only stream, and the headless `--preview-smoke` tool check.

- [ ] **Step 3: Verify**

```bash
nix flake check
```

---

### Task 6: NixOS PAM and Monado 60 Hz

**Files:**
- Modify: `/etc/nixos/modules/nixos/psvr2-vr.nix`
- Mirror: `/home/nate/Projects/homelab-public/modules/nixos/psvr2-vr.nix`

- [ ] **Step 1: Add PAM**

```nix
security.pam.services.weston-remote-access = {
  text = ''
    auth sufficient pam_unix.so likeauth try_first_pass
    auth required pam_deny.so
    account required pam_unix.so
  '';
};
```

- [ ] **Step 2: Patch Monado**

Inside the `monado-psvr2` override:

```nix
postPatch = (oldAttrs.postPatch or "") + ''
  substituteInPlace src/xrt/compositor/null/null_compositor.c \
    --replace-fail 'U_TIME_1S_IN_NS / 20' 'U_TIME_1S_IN_NS / 60'
'';
```

- [ ] **Step 3: Rebuild**

```bash
sudo nixos-rebuild switch
```

- [ ] **Step 4: Apply the same edit to the homelab-public mirror.**

---

### Task 7: Verification

- [ ] Run:

```bash
python3 -m unittest discover -s tests -t .
node --check clank/static/app.js
nix flake check
git diff --check
```

- [ ] Live check with `systemctl --user stop monado.socket`, `./run.sh`, start a session, authenticate, verify VNC canvas, no screenshot loop, 30 fps target, game 60 Hz, manual input, and clean stop. Restore with `systemctl --user start monado.socket`.
- [ ] Failure checks: wrong password, occupied ports, missing noVNC assets.
