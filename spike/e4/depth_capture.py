#!/usr/bin/env python3
"""Reproducible live depth captures from Job Simulator's Vulkan depth buffer.

The VNC/Weston path carries RGB only, so depth has to come from the game's own
Vulkan device. RenderDoc's layer is imported into the Proton pressure-vessel
container by putting a layer manifest on the host's XDG_DATA_DIRS and exporting
PRESSURE_VESSEL_IMPORT_VULKAN_LAYERS=1.

Two details make the launch reliable:

  * The manifest is gated on STEAM_COMPAT_DATA_PATH and on
    GNUTLS_SYSTEM_PRIORITY_FILE=/dev/null. The game process has both; the host
    Steam client and the Wine steam.exe that launches the game do not.
  * The dev shell also exports renderdoc's own share directory. pressure-vessel
    imports that system manifest too, which gives the container a second
    RenderDoc layer. That duplicate loads in steam.exe and stalls
    steam://rungameid. Drop renderdoc's share entries from XDG_DATA_DIRS so only
    the scoped manifest is imported.

Commands::

    # Run the dashboard with the layer available inside the game container.
    nix develop --command python3 spike/e4/depth_capture.py launch --replace

    # Show the live session and the per-process RenderDoc target-control ports.
    python3 spike/e4/depth_capture.py status

    # Trigger, copy and extract one capture from the running game.
    python3 spike/e4/depth_capture.py capture --out research/evidence/depth-live

    # Re-extract an .rdc that was already copied to the host.
    python3 spike/e4/depth_capture.py extract --capture /tmp/game.rdc --out /tmp/depth

The game stalls for the frame being captured. After the first NewCapture message
the tool freezes the RenderDoc-enabled Wine processes briefly, copies the game's
capture out of its private /tmp, deletes all temporary .rdc files (they live in
tmpfs and are ~375 MB each), and resumes them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import depth_post  # noqa: E402

RENDERDOC_PORT_MIN = 38900
RENDERDOC_PORT_MAX = 38999
TARGET_LINE = re.compile(r":(?P<port>\d{5})\s.*pid=(?P<pid>\d+)")


@dataclass
class Target:
    port: int
    pid: int
    name: str


def _load_config():
    from clank.config import load_config

    return load_config(REPO / "config.toml", env=os.environ)


def _qrenderdoc() -> str:
    path = shutil.which("qrenderdoc")
    if path is None:
        raise SystemExit(
            "qrenderdoc is not on PATH; run this inside `nix develop` "
            "(flake.nix now includes renderdoc)"
        )
    return path


def _qrenderdoc_env(**extra: str) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.update(extra)
    return env


def _run_qrenderdoc(script: Path, timeout: float, **extra: str) -> None:
    result = subprocess.run(
        [_qrenderdoc(), "--python", str(script)],
        env=_qrenderdoc_env(**extra),
        timeout=timeout,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{script.name} exited {result.returncode}")


def dashboard_state(cfg) -> dict | None:
    url = f"http://{cfg.dashboard.host}:{cfg.dashboard.port}/api/state"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError):
        return None


def game_pid(cfg) -> int | None:
    """The Wine process running the title, not the reaper/adverb wrappers.

    The game executable also appears in SteamLaunch and pressure-vessel command
    lines, so a plain substring search finds the reaper first. A process with its
    own RenderDoc target control is the one that actually owns the Vulkan device;
    prefer that and fall back to the first substring match.
    """
    exe = Path(cfg.paths.game_exe).name
    owned = {target.pid for target in renderdoc_targets()}
    first: int | None = None
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace"
            )
        except OSError:
            continue
        if exe not in cmdline or entry.name == str(os.getpid()):
            continue
        pid = int(entry.name)
        if first is None:
            first = pid
        if pid in owned:
            return pid
    return first


def renderdoc_targets() -> list[Target]:
    result = subprocess.run(
        ["ss", "-ltnp"], check=True, capture_output=True, text=True
    )
    targets: list[Target] = []
    for line in result.stdout.splitlines():
        match = TARGET_LINE.search(line)
        if match is None:
            continue
        port = int(match.group("port"))
        if not (RENDERDOC_PORT_MIN <= port <= RENDERDOC_PORT_MAX):
            continue
        name = ""
        name_match = re.search(r'users:\(\("([^"]+)"', line)
        if name_match:
            name = name_match.group(1)
        targets.append(Target(port=port, pid=int(match.group("pid")), name=name))
    return sorted(targets, key=lambda target: target.port)


def target_for_pid(pid: int) -> Target | None:
    return next((target for target in renderdoc_targets() if target.pid == pid), None)


def renderdoc_prefix() -> Path:
    prefix = Path(_qrenderdoc()).resolve().parent.parent
    manifest = prefix / "share/vulkan/implicit_layer.d/renderdoc_capture.json"
    if not manifest.exists():
        raise SystemExit(f"RenderDoc layer manifest not found at {manifest}")
    return prefix


def launch(args: argparse.Namespace) -> int:
    cfg = _load_config()
    compatdata = (
        Path.home()
        / ".local/share/Steam/steamapps/compatdata"
        / str(cfg.launch.proton_app_id)
    )
    if not compatdata.exists():
        raise SystemExit(f"compatdata path does not exist: {compatdata}")

    prefix = renderdoc_prefix()
    manifest = json.loads(
        (prefix / "share/vulkan/implicit_layer.d/renderdoc_capture.json").read_text()
    )
    # A unique name keeps the scoped copy independent of renderdoc's system
    # manifest even if one slips through the XDG filter below.
    manifest["layer"]["name"] = "VK_LAYER_RENDERDOC_Capture_Clank"
    # Two keys, both required:
    #   * STEAM_COMPAT_DATA_PATH keeps the host Steam client out (it lacks it);
    #   * GNUTLS_SYSTEM_PRIORITY_FILE=/dev/null is present in the game process but
    #     not in the Wine steam.exe that launches it. Without the second key the
    #     layer also loads in steam.exe and the game launch stalls.
    manifest["layer"]["enable_environment"] = {
        "STEAM_COMPAT_DATA_PATH": str(compatdata),
        "GNUTLS_SYSTEM_PRIORITY_FILE": "/dev/null",
    }

    # Do not use tempfile.mkdtemp here: under `nix develop` TMPDIR is a temporary
    # nix-shell directory that is removed when the wrapper exits, which happens
    # after the dashboard is exec'd. pressure-vessel then imports nothing and the
    # game gets no layer. A stable state directory survives for the session.
    layer_root = Path.home() / ".local/state/Clank/renderdoc-layer"
    layer_dir = layer_root / "vulkan/implicit_layer.d"
    layer_dir.mkdir(parents=True, exist_ok=True)
    (layer_dir / "renderdoc_capture.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    env = os.environ.copy()
    # The dev-shell XDG_DATA_DIRS contains renderdoc's own share directory. If pv
    # imports that manifest too, the container gets a second RenderDoc layer that
    # loads in steam.exe and stalls the game launch. Remove renderdoc's share
    # entries and prepend only our scoped layer root.
    parts = [
        part
        for part in env.get("XDG_DATA_DIRS", "").split(":")
        if part and "renderdoc" not in part
    ]
    env["XDG_DATA_DIRS"] = str(layer_root) + ":" + ":".join(parts)
    env["PRESSURE_VESSEL_IMPORT_VULKAN_LAYERS"] = "1"
    # Without this the game picks the host's ~/.config/openxr/1/active_runtime.json,
    # which points at a separately built Monado. That build still has Monado's
    # RenderDoc integration, so every xrBeginFrame/xrEndFrame starts/ends a
    # capture and fills the container's RAM-backed /tmp. Point the loader at the
    # same patched build the service runs (the flake compiles the integration
    # out with -DXRT_FEATURE_RENDERDOC=OFF).
    if "XR_RUNTIME_JSON" not in env:
        monado_bin = env.get("CLANK_PATHS_MONADO_BIN")
        if monado_bin:
            runtime_json = (
                Path(monado_bin).resolve().parent.parent
                / "share/openxr/1/openxr_monado.json"
            )
            if runtime_json.is_file():
                env["XR_RUNTIME_JSON"] = str(runtime_json)
    existing_disable = env.get("VK_LOADER_LAYERS_DISABLE", "")
    mnd = "VK_LAYER_MND_enable_timeline_semaphore"
    env["VK_LOADER_LAYERS_DISABLE"] = (
        f"{existing_disable}:{mnd}" if existing_disable else mnd
    )
    # Do NOT set ENABLE_VULKAN_RENDERDOC_CAPTURE or VK_INSTANCE_LAYERS here. The
    # layer is activated by STEAM_COMPAT_DATA_PATH inside the container only.
    env.pop("ENABLE_VULKAN_RENDERDOC_CAPTURE", None)
    env.pop("VK_INSTANCE_LAYERS", None)
    env.pop("VK_LAYER_PATH", None)

    argv = [sys.executable, "-m", "clank", *args.clank_args]
    if args.replace:
        argv.append("--replace")
    if args.dry_run:
        print(f"layer manifest: {layer_dir / 'renderdoc_capture.json'}")
        print(f"XDG_DATA_DIRS[:80]: {env['XDG_DATA_DIRS'][:80]}")
        print(f"PRESSURE_VESSEL_IMPORT_VULKAN_LAYERS={env['PRESSURE_VESSEL_IMPORT_VULKAN_LAYERS']}")
        print(f"VK_LOADER_LAYERS_DISABLE={env['VK_LOADER_LAYERS_DISABLE']}")
        print(f"XR_RUNTIME_JSON={env.get('XR_RUNTIME_JSON')}")
        print(f"command: {' '.join(argv)}")
        return 0
    os.execvpe(sys.executable, argv, env)
    return 0


def status(args: argparse.Namespace) -> int:
    cfg = _load_config()
    state = dashboard_state(cfg)
    if state is None:
        print(f"dashboard: not running at {cfg.dashboard.host}:{cfg.dashboard.port}")
    else:
        processes = state.get("processes", {})
        print(
            f"dashboard: {state.get('state')} "
            f"game={processes.get('game', {}).get('alive')} "
            f"vnc={state.get('preview', {}).get('vnc_alive')} "
            f"proxy={state.get('preview', {}).get('proxy_alive')}"
        )
    pid = game_pid(cfg)
    print(f"game pid: {pid}")
    targets = renderdoc_targets()
    if not targets:
        print("RenderDoc target controls: none (launch with `launch` first)")
    for target in targets:
        marker = " <- game" if target.pid == pid else ""
        print(f"  {target.port}  pid={target.pid}  {target.name}{marker}")
    return 0


def _copy_remote_capture(pid: int, remote_path: str, destination: Path) -> Path:
    source = Path(f"/proc/{pid}/root{remote_path}")
    if not source.exists():
        raise RuntimeError(f"capture is not visible at {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def _freeze(targets: list[Target]) -> list[int]:
    stopped = []
    for target in targets:
        try:
            os.kill(target.pid, signal.SIGSTOP)
            stopped.append(target.pid)
        except ProcessLookupError:
            pass
    return stopped


def _resume(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass


def _delete_temp_captures(pid: int) -> None:
    root = Path(f"/proc/{pid}/root/tmp/RenderDoc")
    if not root.is_dir():
        return
    for path in root.glob("*.rdc"):
        try:
            path.unlink()
        except OSError:
            pass


def _remote_capture_dir(pid: int) -> Path:
    return Path(f"/proc/{pid}/root/tmp/RenderDoc")


def _wait_for_completed_capture(pid: int, timeout: float) -> str | None:
    """Wait for a .rdc in the target's tmpfs whose size has stopped growing.

    A trigger on DXVK can write several files, and NewCapture messages are not
    reliable, so watch the directory rather than the target-control stream.
    """
    root = _remote_capture_dir(pid)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not Path(f"/proc/{pid}").exists():
            return None
        files = [path for path in root.glob("*.rdc") if path.is_file()]
        if files:
            newest = max(files, key=lambda path: path.stat().st_mtime)
            try:
                first = newest.stat().st_size
            except OSError:
                first = -1
            time.sleep(1.2)
            try:
                second = newest.stat().st_size if newest.exists() else -1
            except OSError:
                second = -1
            if first > 0 and first == second:
                return f"/tmp/RenderDoc/{newest.name}"
        time.sleep(0.5)
    return None


def capture(args: argparse.Namespace) -> int:
    cfg = _load_config()
    state = dashboard_state(cfg)
    if state is None or state.get("state") != "running":
        raise SystemExit("dashboard is not running; start it with `launch` first")
    pid = game_pid(cfg)
    if pid is None:
        raise SystemExit("game process not found; is the session running?")
    target = target_for_pid(pid)
    if target is None:
        raise SystemExit(
            "the game has no RenderDoc target control; relaunch the dashboard "
            "with `depth_capture.py launch --replace`"
        )

    outdir = Path(args.out).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    # Start from an empty tmpfs so the newest file is certainly the one this
    # trigger produced, not an auto-capture left from launch.
    _delete_temp_captures(pid)
    trigger_out = outdir / "trigger.json"
    with tempfile.TemporaryDirectory(prefix="clank-e4-trigger-") as work:
        trigger_script = Path(work) / "rd_trigger.py"
        shutil.copyfile(Path(__file__).with_name("rd_trigger.py"), trigger_script)
        _run_qrenderdoc(
            trigger_script,
            args.timeout,
            RD_PORT=str(target.port),
            RD_TRIGGER_OUT=str(trigger_out),
            RD_TIMEOUT_SEC=str(args.timeout),
        )
    info = json.loads(trigger_out.read_text())
    if "error" in info:
        raise SystemExit(f"trigger failed: {info['error']}")
    remote_path = _wait_for_completed_capture(pid, min(args.timeout, 120.0))
    if remote_path is None:
        _delete_temp_captures(pid)
        raise SystemExit("no completed capture appeared in the target tmpfs")

    capture_path = outdir / "capture.rdc"
    # The file was already confirmed complete by _wait_for_completed_capture, so
    # no SIGSTOP is needed. Freezing the game around a copy proved to destabilise
    # it, and it is risky to stop a Vulkan process mid-submit.
    try:
        _copy_remote_capture(pid, remote_path, capture_path)
    finally:
        _delete_temp_captures(pid)
        # The layer can keep writing further frames after a trigger. Sweep after
        # a short settle or the tmpfs fills with hundreds of MB per frame.
        time.sleep(1.5)
        _delete_temp_captures(pid)
    # The capture path itself may be gone now; keep the trigger metadata.
    (outdir / "trigger.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"capture: {capture_path} ({capture_path.stat().st_size} bytes)")

    return extract_path(cfg, capture_path, outdir, args)


def extract_path(
    cfg, capture_path: Path, outdir: Path, args: argparse.Namespace
) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.work).resolve() if args.work else Path(
        tempfile.mkdtemp(prefix="clank-e4-extract-")
    )
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        script = Path(__file__).with_name("rd_extract.py")
        _run_qrenderdoc(
            script,
            args.timeout,
            RD_CAPTURE=str(capture_path),
            RD_WORKDIR=str(workdir),
            RD_STEREO_WIDTH=str(cfg.preview.width * 2),
        )
        error = workdir / "error.txt"
        if error.exists():
            raise SystemExit(f"extract failed:\n{error.read_text()}")
        calibration = depth_post.postprocess(
            workdir, outdir, near_override=args.near
        )
        print(
            "depth: "
            f"{calibration['metric_min_m']:.3f}..{calibration['metric_max_m']:.3f} m "
            f"(median {calibration['metric_median_m']:.3f} m, "
            f"near {calibration['near_m']*1000:.2f} mm, "
            f"{calibration['point_count']} points)"
        )
        print(f"artifacts: {outdir}")
        return 0
    finally:
        if not args.keep_work:
            shutil.rmtree(workdir, ignore_errors=True)


def extract(args: argparse.Namespace) -> int:
    cfg = _load_config()
    return extract_path(cfg, Path(args.capture).resolve(), Path(args.out).resolve(), args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="depth_capture")
    sub = parser.add_subparsers(dest="command", required=True)

    launch_parser = sub.add_parser(
        "launch", help="run the dashboard with the RenderDoc layer scoped to the game"
    )
    launch_parser.add_argument("--replace", action="store_true")
    launch_parser.add_argument("--dry-run", action="store_true")
    launch_parser.add_argument("clank_args", nargs=argparse.REMAINDER)
    launch_parser.set_defaults(func=launch)

    status_parser = sub.add_parser("status", help="show session and RenderDoc ports")
    status_parser.set_defaults(func=status)

    def common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--out", default="research/evidence/depth-live")
        target.add_argument("--work", default=None, help="keep raw dumps here")
        target.add_argument("--keep-work", action="store_true")
        target.add_argument("--near", type=float, default=None, help="near plane in metres")
        target.add_argument("--timeout", type=float, default=600.0)

    capture_parser = sub.add_parser("capture", help="trigger and extract a live capture")
    common(capture_parser)
    capture_parser.add_argument(
        "--no-freeze",
        action="store_true",
        help="do not SIGSTOP other Wine processes while copying",
    )
    capture_parser.set_defaults(func=capture)

    extract_parser = sub.add_parser("extract", help="extract from an existing .rdc")
    extract_parser.add_argument("--capture", required=True)
    common(extract_parser)
    extract_parser.set_defaults(func=extract)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
