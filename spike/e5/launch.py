#!/usr/bin/env python3
"""Run the dashboard with the Clank depth-stream Vulkan layer scoped to the game.

The layer manifest is written under ~/.local/state/Clank/depth-layer and the
dashboard is exec'd with an XDG_DATA_DIRS that pressure-vessel will import. The
manifest's enable_environment keys keep the layer out of the host Steam client
and out of the Wine steam.exe that launches the game.

`run.sh` runs this by default, so the dashboard gets the live colour preview
without the VNC fallback. `--no-depth` starts the plain dashboard instead, and
a missing layer library or compatdata falls back to the same thing with a
warning rather than refusing to start.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from clank.config import load_config  # noqa: E402


def _default_near() -> float:
    path = REPO / "research/evidence/depth-live/depth-calibration.json"
    try:
        return float(json.loads(path.read_text())["near_m"])
    except Exception:
        return 0.00992


def _config_path(args: argparse.Namespace) -> Path:
    """The config clank will load.

    An explicit --config wins and resolves against the cwd, exactly as clank
    resolves it. Without one the repository config.toml is used, so the
    launcher works from any directory.
    """
    if args.config:
        path = Path(args.config)
        return path if path.is_absolute() else Path.cwd() / path
    return REPO / "config.toml"


def _clank_argv(args: argparse.Namespace) -> list[str]:
    """The clank command line, with the launcher's own flags left out."""
    argv = [sys.executable, "-m", "clank"]
    if args.replace:
        argv.append("--replace")
    if args.config:
        argv += ["--config", args.config]
    if args.host:
        argv += ["--host", args.host]
    if args.port is not None:
        argv += ["--port", str(args.port)]
    if args.preview_smoke:
        argv.append("--preview-smoke")
    return argv


def _exec(argv: list[str], env: dict[str, str]) -> int:
    """Hand the process over to the dashboard; there is one call site on purpose."""
    os.execvpe(argv[0], argv, env)  # noqa: S606 - fixed argv, exec of sys.executable
    return 0


def _exec_plain(args: argparse.Namespace, argv: list[str]) -> int:
    """Start the dashboard without the layer, leaving the env untouched."""
    if args.dry_run:
        print(f"command: {' '.join(argv)}")
        return 0
    return _exec(argv, os.environ.copy())


def launch(args: argparse.Namespace) -> int:
    argv = _clank_argv(args)
    if args.no_depth:
        return _exec_plain(args, argv)

    cfg = load_config(_config_path(args), env=os.environ)
    compatdata = (
        Path.home()
        / ".local/share/Steam/steamapps/compatdata"
        / str(cfg.launch.proton_app_id)
    )
    layer_so = Path(
        os.environ.get(
            "DEPTH_STREAM_LAYER",
            REPO / "spike/e5/build/depth_stream_layer.so",
        )
    ).resolve()
    problems = []
    if not compatdata.exists():
        problems.append(f"compatdata path does not exist: {compatdata}")
    if not layer_so.is_file():
        problems.append(f"depth layer not built: {layer_so} (run spike/e5/build.sh)")
    if problems:
        print(
            "depth-stream-launch: "
            + "; ".join(problems)
            + "; starting the plain dashboard (the preview falls back to VNC)",
            file=sys.stderr,
        )
        return _exec_plain(args, argv)

    layer_root = Path.home() / ".local/state/Clank/depth-layer"
    layer_dir = layer_root / "vulkan/implicit_layer.d"
    layer_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "file_format_version": "1.1.2",
        "layer": {
            "name": "VK_LAYER_CLANK_depth_stream",
            "type": "GLOBAL",
            "library_path": str(layer_so),
            "api_version": "1.4.0",
            "implementation_version": "1",
            "description": "Clank 480x270 mono depth stream",
            "functions": {
                "vkGetInstanceProcAddr": "clank_depth_GetInstanceProcAddr",
                "vkGetDeviceProcAddr": "clank_depth_GetDeviceProcAddr",
                "vkNegotiateLoaderLayerInterfaceVersion": (
                    "clank_depth_NegotiateLoaderLayerInterfaceVersion"
                ),
            },
            "enable_environment": {
                "STEAM_COMPAT_DATA_PATH": str(compatdata),
                "GNUTLS_SYSTEM_PRIORITY_FILE": "/dev/null",
            },
            "disable_environment": {"DISABLE_DEPTH_STREAM_PATH": "1"},
        },
    }
    (layer_dir / "depth_stream.json").write_text(json.dumps(manifest, indent=2) + "\n")

    env = os.environ.copy()
    # renderdoc's share entry would import its layer too; this launcher only needs
    # the depth layer.
    parts = [
        part
        for part in env.get("XDG_DATA_DIRS", "").split(":")
        if part and "renderdoc" not in part
    ]
    env["XDG_DATA_DIRS"] = str(layer_root) + ":" + ":".join(parts)
    env["PRESSURE_VESSEL_IMPORT_VULKAN_LAYERS"] = "1"
    existing_disable = env.get("VK_LOADER_LAYERS_DISABLE", "")
    mnd = "VK_LAYER_MND_enable_timeline_semaphore"
    env["VK_LOADER_LAYERS_DISABLE"] = (
        f"{existing_disable}:{mnd}" if existing_disable else mnd
    )
    env.pop("ENABLE_VULKAN_RENDERDOC_CAPTURE", None)
    env.pop("VK_INSTANCE_LAYERS", None)
    env.pop("VK_LAYER_PATH", None)

    state = Path.home() / ".local/state/Clank"
    env["DEPTH_STREAM_PATH"] = str(state / "depth-stream.bin")
    env["DEPTH_STREAM_LOG"] = str(state / "depth-stream.log")
    env["DEPTH_STREAM_WIDTH"] = str(args.width)
    env["DEPTH_STREAM_HEIGHT"] = str(args.height)
    env["DEPTH_STREAM_EYE"] = str(args.eye)
    env["DEPTH_STREAM_NEAR"] = f"{args.near if args.near is not None else _default_near():.9f}"
    env["DEPTH_STREAM_COLOR"] = "0" if args.no_color else "1"
    env["DEPTH_STREAM_COLOR_SCALE"] = str(args.color_scale)
    env["DEPTH_STREAM_FPS"] = str(args.fps)

    if args.dry_run:
        print(f"layer manifest: {layer_dir / 'depth_stream.json'}")
        print(f"layer library: {layer_so}")
        print(f"XDG_DATA_DIRS[:96]: {env['XDG_DATA_DIRS'][:96]}")
        print(f"DEPTH_STREAM_PATH={env['DEPTH_STREAM_PATH']}")
        print(f"DEPTH_STREAM_NEAR={env['DEPTH_STREAM_NEAR']}")
        print(f"DEPTH_STREAM_COLOR_SCALE={env['DEPTH_STREAM_COLOR_SCALE']}")
        print(f"DEPTH_STREAM_FPS={env['DEPTH_STREAM_FPS']}")
        print(f"command: {' '.join(argv)}")
        return 0

    return _exec(argv, env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="depth-stream-launch")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--eye", type=int, default=0, choices=(0, 1))
    parser.add_argument("--near", type=float, default=None)
    parser.add_argument("--color-scale", type=int, default=2, choices=(1, 2, 3, 4))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="start the plain dashboard, without the layer (VNC preview only)",
    )
    # run.sh forwards these; resolving them here keeps one code path for both.
    parser.add_argument("--config", help="clank config path, resolved like clank's")
    parser.add_argument("--host", help="override dashboard host")
    parser.add_argument("--port", type=int, help="override dashboard port")
    parser.add_argument("--preview-smoke", action="store_true")
    args = parser.parse_args(argv)
    return launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
