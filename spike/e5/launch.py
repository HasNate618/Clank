#!/usr/bin/env python3
"""Run the dashboard with the Clank depth-stream Vulkan layer scoped to the game.

The layer manifest is written under ~/.local/state/Clank/depth-layer and the
dashboard is exec'd with an XDG_DATA_DIRS that pressure-vessel will import. The
manifest's enable_environment keys keep the layer out of the host Steam client
and out of the Wine steam.exe that launches the game.
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


def launch(args: argparse.Namespace) -> int:
    cfg = load_config(REPO / "config.toml", env=os.environ)
    compatdata = (
        Path.home()
        / ".local/share/Steam/steamapps/compatdata"
        / str(cfg.launch.proton_app_id)
    )
    if not compatdata.exists():
        raise SystemExit(f"compatdata path does not exist: {compatdata}")

    layer_so = Path(
        os.environ.get(
            "DEPTH_STREAM_LAYER",
            REPO / "spike/e5/build/depth_stream_layer.so",
        )
    ).resolve()
    if not layer_so.is_file():
        raise SystemExit(f"depth layer not built: {layer_so} (run spike/e5/build.sh)")

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

    argv = [sys.executable, "-m", "clank"]
    if args.replace:
        argv.append("--replace")
    argv += args.clank_args

    if args.dry_run:
        print(f"layer manifest: {layer_dir / 'depth_stream.json'}")
        print(f"layer library: {layer_so}")
        print(f"XDG_DATA_DIRS[:96]: {env['XDG_DATA_DIRS'][:96]}")
        print(f"DEPTH_STREAM_PATH={env['DEPTH_STREAM_PATH']}")
        print(f"DEPTH_STREAM_NEAR={env['DEPTH_STREAM_NEAR']}")
        print(f"command: {' '.join(argv)}")
        return 0

    os.execvpe(sys.executable, argv, env)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="depth-stream-launch")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--eye", type=int, default=0, choices=(0, 1))
    parser.add_argument("--near", type=float, default=None)
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("clank_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    return launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
