from __future__ import annotations

import math
import os
import shutil
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when config.toml or a CLANK_ override is invalid."""


@dataclass(frozen=True)
class PathsConfig:
    monado_bin: str
    weston_bin: str
    weston_screenshooter_bin: str
    weston_debug_bin: str
    steam_bin: str
    xdotool_bin: str
    novnc_bin: str
    websockify_bin: str
    novnc_web_root: str
    game_exe: str
    game_player_log: str
    #: Directories wiped before every run, relative to the Proton prefix
    #: (<compatdata>/<proton_app_id>/pfx). Absolute entries and `..` are refused at
    #: load: a wipe must never reach outside the prefix.
    game_save_dirs: tuple[str, ...]
    vr_override: str
    pose_client: str
    openxr_active_runtime: str
    session_root: str


@dataclass(frozen=True)
class LaunchConfig:
    shortcut_id: str
    proton_app_id: str
    steam_extra_args: tuple[str, ...]
    steam_settle_sec: float
    steam_ready_timeout_sec: float
    game_process_timeout_sec: float
    openxr_client_timeout_sec: float
    openxr_client_name: str
    viewport_scale_percentage: float
    game_window: str
    startup_timeout_sec: float


@dataclass(frozen=True)
class PreviewConfig:
    width: int
    height: int
    panel_height: int
    fps: float
    vnc_port: int
    proxy_port: int


@dataclass(frozen=True)
class BodyConfig:
    """Third-person MuJoCo view, run beside the C0 session."""

    enabled: bool
    autostart: bool
    python: str
    script: str
    fps: float
    width: int
    height: int
    reel: bool
    render: bool
    views: tuple
    #: The robot's heading in the virtual room, in degrees, positive toward its
    #: left. The desk needs a 90 deg left turn and this body cannot twist that far
    #: (a commanded left look reaches -28.6 deg), so the heading is chosen rather
    #: than walked. 0.0 changes nothing.
    yaw_offset_deg: float = 0.0
    #: The headset's field of view we DECLARE to the runtime, full angles in degrees.
    #: Undeclared by default, so the remote driver falls back to its own default and
    #: the view looks narrow. Both or neither; 0.0 means undeclared. This is a claim
    #: about a headset that does not exist - the value is ours to choose, which is the
    #: point of emulating one.
    fov_h_deg: float = 0.0
    fov_v_deg: float = 0.0
    #: How big the emulated player is reported to the game, as a multiple of the robot.
    #: The game's room is laid out for an adult and the G1 is 1.32 m, so at 1.0 every
    #: object sits about 1.4x further than the arm travels. 1.43 is the measured adult
    #: ratio and the value `research/g1-scale.md` reports; 1.0 changes nothing. This
    #: scales what the game is TOLD, not the robot: see the pivot below.
    player_scale: float = 1.0
    #: The point, in the body's world frame, the scale is measured from. The resting
    #: wrist is the measured choice: travel = |object - pivot| / scale, so a pivot at the
    #: floor lifts the hand faster than it lengthens the arm and makes a reach longer.
    player_pivot: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Whether the camera is scaled with the hands. Measured both ways: with the head
    #: left alone the camera stays where the depth measurements were taken and only the
    #: hands grow; with it scaled the game's whole view moves with the player.
    player_scale_head: bool = True


@dataclass(frozen=True)
class DashboardConfig:
    host: str
    port: int


@dataclass(frozen=True)
class TimeoutsConfig:
    monado_startup_sec: float
    weston_startup_sec: float
    pose_handshake_sec: float
    stop_sec: float


@dataclass(frozen=True)
class RunConfig:
    """One attempt: what resets it, what runs it, and what it needs first."""

    brain_python: str
    brain_script: str
    brain_duration_sec: float
    game_restart_timeout_sec: float
    stop_timeout_sec: float
    rest_ticks: int
    body_timeout_sec: float
    required_model: str
    model_timeout_sec: float
    model_autostart: bool
    model_start_cmd: str


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    launch: LaunchConfig
    preview: PreviewConfig
    dashboard: DashboardConfig
    timeouts: TimeoutsConfig
    run: RunConfig
    body: BodyConfig


DEFAULTS: dict[str, dict[str, Any]] = {
    "paths": {
        "monado_bin": "/run/wrappers/bin/monado-service",
        "weston_bin": "weston",
        "weston_screenshooter_bin": "weston-screenshooter",
        "weston_debug_bin": "weston-debug",
        "steam_bin": os.path.expanduser("~/.local/bin/steam"),
        "xdotool_bin": "xdotool",
        "novnc_bin": "novnc",
        "websockify_bin": "websockify",
        "novnc_web_root": "",
        "game_exe": "JobSimulator.exe",
        # Unity log relative to .../pfx/drive_c/users/steamuser. The dashboard links
        # it. Job Simulator writes output_log.txt; most Unity titles write Player.log.
        "game_player_log": "AppData/LocalLow/Owlchemy Labs/Job Simulator/output_log.txt",
        # Wiped before every run so the job starts at its first frame. Relative to the
        # Proton prefix. The Unity log beside them is deliberately not listed: a failed
        # run is diagnosed from it.
        "game_save_dirs": (
            "drive_c/users/steamuser/AppData/LocalLow/Owlchemy Labs/Job Simulator/User",
            "drive_c/users/steamuser/AppData/LocalLow/Unity/Owlchemy Labs_Job Simulator",
        ),
        # Directory holding the OpenVR shim's bin/linux64/vrclient.so, exported to
        # Proton as VR_OVERRIDE. Empty leaves ~/.config/openvr/openvrpaths.vrpath in
        # charge. The devShell points this at nixpkgs' opencomposite.
        "vr_override": "",
        "pose_client": "spike/c0/c0_client.py",
        "openxr_active_runtime": os.path.expanduser("~/.config/openxr/1/active_runtime.json"),
        "session_root": os.path.expanduser("~/.local/state/Clank/sessions"),
    },
    "launch": {
        "shortcut_id": "14670405696621641728",
        "proton_app_id": "3415720001",
        "steam_extra_args": ("-cef-disable-gpu",),
        "steam_settle_sec": 60.0,
        "steam_ready_timeout_sec": 120.0,
        "game_process_timeout_sec": 180.0,
        "openxr_client_timeout_sec": 120.0,
        # The Monado application_name the game connects as, matched as a prefix of
        # the "application_name: '...'" line. Job Simulator through the OpenComposite
        # shim reports OpenComposite_wine64-preloader, so "OpenComposite" matches.
        "openxr_client_name": "OpenComposite",
        # The remote HMD reports 1920x1080 per eye (r_hmd.c in the flake widens its
        # display to 3840x1080) and the OpenXR runtime scales each view by
        # OXR_VIEWPORT_SCALE_PERCENTAGE. 100 is that native 1080p; 200 supersamples
        # to 3840x2160, which keeps the 1920x1080 mirror crisp.
        "viewport_scale_percentage": 100.0,
        # xdotool --name pattern used to fullscreen the game window. Empty disables.
        "game_window": "Job Simulator",
        "startup_timeout_sec": 300.0,
    },
    "body": {
        # Off by default: the body view needs the SONIC venv and weights, which
        # a bare checkout will not have. Enable it for a "regular session" where
        # you want the robot and the game on screen at the same time.
        "enabled": False,
        "autostart": False,
        "python": "spike/e3/.venv/bin/python",
        "script": "spike/wbc/body_view.py",
        "fps": 10.0,
        "width": 560,
        "height": 420,
        # Cycle the verb reel instead of holding one pose, so the tile shows motion.
        "reel": True,
        # false = physics and verbs only, no third-person image. Needs no GPU,
        # so the body stays drivable on a machine with the graphics stack off.
        "render": True,
        # Camera angles rendered together, first is the primary tile.
        "views": ("three_quarter", "front", "side"),
        # The robot's heading in the virtual room, degrees, positive toward its left.
        # The desk needs a 90 deg left turn, which is far outside what this body can
        # twist (measured: -28.6 deg left, +54.3 right), so the heading is chosen
        # rather than walked. 0.0 leaves every earlier measurement untouched.
        "yaw_offset_deg": 0.0,
        # Field of view declared to the runtime, full angles in degrees. 0.0 = declare
        # nothing, leaving the remote driver's own default in force.
        "fov_h_deg": 0.0,
        "fov_v_deg": 0.0,
        # How big the emulated player is reported to the game; 1.0 changes nothing.
        "player_scale": 1.0,
        "player_pivot": (0.0, 0.0, 0.0),
        "player_scale_head": True,
    },
    "preview": {
        "width": 1280,
        "height": 720,
        "panel_height": 32,
        "fps": 30.0,
        "vnc_port": 5901,
        "proxy_port": 6080,
    },
    "dashboard": {
        "host": "127.0.0.1",
        "port": 8765,
    },
    "timeouts": {
        "monado_startup_sec": 20.0,
        "weston_startup_sec": 20.0,
        # Long enough to outlast a pose client's own handover retries (20 s):
        # the driver serves one client at a time, so a switch can wait that long.
        "pose_handshake_sec": 25.0,
        "stop_sec": 60.0,
    },
    "run": {
        # The brain loop runs in its own venv (torch), like the body view. Both paths
        # stay relative and are resolved by the dashboard's cwd, which is the repo root.
        "brain_python": "spike/e3/.venv/bin/python",
        "brain_script": "spike/wbc_game/auto.py",
        # What ends a run on its own: auto.py's own default, 15 minutes.
        "brain_duration_sec": 900.0,
        # A cold game relaunch: Steam must notice the process is gone first.
        "game_restart_timeout_sec": 180.0,
        "stop_timeout_sec": 10.0,
        # How many `rest` commands the body reset sends before the run starts. The clip
        # answers in under a second and the command's TTL is short, so it is repeated.
        "rest_ticks": 8,
        # `BodyPreview.start()` returns when the process spawns; the sidecar needs
        # seconds to load MuJoCo before it binds its control socket.
        "body_timeout_sec": 120.0,
        # The planner must be serving this model before a run spends a game restart.
        "required_model": "LFM3-Omni",
        # A 2B Q8_0 load takes tens of seconds; that is a wait, not a failure.
        "model_timeout_sec": 180.0,
        # Off: the model, arguments and GPU flags are the operator's choice, so the
        # gate tells them what to start rather than starting one itself.
        "model_autostart": False,
        "model_start_cmd": "",
    },
}


def _coerce(value: Any, default: Any, where: str) -> Any:
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"{where} must be a boolean")
        return value
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where} must be an integer")
        return value
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where} must be a number")
        return float(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise ConfigError(f"{where} must be a string")
        return value
    if isinstance(default, (list, tuple)):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where} must be a list")
        if default and all(
            isinstance(item, (int, float)) and not isinstance(item, bool) for item in default
        ):
            # A numeric triple, e.g. a pivot: the default's own shape is the contract, so a
            # two-value "point" is refused here rather than reaching the relay.
            if len(value) != len(default):
                raise ConfigError(
                    f"{where} must be {len(default)} numbers, got {len(value)}"
                )
            if not all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in value
            ):
                raise ConfigError(f"{where} must be {len(default)} numbers")
            numbers = (float(value[0]), float(value[1]), float(value[2]))
            if len(numbers) == 3:  # noqa: SIM108 - explicit is the point here
                return numbers
            raise ConfigError(f"{where} must be three numbers")
        if not all(isinstance(item, str) for item in value):
            raise ConfigError(f"{where} must be a list of strings")
        return tuple(value)
    raise ConfigError(f"unsupported default for {where}")


def _parse_env_value(section: str, key: str, raw: str, default: Any) -> Any:
    where = f"{section}.{key}"
    try:
        if isinstance(default, bool):
            lowered = raw.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
            raise ConfigError(f"{where} must be a boolean")
        if isinstance(default, int):
            return int(raw)
        if isinstance(default, float):
            return float(raw)
        if isinstance(default, (list, tuple)):
            return tuple(raw.split())
        return raw
    except ValueError as exc:
        raise ConfigError(f"invalid value for {where}: {raw!r}") from exc


def _validate(cfg: Config) -> None:
    if cfg.preview.width <= 0 or cfg.preview.height <= 0:
        raise ConfigError("preview width and height must be positive")
    if cfg.preview.panel_height < 0 or cfg.preview.panel_height >= cfg.preview.height:
        raise ConfigError("preview.panel_height must be >= 0 and less than preview.height")
    if cfg.preview.fps <= 0 or not math.isfinite(cfg.preview.fps):
        raise ConfigError("preview.fps must be positive")
    if cfg.preview.fps != int(cfg.preview.fps):
        raise ConfigError("preview.fps must be a whole number of Hz")
    for field_name in ("vnc_port", "proxy_port"):
        value = getattr(cfg.preview, field_name)
        if not 1 <= value <= 65535:
            raise ConfigError(f"preview.{field_name} must be between 1 and 65535")
    if cfg.preview.vnc_port == cfg.preview.proxy_port:
        raise ConfigError("preview.vnc_port and preview.proxy_port must differ")
    if cfg.dashboard.port != 0 and cfg.dashboard.port in {
        cfg.preview.vnc_port,
        cfg.preview.proxy_port,
    }:
        raise ConfigError("preview ports must not equal dashboard.port")
    host = cfg.dashboard.host.strip()
    if not host:
        raise ConfigError("dashboard.host must not be empty")
    # S104 flags the wildcard literals, but this line is the guard that REFUSES them -
    # it raises rather than binding. The noqa is for the linter, not for the intent.
    if host in {"0.0.0.0", "::", "[::]"}:  # noqa: S104
        raise ConfigError("dashboard.host must be a specific address, not a wildcard")
    if not 0 <= cfg.dashboard.port <= 65535:
        raise ConfigError("dashboard.port must be between 0 and 65535")
    for name, angle in (("fov_h_deg", cfg.body.fov_h_deg),
                        ("fov_v_deg", cfg.body.fov_v_deg)):
        # 0.0 means "declare nothing", which is the documented default. Any other value
        # has to be a real angle: 180 deg is a hemisphere, not a headset.
        if angle and not 0.0 < angle < 180.0:
            raise ConfigError(
                f"body.{name} must be between 0 and 180 degrees, "
                "or 0 to leave the field of view undeclared"
            )
    if bool(cfg.body.fov_h_deg) != bool(cfg.body.fov_v_deg):
        # Refused here rather than at body-mode start, so the session never opens
        # with a field of view nobody intended.
        raise ConfigError(
            "body.fov_h_deg and body.fov_v_deg must be given together, or both 0"
        )
    if not math.isfinite(cfg.body.player_scale) or cfg.body.player_scale <= 0.0:
        # A zero or negative scale collapses the room to a point or mirrors it, and both
        # look like a tracking fault rather than the argument error they are.
        raise ConfigError(
            f"body.player_scale must be finite and positive, got {cfg.body.player_scale!r}"
        )
    pivot = tuple(cfg.body.player_pivot)
    if len(pivot) != 3 or not all(math.isfinite(float(c)) for c in pivot):
        raise ConfigError(
            f"body.player_pivot must be x,y,z, got {cfg.body.player_pivot!r}"
        )
    for field_name in (
        "steam_settle_sec",
        "steam_ready_timeout_sec",
        "game_process_timeout_sec",
        "openxr_client_timeout_sec",
        "startup_timeout_sec",
    ):
        value = getattr(cfg.launch, field_name)
        if value <= 0 or not math.isfinite(value):
            raise ConfigError(f"launch.{field_name} must be positive")
    scale = cfg.launch.viewport_scale_percentage
    if not math.isfinite(scale) or not 0.0 < scale <= 400.0:
        # A percentage, not a fraction: 100 is native and 400 is already 16x the
        # pixels per eye. Refused at load rather than surprising the GPU.
        raise ConfigError("launch.viewport_scale_percentage must be finite and in (0, 400]")
    for field_name in (
        "monado_startup_sec",
        "weston_startup_sec",
        "pose_handshake_sec",
        "stop_sec",
    ):
        value = getattr(cfg.timeouts, field_name)
        if value <= 0 or not math.isfinite(value):
            raise ConfigError(f"timeouts.{field_name} must be positive")
    for entry in cfg.paths.game_save_dirs:
        parts = Path(entry).parts
        # Relative and inside the prefix: a wipe is the one operation here that
        # deletes files, so the shape is checked before it can be resolved.
        if Path(entry).is_absolute() or ".." in parts or not parts:
            raise ConfigError(
                "paths.game_save_dirs entries must be relative to the Proton "
                f"prefix, got {entry!r}"
            )
    if cfg.run.brain_duration_sec <= 0 or not math.isfinite(cfg.run.brain_duration_sec):
        raise ConfigError("run.brain_duration_sec must be positive")
    for field_name in ("game_restart_timeout_sec", "stop_timeout_sec", "model_timeout_sec"):
        value = getattr(cfg.run, field_name)
        if value <= 0 or not math.isfinite(value):
            raise ConfigError(f"run.{field_name} must be positive")
    if cfg.run.rest_ticks < 0:
        raise ConfigError("run.rest_ticks must be >= 0")
    if cfg.run.body_timeout_sec <= 0:
        raise ConfigError("run.body_timeout_sec must be > 0")


def validate_config(cfg: Config) -> None:
    _validate(cfg)


def load_config(
    path: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    base_dir: Path | str | None = None,
) -> Config:
    base = Path.cwd()
    file_data: dict[str, Any] = {}
    candidate: Path | None = None
    if path is not None:
        candidate = Path(path)
        if not candidate.exists():
            raise ConfigError(f"config file not found: {candidate}")
        with candidate.open("rb") as handle:
            try:
                file_data = tomllib.load(handle)
            except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
                raise ConfigError(f"invalid TOML in {candidate}: {exc}") from exc

    if base_dir is not None:
        base = Path(base_dir)
    elif candidate is not None:
        base = candidate.parent

    unknown_sections = set(file_data) - set(DEFAULTS)
    if unknown_sections:
        raise ConfigError(f"unknown config sections: {', '.join(sorted(unknown_sections))}")

    merged = {section: dict(values) for section, values in DEFAULTS.items()}
    for section, values in file_data.items():
        if not isinstance(values, dict):
            raise ConfigError(f"{section} must be a table")
        unknown = set(values) - set(DEFAULTS[section])
        if unknown:
            raise ConfigError(f"unknown keys in {section}: {', '.join(sorted(unknown))}")
        for key, value in values.items():
            merged[section][key] = _coerce(value, DEFAULTS[section][key], f"{section}.{key}")

    for env_key, raw in (env or {}).items():
        if not env_key.startswith("CLANK_"):
            continue
        rest = env_key[len("CLANK_") :]
        if "_" not in rest:
            raise ConfigError(f"invalid environment override {env_key}")
        raw_section, raw_key = rest.split("_", 1)
        section = raw_section.lower()
        key = raw_key.lower()
        if section not in DEFAULTS or key not in DEFAULTS[section]:
            raise ConfigError(f"unknown environment override {env_key}")
        default = DEFAULTS[section][key]
        value = _parse_env_value(section, key, raw, default)
        merged[section][key] = _coerce(value, default, f"{section}.{key}")

    paths = merged["paths"]
    if os.sep in paths["weston_bin"]:
        parent = Path(paths["weston_bin"]).parent
        for key, sibling in (
            ("weston_screenshooter_bin", "weston-screenshooter"),
            ("weston_debug_bin", "weston-debug"),
        ):
            if paths[key] == DEFAULTS["paths"][key]:
                paths[key] = str(parent / sibling)

    for key in ("pose_client", "session_root"):
        candidate = Path(paths[key])
        if not candidate.is_absolute():
            paths[key] = str((base / candidate).resolve())

    if paths["novnc_web_root"]:
        candidate = Path(paths["novnc_web_root"])
        if not candidate.is_absolute():
            paths["novnc_web_root"] = str((base / candidate).resolve())

    # An explicit noVNC web root wins over the one derived from novnc_bin.
    if paths["novnc_web_root"] in ("", DEFAULTS["paths"]["novnc_web_root"]):
        resolved_novnc = _resolve_executable(paths["novnc_bin"])
        if resolved_novnc is not None:
            paths["novnc_web_root"] = str(
                Path(resolved_novnc).resolve().parent.parent / "share" / "webapps" / "novnc"
            )

    cfg = Config(
        paths=PathsConfig(**merged["paths"]),
        launch=LaunchConfig(**merged["launch"]),
        preview=PreviewConfig(**merged["preview"]),
        dashboard=DashboardConfig(**merged["dashboard"]),
        timeouts=TimeoutsConfig(**merged["timeouts"]),
        run=RunConfig(**merged["run"]),
        body=BodyConfig(**merged["body"]),
    )
    _validate(cfg)
    return cfg


def _resolve_executable(value: str) -> Path | None:
    if os.sep in value:
        path = Path(value)
        return path if path.exists() else None
    found = shutil.which(value)
    return Path(found) if found else None


def body_pose_flags(body: BodyConfig) -> list[str]:
    """The pose-relay argv that declares the field of view, or nothing if undeclared.

    This lives beside the fields rather than in the orchestrator, so the "0.0 means
    undeclared" rule has exactly one owner: the same rule `validate_config` enforces.
    Two places deciding what zero means is how a knob ends up half-applied.
    """
    if not (body.fov_h_deg and body.fov_v_deg):
        return []
    return [
        "--fov-h-deg",
        f"{body.fov_h_deg:g}",
        "--fov-v-deg",
        f"{body.fov_v_deg:g}",
    ]


def _accessible(path: Path, mode: int) -> bool:
    """`os.access`, but a hostile or malformed path answers False instead of raising.

    A path with an embedded NUL raises ValueError, and a namespace-isolated mount can
    raise OSError. `missing_requirements` exists to *report* what is missing, so it
    must not be the thing that crashes on the way.
    """
    try:
        return os.access(path, mode)
    except (OSError, ValueError):
        return False


def missing_requirements(cfg: Config) -> list[str]:
    checks = [
        ("monado", cfg.paths.monado_bin, os.X_OK),
        ("weston", cfg.paths.weston_bin, os.X_OK),
        ("weston-debug", cfg.paths.weston_debug_bin, os.X_OK),
        ("steam", cfg.paths.steam_bin, os.X_OK),
        ("novnc", cfg.paths.novnc_bin, os.X_OK),
        ("websockify", cfg.paths.websockify_bin, os.X_OK),
        ("game", cfg.paths.game_exe, os.R_OK),
        ("openxr runtime", cfg.paths.openxr_active_runtime, os.R_OK),
        ("pose client", cfg.paths.pose_client, os.R_OK),
    ]
    if cfg.paths.vr_override:
        checks.append(("vr shim", cfg.paths.vr_override, os.R_OK))
    missing: list[str] = []
    for label, value, mode in checks:
        resolved = _resolve_executable(value)
        if resolved is None or not _accessible(resolved, mode):
            missing.append(f"{label} ({value})")
    web_root = cfg.paths.novnc_web_root
    rfb = Path(web_root) / "core" / "rfb.js" if web_root else None
    if rfb is None or not _accessible(rfb, os.R_OK):
        missing.append(f"noVNC assets ({web_root or 'not found'})")
    return missing
