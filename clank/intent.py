"""The `clank.intent/1` contract: what a writer is allowed to ask for.

This schema exists to keep one boundary honest. A writer - the hub script, an
operator click, or a later model - names **a skill, a side, and optionally a
pixel**. It does not name joint angles, an OpenXR pose, or a world-space point
for the hand. Turning an intent into `BodyCommand`s is the adapter's job, and the
body never learns that Superhot or a pixel exists.

Why a separate schema instead of a field on `BodyCommand`: pixels are not
robotics state. Every existing consumer of `clank.body/1` (executor, relay,
dashboard, tests) would have to learn about an image coordinate it can never use,
and the contract's "authority is a bitmask, unset fields are ignored" would gain
a field no authority bit can gate.

Two things this schema deliberately does **not** carry:

- A 3D point for the target. Design §9 keeps 2D->3D out: a pixel is a ray, and
  the glow closes the depth.
- A grab on glow. Glow true means "in range now"; closing the hand is its own
  intent, because an automatic grab on a false highlight would take the wrong
  object.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .body import FRAMES

SCHEMA = "clank.intent/1"

#: Skills the adapter knows. Closed on purpose: a writer that invents a skill is
#: rejected rather than silently held, so a typo cannot look like "the body refused".
SKILLS = (
    "hold",
    "rest",
    "look",
    "reach",
    "grab",
    "release",
    "fire",
    "strike",
    "dodge",
    "twitch",
)

SIDES = ("none", "left", "right")

#: Skills that address one hand. `hold`/`look`/`twitch` do not, and a side on one
#: of those is rejected rather than ignored - a caller that believes it grabbed
#: something with `hold` has a bug worth seeing.
SIDE_SKILLS = ("reach", "grab", "release", "fire", "strike", "dodge")

SOURCES = ("human", "vlm", "planner", "script")

DEFAULT_TTL_MS = 2000

#: Sanity cap on a local offset, in metres. Not a reach model - the executor's
#: `SATURATED` flag is the authority on "can the arm get there". This only stops a
#: 5 m offset from reaching the solver as if it were a typo-free instruction.
MAX_OFFSET_M = 1.0


@dataclass(frozen=True)
class Intent:
    """One writer request. Frozen: a writer builds a new one rather than mutating."""

    schema: str = SCHEMA
    seq: int = 0
    t_mono_ns: int = 0
    ttl_ms: int = DEFAULT_TTL_MS
    source: str = "script"

    skill: str = "hold"
    side: str = "none"

    #: Frame for `offset`. `root` is the body frame; `head` carries the offset by
    #: however far the torso has twisted, so the hand stays where the camera looks.
    #: Imported from the body contract so the two cannot drift apart - which is how
    #: `hand_target` and `dodge_target` came to disagree about which axis was forward.
    frame: str = "root"

    #: Target in the HMD view, normalised `[0, 1]`, origin top-left. Not a VNC
    #: pixel and not a 3D point; the adapter maps it into its own crop.
    pixel: tuple[float, float] | None = None

    #: Head yaw for a `look` without a pixel, radians, positive toward the robot's
    #: own left - the same sign `SonicExecutor.look_target` takes.
    yaw_rad: float | None = None

    #: Local wrist offset in the pelvis frame, `(forward, up, lateral)` metres.
    #: This is the adapter supplying E3's local offsets, per design §9.
    offset: tuple[float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "seq": self.seq,
            "t_mono_ns": self.t_mono_ns,
            "ttl_ms": self.ttl_ms,
            "source": self.source,
            "skill": self.skill,
            "side": self.side,
            "frame": self.frame,
            "pixel": list(self.pixel) if self.pixel is not None else None,
            "yaw_rad": self.yaw_rad,
            "offset": list(self.offset) if self.offset is not None else None,
        }

    @staticmethod
    def from_dict(raw: Any) -> Intent:
        if not isinstance(raw, dict):
            raise TypeError("intent must be an object")
        pixel = raw.get("pixel")
        offset = raw.get("offset")
        yaw = raw.get("yaw_rad")
        return Intent(
            schema=str(raw.get("schema", SCHEMA)),
            seq=int(raw.get("seq", 0)),
            t_mono_ns=int(raw.get("t_mono_ns", 0)),
            ttl_ms=int(raw.get("ttl_ms", DEFAULT_TTL_MS)),
            source=str(raw.get("source", "script")),
            skill=str(raw.get("skill", "hold")),
            side=str(raw.get("side", "none")),
            frame=str(raw.get("frame", "root")),
            pixel=None
            if pixel is None
            else (float(pixel[0]), float(pixel[1])),
            yaw_rad=None if yaw is None else float(yaw),
            offset=None
            if offset is None
            else (float(offset[0]), float(offset[1]), float(offset[2])),
        )


def _finite(*values: float) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def validate_intent(intent: Intent) -> str:
    """Return a reason string, or "" when the intent may be executed.

    Rejection never raises into the control loop; the caller turns a non-empty
    reason into a hold and counts it for the next digest.
    """
    if intent.schema != SCHEMA:
        return f"schema {intent.schema!r} is not {SCHEMA!r}"
    if intent.source not in SOURCES:
        return f"unknown source {intent.source!r}"
    if intent.skill not in SKILLS:
        return f"unknown skill {intent.skill!r}"
    if intent.side not in SIDES:
        return f"unknown side {intent.side!r}"
    if intent.frame not in FRAMES:
        return f"unknown frame {intent.frame!r}"
    if intent.skill in SIDE_SKILLS and intent.side == "none":
        return f"{intent.skill} needs a side"
    if intent.skill not in SIDE_SKILLS and intent.side != "none":
        return f"{intent.skill} does not take a side"
    if intent.ttl_ms < 0:
        return "negative ttl_ms"

    if intent.pixel is not None:
        if len(intent.pixel) != 2 or not _finite(*intent.pixel):
            return "pixel must be two finite numbers"
        u, v = intent.pixel
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            # A VNC pixel reaching here is the likely cause; say so rather than
            # clamping, because a clamp would servo toward the frame edge.
            return f"pixel ({u:.2f}, {v:.2f}) is outside the unit square"

    if intent.yaw_rad is not None and not _finite(intent.yaw_rad):
        return "non-finite yaw_rad"

    if intent.offset is not None:
        if len(intent.offset) != 3 or not _finite(*intent.offset):
            return "offset must be three finite numbers"
        for value in intent.offset:
            if abs(value) > MAX_OFFSET_M:
                return f"offset {value:.2f} m outside +-{MAX_OFFSET_M} m"

    return ""


#: Default reach: inside the measured ~0.20 m envelope rather than at its edge.
DEFAULT_REACH = (0.20, 0.12, 0.0)
AIM_REACH = (0.22, 0.12, 0.0)
PUNCH_REACH = (0.28, 0.02, 0.0)
DODGE_LEAN = (0.0, 0.0, 0.14)
LOOK_ARC_RAD = math.radians(35.0)


def _side(raw: Mapping[str, Any], default: str = "right") -> str:
    return str(raw.get("side", default))


def verb_to_intent(name: str, raw: Mapping[str, Any]) -> Intent | None:
    """Map a pushed button or an intent-shaped dict to an `Intent`.

    Returns None for an unknown name so the caller can report it rather than
    guess. Buttons (`grab_*`, `release_*`, `fire_*`) deliberately stay out: they
    travel with a hand pose so a null result is attributable to the reach rather
    than to the button, and that pose lives in `body_view.verb_command` where the
    executor is. Parsing them here would split one behaviour across two files.

    Unknown or out-of-range parameters are carried through unvalidated; the caller
    runs `validate_intent` so one place owns rejection.
    """
    pixel = raw.get("pixel")
    pixel_pair = None
    if isinstance(pixel, (list, tuple)) and len(pixel) == 2:
        pixel_pair = (float(pixel[0]), float(pixel[1]))

    def make(skill: str, side: str = "none", **kw: Any) -> Intent:
        return Intent(skill=skill, side=side, source="human", pixel=pixel_pair, **kw)

    if name == "hold":
        return make("hold")
    if name == "rest":
        return make("rest")
    if name == "look":
        return make("look", yaw_rad=math.radians(float(raw.get("yaw_deg", 0.0))))
    if name == "look_left":
        return make("look", yaw_rad=LOOK_ARC_RAD)
    if name == "look_right":
        return make("look", yaw_rad=-LOOK_ARC_RAD)
    if name == "reach":
        return make(
            "reach",
            _side(raw),
            offset=(
                float(raw.get("forward_m", DEFAULT_REACH[0])),
                float(raw.get("up_m", DEFAULT_REACH[1])),
                float(raw.get("lateral_m", DEFAULT_REACH[2])),
            ),
        )
    if name == "aim":
        return make("reach", _side(raw), offset=AIM_REACH)
    if name == "punch":
        return make("strike", _side(raw), offset=PUNCH_REACH)
    if name == "dodge_left":
        return make("dodge", "left", offset=DODGE_LEAN)
    if name == "dodge_right":
        return make("dodge", "right", offset=(-DODGE_LEAN[0], DODGE_LEAN[1], -DODGE_LEAN[2]))
    return None
