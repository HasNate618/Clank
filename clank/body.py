"""The `clank.body/1` command and state contract.

Design: `docs/2026-10-23-clank-design.md` §6, `research/humanoid-command-space.md` §4.

Two rules decide everything else in this module:

1. **Authority is a bitmask, and unset fields follow the last achieved pose.**
   That is how a "mode" is expressed — there is no verb enum below the adapter.
   There are two idle states, and the mask is what tells them apart: `hold`
   claims head and both hands at their last achieved poses, while `rest` is the
   empty mask and returns the executor to its standing clip. An empty mask no
   longer means "executor defaults apply", because those defaults are the clip's
   own VR points and snapping a reached hand back to them is exactly the drift
   this contract exists to prevent.
2. **State reports what the body achieved, never what was asked for.** Callers
   must read `BodyState`, not assume their command landed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SCHEMA = "clank.body/1"

# Authority bits. The only way to select behaviour.
AUTH_ROOT_VEL = 1 << 0  # v_xy, yaw_rate
AUTH_ROOT_HEIGHT = 1 << 1  # height
AUTH_HEAD = 1 << 2  # head pose
AUTH_L_HAND = 1 << 3
AUTH_R_HAND = 1 << 4
AUTH_CLOSURE = 1 << 5  # 0 open .. 1 closed; the adapter maps it to grip/squeeze
AUTH_WAIST = 1 << 6  # yaw, pitch, roll (bounded)
AUTH_GAIT = 1 << 7  # unused in v1; executor reports UNSUPPORTED
AUTH_POSTURE = 1 << 8  # unused in v1 except the stand default
AUTH_TRIGGER = 1 << 9  # 0 released .. 1 pressed; a channel of its own, not a closure value

KNOWN_AUTH = (1 << 10) - 1

# Bits this executor understands but v1 does not honour. Requesting them is
# legal — the executor flags UNSUPPORTED rather than silently pretending.
UNSUPPORTED_IN_V1 = AUTH_GAIT | AUTH_POSTURE

#: The channels a freeze-hold claims: head and both wrists, at their last
#: achieved poses. Deliberately excludes `AUTH_CLOSURE` (grip already latches in
#: the executor, and freezing a hand must not reopen it), `AUTH_ROOT_VEL` and
#: `AUTH_WAIST` (neither can be honoured in a hold session — claiming them would
#: set `FLAG_UNSUPPORTED` on a command that requested nothing).
FREEZE_HOLD_AUTH = AUTH_HEAD | AUTH_L_HAND | AUTH_R_HAND

#: Pose channels, in the order the executor iterates them.
POSE_CHANNELS = (
    ("head", AUTH_HEAD),
    ("left_hand", AUTH_L_HAND),
    ("right_hand", AUTH_R_HAND),
)

# BodyState flags.
FLAG_SATURATED = 1 << 0
FLAG_TTL_EXPIRED = 1 << 1
FLAG_FALL_RISK = 1 << 2
FLAG_UNSUPPORTED = 1 << 3

SOURCES = ("human", "vlm", "planner", "script")
#: Frames a command's poses may be expressed in. `root` is the body frame
#: (`_achieved` de-rotates by the base yaw, so it is the pelvis frame with +X
#: forward). `head` carries hand offsets by the torso twist since the clip, so a
#: hand stays in view when the body turns.
#:
#: `world` used to be listed here and is deliberately gone: the executor never
#: implemented it, so it was accepted and silently ignored - a writer could believe
#: it had asked for world coordinates and receive body ones.
FRAMES = ("root", "head")
DONE_WHEN = ("duration", "target", "external")

# Defaults fixed by the contract.
DEFAULT_TTL_MS = 200
DEFAULT_TOLERANCE_M = 0.02
DEFAULT_MAX_JOINT_SPEED = 2.0  # rad/s, E3
DEFAULT_MAX_HAND_SPEED = 1.0  # m/s


@dataclass(frozen=True)
class Pose:
    """A root-relative pose in metres, quaternion as (x, y, z, w)."""

    p: tuple[float, float, float]
    q_xyzw: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {"p": list(self.p), "q_xyzw": list(self.q_xyzw)}

    @staticmethod
    def from_dict(raw: Any) -> Pose:
        if not isinstance(raw, dict):
            raise TypeError("pose must be an object")
        p = raw["p"]
        q = raw["q_xyzw"]
        if len(p) != 3 or len(q) != 4:
            raise ValueError("pose needs p[3] and q_xyzw[4]")
        return Pose(
            (float(p[0]), float(p[1]), float(p[2])),
            (float(q[0]), float(q[1]), float(q[2]), float(q[3])),
        )


def _pair_or_zero(raw: Any) -> tuple[float, float]:
    """A per-hand pair, tolerating a state file written before the field existed."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return (0.0, 0.0)
    return (float(raw[0]), float(raw[1]))


@dataclass(frozen=True)
class BodyCommand:
    """Writer-agnostic task-space command. Frozen: writers build a new one."""

    schema: str = SCHEMA
    seq: int = 0
    t_mono_ns: int = 0
    ttl_ms: int = DEFAULT_TTL_MS
    source: str = "script"

    authority: int = 0
    frame: str = "root"

    v_xy: tuple[float, float] = (0.0, 0.0)
    yaw_rate: float = 0.0
    height: float | None = None

    head: Pose | None = None
    left_hand: Pose | None = None
    right_hand: Pose | None = None
    closure: tuple[float, float] | None = None
    #: Grip and trigger are separate on purpose. One scalar cannot hold a gun
    #: (closure 1.0, steady) and pull its trigger (0 -> 1 -> 0), and Superhot
    #: needs exactly that. Closure is a grip; trigger is a press.
    trigger: tuple[float, float] | None = None
    waist: tuple[float, float, float] | None = None

    gait: dict[str, Any] | None = None
    posture: dict[str, Any] | None = None

    priority: int = 0
    preempt: bool = False
    duration_ms: int = 0
    done_when: str = "duration"
    tolerance_m: float = DEFAULT_TOLERANCE_M
    max_joint_speed: float = DEFAULT_MAX_JOINT_SPEED
    max_hand_speed: float = DEFAULT_MAX_HAND_SPEED

    # ---- derived helpers -------------------------------------------------

    def is_ballistic(self) -> bool:
        """A command with a duration is a skill that ends by itself."""
        return self.duration_ms > 0

    def is_blocking(self) -> bool:
        """Ballistic and non-preemptible: runs to completion, others queue."""
        return self.is_ballistic() and not self.preempt

    def expires_at_ns(self) -> int:
        return self.t_mono_ns + int(self.ttl_ms) * 1_000_000

    def requested_auth(self) -> int:
        return self.authority & KNOWN_AUTH

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema": self.schema,
            "seq": self.seq,
            "t_mono_ns": self.t_mono_ns,
            "ttl_ms": self.ttl_ms,
            "source": self.source,
            "authority": self.authority,
            "frame": self.frame,
            "v_xy": list(self.v_xy),
            "yaw_rate": self.yaw_rate,
            "height": self.height,
            "head": self.head.to_dict() if self.head else None,
            "left_hand": self.left_hand.to_dict() if self.left_hand else None,
            "right_hand": self.right_hand.to_dict() if self.right_hand else None,
            "closure": list(self.closure) if self.closure is not None else None,
            "trigger": list(self.trigger) if self.trigger is not None else None,
            "waist": list(self.waist) if self.waist is not None else None,
            "gait": self.gait,
            "posture": self.posture,
            "priority": self.priority,
            "preempt": self.preempt,
            "duration_ms": self.duration_ms,
            "done_when": self.done_when,
            "tolerance_m": self.tolerance_m,
            "max_joint_speed": self.max_joint_speed,
            "max_hand_speed": self.max_hand_speed,
        }
        return d

    @staticmethod
    def from_dict(raw: Any) -> BodyCommand:
        if not isinstance(raw, dict):
            raise TypeError("command must be an object")
        pose = lambda k: Pose.from_dict(raw[k]) if raw.get(k) is not None else None  # noqa: E731
        closure = raw.get("closure")
        trigger = raw.get("trigger")
        waist = raw.get("waist")
        v_xy = raw.get("v_xy") or [0.0, 0.0]
        if len(v_xy) != 2:
            raise ValueError("v_xy must have 2 components")
        return BodyCommand(
            schema=str(raw.get("schema", SCHEMA)),
            seq=int(raw.get("seq", 0)),
            t_mono_ns=int(raw.get("t_mono_ns", 0)),
            ttl_ms=int(raw.get("ttl_ms", DEFAULT_TTL_MS)),
            source=str(raw.get("source", "script")),
            authority=int(raw.get("authority", 0)),
            frame=str(raw.get("frame", "root")),
            v_xy=(float(v_xy[0]), float(v_xy[1])),
            yaw_rate=float(raw.get("yaw_rate", 0.0)),
            height=None if raw.get("height") is None else float(raw["height"]),
            head=pose("head"),
            left_hand=pose("left_hand"),
            right_hand=pose("right_hand"),
            closure=None if closure is None else (float(closure[0]), float(closure[1])),
            trigger=None if trigger is None else (float(trigger[0]), float(trigger[1])),
            waist=None
            if waist is None
            else (float(waist[0]), float(waist[1]), float(waist[2])),
            gait=raw.get("gait"),
            posture=raw.get("posture"),
            priority=int(raw.get("priority", 0)),
            preempt=bool(raw.get("preempt", False)),
            duration_ms=int(raw.get("duration_ms", 0)),
            done_when=str(raw.get("done_when", "duration")),
            tolerance_m=float(raw.get("tolerance_m", DEFAULT_TOLERANCE_M)),
            max_joint_speed=float(raw.get("max_joint_speed", DEFAULT_MAX_JOINT_SPEED)),
            max_hand_speed=float(raw.get("max_hand_speed", DEFAULT_MAX_HAND_SPEED)),
        )


@dataclass(frozen=True)
class BodyState:
    """Achieved state, read from `mjData` after `mj_step`. Never desired."""

    seq_ack: int
    t_mono_ns: int
    root_p: tuple[float, float, float]
    root_q: tuple[float, float, float, float]  # MuJoCo order: w, x, y, z
    v_xy: tuple[float, float]
    yaw_rate: float
    q: tuple[float, ...]
    qd: tuple[float, ...]
    head: Pose
    left_hand: Pose
    right_hand: Pose
    #: Commanded, not achieved: the G1 has no fingers, so there is no measured
    #: closure to report. Read this as "what the writer asked the hand to do".
    closure: tuple[float, float] = (0.0, 0.0)
    trigger: tuple[float, float] = (0.0, 0.0)
    residual_head_m: float = 0.0
    residual_left_m: float = 0.0
    residual_right_m: float = 0.0
    flags: int = 0
    supported: int = 0
    invalid_count: int = 0

    def has_flag(self, flag: int) -> bool:
        return bool(self.flags & flag)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq_ack": self.seq_ack,
            "t_mono_ns": self.t_mono_ns,
            "root_p": list(self.root_p),
            "root_q": list(self.root_q),
            "v_xy": list(self.v_xy),
            "yaw_rate": self.yaw_rate,
            "q": list(self.q),
            "qd": list(self.qd),
            "head": self.head.to_dict(),
            "left_hand": self.left_hand.to_dict(),
            "right_hand": self.right_hand.to_dict(),
            "closure": list(self.closure),
            "trigger": list(self.trigger),
            "residual_head_m": self.residual_head_m,
            "residual_left_m": self.residual_left_m,
            "residual_right_m": self.residual_right_m,
            "flags": self.flags,
            "supported": self.supported,
            "invalid_count": self.invalid_count,
        }

    @staticmethod
    def from_dict(raw: Any) -> BodyState:
        if not isinstance(raw, dict):
            raise TypeError("state must be an object")
        return BodyState(
            seq_ack=int(raw["seq_ack"]),
            t_mono_ns=int(raw["t_mono_ns"]),
            root_p=(float(raw["root_p"][0]), float(raw["root_p"][1]), float(raw["root_p"][2])),
            root_q=(
                float(raw["root_q"][0]),
                float(raw["root_q"][1]),
                float(raw["root_q"][2]),
                float(raw["root_q"][3]),
            ),
            v_xy=(float(raw["v_xy"][0]), float(raw["v_xy"][1])),
            yaw_rate=float(raw["yaw_rate"]),
            q=tuple(float(v) for v in raw["q"]),
            qd=tuple(float(v) for v in raw["qd"]),
            head=Pose.from_dict(raw["head"]),
            left_hand=Pose.from_dict(raw["left_hand"]),
            right_hand=Pose.from_dict(raw["right_hand"]),
            closure=_pair_or_zero(raw.get("closure")),
            trigger=_pair_or_zero(raw.get("trigger")),
            residual_head_m=float(raw.get("residual_head_m", 0.0)),
            residual_left_m=float(raw.get("residual_left_m", 0.0)),
            residual_right_m=float(raw.get("residual_right_m", 0.0)),
            flags=int(raw.get("flags", 0)),
            supported=int(raw.get("supported", 0)),
            invalid_count=int(raw.get("invalid_count", 0)),
        )


def hold(
    seq: int = 0,
    now_ns: int = 0,
    ttl_ms: int = DEFAULT_TTL_MS,
    state: BodyState | None = None,
) -> BodyCommand:
    """Freeze-hold: claim head and both hands at their last achieved poses.

    Not "send zero". An empty mask is `rest`, which returns the standing clip —
    and that is what used to snap a reached hand back on every TTL expiry.

    `state` is optional because the writer often does not have it: the bus and
    the adapter fill the poses in when they do, and the executor falls back to
    the last state it achieved. With no state at all, the channels are claimed
    but carry no pose, and `channel_pose` answers with the clip.
    """
    return BodyCommand(
        seq=seq,
        t_mono_ns=now_ns,
        ttl_ms=ttl_ms,
        authority=FREEZE_HOLD_AUTH,
        source="script",
        head=None if state is None else state.head,
        left_hand=None if state is None else state.left_hand,
        right_hand=None if state is None else state.right_hand,
    )


def rest(
    seq: int = 0, now_ns: int = 0, ttl_ms: int = DEFAULT_TTL_MS
) -> BodyCommand:
    """Return the body to its standing clip's own VR points.

    The empty mask, and the only command that means "go back to the clip". Not a
    planner output: a brain that says `hold` means freeze, never reset.
    """
    return BodyCommand(seq=seq, t_mono_ns=now_ns, ttl_ms=ttl_ms, authority=0, source="script")


def commanded_pose(cmd: BodyCommand, name: str, bit: int) -> Pose | None:
    """The pose this command actually asked for, or None if it inherited one.

    The executor needs the distinction: a commanded pose in `frame="head"` still
    has to be carried by the torso twist, but a pose inherited from `BodyState`
    is already an achieved root-frame pose and carrying it again would rotate it
    twice.
    """
    if (cmd.authority & bit) and getattr(cmd, name) is not None:
        return getattr(cmd, name)
    return None


def channel_pose(
    cmd: BodyCommand, last: BodyState | None, name: str, bit: int
) -> Pose | None:
    """The pose one channel should target, or None for the standing clip.

    Four rules, in order:

    1. Empty authority is `rest` — clip, even when a last state exists.
    2. A commanded pose wins.
    3. Otherwise hold the last achieved pose. This is the rule that keeps a hand
       in place while the head turns, and keeps the held card in the hand.
    4. With nothing achieved yet, the clip is the only honest answer.
    """
    if cmd.authority == 0:
        return None
    asked = commanded_pose(cmd, name, bit)
    if asked is not None:
        return asked
    if last is not None:
        return getattr(last, name)
    return None


def commanded_buttons(
    cmd: BodyCommand,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """(closure, trigger) as (left, right), zero where the writer asked for nothing.

    The G1 has no fingers, so there is no achieved closure to report - see
    `BodyState.closure`. This is the commanded value, and a caller must never read
    it as evidence that a hand actually closed.
    """
    grip = (
        cmd.closure
        if (cmd.closure is not None and (cmd.authority & AUTH_CLOSURE) != 0)
        else (0.0, 0.0)
    )
    fire = (
        cmd.trigger
        if (cmd.trigger is not None and (cmd.authority & AUTH_TRIGGER) != 0)
        else (0.0, 0.0)
    )
    return (float(grip[0]), float(grip[1])), (float(fire[0]), float(fire[1]))


def describe_auth(authority: int) -> list[str]:
    """Human-readable authority bits, for logs and the digest."""
    names = [
        (AUTH_ROOT_VEL, "root_vel"),
        (AUTH_ROOT_HEIGHT, "root_height"),
        (AUTH_HEAD, "head"),
        (AUTH_L_HAND, "left_hand"),
        (AUTH_R_HAND, "right_hand"),
        (AUTH_CLOSURE, "closure"),
        (AUTH_WAIST, "waist"),
        (AUTH_GAIT, "gait"),
        (AUTH_POSTURE, "posture"),
        (AUTH_TRIGGER, "trigger"),
    ]
    return [name for bit, name in names if authority & bit]
