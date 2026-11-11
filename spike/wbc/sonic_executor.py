#!/usr/bin/env python3
"""GEAR-SONIC behind the `BodyCommand` contract.

Lives on the spike side because it needs numpy + onnxruntime + mujoco, while
`clank/` stays standard-library only. Phase 3 promotes it once the join
works; until then this is the measured adapter, not product code.

The executor owns feasibility: it clamps, solves, and reports what MuJoCo
actually achieved. It never reports what it was asked for.

Measured facts this file is built on (`research/c3m-results.md`,
`research/sonic-gate-results.md`, `research/wbc-policy-proposal.md`):

- Hold is **not** "freeze the current clip". It switches to the standing clip
  `squat_001__A359` frame 0 and freezes the *anchor index* (`freeze=0`), so the
  encoder's 10-frame lookahead never contains walking. Measured hold RMS
  1.3e-4 m/s. Freezing the wrong thing drifted 0.23-2.07 m.
- That standing clip is the **legs**. The wrists and head are a separate channel:
  `hold` (freeze-hold) claims them at their last achieved poses, and only `rest`
  leaves them on the clip's VR points. Before that split, an empty mask meant
  "clip", so every TTL expiry snapped a reached hand back to rest.
- Wrist aims straight ahead track to 3.51 cm / 7.22 deg; 30 deg off-axis is
  8.63 cm / 28.33 deg. The body must turn to face a target, not aim sideways.
- Arm override *during* walking is not supported by this controller. Only a
  wrist-orientation-only refinement survives (7.22 -> 5.08 deg); the QP joint
  override is dead.

`vr_delta` is a 9-vector (head, left wrist, right wrist - each xyz) added to
the clip's own targets; `vr_orn_delta` is three quats left-multiplied onto
them. That is the only per-tick steering this controller exposes.

With `session_dir` set, the executor is also the player for a motion a
separate process generated (`spike/wbc/reach_ardy.py` -> `mb_mailbox`): each
tick publishes the achieved 36-float pose, a deposited npz clip switches the
runner to mode 0 and plays every row from the current `mjData`, and the end of
the clip returns to the mode-1 freeze-hold. `reset()` is never called on that
path: it writes qpos from a clip's first frame, which is the teleport this
design exists to avoid, and a generated dict carries no `bl`/`ba` for it to
read. The end of a clip is held at the look-ahead (`mb_play.lookahead_rows`),
not by rewriting the clip's rows.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# numpy/mujoco/onnxruntime live in spike/e3/.venv, not in the system interpreter
# this repo's stdlib-only rule applies to `clank/`, not to the spikes.
import numpy as np  # pyright: ignore[reportMissingImports]

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import g1_pack  # noqa: E402
import mb_mailbox  # noqa: E402
import mb_play  # noqa: E402
import sonic_runner as sr  # noqa: E402

from clank.body import (  # noqa: E402
    AUTH_CLOSURE,
    AUTH_HEAD,
    AUTH_L_HAND,
    AUTH_R_HAND,
    AUTH_ROOT_HEIGHT,
    AUTH_TRIGGER,
    FLAG_FALL_RISK,
    FLAG_SATURATED,
    FLAG_UNSUPPORTED,
    UNSUPPORTED_IN_V1,
    BodyCommand,
    BodyState,
    Pose,
    channel_pose,
    commanded_buttons,
    commanded_pose,
)
from clank.projection import carry_with_head, root_offset, yaw_of  # noqa: E402

# Authority this controller can honour today. Walking needs a locomotion clip
# loaded and a live lookahead, so ROOT_VEL is not honoured in a hold session.
SUPPORTED_AUTH = (
    AUTH_HEAD | AUTH_L_HAND | AUTH_R_HAND | AUTH_CLOSURE | AUTH_ROOT_HEIGHT | AUTH_TRIGGER
)

HOLD_CLIP = "squat_001__A359"
HOLD_FRAME = 0

# Measured fall thresholds from the runner: base height and world-up z.
FALL_Z = 0.25
FALL_UP_Z = 0.3

#: Ticks a commanded trigger stays pressed. Long enough for a 50 Hz consumer to
#: sample it, short enough that a second press is a second press.
#: Full-closure finger angles, in `sonic_runner.HAND_ACT_NAMES` order (thumb_0, thumb_1,
#: thumb_2, then two joints per finger). These are GR00T's `grasp_qpos` for the deployed
#: Dex3 hand (`g1_threefinger_hands.py:47,125`), each checked against this XML's joint
#: ranges - every component is inside them, so no clipping is expected in practice.
GRASP_Q = {
    "left": np.array([0.0, 1.0, 1.25, -1.2, -1.2, -1.2, -1.2], dtype=np.float64),
    "right": np.array([0.0, -1.0, -1.25, 1.2, 1.2, 1.2, 1.2], dtype=np.float64),
}

#: Radians of finger travel allowed per control tick (50 Hz). GR00T clamps each hand
#: joint to 0.25 rad per tick in `Dex3Hands::writeOnce`; the same rule keeps a grab a
#: clasp rather than a snap, and keeps current-limited motors off their stops.
HAND_SLEW_RAD = 0.25

FIRE_HOLD_TICKS = 3

# Point order in the VR 3-point interface, from `sonic_runner.vr_targets`:
# it iterates (BODY_L_WRIST, BODY_R_WRIST, BODY_TORSO). `pts` is flat (9,) and
# `orns` is flat (12,), not (3,3) / (3,4).
L_WRIST, R_WRIST, HEAD = 0, 1, 2

IDENTITY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])


def _pose_from(pos: np.ndarray, quat_wxyz: np.ndarray) -> Pose:
    """MuJoCo (w,x,y,z) -> contract (x,y,z,w)."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    return Pose(p=(float(pos[0]), float(pos[1]), float(pos[2])), q_xyzw=(x, y, z, w))


class ExecutorUnavailable(RuntimeError):
    """Raised when the controller or its assets are not present."""


class SonicExecutor:
    """One SONIC tick per `step()`. Owns no timing policy of its own."""

    def __init__(
        self,
        clip: str = HOLD_CLIP,
        model: str = "default",
        verbose: bool = False,
        session_dir: Path | None = None,
    ) -> None:
        self.clip = clip
        self.runner = sr.SonicRunner(mode=1, model=model, verbose=verbose)
        # `g1_pack.pose36_from_qpos` reads fixed qpos addresses up to 49, so a model with a
        # different articulation must fail here, at startup, rather than 50 times a second
        # inside `_publish_pose` once the session is already running.
        assert self.runner.data.qpos.size == g1_pack.QPOS_LEN, (
            f"the pose36 ring needs {g1_pack.QPOS_LEN} qpos values, "
            f"the model has {self.runner.data.qpos.size}"
        )
        self.motion = sr.load_motion(clip)
        #: The freeze-hold reference, restored when a deposited clip has played out.
        #: `self.motion` is what the current tick reads, which during playback is the
        #: generated clip and otherwise this.
        self._hold_motion = self.motion
        self.hold_frame = HOLD_FRAME
        # Anchor-index freeze: the lookahead must never contain walking.
        self.runner.reset(self.motion, start=self.hold_frame)
        self.ticks = 0
        self.fell = False
        self.last_state: BodyState | None = None
        #: Latched (left, right) grip. See the comment in `step()`: the body has no
        #: fingers, so this is a commanded value that persists until released.
        self._buttons: tuple[tuple[float, float], tuple[float, float]] = (
            (0.0, 0.0),
            (0.0, 0.0),
        )
        #: Remaining ticks of trigger hold, per hand. A trigger is momentary: latching
        #: it would let the body fire exactly once per session, because nothing but a
        #: `release` verb would ever bring it back down.
        self._fire_ticks = [0, 0]
        #: Finger angles we are asking for, slewed toward the latched grip each tick.
        #: Kept here rather than in the runner so a restart of the policy cannot
        #: silently leave the hand half shut.
        self._hand_rad = np.zeros(14, dtype=np.float64)
        #: The state `channel_pose` rule 3 is allowed to answer with, and the command
        #: that froze it. See `_fallback`: without this, an unclaimed channel chases
        #: its own measurement.
        self._held_key: tuple[object, ...] | None = None
        self._held_state: BodyState | None = None
        #: The session's output directory. Set, the mailbox is live: achieved poses go
        #: out, deposited clips play, progress is reported. `None` keeps this a plain
        #: hold executor, down to the observation built each tick.
        self.session_dir = Path(session_dir) if session_dir is not None else None
        #: Which deposit is being played and on which row, plus the seq of the request
        #: already answered. `mb_play.Playback` owns the tick ordering that keeps the last
        #: row played and progress reported against the clip that ran, not against a
        #: request that was rejected.
        self._playback = mb_play.Playback()
        #: Set once when any mailbox write fails, so a removed session directory logs once
        #: rather than fifty times a second for the rest of the session.
        self._mailbox_warned = False
        #: The pose36 ring, held in memory so the 50 Hz publish does not read back the
        #: file it wrote one tick ago.
        self._ring: list | None = None

    # ---- geometry -------------------------------------------------------

    def _clip_targets(self) -> tuple[np.ndarray, np.ndarray]:
        return sr.vr_targets(self.motion, self.hold_frame)

    def _achieved(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Achieved root-relative head + both wrists, their quats, base and up."""
        r = self.runner
        base = r.data.qpos[0:3].copy()
        root_inv = sr.quat_conj(r.data.qpos[3:7])
        positions, quats = [], []
        for body, offset in (
            ("left_wrist_yaw_link", sr.VR_OFFSETS[0]),
            ("right_wrist_yaw_link", sr.VR_OFFSETS[1]),
            ("torso_link", sr.VR_OFFSETS[2]),
        ):
            bid = r.model.body(body).id
            pos = r.data.xpos[bid] + r.data.xmat[bid].reshape(3, 3) @ offset - base
            positions.append(sr.quat_rotate(root_inv, pos))
            quats.append(sr.quat_mul(root_inv, r.data.xquat[bid]))
        up = sr.quat_rotate(r.data.qpos[3:7], np.array([0.0, 0.0, 1.0]))
        return np.array(positions), np.array(quats), base, up

    def _head_twist(self) -> float:
        """How far the torso has twisted since the clip's head pose, in radians.

        Zero twist must reproduce the poses measured before `frame="head"` existed,
        so an unknown previous state, or a failed solve, returns exactly 0.0 rather
        than a small number that would quietly shift every later target.

        Composed from the **achieved** head and not the commanded one: the relay
        sends achieved head to the game, so the camera the player sees is the
        achieved pose. Using the command would make the hand and the view disagree
        whenever tracking lags. The cost is one control tick (20 ms) of staleness,
        which is the previous tick's state.
        """
        state = self.last_state
        if state is None or state.head is None:
            return 0.0
        _, orns = self._clip_targets()
        q_ref_xyzw = (
            float(orns[4 * HEAD + 1]),
            float(orns[4 * HEAD + 2]),
            float(orns[4 * HEAD + 3]),
            float(orns[4 * HEAD]),
        )
        delta = yaw_of(state.head.q_xyzw) - yaw_of(q_ref_xyzw)
        # Wrap to (-pi, pi]: 350 deg of twist is 10 deg the other way.
        return math.atan2(math.sin(delta), math.cos(delta))

    def _carry_with_head(
        self, pose: Pose, pivot: tuple[float, float, float], delta_yaw: float
    ) -> Pose:
        """Carry a hand pose with the torso twist: position about Z, and orientation."""
        x, y, z = carry_with_head(pose.p, pivot, delta_yaw)
        qx, qy, qz, qw = pose.q_xyzw
        if delta_yaw != 0.0:
            half = delta_yaw / 2.0
            # wxyz convention for sr.quat_mul, then back to xyzw for the contract.
            q_yaw = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
            qw, qx, qy, qz = sr.quat_mul(q_yaw, np.array([qw, qx, qy, qz]))
        return Pose((x, y, z), (float(qx), float(qy), float(qz), float(qw)))

    def _fallback(self, cmd: BodyCommand) -> BodyState | None:
        """The state rule 3 answers with, frozen for the life of one command.

        `channel_pose` answers any channel the command does not claim with the *last
        achieved* pose. Passing `self.last_state` here refreshed that every tick, so a
        command claiming only the right hand left the left arm tracking its own
        measurement: measured drift 0.444 m over 20 s at 0.0413 m/s RMS in
        `research/p2-soak.md`, against 0.0226 m when latched once. That is a servo
        closing on itself, and it is half of why a reach looked like dancing.

        A command that names some channels therefore means "leave the others where
        they were when I asked". The latch is re-taken whenever the commanded poses
        change - keyed on the poses, not on `seq`, because every command from
        `body_view` carries `seq=0`.
        """
        key = tuple(
            (bit, float(pose.p[0]), float(pose.p[1]), float(pose.p[2]))
            for name, bit in (
                ("head", AUTH_HEAD),
                ("left_hand", AUTH_L_HAND),
                ("right_hand", AUTH_R_HAND),
            )
            for pose in (commanded_pose(cmd, name, bit),)
            if pose is not None
        )
        if key != self._held_key:
            self._held_key = key
            self._held_state = self.last_state
        return self._held_state

    def _targets(self, cmd: BodyCommand) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Absolute VR targets, replacing the clip's own, for the commanded points.

        This is the channel the *measured* aim result used. `gate_common.AimTarget`
        overrides `build_encoder_obs` and REPLACES `vr_pos`/`vr_orn` with absolute
        root-relative targets; it recorded 3.51 cm ahead. Steering with additive
        `vr_delta` instead plateaued at 13.8 cm in a 10 s soak - the offset was
        simply not tracked. So: replace, do not nudge.

        Which pose each channel gets is `channel_pose`'s decision, not this
        method's, so the hold rule has one owner and a stdlib test.
        """
        pts, orns = self._clip_targets()
        pts = np.asarray(pts, dtype=np.float64).copy()
        orns = np.asarray(orns, dtype=np.float64).copy()
        touched = False

        # `frame="head"`: carry hand targets by the torso twist, about the head's
        # vertical axis, so a reached hand stays where the camera is looking.
        delta_yaw = self._head_twist() if cmd.frame == "head" else 0.0
        pivot = (
            float(pts[3 * HEAD]),
            float(pts[3 * HEAD + 1]),
            float(pts[3 * HEAD + 2]),
        )

        for name, bit, index in (
            ("head", AUTH_HEAD, HEAD),
            ("left_hand", AUTH_L_HAND, L_WRIST),
            ("right_hand", AUTH_R_HAND, R_WRIST),
        ):
            pose = channel_pose(cmd, self._fallback(cmd), name, bit)
            if pose is None:
                continue
            # Carry only what the writer asked for. A pose inherited from
            # `BodyState` is already an achieved root-frame pose, so carrying it
            # by the current twist would rotate it a second time.
            if (
                delta_yaw != 0.0
                and bit != AUTH_HEAD
                and commanded_pose(cmd, name, bit) is not None
            ):
                pose = self._carry_with_head(pose, pivot, delta_yaw)
            pts[3 * index:3 * index + 3] = np.asarray(pose.p, dtype=np.float64)
            # Contract is xyzw; the controller wants wxyz.
            x, y, z, w = pose.q_xyzw
            orns[4 * index:4 * index + 4] = np.array([w, x, y, z], dtype=np.float64)
            touched = True

        if not touched:
            return None, None
        return pts, orns

    # ---- one tick -------------------------------------------------------

    def _clasp(self, grip: tuple[float, float]) -> None:
        """Turn the commanded grip into joint motion, so the hand actually closes.

        `grip` is what the XR layer turns into the controller's squeeze button. Until
        now that was the whole journey: `BodyState.closure` carried the command out to
        the game, and the 14 hand motors were never written, so the body grabbed with an
        open hand and the game was the only thing that knew it had happened.

        Target is `g * GRASP_Q` per side, approached at HAND_SLEW_RAD per tick, which
        makes a grab a 5-tick (0.1 s) clasp and a release the same in reverse.
        """
        r = self.runner
        if not len(getattr(r, "hand_act", ())):
            return
        want = np.concatenate(
            [GRASP_Q["left"] * float(grip[0]), GRASP_Q["right"] * float(grip[1])]
        )
        step = np.clip(want - self._hand_rad, -HAND_SLEW_RAD, HAND_SLEW_RAD)
        self._hand_rad = self._hand_rad + step
        r.hand_target = self._hand_rad.copy()

    def step(self, cmd: BodyCommand, now_ns: int = 0) -> BodyState:
        r = self.runner
        if self.session_dir is not None:
            self._publish_pose(now_ns)
            self._accept_play()
        # Mode 0 is the playback mode and only a deposited clip sets it, so it is the
        # single source of truth for "this tick plays a generated clip". With no session
        # directory it is never set and this is exactly the hold executor it always was.
        playing = r.mode == 0
        if playing:
            # No VR overlay: the clip carries the motion, and mode 0 ignores the overlay
            # anyway. The frame is the first row of the clamped look-ahead window, i.e.
            # the row being played and never past the clip's end.
            vr_pos, vr_orn = None, None
            frame = mb_play.lookahead_rows(
                self._playback.frame, self._playback.n
            )[0]
        else:
            vr_pos, vr_orn = self._targets(cmd)
            frame = self.hold_frame

        base_quat = r.data.qpos[3:7].copy()
        enc_obs = r.build_encoder_obs(
            self.motion,
            frame,
            base_quat,
            vr_pos=vr_pos,
            vr_orn=vr_orn,
        )
        # onnxruntime annotates run() as returning SparseTensor | Tensor, but it
        # actually returns a list holding one ndarray (checked at runtime), so the
        # index is fine and the stub is what is wrong here.
        tokens = r.enc.run(None, {r.enc_in: enc_obs[None, :]})[0][0]  # pyright: ignore[reportIndexIssue]
        dec_obs = r.build_decoder_obs(tokens)
        action = r.dec.run(None, {r.dec_in: dec_obs[None, :]})[0][0].astype(np.float64)  # pyright: ignore[reportIndexIssue]
        r.step_policy(action)

        entry = r._state_entry(action)  # noqa: SLF001 - spike exposes no public API
        r.hist.append(entry)
        if len(r.hist) > 20:
            r.hist.pop(0)

        positions, quats, base, up = self._achieved()
        self.ticks += 1

        flags = 0
        if base[2] < FALL_Z or up[2] < FALL_UP_Z:
            flags |= FLAG_FALL_RISK
            self.fell = True
        if (cmd.authority & ~SUPPORTED_AUTH) or (cmd.authority & UNSUPPORTED_IN_V1):
            flags |= FLAG_UNSUPPORTED

        residuals = [0.0, 0.0, 0.0]
        for bit, index, pose, slot in (
            (AUTH_HEAD, HEAD, cmd.head, 0),
            (AUTH_L_HAND, L_WRIST, cmd.left_hand, 1),
            (AUTH_R_HAND, R_WRIST, cmd.right_hand, 2),
        ):
            if (cmd.authority & bit) and pose is not None:
                residuals[slot] = float(np.linalg.norm(np.asarray(pose.p) - positions[index]))
                if residuals[slot] > cmd.tolerance_m:
                    flags |= FLAG_SATURATED

        root_q = r.data.qpos[3:7]
        # Buttons are device state, not a skill, so the GRIP latches: a command that
        # does not mention it leaves it where it was, exactly as a physical button does
        # not spring open because you waved your other arm. Measured before this: every
        # verb except grab reopened the hand on the next tick, and the command bus's TTL
        # did the same after 2 s, so a floppy could be picked up and never carried.
        # `release_*` is how a hand opens; an empty-mask hold is not.
        #
        # The TRIGGER is deliberately not latched. It is a press, and a latched press
        # fires once per session because no verb brings it back down.
        closure, trigger = commanded_buttons(cmd)
        held_grip, _ = self._buttons
        if (cmd.authority & AUTH_CLOSURE) != 0:
            held_grip = closure
        self._buttons = (held_grip, (0.0, 0.0))
        self._clasp(held_grip)
        if (cmd.authority & AUTH_TRIGGER) != 0:
            for hand in (0, 1):
                self._fire_ticks[hand] = FIRE_HOLD_TICKS if trigger[hand] > 0.0 else 0
        held_fire = tuple(
            1.0 if self._fire_ticks[hand] > 0 else 0.0 for hand in (0, 1)
        )
        for hand in (0, 1):
            if self._fire_ticks[hand] > 0:
                self._fire_ticks[hand] -= 1
        state = BodyState(
            seq_ack=cmd.seq,
            t_mono_ns=now_ns,
            root_p=(float(base[0]), float(base[1]), float(base[2])),
            root_q=(float(root_q[0]), float(root_q[1]), float(root_q[2]), float(root_q[3])),
            v_xy=(float(r.data.qvel[0]), float(r.data.qvel[1])),
            yaw_rate=float(r.data.qvel[5]),
            q=tuple(map(float, np.asarray(r._q_policy()).ravel())),  # noqa: SLF001
            qd=tuple(map(float, np.asarray(r._dq_policy()).ravel())),  # noqa: SLF001
            head=_pose_from(positions[HEAD], quats[HEAD]),
            left_hand=_pose_from(positions[L_WRIST], quats[L_WRIST]),
            right_hand=_pose_from(positions[R_WRIST], quats[R_WRIST]),
            residual_head_m=residuals[0],
            residual_left_m=residuals[1],
            residual_right_m=residuals[2],
            flags=flags,
            supported=SUPPORTED_AUTH,
            closure=held_grip,
            trigger=(held_fire[0], held_fire[1]),
            invalid_count=0,
        )
        self.last_state = state
        if playing:
            self._advance_clip()
        return state

    # ---- mailbox: play a deposited clip ---------------------------------

    def _mailbox_dir(self) -> Path:
        """The session directory. `step()` only reaches the mailbox when it is set."""
        assert self.session_dir is not None, "mailbox used without a session directory"
        return self.session_dir

    def _publish_pose(self, now_ns: int) -> None:
        """Hand the adapter the achieved pose ring it generates from.

        The achieved pose, not the commanded one: the adapter continues from where the
        body actually is, and the body is what the next clip has to start from.

        A publish that fails drops this tick's ring update and carries on. Both failures
        are realistic mid-session and neither is worth the 50 Hz loop: a session directory
        rotated or removed under a running body raises `OSError` from the atomic replace,
        and a qpos of unexpected width raises `ValueError` from the packer. The adapter
        sees the tick missing from the ring's `seq`, which is recoverable; losing the body
        process is not.
        """
        try:
            self._ring = mb_mailbox.write_pose36(
                self._mailbox_dir(),
                g1_pack.pose36_from_qpos(self.runner.data.qpos),
                self.ticks,
                now_ns,
                prev=self._ring,
            )
        except (OSError, ValueError) as exc:
            if not self._mailbox_warned:
                self._mailbox_warned = True
                print(
                    f"sonic_executor: pose36 publish failed, carrying on: {exc}",
                    file=sys.stderr,
                )

    def _write_status(self, seq: int, status: str, frame: int, n: int) -> None:
        """Publish progress, or lose this tick's report without losing the body.

        The same rule as `_publish_pose`, and the same trigger: a session directory that
        is rotated or removed under a running body makes the mailbox's atomic replace
        raise `OSError`. Here that would take the 50 Hz loop down from the middle of a
        clip. Warned once, because a directory that has gone away does not come back -
        which means status stops being observable at all. That converts a crash into
        silence, so the adapter is expected to notice through the age of its clip rather
        than through a `failed` status it will never receive.
        """
        why = mb_play.write_progress(self._mailbox_dir(), seq, status, frame, n)
        if why is None or self._mailbox_warned:
            return
        self._mailbox_warned = True
        print(f"sonic_executor: status write failed, carrying on: {why}", file=sys.stderr)

    def _accept_play(self) -> None:
        """Load a clip the adapter deposited and start playing it, without resetting.

        The current `mjData` is the state the clip is played *from*; `reset()` would write
        qpos from the clip's first frame, teleporting the body and the game's view, and it
        reads `bl`/`ba`, which `clip_from_gen.sonic_dict` does not emit.
        """
        session_dir = self._mailbox_dir()
        play = self._playback.pending(mb_mailbox.read_play(session_dir))
        if play is None:
            return
        seq = int(play["seq"])
        path = session_dir / play["path"]
        try:
            motion = self._load_clip(path)
        except Exception as exc:  # noqa: BLE001 - see below
            # Anything, because a truncated clip is a torn zip container and `np.load`
            # raises `zipfile.BadZipFile`, `EOFError` or `UnpicklingError` out of it - none
            # of them an `OSError`, and all of them the realistic result of an interrupted
            # adapter write. Nothing here can legitimately raise something worth
            # re-raising into the 50 Hz loop.
            #
            # A request is taken once even when it cannot be played: the mailbox is
            # latest-wins, so the next attempt arrives with a new seq rather than being
            # retried here every 20 ms. `rejected` deliberately leaves any clip that is
            # still playing owning its own seq, so its progress is not renamed.
            print(f"sonic_executor: clip {path} rejected: {exc}", file=sys.stderr)
            self._playback.rejected(seq)
            self._write_status(seq, "failed", 0, 0)
            return
        self.motion = motion
        self._playback.started(seq, int(motion["n"]))
        # The command that froze the arms before the clip describes a body that has moved
        # by the time the clip ends, so let `_fallback` re-take it from the achieved pose
        # rather than snapping the hands back to their pre-clip target.
        self._held_key = None
        self.runner.mode = 0
        self._write_status(seq, mb_play.PLAYING, 0, int(motion["n"]))

    def _load_clip(self, path: Path) -> dict:
        """Read the adapter's resampled npz into SONIC's motion dict.

        `n` is derived from the arrays and every array is required to agree with it: the
        encoder's look-ahead is clamped against `n`, so a clip whose arrays disagree would
        read rows that are not there instead of holding on the last one.
        """
        with np.load(path) as data:
            rows = {
                key: np.asarray(data[key], dtype=np.float64)
                for key in ("jp", "jv", "bp", "bq")
            }
        lengths = {key: int(rows_array.shape[0]) for key, rows_array in rows.items()}
        n = lengths["jp"]
        if n < 1:
            raise ValueError(f"{path}: clip has no frames")
        if set(lengths.values()) != {n}:
            raise ValueError(f"{path}: clip arrays disagree on length: {lengths}")
        motion: dict[str, object] = dict(rows)
        motion["n"] = n
        return motion

    def _advance_clip(self) -> None:
        """Play out this tick's row, report progress, and leave mode 0 when it plays out.

        The clip's rows are played in full and never rewritten. The hold at the end lives
        in the look-ahead (`mb_play.lookahead_rows`): on the last frame all ten rows the
        encoder reads are that frame, so the reference is constant and the body cannot be
        told to keep walking (`research/c3m-results.md`, H1f).

        Progress is reported against the clip that is playing (`clip_seq`), never against
        the last request taken: a reload that failed leaves an older clip running, and that
        clip must not acknowledge a request that never ran.
        """
        status = self._playback.advance()
        seq = self._playback.clip_seq
        if seq is None:  # unreachable: `advance()` is only called with a clip playing
            raise RuntimeError("progress reported with no clip playing")
        self._write_status(seq, status, self._playback.frame, self._playback.n)
        if status == mb_play.DONE:
            self._finish_clip()

    def _finish_clip(self) -> None:
        """Return to the mode-1 freeze-hold where the clip left the body.

        Putting the standing clip back is all this takes: mode 1 reads its reference from
        `self.motion` at `hold_frame`, and `hold` claims the achieved head and wrists
        through `channel_pose`, so the body stays where it is. Touching qpos here - which
        is what the constructor's `reset()` call does at startup - would be the teleport
        the clip was generated to avoid.
        """
        self.runner.mode = 1
        self.motion = self._hold_motion

    # ---- command builders the reel uses --------------------------------

    def look_target(self, yaw_rad: float) -> Pose:
        """The clip's head pose, yawed. Position held so the neck does the work."""
        pts, orns = self._clip_targets()
        q = orns[4 * HEAD:4 * HEAD + 4]
        half = yaw_rad / 2.0
        # Yaw is about Z in a Z-up frame (MuJoCo/OpenXR stage): [w, 0, 0, z].
        # Building it about Y instead rotates pitch/roll and produced ~0.3 deg
        # of torso yaw for a 35 deg command - measured, not guessed.
        q_yaw = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
        return _pose_from(pts[3 * HEAD:3 * HEAD + 3], sr.quat_mul(q_yaw, q))

    def hand_target_at(
        self, side: str, p: tuple[float, float, float], yaw_deg: float = 0.0
    ) -> Pose:
        """A wrist pose at an absolute point in the pelvis frame, `(x, y, z)`.

        The point is the **VR point**, which is what the game draws the hand with: the
        relay sends `_achieved`'s poses, and those are the wrist link plus
        `VR_OFFSETS[side]` (a 0.18 m lever in the wrist's own frame, measured today as
        a 0.182 m gap between the link and the reported hand). Commanding this point at
        the object's point therefore puts the game's hand on the object; commanding the
        physical link there would put it 0.18 m away and sweeping as the wrist turned.

        Absolute rather than offset because the offset path is measured worse: replacing
        the clip's `vr_pos` outright tracks to 3.51 cm, while additive `vr_delta` offsets
        plateaued at 13.8 cm in a 10 s soak.
        """
        index = L_WRIST if side == "left" else R_WRIST
        _, orns = self._clip_targets()
        quat = orns[4 * index:4 * index + 4]
        if yaw_deg:
            half = math.radians(yaw_deg) / 2.0
            quat = sr.quat_mul(np.array([math.cos(half), 0.0, 0.0, math.sin(half)]), quat)
        return _pose_from(np.asarray(p, dtype=np.float64), quat)

    def hand_target(
        self,
        side: str,
        forward_m: float,
        up_m: float = 0.0,
        lateral_m: float = 0.0,
        yaw_deg: float = 0.0,
    ) -> Pose:
        """A wrist pose offset from the clip's wrist, optionally re-oriented.

        The offset is in the model's root frame: **+X is forward, +Y is the robot's
        left, +Z is up** (`clank.projection.root_offset` owns that mapping). This
        docstring previously stated the opposite - "+Y is forward, +X is the
        robot's left" - and the code implemented *that*, so `forward_m` moved the
        wrist across the body while `lateral_m` moved it forward. Measured
        2026-10-30: `forward_m=0.15` took the right wrist from y=-0.180 to
        y=-0.030 with x unchanged. `dodge_target` below documents the correct
        frame and was already fixed for it (f4ca2e7); the fix had not reached here.

        `yaw_deg` turns the wrist about Z so an "aim" can point somewhere other
        than the clip's palm direction.
        """
        index = L_WRIST if side == "left" else R_WRIST
        pts, orns = self._clip_targets()
        start = 3 * index
        # One mapping, borrowed rather than restated: the previous duplicate here
        # is what made two modules disagree about which axis was which.
        offset = root_offset(
            forward_m=forward_m, up_m=up_m, lateral_m=lateral_m
        )
        target = pts[start:start + 3] + np.asarray(offset, dtype=np.float64)
        quat = orns[4 * index:4 * index + 4]
        if yaw_deg:
            half = math.radians(yaw_deg) / 2.0
            quat = sr.quat_mul(np.array([math.cos(half), 0.0, 0.0, math.sin(half)]), quat)
        return _pose_from(target, quat)

    def aim_target(self, side: str, forward_m: float = 0.22, yaw_deg: float = 0.0) -> Pose:
        """Wrist raised toward the eye line and pointed - the aiming pose.

        Deliberately inside the measured ~0.20 m reliable envelope. Aiming is
        the axis the body is weakest on off-centre (8.63 cm / 28.33 deg at
        30 deg off-axis), which is why the body turns to face first.
        """
        return self.hand_target(side, forward_m=forward_m, up_m=0.12, yaw_deg=yaw_deg)

    def punch_target(self, side: str, forward_m: float = 0.28) -> Pose:
        """Maximum forward wrist reach. E3 measured 9.5 mm at +0.30 m with hard
        IK; under the learned policy this is unmeasured, so read the residual."""
        return self.hand_target(side, forward_m=forward_m, up_m=0.02)

    def dodge_target(self, lateral_m: float) -> Pose:
        """Lateral torso/head offset: a lean, not a step.

        AUTH_ROOT_VEL is not honoured in a hold session, so the base barely moves.
        Measured 2026-10-29 with a 0.14 m command: the head shifted 0.106-0.134 m
        laterally while the base moved 0.008 m or less. That is the whole dodge.

        The offset is on Y, not X. This frame is Z-up with +X forward and +Y left,
        so the first version leaned the head forwards and back while calling it a
        dodge - and the head channel barely honoured that, so the verb moved about
        a centimetre in any direction. Fixing the axis took it to 10-13 cm.
        """
        pts, orns = self._clip_targets()
        start = 3 * HEAD
        target = pts[start:start + 3] + np.array([0.0, lateral_m, -0.04])
        return _pose_from(target, orns[4 * HEAD:4 * HEAD + 4])
