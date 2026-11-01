# Re-indexed: AUTH_TRIGGER, commanded_buttons and the BodyState button
# fields live in clank/body.py and are exercised below.
import unittest

from clank.body import (
    AUTH_CLOSURE,
    AUTH_HEAD,
    AUTH_L_HAND,
    AUTH_R_HAND,
    AUTH_ROOT_VEL,
    AUTH_TRIGGER,
    FREEZE_HOLD_AUTH,
    KNOWN_AUTH,
    SCHEMA,
    BodyCommand,
    BodyState,
    Pose,
    channel_pose,
    commanded_buttons,
    describe_auth,
    hold,
    rest,
)

POSE = Pose((0.1, 0.2, 1.0), (0.0, 0.0, 0.0, 1.0))


def sample_state() -> BodyState:
    return BodyState(
        seq_ack=7,
        t_mono_ns=123456789,
        root_p=(0.0, 0.0, 0.793),
        root_q=(1.0, 0.0, 0.0, 0.0),
        v_xy=(0.01, -0.02),
        yaw_rate=0.003,
        q=tuple(0.1 * i for i in range(29)),
        qd=tuple(0.01 * i for i in range(29)),
        head=POSE,
        left_hand=POSE,
        right_hand=POSE,
        residual_head_m=0.004,
        residual_left_m=0.031,
        residual_right_m=0.028,
        flags=0,
        supported=AUTH_ROOT_VEL | AUTH_HEAD,
    )


class BodyTestCase(unittest.TestCase):
    def test_schema_name_is_clank_body_1(self) -> None:
        self.assertEqual(SCHEMA, "clank.body/1")
        self.assertEqual(BodyCommand().schema, "clank.body/1")

    def test_command_roundtrip(self) -> None:
        cmd = BodyCommand(
            seq=4,
            t_mono_ns=99,
            authority=AUTH_HEAD | AUTH_L_HAND,
            v_xy=(0.2, -0.1),
            yaw_rate=0.4,
            height=1.221,
            head=POSE,
            left_hand=POSE,
            closure=(0.0, 1.0),
            waist=(0.1, 0.0, -0.1),
            duration_ms=1500,
            preempt=False,
            done_when="target",
        )
        self.assertEqual(BodyCommand.from_dict(cmd.to_dict()), cmd)

    def test_state_roundtrip(self) -> None:
        state = sample_state()
        self.assertEqual(BodyState.from_dict(state.to_dict()), state)

    def test_unset_fields_are_none_not_zero(self) -> None:
        cmd = BodyCommand.from_dict({"seq": 1})
        self.assertIsNone(cmd.height)
        self.assertIsNone(cmd.head)
        self.assertIsNone(cmd.left_hand)
        self.assertIsNone(cmd.right_hand)
        self.assertIsNone(cmd.closure)
        self.assertIsNone(cmd.waist)
        # And they survive the return trip as null, not 0.0.
        self.assertIsNone(cmd.to_dict()["height"])

    def test_hold_requests_head_and_both_hands(self) -> None:
        # Freeze-hold, not an empty mask. An empty mask is `rest` and means
        # "standing clip", which is what used to drop a reached hand on TTL expiry.
        cmd = hold(seq=5, now_ns=10)
        self.assertEqual(cmd.authority, FREEZE_HOLD_AUTH)
        self.assertEqual(describe_auth(cmd.authority), ["head", "left_hand", "right_hand"])
        self.assertFalse(cmd.is_ballistic())
        self.assertFalse(cmd.is_blocking())

    def test_is_blocking_only_for_ballistic_without_preempt(self) -> None:
        self.assertFalse(BodyCommand(duration_ms=0).is_blocking())
        self.assertTrue(BodyCommand(duration_ms=800, preempt=False).is_blocking())
        self.assertFalse(BodyCommand(duration_ms=800, preempt=True).is_blocking())

    def test_expires_at_uses_ttl(self) -> None:
        cmd = BodyCommand(t_mono_ns=1_000_000, ttl_ms=200)
        self.assertEqual(cmd.expires_at_ns(), 1_000_000 + 200 * 1_000_000)

    def test_describe_auth_names_bits(self) -> None:
        self.assertEqual(
            describe_auth(AUTH_ROOT_VEL | AUTH_HEAD | AUTH_L_HAND),
            ["root_vel", "head", "left_hand"],
        )

    def test_trigger_is_its_own_authority_bit(self) -> None:
        self.assertEqual(AUTH_TRIGGER, 1 << 9)
        self.assertEqual(AUTH_TRIGGER & AUTH_CLOSURE, 0)
        self.assertEqual(AUTH_TRIGGER & ~KNOWN_AUTH, 0, "trigger must be inside KNOWN_AUTH")
        self.assertEqual(describe_auth(AUTH_TRIGGER), ["trigger"])

    def test_commanded_buttons_are_per_hand_and_authority_gated(self) -> None:
        grip_and_fire = BodyCommand(
            authority=AUTH_CLOSURE | AUTH_TRIGGER,
            closure=(0.0, 1.0),
            trigger=(1.0, 0.0),
        )
        self.assertEqual(commanded_buttons(grip_and_fire), ((0.0, 1.0), (1.0, 0.0)))

    def test_commanded_buttons_are_zero_without_their_bit(self) -> None:
        no_authority = BodyCommand(closure=(1.0, 1.0), trigger=(1.0, 1.0))
        self.assertEqual(commanded_buttons(no_authority), ((0.0, 0.0), (0.0, 0.0)))

        absent = BodyCommand(authority=AUTH_CLOSURE | AUTH_TRIGGER)
        self.assertEqual(commanded_buttons(absent), ((0.0, 0.0), (0.0, 0.0)))

    def test_command_round_trips_the_trigger(self) -> None:
        # to_dict wrote `trigger` and from_dict never read it, so any JSON writer
        # (the bus, a future brain) had its trigger silently discarded.
        cmd = BodyCommand(authority=AUTH_CLOSURE | AUTH_TRIGGER,
                          closure=(0.0, 1.0), trigger=(1.0, 0.0))
        back = BodyCommand.from_dict(cmd.to_dict())
        self.assertEqual(back.closure, (0.0, 1.0))
        self.assertEqual(back.trigger, (1.0, 0.0))
        self.assertEqual(commanded_buttons(back), ((0.0, 1.0), (1.0, 0.0)))

    def test_state_round_trips_buttons(self) -> None:
        state = sample_state()
        state = BodyState(**{**state.__dict__, "closure": (0.0, 1.0), "trigger": (1.0, 0.0)})
        back = BodyState.from_dict(state.to_dict())
        self.assertEqual(back.closure, (0.0, 1.0))
        self.assertEqual(back.trigger, (1.0, 0.0))


REACHED_RIGHT = Pose((0.18, -0.20, 0.40), (0.0, 0.0, 0.0, 1.0))
REACHED_LEFT = Pose((0.18, 0.20, 0.40), (0.0, 0.0, 0.0, 1.0))
LOOKED_HEAD = Pose((0.0, 0.0, 0.40), (0.0, 0.0, 0.1, 0.995))


def reached_state() -> BodyState:
    base = sample_state()
    return BodyState(
        **{
            **base.__dict__,
            "head": LOOKED_HEAD,
            "left_hand": REACHED_LEFT,
            "right_hand": REACHED_RIGHT,
        }
    )


class FreezeHold(unittest.TestCase):
    """`hold` is freeze-hold; `rest` is the standing clip.

    The bug this pins: `hold()` used to request no authority, so the executor
    fell through to the standing clip's VR points and a reached hand snapped
    back to rest. That fired on every TTL expiry (the production path) and on
    every command that named only one channel - `look` after `reach` being the
    one that would drop the card.
    """

    def test_hold_copies_poses_from_state(self) -> None:
        cmd = hold(state=reached_state())
        self.assertEqual(cmd.head, LOOKED_HEAD)
        self.assertEqual(cmd.left_hand, REACHED_LEFT)
        self.assertEqual(cmd.right_hand, REACHED_RIGHT)
        self.assertEqual(cmd.authority, FREEZE_HOLD_AUTH)

    def test_hold_does_not_touch_grip(self) -> None:
        # Grip already latches in the executor. Freezing a hand must not reopen it.
        self.assertEqual(hold().authority & AUTH_CLOSURE, 0)
        self.assertIsNone(hold().closure)

    def test_hold_does_not_claim_root_or_waist(self) -> None:
        # Neither bit can be honoured in a hold session (FLAG_UNSUPPORTED), so
        # claiming them would be a lie to `validate_command`.
        self.assertEqual(FREEZE_HOLD_AUTH & AUTH_ROOT_VEL, 0)
        self.assertEqual(FREEZE_HOLD_AUTH & KNOWN_AUTH, FREEZE_HOLD_AUTH)

    def test_rest_requests_no_authority(self) -> None:
        cmd = rest(seq=5, now_ns=10)
        self.assertEqual(cmd.authority, 0)
        self.assertIsNone(cmd.head)

    def test_rest_ignores_last_state(self) -> None:
        cmd = rest()
        self.assertIsNone(channel_pose(cmd, reached_state(), "head", AUTH_HEAD))
        self.assertIsNone(
            channel_pose(cmd, reached_state(), "right_hand", AUTH_R_HAND)
        )

    def test_look_keeps_the_hands(self) -> None:
        # The card-dropper: a look names only the head, so both hands are
        # unmentioned and must hold where they achieved, not return to clip.
        cmd = BodyCommand(authority=AUTH_HEAD, head=LOOKED_HEAD)
        last = reached_state()
        self.assertEqual(channel_pose(cmd, last, "left_hand", AUTH_L_HAND), REACHED_LEFT)
        self.assertEqual(
            channel_pose(cmd, last, "right_hand", AUTH_R_HAND), REACHED_RIGHT
        )
        self.assertEqual(channel_pose(cmd, last, "head", AUTH_HEAD), LOOKED_HEAD)

    def test_reach_keeps_the_other_hand_and_head(self) -> None:
        cmd = BodyCommand(authority=AUTH_R_HAND, right_hand=REACHED_RIGHT)
        last = reached_state()
        self.assertEqual(channel_pose(cmd, last, "left_hand", AUTH_L_HAND), REACHED_LEFT)
        self.assertEqual(channel_pose(cmd, last, "head", AUTH_HEAD), LOOKED_HEAD)
        self.assertEqual(
            channel_pose(cmd, last, "right_hand", AUTH_R_HAND), REACHED_RIGHT
        )

    def test_no_last_state_is_clip_on_unset_channels(self) -> None:
        # First tick of a session: nothing has been achieved yet, so the clip is
        # the only honest answer. A commanded channel still wins.
        cmd = BodyCommand(authority=AUTH_HEAD, head=LOOKED_HEAD)
        self.assertIsNone(channel_pose(cmd, None, "right_hand", AUTH_R_HAND))
        self.assertEqual(channel_pose(cmd, None, "head", AUTH_HEAD), LOOKED_HEAD)

    def test_freeze_hold_bits_with_no_poses_use_last(self) -> None:
        # `hold()` with no state names the channels but carries no poses, so the
        # caller that has the state fills it in. Both paths reach the same pose.
        cmd = hold()
        self.assertIsNone(cmd.right_hand)
        last = reached_state()
        self.assertEqual(
            channel_pose(cmd, last, "right_hand", AUTH_R_HAND), REACHED_RIGHT
        )

    def test_a_commanded_pose_beats_the_last_one(self) -> None:
        cmd = BodyCommand(authority=AUTH_R_HAND, right_hand=REACHED_LEFT)
        self.assertEqual(
            channel_pose(cmd, reached_state(), "right_hand", AUTH_R_HAND), REACHED_LEFT
        )

    def test_set_bit_with_no_pose_still_falls_back_to_last(self) -> None:
        # A writer may request a channel without supplying a pose; that is not
        # permission to snap to clip.
        cmd = BodyCommand(authority=AUTH_R_HAND)
        self.assertEqual(
            channel_pose(cmd, reached_state(), "right_hand", AUTH_R_HAND), REACHED_RIGHT
        )


if __name__ == "__main__":
    unittest.main()
