"""`clank.intent/1`: what a writer names, and nothing a robot cannot obey.

The point of this schema is the boundary. A writer (script, operator click, or a
later model) says *which skill on which side, and optionally which pixel*. It does
not say joint angles, an OpenXR pose, or a world-space point for the hand. The
adapter owns turning that into `BodyCommand`s, and the body never sees a pixel.
"""

from __future__ import annotations

import math
import unittest

from clank import intent


class IntentRoundTrip(unittest.TestCase):
    def test_intent_round_trips(self) -> None:
        i = intent.Intent(skill="reach", side="right", pixel=(0.4, 0.5))
        self.assertEqual(intent.Intent.from_dict(i.to_dict()), i)

    def test_an_offset_round_trips(self) -> None:
        i = intent.Intent(skill="reach", side="left", offset=(0.18, 0.12, -0.04))
        self.assertEqual(intent.Intent.from_dict(i.to_dict()), i)

    def test_a_pixel_serialises_as_a_list(self) -> None:
        d = intent.Intent(skill="reach", side="right", pixel=(0.4, 0.5)).to_dict()
        self.assertEqual(d["pixel"], [0.4, 0.5])
        self.assertEqual(d["schema"], intent.SCHEMA)


class Validation(unittest.TestCase):
    def test_pixel_outside_the_unit_square_is_rejected(self) -> None:
        self.assertTrue(
            intent.validate_intent(
                intent.Intent(skill="reach", side="right", pixel=(1.4, 0.5))
            )
        )

    def test_a_negative_pixel_is_rejected(self) -> None:
        self.assertTrue(
            intent.validate_intent(
                intent.Intent(skill="reach", side="right", pixel=(-0.1, 0.5))
            )
        )

    def test_a_hand_skill_needs_a_side(self) -> None:
        self.assertTrue(intent.validate_intent(intent.Intent(skill="grab", side="none")))

    def test_hold_needs_no_side(self) -> None:
        self.assertEqual(intent.validate_intent(intent.Intent(skill="hold")), "")

    def test_an_unknown_skill_is_rejected(self) -> None:
        self.assertTrue(intent.validate_intent(intent.Intent(skill="teleport")))

    def test_an_unknown_side_is_rejected(self) -> None:
        self.assertTrue(
            intent.validate_intent(intent.Intent(skill="grab", side="both"))
        )

    def test_a_non_finite_yaw_is_rejected(self) -> None:
        self.assertTrue(
            intent.validate_intent(
                intent.Intent(skill="look", yaw_rad=math.inf)
            )
        )

    def test_an_absurd_offset_is_rejected(self) -> None:
        # A metre-scale offset is outside the arm, not merely ambitious.
        self.assertTrue(
            intent.validate_intent(
                intent.Intent(skill="reach", side="right", offset=(5.0, 0.0, 0.0))
            )
        )

    def test_a_wrong_schema_is_rejected(self) -> None:
        self.assertTrue(
            intent.validate_intent(intent.Intent(skill="hold", schema="clank.body/1"))
        )


class VerbParsing(unittest.TestCase):
    def test_look_carries_its_yaw(self) -> None:
        i = intent.verb_to_intent("look", {"yaw_deg": 20})
        assert i is not None
        self.assertEqual(i.skill, "look")
        self.assertAlmostEqual(i.yaw_rad or 0.0, math.radians(20))

    def test_reach_carries_its_offset_and_side(self) -> None:
        i = intent.verb_to_intent(
            "reach", {"side": "left", "forward_m": 0.18, "up_m": 0.1, "lateral_m": -0.02}
        )
        assert i is not None
        self.assertEqual((i.skill, i.side), ("reach", "left"))
        self.assertEqual(i.offset, (0.18, 0.1, -0.02))

    def test_reach_can_carry_the_yaw_it_was_marked_at(self) -> None:
        # A pixel is only valid from the orientation it was marked in. `look` is
        # derived from the clip, so naming the yaw on the reach itself lets the
        # adapter rebuild *that same camera* while the arm moves, instead of
        # letting the body drag the head somewhere else.
        i = intent.verb_to_intent("reach", {"forward_m": 0.2, "look_yaw_deg": -20})
        assert i is not None
        self.assertAlmostEqual(i.yaw_rad, math.radians(-20))

    def test_reach_without_a_yaw_says_nothing_about_the_head(self) -> None:
        # None means "not my business": the caller keeps the existing hold
        # behaviour. A default of 0.0 would silently recentre the camera.
        i = intent.verb_to_intent("reach", {"forward_m": 0.2})
        assert i is not None
        self.assertIsNone(i.yaw_rad)

    def test_reach_defaults_to_the_measured_envelope(self) -> None:
        i = intent.verb_to_intent("reach", {})
        assert i is not None
        offset = i.offset
        assert offset is not None
        self.assertLessEqual(offset[0], 0.22)

    def test_reach_carries_an_optional_pixel(self) -> None:
        i = intent.verb_to_intent("reach", {"side": "right", "pixel": [0.3, 0.7]})
        assert i is not None
        self.assertEqual(i.pixel, (0.3, 0.7))

    def test_the_existing_names_still_map(self) -> None:
        for name in ("hold", "look_left", "look_right", "aim", "punch",
                     "dodge_left", "dodge_right"):
            self.assertIsNotNone(intent.verb_to_intent(name, {}), name)

    def test_unknown_verb_is_none(self) -> None:
        self.assertIsNone(intent.verb_to_intent("teleport", {}))

    def test_a_parsed_verb_validates(self) -> None:
        i = intent.verb_to_intent("aim", {})
        assert i is not None
        self.assertEqual(intent.validate_intent(i), "")

    def test_rest_needs_no_side(self) -> None:
        self.assertEqual(intent.validate_intent(intent.Intent(skill="rest")), "")

    def test_rest_rejects_a_side(self) -> None:
        # A caller that believes `rest` addresses a hand has a bug worth seeing.
        self.assertTrue(intent.validate_intent(intent.Intent(skill="rest", side="right")))

    def test_rest_verb_maps(self) -> None:
        i = intent.verb_to_intent("rest", {})
        assert i is not None
        self.assertEqual(i.skill, "rest")
        self.assertEqual(intent.validate_intent(i), "")

    def test_rest_is_not_a_side_skill(self) -> None:
        self.assertIn("rest", intent.SKILLS)
        self.assertNotIn("rest", intent.SIDE_SKILLS)


if __name__ == "__main__":
    unittest.main()


class FrameField(unittest.TestCase):
    """`offset` needs a frame, and the frame vocabulary is the body contract's.

    Importing `FRAMES` rather than restating it is deliberate: this session found
    two modules disagreeing about which axis was forward, and a second copy of the
    frame list would be the same mistake with a different noun.
    """

    def test_default_frame_is_root(self) -> None:
        self.assertEqual(intent.Intent(skill="reach", side="right").frame, "root")

    def test_head_frame_survives_a_round_trip(self) -> None:
        before = intent.Intent(
            skill="reach", side="right", frame="head", offset=(0.3, 0.2, 0.0)
        )
        after = intent.Intent.from_dict(before.to_dict())
        self.assertEqual(after.frame, "head")
        self.assertEqual(after, before)

    def test_an_unknown_frame_is_rejected(self) -> None:
        reason = intent.validate_intent(
            intent.Intent(skill="reach", side="right", frame="world")
        )
        self.assertIn("unknown frame", reason)

    def test_the_body_contract_owns_the_vocabulary(self) -> None:
        # If `FRAMES` ever grows, this test must not need editing.
        from clank.body import FRAMES

        for frame in FRAMES:
            self.assertEqual(
                intent.validate_intent(
                    intent.Intent(skill="reach", side="right", frame=frame)
                ),
                "",
            )


class AbsoluteTargetTests(unittest.TestCase):
    """`target` is where the hand must be, in the pelvis frame.

    Added because an offset from the clip's wrist is measured worse than an absolute
    point: replacing `vr_pos` outright tracks to 3.51 cm, additive offsets plateaued at
    13.8 cm. A depth-derived object position is already absolute, so it belongs here.
    """

    def test_a_target_round_trips(self) -> None:
        raw = intent.Intent(
            skill="reach", side="right", target=(0.32, -0.11, 0.05)
        ).to_dict()
        self.assertEqual(intent.Intent.from_dict(raw).target, (0.32, -0.11, 0.05))

    def test_no_target_survives_as_none(self) -> None:
        self.assertIsNone(intent.Intent(skill="reach", side="right").target)

    def test_a_target_is_accepted_inside_the_frame(self) -> None:
        self.assertEqual(
            intent.validate_intent(
                intent.Intent(skill="reach", side="right", target=(0.3, -0.2, 0.1))
            ),
            "",
        )

    def test_a_non_finite_target_is_refused(self) -> None:
        for bad in ((float("nan"), 0.0, 0.0), (0.0, float("inf"), 0.0), (0.0, 0.0)):
            if len(bad) != 3:
                continue
            self.assertNotEqual(
                intent.validate_intent(
                    intent.Intent(skill="reach", side="right", target=bad)
                ),
                "",
            )

    def test_a_target_far_outside_the_body_is_refused(self) -> None:
        # Almost always a frame mix-up - world coordinates, or a game-space depth that was
        # never converted - so it is refused rather than driven at.
        self.assertIn(
            "from the pelvis",
            intent.validate_intent(
                intent.Intent(skill="reach", side="right", target=(3.0, 0.0, 0.0))
            ),
        )


class ReachVerbTargetTests(unittest.TestCase):
    """A `reach` verb carrying a point must reach that point, not the default offset.

    Found live: the parser only looked for `forward_m`/`up_m`/`lateral_m`, so
    `{"verb": "reach", "target": [...]}` fell through to `DEFAULT_REACH` and moved the hand
    0.20 m forward of the clip instead. The 0.45 m miss looked exactly like the policy
    refusing to reach, which is how a parser bug survives a live test.
    """

    def test_a_target_becomes_the_target_and_not_an_offset(self) -> None:
        got = intent.verb_to_intent("reach", {"target": [0.3, -0.2, 0.1], "side": "right"})
        assert got is not None
        self.assertEqual(got.target, (0.3, -0.2, 0.1))
        self.assertIsNone(got.offset)

    def test_a_target_pins_the_head_at_zero_yaw(self) -> None:
        # The pixel was marked in the current view; the camera must not move under it.
        got = intent.verb_to_intent("reach", {"target": [0.3, -0.2, 0.1]})
        assert got is not None
        self.assertEqual(got.yaw_rad, 0.0)

    def test_an_offset_reach_still_uses_the_offset(self) -> None:
        got = intent.verb_to_intent("reach", {"forward_m": 0.1, "up_m": 0.05})
        assert got is not None
        self.assertIsNone(got.target)
        self.assertEqual(got.offset, (0.1, 0.05, intent.DEFAULT_REACH[2]))

    def test_a_malformed_target_is_left_for_the_validator(self) -> None:
        got = intent.verb_to_intent("reach", {"target": [0.3, -0.2]})
        assert got is not None
        self.assertIsNone(got.target)  # not silently half-accepted
