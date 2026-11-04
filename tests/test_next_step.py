"""NEXT STEP: deciding from the scene, and the words-to-skill mapping.

The mapping is the part that has to be a test and not a prompt: measured
2026-11-04, a skill enum in the prompt collapsed to `{"skill": "grab"}` on a
situation where the hand was already closed, while asking for the goal in words
produced the destination. So the model talks, and this code picks.
"""

from __future__ import annotations

import unittest

from clank import brain
from clank.brain import BrainConfig, Frame, NextStep, next_step, skill_from_goal
from tests.test_brain import ChatPatchMixin

FRAME = Frame(index=0, t_s=0.0, jpeg=b"not-a-real-jpeg", source="fixture")


class SkillFromGoalTest(unittest.TestCase):
    def test_insert_place_put_slot_load_all_mean_place(self) -> None:
        for goal in (
            "insert the cartridge into the game console",
            "put the cartridge in the slot",
            "place it in the receptacle",
            "slot the cartridge",
            "load the cartridge",
        ):
            self.assertEqual(skill_from_goal(goal), "place", goal)

    def test_take_and_pick_mean_grab(self) -> None:
        for goal in ("take my card", "pick up a cartridge", "grab the lever"):
            self.assertEqual(skill_from_goal(goal), "grab", goal)

    def test_release_words(self) -> None:
        self.assertEqual(skill_from_goal("release the cartridge"), "release")
        self.assertEqual(skill_from_goal("let go"), "release")
        self.assertEqual(skill_from_goal("drop it"), "release")

    def test_anything_else_is_a_reach(self) -> None:
        self.assertEqual(skill_from_goal("look at the screen"), "reach")
        self.assertEqual(skill_from_goal(""), "reach")

    def test_place_wins_over_grab_when_both_appear(self) -> None:
        # "put the cartridge I took" is a place; the earlier word must not steal it.
        self.assertEqual(skill_from_goal("put the cartridge you took into the slot"), "place")


class NextStepParsingTest(ChatPatchMixin, unittest.TestCase):
    reply = '{"goal": "insert the cartridge into the game console", ' \
            '"object": "cartridge", "destination": "the game console", "side": "right"}'

    def test_destination_becomes_the_target_for_a_place(self) -> None:
        step = next_step(BrainConfig(), FRAME, "you are holding a cartridge")
        self.assertTrue(step.ok)
        self.assertEqual(step.skill, "place")
        self.assertEqual(step.target, "the game console")
        self.assertEqual(step.destination, "the game console")

    def test_the_object_becomes_the_target_for_a_grab(self) -> None:
        self.reply = ('{"goal": "take my card", "object": "JobBot card", '
                      '"destination": "none", "side": "right"}')
        step = next_step(BrainConfig(), FRAME, "standing at the counter")
        self.assertEqual(step.skill, "grab")
        self.assertEqual(step.target, "JobBot card")

    def test_place_without_a_destination_is_an_error_not_a_reach(self) -> None:
        # The failure that looks like success: a place with nowhere to go must not
        # quietly become a reach to whatever the object was.
        self.reply = ('{"goal": "insert the cartridge", "object": "cartridge", '
                      '"destination": "none", "side": "right"}')
        step = next_step(BrainConfig(), FRAME, "holding a cartridge")
        self.assertFalse(step.ok)
        self.assertIn("no destination", step.error)

    def test_a_side_of_center_falls_back_to_the_default(self) -> None:
        self.reply = ('{"goal": "insert the cartridge", "object": "cartridge", '
                      '"destination": "the console", "side": "center"}')
        step = next_step(BrainConfig(), FRAME, "holding a cartridge", default_side="left")
        self.assertEqual(step.side, "left")

    def test_no_json_is_an_error_with_the_raw_text(self) -> None:
        self.reply = "I would insert the cartridge."
        step = next_step(BrainConfig(), FRAME, "holding a cartridge")
        self.assertFalse(step.ok)
        self.assertIn("no JSON", step.error)
        self.assertEqual(step.raw, "I would insert the cartridge.")

    def test_transport_error_is_reported(self) -> None:
        self.reply_error = "connection refused"
        step = next_step(BrainConfig(), FRAME, "holding a cartridge")
        self.assertEqual(step.error, "connection refused")

    def test_the_prompt_carries_the_situation_and_no_skill_enum(self) -> None:
        step = next_step(BrainConfig(), FRAME, "you are holding a cartridge")
        self.assertTrue(step.ok)
        text = self.user_text()
        self.assertIn("you are holding a cartridge", text)
        self.assertNotIn("left|right|", text)
        # The enum that was measured to collapse the answer must not come back.
        self.assertNotIn('"skill": "grab', text)


class NextStepShapeTest(unittest.TestCase):
    def test_not_ok_without_a_skill_or_target(self) -> None:
        self.assertFalse(NextStep(goal="x", skill=None, target="y").ok)
        self.assertFalse(NextStep(goal="x", skill="place", target=None).ok)
        self.assertTrue(NextStep(goal="x", skill="place", target="y").ok)


if __name__ == "__main__":
    unittest.main()
