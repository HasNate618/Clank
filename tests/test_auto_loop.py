"""The loop must not quietly become a parser again.

`task_intent` read the verb out of JobBot's sentence, `sequence` then always ran locate and always
reached, and the model was decoration. These are grep tests on purpose: the failure they guard
against is an import and four lines of glue, which no unit test of the brain would ever see, and
which reads perfectly sensibly at the time.
"""

from __future__ import annotations

import pathlib
import unittest

SRC = pathlib.Path(__file__).resolve().parents[1] / "spike/wbc_game/auto.py"


class AutoLoop(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SRC.read_text()

    def test_does_not_parse_the_instruction(self) -> None:
        for banned in ("brain.task_intent", "brain.next_step", "brain.sequence(",
                       "task_intent(", "next_step(", "sequence("):
            self.assertNotIn(banned, self.source, f"{banned} puts the choreography back in code")

    def test_thinks_before_it_looks(self) -> None:
        self.assertIn("brain.think", self.source)
        self.assertIn("SENSE_BUDGET", self.source)
        turn = self.source[self.source.index("def one_turn("):]
        self.assertLess(turn.index("brain.think"), turn.index("sense(target"),
                        "the model decides before the body finds anything")

    def test_does_not_sweep_head_angles_to_locate(self) -> None:
        # The old sweep located at five yaw offsets and reached at whichever answered first.
        # That is a search the brain never chose, and it is why a wrong object was sometimes
        # reached for with total confidence.
        self.assertNotIn("LOOK_DEG", self.source)

    def test_locates_only_from_the_sense_step(self) -> None:
        calls = [i for i in range(len(self.source)) if self.source.startswith("locate.locate", i)]
        self.assertEqual(len(calls), 1, "one place runs the locator, and it is sense()")
        sense_at = self.source.index("def sense(")
        turn_at = self.source.index("def one_turn(")
        self.assertGreater(calls[0], sense_at)
        self.assertLess(calls[0], turn_at)

    def test_the_grab_close_stays_inside_the_reach(self) -> None:
        # MEASURED 2026-11-09: closing from a separate process left ~2 s in which the hand
        # drifted 0.09 m - more than the grab radius - so the fingers shut on air.
        self.assertIn("grab=skill == \"grab\"", self.source)


if __name__ == "__main__":
    unittest.main()
