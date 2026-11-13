"""`--move FWD UP LAT`: a reach to where the hand already is, displaced.

A pull is not a new motion. It is the hand that is already closed on a handle travelling, and
the only thing it needs from the caller is a direction and a distance. `clank.workspace` owns
both limits; these tests pin the *wiring* - that the two scripts read the flag, that the target
comes from the measured hand rather than from an image, and that a move with no hand to start
from refuses instead of guessing.

Source-text assertions rather than imports, as `tests/test_auto_loop.py` does for `spike/`: these
modules import MuJoCo and torch at module scope and cannot be imported in the stdlib test run.
"""

from __future__ import annotations

import math
import pathlib
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]


class TheMoveFlag(unittest.TestCase):
    def setUp(self) -> None:
        self.source = (REPO / "spike/wbc_game/reach_ardy.py").read_text()

    def test_the_flag_takes_three_metres(self) -> None:
        self.assertIn('"--move", type=float, nargs=3', self.source)

    def test_a_move_target_comes_from_the_hand_not_from_a_pixel(self) -> None:
        body = self.source[self.source.index("def move_target("):self.source.index("def main(")]
        self.assertIn("moved_point", body)
        self.assertIn("to_world", body)

    def test_a_move_with_no_reported_hand_refuses(self) -> None:
        body = self.source[self.source.index("def move_target("):self.source.index("def main(")]
        self.assertIn("return None", body)

    def test_a_move_and_a_pixel_together_are_refused(self) -> None:
        errors = self.source[self.source.index("parser.error("):
                              self.source.index("session_dir = Path(")]
        self.assertIn("move", errors)


class TheCruiseReadsItToo(unittest.TestCase):
    """The left hand goes through the cruise, so it needs the same flag or a move has a side."""

    def test_the_cruise_takes_a_move(self) -> None:
        source = (REPO / "spike/wbc_game/reach_vr.py").read_text()
        self.assertIn('"--move", type=float, nargs=3', source)
        self.assertIn("moved_point", source)


class TheFrameOfAMove(unittest.TestCase):
    """Review Focus 5: a half-turned robot's "back" is not the world's -X axis."""

    def setUp(self) -> None:
        sys.path.insert(0, str(REPO / "spike/wbc_game"))
        sys.path.insert(0, str(REPO))
        try:
            from spike.wbc_game import reach_vr
        except Exception as exc:  # noqa: BLE001 - the spike's own deps, not this plan's
            self.skipTest(f"reach_vr is not importable here: {exc}")
        self.reach_vr = reach_vr
        from clank import workspace
        self.workspace = workspace

    def test_a_backward_move_follows_the_root_quaternion(self) -> None:
        half = math.radians(45.0)
        st = {"root_p": (0.0, 0.0, 0.0),
              "root_q": (math.cos(half), 0.0, 0.0, math.sin(half))}
        here = (0.40, 0.0, 0.20)
        # 0.20 m, not the 0.40 m cap: the box floor is 0.079 m, so a full-cap backward pull
        # from 0.40 lands on the clamp and the travel measured here would be the clamp's.
        travel = 0.20
        moved = self.workspace.moved_point(here, -travel, 0.0, 0.0)
        world_here = self.reach_vr.to_world(here, st)
        world_moved = self.reach_vr.to_world(moved, st)
        # The sign depends on the quaternion's handedness; the property does not: a pull under a
        # quarter turn leaves the world X axis alone and spends the travel on Y.
        self.assertAlmostEqual(world_moved[0] - world_here[0], 0.0, places=6)
        self.assertAlmostEqual(abs(world_moved[1] - world_here[1]), travel, places=6)


if __name__ == "__main__":
    unittest.main()
