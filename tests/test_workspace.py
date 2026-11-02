"""The measured wrist workspace, and the clamp that keeps the servo inside it.

Provenance for every number asserted here is `research/sonic-reach-envelope.md`
(forward sweep) and `spike/wbc_game/side_envelope_sonic.py` (lateral and vertical
sweeps). These tests are the regression guard on those measurements: if somebody
changes a bound, a test citing the measurement fails and they have to go and measure
again.
"""

import unittest

from clank.workspace import WRIST_BOX, box_bounds, clamp_wrist, in_workspace

#: Settled right wrist at the standing clip, root-relative metres.
STANDING_REST = (0.079, -0.308, -0.203)

#: Where the live servo asked the hand to go and the loop diverged
#: (`research/live-pixel-servo.md`). Behind the body, so unreachable.
LIVE_DIVERGENT_REQUEST = (-0.373, 0.442, 0.269)


class BoxShape(unittest.TestCase):
    def test_bounds_are_the_measured_extremes(self):
        """The bounds are measurements, not round numbers somebody liked."""
        x, y, z = box_bounds()
        # forward sweep: the standing rest is 0.079 and the first four reach rows
        # track to 0.587; 0.60+ plateaus with a 0.32-0.38 m residual, so it is not
        # a positioned wrist. The floor is the rest, so idle is inside the box.
        self.assertAlmostEqual(x[0], 0.079, places=3)
        self.assertAlmostEqual(x[1], 0.60, places=2)
        # lateral sweep: -0.280 at lateral 0.00 .. +0.419 at lateral 0.60.
        self.assertAlmostEqual(y[0], -0.31, places=2)
        self.assertAlmostEqual(y[1], 0.42, places=2)
        # vertical sweep: -0.149 at up -0.10 .. +0.449 at up 0.60, plus rest.
        self.assertAlmostEqual(z[0], -0.23, places=2)
        self.assertAlmostEqual(z[1], 0.45, places=2)

    def test_every_axis_is_ordered(self):
        for lo, hi in box_bounds():
            self.assertLess(lo, hi)

    def test_the_box_contains_the_standing_rest(self):
        """If it did not, the adapter would clamp away its own idle pose."""
        self.assertTrue(in_workspace(STANDING_REST))

    def test_the_box_excludes_the_request_that_diverged(self):
        self.assertFalse(in_workspace(LIVE_DIVERGENT_REQUEST))


class Clamping(unittest.TestCase):
    def test_a_point_inside_is_returned_unchanged(self):
        pt = (0.40, 0.10, 0.05)
        self.assertEqual(clamp_wrist(pt), pt)

    def test_a_point_behind_the_body_is_pulled_to_the_front_face(self):
        self.assertEqual(clamp_wrist(LIVE_DIVERGENT_REQUEST)[0], WRIST_BOX.x[0])

    def test_a_point_beyond_the_lateral_limit_is_pulled_in(self):
        self.assertEqual(clamp_wrist((0.40, 0.90, 0.0))[1], WRIST_BOX.y[1])
        self.assertEqual(clamp_wrist((0.40, -0.90, 0.0))[1], WRIST_BOX.y[0])

    def test_a_point_above_the_box_is_pulled_down(self):
        self.assertEqual(clamp_wrist((0.40, 0.0, 1.20))[2], WRIST_BOX.z[1])

    def test_clamping_is_idempotent(self):
        once = clamp_wrist(LIVE_DIVERGENT_REQUEST)
        self.assertEqual(clamp_wrist(once), once)

    def test_only_violating_coordinates_move(self):
        out = clamp_wrist((0.40, 0.90, 0.05))
        self.assertEqual(out[0], 0.40)  # inside: untouched
        self.assertEqual(out[2], 0.05)  # inside: untouched
        self.assertEqual(out[1], WRIST_BOX.y[1])  # outside: onto the face

    def test_in_workspace_agrees_with_clamp(self):
        for pt in (STANDING_REST, LIVE_DIVERGENT_REQUEST, (0.5, 0.0, 0.0),
                   (0.5, 0.0, 0.50), (0.5, 0.50, 0.0)):
            self.assertEqual(in_workspace(pt), clamp_wrist(pt) == pt)


class Overloads(unittest.TestCase):
    def test_a_short_tuple_is_rejected(self):
        with self.assertRaises(ValueError):
            clamp_wrist((0.4, 0.0))

    def test_a_non_finite_coordinate_is_rejected(self):
        with self.assertRaises(ValueError):
            clamp_wrist((float("nan"), 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
