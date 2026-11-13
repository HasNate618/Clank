"""The measured wrist workspace, and the clamp that keeps the servo inside it.

Provenance for every number asserted here is `research/sonic-reach-envelope.md`
(forward sweep) and `spike/wbc_game/side_envelope_sonic.py` (lateral and vertical
sweeps). These tests are the regression guard on those measurements: if somebody
changes a bound, a test citing the measurement fails and they have to go and measure
again.
"""

import math
import unittest

from clank.workspace import (
    APPROACH_M,
    MAX_LATERAL_OFFSET_M,
    MAX_OFFSET_M,
    WRIST_BOX,
    approach_point,
    box_bounds,
    clamp_offset,
    clamp_wrist,
    in_workspace,
    moved_point,
)

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


class ClampOffset(unittest.TestCase):
    """Limits on a *commanded* offset, which is a different thing from a wrist point.

    Measured 2026-11-03: a raw Jacobian inverse asked for (-0.761, +0.798, +0.124) m of
    offset, the body leaned, and the executor stopped with `FALL detected`. Lateral is
    the axis that does it - it is coupled ~45 deg into forward (Jacobian column
    [+0.50, +0.71, +0.10]), so a lateral request becomes a lean rather than a move.
    """

    def test_a_small_offset_passes_through(self) -> None:
        self.assertEqual(clamp_offset(0.10, 0.05, 0.05), (0.10, 0.05, 0.05))

    def test_lateral_is_clamped_on_its_own_axis_first(self) -> None:
        got = clamp_offset(0.0, -0.761, 0.0)
        self.assertEqual(got, (0.0, -MAX_LATERAL_OFFSET_M, 0.0))

    def test_the_remainder_is_scaled_without_changing_direction(self) -> None:
        # Lateral is capped first, *then* the whole vector is scaled. So the direction
        # that survives is the lateral-capped one, not the original - and it must be
        # preserved exactly, because clamping the axes independently would aim the reach
        # somewhere it was never asked to go.
        fwd, lat, up = 0.798, -0.761, 0.124
        got = clamp_offset(fwd, lat, up)
        pre = (fwd, -MAX_LATERAL_OFFSET_M, up)
        self.assertAlmostEqual(math.dist((0.0, 0.0, 0.0), got), MAX_OFFSET_M, places=9)
        scale = MAX_OFFSET_M / math.dist((0.0, 0.0, 0.0), pre)
        for k in range(3):
            self.assertAlmostEqual(got[k], pre[k] * scale, places=9)

    def test_the_dangerous_request_is_now_survivable(self) -> None:
        # This exact vector knocked the robot over when it was sent unclamped.
        got = clamp_offset(0.798, -0.761, 0.124)
        self.assertLessEqual(abs(got[1]), MAX_LATERAL_OFFSET_M)
        self.assertLessEqual(math.dist((0.0, 0.0, 0.0), got), MAX_OFFSET_M + 1e-9)

    def test_non_finite_is_a_caller_bug(self) -> None:
        for bad in ((float("nan"), 0.0, 0.0), (0.0, float("inf"), 0.0)):
            with self.assertRaises(ValueError):
                clamp_offset(*bad)

    def test_a_non_positive_limit_is_a_caller_bug(self) -> None:
        with self.assertRaises(ValueError):
            clamp_offset(0.1, 0.0, 0.0, max_m=0.0)


class ApproachTest(unittest.TestCase):
    """`place` hovers above its destination; the height is a constant, not a prompt value."""

    def test_approach_is_above_and_inside_the_workspace(self) -> None:
        below = (0.30, -0.30, 0.10)
        above = approach_point(below)
        self.assertAlmostEqual(above[2], below[2] + APPROACH_M, places=6)
        self.assertEqual(above[:2], below[:2])
        self.assertTrue(in_workspace(above))

    def test_approach_clamps_instead_of_leaving_the_box(self) -> None:
        # A destination near the top of the box must not put the hover out of reach.
        high = (0.30, -0.30, 10.0)
        above = approach_point(high)
        self.assertTrue(in_workspace(above))
        self.assertLessEqual(above[2], box_bounds()[2][1])

    def test_approach_rejects_a_short_point(self) -> None:
        with self.assertRaises(ValueError):
            approach_point((0.1, 0.2))


class MovedPointTest(unittest.TestCase):
    """A directional move: the same box and the same cap as every other wrist point.

    `Intent.offset` is `(forward, up, lateral)`; `WRIST_BOX` is `(x, y, z)` with +Y the robot's
    left, so the lateral argument lands on y. These four tests are the only guard that the two
    orders cannot be swapped by a later edit.
    """

    def assertPoint(self, got: tuple[float, ...], want: tuple[float, ...]) -> None:
        """IEEE subtraction is exact only at the values it likes: 0.4 - 0.25 is not 0.15."""
        for mine, theirs in zip(got, want):
            self.assertAlmostEqual(mine, theirs, places=9)

    def test_a_backward_move_subtracts_from_forward(self) -> None:
        self.assertPoint(moved_point((0.40, 0.0, 0.20), -0.25, 0.0, 0.0), (0.15, 0.0, 0.20))

    def test_lateral_is_the_robots_left(self) -> None:
        self.assertPoint(moved_point((0.40, 0.0, 0.20), 0.0, 0.0, 0.10), (0.40, 0.10, 0.20))

    def test_the_offset_is_capped_before_it_is_applied(self) -> None:
        # A 0.60 m pull is more than the arm is measured to travel (0.41-0.51 m).
        self.assertPoint(moved_point((0.55, 0.0, 0.20), -0.60, 0.0, 0.0),
                         clamp_wrist((0.55 - MAX_OFFSET_M, 0.0, 0.20)))

    def test_the_result_is_inside_the_box(self) -> None:
        moved = moved_point((0.10, 0.0, 0.20), -0.25, 0.0, 0.0)
        self.assertTrue(in_workspace(moved))
        self.assertEqual(moved[0], box_bounds()[0][0])
