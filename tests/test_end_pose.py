"""The anti-dance rule: a grab the arm could crane to is still a step if it is off-axis.

`classify` reads the object in the pelvis frame and says whether a still pose with planted
feet serves the grab or whether the root has to step. The yaw gate is what stops a 90 deg
reach from being served by twisting the arm around the torso, and the arm-residual gate is
what stops a legal-direction but too-far object from being served by a crane.

`step_root` is plain planar geometry from the design spec §6.4 steps 1-2: turn until the
preferred hand direction points at the object, then back the root off by that rotated
preference. Everything here is robot-relative metres and radians, stdlib only; the mink
solves live behind `solve_planted` / `solve_stepped`.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import unittest

WBC = pathlib.Path(__file__).resolve().parents[1] / "spike" / "wbc"
_SPEC = importlib.util.spec_from_file_location("end_pose_test", WBC / "end_pose.py")
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - a missing file, not a branch
    raise RuntimeError("could not load spike/wbc/end_pose.py")
end_pose = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(end_pose)


def rotate_z(xy: tuple[float, float], yaw: float) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * xy[0] - s * xy[1], s * xy[0] + c * xy[1])


class Envelope(unittest.TestCase):
    def test_in_front_short_reach_is_planted(self) -> None:
        self.assertEqual(end_pose.classify((0.35, -0.05, 0.15), 0.02), "planted")

    def test_ninety_degrees_steps_even_if_arm_can_crane(self) -> None:
        self.assertEqual(end_pose.classify((0.05, -0.40, 0.15), 0.01), "step")

    def test_in_front_but_short_arm_steps(self) -> None:
        self.assertEqual(end_pose.classify((0.80, 0.0, 0.15), 0.12), "step")

    def test_yaw_error_is_the_angle_in_the_xy_plane(self) -> None:
        self.assertAlmostEqual(math.degrees(end_pose.yaw_error((1.0, 1.0, 9.0))), 45.0)
        self.assertAlmostEqual(math.degrees(end_pose.yaw_error((0.0, -2.0, 0.15))), -90.0)

    def test_the_yaw_gate_is_35_degrees_either_side(self) -> None:
        inside = math.radians(34.9)
        outside = math.radians(35.1)
        self.assertEqual(end_pose.classify((math.cos(inside), math.sin(inside), 0.15), 0.01),
                         "planted")
        self.assertEqual(end_pose.classify((math.cos(inside), -math.sin(inside), 0.15), 0.01),
                         "planted")
        self.assertEqual(end_pose.classify((math.cos(outside), math.sin(outside), 0.15), 0.01),
                         "step")

    def test_an_unsolved_planted_candidate_asks_for_a_planted_solve(self) -> None:
        # Residual is None before mink has run. Falling through to "step" here would step
        # for every grab that is merely unsolved.
        self.assertEqual(end_pose.classify((0.35, -0.05, 0.15), None), "need_planted_solve")

    def test_an_unsolved_off_axis_object_steps_without_a_planted_solve(self) -> None:
        self.assertEqual(end_pose.classify((0.05, -0.40, 0.15), None), "step")

    def test_a_residual_on_the_5_cm_boundary_steps(self) -> None:
        self.assertEqual(end_pose.classify((0.35, 0.0, 0.15), end_pose.PLANTED_RESIDUAL_M),
                         "step")


class StepPlacement(unittest.TestCase):
    def test_step_root_puts_preferred_hand_on_the_object(self) -> None:
        obj = (1.0, 0.0)
        root_xy, yaw = end_pose.step_root(obj, "right", (0.0, 0.0))
        hand = rotate_z(end_pose.HAND_PREF_RIGHT[:2], yaw)
        self.assertAlmostEqual(root_xy[0] + hand[0], obj[0], delta=0.001)
        self.assertAlmostEqual(root_xy[1] + hand[1], obj[1], delta=0.001)
        self.assertFalse(end_pose.step_too_far(root_xy, (0.0, 0.0)))

    def test_step_root_turns_towards_an_object_behind_the_guess(self) -> None:
        # Object at +Y, root guessed at the origin: yaw is 90 deg plus the 39 deg the right
        # hand sits off forward, so the hand - not the chest - ends up on the object.
        obj = (0.0, 1.0)
        root_xy, yaw = end_pose.step_root(obj, "right", (0.0, 0.0))
        hand = rotate_z(end_pose.HAND_PREF_RIGHT[:2], yaw)
        self.assertAlmostEqual(root_xy[0] + hand[0], obj[0], delta=0.001)
        self.assertAlmostEqual(root_xy[1] + hand[1], obj[1], delta=0.001)
        self.assertGreater(yaw, math.radians(90.0))
        self.assertFalse(end_pose.step_too_far(root_xy, (0.0, 0.0)))

    def test_the_left_hand_mirrors_the_right(self) -> None:
        obj = (1.0, 0.0)
        root_xy, yaw = end_pose.step_root(obj, "left", (0.0, 0.0))
        hand = rotate_z(end_pose.HAND_PREF_LEFT[:2], yaw)
        self.assertAlmostEqual(root_xy[0] + hand[0], obj[0], delta=0.001)
        self.assertAlmostEqual(root_xy[1] + hand[1], obj[1], delta=0.001)
        self.assertAlmostEqual(yaw, -end_pose.step_root(obj, "right", (0.0, 0.0))[1])

    def test_three_metre_unprojection_is_too_far(self) -> None:
        root_xy, yaw = end_pose.step_root((3.0, 0.0), "right", (0.0, 0.0))
        self.assertTrue(end_pose.step_too_far(root_xy, (0.0, 0.0)))

    def test_the_step_cap_is_1_5_metres(self) -> None:
        self.assertTrue(end_pose.step_too_far((1.51, 0.0), (0.0, 0.0)))
        self.assertFalse(end_pose.step_too_far((1.49, 0.0), (0.0, 0.0)))

    def test_an_unknown_hand_side_is_an_error_not_a_default(self) -> None:
        with self.assertRaises(ValueError):
            end_pose.step_root((1.0, 0.0), "centre", (0.0, 0.0))

    def test_the_constants_are_the_plan_values(self) -> None:
        self.assertAlmostEqual(end_pose.PLANTED_YAW_RAD, math.radians(35))
        self.assertEqual(end_pose.PLANTED_RESIDUAL_M, 0.05)
        self.assertEqual(end_pose.TRACK_SUCCESS_M, 0.08)
        self.assertEqual(end_pose.CLASP_START_M, 0.08)
        self.assertEqual(end_pose.MAX_STEP_M, 1.5)
        self.assertEqual(end_pose.MAX_STEP_GEN, 4)
        self.assertEqual(end_pose.STEP_CHAIN_S, 8.0)
        self.assertEqual(end_pose.HAND_PREF_RIGHT, (0.35, -0.28, 0.15))
        self.assertEqual(end_pose.HAND_PREF_LEFT, (0.35, 0.28, 0.15))


class StepWorth(unittest.TestCase):
    """A root goal is worth a generation when it buys a walk or a turn, not a shuffle."""

    def test_a_shuffle_with_no_turn_is_not_worth_a_generation(self) -> None:
        # 0.02-0.06 m of travel on an object the body already points at: the measured runs
        # that moved the hand by less than the generator's own spread.
        self.assertFalse(end_pose.step_worthwhile(0.02, 0.0))
        self.assertFalse(end_pose.step_worthwhile(0.06, math.radians(5)))

    def test_either_quantity_alone_is_enough(self) -> None:
        # A real walk with no turn, and a real turn with no walk: both are worth the goal.
        self.assertTrue(end_pose.step_worthwhile(0.55, 0.0))
        self.assertTrue(end_pose.step_worthwhile(0.0, math.radians(53.5)))
        self.assertFalse(end_pose.step_worthwhile(0.03, math.radians(-3.0)))

    def test_the_boundaries_are_inclusive_on_both_quantities(self) -> None:
        self.assertFalse(end_pose.step_worthwhile(end_pose.STEP_WORTH_M - 0.001,
                                                 end_pose.STEP_WORTH_RAD - 0.001))
        self.assertTrue(end_pose.step_worthwhile(end_pose.STEP_WORTH_M, 0.0))
        self.assertTrue(end_pose.step_worthwhile(0.0, end_pose.STEP_WORTH_RAD))
        self.assertTrue(end_pose.step_worthwhile(0.0, -end_pose.STEP_WORTH_RAD))

    def test_a_real_step_is_worth_it(self) -> None:
        # A 0.55 m walk to an object at arm's length + 0.55 m is what the goal exists for.
        placed, _ = end_pose.step_root((1.0, 0.0), "right", (0.0, 0.0))
        self.assertTrue(end_pose.step_worthwhile(math.dist(placed, (0.0, 0.0)), 0.0))

    def test_the_thresholds_are_10_cm_and_20_degrees(self) -> None:
        self.assertEqual(end_pose.STEP_WORTH_M, 0.10)
        self.assertAlmostEqual(end_pose.STEP_WORTH_RAD, math.radians(20))

    def test_on_bearing_and_out_of_reach_is_refused_as_a_measurement(self) -> None:
        # The Task 8 case: the object is nearly straight ahead, so the goal is a walk of a
        # few centimetres, and it was escalating on the residual alone. Nothing to buy.
        p_root = (0.42, -0.02, 0.15)
        placed, _ = end_pose.step_root((p_root[0], p_root[1]), "right", (0.0, 0.0))
        travel = math.dist(placed, (0.0, 0.0))
        self.assertLess(travel, end_pose.STEP_WORTH_M)
        self.assertFalse(end_pose.step_worthwhile(travel, end_pose.yaw_error(p_root)))

    def test_yaw_dominant_and_in_reach_still_escalates_for_the_turn(self) -> None:
        # 53.5 deg off the heading, 0.41 m away: the walk is 0.08 m, but the turn is the
        # thing being bought, so this goal is worth a generation.
        p_root = (0.22, -0.30, 0.15)
        self.assertEqual(end_pose.classify(p_root, None), "step")
        placed, _ = end_pose.step_root((p_root[0], p_root[1]), "right", (0.0, 0.0))
        travel = math.dist(placed, (0.0, 0.0))
        self.assertLess(travel, end_pose.STEP_WORTH_M)
        self.assertTrue(end_pose.step_worthwhile(travel, end_pose.yaw_error(p_root)))


class Escalation(unittest.TestCase):
    def test_success_stops(self) -> None:
        self.assertEqual(end_pose.decide_next(kind="planted", residual_m=0.06, escalated=False,
                                              gen=1, elapsed_s=1.0, generated=1), "stop")

    def test_short_hand_escalates_once(self) -> None:
        self.assertEqual(end_pose.decide_next(kind="planted", residual_m=0.25, escalated=False,
                                              gen=1, elapsed_s=1.0, generated=1), "escalate")

    def test_already_escalated_is_saturated(self) -> None:
        # One root goal is the whole budget, so a second request for one is a dead end, not
        # a quiet stop: the caller has to report the residual it could not beat.
        self.assertEqual(end_pose.decide_next(kind="planted", residual_m=0.25, escalated=True,
                                              gen=2, elapsed_s=2.0, generated=2), "saturated")

    def test_generation_budget_is_bounded(self) -> None:
        self.assertEqual(end_pose.decide_next(kind="planted", residual_m=0.25, escalated=False,
                                              gen=1, elapsed_s=1.0,
                                              generated=end_pose.MAX_STEP_GEN), "saturated")

    def test_a_planted_reach_just_short_stops_instead_of_escalating(self) -> None:
        # 0.09 m is inside the 33-80 mm spread the unseeded generator shows, so spending
        # a regeneration on it would be chasing sampling noise.
        self.assertEqual(end_pose.decide_next(kind="planted", residual_m=0.09, escalated=False,
                                              gen=1, elapsed_s=1.0, generated=1), "stop")

    def test_a_residual_on_the_10_cm_gate_stops(self) -> None:
        # "more than ROOT_GOAL_RESIDUAL_M short" is strict: exactly 10 cm is not a shortfall.
        self.assertEqual(end_pose.decide_next(kind="planted", residual_m=0.10, escalated=False,
                                              gen=1, elapsed_s=1.0, generated=1), "stop")

    def test_an_out_of_reach_object_escalates_even_on_a_good_residual(self) -> None:
        # The object sits outside the planted envelope: no hand-only generation can serve it,
        # however close this one came.
        self.assertEqual(end_pose.decide_next(kind="step", residual_m=0.09, escalated=False,
                                              gen=1, elapsed_s=1.0, generated=1), "escalate")

    def test_a_reached_object_does_not_escalate_because_the_pose_was_a_step(self) -> None:
        # Success trumps the classification. The hand is on the object; stepping now would
        # walk away from a grab that already works.
        self.assertEqual(end_pose.decide_next(kind="step", residual_m=0.05, escalated=False,
                                              gen=1, elapsed_s=1.0, generated=1), "stop")

    def test_the_root_goal_gate_is_10_cm(self) -> None:
        self.assertEqual(end_pose.ROOT_GOAL_RESIDUAL_M, 0.10)

    def test_the_time_budget_alone_is_saturated(self) -> None:
        # A chain that has spent its 8 s has no room for one more root goal.
        self.assertEqual(end_pose.decide_next(kind="planted", residual_m=0.25, escalated=False,
                                              gen=1, elapsed_s=end_pose.STEP_CHAIN_S,
                                              generated=1), "saturated")

    def test_a_residual_on_the_success_boundary_stops(self) -> None:
        # `<= TRACK_SUCCESS_M` is a success; 8 cm is inside SONIC's own tracking error.
        self.assertEqual(end_pose.decide_next(kind="planted", residual_m=end_pose.TRACK_SUCCESS_M,
                                              escalated=False, gen=1, elapsed_s=1.0,
                                              generated=1), "stop")

    def test_saturating_does_not_override_a_success(self) -> None:
        self.assertEqual(end_pose.decide_next(kind="planted", residual_m=0.02, escalated=False,
                                              gen=end_pose.MAX_STEP_GEN, elapsed_s=1.0,
                                              generated=end_pose.MAX_STEP_GEN), "stop")


if __name__ == "__main__":
    unittest.main()
