"""ARDY's constraint vocabulary, and the wrapper's pure parts.

Three of the things this guards are silent failures that cost real time when they happen:

* With ``cfg_type="separated"`` a *scalar* ``cfg_weight`` sets the **constraint** guidance
  weight to zero (``ardy_model.py:274-279``), so a constrained run under the default returns
  motion that ignores its constraints - measured 870 mm of hand error instead of 60.
* ``num_denoising_steps`` above the checkpoint's ``num_base_steps`` trips a CUDA device-side
  assert rather than raising, and it takes the GPU with it. ``check_steps`` refuses in Python.
* A hand target is built relative to where the hand already is, and the arithmetic happens in
  ARDY space (y up, z forward), not in our MuJoCo space (z up, x forward). ``hand_delta`` and
  ``world_root`` are the two places that conversion happens, so both are pinned here.

The rest is the root waypoint path: a straight run from the body's current root, `forward_m`
along its heading, at the window's first, middle and last frame. That is pure geometry and has
no business needing a GPU.

This module must be importable without torch: `generate`/`load_ardy` import it inside the call,
so a stdlib test suite can hold the whole surface above.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import subprocess
import sys
import unittest

WBC = pathlib.Path(__file__).resolve().parents[1] / "spike" / "wbc"

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_test", WBC / f"{name}.py")
    if spec is None or spec.loader is None:  # pragma: no cover - a missing file, not a branch
        raise RuntimeError(f"could not load spike/wbc/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ardy_gen = _load("ardy_gen")


def pose36(root=(0.0, 0.0, 0.0), quat=IDENTITY, hinges=None) -> tuple[float, ...]:
    """A pose with a chosen root and root rotation; hinges default to the neutral stand."""
    return tuple(root) + tuple(quat) + tuple(hinges if hinges is not None else (0.0,) * 29)


def yaw_quat(angle: float) -> tuple[float, float, float, float]:
    """`xyzw` quaternion for `angle` about MuJoCo's z (the body's yaw)."""
    return (0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0))


def rot_z(angle: float) -> tuple[tuple[float, float, float], ...]:
    """A rotation matrix of `angle` about z, the axis a yaw hinge turns on."""
    c, s = math.cos(angle), math.sin(angle)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


class ConstantsTest(unittest.TestCase):
    def test_cfg_weight_is_a_pair_not_a_scalar(self) -> None:
        self.assertEqual(ardy_gen.CFG_WEIGHT, (1.0, 2.0))
        self.assertIsInstance(ardy_gen.CFG_WEIGHT[1], float)
        self.assertGreater(ardy_gen.CFG_WEIGHT[1], 0.0)

    def test_checkpoint_constants(self) -> None:
        self.assertEqual(ardy_gen.ARDY_MODEL, "ARDY-G1-RP-25FPS-Horizon52")
        self.assertEqual((ardy_gen.ARDY_FPS, ardy_gen.ARDY_HORIZON), (25, 52))
        self.assertEqual((ardy_gen.HAND_DELTA_FWD, ardy_gen.HAND_DELTA_UP), (0.35, 0.15))


class CheckStepsTest(unittest.TestCase):
    def test_steps_above_the_schedule_are_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            ardy_gen.check_steps(25, num_base_steps=10)
        self.assertIn("10", str(caught.exception))

    def test_the_schedule_itself_is_allowed(self) -> None:
        self.assertEqual(ardy_gen.check_steps(10, num_base_steps=10), 10)
        self.assertEqual(ardy_gen.check_steps(4, num_base_steps=10), 4)

    def test_no_steps_at_all_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            ardy_gen.check_steps(0, num_base_steps=10)


class HandDeltaTest(unittest.TestCase):
    def test_hand_delta_is_forward_and_up_in_ardy_space(self) -> None:
        d = ardy_gen.hand_delta(0.35, 0.15)
        self.assertEqual(d, (0.0, 0.15, 0.35))  # y up, z forward

    def test_hand_delta_keeps_the_signs_it_is_given(self) -> None:
        self.assertEqual(ardy_gen.hand_delta(-0.2, -0.1), (0.0, -0.1, -0.2))

    def test_hand_delta_does_not_move_the_hand_sideways_and_returns_floats(self) -> None:
        d = ardy_gen.hand_delta(1, 2)
        self.assertEqual(d[0], 0.0)
        self.assertIsInstance(d[2], float)


class RootFrameTest(unittest.TestCase):
    def test_world_root_swaps_mujoco_xyz_into_ardy_xz_and_drops_height(self) -> None:
        # MuJoCo is (x forward, y left, z up); ARDY is (x left, y up, z forward). The model
        # asserts a zero y in its initial translation, so the height is not ours to pass.
        self.assertEqual(ardy_gen.world_root(pose36(root=(0.6, 0.0, 0.793))), (0.0, 0.0, 0.6))
        self.assertEqual(ardy_gen.world_root(pose36(root=(-1.5, 2.0, 0.8))), (2.0, 0.0, -1.5))

    def test_heading_angle_is_the_root_yaw(self) -> None:
        self.assertAlmostEqual(ardy_gen.heading_angle(pose36()), 0.0)
        self.assertAlmostEqual(ardy_gen.heading_angle(pose36(quat=yaw_quat(0.8))), 0.8)
        self.assertAlmostEqual(ardy_gen.heading_angle(pose36(quat=yaw_quat(-2.1))), -2.1)

    def test_heading_angle_wraps_at_half_a_turn(self) -> None:
        self.assertAlmostEqual(abs(ardy_gen.heading_angle(pose36(quat=yaw_quat(math.pi)))), math.pi)


class BearingTest(unittest.TestCase):
    """Facing is judged in the same yaw convention the placement and the path use."""

    def test_a_target_straight_ahead_is_zero_bearing(self) -> None:
        # ARDY world is (x left, y up, z forward) and heading 0 faces +z.
        self.assertAlmostEqual(ardy_gen.bearing_to(pose36(), (0.0, 0.9, 1.0)), 0.0)

    def test_a_target_to_the_left_is_a_positive_yaw(self) -> None:
        self.assertAlmostEqual(ardy_gen.bearing_to(pose36(), (1.0, 0.9, 0.0)), math.pi / 2)
        self.assertAlmostEqual(ardy_gen.bearing_to(pose36(), (-1.0, 0.9, 0.0)), -math.pi / 2)

    def test_the_bearing_is_measured_from_the_root_not_the_world_origin(self) -> None:
        pose = pose36(root=(0.7, -0.2, 0.787))  # ARDY root: z = 0.7 (MuJoCo x), x = -0.2 (MuJoCo y)
        self.assertAlmostEqual(ardy_gen.bearing_to(pose, (-0.2, 0.9, 0.7 + 0.5)), 0.0)
        self.assertAlmostEqual(ardy_gen.bearing_to(pose, (-0.2, 0.9, 0.7 - 0.5)), math.pi)

    def test_off_bearing_subtracts_the_body_heading_and_wraps(self) -> None:
        pose = pose36(quat=yaw_quat(0.5))
        self.assertAlmostEqual(ardy_gen.off_bearing(pose, (0.0, 0.9, 1.0)), -0.5)
        # A heading and a bearing either side of half a turn must not read as a big turn.
        pose = pose36(quat=yaw_quat(math.pi - 0.1))
        self.assertAlmostEqual(
            ardy_gen.off_bearing(pose, (math.sin(0.1), 0.9, -math.cos(0.1))), 0.0, places=6
        )

    def test_wrap_rad_keeps_the_signed_turn(self) -> None:
        self.assertAlmostEqual(ardy_gen.wrap_rad(math.radians(350)), math.radians(-10))
        self.assertAlmostEqual(ardy_gen.wrap_rad(math.radians(-350)), math.radians(10))


class RotationAngleTest(unittest.TestCase):
    def test_the_same_rotation_is_zero_degrees(self) -> None:
        self.assertAlmostEqual(ardy_gen.rotation_angle_deg(rot_z(0.8), rot_z(0.8)), 0.0)

    def test_a_quarter_turn_is_ninety_degrees_either_way(self) -> None:
        self.assertAlmostEqual(
            ardy_gen.rotation_angle_deg(rot_z(math.pi / 2), rot_z(0.0)), 90.0
        )
        self.assertAlmostEqual(
            ardy_gen.rotation_angle_deg(rot_z(0.0), rot_z(math.pi / 2)), 90.0
        )

    def test_a_flipped_leaf_joint_reads_twice_its_angle(self) -> None:
        # What a wrong hinge sign looks like: 0.3 rad the other way is 0.6 rad apart. This is
        # the error the position-only round trip cannot see - a wrist yaw's anchor never moves.
        self.assertAlmostEqual(
            ardy_gen.rotation_angle_deg(rot_z(0.3), rot_z(-0.3)), math.degrees(0.6), places=4
        )

    def test_a_matrix_that_is_not_exactly_orthonormal_is_clamped(self) -> None:
        scaled = ((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0))
        self.assertEqual(ardy_gen.rotation_angle_deg(scaled, rot_z(0.0)), 0.0)


class LookTest(unittest.TestCase):
    """The look goal is the torso's, and the torso is the root plus the waist hinge."""

    def test_the_torso_is_the_root_yaw_plus_the_waist_hinge(self) -> None:
        pose = list(pose36(quat=yaw_quat(0.4)))
        pose[7 + ardy_gen.WAIST_YAW_HINGE] = 0.2
        self.assertAlmostEqual(ardy_gen.torso_yaw(tuple(pose)), 0.6)

    def test_a_waist_twist_moves_the_look_but_not_the_bearing(self) -> None:
        # This is the measured failure: the body faces the object and the waist turns away.
        pose = list(pose36())
        pose[7 + ardy_gen.WAIST_YAW_HINGE] = math.radians(15)
        target = (0.0, 0.9, 1.0)
        self.assertAlmostEqual(ardy_gen.off_bearing(tuple(pose), target), 0.0, places=6)
        self.assertAlmostEqual(ardy_gen.look_error(tuple(pose), target), math.radians(15))

    def test_the_look_gate_is_20_degrees(self) -> None:
        self.assertAlmostEqual(ardy_gen.LOOK_AT_OBJECT_RAD, math.radians(20))


class Frame0Test(unittest.TestCase):
    """A clip starts where the body is; this is the number that says whether it did."""

    def test_an_identical_pose_is_no_distance_and_no_turn(self) -> None:
        pose = pose36(root=(0.4, -0.2, 0.787), quat=yaw_quat(-0.5))
        distance, heading = ardy_gen.frame0_delta(pose, pose)
        self.assertAlmostEqual(distance, 0.0)
        self.assertAlmostEqual(heading, 0.0)

    def test_distance_is_metres_and_heading_is_a_signed_radian_turn(self) -> None:
        live = pose36(root=(0.0, 0.0, 0.787))
        turned = pose36(root=(0.03, 0.04, 0.787), quat=yaw_quat(0.25))
        distance, heading = ardy_gen.frame0_delta(live, turned)
        self.assertAlmostEqual(distance, 0.05)
        self.assertAlmostEqual(heading, 0.25)


class RootWaypointsTest(unittest.TestCase):
    def assertPoints(self, got, want) -> None:
        self.assertEqual(len(got), len(want))
        for actual, expected in zip(got, want, strict=True):
            self.assertAlmostEqual(actual[0], expected[0])
            self.assertAlmostEqual(actual[1], expected[1])

    def test_three_waypoints_at_the_first_middle_and_last_frame(self) -> None:
        frames, points = ardy_gen.root_waypoints(52, 0.6)
        self.assertEqual(frames, (0, 26, 51))
        self.assertPoints(points, ((0.0, 0.0), (0.0, 0.3), (0.0, 0.6)))

    def test_the_path_starts_where_the_body_stands(self) -> None:
        _, points = ardy_gen.root_waypoints(52, 0.6, start_xy=(0.6, -1.2))
        self.assertPoints(points, ((0.6, -1.2), (0.6, -0.9), (0.6, -0.6)))

    def test_forward_follows_the_body_heading_not_the_world_z_axis(self) -> None:
        _, points = ardy_gen.root_waypoints(52, 0.4, heading=math.pi / 2.0)
        self.assertPoints(points, ((0.0, 0.0), (0.2, 0.0), (0.4, 0.0)))

    def test_a_backwards_step_is_allowed(self) -> None:
        _, points = ardy_gen.root_waypoints(52, -0.6)
        self.assertAlmostEqual(points[-1][1], -0.6)


class CheckNumFramesTest(unittest.TestCase):
    """ARDY always returns its horizon and asserts whole tokens, so `generate` refuses first."""

    def test_the_checkpoints_horizon_is_allowed(self) -> None:
        self.assertEqual(ardy_gen.check_num_frames(52, 52, 4), 52)

    def test_a_shorter_window_is_refused_naming_the_horizon(self) -> None:
        # 40 would come back as a 52-frame clip with the hand pinned at frame 39.
        with self.assertRaises(ValueError) as caught:
            ardy_gen.check_num_frames(40, 52, 4)
        self.assertIn("52", str(caught.exception))
        self.assertIn("40", str(caught.exception))

    def test_a_window_off_the_token_grid_never_reaches_the_models_assert(self) -> None:
        # `autoregressive_step` asserts this itself, one layer too deep to catch in Python.
        with self.assertRaises(ValueError) as caught:
            ardy_gen.check_num_frames(50, 52, 4)
        self.assertIn("52", str(caught.exception))

    def test_the_token_guard_names_the_divisor(self) -> None:
        with self.assertRaises(ValueError) as caught:
            ardy_gen.check_num_frames(52, 52, 5)
        self.assertIn("5", str(caught.exception))


class ContextValidationTest(unittest.TestCase):
    """`generate` checks the context before it imports torch, so these stay stdlib-cheap."""

    def test_an_empty_context_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            ardy_gen.generate([], hand_delta=(0.0, 0.15, 0.35), root_forward_m=None)

    def test_a_short_pose_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            ardy_gen.generate([(0.0,) * 35], hand_delta=(0.0, 0.15, 0.35), root_forward_m=None)

    def test_history_features_are_refused_rather_than_misread(self) -> None:
        # ARDY returns init_history_len + gen_horizon_len frames while the hand pin and the
        # residual are indexed at num_frames - 1: a history would silently pin the last
        # history frame, so there is no encoder and the argument is not accepted.
        with self.assertRaises(NotImplementedError) as caught:
            ardy_gen.generate(
                [pose36()],
                hand_delta=(0.0, 0.15, 0.35),
                root_forward_m=None,
                history_features=object(),
            )
        self.assertIn("encoder", str(caught.exception))


class LazyImportTest(unittest.TestCase):
    def test_importing_the_module_does_not_import_torch(self) -> None:
        code = (
            f"import sys; sys.path.insert(0, {str(WBC)!r}); import ardy_gen; "
            "raise SystemExit(1 if 'torch' in sys.modules else 0)"
        )
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == "__main__":
    unittest.main()
