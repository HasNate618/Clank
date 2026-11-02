"""Where the wrist lands in the view we ourselves emit.

This is the whole reason a slow model can still be useful: we own the camera. The
HMD pose is a value we send, so a wrist from `mjData` and a pixel named by a writer
are comparable through arithmetic - no detector, no depth, no 3D reconstruction.

The root cancels. `BodyState.head` and `BodyState.right_hand` are both
root-relative, so the projection needs no root composition at all; a test below
pins that, because recomposing the root would silently reintroduce the
`(w,x,y,z)`-vs-`(x,y,z,w)` trap that already cost one debugging session.
"""

from __future__ import annotations

import math
import unittest

import clank.projection as projection
from clank.body import Pose
from clank.projection import servo_step

IDENTITY = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
HALF_45 = projection.Fov.from_degrees(45.0, 45.0, 45.0, 45.0)


def yawed(degrees: float) -> Pose:
    """A head pose turned about +Z, the frame's up axis."""
    half = math.radians(degrees) / 2.0
    return Pose((0.0, 0.0, 0.0), (0.0, 0.0, math.sin(half), math.cos(half)))


class ProjectPoint(unittest.TestCase):
    def test_a_point_on_the_gaze_axis_is_the_centre(self) -> None:
        u, v = projection.project_point(IDENTITY, (1.0, 0.0, 0.0), HALF_45)  # type: ignore[misc]
        self.assertAlmostEqual(u, 0.5)
        self.assertAlmostEqual(v, 0.5)

    def test_a_point_at_the_left_edge_maps_to_u_zero(self) -> None:
        u, _ = projection.project_point(IDENTITY, (1.0, 1.0, 0.0), HALF_45)  # type: ignore[misc]
        self.assertAlmostEqual(u, 0.0)

    def test_a_point_at_the_top_edge_maps_to_v_zero(self) -> None:
        _, v = projection.project_point(IDENTITY, (1.0, 0.0, 1.0), HALF_45)  # type: ignore[misc]
        self.assertAlmostEqual(v, 0.0)

    def test_a_point_behind_the_camera_is_not_projected(self) -> None:
        # Returning a mirrored pixel here would drive the servo away from the target.
        self.assertIsNone(projection.project_point(IDENTITY, (-1.0, 0.0, 0.0), HALF_45))

    def test_head_yaw_moves_a_fixed_point_to_the_right(self) -> None:
        # Turning the head left (+Z) moves a fixed world point right in the image.
        u, _ = projection.project_point(yawed(20.0), (1.0, 0.0, 0.0), HALF_45)  # type: ignore[misc]
        self.assertGreater(u, 0.5)

    def test_the_root_cancels(self) -> None:
        # Head and wrist offset together: same answer as at the origin.
        head = Pose((0.0, 0.0, 1.5), (0.0, 0.0, 0.0, 1.0))
        u, v = projection.project_point(head, (1.0, 0.0, 1.5), HALF_45)  # type: ignore[misc]
        self.assertAlmostEqual(u, 0.5)
        self.assertAlmostEqual(v, 0.5)

    def test_an_asymmetric_fov_puts_the_centre_off_the_middle(self) -> None:
        wide_left = projection.Fov.from_degrees(60.0, 30.0, 40.0, 40.0)
        u, _ = projection.project_point(IDENTITY, (1.0, 0.0, 0.0), wide_left)  # type: ignore[misc]
        self.assertNotAlmostEqual(u, 0.5)
        self.assertGreater(u, 0.0)
        self.assertLess(u, 1.0)


class RectilinearProjection(unittest.TestCase):
    """An OpenXR frustum is rectilinear: a pixel tracks `tan(angle)`, not `angle`.

    The tests above cannot tell the two models apart - every *edge* angle lands on
    the same pixel either way, and the centre of a symmetric frustum is 0.5 either
    way. These four discriminate. Measured on the live game 2026-11-01: the field is
    about 140 deg, where the difference at 35 deg off-axis is 0.12 of the frame.
    """

    def test_halfway_to_the_edge_is_not_halfway_in_pixels(self) -> None:
        # 22.5 deg in a 45 deg half-frustum: linear says 0.25, tangent says 0.2929.
        point = (1.0, math.tan(math.radians(22.5)), 0.0)
        u, _ = projection.project_point(IDENTITY, point, HALF_45)  # type: ignore[misc]
        self.assertAlmostEqual(u, (1.0 - math.tan(math.radians(22.5))) / 2.0, places=9)
        self.assertNotAlmostEqual(u, 0.25, places=3)

    def test_a_wide_frustum_compresses_off_axis_points_toward_the_middle(self) -> None:
        wide = projection.Fov.from_degrees(70.0, 70.0, 70.0, 70.0)
        point = (1.0, math.tan(math.radians(35.0)), 0.0)
        u, _ = projection.project_point(IDENTITY, point, wide)  # type: ignore[misc]
        tan70, tan35 = math.tan(math.radians(70.0)), math.tan(math.radians(35.0))
        self.assertAlmostEqual(u, (tan70 - tan35) / (2.0 * tan70), places=9)
        # The linear model would have said (70 - 35) / 140 = 0.25, which is wrong
        # by 0.12 of the frame, about 236 px of 1920.
        self.assertNotAlmostEqual(u, 0.25, places=2)

    def test_an_asymmetric_frustum_centres_on_the_tangent(self) -> None:
        left_right = projection.Fov.from_degrees(60.0, 30.0, 40.0, 40.0)
        u, _ = projection.project_point(IDENTITY, (1.0, 0.0, 0.0), left_right)  # type: ignore[misc]
        tan60, tan30 = math.tan(math.radians(60.0)), math.tan(math.radians(30.0))
        self.assertAlmostEqual(u, tan60 / (tan60 + tan30), places=9)

    def test_unproject_inverts_the_tangent_not_the_angle(self) -> None:
        # Pixel 0.25 in a 45 deg half-frustum is atan(1 - 0.25 * 2) = 26.565 deg.
        point = projection.unproject((0.25, 0.5), IDENTITY, HALF_45, 1.0)
        assert point is not None
        self.assertAlmostEqual(point[1], 0.5, places=9)
        self.assertNotAlmostEqual(point[1], math.tan(math.radians(22.5)), places=3)

    def test_round_trip_survives_a_wide_asymmetric_frustum(self) -> None:
        wide = projection.Fov.from_degrees(70.0, 40.0, 60.0, 30.0)
        for pixel in ((0.0, 0.0), (1.0, 1.0), (0.13, 0.87), (0.5, 0.5)):
            for head in (IDENTITY, yawed(35.0), yawed(-70.0)):
                point = projection.unproject(pixel, head, wide, 2.0)
                assert point is not None
                back = projection.project_point(head, point, wide)
                assert back is not None
                self.assertAlmostEqual(back[0], pixel[0], places=9)
                self.assertAlmostEqual(back[1], pixel[1], places=9)


class FovConstruction(unittest.TestCase):
    def test_degrees_are_converted(self) -> None:
        self.assertAlmostEqual(HALF_45.angle_left, math.radians(45.0))

    def test_a_degenerate_fov_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            projection.Fov.from_degrees(0.0, 0.0, 30.0, 30.0)

    def test_a_non_finite_angle_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            projection.Fov.from_degrees(math.inf, 30.0, 30.0, 30.0)

    def test_negative_angles_are_treated_as_magnitudes(self) -> None:
        # OpenXR implementations disagree about the sign of angle_right; taking
        # magnitudes means a symmetric frustum projects the same either way.
        signed = projection.Fov.from_degrees(45.0, -45.0, 45.0, -45.0)
        u, _ = projection.project_point(IDENTITY, (1.0, 1.0, 0.0), signed)  # type: ignore[misc]
        self.assertAlmostEqual(u, 0.0)


class RootOffset(unittest.TestCase):
    """The one place a named offset becomes a model-frame vector.

    This class exists because the same convention was stated two ways in two
    modules: `hand_target` claimed "+Y is forward, +X is the robot's left" while
    `dodge_target` and this module use +X forward, +Y left. Measured 2026-10-30,
    the model follows the latter (clip left wrist y = +0.190, right y = -0.180),
    so `forward_m` was moving the wrist across the body and `lateral_m` was moving
    it forward. A test asserting only `(0.15, 0, 0)` would not have caught that,
    so the tests below pin each argument to a *named* axis, and one pins the sign
    of lateral against the frame `dodge_target` documents.
    """

    def test_forward_is_plus_x(self) -> None:
        self.assertEqual(projection.root_offset(forward_m=0.15), (0.15, 0.0, 0.0))

    def test_lateral_is_plus_y_the_robots_left(self) -> None:
        # Positive lateral is the robot's left, matching the yaw sign that turns
        # left. The clip's left wrist has larger +Y than its right wrist, which is
        # what makes +Y "left" in this model rather than a guess.
        self.assertEqual(projection.root_offset(lateral_m=0.15), (0.0, 0.15, 0.0))

    def test_up_is_plus_z(self) -> None:
        self.assertEqual(projection.root_offset(up_m=0.15), (0.0, 0.0, 0.15))

    def test_components_add(self) -> None:
        self.assertEqual(
            projection.root_offset(forward_m=0.2, up_m=0.1, lateral_m=-0.3),
            (0.2, -0.3, 0.1),
        )

    def test_defaults_are_a_zero_offset(self) -> None:
        self.assertEqual(projection.root_offset(), (0.0, 0.0, 0.0))

    def test_the_axes_are_orthonormal(self) -> None:
        # If someone "fixes" a sign without thinking, two axes can collapse onto
        # each other and every offset becomes unrecoverable.
        for a, b in (
            (projection.FORWARD, projection.LEFT),
            (projection.FORWARD, projection.UP),
            (projection.LEFT, projection.UP),
        ):
            self.assertAlmostEqual(sum(x * y for x, y in zip(a, b)), 0.0)
            self.assertAlmostEqual(math.sqrt(sum(x * x for x in a)), 1.0)

    def test_lateral_agrees_with_the_dodge_frame(self) -> None:
        # dodge_target leans on +Y and is measured at 10-13 cm of head shift.
        # The hand must not silently disagree with the verb that already works.
        self.assertEqual(projection.root_offset(lateral_m=1.0)[1], 1.0)


class YawOf(unittest.TestCase):
    """Heading about +Z, extracted from a body-frame quaternion.

    Needed because "hand relative to head" is a rotation about the vertical axis by
    however far the torso has twisted since the clip's reference pose. Poses in
    `BodyState` are de-rotated into the base frame (`_achieved` applies `root_inv`),
    so a torso twist moves the head and its camera while a fixed body-frame wrist
    target does not - which is precisely why the hand leaves the view when the
    body turns.
    """

    def test_identity_has_no_yaw(self) -> None:
        self.assertAlmostEqual(projection.yaw_of((0.0, 0.0, 0.0, 1.0)), 0.0)

    def test_a_left_turn_is_positive(self) -> None:
        # Matches the sign `look_target` uses for a left turn and root_offset's
        # lateral-is-left convention.
        half = math.pi / 4.0
        self.assertAlmostEqual(
            projection.yaw_of((0.0, 0.0, math.sin(half), math.cos(half))), math.pi / 2.0
        )

    def test_a_right_turn_is_negative(self) -> None:
        half = -math.pi / 4.0
        self.assertAlmostEqual(
            projection.yaw_of((0.0, 0.0, math.sin(half), math.cos(half))), -math.pi / 2.0
        )

    def test_a_pure_pitch_has_no_yaw(self) -> None:
        half = math.pi / 4.0
        self.assertAlmostEqual(
            projection.yaw_of((0.0, math.sin(half), 0.0, math.cos(half))), 0.0
        )

    def test_a_pure_roll_has_no_yaw(self) -> None:
        half = math.pi / 4.0
        self.assertAlmostEqual(
            projection.yaw_of((math.sin(half), 0.0, 0.0, math.cos(half))), 0.0
        )

    def test_yaw_wraps_rather_than_unwinding(self) -> None:
        # 270 deg right and 90 deg left are the same heading.
        half = math.radians(-270.0) / 2.0
        self.assertAlmostEqual(
            projection.yaw_of((0.0, 0.0, math.sin(half), math.cos(half))), math.pi / 2.0
        )


class RotateZ(unittest.TestCase):
    def test_zero_is_exactly_the_identity(self) -> None:
        # Exact, not approximate: the compatibility guarantee for frame="head" is
        # that a zero twist reproduces today's poses bit for bit.
        self.assertEqual(
            projection.rotate_z((0.3, -0.2, 0.1), 0.0), (0.3, -0.2, 0.1)
        )

    def test_a_quarter_turn_sends_forward_to_left(self) -> None:
        x, y, z = projection.rotate_z((1.0, 0.0, 0.0), math.pi / 2.0)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 1.0)
        self.assertAlmostEqual(z, 0.0)

    def test_height_and_radius_are_preserved(self) -> None:
        v = (0.4, -0.3, 0.25)
        out = projection.rotate_z(v, 0.7)
        self.assertAlmostEqual(out[2], 0.25)
        self.assertAlmostEqual(
            math.hypot(out[0], out[1]), math.hypot(0.4, 0.3)
        )


class CarryWithHead(unittest.TestCase):
    """The hand-follows-the-head rule, as arithmetic.

    A wrist offset is carried by the torso's twist: the point is rotated about the
    vertical axis *through the pivot*, so a hand at the pivot does not move and a
    hand out in front swings with the gaze.
    """

    def test_zero_twist_leaves_the_point_exactly_alone(self) -> None:
        self.assertEqual(
            projection.carry_with_head((0.4, -0.2, 0.1), (0.1, 0.0, 0.0), 0.0),
            (0.4, -0.2, 0.1),
        )

    def test_the_pivot_itself_never_moves(self) -> None:
        pivot = (0.1, 0.05, 0.3)
        self.assertEqual(
            projection.carry_with_head(pivot, pivot, 1.2), pivot
        )

    def test_a_left_twist_carries_a_forward_hand_to_the_left(self) -> None:
        x, y, _ = projection.carry_with_head((0.5, 0.0, 0.0), (0.0, 0.0, 0.0), math.pi / 2.0)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 0.5)

    def test_radius_about_the_pivot_is_preserved(self) -> None:
        pivot = (0.1, -0.05, 0.0)
        before = (0.45, 0.2, 0.3)
        after = projection.carry_with_head(before, pivot, 0.9)
        self.assertAlmostEqual(
            math.hypot(before[0] - pivot[0], before[1] - pivot[1]),
            math.hypot(after[0] - pivot[0], after[1] - pivot[1]),
        )
        self.assertAlmostEqual(after[2], before[2])


class PixelError(unittest.TestCase):
    def test_error_is_euclidean_in_normalised_pixels(self) -> None:
        self.assertAlmostEqual(
            projection.pixel_error((0.4, 0.5), (0.7, 0.9)), 0.5
        )

    def test_a_perfect_match_is_zero(self) -> None:
        self.assertEqual(projection.pixel_error((0.4, 0.5), (0.4, 0.5)), 0.0)


class Unproject(unittest.TestCase):
    """The inverse: a pixel plus a depth becomes a root-relative wrist target.

    `project_point` is only half a servo. The adapter can watch the wrist's pixel
    drift toward a target pixel, but without this it has no way to say *where* to
    put the hand - the `reach` verb takes offsets in metres from a human. So every
    intent the brain can emit had no compile path.
    """

    def test_the_centre_pixel_is_a_point_on_the_gaze_axis(self) -> None:
        point = projection.unproject((0.5, 0.5), IDENTITY, HALF_45, 2.0)
        assert point is not None
        for got, want in zip(point, (2.0, 0.0, 0.0)):
            self.assertAlmostEqual(got, want)

    def test_the_left_edge_is_the_robot_left(self) -> None:
        # Pinned because it is easy to get backwards, and backwards drives the
        # servo away from the target instead of toward it.
        point = projection.unproject((0.0, 0.5), IDENTITY, HALF_45, 1.0)
        assert point is not None
        for got, want in zip(point, (1.0, 1.0, 0.0)):
            self.assertAlmostEqual(got, want)

    def test_the_top_edge_is_up(self) -> None:
        point = projection.unproject((0.5, 0.0), IDENTITY, HALF_45, 1.0)
        assert point is not None
        for got, want in zip(point, (1.0, 0.0, 1.0)):
            self.assertAlmostEqual(got, want)

    def test_round_trip_through_project_point(self) -> None:
        for pixel in ((0.5, 0.5), (0.0, 0.0), (1.0, 1.0), (0.2, 0.75)):
            for head in (IDENTITY, yawed(30.0), yawed(-55.0)):
                point = projection.unproject(pixel, head, HALF_45, 1.5)
                assert point is not None
                back = projection.project_point(head, point, HALF_45)
                assert back is not None
                self.assertAlmostEqual(back[0], pixel[0], places=6)
                self.assertAlmostEqual(back[1], pixel[1], places=6)

    def test_the_forward_depth_is_measured_along_the_gaze(self) -> None:
        # Documented semantics: depth_m is the head-frame forward component, not
        # the straight-line distance. A far-off-axis pixel therefore needs *less*
        # depth to sit at the same range, and the projection must agree.
        point = projection.unproject((0.0, 0.5), IDENTITY, HALF_45, 1.0)
        assert point is not None
        self.assertAlmostEqual(point[0], 1.0)          # forward component is depth
        self.assertAlmostEqual(math.dist(point, (0.0, 0.0, 0.0)), math.sqrt(2.0))

    def test_the_head_translation_is_added(self) -> None:
        head = Pose((0.1, -0.2, 1.2), (0.0, 0.0, 0.0, 1.0))
        point = projection.unproject((0.5, 0.5), head, HALF_45, 1.0)
        assert point is not None
        for got, want in zip(point, (1.1, -0.2, 1.2)):
            self.assertAlmostEqual(got, want)

    def test_a_depth_at_or_behind_the_camera_has_no_answer(self) -> None:
        for depth in (0.0, -1.0):
            self.assertIsNone(projection.unproject((0.5, 0.5), IDENTITY, HALF_45, depth))

    def test_a_pixel_outside_the_frame_is_a_caller_bug(self) -> None:
        # Not None: None would mean "no answer" and a caller would retry forever.
        # A pixel outside [0, 1] cannot come from `clank.intent/1`.
        for pixel in ((-0.1, 0.5), (1.1, 0.5), (0.5, -0.01), (0.5, 1.01)):
            with self.assertRaises(ValueError):
                projection.unproject(pixel, IDENTITY, HALF_45, 1.0)

    def test_a_non_finite_pixel_or_depth_is_a_caller_bug(self) -> None:
        for pixel in ((float("nan"), 0.5), (0.5, float("inf"))):
            with self.assertRaises(ValueError):
                projection.unproject(pixel, IDENTITY, HALF_45, 1.0)
        with self.assertRaises(ValueError):
            projection.unproject((0.5, 0.5), IDENTITY, HALF_45, float("nan"))


class ServoStep(unittest.TestCase):
    """One iteration of the pixel servo: where to put the wrist, and how wrong we are."""

    def test_a_wrist_already_on_target_stops(self) -> None:
        target, error = projection.servo_step(
            (0.5, 0.5), (0.5, 0.5), IDENTITY, HALF_45, 1.0
        )
        self.assertIsNone(target)
        self.assertAlmostEqual(error, 0.0)

    def test_a_wrist_inside_the_deadband_stops(self) -> None:
        target, error = projection.servo_step(
            (0.5, 0.5), (0.503, 0.5), IDENTITY, HALF_45, 1.0, deadband=0.01
        )
        self.assertIsNone(target)
        self.assertAlmostEqual(error, 0.003)

    def test_an_off_target_wrist_gets_the_point_under_the_target_pixel(self) -> None:
        target, error = projection.servo_step(
            (0.2, 0.75), (0.5, 0.5), IDENTITY, HALF_45, 1.5
        )
        assert target is not None
        self.assertGreater(error, 0.0)
        back = projection.project_point(IDENTITY, target, HALF_45)
        assert back is not None
        self.assertAlmostEqual(back[0], 0.2, places=6)
        self.assertAlmostEqual(back[1], 0.75, places=6)

    def test_it_still_reports_the_error_when_it_cannot_answer(self) -> None:
        # A zero depth is unanswerable, but swallowing the error would hide why.
        target, error = projection.servo_step(
            (0.2, 0.75), (0.5, 0.5), IDENTITY, HALF_45, 0.0
        )
        self.assertIsNone(target)
        self.assertGreater(error, 0.0)

    def test_a_negative_deadband_is_a_caller_bug(self) -> None:
        with self.assertRaises(ValueError):
            projection.servo_step(
                (0.5, 0.5), (0.6, 0.5), IDENTITY, HALF_45, 1.0, deadband=-0.1
            )


class ServoFeedback(unittest.TestCase):
    """`servo_step` names a point on the target ray *as seen from `head`*.

    That dependency is the mechanism behind a live divergence: `reach` moves the
    head, so re-issuing the step every iteration chases a ray that keeps turning.
    Pinning the behaviour here means an adapter that loops on it has to be an
    explicit decision, not an accident.
    """

    def test_the_point_moves_when_the_head_moves(self):
        target, wrist = (0.30, 0.50), (0.50, 0.50)
        a, _ = servo_step(target, wrist, IDENTITY, HALF_45, 0.45)
        b, _ = servo_step(target, wrist, yawed(10.0), HALF_45, 0.45)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertNotEqual(tuple(round(v, 6) for v in a),
                            tuple(round(v, 6) for v in b))

    def test_the_point_does_not_depend_on_the_wrist_pixel(self):
        """Only the error does. So a wrist that drifts cannot move the target."""
        target = (0.30, 0.50)
        a, ea = servo_step(target, (0.50, 0.50), IDENTITY, HALF_45, 0.45)
        b, eb = servo_step(target, (0.80, 0.30), IDENTITY, HALF_45, 0.45)
        self.assertEqual(a, b)
        self.assertNotEqual(ea, eb)


if __name__ == "__main__":
    unittest.main()
