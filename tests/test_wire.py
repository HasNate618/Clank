from __future__ import annotations

import math
import struct
import unittest

from clank import wire


class LayoutTests(unittest.TestCase):
    def test_layout_matches_the_remote_driver(self) -> None:
        wire.assert_layout()
        self.assertEqual(wire.SIZE, 376)
        self.assertEqual(wire.C.sizeof(wire.HeadData), 128)
        self.assertEqual(wire.C.sizeof(wire.ControllerData), 120)

    def test_header_is_the_remote_driver_magic(self) -> None:
        self.assertEqual(struct.pack("<Q", wire.HEADER), b"mndrmt3\0")

    def test_layout_check_notices_a_shrunken_struct(self) -> None:
        original = wire.EXPECTED["r_remote_data"]
        wire.EXPECTED["r_remote_data"] = (376, 375)
        try:
            with self.assertRaises(ValueError):
                wire.assert_layout()
        finally:
            wire.EXPECTED["r_remote_data"] = original


class FrameMapTests(unittest.TestCase):
    def test_basis_vectors_follow_the_measured_map(self) -> None:
        self.assertEqual(wire.mj_to_xr_vec((1.0, 0.0, 0.0)), (0.0, 0.0, -1.0))
        self.assertEqual(wire.mj_to_xr_vec((0.0, 1.0, 0.0)), (-1.0, 0.0, 0.0))
        self.assertEqual(wire.mj_to_xr_vec((0.0, 0.0, 1.0)), (0.0, 1.0, 0.0))

    def test_wrist_rest_pose_lands_on_the_recorded_xr_point(self) -> None:
        # c0_client: MuJoCo L (-0.017, 0.222, 0.721) -> XR (-0.222, 0.721, 0.017)
        self.assertEqual(wire.mj_to_xr_vec((-0.017, 0.222, 0.721)), (-0.222, 0.721, 0.017))

    def test_quaternion_map_agrees_with_the_vector_map(self) -> None:
        # Rotating in MuJoCo then mapping must equal mapping then rotating with
        # the mapped quaternion; that is what makes the map a frame change and
        # not a lucky set of three vectors.
        half = math.radians(37.0) / 2.0
        q_mj = (math.cos(half), 0.0, 0.0, math.sin(half))  # 37 deg about +Z
        q_xr = wire.mj_to_xr_quat(q_mj)
        for v in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.3, -0.7, 0.2)):
            rotated_then_mapped = wire.mj_to_xr_vec(wire.quat_rotate(q_mj, v))
            mapped_then_rotated = wire.quat_rotate(
                (q_xr[3], q_xr[0], q_xr[1], q_xr[2]), wire.mj_to_xr_vec(v)
            )
            for got, want in zip(mapped_then_rotated, rotated_then_mapped, strict=True):
                self.assertAlmostEqual(got, want, places=9)

    def test_identity_stays_identity(self) -> None:
        q = wire.mj_to_xr_quat((1.0, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(q[0], 0.0)
        self.assertAlmostEqual(q[1], 0.0)
        self.assertAlmostEqual(q[2], 0.0)
        self.assertAlmostEqual(q[3], 1.0)


class BodyStateConversionTests(unittest.TestCase):
    @staticmethod
    def state(root_p=(0.0, 0.0, 0.8), root_q=(1.0, 0.0, 0.0, 0.0),
              head_p=(0.0, 0.0, 0.4), head_q=(0.0, 0.0, 0.0, 1.0)) -> dict:
        return {
            "root_p": list(root_p),
            "root_q": list(root_q),
            "head": {"p": list(head_p), "q_xyzw": list(head_q)},
            "left_hand": {"p": [0.0, 0.2, -0.2], "q_xyzw": [0.0, 0.0, 0.0, 1.0]},
            "right_hand": {"p": [0.0, -0.2, -0.2], "q_xyzw": [0.0, 0.0, 0.0, 1.0]},
        }

    def test_root_translation_is_added(self) -> None:
        poses = wire.body_state_to_xr(self.state(root_p=(1.0, 2.0, 0.8), head_p=(0.0, 0.0, 0.4)))
        self.assertAlmostEqual(poses["head"][0][0], -2.0)  # xr_x = -mj_y
        self.assertAlmostEqual(poses["head"][0][1], 1.2)  # xr_y = mj_z (0.8 + 0.4)
        self.assertAlmostEqual(poses["head"][0][2], -1.0)  # xr_z = -mj_x

    def test_root_rotation_is_composed_before_mapping(self) -> None:
        # Root turned 180 deg about Z: a head 1 m along +X relative to the root is
        # 1 m along -X in the world, which maps to +Z. Skipping the root rotation
        # would leave it at -Z and look plausible while being wrong.
        root_q_180 = (0.0, 0.0, 0.0, 1.0)  # w, x, y, z
        poses = wire.body_state_to_xr(self.state(root_q=root_q_180, head_p=(1.0, 0.0, 0.0)))
        p = poses["head"][0]
        self.assertAlmostEqual(p[0], 0.0, places=6)
        self.assertAlmostEqual(p[1], 0.8, places=6)
        self.assertAlmostEqual(p[2], 1.0, places=6)

    def test_head_orientation_inherits_the_root_rotation(self) -> None:
        half = math.radians(90.0) / 2.0
        root_q_90 = (math.cos(half), 0.0, 0.0, math.sin(half))  # w, x, y, z
        poses = wire.body_state_to_xr(self.state(root_q=root_q_90))
        expected = wire.mj_to_xr_quat(root_q_90)
        for got, want in zip(poses["head"][1], expected, strict=True):
            self.assertAlmostEqual(got, want, places=9)

    def test_rest_poses_describe_a_standing_humanoid(self) -> None:
        head, left, right = wire.REST_POSES["head"], wire.REST_POSES["left"], wire.REST_POSES["right"]
        self.assertAlmostEqual(head[0][1], 1.221)  # E3 HMD height
        self.assertLess(left[0][0], 0.0)
        self.assertGreater(right[0][0], 0.0)
        for pose in (left, right):
            self.assertLess(pose[0][1], head[0][1])


class FillTests(unittest.TestCase):
    def test_fill_writes_poses_and_buttons(self) -> None:
        poses = wire.body_state_to_xr(BodyStateConversionTests.state())
        d = wire.fill(
            wire.blank_packet(), poses, trigger=(1.0, 1.0), trigger_click=(True, True)
        )
        self.assertEqual(d.header, wire.HEADER)
        self.assertAlmostEqual(d.head.center.position.x, poses["head"][0][0], places=5)
        self.assertAlmostEqual(d.left.pose.position.y, poses["left"][0][1], places=5)
        self.assertTrue(d.right.trigger_click)
        self.assertAlmostEqual(d.right.trigger_value.x, 1.0)
        self.assertTrue(d.left.active)
        self.assertFalse(d.head.per_view_data_valid)

    def test_fill_sends_buttons_per_hand(self) -> None:
        packet = wire.fill(
            wire.blank_packet(),
            wire.REST_POSES,
            grip=(0.0, 1.0),
            trigger=(1.0, 0.0),
            trigger_click=(True, False),
        )
        self.assertAlmostEqual(packet.left.squeeze_value.x, 0.0)
        self.assertAlmostEqual(packet.right.squeeze_value.x, 1.0)
        self.assertAlmostEqual(packet.left.trigger_value.x, 1.0)
        self.assertAlmostEqual(packet.right.trigger_value.x, 0.0)
        self.assertTrue(packet.left.trigger_click)
        self.assertFalse(packet.right.trigger_click)

    def test_curl_and_squeeze_are_separately_controllable(self) -> None:
        squeeze_only = wire.fill(
            wire.blank_packet(), wire.REST_POSES, grip=(1.0, 1.0), curl=(0.0, 0.0)
        )
        self.assertAlmostEqual(squeeze_only.left.squeeze_value.x, 1.0)
        self.assertAlmostEqual(squeeze_only.left.squeeze_force.x, 1.0)
        self.assertAlmostEqual(squeeze_only.left.hand_curl[0], 0.0)

    def test_curl_alone_moves_no_button(self) -> None:
        curl_only = wire.fill(wire.blank_packet(), wire.REST_POSES, curl=(1.0, 0.0))
        self.assertAlmostEqual(curl_only.left.hand_curl[4], 1.0)
        self.assertAlmostEqual(curl_only.left.squeeze_value.x, 0.0)

    def test_held_trigger_does_not_repeat_the_click(self) -> None:
        held = wire.fill(wire.blank_packet(), wire.REST_POSES, trigger=(1.0, 1.0))
        self.assertFalse(held.left.trigger_click, "a click is an edge, not a level")
        self.assertAlmostEqual(held.left.trigger_value.x, 1.0)
        self.assertTrue(held.left.trigger_touch)


if __name__ == "__main__":
    unittest.main()


class DeclaredFov(unittest.TestCase):
    """We emit the HMD data, so the field of view can be a decision, not a guess.

    `per_view_data_valid` is the driver's own switch for "use `views[].fov`". Upstream
    `drivers/remote/r_interface.h` documents it as "Is the per view data valid and
    should be used?", and `r_head_data.views[2]` carries `{fov, pose, _pad}`. CITED from
    https://monado.pages.freedesktop.org/monado/r__interface_8h_source.html
    (accessed 2026-10-30).

    What is NOT cited: the code that consumes it. That path could not be read from the
    installed store or the raw source, so honouring it is an ASSUMPTION until the
    two-FOV experiment measures it. These tests pin our side of the contract only.
    """

    def test_the_default_leaves_per_view_data_invalid(self) -> None:
        # Unchanged behaviour: a caller that says nothing about FOV keeps the
        # compositor's own default, exactly as before this field existed.
        d = wire.fill(wire.blank_packet(), wire.REST_POSES)
        self.assertFalse(d.head.per_view_data_valid)

    def test_supplying_a_fov_marks_the_per_view_data_valid(self) -> None:
        d = wire.fill(wire.blank_packet(), wire.REST_POSES, fov=(math.radians(55.0), math.radians(45.0)))
        self.assertTrue(d.head.per_view_data_valid)

    def test_a_symmetric_fov_writes_all_four_half_angles(self) -> None:
        half_h = math.radians(55.0)
        half_v = math.radians(45.0)
        d = wire.fill(wire.blank_packet(), wire.REST_POSES, fov=(half_h, half_v))
        for view in d.head.views:
            self.assertAlmostEqual(view.fov.angle_left, half_h, places=6)
            self.assertAlmostEqual(view.fov.angle_right, half_h, places=6)
            self.assertAlmostEqual(view.fov.angle_up, half_v, places=6)
            self.assertAlmostEqual(view.fov.angle_down, half_v, places=6)

    def test_a_negative_or_zero_fov_is_rejected(self) -> None:
        for bad in ((0.0, 0.5), (-0.5, 0.5), (0.5, 0.0), (float("nan"), 0.5), (0.5, float("inf"))):
            with self.assertRaises(ValueError):
                wire.fill(wire.blank_packet(), wire.REST_POSES, fov=bad)

    def test_declaring_a_fov_does_not_move_anything(self) -> None:
        # The experiment changes one variable. If declaring a FOV also perturbed the
        # head pose, a changed image could not be attributed to the field of view.
        plain = wire.fill(wire.blank_packet(), wire.REST_POSES)
        fovved = wire.fill(wire.blank_packet(), wire.REST_POSES, fov=(1.0, 0.8))
        self.assertEqual(plain.head.center.position.x, fovved.head.center.position.x)
        self.assertEqual(plain.head.center.position.y, fovved.head.center.position.y)
        self.assertEqual(plain.head.center.position.z, fovved.head.center.position.z)
        self.assertEqual(plain.head.center.orientation.w, fovved.head.center.orientation.w)


class YawOffset(unittest.TestCase):
    """Choosing the robot's heading in the virtual room.

    The desk needs a 90 deg left turn (operator, 2026-10-30) and the body cannot twist
    that far: a commanded left look reaches only -28.6 deg from a -86.79 baseline
    (measured at the join), against +54.3 right. Torso yaw saturates long before 90.

    So instead of turning the robot, we place it in the room already facing the desk:
    a fixed yaw applied to every emitted pose. This is NOT the same as rotating the
    camera alone - that would leave the hands 90 deg off the view. Rotating the camera
    and both hands together preserves the geometry the camera sees, while the desk's
    bearing relative to the robot changes. The state below pins exactly that: the
    hand's position *in the head's own frame* must not move.
    """

    @staticmethod
    def state(**kw) -> dict:
        return BodyStateConversionTests.state(**kw)

    def test_zero_offset_is_bit_identical(self) -> None:
        # Every earlier measurement was taken at zero offset, so this must be exact.
        plain = wire.body_state_to_xr(self.state())
        offset = wire.body_state_to_xr(self.state(), yaw_offset_rad=0.0)
        self.assertEqual(plain, offset)

    def test_ninety_degrees_left_sends_the_gaze_to_the_robots_left(self) -> None:
        # Head 1 m along the robot's forward (+X in the MJ frame). Turning the robot
        # 90 deg left about Z puts it along +Y, which maps to XR -X.
        poses = wire.body_state_to_xr(
            self.state(head_p=(1.0, 0.0, 0.0)), yaw_offset_rad=math.radians(90.0)
        )
        p = poses["head"][0]
        self.assertAlmostEqual(p[0], -1.0, places=6)
        self.assertAlmostEqual(p[1], 0.8, places=6)
        self.assertAlmostEqual(p[2], 0.0, places=6)

    def test_the_hand_stays_where_the_camera_sees_it(self) -> None:
        # THE invariant. If this moves, the arms no longer point where the view looks
        # and the whole approach is worthless.
        def hand_in_head_frame(yaw_rad: float) -> tuple[float, float, float]:
            poses = wire.body_state_to_xr(
                self.state(root_p=(0.0, 0.0, 0.0), head_p=(0.0, 0.0, 0.0)),
                yaw_offset_rad=yaw_rad,
            )
            head_p, head_q_xyzw = poses["head"]
            hand_p = poses["right"][0]
            # Poses come back in XR order; the quaternion is (x, y, z, w).
            q_wxyz = (head_q_xyzw[3], head_q_xyzw[0], head_q_xyzw[1], head_q_xyzw[2])
            delta = tuple(h - g for h, g in zip(hand_p, head_p, strict=True))
            return wire.quat_rotate(wire.quat_conj(q_wxyz), delta)

        base = hand_in_head_frame(0.0)
        for yaw in (math.radians(90.0), math.radians(-45.0), math.radians(180.0)):
            for got, want in zip(hand_in_head_frame(yaw), base, strict=True):
                self.assertAlmostEqual(got, want, places=6)

    def test_the_head_orientation_is_rotated_too(self) -> None:
        # Rotating the position but not the orientation would make the camera face the
        # old way while standing somewhere else.
        plain = wire.body_state_to_xr(self.state(), yaw_offset_rad=0.0)
        turned = wire.body_state_to_xr(self.state(), yaw_offset_rad=math.radians(90.0))
        self.assertNotEqual(plain["head"][1], turned["head"][1])
        # A quarter turn of the orientation must still be a unit quaternion.
        norm = math.sqrt(sum(c * c for c in turned["head"][1]))
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_a_full_turn_is_the_identity(self) -> None:
        plain = wire.body_state_to_xr(self.state())
        full = wire.body_state_to_xr(self.state(), yaw_offset_rad=2.0 * math.pi)
        for key in ("head", "left", "right"):
            for got, want in zip(full[key][0], plain[key][0], strict=True):
                self.assertAlmostEqual(got, want, places=6)


class DeclaredViewPoses(unittest.TestCase):
    """Declaring a FOV also validates the per-view POSE block, not just the frustum.

    `per_view_data_valid` is documented as "Is the per view data valid and should be
    used?" - the whole `views[]` entry, which is `{fov, pose, _pad}`. While the flag was
    False those poses were zeroed and ignored, so it did not matter. Turning the flag on
    made them authoritative: both eyes at the origin, and an orientation quaternion of
    (0, 0, 0, 0) - norm zero, which is not a rotation at all.

    That is a plausible cause of the client dying the moment the wider view appeared: a
    wider FOV is visible immediately, and degenerate transforms propagate right after.
    """

    def _packet(self, **kw):
        return wire.fill(wire.blank_packet(), wire.REST_POSES, **kw)

    def _fov(self):
        return (math.radians(55.0), math.radians(39.0))

    def test_each_eye_gets_a_unit_rotation(self) -> None:
        # A zero quaternion is not identity. Anything that normalises it divides by zero.
        d = self._packet(fov=self._fov())
        for view in d.head.views:
            q = view.pose.orientation
            norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
            self.assertAlmostEqual(norm, 1.0, places=6)

    def test_the_eyes_are_separated_by_the_interpupillary_distance(self) -> None:
        d = self._packet(fov=self._fov(), ipd_m=0.063)
        left, right = d.head.views
        # OpenXR eye space: +X is the viewer's right.
        self.assertAlmostEqual(left.pose.position.x, -0.0315, places=6)
        self.assertAlmostEqual(right.pose.position.x, 0.0315, places=6)
        self.assertAlmostEqual(
            right.pose.position.x - left.pose.position.x, 0.063, places=6
        )

    def test_the_eye_poses_do_not_drift_vertically_or_forward(self) -> None:
        # Symmetric eyes, level with the head centre: no toe-in, no elevation.
        d = self._packet(fov=self._fov())
        for view in d.head.views:
            self.assertEqual(view.pose.position.y, 0.0)
            self.assertEqual(view.pose.position.z, 0.0)

    def test_an_implausible_ipd_is_refused(self) -> None:
        for bad in (0.0, -0.063, float("nan"), float("inf"), 0.5):
            with self.subTest(ipd=bad), self.assertRaises(ValueError):
                self._packet(fov=self._fov(), ipd_m=bad)

    def test_undeclared_leaves_the_view_poses_alone(self) -> None:
        # The flag is False, so the driver uses its own eyes; writing ours would be
        # inventing data nobody asked for.
        d = self._packet()
        self.assertFalse(d.head.per_view_data_valid)
        for view in d.head.views:
            self.assertEqual(view.pose.position.x, 0.0)
            self.assertEqual(view.pose.orientation.w, 0.0)


class PlayerScaleTests(unittest.TestCase):
    """`scale` sizes the emulated player against a room laid out for an adult.

    The G1 is 1.32 m with ~0.45 m of arm travel and Job Simulator's counter is an adult's
    counter, so the card sat 0.486 m from the wrist - past the arm, which made the body step
    to balance and drag the camera with it. Scaling the head and both hands together makes
    the game-space arm `scale` times longer, so the reach a given object demands falls by the
    same factor. The robot's own dimensions are untouched.
    """

    def setUp(self) -> None:
        self.state = {
            "root_p": [0.0, 0.0, 0.787],
            "root_q": [1.0, 0.0, 0.0, 0.0],
            "head": {"p": [0.0, 0.0, 0.403], "q_xyzw": [0.0, 0.0, 0.0, 1.0]},
            "left_hand": {"p": [0.3, 0.2, 0.1], "q_xyzw": [0.0, 0.0, 0.0, 1.0]},
            "right_hand": {"p": [0.3, -0.2, 0.1], "q_xyzw": [0.0, 0.0, 0.0, 1.0]},
        }

    def test_the_eye_height_grows_with_the_scale(self) -> None:
        plain = wire.body_state_to_xr(self.state)
        tall = wire.body_state_to_xr(self.state, scale=1.43)
        # mj_to_xr_vec puts MuJoCo z on the driver's y, so the height is the middle one.
        self.assertAlmostEqual(tall["head"][0][1] / plain["head"][0][1], 1.43, places=6)

    def test_the_hand_to_head_offset_grows_too_which_is_the_point(self) -> None:
        # If only the head were scaled the hands would detach from the view; if neither were,
        # nothing would change. Both hands and the head move together, so the arm *in game
        # space* gets longer by exactly the scale.
        plain = wire.body_state_to_xr(self.state)
        tall = wire.body_state_to_xr(self.state, scale=1.43)
        for hand in ("left", "right"):
            for axis in range(3):
                offset_plain = plain[hand][0][axis] - plain["head"][0][axis]
                offset_tall = tall[hand][0][axis] - tall["head"][0][axis]
                self.assertAlmostEqual(offset_tall / offset_plain, 1.43, places=6)

    def test_orientation_is_untouched_by_scaling(self) -> None:
        self.state["head"]["q_xyzw"] = [0.0, 0.0, 0.3826834, 0.9238795]
        plain = wire.body_state_to_xr(self.state)
        tall = wire.body_state_to_xr(self.state, scale=1.43)
        self.assertEqual(plain["head"][1], tall["head"][1])

    def test_the_default_leaves_the_poses_bit_identical(self) -> None:
        self.assertEqual(
            wire.body_state_to_xr(self.state), wire.body_state_to_xr(self.state, scale=1.0)
        )

    def test_a_scale_that_would_mirror_or_collapse_is_rejected(self) -> None:
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                wire.body_state_to_xr(self.state, scale=bad)


class PlayerScalePivotTests(unittest.TestCase):
    """Where the scale is measured from decides whether it helps at all.

    Measured 2026-11-03 with the same body pose: scaling about the floor lifted the camera
    (card depth 0.342 -> 0.527 m) and left the card needing 0.566 m of reach, i.e. *worse*
    than the 0.486 m unscaled, because a hand raised 0.25 m gains only 0.19 m of arm. About
    the resting wrist the same scale asks for |O-H|/scale = 0.34 m.
    """

    STATE = {
        "root_p": [0.0, 0.0, 0.787],
        "root_q": [1.0, 0.0, 0.0, 0.0],
        "head": {"p": [0.0, 0.0, 0.403], "q_xyzw": [0.0, 0.0, 0.0, 1.0]},
        "left_hand": {"p": [0.16, 0.257, -0.204], "q_xyzw": [0.0, 0.0, 0.0, 1.0]},
        "right_hand": {"p": [0.16, -0.257, -0.204], "q_xyzw": [0.0, 0.0, 0.0, 1.0]},
    }
    WRIST = (0.16, -0.257, 0.583)

    def test_the_pivot_itself_does_not_move(self) -> None:
        # the whole reason for a pivot: the hand must stay where the body put it
        tall = wire.body_state_to_xr(self.STATE, scale=1.43, pivot=self.WRIST)
        # mj_to_xr_vec: x -> -y, y -> z, z -> -x
        self.assertAlmostEqual(tall["right"][0][2], -self.WRIST[0], places=6)
        self.assertAlmostEqual(tall["right"][0][1], self.WRIST[2], places=6)
        self.assertAlmostEqual(tall["right"][0][0], -self.WRIST[1], places=6)

    def test_a_point_far_from_the_wrist_scales_about_it_not_the_floor(self) -> None:
        plain = wire.body_state_to_xr(self.STATE)
        tall = wire.body_state_to_xr(self.STATE, scale=1.43, pivot=self.WRIST)
        # the head must move away from the wrist by exactly the scale
        for axis in range(3):
            d_plain = plain["head"][0][axis] - plain["right"][0][axis]
            d_tall = tall["head"][0][axis] - tall["right"][0][axis]
            self.assertAlmostEqual(d_tall / d_plain, 1.43, places=6)

    def test_the_floor_pivot_lifts_the_hand_and_the_wrist_pivot_does_not(self) -> None:
        floor = wire.body_state_to_xr(self.STATE, scale=1.43)
        wrist = wire.body_state_to_xr(self.STATE, scale=1.43, pivot=self.WRIST)
        self.assertGreater(floor["right"][0][1], wrist["right"][0][1] + 0.20)

    def test_a_bad_pivot_is_rejected(self) -> None:
        for bad in ((0.0, 0.0), (0.0, float("nan"), 0.0), (1, 2, 3, 4)):
            with self.assertRaises(ValueError):
                wire.body_state_to_xr(self.STATE, scale=1.43, pivot=bad)


class HandsOnlyScaleTests(unittest.TestCase):
    """Scale the hands and leave the head where it is.

    Measured 2026-11-03: with the head scaled too, the camera rises 0.29 m, the depth the
    camera reports changes, the object's apparent position in the robot frame moves with
    it, and 1.43 bought only 0.500 -> 0.472 m. The card is 0.45 m above the resting hand
    and 0.12 m away, so almost the whole reach is vertical and the camera's own motion
    cancels the longer arm. Hands only: camera unchanged, depth still valid, and the hand
    still travels `scale` metres per real metre.
    """

    STATE = {
        "root_p": [0.0, 0.0, 0.787],
        "root_q": [1.0, 0.0, 0.0, 0.0],
        "head": {"p": [0.0, 0.0, 0.403], "q_xyzw": [0.0, 0.0, 0.0, 1.0]},
        "left_hand": {"p": [0.16, 0.257, -0.204], "q_xyzw": [0.0, 0.0, 0.0, 1.0]},
        "right_hand": {"p": [0.30, -0.257, -0.204], "q_xyzw": [0.0, 0.0, 0.0, 1.0]},
    }
    WRIST = (0.16, -0.257, 0.583)

    def test_the_head_does_not_move(self) -> None:
        plain = wire.body_state_to_xr(self.STATE)
        hands = wire.body_state_to_xr(
            self.STATE, scale=1.43, pivot=self.WRIST, scale_head=False
        )
        self.assertEqual(plain["head"][0], hands["head"][0])

    def test_a_metre_of_hand_motion_becomes_one_scale_of_room(self) -> None:
        # the entire point: the reach a given object demands divides by the scale
        plain = wire.body_state_to_xr(self.STATE)
        hands = wire.body_state_to_xr(
            self.STATE, scale=1.43, pivot=self.WRIST, scale_head=False
        )
        right = self.STATE["right_hand"]["p"]
        moved = dict(self.STATE)
        moved["right_hand"] = {"p": [right[0], right[1], right[2] + 0.10],
                              "q_xyzw": [0.0, 0.0, 0.0, 1.0]}
        plain2 = wire.body_state_to_xr(moved)
        hands2 = wire.body_state_to_xr(
            moved, scale=1.43, pivot=self.WRIST, scale_head=False
        )
        d_plain = abs(plain2["right"][0][1] - plain["right"][0][1])
        d_hands = abs(hands2["right"][0][1] - hands["right"][0][1])
        self.assertAlmostEqual(d_hands / d_plain, 1.43, places=6)

    def test_the_default_still_scales_the_head(self) -> None:
        both = wire.body_state_to_xr(self.STATE, scale=1.43, pivot=self.WRIST)
        plain = wire.body_state_to_xr(self.STATE)
        self.assertNotEqual(both["head"][0], plain["head"][0])
