"""Pixel + depth -> 3D point.

The depth stream (`clank/depth_stream.py`) publishes 480x270 linear metres from the same
eye as the colour it carries, so the depth no longer has to be guessed: a marked pixel
becomes a point in one lookup. What needs care is the *sampling*: a single texel is
unreliable on an edge, and averaging across a depth discontinuity invents a surface that
is not there - the card is thin enough for that to matter.
"""

import math
import unittest

from clank.body import Pose
from clank.pixel3d import point_from_pixel, sample_depth
from clank.projection import Fov, project_point, unproject

FOV = Fov.from_degrees(70.0, 70.0, 57.11, 57.11)
HEAD = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def frame(value: float, width: int = 48, height: int = 27) -> dict:
    return {"depth": [value] * (width * height), "width": width, "height": height}


def sample(f: dict, u: float = 0.5, v: float = 0.5, **kw) -> float | None:
    return sample_depth(f["depth"], f["width"], f["height"], u, v, **kw)


class SampleDepth(unittest.TestCase):
    def test_a_uniform_region_returns_its_value(self) -> None:
        got = sample(frame(0.42))
        assert got is not None
        self.assertAlmostEqual(got, 0.42, places=6)

    def test_no_return_is_not_a_distance(self) -> None:
        # 0.0 is "nothing there" in this stream, not a surface at zero metres.
        self.assertIsNone(sample(frame(0.0)))

    def test_nan_at_the_centre_is_no_answer(self) -> None:
        self.assertIsNone(sample(frame(float("nan"))))

    def test_an_edge_does_not_average_two_surfaces(self) -> None:
        # Left half of the patch is 0.40 m, right half 3.00 m. The centre texel sits on
        # the near surface, so the answer must be the near surface - an average would
        # report a 1.70 m surface that does not exist anywhere in the scene.
        w, h = 48, 27
        depth = [0.40] * (w * h)
        cx, cy = int(0.5 * w), int(0.5 * h)
        # Start at cx + 1: the centre texel must sit on the *near* surface, otherwise the
        # test measures the window rather than the rule.
        for y in range(cy - 1, cy + 2):
            for x in range(cx + 1, cx + 3):
                depth[y * w + x] = 3.00
        got = sample_depth(depth, w, h, 0.5, 0.5, patch=3)
        assert got is not None
        self.assertAlmostEqual(got, 0.40, places=6)

    def test_an_isolated_spike_near_the_centre_is_rejected(self) -> None:
        w, h = 48, 27
        depth = [0.50] * (w * h)
        cx, cy = int(0.5 * w), int(0.5 * h)
        depth[(cy + 1) * w + cx + 1] = 0.01  # a stray texel from an edge
        got = sample_depth(depth, w, h, 0.5, 0.5, patch=3)
        assert got is not None
        self.assertAlmostEqual(got, 0.50, places=6)


class PointFromPixel(unittest.TestCase):
    def pixel_of(self, camera: Pose, point: tuple) -> tuple[float, float]:
        pixel = project_point(camera, point, FOV)
        assert pixel is not None
        return pixel[0], pixel[1]

    def test_depth_is_the_forward_component_not_the_range(self) -> None:
        # A point 1.0 m forward and 1.0 m to the left is 1.414 m away but has a forward
        # component of exactly 1.0. `unproject` wants the forward one, so feeding this
        # depth must recover that point exactly.
        u, v = self.pixel_of(HEAD, (1.0, 1.0, 0.0))
        got = point_from_pixel(frame(1.0), u, v, HEAD, FOV)
        assert got is not None
        for k in range(3):
            self.assertAlmostEqual(got[k], (1.0, 1.0, 0.0)[k], places=6)

    def test_a_round_trip_recovers_the_pixel(self) -> None:
        point = (0.8, -0.3, 0.2)
        u, v = self.pixel_of(HEAD, point)
        got = point_from_pixel(frame(0.8), u, v, HEAD, FOV)
        assert got is not None
        back = project_point(HEAD, got, FOV)
        assert back is not None
        self.assertAlmostEqual(back[0], u, places=6)
        self.assertAlmostEqual(back[1], v, places=6)

    def test_no_depth_gives_no_point(self) -> None:
        self.assertIsNone(point_from_pixel(frame(0.0), 0.5, 0.5, HEAD, FOV))


class PlayerScale(unittest.TestCase):
    """The stream measures game space; the robot frame is game space divided by `scale`.

    `clank.wire.body_state_to_xr` sizes the emulated player against the room, so the game's
    camera sits `scale` times the robot's head away. A depth read from that camera is
    therefore `scale` times the robot-frame distance along the same gaze, and forgetting to
    divide brings every reach back `scale` times too large - straight past the arm and into
    the body-step regime that invalidated the camera in the first place.
    """

    def test_a_marked_pixel_comes_back_scale_times_closer(self) -> None:
        f = frame(0.49)
        plain = point_from_pixel(f, 0.5, 0.5, HEAD, FOV)
        scaled = point_from_pixel(f, 0.5, 0.5, HEAD, FOV, head_scale=1.43)
        assert plain is not None and scaled is not None
        self.assertAlmostEqual(plain[0] / scaled[0], 1.43, places=6)

    def test_off_axis_pixels_scale_too(self) -> None:
        # Not just the forward axis: the lateral component is what decides whether a reach
        # lands on the object, and it is the component the body leans to supply.
        f = frame(0.60)
        plain = point_from_pixel(f, 0.72, 0.38, HEAD, FOV)
        scaled = point_from_pixel(f, 0.72, 0.38, HEAD, FOV, head_scale=1.6)
        assert plain is not None and scaled is not None
        for a, b in zip(plain, scaled, strict=True):
            self.assertAlmostEqual(a / b, 1.6, places=6)

    def test_the_default_is_the_robot_frame_unchanged(self) -> None:
        f = frame(0.42)
        self.assertEqual(
            point_from_pixel(f, 0.5, 0.5, HEAD, FOV),
            point_from_pixel(f, 0.5, 0.5, HEAD, FOV, head_scale=1.0),
        )

    def test_a_scale_that_would_mirror_the_world_is_rejected(self) -> None:
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                point_from_pixel(frame(0.4), 0.5, 0.5, HEAD, FOV, head_scale=bad)

    def test_the_card_is_inside_the_arm_once_the_player_is_adult_sized(self) -> None:
        # The measurement this change exists for: 0.486 m of game distance is 0.34 m of
        # real arm travel at 1.43, against a 0.40 m standing clamp.
        f = frame(0.486)
        p = point_from_pixel(f, 0.5, 0.5, HEAD, FOV, head_scale=1.43)
        assert p is not None
        self.assertLess(math.dist((0.0, 0.0, 0.0), p), 0.40)


class PlayerScaleComposition(unittest.TestCase):
    """Undoing `pivot + scale * (point - pivot)` is two steps, not one.

    The camera may have moved and the hands may have grown, and `--player-scale-head off`
    is exactly the configuration where those differ. Getting one without the other is the
    silent 1.43x error: `research/g1-scale.md` measured a 4% gain once, and the camera was
    unprojected through the robot's own head while the game rendered from a taller one.
    """

    PIVOT = (0.10, -0.20, 0.35)
    K = 1.43
    HEAD = Pose((0.02, 0.00, 0.40), (0.0, 0.0, 0.0, 1.0))
    OBJECT = (0.32, -0.36, 0.05)

    def game_point(self, point: tuple, scale: float | None = None) -> tuple:
        k = self.K if scale is None else scale
        return tuple(self.PIVOT[i] + (point[i] - self.PIVOT[i]) * k for i in range(3))

    def look_at(self, camera: Pose, target: tuple) -> tuple[float, float, float]:
        """The pixel and forward depth a camera at `camera` would report for `target`."""
        pixel = project_point(camera, target, FOV)
        assert pixel is not None
        # Identity rotation, so the body x axis is the gaze and the forward component is
        # just the difference along it - the same quantity `depth` carries in the stream.
        return pixel[0], pixel[1], target[0] - camera.p[0]

    def test_a_scaled_player_round_trips_to_the_robot_frame(self) -> None:
        camera = Pose(self.game_point(self.HEAD.p), self.HEAD.q_xyzw)
        target = self.game_point(self.OBJECT)
        u, v, depth = self.look_at(camera, target)
        got = point_from_pixel(frame(depth), u, v, self.HEAD, FOV,
                               head_scale=self.K, pivot=self.PIVOT, hands_scale=self.K)
        assert got is not None
        for i in range(3):
            self.assertAlmostEqual(got[i], self.OBJECT[i], places=6)

    def test_hands_only_scaling_still_needs_the_division(self) -> None:
        # `--player-scale-head off`: the camera is the robot's own, the objects are where
        # the game put them, and the hand travels scale metres per commanded metre. The
        # camera composition is 1.0 and the hand division is not.
        target = self.game_point(self.OBJECT)
        u, v, depth = self.look_at(self.HEAD, target)
        got = point_from_pixel(frame(depth), u, v, self.HEAD, FOV,
                               pivot=self.PIVOT, hands_scale=self.K)
        assert got is not None
        for i in range(3):
            self.assertAlmostEqual(got[i], self.OBJECT[i], places=6)

    def test_forgetting_the_hands_aims_scale_times_too_far(self) -> None:
        # The failure this guards, in the configuration where the halves differ. With
        # `--player-scale-head off` the camera is unscaled, so `hands_scale` has no value
        # to inherit and an omission silently returns the game-space point: aim, and land
        # 1.43x out. Pin the size of that mistake.
        target = self.game_point(self.OBJECT)
        u, v, depth = self.look_at(self.HEAD, target)
        good = point_from_pixel(frame(depth), u, v, self.HEAD, FOV,
                                pivot=self.PIVOT, hands_scale=self.K)
        forgot = point_from_pixel(frame(depth), u, v, self.HEAD, FOV, pivot=self.PIVOT)
        assert good is not None and forgot is not None
        self.assertAlmostEqual(
            math.dist(self.PIVOT, forgot) / math.dist(self.PIVOT, good), self.K, places=6
        )

    def test_a_scaled_camera_lends_its_scale_to_the_hands(self) -> None:
        # And the configuration where nothing is droppable: when the whole player is
        # scaled, both halves are the same number and the default is the right one. If
        # that ever stops being true the two-scale test above fails, not this one.
        target = self.game_point(self.OBJECT)
        u, v, depth = self.look_at(Pose(self.game_point(self.HEAD.p), self.HEAD.q_xyzw),
                                   target)
        self.assertEqual(
            point_from_pixel(frame(depth), u, v, self.HEAD, FOV, head_scale=self.K,
                             pivot=self.PIVOT),
            point_from_pixel(frame(depth), u, v, self.HEAD, FOV, head_scale=self.K,
                             pivot=self.PIVOT, hands_scale=self.K),
        )

    def test_an_eye_pivot_is_exactly_the_depth_division(self) -> None:
        # The old implementation, which is why the default is still that behaviour.
        u, v = 0.72, 0.38
        old = point_from_pixel(frame(0.60), u, v, HEAD, FOV, head_scale=2.0)
        new = point_from_pixel(frame(0.60), u, v, HEAD, FOV, head_scale=2.0,
                               pivot=HEAD.p, hands_scale=2.0)
        self.assertEqual(old, new)

    def test_a_pivot_at_the_eye_matches_unproject_directly(self) -> None:
        depth, u, v = 0.55, 0.66, 0.44
        want = unproject((u, v), HEAD, FOV, depth / 1.6)
        got = point_from_pixel(frame(depth), u, v, HEAD, FOV, head_scale=1.6,
                               pivot=HEAD.p)
        assert want is not None and got is not None
        for i in range(3):
            self.assertAlmostEqual(got[i], want[i], places=9)

    def test_a_bad_pivot_or_hand_scale_is_rejected(self) -> None:
        for bad in (0.0, -2.0, float("nan")):
            with self.assertRaises(ValueError):
                point_from_pixel(frame(0.4), 0.5, 0.5, HEAD, FOV, hands_scale=bad)
        for bad in ((1.0, 2.0), (1.0, 2.0, float("inf"))):
            with self.assertRaises(ValueError):
                point_from_pixel(frame(0.4), 0.5, 0.5, HEAD, FOV, pivot=bad)
