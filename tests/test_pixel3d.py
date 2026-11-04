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
        self.assertAlmostEqual(sample(frame(0.42)), 0.42, places=6)

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
    def test_depth_is_the_forward_component_not_the_range(self) -> None:
        # A point 1.0 m forward and 1.0 m to the left is 1.414 m away but has a forward
        # component of exactly 1.0. `unproject` wants the forward one, so feeding this
        # depth must recover that point exactly.
        u, v = project_point(HEAD, (1.0, 1.0, 0.0), FOV)
        self.assertIsNotNone(u)
        got = point_from_pixel(frame(1.0), u, v, HEAD, FOV)
        assert got is not None
        for k in range(3):
            self.assertAlmostEqual(got[k], (1.0, 1.0, 0.0)[k], places=6)

    def test_a_round_trip_recovers_the_pixel(self) -> None:
        point = (0.8, -0.3, 0.2)
        u, v = project_point(HEAD, point, FOV)
        self.assertIsNotNone(u)
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
        for a, b in zip(plain, scaled):
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
