"""The player-scale declaration: written by the relay, read by anything that unprojects.

The declaration is the only thing that says whether the depth stream's metres are the
robot's or an adult-sized emulated player's. A reader that assumes 1.0 aims 1.43x too far,
so absence, corruption and a real declaration have to be three different answers.
"""

import json
import tempfile
import unittest
from pathlib import Path

from clank.body import Pose
from clank.player_scale import FILENAME, PlayerScale, apply, path, read, write
from clank.projection import Fov, project_point


class Defaults(unittest.TestCase):
    def test_a_session_that_declared_nothing_is_unscaled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            declared = read(tmp)
        self.assertEqual(declared.scale, 1.0)
        self.assertEqual(declared.camera_scale, 1.0)
        self.assertEqual(declared.pivot, (0.0, 0.0, 0.0))

    def test_the_path_sits_beside_the_body_state(self) -> None:
        # A relative directory, so this is a naming test and not a claim about the host.
        self.assertEqual(path("body-state").name, FILENAME)
        self.assertEqual(path("body-state"), Path("body-state") / FILENAME)


class RoundTrip(unittest.TestCase):
    def test_what_the_relay_writes_is_what_a_reader_gets(self) -> None:
        written = PlayerScale(scale=1.43, pivot=(0.1, -0.2, 0.35),
                              head_scaled=False, yaw_offset_deg=90.0)
        with tempfile.TemporaryDirectory() as tmp:
            where = write(written, tmp)
            self.assertTrue(where.exists())
            got = read(tmp)
        self.assertEqual(got, written)

    def test_the_file_is_readable_json_and_leaves_no_scratch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            where = write(PlayerScale(scale=1.2), tmp)
            data = json.loads(where.read_text())
            leftovers = [p.name for p in Path(tmp).iterdir() if p.name != FILENAME]
        self.assertEqual(data["scale"], 1.2)
        self.assertEqual(data["head_scaled"], True)
        self.assertEqual(leftovers, [])


class CameraScale(unittest.TestCase):
    def test_the_camera_is_only_scaled_when_the_head_is(self) -> None:
        # The distinction `research/g1-scale.md` turns on: hands-only scaling leaves the
        # camera where every depth measurement was taken.
        self.assertEqual(PlayerScale(scale=1.43, head_scaled=True).camera_scale, 1.43)
        self.assertEqual(PlayerScale(scale=1.43, head_scaled=False).camera_scale, 1.0)


class Apply(unittest.TestCase):
    def test_a_game_point_is_divided_about_the_pivot(self) -> None:
        declared = PlayerScale(scale=2.0, pivot=(0.0, 0.0, 0.0))
        self.assertEqual(apply((0.8, -0.4, 0.2), declared), (0.4, -0.2, 0.1))

    def test_the_pivot_itself_does_not_move(self) -> None:
        declared = PlayerScale(scale=1.43, pivot=(0.1, -0.2, 0.35))
        self.assertEqual(apply((0.1, -0.2, 0.35), declared), (0.1, -0.2, 0.35))


class Refusals(unittest.TestCase):
    def test_a_mirroring_scale_is_rejected(self) -> None:
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                PlayerScale(scale=bad)

    def test_a_pivot_that_is_not_a_point_is_rejected(self) -> None:
        for bad in ((1.0, 2.0), (1.0, 2.0, float("inf"))):
            with self.assertRaises(ValueError):
                # Deliberately the wrong shape: the dataclass has to refuse it at runtime,
                # which is the whole point of the check.
                PlayerScale(pivot=bad)  # type: ignore[arg-type]

    def test_corrupt_json_raises_rather_than_assuming_unscaled(self) -> None:
        # Silently falling back to 1.0 would aim every reach 1.43x too far and look like a
        # grounding failure. Stopping is the honest answer.
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / FILENAME).write_text("{not json")
            with self.assertRaises(ValueError):
                read(tmp)

    def test_a_bad_pivot_in_the_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / FILENAME).write_text(json.dumps({"scale": 1.4, "pivot": [1, 2]}))
            with self.assertRaises(ValueError):
                read(tmp)

    def test_a_bad_scale_in_the_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / FILENAME).write_text(json.dumps({"scale": 0.0}))
            with self.assertRaises(ValueError):
                read(tmp)


class PixelPoint(unittest.TestCase):
    """The seam that applies the declaration, not just its two numbers.

    `scale` and `camera_scale` are exercised above, and on their own they let an uncallable
    `pixel_point` sit under a green suite: the module was restored against a
    `point_from_pixel` that takes `pivot`/`hands_scale`, and the one on this branch did not,
    so every call raised TypeError. `reach_vr`'s 1.43x miss is the same seam, so the two
    configurations get a call each.
    """

    PIVOT = (0.10, -0.20, 0.35)
    SCALE = 1.43
    HEAD = Pose((0.02, 0.00, 0.40), (0.0, 0.0, 0.0, 1.0))
    OBJECT = (0.32, -0.36, 0.05)
    FOV = Fov.from_degrees(70.0, 70.0, 57.11, 57.11)

    def game_point(self, point: tuple[float, float, float]) -> tuple[float, float, float]:
        """The game-space point for a robot-frame `point`: `pivot + scale * (point - pivot)`."""
        return (
            self.PIVOT[0] + (point[0] - self.PIVOT[0]) * self.SCALE,
            self.PIVOT[1] + (point[1] - self.PIVOT[1]) * self.SCALE,
            self.PIVOT[2] + (point[2] - self.PIVOT[2]) * self.SCALE,
        )

    def frame(self, depth: float) -> dict:
        return {"depth": [depth] * (48 * 27), "width": 48, "height": 27}

    def pixel_and_depth(self, camera: Pose, target: tuple[float, float, float]):
        pixel = project_point(camera, target, self.FOV)
        assert pixel is not None
        # Identity rotation, so the body +x axis is the gaze and the depth the stream
        # carries is the forward component, not the range.
        return pixel[0], pixel[1], target[0] - camera.p[0]

    def test_a_scaled_camera_and_hands_round_trip_to_the_robot_frame(self) -> None:
        declared = PlayerScale(scale=self.SCALE, pivot=self.PIVOT, head_scaled=True)
        camera = Pose(self.game_point(self.HEAD.p), self.HEAD.q_xyzw)
        u, v, depth = self.pixel_and_depth(camera, self.game_point(self.OBJECT))
        got = declared.pixel_point(self.frame(depth), u, v, self.HEAD, self.FOV)
        assert got is not None
        for i in range(3):
            self.assertAlmostEqual(got[i], self.OBJECT[i], places=6)

    def test_hands_only_scaling_round_trips_with_the_camera_left_alone(self) -> None:
        # `--player-scale-head off`: the camera is the robot's own, and only the hand's
        # travel is scaled, so the division about the pivot is the whole transform.
        declared = PlayerScale(scale=self.SCALE, pivot=self.PIVOT, head_scaled=False)
        u, v, depth = self.pixel_and_depth(self.HEAD, self.game_point(self.OBJECT))
        got = declared.pixel_point(self.frame(depth), u, v, self.HEAD, self.FOV)
        assert got is not None
        for i in range(3):
            self.assertAlmostEqual(got[i], self.OBJECT[i], places=6)
