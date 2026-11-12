"""Path shape, because the reach was rejected for how it moved and every gate measured where it stopped.

The definitions are pinned to the ones already on record - projection onto the start-to-end axis,
4 mm deadband - so a generated path's reversals can be compared with the cruise's 2 / 4 / 13-27
without re-deriving what a reversal is.
"""

import importlib.util
import pathlib
import unittest

WBC = pathlib.Path(__file__).resolve().parents[1] / "spike" / "wbc"
_SPEC = importlib.util.spec_from_file_location("path_shape_test", WBC / "path_shape.py")
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - a missing file, not a branch
    raise RuntimeError("could not load spike/wbc/path_shape.py")
path_shape = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(path_shape)  # type: ignore[union-attr]


def line(start, step, count):
    return [tuple(start[k] + step[k] * i for k in range(3)) for i in range(count)]


class StraightPathTest(unittest.TestCase):
    def test_a_straight_line_is_straight_and_does_not_reverse(self) -> None:
        shape = path_shape.path_shape(line((0.0, 0.0, 0.0), (0.01, 0.0, 0.0), 20))
        self.assertAlmostEqual(shape["straightness"], 1.0, places=3)
        self.assertEqual(shape["reversals"], 0)
        self.assertAlmostEqual(shape["net_m"], 0.19, places=6)

    def test_an_out_and_back_is_half_wasted(self) -> None:
        points = line((0.0, 0.0, 0.0), (0.01, 0.0, 0.0), 20) + line(
            (0.19, 0.0, 0.0), (-0.01, 0.0, 0.0), 19
        )
        shape = path_shape.path_shape(points)
        self.assertLess(shape["straightness"], 0.05)
        self.assertEqual(shape["reversals"], 1)

    def test_a_zig_zag_reverses_often(self) -> None:
        # Forward, back, forward, back: the stutter the generated path is supposed to remove.
        points = []
        for leg in range(6):
            start = 0.1 if leg % 2 else 0.0
            end = 0.0 if leg % 2 else 0.1
            points += [(start, 0.0, 0.0), (end, 0.0, 0.0)]
        self.assertGreaterEqual(path_shape.path_shape(points)["reversals"], 5)


class DeadbandTest(unittest.TestCase):
    def test_steps_under_the_deadband_are_not_intent(self) -> None:
        # A 1 mm wobble each way is tracking noise on a held hand, not a reversal.
        points = []
        for i in range(20):
            points.append((0.001 * (i % 2), 0.0, 0.0))
        self.assertEqual(path_shape.path_shape(points)["reversals"], 0)

    def test_the_deadband_is_reachable_from_the_caller(self) -> None:
        points = []
        for i in range(20):
            points.append((0.001 * (i % 2), 0.0, 0.0))
        self.assertGreater(path_shape.path_shape(points, deadband_m=0.0005)["reversals"], 5)


class DegenerateTest(unittest.TestCase):
    def test_a_body_that_never_moved_reports_zeros(self) -> None:
        shape = path_shape.path_shape([(1.0, 2.0, 3.0)] * 10)
        self.assertEqual(shape["path_m"], 0.0)
        self.assertEqual(shape["straightness"], 1.0)
        self.assertEqual(shape["reversals"], 0)

    def test_a_single_sample_has_no_shape_and_no_crash(self) -> None:
        shape = path_shape.path_shape([(1.0, 2.0, 3.0)])
        self.assertEqual(shape["samples"], 1)
        self.assertEqual(shape["net_m"], 0.0)

    def test_no_samples_at_all(self) -> None:
        self.assertEqual(path_shape.path_shape([])["samples"], 0)


class AxisTest(unittest.TestCase):
    def test_an_arc_forward_does_not_read_as_a_reversal(self) -> None:
        # Sideways travel with forward progress is an arc, not a change of mind.
        points = [(0.0, 0.05 * i / 10, 0.01 * i) for i in range(11)]
        self.assertEqual(path_shape.path_shape(points)["reversals"], 0)
        self.assertGreater(path_shape.path_shape(points)["path_m"], 0.05)


if __name__ == "__main__":
    unittest.main()
