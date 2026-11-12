"""Resampling generated 25 Hz motion up to SONIC's 50 Hz, and the flat hold at a clip's end.

Two silent failures this guards. The root orientation must slerp: a lerped quaternion
between two rotations is not a unit quaternion, and SONIC's observation carries per-frame
joint velocities, so the artefact reaches the policy as a velocity it never commanded. And
a clip that simply ends leaves the encoder's ten-frame lookahead reading the final rows -
or past them once a stale index arrives - so the end of a clip is held by repeating the
last row in all ten slots, the measured hold that does not walk (`research/c3m-results.md`,
H1f).

The `jp` order is the part that looks right while being wrong: SONIC's per-frame array is
policy/CSV order, an interleaved left/right permutation of a pose36's hardware-order
hinges, so the tests compare against `g1_pack.jp_from_pose36` rather than to hand indices.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import unittest

WBC = pathlib.Path(__file__).resolve().parents[1] / "spike" / "wbc"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_test", WBC / f"{name}.py")
    if spec is None or spec.loader is None:  # pragma: no cover - a missing file, not a branch
        raise RuntimeError(f"could not load spike/wbc/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


g1_pack = _load("g1_pack")
clip_from_gen = _load("clip_from_gen")

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def ramp(n: int, hz: float) -> list[tuple[float, ...]]:
    """`n` poses at `hz`: root x advances one frame per step, pelvis stays at the identity."""
    return [(i / hz, 0.0, 0.0) + IDENTITY + (0.0,) * 29 for i in range(n)]


def one_hinge(index: int, value: float) -> tuple[float, ...]:
    """A pose at the origin with one hardware-order hinge set and everything else zero."""
    return (0.0, 0.0, 0.0) + IDENTITY + tuple(
        value if i == index else 0.0 for i in range(29)
    )


class Slerp(unittest.TestCase):
    def test_the_half_way_quaternion_is_a_half_turn(self) -> None:
        # identity -> 90 deg about z: halfway is 45 deg about z.
        q = clip_from_gen.slerp_xyzw(IDENTITY, (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)), 0.5)
        self.assertAlmostEqual(q[0], 0.0, places=9)
        self.assertAlmostEqual(q[1], 0.0, places=9)
        self.assertAlmostEqual(q[2], math.sin(math.pi / 8), places=9)
        self.assertAlmostEqual(q[3], math.cos(math.pi / 8), places=9)

    def test_it_takes_the_short_way_round(self) -> None:
        # The same 90 deg rotation written with a negated quaternion: the short arc is still
        # 45 deg forward, not 135 deg back.
        q = clip_from_gen.slerp_xyzw(
            IDENTITY, (0.0, 0.0, -math.sqrt(0.5), -math.sqrt(0.5)), 0.5
        )
        self.assertAlmostEqual(q[2], math.sin(math.pi / 8), places=9)
        self.assertAlmostEqual(q[3], math.cos(math.pi / 8), places=9)

    def test_the_unit_quaternion_among_the_endpoints_is_the_start(self) -> None:
        # q1 == -identity is the same rotation; the interpolation must not move.
        q = clip_from_gen.slerp_xyzw(IDENTITY, (0.0, 0.0, 0.0, -1.0), 0.5)
        self.assertAlmostEqual(abs(q[3]), 1.0, places=9)

    def test_the_ends_are_the_inputs(self) -> None:
        q0, q1 = IDENTITY, (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
        self.assertEqual(clip_from_gen.slerp_xyzw(q0, q1, 0.0), q0)
        self.assertEqual(clip_from_gen.slerp_xyzw(q0, q1, 1.0), q1)

    def test_lerp_is_linear(self) -> None:
        self.assertAlmostEqual(clip_from_gen.lerp(2.0, 4.0, 0.25), 2.5)


class Resample(unittest.TestCase):
    def test_resample_25_to_50_length(self) -> None:
        # 25 frames @ 25 Hz (ARDY) = 1.0 s -> 50 frames.
        frames = ramp(25, 25.0)
        out = clip_from_gen.resample_pose36(frames)
        self.assertEqual(len(out), 50)

    def test_resample_30_to_50_still_supported(self) -> None:
        frames = ramp(30, 30.0)
        out = clip_from_gen.resample_pose36(frames, src_hz=30.0)
        self.assertEqual(len(out), 50)
        self.assertAlmostEqual(out[0][0], frames[0][0])
        self.assertAlmostEqual(out[-1][0], frames[-1][0], places=5)

    def test_upsampling_keeps_the_original_frames_and_adds_midpoints(self) -> None:
        frames = ramp(25, 25.0)
        out = clip_from_gen.resample_pose36(frames)
        self.assertAlmostEqual(out[0][0], frames[0][0])
        self.assertAlmostEqual(out[2][0], frames[1][0])
        self.assertAlmostEqual(out[3][0], (frames[1][0] + frames[2][0]) / 2.0)
        # The last row is held rather than read past the end.
        self.assertAlmostEqual(out[-1][0], frames[-1][0])

    def test_every_resampled_quaternion_stays_a_unit_quaternion(self) -> None:
        frames = []
        for i in range(25):
            a = i * math.radians(7.0)
            frames.append((0.0, 0.0, 0.0, 0.0, 0.0, math.sin(a / 2), math.cos(a / 2))
                          + (0.0,) * 29)
        out = clip_from_gen.resample_pose36(frames)
        for pose in out:
            norm = math.sqrt(sum(v * v for v in pose[3:7]))
            self.assertAlmostEqual(norm, 1.0, places=9)

    def test_hinges_interpolate_too(self) -> None:
        frames = [one_hinge(0, 0.0), one_hinge(0, 0.4)]
        out = clip_from_gen.resample_pose36(frames, src_hz=25.0)
        self.assertEqual(len(out), 4)
        self.assertAlmostEqual(out[1][7], 0.2)
        self.assertAlmostEqual(out[3][7], 0.4)

    def test_an_empty_clip_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            clip_from_gen.resample_pose36([])


class SonicDict(unittest.TestCase):
    def test_sonic_jp_is_csv_order(self) -> None:
        pose = tuple(float(i) for i in range(36))
        d = clip_from_gen.sonic_dict([pose, pose])
        self.assertEqual(d["n"], 2)
        self.assertEqual(d["jp"][0], list(g1_pack.jp_from_pose36(pose)))
        self.assertEqual(d["bq"][0][0], list(g1_pack.xyzw_to_wxyz(pose[3:7])))

    def test_the_root_rides_in_the_pelvis_row_only(self) -> None:
        pose = one_hinge(0, 0.0)
        d = clip_from_gen.sonic_dict([pose, pose])
        self.assertEqual(len(d["bp"]), 2)
        self.assertEqual(len(d["bp"][0]), 14)
        self.assertEqual(len(d["bq"][0]), 14)
        self.assertEqual(d["bp"][0][0], [0.0, 0.0, 0.0])
        self.assertEqual(d["bq"][0][0], [1.0, 0.0, 0.0, 0.0])
        for body in range(1, 14):
            self.assertEqual(d["bp"][0][body], [0.0, 0.0, 0.0])
            self.assertEqual(d["bq"][0][body], [1.0, 0.0, 0.0, 0.0])

    def test_the_root_xyz_and_quaternion_are_carried_across(self) -> None:
        pose = (1.5, -0.25, 0.75, 0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)) + (0.0,) * 29
        d = clip_from_gen.sonic_dict([pose])
        self.assertEqual(d["bp"][0][0], [1.5, -0.25, 0.75])
        self.assertAlmostEqual(d["bq"][0][0][0], math.sqrt(0.5))  # w first
        self.assertAlmostEqual(d["bq"][0][0][3], math.sqrt(0.5))

    def test_joint_velocity_is_per_second_and_csv_ordered(self) -> None:
        # 0.5 rad per frame at 50 Hz is 25 rad/s, on hardware hinge 6 == CSV index 1.
        d = clip_from_gen.sonic_dict(
            [one_hinge(6, 0.0), one_hinge(6, 0.5), one_hinge(6, 1.0)]
        )
        self.assertEqual(len(d["jv"]), 3)
        self.assertAlmostEqual(d["jv"][1][1], 25.0)
        self.assertAlmostEqual(d["jv"][0][1], 25.0)

    def test_the_last_velocity_row_is_a_copy_of_the_one_before(self) -> None:
        d = clip_from_gen.sonic_dict(
            [one_hinge(6, 0.0), one_hinge(6, 0.5), one_hinge(6, 1.0)]
        )
        self.assertEqual(d["jv"][-1], d["jv"][-2])

    def test_a_still_clip_has_no_velocity(self) -> None:
        pose = one_hinge(0, 0.3)
        d = clip_from_gen.sonic_dict([pose, pose, pose])
        self.assertEqual(d["jv"], [[0.0] * 29] * 3)

    def test_resampled_motion_is_fifty_hertz(self) -> None:
        out = clip_from_gen.resample_pose36(ramp(25, 25.0))
        self.assertEqual(clip_from_gen.sonic_dict(out)["n"], 50)


class Lookahead(unittest.TestCase):
    def test_lookahead_flat_at_end(self) -> None:
        self.assertEqual(clip_from_gen.lookahead_idx(99, 100, True), (99,) * 10)
        live = clip_from_gen.lookahead_idx(0, 100, False)
        self.assertEqual(live[0], 0)
        self.assertEqual(live[1], 5)
        self.assertEqual(live[-1], 45)

    def test_a_live_lookahead_saturates_at_the_last_frame(self) -> None:
        live = clip_from_gen.lookahead_idx(95, 100, False)
        self.assertEqual(len(live), 10)
        self.assertEqual(live[0], 95)
        self.assertEqual(live[1:], (99,) * 9)

    def test_clip_tick_flattens_on_last_frame(self) -> None:
        self.assertEqual(clip_from_gen.clip_tick(0, 10), (1, False))
        self.assertEqual(clip_from_gen.clip_tick(9, 10), (9, True))

    def test_a_one_frame_clip_is_flat_from_the_start(self) -> None:
        self.assertEqual(clip_from_gen.clip_tick(0, 1), (0, True))


if __name__ == "__main__":
    unittest.main()
