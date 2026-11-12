"""The 36-float pose conversion between MuJoCo `qpos`, MotionBricks, and SONIC's `jp`.

Every failure this guards against is silent: the deployed model interleaves a 7-joint hand
after each arm, so the 29 body joints are not contiguous in `qpos`; MuJoCo stores the base
quaternion `wxyz` while MotionBricks stores `xyzw`; and SONIC's per-frame `jp` is in policy
order, which is a different permutation of the same 29 hinges again. Each mismatch still
produces a plausible-looking 29-vector, so the tests compare exact indices.
"""

from __future__ import annotations

import importlib.util
import pathlib
import unittest

WBC = pathlib.Path(__file__).resolve().parents[1] / "spike" / "wbc"
_SPEC = importlib.util.spec_from_file_location("g1_pack_test", WBC / "g1_pack.py")
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - a missing file, not a branch
    raise RuntimeError("could not load spike/wbc/g1_pack.py")
g1_pack = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(g1_pack)


class Packing(unittest.TestCase):
    def test_qadr_skips_the_hands(self) -> None:
        self.assertEqual(g1_pack.QPOS_QADR[:22], tuple(range(7, 29)))
        self.assertEqual(g1_pack.QPOS_QADR[22:], tuple(range(36, 43)))
        self.assertEqual(len(g1_pack.JOINT_NAMES), 29)
        self.assertEqual(g1_pack.JOINT_NAMES[15], "left_shoulder_pitch_joint")
        self.assertEqual(g1_pack.JOINT_NAMES[22], "right_shoulder_pitch_joint")

    def test_round_trip_leaves_hands_untouched(self) -> None:
        qpos = [float(i) for i in range(50)]
        pose = g1_pack.pose36_from_qpos(qpos)
        self.assertEqual(len(pose), 36)
        self.assertEqual(pose[3:7], g1_pack.wxyz_to_xyzw(tuple(qpos[3:7])))
        out = g1_pack.apply_pose36(qpos, pose)
        self.assertEqual(out[29:36], qpos[29:36])
        self.assertEqual(out[43:50], qpos[43:50])
        self.assertEqual(g1_pack.pose36_from_qpos(out), pose)

    def test_jp_uses_csv_order_not_hardware(self) -> None:
        pose = tuple(float(i) for i in range(36))
        jp = g1_pack.jp_from_pose36(pose)
        self.assertEqual(jp[0], pose[7 + 0])   # left_hip_pitch
        self.assertEqual(jp[1], pose[7 + 6])   # right_hip_pitch (CSV2MJ[1] == 6)
        self.assertEqual(jp[2], pose[7 + 12])  # waist_yaw

    def test_policy_q_is_not_a_pose36(self) -> None:
        # 29-vector (policy q) must not be accepted as pose36
        with self.assertRaises(ValueError):
            g1_pack.pose36_from_qpos([0.0] * 29)

    def test_context4_repeats_when_short(self) -> None:
        one = tuple(float(i) for i in range(36))
        ctx = g1_pack.context4_from_ring([one])
        self.assertEqual(ctx, (one, one, one, one))


class RoundTrip(unittest.TestCase):
    """The full qpos -> pose36 -> jp -> pose36 -> qpos loop the plan's packing test runs."""

    qpos = [0.1 * i for i in range(50)]

    def test_pose36_and_jp_are_inverse(self) -> None:
        pose = g1_pack.pose36_from_qpos(self.qpos)
        jp = g1_pack.jp_from_pose36(pose)
        back = g1_pack.pose36_from_jp(pose[0:3], pose[3:7], jp)
        self.assertEqual(back, pose)

    def test_hands_hold_their_own_values_through_the_loop(self) -> None:
        pose = g1_pack.pose36_from_qpos(self.qpos)
        out = g1_pack.apply_pose36(self.qpos, pose)
        self.assertEqual(out[29:36], self.qpos[29:36])
        self.assertEqual(out[36:43], self.qpos[36:43])
        self.assertEqual(out[43:50], self.qpos[43:50])

    def test_the_base_quaternion_is_reordered_both_ways(self) -> None:
        q = (1.0, 2.0, 3.0, 4.0)
        self.assertEqual(g1_pack.wxyz_to_xyzw(q), (2.0, 3.0, 4.0, 1.0))
        self.assertEqual(g1_pack.xyzw_to_wxyz(q), (4.0, 1.0, 2.0, 3.0))
        self.assertEqual(g1_pack.xyzw_to_wxyz(g1_pack.wxyz_to_xyzw(q)), q)


class Context(unittest.TestCase):
    def ring(self) -> list[tuple[float, ...]]:
        return [tuple(float(i * 1000 + j) for j in range(36)) for i in range(8)]

    def test_takes_four_oldest_first_at_a_third_of_the_rate(self) -> None:
        ring = self.ring()
        ctx = g1_pack.context4_from_ring(ring)
        self.assertEqual(len(ctx), 4)
        self.assertEqual(ctx[-1], ring[-1])
        self.assertEqual(ctx, (ring[1], ring[3], ring[5], ring[7]))

    def test_at_thirty_hertz_every_frame_is_a_context_slot(self) -> None:
        ring = self.ring()
        self.assertEqual(g1_pack.context4_from_ring(ring, hz=30.0),
                         (ring[4], ring[5], ring[6], ring[7]))

    def test_an_empty_ring_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            g1_pack.context4_from_ring([])


if __name__ == "__main__":
    unittest.main()
