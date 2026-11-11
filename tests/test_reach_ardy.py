"""What the ARDY reach adapter must not do, and the policy it must not re-derive.

Neither the adapter nor the probe can be imported here: `reach_ardy` imports the ARDY
wrapper and writes NumPy clips, and `probe_ardy_track` drives a MuJoCo body process. So the
source text is the contract for the wiring, and the one piece of policy the adapter shares -
the escalation chain - is the real `end_pose.decide_next`, called here the way the adapter
calls it.

The greps are named after measured failures, not style: `STEP_GUARD` and `WRIST_BOX` belong
to the cruise this replaces, and a streamed head pin twisted the torso and moved the frame
the target lived in.
"""

import importlib.util
import pathlib
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
WBC = REPO / "spike" / "wbc"
_SPEC = importlib.util.spec_from_file_location("end_pose_reach_test", WBC / "end_pose.py")
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - a missing file, not a branch
    raise RuntimeError("could not load spike/wbc/end_pose.py")
end_pose = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(end_pose)  # type: ignore[union-attr]

ADAPTER = REPO / "spike" / "wbc_game" / "reach_ardy.py"
PROBE = REPO / "spike" / "wbc_game" / "probe_ardy_track.py"


def function_body(text: str, name: str) -> str:
    """The source of `name`'s function, so a test can say where a decision lives."""
    start = text.index(f"def {name}(")
    end = text.find("\ndef ", start + 1)
    return text[start:] if end < 0 else text[start:end]


class NoCruiseArtefacts(unittest.TestCase):
    def test_no_step_guard_or_wrist_box(self) -> None:
        # Both are artefacts of streaming a Cartesian reference: the step guard caught the
        # walk that cruise induced, and WRIST_BOX is its steering clamp, not the arm's
        # reach. A generated clip has neither.
        text = ADAPTER.read_text()
        self.assertNotIn("STEP_GUARD", text)
        self.assertNotIn("WRIST_BOX", text)

    def test_no_head_pin_and_no_cartesian_cruise(self) -> None:
        text = ADAPTER.read_text()
        for absent in ("AUTH_HEAD", "STREAM_HZ", 'verb="reach"', "clamp_wrist"):
            self.assertNotIn(absent, text)
        self.assertNotIn("STEP_GUARD", PROBE.read_text())


class ChainPolicy(unittest.TestCase):
    def test_the_adapter_obeys_the_real_decision_function(self) -> None:
        text = ADAPTER.read_text()
        self.assertIn("end_pose.decide_next(", text)
        # The older brief's vocabulary was a different function; nothing here re-implements
        # the policy under it.
        for stale in ("planted_once", "def decide_next"):
            self.assertNotIn(stale, text)

    def test_one_escalation_then_the_best_residual(self) -> None:
        # The adapter's two attempts, at the residuals a tracked reach produces: the first
        # spends the root goal, the second has no budget left and must report, not retry.
        self.assertEqual(
            end_pose.decide_next(
                kind="planted", residual_m=0.25, escalated=False,
                gen=1, elapsed_s=1.0, generated=1,
            ),
            "escalate",
        )
        self.assertEqual(
            end_pose.decide_next(
                kind="planted", residual_m=0.20, escalated=True,
                gen=2, elapsed_s=4.0, generated=2,
            ),
            "saturated",
        )

    def test_an_on_target_reach_stops(self) -> None:
        self.assertEqual(
            end_pose.decide_next(
                kind="planted", residual_m=0.06, escalated=False,
                gen=1, elapsed_s=1.0, generated=1,
            ),
            "stop",
        )


class ClaspOnApproach(unittest.TestCase):
    def test_the_clasp_is_decided_while_the_clip_plays(self) -> None:
        # Two seconds between reach and close once cost 0.09 m of drift, so the closure is
        # sent from inside the playback wait, on the achieved hand, not after the clip ends.
        text = ADAPTER.read_text()
        wait = function_body(text, "wait_for_play")
        self.assertIn("AUTH_CLOSURE", wait)
        self.assertIn("end_pose.CLASP_START_M", wait)
        # The hand points in `body_state` are root-relative and the object is a world point;
        # comparing them raw measures nothing (a standing hand sits 0.2 m below the root).
        self.assertIn("reach_vr.to_world(", wait)
        self.assertLess(text.index("AUTH_CLOSURE"), text.index("end_pose.decide_next("))

    def test_the_clasp_threshold_is_the_policy_constant(self) -> None:
        # One number, defined next to the chain the adapter already imports; a local 0.08
        # is how two copies of "close enough" drift apart.
        self.assertEqual(end_pose.CLASP_START_M, 0.08)


class MailboxAndScale(unittest.TestCase):
    def test_the_clip_goes_through_the_session_mailbox(self) -> None:
        text = ADAPTER.read_text()
        for seam in (
            "mb_mailbox.read_pose36(",
            "mb_mailbox.write_play(",
            "mb_mailbox.read_status(",
            "clip_from_gen.resample_pose36(",
        ):
            self.assertIn(seam, text)

    def test_the_object_goes_through_the_declared_player_scale(self) -> None:
        # A reader that assumes 1.0 aims 1.43x too far; the declaration is the only thing
        # that says whether the stream's metres are the robot's.
        text = ADAPTER.read_text()
        self.assertIn("read_player_scale(", text)
        self.assertIn(".pixel_point(", text)
        self.assertIn("--world", text)

    def test_the_relay_yaw_offset_is_never_applied_to_the_camera(self) -> None:
        # Applying it mirrored every left-hand reach; the pixel is valid for the camera that
        # saw it, and the relay's rotation cancels.
        self.assertNotIn("yaw_offset_deg", ADAPTER.read_text())


class Probe(unittest.TestCase):
    def test_the_probe_deposits_the_same_clip_format(self) -> None:
        text = PROBE.read_text()
        self.assertIn("reach_ardy.deposit(", text)
        self.assertIn("mb_mailbox.read_status(", text)
        self.assertIn("mb_mailbox.read_pose36(", text)

    def test_the_probe_measures_the_gate_and_stops_on_a_bad_one(self) -> None:
        text = PROBE.read_text()
        for number in ("TRACK_SUCCESS_M", "STOP_M", "FLAG_FALL_RISK"):
            self.assertIn(number, text)

    def test_the_probe_has_no_cruise_fallback(self) -> None:
        text = PROBE.read_text()
        for absent in ("WRIST_BOX", "STEP_GUARD", "reach_vr.command", "reach_vr.hold"):
            self.assertNotIn(absent, text)
