"""The loop must not quietly become a parser again.

`task_intent` read the verb out of JobBot's sentence, `sequence` then always ran locate and always
reached, and the model was decoration. These are grep tests on purpose: the failure they guard
against is an import and four lines of glue, which no unit test of the brain would ever see, and
which reads perfectly sensibly at the time.
"""

from __future__ import annotations

import ast
import pathlib
import unittest

SRC = pathlib.Path(__file__).resolve().parents[1] / "spike/wbc_game/auto.py"


class AutoLoop(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SRC.read_text()

    def test_does_not_parse_the_instruction(self) -> None:
        for banned in ("brain.task_intent", "brain.next_step", "brain.sequence(",
                       "task_intent(", "next_step(", "sequence("):
            self.assertNotIn(banned, self.source, f"{banned} puts the choreography back in code")

    def test_thinks_before_it_looks(self) -> None:
        self.assertIn("brain.think", self.source)
        self.assertIn("SENSE_BUDGET", self.source)
        turn = self.source[self.source.index("def one_turn("):]
        self.assertLess(turn.index("brain.think"), turn.index("sense(target"),
                        "the model decides before the body finds anything")

    def test_does_not_sweep_head_angles_to_locate(self) -> None:
        # The old sweep located at five yaw offsets and reached at whichever answered first.
        # That is a search the brain never chose, and it is why a wrong object was sometimes
        # reached for with total confidence.
        self.assertNotIn("LOOK_DEG", self.source)

    def test_locates_only_from_the_sense_step(self) -> None:
        calls = [i for i in range(len(self.source)) if self.source.startswith("locate.locate", i)]
        self.assertEqual(len(calls), 1, "one place runs the locator, and it is sense()")
        sense_at = self.source.index("def sense(")
        turn_at = self.source.index("def one_turn(")
        self.assertGreater(calls[0], sense_at)
        self.assertLess(calls[0], turn_at)

    def test_the_grab_close_stays_inside_the_reach(self) -> None:
        # MEASURED 2026-11-09: closing from a separate process left ~2 s in which the hand
        # drifted 0.09 m - more than the grab radius - so the fingers shut on air.
        self.assertIn("grab=skill == \"grab\"", self.source)


class TheActUsesTheGeneratedPath(unittest.TestCase):
    """The generated reach was merged and live on 2026-11-12, and the loop still called
    `reach_vr.py` - so a live run planned `reach(JobBot card)`, located it at 0.42 m, and the
    cruise hand finished more than 20 mm off (exit 2). These are grep tests because importing
    `auto.py` pulls torch, and the failure they guard is one path name.
    """

    def setUp(self) -> None:
        self.source = SRC.read_text()
        self.act = self.source[self.source.index("def grab_pixel("):
                               self.source.index("def _ardy_env(")]
        self.generated = self.source[self.source.index("def _generated_argv("):
                                     self.source.index("def _cruise_argv(")]
        self.cruise = self.source[self.source.index("def _cruise_argv("):]

    def test_the_right_hand_is_generated(self) -> None:
        self.assertIn("reach_ardy.py", self.generated)
        # ardy_gen pins the right hand only, so a left reach would be refused anyway.
        self.assertIn('side == "right"', self.act)

    def test_the_left_hand_keeps_the_cruise(self) -> None:
        self.assertIn("reach_vr.py", self.cruise)

    def test_it_runs_under_the_interpreter_that_has_torch(self) -> None:
        # The body's venv has no torch. `preflight` imports it, and on NixOS torch reports
        # `cuda False` on a working driver unless LD_LIBRARY_PATH carries libcuda.so.1.
        self.assertIn("ARDY_PY", self.source)
        self.assertIn("/run/opengl-driver/lib", self.source)
        self.assertIn("env=_ardy_env()", self.act)

    def test_a_preflight_refusal_falls_back_instead_of_failing_the_act(self) -> None:
        # Exit 3 happens before the body moves (no ring, no right-hand support, no CUDA),
        # so the cruise can carry the act without a half-reach already in flight.
        self.assertIn("ARDY_UNSUPPORTED", self.act)
        self.assertIn("code != ARDY_UNSUPPORTED", self.act)

    def test_the_marked_camera_reaches_both_paths(self) -> None:
        # A pixel is only valid through the camera that saw it; the generated path was the
        # one missing the option, not the cruise.
        self.assertEqual(self.source.count("_marked_camera(camera)"), 2)

    def test_the_generated_reach_does_not_narrow_the_policy_band(self) -> None:
        # The cruise is called with --tol 0.02; the generated path must not be, or the live
        # grab (24.7 mm achieved) would have been reported as a failure. The constants are
        # read from the source tree so the docstring that says "--tol" cannot pass this.
        tree = ast.parse(self.source)
        flags = {
            node.name: {c.value for c in ast.walk(node) if isinstance(c, ast.Constant)
                        and isinstance(c.value, str)}
            for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("--tol", flags["_cruise_argv"], "the cruise keeps its tight band")
        self.assertNotIn("--tol", flags["_generated_argv"])

    def test_confirm_records_a_window_instead_of_listening(self) -> None:
        # The gate's job is "wait for the next line"; confirm's question is "what did the game
        # just say", and the game talks over the grab. MEASURED 2026-11-13: a listen that opened
        # on JobBot measured its floor on the speech and reported silence over an audible line.
        confirm = self.source[self.source.index("def confirm("):self.source.index("def grab_pixel(")]
        self.assertIn("ears.record(", confirm)
        self.assertNotIn("ears.listen(", confirm)
        self.assertIn("max_s: float = 8.0", confirm)

    def test_one_session_memory_lives_in_main(self) -> None:
        main = self.source[self.source.index("def main("):]
        self.assertIn("brain.SessionMemory()", main)
        self.assertEqual(main.count("SessionMemory()"), 1, "one conversation per run")
        # WorkingMemory stays per turn: its pixels are the turn's geometry, not the run's words.
        self.assertNotIn("WorkingMemory()", main)
        self.assertIn("WorkingMemory()", self.source[self.source.index("def one_turn("):])

    def test_think_receives_the_session(self) -> None:
        turn = self.source[self.source.index("def one_turn("):]
        self.assertIn("session=session", turn)

    def test_confirm_feeds_the_next_turn(self) -> None:
        # JobBot's reply after the grab is the next instruction, already transcribed.
        self.assertIn("pending_heard", self.source)
        self.assertIn("pending=spoken_after", self.source)

    def test_the_body_is_reset_once_not_once_per_turn(self) -> None:
        main = self.source[self.source.index("def main("):]
        self.assertIn("ready()", main, "the run stands the body up once")
        turn = self.source[self.source.index("def one_turn("):self.source.index("def main(")]
        self.assertIn("_body_needs_reset(last_code)", turn, "and only on recovery after that")
        # Nothing after the grab may reset: a reset re-opens the hand that holds the card.
        after = turn[turn.index("grab_pixel("):]
        self.assertNotIn("ready(", after)

    def test_refusals_are_results_not_acts(self) -> None:
        self.assertIn('log("result", f"refused ', self.source)
        self.assertNotIn('hold ({thought.decision.reason})', self.source)

    def test_results_reach_the_session(self) -> None:
        self.assertIn("session.note_result(", self.source)
        self.assertIn("session.cap()", self.source)


if __name__ == "__main__":
    unittest.main()
