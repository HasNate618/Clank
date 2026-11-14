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

    def test_the_marked_camera_reaches_every_path(self) -> None:
        # A pixel is only valid through the camera that saw it; the generated path was the one
        # missing the option, not the cruise. A count would break every time a path is added,
        # which is what happened when `_move_argv` arrived - so the invariant is per builder.
        builders = ("_generated_argv", "_cruise_argv", "_move_argv")
        for name in builders:
            body = self.source[self.source.index(f"def {name}("):
                                self.source.index("def ", self.source.index(f"def {name}(") + 1)]
            self.assertIn("_marked_camera(camera)", body, name)

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

    def test_confirm_reads_the_act_back_out_of_the_ring(self) -> None:
        # MEASURED, run 181738-ee1d6d: JobBot's reply ran 14.9 s *during* a generated reach and
        # the old fixed window opened one second after it ended, so the diary said "no word from
        # JobBot" over a line he had just finished saying. The ring holds the act's own seconds.
        confirm = self.source[self.source.index("def confirm("):self.source.index("def grab_pixel(")]
        self.assertIn("RING.read_utterance(after", confirm,
                      "the read must end where the speech ends, not at the clock")
        self.assertIn('source="act"', confirm)
        # And the old window stays as the fallback, for a run with no ring.
        self.assertIn("ears.record(", confirm)
        self.assertNotIn("ears.listen(", confirm, "a listen opens on JobBot and gates itself deaf")

    def test_the_mark_is_taken_before_the_act(self) -> None:
        turn = self.source[self.source.index("def one_turn("):self.source.index("def main(")]
        self.assertLess(turn.index("act_mark = RING.mark()"), turn.index("grab_pixel("),
                        "a mark taken after the act cannot contain the act's speech")
        self.assertIn("confirm(pattern, after=act_mark)", turn)

    def test_the_ring_starts_before_the_body_is_reset(self) -> None:
        # The greeting arrives within seconds of the game's window and the reset costs ~6 s:
        # measured, 12.8 s of JobBot's opening played while the body was being reset.
        main = self.source[self.source.index("def main("):]
        self.assertLess(main.index("_start_ring("), main.index("ready()"),
                        "a ring started after the reset has already missed the opening line")
        self.assertIn("ring.stop()", main)

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

    def test_the_grounder_sees_the_faded_still_and_the_planner_does_not(self) -> None:
        # MEASURED 2026-11-13: raw -> BOTTOM-LEFT, refused, 3/3; faded at the module's own
        # default band -> BOTTOM-LEFT, refused, 3/3; faded at the gate's band -> BOTTOM-RIGHT,
        # 3/3. The band was the whole difference, and the planner needs the room.
        self.assertIn("faded_for_grounding(", self.source)
        sense = self.source[self.source.index("def sense("):self.source.index("def hands_holding(")]
        self.assertIn("faded_for_grounding(png, depth_frame)", sense)
        self.assertIn("locate.locate(", sense)
        # the planner's still is the raw one
        turn = self.source[self.source.index("def one_turn("):self.source.index("def main(")]
        self.assertIn('Frame(0, 0.0, jpeg, "live")', turn)

    def test_the_fade_band_is_the_gate_band(self) -> None:
        # One pair of numbers for "what the model is shown" and "what the body accepts".
        # Otherwise the grounder looks at a world the gate judges by different rules.
        self.assertIn("GROUNDING_BAND_M = (reach_vr.MIN_SURFACE_M, MAX_OBJ_M)", self.source)
        body = self.source[self.source.index("def faded_for_grounding("):self.source.index("def mark(")]
        self.assertIn("GROUNDING_BAND_M", body)
        self.assertIn("fade.fade(", body)

    def test_place_is_wired(self) -> None:
        # `place` was in the skill vocabulary and refused by assemble, so a model that asked for
        # it got a hold and the diary said "not wired yet" - a working vocabulary entry with no
        # implementation behind it.
        turn = self.source[self.source.index("def one_turn("):self.source.index("def main(")]
        self.assertIn('skill == "place"', turn)
        # and the branch runs BEFORE the "not wired yet" line, which no longer covers place
        self.assertLess(turn.index('skill == "place"'), turn.index("is not wired yet"))

    def test_place_hovers_above_the_destination_and_opens(self) -> None:
        body = self.source[self.source.index("def _place("):self.source.index("def one_turn(")]
        self.assertIn("workspace.APPROACH_M", body, "a reach to the pixel puts the hand inside it")
        self.assertIn("place.hover(", body)
        self.assertIn('send(verb=f"open_{side}")', body)
        # An empty hand cannot place, and saying so is the whole difference from a silent no-op.
        self.assertIn("hands empty", body)
        self.assertIn("no destination", body)
        self.assertNotIn("sequence(", body)

    def test_results_reach_the_session(self) -> None:
        self.assertIn("session.note_result(", self.source)
        self.assertIn("session.cap()", self.source)


if __name__ == "__main__":
    unittest.main()


class ATravelIsItsOwnVerb(unittest.TestCase):
    """A pull is a reach that travels, so the loop runs it before it looks for anything.

    The decision lives in `clank/brain.py` as `Intent.offset` - the adapter's own field for a
    displacement - and this file only carries it to the two executables. That split is why the
    assertion is "the offset is consulted before `sense`", not "a goal is parsed here".
    """

    def setUp(self) -> None:
        self.source = SRC.read_text()

    def test_the_move_argv_carries_three_metres_and_keeps_the_grip(self) -> None:
        body = self.source[self.source.index("def _move_argv("):
                            self.source.index("def _number(")]
        self.assertIn('"--move"', body)
        self.assertIn("--grab", body)

    def test_a_travel_runs_before_anything_is_grounded(self) -> None:
        body = self.source[self.source.index("def one_turn("):]
        self.assertIn("intent.offset is not None", body)
        self.assertLess(body.index("intent.offset is not None"), body.index("if seen is None"))

    def test_the_cruise_carries_the_travel_for_both_hands(self) -> None:
        """The cruise, and not the generator, and this one is a measurement rather than a taste.

        MEASURED 2026-11-13, live body, same hand and same 0.20 m request: a generated clip
        travelled 0.20 m and was back at its start 10 s after the process exited, because
        `CommandBus` retires its freeze-hold snapshot only on a real command and a deposited clip
        is not one - so the hold after a clip bakes the *previous* reach's pose and the arm walks
        home. The cruise, which re-commands an absolute target every keepalive, was 7.4 mm off
        and still there 10 s later. Asserting `reach_ardy.py` here again would re-break a pull
        while looking like a regression test.
        """
        body = self.source[self.source.index("def _move_argv("):
                            self.source.index("def _number(")]
        self.assertIn("reach_vr.py", body)
        self.assertIn('"--move"', body)
        self.assertNotIn("reach_ardy.py", body.replace("(`reach_ardy --move`)", ""))


class TheRingIsConsumedOnce(unittest.TestCase):
    """Audio read back for `confirm` must not be heard again by the next `listen`.

    MEASURED, run 193516-ca1fde: JobBot's "grab one of the job cartridges" arrived three times,
    once as the act's own reply and twice more when later turns resumed listening from a cursor
    that `confirm` had not advanced. Every copy went into the session and the planner's prompt.
    """

    def test_confirm_advances_the_cursor(self) -> None:
        source = (pathlib.Path(__file__).resolve().parents[1]
                  / "spike" / "wbc_game" / "auto.py").read_text()
        confirm = source[source.index("def confirm("):source.index("def grab_pixel(")]
        self.assertIn("RING.cursor = RING.mark()", confirm)
        read_at = confirm.index("RING.read_utterance(after")
        self.assertLess(read_at, confirm.index("RING.cursor = RING.mark()"),
                        "the cursor moves after the read, so the read is not skipped")

    def test_the_carried_reply_is_not_written_as_a_second_hearing(self) -> None:
        """JobBot said it once, so the diary says it once.

        MEASURED, run 193516-ca1fde: "Hey, Hillman, take my card. All right, why don't you grab
        one of the job cartridges..." produced three `heard` rows - one from `confirm` reading
        the act back, one from this carry-forward, and one from a `listen` that replayed audio
        the cursor had not consumed. Only the first was a hearing.
        """
        source = (pathlib.Path(__file__).resolve().parents[1]
                  / "spike" / "wbc_game" / "auto.py").read_text()
        turn = source[source.index("def one_turn("):source.index("def main(")]
        start = turn.index("elif pending_heard:")
        carried = turn[start:turn.index("    else:", start)]
        self.assertNotIn('brainlog.log("heard"', carried)


class ARefusalEndsTheTurn(unittest.TestCase):
    """The retry loop must not re-ask a question the frame has already answered.

    MEASURED, run 193516-ca1fde: three `continue`s per turn and three turns, seven identical
    attempts at the console the robot was standing at, each one ~3 s of model calls.
    """

    def setUp(self) -> None:
        self.source = (pathlib.Path(__file__).resolve().parents[1]
                       / "spike" / "wbc_game" / "auto.py").read_text()

    def test_a_refused_locate_does_not_continue_the_search_loop(self) -> None:
        turn = self.source[self.source.index("def one_turn("):self.source.index("def main(")]
        for marker in ('if found.outcome != "present":',):
            parts = turn.split(marker)
            self.assertGreaterEqual(len(parts), 3, "both sense call sites check the outcome")
            for after in parts[1:]:
                head = after[:1000]
                self.assertIn("return Turn(", head,
                              "a refusal must end the turn, not re-plan the same question")
                self.assertNotIn("\n                    continue", head)

    def test_the_refused_place_is_carried_into_the_next_search(self) -> None:
        self.assertIn("REFUSED.setdefault(target, []).append", self.source)
        self.assertEqual(self.source.count("avoid=tuple(REFUSED.get(target, ()))"), 3,
                         "both sense call sites and the retry")

    def test_a_thing_too_close_is_named_as_such(self) -> None:
        # "not visible at 0.19 m" read like a grounding failure. It was the console the robot
        # was standing at, and stage 3 had named it correctly off its own INSERT CARTRIDGE panel.
        sense = self.source[self.source.index("def sense("):self.source.index("def _place(")]
        self.assertIn('outcome = "too_close"', sense)

    def test_the_cartridge_label_is_code_and_not_prompt(self) -> None:
        # Asked for in the prompt, twice; it broke the card step both times, because a prop named
        # in the prompt outranks the sentence the model was told to obey. The loop applies it in
        # code instead, on the way to the grounder.
        from clank import brain
        self.assertNotIn("STORE CLERK", brain.NEXT_STEP_PROMPT)
        self.assertEqual(brain.refine_target("job cartridge"), "STORE CLERK cartridge")
        self.assertIn("brain.refine_target(", self.source)


class ARefusalIsNotTheEndOfTheInitiative(unittest.TestCase):
    """A quiet window after a refusal is spent looking again, not idling.

    MEASURED, run 204320-0213a5: locate refused the cartridge at 20:44:07 and the run then took
    no action at all - `one_turn` returns an empty Turn whenever the gate is quiet, so a refusal
    ended the initiative. The constitution already says what to do: "If find said it is not
    there, look." The body knows where it already looked, so it looks beside it.

    Read as source, like the rest of this file: `auto.py` imports numpy, which lives in the
    spike venv and not in the interpreter the suite runs under.
    """

    def setUp(self) -> None:
        self.source = (pathlib.Path(__file__).resolve().parents[1]
                       / "spike" / "wbc_game" / "auto.py").read_text()

    def retry(self) -> str:
        return self.source[self.source.index("def _proactive_retry("):
                           self.source.index("def _place(")]

    def test_the_quiet_window_is_where_the_retry_happens(self) -> None:
        quiet = self.source[self.source.index("if not hearing.ok:"):]
        self.assertIn("_proactive_retry(", quiet[:600])

    def test_it_looks_once_and_clears_the_errand_first(self) -> None:
        body = self.retry()
        self.assertIn("PENDING_SEARCH = None", body)
        self.assertLess(body.index("PENDING_SEARCH = None"), body.index("sense("),
                        "cleared before the search, so a second refusal cannot loop")

    def test_the_refused_place_rides_into_the_retry(self) -> None:
        self.assertIn("avoid=tuple(REFUSED.get(target, ()))", self.retry())

    def test_a_refusal_arms_the_retry(self) -> None:
        self.assertEqual(self.source.count("PENDING_SEARCH = (target,"), 2,
                         "both refusal sites remember what to look for")

    def test_the_retry_acts_rather_than_holding(self) -> None:
        self.assertIn("grab_pixel(", self.retry())
