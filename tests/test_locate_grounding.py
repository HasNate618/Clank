"""The parts of `spike/wbc_game/locate.py` that hold a decision, tested without the model.

The module talks to a local server, so anything that asks a question is untestable here. What
is testable is everything that decides *after* an answer arrives, and that is where the
pipeline's two confident errors were: a box parser that read the `0` in `x0`, and a stage 3
that believed the model's YES about an object the model had just described as something else.
"""

from __future__ import annotations

import importlib.util
import pathlib
import unittest
from collections.abc import Mapping
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "clank_locate", ROOT / "spike" / "wbc_game" / "locate.py"
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - a missing file, not a branch
    raise RuntimeError("could not load spike/wbc_game/locate.py")
locate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(locate)


class DescribeMatchesTests(unittest.TestCase):
    """Measured descriptions, and the verdict each must produce.

    Every description here was actually returned by the model on a crop of the live frame
    (2026-11-09): the badge is "Job Bot card" / "JobBot Museum Curator card", the lever knob
    is "Blue mushroom" / "Blue sphere on a pole", the desk object is "Book" / "Control panel".
    """

    CASES = (
        ("card", "Job Bot card", True),
        ("card", "JobBot Museum Curator card", True),
        ("job cartridge", "JobBot Museum Curator card", False),
        ("job cartridge", "JobBot Museum Curator badge", False),
        ("job cartridge", "Book", False),
        ("job cartridge", "Control panel", False),
        ("blue lever knob", "Blue mushroom", True),
        ("blue lever knob", "Blue sphere on a pole", True),
        ("banana", "Job Bot card", False),
        # Found by the generalisation frame: a shared colour is not an identity.
        ("job cartridge", "green robot computer", False),
        ("blue lever knob", "Blue sign", False),
        ("the JobBot ID card: a white badge with barcodes", "Job Bot card", True),
        ("cartridge", "cartridges", True),
    )

    def test_measured_descriptions_get_the_right_verdict(self) -> None:
        for noun, description, want in self.CASES:
            with self.subTest(noun=noun, description=description):
                matched, why = locate.describe_matches(noun, description)
                self.assertIs(matched, want, why)

    def test_the_reason_names_what_was_seen(self) -> None:
        matched, why = locate.describe_matches("job cartridge", "JobBot Museum Curator card")
        self.assertFalse(matched)
        self.assertIn("curator", why)
        self.assertIn("job cartridge", why)

    def test_an_empty_description_is_not_a_match(self) -> None:
        matched, why = locate.describe_matches("card", "   ")
        self.assertFalse(matched)
        self.assertIn("nothing described", why)

    def test_the_summoner_name_alone_does_not_match(self) -> None:
        # "job" is in both strings; only the noun's own words carry the identity.
        matched, _ = locate.describe_matches("job cartridge", "Job Bot")
        self.assertFalse(matched)

    def test_a_shared_colour_is_not_a_match(self) -> None:
        # Measured on a second frame: this exact pair was accepted before the weak tier was
        # removed, and it pointed the robot at a green computer.
        matched, why = locate.describe_matches("job cartridge", "green robot computer")
        self.assertFalse(matched, why)


class ParseBoxTests(unittest.TestCase):
    def test_keyed_reply_is_read_by_key_name(self) -> None:
        # A parser that takes the first four numbers reads the 0 in `x0` and returns
        # a box at negative zero - measured, and the reason this parser exists.
        self.assertEqual(
            locate.parse_box('{"x0": 105, "y0": 220, "x1": 330, "y1": 405}'),
            (105.0, 220.0, 330.0, 405.0),
        )

    def test_keyed_reply_with_markdown_fence(self) -> None:
        self.assertEqual(
            locate.parse_box('```json\n{"x1": 330, "x0": 105, "y1": 405, "y0": 220}\n```'),
            (105.0, 220.0, 330.0, 405.0),
        )

    def test_bare_numbers_still_parse(self) -> None:
        self.assertEqual(locate.parse_box("105, 220 - 330, 405"), (105.0, 220.0, 330.0, 405.0))

    def test_a_refusal_is_not_a_box(self) -> None:
        self.assertIsNone(locate.parse_box('{"none": true}'))


class RefusalTests(unittest.TestCase):
    """The refusal contract: a box the model cannot have is what the old pipeline invented."""

    FRAME = None  # ask_box is patched, so nothing is ever sent anywhere

    def _read(self, replies: tuple[tuple[tuple[float, float, float, float] | None, str], ...]):
        calls = {"n": 0}

        def fake_ask_box(config, frame, noun, box, width, pad):
            reply = replies[min(calls["n"], len(replies) - 1)]
            calls["n"] += 1
            return reply

        with patch.object(locate, "ask_box", fake_ask_box):
            # Serial on purpose: the script means "reply N for call N", which concurrency
            # cannot promise. ParallelReadTests owns the concurrent contract.
            return locate.read_box(None, None, "job cartridge", (0.0, 0.0, 1.0, 1.0),
                                   parallel=1)

    def test_a_majority_of_refusals_is_a_refusal(self) -> None:
        median, boxes, raws, refused = self._read(((None, '{"none": true}'),))
        self.assertTrue(refused)
        self.assertIsNone(median)
        self.assertEqual(boxes, [])
        self.assertEqual(len(raws), 5)

    def test_two_refusals_out_of_five_still_yield_a_box(self) -> None:
        replies = (
            (None, '{"none": true}'),
            (None, '{"none": true}'),
            ((0.2, 0.2, 0.4, 0.4), '{"x0": 200, "y0": 200, "x1": 400, "y1": 400}'),
        )
        median, boxes, _, refused = self._read(replies)
        self.assertFalse(refused)
        self.assertEqual(median, (0.2, 0.2, 0.4, 0.4))
        self.assertEqual(len(boxes), 3)

    def test_the_median_is_taken_per_edge(self) -> None:
        # One wild read in five cannot move an edge the way it moves a mean. With four
        # wild reads it would, which is the failure mode the ensemble count guards.
        wild = (0.0, 0.0, 1.0, 1.0)
        good = (0.70, 0.70, 0.90, 0.90)
        good_raw = '{"x0": 700, "y0": 700, "x1": 900, "y1": 900}'
        wild_raw = '{"x0": 0, "y0": 0, "x1": 1000, "y1": 1000}'
        replies = ((good, good_raw), (good, good_raw), (wild, wild_raw),
                   (good, good_raw), (good, good_raw))
        median, boxes, _, refused = self._read(replies)
        self.assertFalse(refused)
        self.assertEqual(median, good)
        self.assertEqual(len(boxes), 5)

    def test_a_majority_of_wild_reads_is_not_rescued_by_the_median(self) -> None:
        # The honest limit: the median survives a minority of bad reads, not a majority.
        wild = (0.0, 0.0, 1.0, 1.0)
        good = (0.70, 0.70, 0.90, 0.90)
        good_raw = '{"x0": 700, "y0": 700, "x1": 900, "y1": 900}'
        wild_raw = '{"x0": 0, "y0": 0, "x1": 1000, "y1": 1000}'
        replies = ((good, good_raw), (wild, wild_raw), (wild, wild_raw),
                   (wild, wild_raw), (wild, wild_raw))
        median, _, _, _ = self._read(replies)
        self.assertEqual(median, wild)


class ParallelReadTests(unittest.TestCase):
    """Five reads at once must decide exactly what five reads in a row decided.

    The server serves four slots (`/props` reports `total_slots: 4`), and the five stage-2
    framings are five independent questions, so they can be asked in one wave. Both the
    per-edge median and the majority refusal are decisions about the *set* of replies, so
    the concurrency must be invisible in the answer - only in `times`.
    """

    BOX = (0.70, 0.70, 0.90, 0.90)
    BOX_RAW = '{"x0": 700, "y0": 700, "x1": 900, "y1": 900}'
    WILD = (0.00, 0.00, 1.00, 1.00)
    WILD_RAW = '{"x0": 0, "y0": 0, "x1": 1000, "y1": 1000}'
    REFUSAL = (None, '{"none": true}')

    #: One reply per send width, so the fake does not depend on the order it is called in.
    BOXES = {
        640: (BOX, BOX_RAW),
        720: ((0.72, 0.70, 0.92, 0.90), '{"x0": 720, "y0": 700, "x1": 920, "y1": 900}'),
        800: (WILD, WILD_RAW),
        880: ((0.70, 0.72, 0.90, 0.92), '{"x0": 700, "y0": 720, "x1": 900, "y1": 920}'),
        960: ((0.71, 0.70, 0.91, 0.90), '{"x0": 710, "y0": 700, "x1": 910, "y1": 900}'),
    }
    REFUSED = {640: REFUSAL, 720: REFUSAL, 800: REFUSAL, 880: (BOX, BOX_RAW), 960: (BOX, BOX_RAW)}

    def _read(
        self,
        replies: Mapping[int, tuple[tuple[float, float, float, float] | None, str]],
        parallel: int,
    ):
        def fake_ask_box(config, frame, noun, box, width, pad):
            return replies[width]

        times: list[float] = []
        with patch.object(locate, "ask_box", fake_ask_box):
            got = locate.read_box(
                None, None, "card", (0.0, 0.0, 1.0, 1.0), times=times, parallel=parallel
            )
        return got, times

    def test_the_median_does_not_depend_on_read_order(self) -> None:
        for name, replies in (("five boxes", self.BOXES), ("three refusals", self.REFUSED)):
            with self.subTest(name):
                serial, serial_times = self._read(replies, parallel=1)
                at_once, at_once_times = self._read(replies, parallel=4)
                self.assertEqual(at_once[0], serial[0], "median box")
                # Order-sensitive, and the canned replies are distinct per width: an
                # implementation that collected in completion order would match on median and
                # refusal but fail here, and `stage2_raw` would stop lining up with `reads`.
                self.assertEqual(at_once[1], serial[1], "boxes")
                self.assertEqual(at_once[2], serial[2], "raws")
                self.assertIs(at_once[3], serial[3], "refusal flag")
                self.assertEqual(len(serial_times), 5)
                self.assertEqual(len(at_once_times), 5)


if __name__ == "__main__":
    unittest.main()


class StageThreeVoteTests(unittest.TestCase):
    """Stage 3 is a vote of closed questions, and every one of these was measured.

    MEASURED 2026-11-13 on JobBot's card: `DESCRIBE_PROMPT` on that crop reads the card's own
    printed text back - "JobBot", "Job Bot sign", "A label" - so the right object is refused. The
    closed question is right on every case measured: the card says "Card" 3/3, the console's OFF
    button "Sign" 3/3, the glove and the console "Something else" and "Sign" 3/3 each.
    """

    def test_the_question_offers_a_way_to_say_no(self) -> None:
        prompt = locate.VERIFY_PROMPT
        self.assertIn("{noun}", prompt)
        self.assertIn("something else", prompt.lower(),
                      "a two-way question let the OFF button answer 'card' 3/3")
        self.assertNotIn("yes or no", prompt.lower())

    def _locate(self, descriptions: list[str]):
        with patch.object(locate, "ask_quadrant") as quadrant, \
                patch.object(locate, "read_box") as read_box, \
                patch.object(locate, "describe_crop") as describe:
            quadrant.return_value = locate.brain.Ground(
                noun="card", box=(0.5, 0.5, 1.0, 1.0), quadrant="BOTTOM-RIGHT", raw="BOTTOM-RIGHT")
            box = (0.80, 0.77, 0.99, 0.95)
            read_box.return_value = (box, [box], ['{"x0": 800, "y0": 770, "x1": 990, "y1": 950}'], False)
            describe.side_effect = lambda *a, **k: descriptions.pop(0)
            return locate.locate(None, None, "card", (960, 540))

    def test_a_majority_of_card_reads_is_a_present(self) -> None:
        out = self._locate(["Card", "Card", "Sign"])
        self.assertIs(out["present"], True)
        # The reads run on threads, so the votes arrive in scheduling order: count them.
        self.assertEqual(sum(1 for v in out["verify_votes"] if v), 2)

    def test_a_majority_of_other_things_is_a_refusal(self) -> None:
        out = self._locate(["Sign", "Something else", "Card"])
        self.assertIs(out["present"], False)

    def test_reads_that_name_nothing_are_not_a_refusal(self) -> None:
        # "JobBot" alone is all stop words: the model told us nothing, which is not a denial.
        out = self._locate(["JobBot", "JobBot", "JobBot"])
        self.assertIsNone(out["present"])
        self.assertEqual(out["match"], "the reads did not name it")


class TheRefusalChangesTheQuestion(unittest.TestCase):
    """A refused candidate must not be looked at again.

    MEASURED, run 193516-ca1fde: seven attempts, one picture, one question, and the same console
    every time - the terminal the robot stands at, panel-labelled INSERT CARTRIDGE at 0.19 m.
    The rack of cartridges was a metre to the right the whole time.
    """

    def test_a_box_in_a_refused_place_sends_stage_two_beside_it(self) -> None:
        console = (0.45, 0.46, 0.59, 0.80)          # the box the model kept returning
        calls: list[tuple[float, float, float, float]] = []

        def fake_read(config, frame, noun, box, plan, times=None, parallel=4):
            calls.append(box)
            # The model answers with the console box again, whatever crop it is given: that is
            # what temperature 0 on an unchanged picture does.
            return console, [console], ["{}"], False

        with patch.object(locate, "ask_quadrant") as quadrant, \
                patch.object(locate, "read_box", side_effect=fake_read), \
                patch.object(locate, "describe_crop", return_value="Job cartridge"):
            quadrant.return_value = locate.brain.Ground(
                noun="job cartridge", box=(0.0, 0.5, 0.5, 1.0), quadrant="BOTTOM-LEFT",
                raw="BOTTOM-LEFT")
            out = locate.locate(None, None, "job cartridge", (960, 540), avoid=(console,))

        self.assertEqual(out["quadrant"], "BOTTOM-RIGHT", "the search must move beside the refusal")
        self.assertEqual(len(calls), 2, "one re-ask, not a loop")
        # Quadrants overlap by MARGIN, so compare where each crop is centred, not its edges.
        self.assertLess((calls[0][0] + calls[0][2]) / 2, 0.35, "the first crop is the left")
        self.assertGreater((calls[1][0] + calls[1][2]) / 2, 0.65, "the second crop is the right")

    def test_nothing_is_re_asked_when_the_box_is_new(self) -> None:
        fresh = (0.80, 0.20, 0.95, 0.40)
        with patch.object(locate, "ask_quadrant") as quadrant, \
                patch.object(locate, "read_box") as read_box, \
                patch.object(locate, "describe_crop", return_value="Job cartridge"):
            quadrant.return_value = locate.brain.Ground(
                noun="job cartridge", box=(0.5, 0.0, 1.0, 0.5), quadrant="TOP-RIGHT",
                raw="TOP-RIGHT")
            read_box.return_value = (fresh, [fresh], ["{}"], False)
            out = locate.locate(None, None, "job cartridge", (960, 540),
                                avoid=((0.45, 0.46, 0.59, 0.80),))
        self.assertEqual(out["quadrant"], "TOP-RIGHT")
        self.assertEqual(read_box.call_count, 1)

    def test_a_box_touching_the_refusal_at_its_edge_is_not_a_repeat(self) -> None:
        # Neighbours in one quadrant share an edge constantly; only the middle decides.
        self.assertFalse(locate._overlaps((0.59, 0.46, 0.70, 0.60), ((0.45, 0.46, 0.59, 0.80),)))
        self.assertTrue(locate._overlaps((0.50, 0.50, 0.60, 0.60), ((0.45, 0.46, 0.59, 0.80),)))

    def test_beside_mirrors_and_keeps_the_row(self) -> None:
        self.assertEqual(locate._beside("BOTTOM-LEFT"), "BOTTOM-RIGHT")
        self.assertEqual(locate._beside("TOP-RIGHT"), "TOP-LEFT")
