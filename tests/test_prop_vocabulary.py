"""The plan may only aim at what the locator can confirm.

Measured live on 2026-11-12 (`research/evidence/ardy-in-the-loop/`): the run heard JobBot, and
the plan answered `reach(the Job Simulator display/interface (central teal 'JOB SIMULATOR' sign
and console))`. `locate` refused that noun three times — correctly, it is not in this job's
vocabulary — while the card sat 0.28 m away in the same frame, and the loop ended on `hold`.

The console is a destination. A target has to be one of the props the locator knows, so the
seed and the step prompt name them, and this test binds that list to the locator's own so the
two cannot drift apart in silence.
"""

from __future__ import annotations

import importlib.util
import pathlib
import unittest

from clank import brain

ROOT = pathlib.Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "clank_locate_vocabulary", ROOT / "spike" / "wbc_game" / "locate.py"
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - a missing file, not a branch
    raise RuntimeError("could not load spike/wbc_game/locate.py")
locate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(locate)


class TheVocabularyIsShared(unittest.TestCase):
    def test_every_prop_the_prompt_names_is_one_the_locator_recognises(self) -> None:
        for noun in brain.PROP_NOUNS:
            self.assertIn(noun, locate._MATCH_SYNONYMS,
                          f"{noun} is offered to the plan but stage 3 cannot confirm it")

    def test_every_recognised_noun_is_offered_to_the_plan(self) -> None:
        # The other direction: a *prop* the locator knows and the prompt never mentions is a
        # target that only arrives by luck. Keys are per-word, and several words name the same
        # prop (`badge` is the card, `knob` is the lever), so compare the distinct synonym sets
        # and require a prop the prompt names to carry each of them.
        offered = {frozenset(locate._MATCH_SYNONYMS[noun]) for noun in brain.PROP_NOUNS}
        for word, synonyms in locate._MATCH_SYNONYMS.items():
            self.assertIn(frozenset(synonyms), offered,
                          f"locate can confirm {word} but the plan is never told it is there")

    def test_the_seed_names_every_prop_and_says_what_the_console_is(self) -> None:
        # A prop has several names (`badge` is the card, `knob` is the lever), and the seed may
        # use whichever reads naturally: what matters is that each prop the locator confirms is
        # named once, and that the console is ruled out as a target.
        seed = brain.DEFAULT_SEED.lower()
        for noun in brain.PROP_NOUNS:
            self.assertTrue(any(word in seed for word in locate._MATCH_SYNONYMS[noun]),
                            f"the seed never names {noun} or any of its names")
        self.assertIn("console", seed)
        self.assertIn("not a thing to pick up", seed,
                      "without this the console reads as the most prominent thing to reach")

    def test_the_step_prompt_names_them_too(self) -> None:
        prompt = brain.NEXT_STEP_PROMPT.lower()
        for noun in brain.PROP_NOUNS:
            self.assertIn(noun, prompt)


class TheNounThatFailedIsStillNotLocatable(unittest.TestCase):
    """The regression fixture is the phrase from the run, not a paraphrase of it."""

    FAILED = ("the Job Simulator display/interface "
              "(central teal 'JOB SIMULATOR' sign and console)")

    def test_the_console_is_not_a_synonym_of_any_prop(self) -> None:
        words = set()
        for synonyms in locate._MATCH_SYNONYMS.values():
            words |= set(synonyms)
        for word in ("console", "display", "interface", "sign", "simulator"):
            self.assertNotIn(word, words)

    def test_the_failed_phrase_matches_no_prop(self) -> None:
        for noun in brain.PROP_NOUNS:
            matched, _why = locate.describe_matches(noun, self.FAILED)
            self.assertFalse(matched, f"{noun} accepted a console description")
