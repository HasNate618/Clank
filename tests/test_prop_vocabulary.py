"""What the locator will confirm, and the noun it must keep refusing.

Measured live on 2026-11-12 (`research/evidence/ardy-in-the-loop/`): the run heard JobBot, and
the plan answered `reach(the Job Simulator display/interface (central teal 'JOB SIMULATOR' sign
and console))`. `locate` refused that noun three times — correctly, it is not in this job's
vocabulary — while the card sat 0.28 m away in the same frame, and the loop ended on `hold`.

Since 2026-11-13 the planner prompt names no props at all (spec §3.2): the model chooses the
noun in its own words and `locate` is the only place that decides whether it can be confirmed.
So what is pinned here is the refusal, not a menu shared with the prompt.
"""

from __future__ import annotations

import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "clank_locate_vocabulary", ROOT / "spike" / "wbc_game" / "locate.py"
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - a missing file, not a branch
    raise RuntimeError("could not load spike/wbc_game/locate.py")
locate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(locate)


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
        for noun in locate._MATCH_SYNONYMS:
            matched, _why = locate.describe_matches(noun, self.FAILED)
            self.assertFalse(matched, f"{noun} accepted a console description")


if __name__ == "__main__":
    unittest.main()
