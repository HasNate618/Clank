"""The diary the brain writes to: one JSON row per sentence."""

import io
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from clank import brainlog


class LogTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self._tmp.name) / "brain" / "messages.jsonl"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def rows(self) -> list[dict]:
        return brainlog.read(self.path)

    def test_one_line_per_message(self):
        brainlog.log("think", "I will pick up the card", path=self.path, echo=False)
        brainlog.log("act", "reaching", path=self.path, echo=False)
        self.assertEqual([r["text"] for r in self.rows()],
                         ["I will pick up the card", "reaching"])

    def test_rows_are_json_objects_on_disk(self):
        brainlog.log("heard", "take my card", path=self.path, echo=False)
        raw = self.path.read_text().strip().splitlines()
        self.assertEqual(len(raw), 1)
        self.assertEqual(json.loads(raw[0])["kind"], "heard")

    def test_parent_directories_are_created(self):
        self.assertFalse(self.path.parent.exists())
        brainlog.log("heard", "hello human", path=self.path, echo=False)
        self.assertTrue(self.path.exists())

    def test_the_line_is_flushed_to_stderr_as_it_happens(self):
        err = io.StringIO()
        brainlog.log("think", "I will pick up the card", path=self.path, stream=err)
        self.assertIn("think: I will pick up the card", err.getvalue())

    def test_whitespace_is_collapsed_into_one_sentence(self):
        row = brainlog.log("heard", "take my\n  card\tnow", path=self.path, echo=False)
        self.assertEqual(row["text"], "take my card now")

    def test_unknown_kind_becomes_think(self):
        row = brainlog.log("muse", "hmm", path=self.path, echo=False)
        self.assertEqual(row["kind"], "think")

    def test_numbers_are_rounded_and_none_is_dropped(self):
        row = brainlog.log("result", "ok", path=self.path, echo=False,
                           ms=12.3456, code=None, skill="grab")
        self.assertEqual(row["ms"], 12.35)
        self.assertNotIn("code", row)
        self.assertEqual(row["skill"], "grab")

    def test_carries_time_so_a_reader_can_order_it(self):
        row = brainlog.log("heard", "hi", path=self.path, echo=False)
        self.assertIsInstance(row["t"], float)
        self.assertEqual(len(row["iso"]), 8)

    def test_a_torn_last_line_does_not_hide_the_rest(self):
        brainlog.log("heard", "one", path=self.path, echo=False)
        with self.path.open("a") as handle:
            handle.write('{"kind": "thin')
        self.assertEqual([r["text"] for r in self.rows()], ["one"])

    def test_reading_a_missing_file_is_empty_not_an_error(self):
        self.assertEqual(brainlog.read(self.path), [])

    def test_tail_returns_the_last_n(self):
        for word in ("one", "two", "three"):
            brainlog.log("heard", word, path=self.path, echo=False)
        self.assertEqual([r["text"] for r in brainlog.tail(2, self.path)], ["two", "three"])

    def test_an_unwritable_diary_does_not_break_the_run(self):
        blocked = pathlib.Path(self._tmp.name) / "readonly"
        blocked.mkdir()
        blocked.chmod(0o500)
        err = io.StringIO()
        try:
            with mock.patch("sys.stderr", err):
                row = brainlog.log("think", "still thinking", path=blocked / "no" / "x.jsonl")
            self.assertEqual(row["text"], "still thinking")
            self.assertIn("cannot write", err.getvalue())
        finally:
            blocked.chmod(0o700)

    def test_env_override_points_the_diary_elsewhere(self):
        with mock.patch.dict(os.environ, {"CLANK_BRAIN_LOG": str(self.path)}):
            self.assertEqual(brainlog.default_path(), self.path)
            brainlog.log("heard", "hi", echo=False)
        self.assertEqual(len(self.rows()), 1)


class RenderTest(unittest.TestCase):
    def test_extras_are_shown_so_the_line_stands_alone(self):
        line = brainlog.render({"kind": "act", "text": "grab 'card'", "skill": "grab",
                                "side": "right", "code": 0, "iso": "11:22:33"})
        self.assertIn("act: grab 'card'", line)
        self.assertIn("skill=grab", line)
        self.assertIn("code=0", line)

    def test_no_extras_is_just_the_sentence(self):
        self.assertEqual(brainlog.render({"kind": "think", "text": "ok"}).count("("), 0)


class CliTest(unittest.TestCase):
    def test_tail_prints_the_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "m.jsonl"
            brainlog.log("think", "I will pick up the card", path=path, echo=False)
            out = io.StringIO()
            with mock.patch("sys.stdout", out), mock.patch("sys.stderr", io.StringIO()):
                self.assertEqual(brainlog.main(["--path", str(path)]), 0)
            self.assertIn("I will pick up the card", out.getvalue())

    def test_empty_diary_exits_one(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("sys.stdout", io.StringIO()), \
                mock.patch("sys.stderr", io.StringIO()):
            self.assertEqual(brainlog.main(["--path", str(pathlib.Path(tmp) / "none.jsonl")]), 1)


if __name__ == "__main__":
    unittest.main()
