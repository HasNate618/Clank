"""The executor's half of a deposited clip: when to load one, when it is played out, and
the look-ahead that holds the body still once it is.

`sonic_executor.py` pulls MuJoCo and ONNX, so nothing here constructs it; the decisions it
makes are extracted into `spike/wbc/mb_play.py` and tested there. The one exception is the
text checks at the end, which read the executor as source to pin the two rules that matter
most: a generated clip is never handed to `SonicRunner.reset` (it teleports the body, and a
generated dict has no `bl`/`ba` so it raises), and the 50 Hz process never gains the
generator's import.

The hold itself is a property of the look-ahead, not of the clip data. The clip's own rows
are played in full and never rewritten; every one of the ten rows the mode-0 encoder reads
is clamped to the clip's last row, so the window saturates on the final pose instead of
promising motion the clip no longer has (`research/c3m-results.md`, H1f).
"""

from __future__ import annotations

import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WBC = ROOT / "spike" / "wbc"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_play", WBC / f"{name}.py")
    if spec is None or spec.loader is None:  # pragma: no cover - a missing file, not a branch
        raise RuntimeError(f"could not load spike/wbc/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mb_play = _load("mb_play")


class PlaySeqChange(unittest.TestCase):
    def test_the_first_request_is_new(self) -> None:
        self.assertTrue(mb_play.on_play_seq_change(None, {"seq": 1, "path": "mb_clip.npz"}))

    def test_the_request_already_loaded_is_not_new(self) -> None:
        self.assertFalse(mb_play.on_play_seq_change(1, {"seq": 1, "path": "mb_clip.npz"}))

    def test_a_later_seq_is_new(self) -> None:
        self.assertTrue(mb_play.on_play_seq_change(1, {"seq": 2, "path": "mb_clip.npz"}))

    def test_an_empty_mailbox_is_not_new(self) -> None:
        self.assertFalse(mb_play.on_play_seq_change(3, None))
        self.assertFalse(mb_play.on_play_seq_change(None, None))


class AfterClip(unittest.TestCase):
    def test_the_last_frame_is_done(self) -> None:
        self.assertEqual(mb_play.after_clip(9, 10), "done")

    def test_an_earlier_frame_is_playing(self) -> None:
        self.assertEqual(mb_play.after_clip(0, 10), "playing")
        self.assertEqual(mb_play.after_clip(8, 10), "playing")

    def test_a_frame_past_the_end_stays_done(self) -> None:
        self.assertEqual(mb_play.after_clip(10, 10), "done")
        self.assertEqual(mb_play.after_clip(99, 10), "done")

    def test_a_one_frame_clip_is_done_at_frame_zero(self) -> None:
        self.assertEqual(mb_play.after_clip(0, 1), "done")


class LookaheadRows(unittest.TestCase):
    def test_while_the_clip_lasts_no_row_is_clamped(self) -> None:
        # 60 rows, window 0..45: the whole 0.9 s is still inside the clip.
        self.assertEqual(
            mb_play.lookahead_rows(0, 60), (0, 5, 10, 15, 20, 25, 30, 35, 40, 45)
        )

    def test_on_the_last_frame_every_row_is_the_final_pose(self) -> None:
        self.assertEqual(mb_play.lookahead_rows(59, 60), (59,) * 10)

    def test_only_the_rows_past_the_end_saturate(self) -> None:
        # Frame 54 of 60: the near row is real, the nine beyond it are the final pose.
        self.assertEqual(mb_play.lookahead_rows(54, 60), (54,) + (59,) * 9)

    def test_no_row_indexes_past_the_clip(self) -> None:
        for frame in range(60):
            for row in mb_play.lookahead_rows(frame, 60):
                self.assertGreaterEqual(row, 0)
                self.assertLess(row, 60)

    def test_a_frame_past_the_end_reads_the_last_row(self) -> None:
        self.assertEqual(mb_play.lookahead_rows(99, 60), (59,) * 10)

    def test_a_one_row_clip_is_a_constant_window(self) -> None:
        self.assertEqual(mb_play.lookahead_rows(0, 1), (0,) * 10)

    def test_an_empty_clip_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            mb_play.lookahead_rows(0, 0)


class ExecutorSource(unittest.TestCase):
    def setUp(self) -> None:
        self.text = (WBC / "sonic_executor.py").read_text()

    def test_executor_does_not_import_the_generator(self) -> None:
        self.assertNotIn("ardy_gen", self.text)
        self.assertNotIn("end_pose", self.text)

    def test_executor_resets_only_at_construction(self) -> None:
        # One reset, in `__init__`, puts the body on the standing clip at startup. A second
        # call site would be playback calling it, which teleports the body and raises on a
        # generated dict's missing `bl`/`ba`.
        self.assertEqual(self.text.count(".reset("), 1, "playback must not call reset()")


if __name__ == "__main__":
    unittest.main()
