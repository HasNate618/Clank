"""The executor's half of a deposited clip: when to load one, when it is played out, and
the look-ahead that holds the body still once it is.

`sonic_executor.py` pulls MuJoCo and ONNX, so nothing here constructs it; the decisions it
makes are extracted into `spike/wbc/mb_play.py` and tested there - including
`Playback`, the tick ordering that decides when the last clip row is played and which seq
a progress write may name. The one exception is the text checks at the end, which read the
executor as source to pin the two rules that matter most: a generated clip is never handed
to `SonicRunner.reset` (it teleports the body, and a generated dict has no `bl`/`ba` so it
raises), and the 50 Hz process never gains the generator's import.

The hold itself is a property of the look-ahead, not of the clip data. The clip's own rows
are played in full and never rewritten; every one of the ten rows the mode-0 encoder reads
is clamped to the clip's last row, so the window saturates on the final pose instead of
promising motion the clip no longer has (`research/c3m-results.md`, H1f).
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil
import tempfile
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
mb_mailbox = _load("mb_mailbox")


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


def _play(seq: int) -> dict:
    return {"seq": seq, "path": "mb_clip.npz"}


class PlaybackClip(unittest.TestCase):
    """The 50 Hz ordering: a tick observes `frame`, then `advance()` reports on it."""

    def test_the_last_row_is_played_before_done(self) -> None:
        play = mb_play.Playback()
        play.started(1, 3)
        reported = []
        for _ in range(3):
            frame = play.frame  # the row this tick builds its observation from
            reported.append((frame, play.advance()))
        # Frame 2 is observed and then reported `done`. Advancing before deciding skipped
        # it: frames 0..1 played, mode 1 took the body back on the third tick.
        self.assertEqual(
            reported, [(0, "playing"), (1, "playing"), (2, "done")]
        )

    def test_every_row_of_a_longer_clip_is_played(self) -> None:
        play = mb_play.Playback()
        play.started(1, 10)
        played = []
        while True:
            played.append(play.frame)
            if play.advance() == "done":
                break
        self.assertEqual(played, list(range(10)))

    def test_a_one_row_clip_plays_its_row_then_finishes(self) -> None:
        play = mb_play.Playback()
        play.started(1, 1)
        self.assertEqual(play.frame, 0)
        self.assertEqual(play.advance(), "done")
        self.assertEqual(play.frame, 0)

    def test_a_tick_after_done_stays_done_on_the_last_row(self) -> None:
        play = mb_play.Playback()
        play.started(1, 2)
        play.advance()
        self.assertEqual(play.advance(), "done")
        self.assertEqual(play.advance(), "done")
        self.assertEqual(play.frame, 1)

    def test_a_started_request_is_not_loaded_twice(self) -> None:
        play = mb_play.Playback()
        self.assertTrue(play.pending(_play(1)))
        play.started(1, 4)
        self.assertFalse(play.pending(_play(1)))
        self.assertTrue(play.pending(_play(2)))

    def test_a_rejected_request_is_not_retried_every_tick(self) -> None:
        play = mb_play.Playback()
        self.assertTrue(play.pending(_play(7)))
        play.rejected(7)
        self.assertFalse(play.pending(_play(7)))
        self.assertTrue(play.pending(_play(8)))

    def test_a_failed_reload_keeps_the_playing_clips_seq(self) -> None:
        play = mb_play.Playback()
        play.started(1, 4)
        play.advance()
        # Seq 2 is deposited and cannot be read: the request is taken, the clip is not.
        self.assertTrue(play.pending(_play(2)))
        play.rejected(2)
        self.assertEqual(play.clip_seq, 1, "progress must name the clip that ran")
        self.assertEqual(play.frame, 1)
        self.assertEqual(play.advance(), "playing")
        self.assertEqual(play.clip_seq, 1)
        self.assertEqual(play.frame, 2)


class ProgressWrite(unittest.TestCase):
    """Reporting is best-effort: the 50 Hz tick owns the physics, not the mailbox."""

    def test_a_report_that_lands_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = pathlib.Path(tmp)
            self.assertIsNone(mb_play.write_progress(session, 1, "playing", 0, 4))
            self.assertEqual(mb_mailbox.read_status(session)["status"], "playing")

    def test_a_removed_session_directory_is_returned_not_raised(self) -> None:
        # The measured failure: a session directory rotated or removed under a running
        # body made the atomic replace raise FileNotFoundError out of `step()`.
        with tempfile.TemporaryDirectory() as tmp:
            session = pathlib.Path(tmp) / "body"
            session.mkdir()
            shutil.rmtree(session)
            self.assertIsNotNone(mb_play.write_progress(session, 1, "playing", 0, 4))

    def test_an_unknown_status_is_returned_not_raised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNotNone(
                mb_play.write_progress(pathlib.Path(tmp), 1, "halfway", 0, 4)
            )

    def test_a_failed_report_does_not_stop_the_clip(self) -> None:
        play = mb_play.Playback()
        play.started(1, 3)
        with tempfile.TemporaryDirectory() as tmp:
            session = pathlib.Path(tmp) / "body"
            session.mkdir()
            shutil.rmtree(session)
            for _ in range(3):
                status = play.advance()
                self.assertIsNotNone(
                    mb_play.write_progress(session, play.clip_seq, status, play.frame, play.n)
                )
        # The rows were still played out and the clip still owns its own seq.
        self.assertEqual((play.clip_seq, play.frame, play.n), (1, 2, 3))


class WarnLatchOnce(unittest.TestCase):
    """One failed mailbox write is worth one line, and each write needs its own latch."""

    def test_a_failure_warns_once(self) -> None:
        latch = mb_play.WarnLatch()
        self.assertEqual(latch.warn("directory gone"), "directory gone")
        self.assertIsNone(latch.warn("directory gone"))

    def test_a_success_says_nothing(self) -> None:
        latch = mb_play.WarnLatch()
        self.assertIsNone(latch.warn(None))
        self.assertFalse(latch.fired)
        self.assertEqual(latch.warn("directory gone"), "directory gone")

    def test_a_success_after_a_failure_does_not_reopen_the_latch(self) -> None:
        # A directory that has gone away does not come back, so nothing unlatches: two
        # consecutive failures each still produce exactly one message.
        latch = mb_play.WarnLatch()
        latch.warn("disk full")
        self.assertIsNone(latch.warn(None))
        self.assertIsNone(latch.warn("disk full"))

    def test_the_pose_and_status_latches_are_independent(self) -> None:
        # Both writes share the directory and both fail on the same tick, with the pose
        # publish first. One shared latch swallowed the status message entirely, and the
        # log then said nothing about the status stream having stopped.
        pose, status = mb_play.WarnLatch(), mb_play.WarnLatch()
        self.assertIsNotNone(pose.warn("directory gone"))
        self.assertIsNotNone(status.warn("directory gone"))
        self.assertEqual((pose.fired, status.fired), (True, True))


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

    def test_playback_progress_is_reported_against_the_playing_clip(self) -> None:
        # `_mb_seq` used to be assigned before the load, so a clip that was still running
        # wrote `playing`/`done` naming a seq that had already been rejected. The accepted
        # request and the playing clip are two pieces of state; progress names the second.
        self.assertIn("self._playback.clip_seq", self.text)
        self.assertNotIn("self._mb_seq", self.text)

    def test_progress_writes_go_through_the_guarded_writer(self) -> None:
        # A status write that raises is how the tick died when the session directory went
        # away mid-session, so every site goes through the one writer that swallows it.
        self.assertNotIn("mb_mailbox.write_status(", self.text)
        self.assertEqual(self.text.count("self._write_status("), 3)

    def test_the_two_mailbox_writes_have_their_own_warn_latch(self) -> None:
        # The pose publish runs before the status write every tick and both write into the
        # same directory, so one latch made the status message unreachable.
        self.assertNotIn("_mailbox_warned", self.text)
        self.assertEqual(self.text.count("mb_play.WarnLatch()"), 2)

    def test_the_mailbox_is_cleared_at_construction_before_any_read(self) -> None:
        # A deposit left in the persistent `--out` directory by the previous run played its
        # clip on the first tick of a new body, with no adapter running. Construction is
        # the only clearing site and it reads nothing, so nothing can be taken first.
        init = self.text.split("def __init__(", 1)[1].split("\n    def ", 1)[0]
        self.assertIn("mb_mailbox.clear_stale(", init)
        self.assertNotIn("read_play", init)
        self.assertNotIn("read_pose36", init)

    def test_advancing_the_clip_does_not_unwind_the_tick(self) -> None:
        # `_advance_clip` is reached from inside the 50 Hz tick, which owns the physics:
        # no path out of it may throw, including the documented-unreachable one.
        advance = self.text.split("def _advance_clip(", 1)[1].split("\n    def ", 1)[0]
        self.assertNotIn("raise", advance)

    def test_executor_resets_only_at_construction(self) -> None:
        # One reset, in `__init__`, puts the body on the standing clip at startup. A second
        # call site would be playback calling it, which teleports the body and raises on a
        # generated dict's missing `bl`/`ba`.
        self.assertEqual(self.text.count(".reset("), 1, "playback must not call reset()")


if __name__ == "__main__":
    unittest.main()
