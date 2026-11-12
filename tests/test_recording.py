"""Turning a live run into a watchable file: the pure parts and the recorder."""

from __future__ import annotations

import struct
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

from clank import recording
from clank.config import load_config
from clank.ears import EarsError
from clank.recording import (
    GAME_HEIGHT,
    GAME_WIDTH,
    Recorder,
    composite_argv,
    parse_range,
    read_manifest,
)


class CompositeArgvTests(unittest.TestCase):
    def argv(self, **kw) -> list[str]:
        base: dict = {"game": Path("/r/game.mp4"), "robot": Path("/r/robot.mp4"),
                      "audio": Path("/r/audio.m4a"), "out": Path("/r/recording.mp4"),
                      "fps": 15, "crf": 28, "preset": "veryfast"}
        base.update(kw)
        return composite_argv(**base)

    def test_both_pictures_are_stacked_at_a_fixed_size(self):
        argv = self.argv()
        graph = argv[argv.index("-filter_complex") + 1]
        self.assertIn("hstack", graph)
        self.assertIn("scale=960:540", graph)
        self.assertIn("scale=720:540", graph)
        self.assertEqual(argv[argv.index("-movflags") + 1], "+faststart")

    def test_audio_is_muxed_and_the_shorter_stream_wins(self):
        argv = self.argv()
        self.assertIn("-c:a", argv)
        self.assertIn("aac", argv)
        self.assertIn("-shortest", argv)

    def test_a_missing_source_is_black_padding(self):
        argv = self.argv(robot=None, audio=None)
        graph = argv[argv.index("-filter_complex") + 1]
        self.assertIn("color=black", graph)
        self.assertNotIn("-c:a", argv)

    def test_the_encoder_is_named_in_the_argv(self):
        self.assertEqual(self.argv(ffmpeg="/opt/ff")[0], "/opt/ff")

    def test_a_missing_robot_does_not_shift_the_audio_map(self):
        argv = self.argv(robot=None)
        maps = [argv[i + 1] for i, arg in enumerate(argv) if arg == "-map"]
        self.assertEqual(maps, ["[v]", "1:a"])
        self.assertNotIn("/r/robot.mp4", argv)


class GeometryTests(unittest.TestCase):
    def test_the_frame_is_the_two_pictures_side_by_side(self):
        self.assertEqual(recording.GAME_WIDTH, 960)
        self.assertEqual(recording.GAME_HEIGHT, 540)
        self.assertEqual(recording.ROBOT_WIDTH, 720)
        self.assertEqual(recording.HEIGHT, 540)
        self.assertEqual(recording.WIDTH, 1680)


class ParseRangeTests(unittest.TestCase):
    def test_the_three_shapes_a_video_player_sends(self):
        self.assertEqual(parse_range("bytes=0-1", 1000), (0, 1))
        self.assertEqual(parse_range("bytes=100-", 1000), (100, 999))
        self.assertEqual(parse_range("bytes=-500", 1000), (500, 999))

    def test_a_plain_request_is_not_a_range(self):
        self.assertIsNone(parse_range("", 1000))

    def test_multi_range_is_served_whole_rather_than_guessed(self):
        self.assertIsNone(parse_range("bytes=0-1,5-6", 1000))

    def test_unsatisfiable_and_malformed_are_refused(self):
        with self.assertRaises(ValueError):
            parse_range("bytes=5000-", 1000)
        with self.assertRaises(ValueError):
            parse_range("bytes=abc", 1000)


# --------------------------------------------------------------------------- #
# The recorder, against a fake encoder
# --------------------------------------------------------------------------- #

FAKE_FFMPEG = """#!/bin/sh
# The last argument is the output file; copy stdin there so the test can see the
# bytes. `-version` is the one call with no output, and answering it keeps the
# probe from writing a file named "-version".
for a in "$@"; do out="$a"; done
case "$out" in
  *-version) echo "fake-ffmpeg version 1.0"; exit 0 ;;
esac
cat > "$out"
echo "$@" > "$out.argv"
"""

#: `ears.capture()` pieces below need a peak `level()` reads back unchanged.
KNOWN_PEAK = 9000


def pcm(peak: int = KNOWN_PEAK, samples: int = 8) -> bytes:
    values = [peak, -peak] + [0] * max(0, samples - 2)
    return struct.pack(f"<{len(values)}h", *values)


#: An encoder that answers the probe and then exits without reading its pipe.
DEAD_FFMPEG = """#!/bin/sh
for a in "$@"; do out="$a"; done
case "$out" in
  *-version) echo "dead-ffmpeg 0"; exit 0 ;;
esac
exit 0
"""

#: The game's encoder never reads its pipe, so the frame writer fills it and
#: blocks there. The audio and robot encoders behave.
WEDGED_FFMPEG = """#!/bin/sh
for a in "$@"; do out="$a"; done
case "$out" in
  *-version) echo "wedged-ffmpeg 0"; exit 0 ;;
esac
case " $* " in
  *" rawvideo "*) exec sleep 300 ;;
esac
cat > "$out"
"""

class FakeProc:
    def wait(self, timeout=None) -> int:
        return 0

    def poll(self) -> int:
        return 0


class FakeReader:
    """The capture's reader: known PCM, then nothing."""

    def __init__(self, blocks: list[bytes]) -> None:
        self.blocks = list(blocks)

    def read(self, size: int) -> bytes:
        return self.blocks.pop(0) if self.blocks else b""


class FakeCapture:
    """Stands in for `clank.ears.capture`, held open for the run."""

    def __init__(self, blocks: int = 8, peak: int = KNOWN_PEAK, error: Exception | None = None):
        self.blocks = blocks
        self.peak = peak
        self.error = error
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        if self.error is not None:
            raise self.error
        self.entered += 1
        return FakeReader([pcm(self.peak) for _ in range(self.blocks)]), FakeProc(), {"device": "fake"}

    def __exit__(self, *exc):
        self.exited += 1
        return None


class FakeStream:
    """A depth stream publishing `frames` bottom-up RGBA frames, then nothing."""

    def __init__(self, frames: int = 1) -> None:
        self.frames = frames
        self.seq = 0
        self.flips: list[bool] = []
        self.closed = False
        #: Set once a read is in progress: the frame tick has begun.
        self.reading = threading.Event()

    def read(self, timeout: float = 5.0, after_seq: int = 0, flip: bool = True) -> dict | None:
        self.reading.set()
        self.flips.append(flip)
        if self.seq >= self.frames * 2:
            return None
        self.seq += 2
        return {
            "color": bytes(GAME_WIDTH * GAME_HEIGHT * 4),
            "seq": self.seq,
            "color_width": GAME_WIDTH,
            "color_height": GAME_HEIGHT,
            "width": 480,
            "height": 270,
        }

    def close(self) -> None:
        self.closed = True


class BrokenStream(FakeStream):
    """A depth stream whose reads raise, the way a closed mmap's do.

    `failures` reads fail and every later one behaves, so one fake covers both
    the blink the recorder notes and the death it has to report.
    """

    def __init__(self, failures: int = 1) -> None:
        super().__init__(frames=1)
        self.failures = failures

    def read(self, timeout: float = 5.0, after_seq: int = 0, flip: bool = True) -> dict | None:
        if self.failures <= 0:
            return super().read(timeout, after_seq, flip)
        self.failures -= 1
        self.reading.set()
        raise OSError("mmap closed")


class SteppingClock:
    """A clock the test owns: every read advances it by one step.

    `Recorder(clock=...)` exists so a test can decide the schedule instead of
    waiting for it. A pump then stops when *this* clock passes its deadline, so
    the number of due frames is the same on every machine.
    """

    def __init__(self, step: float = 0.02) -> None:
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


class ExclusiveStream(FakeStream):
    """A depth stream that counts any two frame ticks let in at the same time.

    A read is held open for a millisecond, which is the window in which a second
    pump, ticking on its own, would reach this stream too. The count is of
    violations that really happened, so it cannot cry wolf: it can only miss.
    """

    def __init__(self) -> None:
        super().__init__(frames=1)
        self.inside = threading.Lock()
        self.violations = 0

    def read(self, timeout: float = 5.0, after_seq: int = 0, flip: bool = True) -> dict | None:
        if not self.inside.acquire(blocking=False):
            self.violations += 1
            return super().read(timeout, after_seq, flip)
        try:
            time.sleep(0.001)
            return super().read(timeout, after_seq, flip)
        finally:
            self.inside.release()


class FakeStreamFactory:
    """`stream_factory` as the recorder calls it, keeping the streams it made."""

    def __init__(self, frames: int = 1) -> None:
        self.frames = frames
        self.made: list[FakeStream] = []

    def __call__(self) -> FakeStream:
        stream = FakeStream(self.frames)
        self.made.append(stream)
        return stream


class RecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        cfg = load_config(None, env={})
        # The body view writes its newest tile beside the session directories,
        # so the recorder finds the robot's picture through the config.
        cfg = replace(cfg, paths=replace(cfg.paths, session_root=str(self.base / "sessions")))
        self.run_dir = Path(cfg.paths.session_root) / "101010-a1b2c3" / "runs" / "101010-deadbe"
        self.run_dir.mkdir(parents=True)
        self.ffmpeg = self.base / "fake-ffmpeg"
        self.ffmpeg.write_text(FAKE_FFMPEG)
        self.ffmpeg.chmod(0o755)
        self.cfg = replace(cfg, recording=replace(cfg.recording, ffmpeg=str(self.ffmpeg)))
        self.body_dir = Path(cfg.paths.session_root) / "body"
        self.body_dir.mkdir(parents=True)
        self.tile = self.body_dir / "body_latest.jpg"
        self.tile.write_bytes(b"\xff\xd8body-tile\xff\xd9")
        self.streams = FakeStreamFactory()

    def argv_of(self, part: str) -> str:
        return (self.run_dir / "parts" / f"{part}.argv").read_text()

    def test_start_records_and_stop_composes_a_manifest(self):
        rec = Recorder(self.cfg, self.run_dir, stream_factory=self.streams, capture=FakeCapture)
        rec.start()
        recording_manifest = read_manifest(self.run_dir)
        if recording_manifest is None:
            self.fail("start() must leave a manifest behind")
        self.assertEqual(recording_manifest["state"], "recording")
        rec.pump(0.4)
        manifest = rec.stop()
        self.assertEqual(manifest["state"], "composing")
        self.assertEqual(manifest["width"], 1680)
        self.assertEqual(manifest["height"], 540)
        self.assertEqual(manifest["fps"], 15)
        self.assertEqual(manifest["sources"], {"game": "ok", "robot": "ok", "audio": "ok"})
        self.assertEqual(manifest["audio_peak"], KNOWN_PEAK)
        self.assertEqual(manifest["parts"], {
            "game": "parts/game.mp4",
            "robot": "parts/robot.mp4",
            "audio": "parts/audio.m4a",
        })
        self.assertEqual(manifest["bytes"],
                         sum((self.run_dir / name).stat().st_size
                             for name in manifest["parts"].values()))
        self.assertIsNone(manifest["error"])
        self.assertTrue(manifest["available"])
        self.assertIn("fake-ffmpeg", manifest["encoder"])
        self.assertTrue(manifest["started_at"].endswith("Z"))
        self.assertTrue(manifest["ended_at"].endswith("Z"))

    def test_a_missing_encoder_is_a_manifest_not_a_crash(self):
        cfg = replace(self.cfg, recording=replace(self.cfg.recording,
                                                   ffmpeg=str(self.base / "nope")))
        rec = Recorder(cfg, self.run_dir)
        rec.start()  # must not raise
        manifest = rec.stop()
        self.assertEqual(manifest["state"], "failed")
        self.assertIn("ffmpeg", manifest["error"])
        self.assertEqual(manifest["parts"], {})
        self.assertFalse(manifest["available"])

    def test_no_depth_stream_records_the_robot_side_only(self):
        rec = Recorder(self.cfg, self.run_dir, stream_factory=None, capture=FakeCapture)
        rec.start()
        manifest = rec.stop()
        self.assertEqual(manifest["sources"]["game"], "missing")
        self.assertEqual(manifest["state"], "composing")
        self.assertEqual(manifest["parts"],
                         {"robot": "parts/robot.mp4", "audio": "parts/audio.m4a"})
        self.assertIsNone(manifest["error"])

    def test_a_frame_that_did_not_arrive_is_written_again_when_it_is_due(self):
        """The part's length is the schedule's, exactly: one frame per slot.

        The clock is this test's and it drives the only pump. The recorder's own
        thread would otherwise supply the same frames off the same clock, so the
        counts would be true whether or not this pump wrote anything at all.
        """
        clock = SteppingClock()
        rec = Recorder(self.cfg, self.run_dir, stream_factory=self.streams,
                       capture=FakeCapture, clock=clock)
        rec._start_thread = lambda target, name: None   # this test drives the only pump
        rec.start()
        rec.pump(0.38)
        rec.stop()
        frame = GAME_WIDTH * GAME_HEIGHT * 4
        game = (self.run_dir / "parts" / "game.mp4").stat().st_size
        # The start slot is 0.02 s in and the pump reaches 0.40 s: six slots at
        # 15 fps, and not one frame more for the hand-driven pump beside the
        # schedule.
        self.assertEqual(game, 6 * frame)
        tile = len(self.tile.read_bytes())
        robot = (self.run_dir / "parts" / "robot.mp4").stat().st_size
        # Four slots at 10 fps over the same 0.40 s, and the tile stop() writes
        # so a run shorter than a tick still has the robot's picture.
        self.assertEqual(robot, 5 * tile)

    def test_a_depth_stream_that_keeps_raising_is_not_recorded_as_a_success(self):
        stream = BrokenStream(failures=10**6)
        rec = Recorder(self.cfg, self.run_dir, stream_factory=lambda: stream,
                       capture=FakeCapture)
        rec.start()
        # How many ticks pump() gets out of a duration is the wall clock's, so
        # the bound is driven one tick at a time.
        now = rec._started_monotonic
        for tick in range(recording.STREAM_ERROR_LIMIT + 1):
            rec._game_tick(now + tick * rec._period)
        manifest = rec.stop()
        self.assertEqual(manifest["sources"]["game"], "failed")
        self.assertIn("stopped answering", manifest["notes"]["game"])
        self.assertNotIn("game", manifest["parts"])
        self.assertEqual(manifest["state"], "composing")

    def test_a_read_error_is_in_the_manifest_even_when_the_stream_recovers(self):
        stream = BrokenStream(failures=1)
        rec = Recorder(self.cfg, self.run_dir, stream_factory=lambda: stream,
                       capture=FakeCapture)
        rec.start()
        rec.pump(0.2)
        manifest = rec.stop()
        # One failed read is a note, not a failure: the stream answered again
        # and the picture kept coming. The note survives so the reason for the
        # gap is not lost.
        self.assertEqual(manifest["sources"]["game"], "ok")
        self.assertIn("stopped answering", manifest["notes"]["game"])

    def test_two_pumps_never_write_into_the_tick_at_the_same_time(self):
        stream = ExclusiveStream()
        rec = Recorder(self.cfg, self.run_dir, stream_factory=lambda: stream,
                       capture=FakeCapture, clock=SteppingClock())
        rec.start()
        # The recorder's own thread is pumping this same schedule, so this pump
        # beside it is exactly the second caller the tick lock has to serialise.
        rec.pump(0.38)
        rec.stop()
        self.assertEqual(stream.violations, 0)

    def test_a_wedged_game_writer_leaves_the_audio_track_on_disk(self):
        wedged = self.base / "wedged-ffmpeg"
        wedged.write_text(WEDGED_FFMPEG)
        wedged.chmod(0o755)
        cfg = replace(self.cfg, recording=replace(self.cfg.recording, ffmpeg=str(wedged)))
        rec = Recorder(cfg, self.run_dir, stream_factory=self.streams, capture=FakeCapture)
        rec.start()
        self.assertTrue(self.streams.made[0].reading.wait(5.0))
        # The frame writer is now blocked in the game's full pipe, and stays
        # blocked until stop() kills that encoder. The audio encoder never
        # stopped reading, so its part is the whole run.
        manifest = rec.stop()
        self.assertEqual(manifest["sources"]["audio"], "ok")
        self.assertIn("audio", manifest["parts"])
        self.assertGreater((self.run_dir / "parts" / "audio.m4a").stat().st_size, 0)
        self.assertNotEqual(manifest["sources"]["game"], "ok")
        self.assertNotIn("game", manifest["parts"])

    def test_the_reader_is_not_flipped_and_the_encoders_are_fed_their_pipes(self):
        rec = Recorder(self.cfg, self.run_dir, stream_factory=self.streams, capture=FakeCapture)
        rec.start()
        rec.pump(0.1)
        rec.stop()
        self.assertTrue(self.streams.made)
        self.assertIn(False, self.streams.made[0].flips)  # 210 fps, not 58
        self.assertTrue(self.streams.made[0].closed)
        game = self.argv_of("game.mp4")
        self.assertIn("rawvideo", game)
        self.assertIn("-s 960x540", game)
        self.assertIn("-vf vflip", game)
        robot = self.argv_of("robot.mp4")
        self.assertIn("image2pipe", robot)
        self.assertIn("-framerate 10", robot)
        self.assertIn("scale=720:540", robot)
        audio = self.argv_of("audio.m4a")
        self.assertIn("s16le", audio)
        self.assertIn("-ar 16000", audio)
        self.assertIn("aac", audio)

    def test_a_body_view_that_never_wrote_a_tile_is_not_in_the_manifest(self):
        self.tile.unlink()
        rec = Recorder(self.cfg, self.run_dir, stream_factory=self.streams, capture=FakeCapture)
        rec.start()
        manifest = rec.stop()
        self.assertEqual(manifest["sources"]["robot"], "missing")
        self.assertNotIn("robot", manifest["parts"])
        self.assertFalse((self.run_dir / "parts" / "robot.mp4").exists())
        self.assertEqual(manifest["state"], "composing")

    def test_a_dead_pipe_is_a_failed_source_not_an_exception(self):
        dead = self.base / "dead-ffmpeg"
        dead.write_text(DEAD_FFMPEG)
        dead.chmod(0o755)
        cfg = replace(self.cfg, recording=replace(self.cfg.recording, ffmpeg=str(dead)))
        rec = Recorder(cfg, self.run_dir, stream_factory=self.streams, capture=None)
        rec.start()
        rec.pump(0.2)  # nothing is reading the game pipe; the write must not raise
        manifest = rec.stop()
        self.assertEqual(manifest["sources"]["game"], "failed")
        self.assertNotIn("game", manifest["parts"])
        self.assertIsNone(manifest["error"])

    def test_a_capture_that_cannot_start_is_a_missing_audio_track(self):
        rec = Recorder(self.cfg, self.run_dir, stream_factory=self.streams,
                       capture=lambda: FakeCapture(error=EarsError("no playback stream")))
        rec.start()
        manifest = rec.stop()
        self.assertEqual(manifest["sources"]["audio"], "missing")
        self.assertNotIn("audio", manifest["parts"])
        self.assertEqual(manifest["state"], "composing")
