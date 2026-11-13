"""Turning a live run into a watchable file: the pure parts and the recorder."""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from clank import recording
from clank.config import load_config
from clank.ears import EarsError
from clank.recording import (
    GAME_HEIGHT,
    GAME_WIDTH,
    Recorder,
    composite_argv,
    list_recordings,
    parse_range,
    probe_part,
    prune,
    read_manifest,
    pending_composites,
    run_composite,
    sweep_stale,
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
        self.assertIn(f"scale={recording.GAME_WIDTH}:{recording.HEIGHT}", graph)
        self.assertIn(f"scale={recording.ROBOT_WIDTH}:{recording.HEIGHT}", graph)
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


class WrongSizeStream(FakeStream):
    """A live layer launched with `--no-color` or a different `--color-scale`.

    The reads answer perfectly well; the frames are simply not the size the
    composite is built for, which is an operator's launch flag and not a dead
    layer.
    """

    def read(self, timeout: float = 5.0, after_seq: int = 0, flip: bool = True) -> dict | None:
        self.reading.set()
        self.flips.append(flip)
        self.seq += 2
        return {
            "color": bytes(480 * 270 * 4),
            "seq": self.seq,
            "color_width": 480,
            "color_height": 270,
            "width": 480,
            "height": 270,
        }


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


class DyingStream(FakeStream):
    """A depth stream that answers `reads` times and then raises for good.

    The shape the ruling is about: a depth layer that recorded for a while and
    then died, so the source is no longer ``ok`` but its part holds real frames.
    """

    def __init__(self, reads: int = 2) -> None:
        super().__init__(frames=reads)
        self.reads = reads

    def read(self, timeout: float = 5.0, after_seq: int = 0, flip: bool = True) -> dict | None:
        if self.reads <= 0:
            self.reading.set()
            raise OSError("mmap closed")
        self.reads -= 1
        return super().read(timeout, after_seq, flip)


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

    def __init__(self, frames: int = 1, factory: type[FakeStream] = FakeStream) -> None:
        self.frames = frames
        self.factory = factory
        self.made: list[FakeStream] = []

    def __call__(self) -> FakeStream:
        stream = self.factory(self.frames)
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
        self.assertEqual(manifest["width"], recording.WIDTH)
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

    def test_a_stream_of_the_wrong_size_names_the_mismatch(self):
        """`--no-color` and `--color-scale` are launch flags, not a dead layer:
        "nothing arrived" would send the operator looking in the wrong place."""
        streams = FakeStreamFactory(factory=WrongSizeStream)
        rec = Recorder(self.cfg, self.run_dir, stream_factory=streams, capture=FakeCapture)
        rec.start()
        rec.pump(2.2)  # past STREAM_ERROR_LIMIT ticks of the wrong size
        manifest = rec.stop()
        # No frame was written, so the source is `missing`; the note is what says
        # why, and it names the sizes rather than claiming nothing arrived.
        self.assertEqual(manifest["sources"]["game"], "missing")
        note = manifest["notes"]["game"]
        self.assertIn("480x270", note)
        self.assertIn(f"{GAME_WIDTH}x{GAME_HEIGHT}", note)

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

    def test_a_demoted_source_keeps_the_part_it_already_wrote(self):
        """Bytes beat status: the picture recorded before the stream died is kept.

        Demotion is about reporting, never deletion. The game source ends
        ``failed`` with its note, and the part still holds every frame written
        before the stream died, with `part_seconds` saying how much that is so
        Task 4 can pad the span instead of letting `-shortest` truncate the
        composite to it.
        """
        stream = DyingStream(reads=2)
        rec = Recorder(self.cfg, self.run_dir, stream_factory=lambda: stream,
                       capture=FakeCapture)
        rec._start_thread = lambda target, name: None   # this test drives the ticks
        rec.start()
        now = rec._started_monotonic
        for tick in range(recording.STREAM_ERROR_LIMIT + 2):
            rec._game_tick(now + tick * rec._period)
        manifest = rec.stop()
        self.assertEqual(manifest["sources"]["game"], "failed")
        self.assertIn("stopped answering", manifest["notes"]["game"])
        part = self.run_dir / "parts" / "game.mp4"
        self.assertTrue(part.exists())
        frames = part.stat().st_size // (GAME_WIDTH * GAME_HEIGHT * 4)
        self.assertGreater(frames, 0)
        self.assertIn("game", manifest["parts"])
        self.assertEqual(manifest["parts"]["game"], "parts/game.mp4")
        self.assertAlmostEqual(manifest["part_seconds"]["game"], frames / 15, places=2)
        self.assertEqual(manifest["state"], "composing")

    def test_a_demoted_source_that_wrote_nothing_has_no_part(self):
        """A part with no bytes is not evidence, and Task 4 cannot composite it."""
        stream = BrokenStream(failures=10**6)
        rec = Recorder(self.cfg, self.run_dir, stream_factory=lambda: stream,
                       capture=FakeCapture)
        rec._start_thread = lambda target, name: None   # this test drives the ticks
        rec.start()
        now = rec._started_monotonic
        for tick in range(recording.STREAM_ERROR_LIMIT + 2):
            rec._game_tick(now + tick * rec._period)
        manifest = rec.stop()
        self.assertEqual(manifest["sources"]["game"], "failed")
        self.assertNotIn("game", manifest["parts"])
        self.assertFalse((self.run_dir / "parts" / "game.mp4").exists())
        self.assertIn("part_seconds", manifest)
        self.assertNotIn("game", manifest["part_seconds"])

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
        self.assertIn(f"scale={recording.ROBOT_WIDTH}:{recording.HEIGHT}", robot)
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


# --------------------------------------------------------------------------- #
# The offline half: the composite, the sweep, the prune and the listing
# --------------------------------------------------------------------------- #

#: A fake encoder and a fake probe, both needed. The composite probes every part
#: before it composes them, so a test that faked only ffmpeg would hand real
#: ffprobe a file that is not an MP4 and watch every part be dropped.
FAKE_COMPOSITOR = """#!/bin/sh
# The last argument is the output: write a marker there and record the argv
# beside it, one argument per line so a path with a space stays one argument.
for a in "$@"; do out="$a"; done
case "$out" in
  *-version) echo "fake-ffmpeg version 1.0"; exit 0 ;;
esac
printf 'fake-composite' > "$out"
printf '%s\\n' "$@" > "$out.argv"
"""

#: A composite that reports success and writes nothing: the manifest must not
#: call that playable.
EMPTY_COMPOSITOR = """#!/bin/sh
for a in "$@"; do out="$a"; done
case "$out" in
  *-version) echo "empty-ffmpeg 0"; exit 0 ;;
esac
: > "$out"
"""

#: The fake probe answers `ffprobe -print_format json -show_entries ...`. A part
#: whose name says "broken" has no moov atom to be found; the game part is a
#: minute long and the rest are four, so the padding arithmetic is visible in the
#: composite's argv. A part named "short-robot" is a minute too, which is the
#: demotion the right-hand tile has to survive, and the composite's own output
#: measures 238.5 s -- deliberately not the 240 s the run asked for, so a test can
#: tell a measured length from the target.
FAKE_PROBE = """#!/bin/sh
for a in "$@"; do part="$a"; done
case "$part" in
  *broken*)
    echo "moov atom not found" >&2
    exit 1
    ;;
  *short-robot*)
    printf '{"streams":[{"codec_type":"video"}],"format":{"duration":"60.0"}}'
    ;;
  *recording.*mp4)
    printf '{"streams":[{"codec_type":"video"}],"format":{"duration":"238.5"}}'
    ;;
  *game.mp4)
    printf '{"streams":[{"codec_type":"video"}],"format":{"duration":"60.0"}}'
    ;;
  *.m4a)
    printf '{"streams":[{"codec_type":"audio"}],"format":{"duration":"240.0"}}'
    ;;
  *)
    printf '{"streams":[{"codec_type":"video"}],"format":{"duration":"240.0"}}'
    ;;
esac
"""

#: The parts a manifest names by default, keyed the way `Recorder` keys them.
PART_PATHS = {name: f"{recording.PARTS_DIR}/{filename}"
              for name, filename in recording.PART_NAMES.items()}


def write_recording(
    run_dir: Path,
    *,
    state: str = "composing",
    started_at: str = "2026-11-12T15:00:00.000Z",
    duration_s: float = 240.0,
    parts: dict[str, str] | None = None,
    write_parts: bool = True,
    **fields,
) -> dict:
    """A `recording.json` in the shape `Recorder.stop()` leaves, with its parts.

    `write_parts=False` leaves the files alone: a test that built real ones with
    ffmpeg wants the manifest only.
    """
    parts = dict(PART_PATHS if parts is None else parts)
    paths = parts.values() if write_parts else ()
    for relative in paths:
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"part-bytes")
    doc = {
        "available": bool(parts),
        "file": None,
        "started_at": started_at,
        "ended_at": "2026-11-12T15:04:00.000Z",
        "duration_s": duration_s,
        "state": state,
        "width": recording.WIDTH,
        "height": recording.HEIGHT,
        "fps": 15,
        "bytes": 0,
        "sources": {"game": "ok", "robot": "ok", "audio": "ok"},
        "audio_peak": 9000,
        "encoder": "fake-ffmpeg version 1.0",
        "parts": parts,
        "part_seconds": dict.fromkeys(parts, duration_s),
        "notes": {},
        "error": None,
    }
    doc.update(fields)
    (Path(run_dir) / recording.MANIFEST_NAME).write_text(json.dumps(doc), encoding="utf-8")
    return doc


def manifest_of(test: unittest.TestCase, run_dir: Path) -> dict:
    """`read_manifest` with the test failed rather than a TypeError when it is gone."""
    doc = read_manifest(run_dir)
    if doc is None:
        test.fail(f"no {recording.MANIFEST_NAME} in {run_dir}")
    return doc


class GeometryTests(unittest.TestCase):
    def test_the_composite_is_a_portrait_tile_beside_the_game(self):
        self.assertEqual(recording.WIDTH, 1264)
        self.assertEqual(recording.HEIGHT, 540)
        self.assertEqual(recording.ROBOT_WIDTH, 304)
        self.assertEqual(recording.GAME_WIDTH, 960)

    def test_the_robot_part_scales_into_the_portrait_tile(self):
        argv = composite_argv(game=Path("g.mp4"), robot=Path("r.mp4"), audio=None,
                              out=Path("o.mp4"), fps=15, crf=28, preset="veryfast")
        graph = argv[argv.index("-filter_complex") + 1]
        self.assertIn("scale=304:540", graph)
        self.assertIn("scale=960:540", graph)


class CompositeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.sessions = self.base / "sessions"
        self.run_dir = self.sessions / "101010-a1b2c3" / "runs" / "101011-deadbe"
        self.run_dir.mkdir(parents=True)
        cfg = load_config(None, env={})
        cfg = replace(cfg, paths=replace(cfg.paths, session_root=str(self.sessions)))
        self.compositor = self.base / "fake-ffmpeg"
        self.compositor.write_text(FAKE_COMPOSITOR)
        self.compositor.chmod(0o755)
        # The probe is found beside the configured encoder, so the fake pair sits
        # in one directory and `cfg.recording.ffmpeg` names both.
        probe = self.base / "ffprobe"
        probe.write_text(FAKE_PROBE)
        probe.chmod(0o755)
        self.cfg = replace(cfg, recording=replace(cfg.recording, ffmpeg=str(self.compositor)))
        write_recording(self.run_dir)

    def composite_argv_of(self, name: str = "recording.mp4") -> list[str]:
        # The composite writes beside its target and renames, so the argv the fake
        # recorded is named after whichever output path it was handed.
        path = self.run_dir / f"{name}.argv"
        if not path.is_file():
            # The composite writes beside its target and renames, so the argv the
            # fake recorded is named after the temporary output it was handed.
            candidates = sorted(self.run_dir.glob("*.argv"))
            self.assertTrue(candidates, f"no argv was written for {name}")
            path = candidates[-1]
        return path.read_text().splitlines()

    def graph_of(self, name: str = "recording.mp4") -> str:
        argv = self.composite_argv_of(name)
        return argv[argv.index("-filter_complex") + 1]

    def test_the_composite_turns_composing_into_ready(self):
        manifest = run_composite(self.run_dir, self.cfg)
        self.assertEqual(manifest["state"], "ready")
        self.assertEqual(manifest["file"], "recording.mp4")
        self.assertTrue((self.run_dir / "recording.mp4").is_file())
        self.assertFalse((self.run_dir / "parts").exists())  # keep_parts false
        self.assertTrue(manifest["available"])
        self.assertIsNone(manifest["error"])
        self.assertEqual(manifest["width"], recording.WIDTH)
        self.assertEqual(manifest["height"], recording.HEIGHT)
        self.assertEqual(manifest["fps"], 15)
        self.assertEqual(manifest["bytes"], (self.run_dir / "recording.mp4").stat().st_size)
        # The run asked for 240 s; the probe measured the file at 238.5, and the
        # measurement is what the page shows.
        self.assertEqual(manifest["duration_s"], 238.5)
        # The manifest a dashboard reads back says so too.
        self.assertEqual(manifest_of(self, self.run_dir)["state"], "ready")

    def test_the_composite_names_the_parts_by_absolute_path(self):
        """The dashboard's cwd is not the run directory: a bare part name is a
        file the composite cannot find."""
        run_composite(self.run_dir, self.cfg)
        argv = self.composite_argv_of()
        for name in ("game.mp4", "robot.mp4", "audio.m4a"):
            self.assertIn(str((self.run_dir / "parts" / name).resolve()), argv)
            self.assertNotIn(f"parts/{name}", argv)

    def test_a_short_part_is_padded_so_the_video_covers_the_run(self):
        """`composite_argv` ends at `-shortest`, so a part demoted at minute one of
        twenty would otherwise composite to one minute."""
        run_composite(self.run_dir, self.cfg)
        graph = self.graph_of()
        argv = self.composite_argv_of()
        # The game part probes at 60 s against a 240 s run: hold its last frame
        # for the other 180 s, and cut the result at the run's own length.
        self.assertIn("tpad=stop_mode=clone:stop_duration=181.000", graph)
        self.assertEqual(argv[argv.index("-t") + 1], "240.000")
        self.assertNotIn("-shortest", argv)

    def test_a_part_that_cannot_be_demuxed_is_left_out_with_a_note(self):
        broken = {"game": "parts/broken-game.mp4",
                  "robot": PART_PATHS["robot"], "audio": PART_PATHS["audio"]}
        write_recording(self.run_dir, parts=broken)
        manifest = run_composite(self.run_dir, self.cfg)
        self.assertEqual(manifest["state"], "ready")
        self.assertNotIn("game", manifest["parts"])
        self.assertIn("cannot be demuxed", manifest["notes"]["game"])
        argv = self.composite_argv_of()
        self.assertNotIn(str(self.run_dir / "parts" / "broken-game.mp4"), argv)
        # The picture the probe did trust is still composited, black on the left.
        self.assertIn("color=black:s=960x540", self.graph_of())

    def test_nothing_playable_is_a_failed_run_not_a_ready_one(self):
        broken = {name: f"parts/broken-{filename}"
                  for name, filename in recording.PART_NAMES.items()}
        write_recording(self.run_dir, parts=broken)
        manifest = run_composite(self.run_dir, self.cfg)
        self.assertEqual(manifest["state"], "failed")
        self.assertFalse(manifest["available"])
        self.assertIsNone(manifest["file"])
        self.assertIn("no picture", manifest["error"])
        self.assertFalse((self.run_dir / "recording.mp4").exists())

    def test_a_failed_composite_keeps_the_parts_and_says_why(self):
        cfg = replace(self.cfg, recording=replace(self.cfg.recording,
                                                   ffmpeg=str(self.base / "nope")))
        manifest = run_composite(self.run_dir, cfg)
        self.assertEqual(manifest["state"], "failed")
        self.assertTrue((self.run_dir / "parts" / "game.mp4").is_file())
        self.assertIn("nope", manifest["error"])
        self.assertFalse(manifest["available"])

    def test_a_recorder_that_failed_is_not_recomposed(self):
        """The recorder's reason is precise; "nothing survived the probe" is not."""
        write_recording(self.run_dir, state="failed", parts={},
                        error="ffmpeg is not runnable: [Errno 2] no such file")
        manifest = run_composite(self.run_dir, self.cfg)
        self.assertEqual(manifest["state"], "failed")
        self.assertEqual(manifest["error"], "ffmpeg is not runnable: [Errno 2] no such file")
        self.assertFalse((self.run_dir / "recording.mp4").exists())

    def test_a_short_robot_part_is_padded_on_the_robot_side(self):
        """The game's picture is not the only one that can be demoted mid-run:
        the robot's hold has to land on the `[r]` chain, not only on `[g]`."""
        write_recording(self.run_dir, parts={
            "game": "parts/capture.mp4",        # 240 s: the fake probe's default
            "robot": "parts/short-robot.mp4",   # 60 s: the part a demotion leaves
            "audio": "parts/audio.m4a",
        })
        run_composite(self.run_dir, self.cfg)
        chains = self.graph_of().split(";")
        right = next(chain for chain in chains if chain.endswith("[r]"))
        left = next(chain for chain in chains if chain.endswith("[g]"))
        self.assertIn("tpad=stop_mode=clone:stop_duration=181.000", right)
        self.assertNotIn("181.000", left)  # the game's side is full length

    def test_a_composite_that_writes_nothing_is_not_published(self):
        empty = self.base / "empty-ffmpeg"
        empty.write_text(EMPTY_COMPOSITOR)
        empty.chmod(0o755)
        cfg = replace(self.cfg, recording=replace(self.cfg.recording, ffmpeg=str(empty)))
        manifest = run_composite(self.run_dir, cfg)
        self.assertEqual(manifest["state"], "failed")
        self.assertIn("wrote nothing", manifest["error"])
        self.assertFalse(manifest["available"])
        self.assertFalse((self.run_dir / "recording.mp4").exists())
        # What is left is the parts, which are the evidence.
        self.assertTrue((self.run_dir / "parts" / "game.mp4").is_file())

    def test_a_composite_that_fails_leaves_no_truncated_video(self):
        """A half-written `recording.mp4` would carry `file: null`, so neither the
        sweep nor the prune would ever come back for it."""
        crippled = self.base / "crippled-ffmpeg"
        crippled.write_text(
            "#!/bin/sh\n"
            "for a in \"$@\"; do out=\"$a\"; done\n"
            "printf 'half-an-mp4' > \"$out\"\n"
            "echo 'boom: the encoder died' >&2\n"
            "exit 1\n"
        )
        crippled.chmod(0o755)
        cfg = replace(self.cfg, recording=replace(self.cfg.recording, ffmpeg=str(crippled)))
        manifest = run_composite(self.run_dir, cfg)
        self.assertEqual(manifest["state"], "failed")
        self.assertIn("boom", manifest["error"])
        self.assertFalse((self.run_dir / "recording.mp4").exists())
        self.assertTrue((self.run_dir / "parts" / "game.mp4").is_file())

    def test_a_composite_whose_output_cannot_be_demuxed_is_not_published(self):
        """Exit 0 and non-zero bytes is not "playable": the output is probed too."""
        encoder_dir = self.base / "corrupt"
        encoder_dir.mkdir()
        encoder = encoder_dir / "ffmpeg"
        encoder.write_text(
            "#!/bin/sh\n"
            "for a in \"$@\"; do out=\"$a\"; done\n"
            "printf 'not-an-mp4' > \"$out\"\n"
        )
        encoder.chmod(0o755)
        probe = encoder_dir / "ffprobe"
        probe.write_text(
            "#!/bin/sh\n"
            "for a in \"$@\"; do part=\"$a\"; done\n"
            "case \"$part\" in\n"
            "  *recording.*mp4) echo 'moov atom not found' >&2; exit 1 ;;\n"
            "  *.m4a) printf '{\"streams\":[{\"codec_type\":\"audio\"}],"
            "\"format\":{\"duration\":\"240.0\"}}' ;;\n"
            "  *) printf '{\"streams\":[{\"codec_type\":\"video\"}],"
            "\"format\":{\"duration\":\"240.0\"}}' ;;\n"
            "esac\n"
        )
        probe.chmod(0o755)
        cfg = replace(self.cfg, recording=replace(self.cfg.recording, ffmpeg=str(encoder)))
        manifest = run_composite(self.run_dir, cfg)
        self.assertEqual(manifest["state"], "failed")
        self.assertIn("no playable video", manifest["error"])
        self.assertIn("moov atom not found", manifest["error"])
        self.assertFalse(manifest["available"])
        self.assertIsNone(manifest["file"])
        self.assertFalse((self.run_dir / "recording.mp4").exists())

    def test_a_prune_that_raises_does_not_undo_the_publication(self):
        """Housekeeping is not part of the publication: the recording is `ready`
        whether or not the cap could be enforced."""
        with patch.object(recording, "prune", side_effect=OSError("disk gone")):
            manifest = run_composite(self.run_dir, self.cfg)
        self.assertEqual(manifest["state"], "ready")
        self.assertEqual(manifest_of(self, self.run_dir)["state"], "ready")
        self.assertTrue((self.run_dir / "recording.mp4").is_file())

    def test_the_run_just_composed_is_never_pruned(self):
        """The cap deletes the oldest, and `keep` is the run that just finished:
        deleting the thing that was just asked for is not pruning."""
        other = self.sessions / "101010-a1b2c3" / "runs" / "101500-ffffff"
        other.mkdir(parents=True)
        write_recording(other, state="ready", started_at="2026-11-12T15:05:00.000Z",
                        file="recording.mp4")
        (other / "recording.mp4").write_bytes(b"x" * (2 * 1024 * 1024))
        cfg = replace(self.cfg, recording=replace(self.cfg.recording, max_total_mb=1))
        manifest = run_composite(self.run_dir, cfg)
        self.assertEqual(manifest["state"], "ready")
        # Over the cap, so its video goes: the run that was just asked for stays.
        self.assertFalse((other / "recording.mp4").exists())
        self.assertEqual(manifest_of(self, other)["state"], "pruned")
        self.assertTrue((self.run_dir / "recording.mp4").is_file())


class SessionsRootTests(unittest.TestCase):
    """The sweep, the prune and the listing, over a sessions root of runs."""

    SESSION = "20261112-abc123"

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.sessions = self.base / "sessions"

    def dir_of(self, run_id: str, session: str | None = None) -> Path:
        return self.sessions / (session or self.SESSION) / "runs" / run_id

    def make_run(self, run_id: str, *, session: str | None = None, **fields) -> Path:
        run_dir = self.dir_of(run_id, session)
        run_dir.mkdir(parents=True)
        write_recording(run_dir, **fields)
        return run_dir

    def make_video(self, run_dir: Path, size: int) -> None:
        (run_dir / "recording.mp4").write_bytes(b"x" * size)

    def test_a_run_left_mid_composite_is_kept_for_the_next_dashboard(self):
        """Exiting inside a composite window is ordinary; the video is not."""
        run_dir = self.make_run("101010-aaaaaa", state="recording")
        self.make_video(run_dir, 32)  # a composite killed mid-write, no moov atom
        self.assertEqual(sweep_stale(self.sessions), [])
        manifest = manifest_of(self, run_dir)
        self.assertEqual(manifest["state"], "recording")
        self.assertTrue((run_dir / "parts" / "game.mp4").is_file())
        self.assertTrue((run_dir / "recording.mp4").is_file())

    def test_a_stale_run_with_nothing_to_compose_is_marked_failed(self):
        run_dir = self.make_run("101010-aaaaaa", state="composing", parts={})
        removed = sweep_stale(self.sessions)
        self.assertEqual(removed, [run_dir])
        manifest = manifest_of(self, run_dir)
        self.assertEqual(manifest["state"], "failed")
        self.assertIn("dashboard exited", manifest["error"])
        self.assertFalse(manifest["available"])
        self.assertFalse((run_dir / "parts").exists())
        self.assertFalse((run_dir / "recording.mp4").exists())

    def test_a_composite_still_running_is_left_entirely_alone(self):
        run_dir = self.make_run("101010-aaaaaa", state="composing")
        (run_dir / recording.COMPOSITE_LOCK).write_text(str(os.getpid()), encoding="utf-8")
        self.assertEqual(sweep_stale(self.sessions), [])
        self.assertEqual(pending_composites(self.sessions), [])
        self.assertTrue((run_dir / "parts" / "game.mp4").is_file())
        self.assertEqual(manifest_of(self, run_dir)["state"], "composing")

    def test_pending_composites_hands_a_dead_dashboards_work_on(self):
        run_dir = self.make_run("101010-aaaaaa", state="composing")
        # A lock naming a pid that cannot exist: the dashboard that wrote it is gone.
        (run_dir / recording.COMPOSITE_LOCK).write_text("999999", encoding="utf-8")
        self.assertEqual(pending_composites(self.sessions), [run_dir])
        self.assertEqual(sweep_stale(self.sessions), [])
        # The next dashboard takes a dead owner's lock over, and refuses a live one.
        self.assertTrue(recording._acquire_composite(run_dir))
        self.assertFalse(recording._acquire_composite(run_dir))
        recording._release_composite(run_dir)
        self.assertFalse((run_dir / recording.COMPOSITE_LOCK).exists())

    def test_an_unfinished_temporary_from_a_dead_process_is_removed(self):
        run_dir = self.make_run("101010-aaaaaa", state="ready", file="recording.mp4")
        self.make_video(run_dir, 64)
        dead = run_dir / "recording.part.999999.mp4"
        dead.write_bytes(b"x" * 8)
        sweep_stale(self.sessions)
        self.assertFalse(dead.exists())
        self.assertTrue((run_dir / "recording.mp4").is_file())

    def test_a_swept_run_with_no_manifest_still_gets_a_time(self):
        """A parts directory with no manifest has no `started_at` of its own; an
        undated row sorts to the bottom of the page as if it were the oldest."""
        run_dir = self.dir_of("101010-bbbbbb")
        (run_dir / "parts").mkdir(parents=True)
        (run_dir / "parts" / "game.mp4").write_bytes(b"x")
        self.assertEqual(sweep_stale(self.sessions), [run_dir])
        started = manifest_of(self, run_dir)["started_at"]
        self.assertTrue(started and started.endswith("Z"), started)

    def test_a_parts_directory_with_no_manifest_is_swept_too(self):
        run_dir = self.dir_of("101010-bbbbbb")
        (run_dir / "parts").mkdir(parents=True)
        (run_dir / "parts" / "game.mp4").write_bytes(b"x")
        removed = sweep_stale(self.sessions)
        self.assertEqual(removed, [run_dir])
        self.assertFalse((run_dir / "parts").exists())
        self.assertIn("dashboard exited", manifest_of(self, run_dir)["error"])

    def test_a_ready_run_and_its_video_are_left_alone(self):
        run_dir = self.make_run("101010-cccccc", state="ready", file="recording.mp4")
        self.make_video(run_dir, 64)
        self.assertEqual(sweep_stale(self.sessions), [])
        self.assertTrue((run_dir / "recording.mp4").is_file())
        self.assertEqual(manifest_of(self, run_dir)["state"], "ready")

    def test_a_failed_run_keeps_its_parts_as_evidence(self):
        run_dir = self.make_run("101010-dddddd", state="failed",
                                error="the composite failed: boom")
        self.assertEqual(sweep_stale(self.sessions), [])
        self.assertTrue((run_dir / "parts" / "game.mp4").is_file())
        self.assertEqual(manifest_of(self, run_dir)["error"], "the composite failed: boom")

    def test_pruning_takes_the_oldest_and_never_the_run_just_recorded(self):
        oldest = self.make_run("101010-aaaaaa", state="ready", file="recording.mp4",
                               started_at="2026-11-12T10:00:00.000Z")
        middle = self.make_run("110000-bbbbbb", state="ready", file="recording.mp4",
                               started_at="2026-11-12T11:00:00.000Z")
        newest = self.make_run("120000-cccccc", state="ready", file="recording.mp4",
                               started_at="2026-11-12T12:00:00.000Z")
        for run_dir in (oldest, middle, newest):
            self.make_video(run_dir, 1024 * 1024)
        removed = prune(self.sessions, 2, keep=newest)
        self.assertEqual([path.name for path in removed], ["101010-aaaaaa"])
        # The video goes and the run stays; see the diary test below.
        self.assertFalse((oldest / "recording.mp4").exists())
        self.assertEqual(manifest_of(self, oldest)["state"], "pruned")
        self.assertTrue(middle.exists())
        self.assertTrue((newest / "recording.mp4").is_file())  # never the kept run

    def test_pruning_keeps_the_diary_and_the_run_record(self):
        """The cap measures video. The brain logs are the operator's other named
        requirement, cost kilobytes beside the megabytes, and are not what the
        cap is holding down."""
        old = self.make_run("101010-aaaaaa", state="ready", file="recording.mp4",
                            started_at="2026-11-12T10:00:00.000Z")
        self.make_video(old, 1024 * 1024)
        (old / "brain.jsonl").write_text('{"row": 1}\n')
        (old / "brain.log").write_text("brain output\n")
        (old / "run.json").write_text('{"run": "101010-aaaaaa"}\n')
        (old / "frames").mkdir()
        (old / "frames" / "0001.jpg").write_bytes(b"frame")
        new = self.make_run("110000-bbbbbb", state="ready", file="recording.mp4",
                            started_at="2026-11-12T11:00:00.000Z")
        self.make_video(new, 1024 * 1024)

        self.assertEqual([path.name for path in prune(self.sessions, 1)], ["101010-aaaaaa"])

        self.assertFalse((old / "recording.mp4").exists())
        self.assertFalse((old / "parts").exists())
        self.assertEqual((old / "brain.jsonl").read_text(), '{"row": 1}\n')
        self.assertTrue((old / "brain.log").is_file())
        self.assertTrue((old / "run.json").is_file())
        self.assertTrue((old / "frames" / "0001.jpg").is_file())
        manifest = manifest_of(self, old)
        self.assertEqual(manifest["state"], "pruned")
        self.assertFalse(manifest["available"])
        self.assertIsNone(manifest["file"])
        self.assertEqual(manifest["error"], "pruned to make room")
        # The page still lists the run, and says there is nothing to play.
        row = next(row for row in list_recordings(self.sessions)
                   if row["run_id"] == "101010-aaaaaa")
        self.assertFalse(row["has_file"])
        self.assertTrue(row["has_diary"])

    def test_pruning_logs_each_deletion_to_the_session_log(self):
        old = self.make_run("101010-aaaaaa", state="ready", file="recording.mp4",
                            started_at="2026-11-12T10:00:00.000Z")
        new = self.make_run("110000-bbbbbb", state="ready", file="recording.mp4",
                            started_at="2026-11-12T11:00:00.000Z")
        self.make_video(old, 1024 * 1024)
        self.make_video(new, 1024 * 1024)
        prune(self.sessions, 1)
        log = (self.sessions / self.SESSION / "session.log").read_text()
        lines = [line for line in log.splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)
        self.assertIn("101010-aaaaaa", lines[0])
        self.assertIn(str(1024 * 1024), lines[0])

    def test_pruning_leaves_a_composite_that_has_not_finished(self):
        composing = self.make_run("101010-aaaaaa", state="composing")
        self.make_video(composing, 2 * 1024 * 1024)  # a partial file, mid-write
        ready = self.make_run("110000-bbbbbb", state="ready", file="recording.mp4")
        self.make_video(ready, 1024 * 1024)
        self.assertEqual(prune(self.sessions, 1), [])
        self.assertTrue(composing.exists())
        self.assertTrue(ready.exists())

    def test_zero_disables_pruning(self):
        run_dir = self.make_run("101010-aaaaaa", state="ready", file="recording.mp4")
        self.make_video(run_dir, 2 * 1024 * 1024)
        self.assertEqual(prune(self.sessions, 0), [])
        self.assertTrue(run_dir.exists())

    def test_the_listing_is_newest_first_and_says_what_is_missing(self):
        self.make_run("101010-aaaaaa", state="failed", error="dashboard exited",
                      started_at="2026-11-12T10:00:00.000Z")
        played = self.make_run("120000-bbbbbb", state="ready", file="recording.mp4",
                               started_at="2026-11-12T12:00:00.000Z")
        self.make_video(played, 512)
        (played / "brain.jsonl").write_text("")
        rows = list_recordings(self.sessions)
        self.assertEqual([row["run_id"] for row in rows], ["120000-bbbbbb", "101010-aaaaaa"])
        self.assertEqual(rows[0]["state"], "ready")
        self.assertTrue(rows[0]["has_file"])
        self.assertTrue(rows[0]["has_diary"])
        self.assertEqual(rows[0]["session_id"], self.SESSION)
        self.assertEqual(rows[0]["fps"], 15)
        self.assertEqual(rows[-1]["state"], "failed")
        self.assertIn("dashboard exited", rows[-1]["error"])
        self.assertFalse(rows[-1]["has_file"])

    def test_a_run_with_a_diary_and_no_video_is_still_listed(self):
        self.make_run("101010-aaaaaa", state="ready", file="recording.mp4")
        diary_only = self.dir_of("091500-cccccc")
        diary_only.mkdir(parents=True)
        (diary_only / "brain.jsonl").write_text(
            '{"kind": "think", "text": "nothing was recorded"}\n', encoding="utf-8"
        )
        rows = list_recordings(self.sessions)
        row = next(item for item in rows if item["run_id"] == "091500-cccccc")
        self.assertIsNone(row["state"])
        self.assertFalse(row["has_file"])
        self.assertTrue(row["has_diary"])
        # No manifest means no started_at of its own; the diary's mtime is used
        # so the row sorts and prints a time instead of landing at the bottom.
        self.assertTrue(row["started_at"])
        self.assertEqual(
            {item["run_id"] for item in rows}, {"101010-aaaaaa", "091500-cccccc"}
        )

    def test_the_listing_skips_a_run_with_no_usable_manifest(self):
        self.make_run("101010-aaaaaa", state="ready", file="recording.mp4")
        blank = self.dir_of("101010-bbbbbb")
        blank.mkdir(parents=True)
        (blank / recording.MANIFEST_NAME).write_text("{not json")
        rows = list_recordings(self.sessions)
        self.assertEqual([row["run_id"] for row in rows], ["101010-aaaaaa"])


class ProbeTests(unittest.TestCase):
    def test_a_part_without_a_moov_atom_is_refused_with_the_demuxers_words(self):
        probe = self.probe()
        part = self.base / "broken-game.mp4"
        part.write_bytes(b"x" * 32)
        caught, problem = probe_part(part, want="video", ffprobe=str(probe))
        self.assertIsNone(caught)
        self.assertIn("moov atom not found", problem)

    def test_a_part_without_its_stream_is_refused(self):
        probe = self.probe()
        part = self.base / "game.mp4"
        part.write_bytes(b"x")
        caught, problem = probe_part(part, want="audio", ffprobe=str(probe))
        self.assertIsNone(caught)
        self.assertIn("no audio stream", problem)

    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)

    def probe(self) -> Path:
        probe = self.base / "ffprobe"
        probe.write_text(FAKE_PROBE)
        probe.chmod(0o755)
        return probe


class ConfinementTests(unittest.TestCase):
    """These two functions delete files unattended, so a path that leaves the
    sessions root by a symlink or by a `..` in a manifest is refused: only a run
    directory found under the root, and only a fixed name inside it."""

    SESSION = "20261112-abc123"

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.sessions = self.base / "sessions"
        self.sessions.mkdir()
        self.outside = self.base / "outside"
        (self.outside / "parts").mkdir(parents=True)
        self.victim = self.outside / "parts" / "game.mp4"
        self.victim.write_bytes(b"evidence")

    def test_a_symlinked_session_directory_is_not_swept(self):
        (self.sessions / self.SESSION).symlink_to(self.outside)
        self.assertEqual(sweep_stale(self.sessions), [])
        self.assertTrue(self.victim.is_file())

    def test_a_symlinked_run_directory_is_not_swept(self):
        runs = self.sessions / self.SESSION / "runs"
        runs.mkdir(parents=True)
        write_recording(self.outside, state="recording")
        (runs / "101010-aaaaaa").symlink_to(self.outside)
        self.assertEqual(sweep_stale(self.sessions), [])
        self.assertTrue(self.victim.is_file())
        self.assertEqual(manifest_of(self, self.outside)["state"], "recording")

    def test_a_symlinked_run_directory_is_not_pruned(self):
        runs = self.sessions / self.SESSION / "runs"
        runs.mkdir(parents=True)
        write_recording(self.outside, state="ready", file="recording.mp4")
        (self.outside / "recording.mp4").write_bytes(b"x" * (2 * 1024 * 1024))
        (runs / "101010-aaaaaa").symlink_to(self.outside)
        self.assertEqual(prune(self.sessions, 1), [])
        self.assertTrue((self.outside / "recording.mp4").is_file())

    def test_a_symlinked_parts_directory_is_unlinked_not_emptied(self):
        run_dir = self.sessions / self.SESSION / "runs" / "101010-bbbbbb"
        run_dir.mkdir(parents=True)
        write_recording(run_dir, state="composing", parts={})
        (run_dir / "parts").symlink_to(self.outside / "parts")
        self.assertEqual(sweep_stale(self.sessions), [run_dir])
        self.assertFalse((run_dir / "parts").exists())
        self.assertTrue(self.victim.is_file())

    def test_a_part_path_that_climbs_out_of_the_run_is_not_deleted(self):
        run_dir = self.sessions / self.SESSION / "runs" / "101010-cccccc"
        run_dir.mkdir(parents=True)
        write_recording(run_dir, state="composing",
                        parts={"game": "../../../../outside/parts/game.mp4"})
        sweep_stale(self.sessions)
        self.assertTrue(self.victim.is_file())
        self.assertEqual(manifest_of(self, run_dir)["state"], "failed")


class RealCompositeTests(unittest.TestCase):
    """One composite through real ffmpeg, because the padding has to be measured.

    The fake pair above proves the argv; this proves the file. Skipped on a
    machine without ffmpeg, which is the same machine that cannot record.
    """

    def setUp(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if ffmpeg is None or ffprobe is None:
            self.skipTest("ffmpeg is not installed")
        self.ffmpeg: str = ffmpeg
        self.ffprobe: str = ffprobe
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.sessions = self.base / "sessions"
        self.run_dir = self.sessions / "101010-a1b2c3" / "runs" / "101011-deadbe"
        self.run_dir.mkdir(parents=True)
        cfg = load_config(None, env={})
        cfg = replace(cfg, paths=replace(cfg.paths, session_root=str(self.sessions)))
        # keep_parts: the second test composites the same three files again.
        self.cfg = replace(cfg, recording=replace(cfg.recording, ffmpeg=self.ffmpeg,
                                                   keep_parts=True))
        self.parts = {
            "game": self.build("game.mp4", "testsrc=size=960x540:rate=15:duration=1",
                               ["-c:v", "libx264", "-pix_fmt", "yuv420p"]),
            "robot": self.build("robot.mp4", "testsrc2=size=720x540:rate=10:duration=3",
                                ["-c:v", "libx264", "-pix_fmt", "yuv420p"]),
            "audio": self.build("audio.m4a", "sine=frequency=440:duration=2",
                                ["-c:a", "aac"]),
        }
        write_recording(self.run_dir, duration_s=3.0, write_parts=False,
                        parts={name: f"parts/{path.name}" for name, path in self.parts.items()},
                        part_seconds={"game": 1.0, "robot": 3.0, "audio": 2.0})

    def build(self, name: str, source: str, codec: list[str]) -> Path:
        out = self.run_dir / "parts" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [self.cfg.recording.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
             "-y", "-f", "lavfi", "-i", source, *codec, str(out)],
            check=True, stdin=subprocess.DEVNULL, capture_output=True,
        )
        return out

    def length_of(self, path: Path) -> float:
        done = subprocess.run(
            [self.ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            check=True, stdin=subprocess.DEVNULL, capture_output=True, text=True,
        )
        return float(done.stdout.strip())

    def test_a_short_part_does_not_truncate_the_composite(self):
        manifest = run_composite(self.run_dir, self.cfg)
        self.assertEqual(manifest["state"], "ready")
        self.assertAlmostEqual(self.length_of(self.run_dir / "recording.mp4"), 3.0, delta=0.3)

    def test_the_unpadded_argv_would_have_stopped_at_the_short_part(self):
        """The measurement the padding answers: `-shortest` ends the file at the
        shortest stream, which is a part that was demoted mid-run."""
        out = self.base / "naive.mp4"
        argv = composite_argv(
            game=self.parts["game"], robot=self.parts["robot"], audio=self.parts["audio"],
            out=out, fps=15, crf=28, preset="veryfast", ffmpeg=self.cfg.recording.ffmpeg,
        )
        subprocess.run(argv, check=True, stdin=subprocess.DEVNULL, capture_output=True)
        self.assertLess(self.length_of(out), 2.5)
