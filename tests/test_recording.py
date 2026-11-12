"""Turning a live run into a watchable file: pure parts first."""

from __future__ import annotations

import unittest
from pathlib import Path

from clank import recording
from clank.recording import composite_argv, parse_range


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
