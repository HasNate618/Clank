"""The ear: which node is the game, and is there speech in it.

No PipeWire needed. `listen` takes a reader, so a fake stream of PCM blocks exercises the
gate, the pre-roll and the trim; `_spawn` is patched for `record`.
"""

import array
import io
import math
import unittest
from unittest import mock

from clank import ears

BLOCK = int(ears.BLOCK_S * ears.RATE) * 2  # 1600 bytes, 800 s16 samples


def pcm(amplitude: int, seconds: float) -> bytes:
    samples = array.array("h", [amplitude] * int(ears.RATE * seconds))
    return samples.tobytes()


def silence(seconds: float) -> bytes:
    return pcm(0, seconds)


class FakeStream:
    """What pw-record's stdout looks like: fixed-size blocks then EOF."""

    def __init__(self, blocks: list[bytes]):
        self.blocks = list(blocks)
        self.read_sizes: list[int] = []

    def read(self, n: int) -> bytes:
        self.read_sizes.append(n)
        if not self.blocks:
            return b""
        block = self.blocks.pop(0)
        if len(block) != n:
            self.blocks.insert(0, block[n:])
            return block[:n]
        return block


def node(node_id: int, name: str, media_class: str = ears.PLAYBACK_CLASS) -> dict:
    return {
        "id": node_id,
        "type": "PipeWire:Interface:Node",
        "info": {"props": {"media.class": media_class, "application.name": name,
                           "node.name": name}},
    }


def dump_nodes() -> list[dict]:
    """A pw-dump excerpt: one microphone, one app that is not the game, the game."""
    return [
        node(151, "alsa_input.pci", "Audio/Source"),
        node(268, "Zen"),
        node(161, "JobSimulator.exe"),
    ]


class PlaybackNodesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dump = dump_nodes()

    def test_only_playback_streams_are_candidates(self):
        found = ears.playback_nodes(self.dump)
        self.assertEqual([n for n, _, _ in found], [161, 268])

    def test_picks_the_game_by_substring(self):
        self.assertEqual(ears.pick_node(self.dump, "JobSimulator"), (161, "JobSimulator.exe"))

    def test_match_is_case_insensitive(self):
        self.assertIsNotNone(ears.pick_node(self.dump, "jobs"))

    def test_microphone_never_matches(self):
        dump = [node(151, "JobSimulator", "Audio/Source")]
        self.assertIsNone(ears.pick_node(dump, "JobSimulator"))

    def test_no_match_returns_none_rather_than_guessing(self):
        self.assertIsNone(ears.pick_node(self.dump, "Factorio"))

    def test_empty_pattern_matches_nothing(self):
        self.assertIsNone(ears.pick_node(self.dump, "   "))

    def test_ignores_non_node_objects(self):
        dump = [{"id": 5, "type": "PipeWire:Interface:Link"}, {"id": 161}, "nonsense"]
        self.assertEqual(ears.playback_nodes(dump), [])

    def test_game_node_reports_what_is_playing_instead(self):
        with mock.patch.object(ears, "audio_dump", return_value=self.dump):
            self.assertEqual(ears.game_node("JobSimulator")[0], 161)
            with self.assertRaises(ears.EarsError) as caught:
                ears.game_node("Factorio")
        self.assertIn("Zen", str(caught.exception))

    def test_game_node_survives_pw_dump_missing(self):
        with mock.patch.object(ears.subprocess, "run", side_effect=FileNotFoundError), \
                self.assertRaises(ears.EarsError) as caught:
            ears.audio_dump()
        self.assertIn("pw-dump", str(caught.exception))


class LevelTest(unittest.TestCase):
    def test_silence_is_zero(self):
        self.assertEqual(ears.level(pcm(0, 0.1)), (0, 0.0))

    def test_peak_and_rms_of_a_square_wave_agree(self):
        peak, rms = ears.level(pcm(1000, 0.1))
        self.assertEqual(peak, 1000)
        self.assertTrue(math.isclose(rms, 1000.0, rel_tol=1e-6))

    def test_odd_length_is_tolerated(self):
        peak, rms = ears.level(pcm(500, 0.05) + b"\x01")
        self.assertEqual(peak, 500)
        self.assertGreater(rms, 0.0)

    def test_empty_is_zero(self):
        self.assertEqual(ears.level(b""), (0, 0.0))


class WavTest(unittest.TestCase):
    def test_round_trip(self):
        raw = pcm(250, 0.2)
        data = ears.wav_bytes(raw)
        back, rate, channels = ears.wav_pcm(data)
        self.assertEqual(rate, ears.RATE)
        self.assertEqual(channels, 1)
        self.assertEqual(back, raw)

    def test_header_is_what_a_model_expects(self):
        self.assertTrue(ears.wav_bytes(pcm(0, 0.05)).startswith(b"RIFF"))


class ListenTest(unittest.TestCase):
    def test_finds_one_utterance_and_stops_after_the_hang(self):
        stream = FakeStream(
            [silence(ears.BLOCK_S)] * 12
            + [pcm(8000, ears.BLOCK_S)] * 10
            + [silence(ears.BLOCK_S)] * 40
        )
        hearing = ears.listen(reader=stream, max_s=10.0)
        self.assertTrue(hearing.ok, hearing.error)
        self.assertGreater(hearing.speech_s, 0.4)
        self.assertLess(hearing.speech_s, 0.7)
        # the hang ended it: the remaining quiet blocks were never read
        self.assertLess(len(stream.read_sizes), 62)
        self.assertTrue(hearing.wav.startswith(b"RIFF"))

    def test_keeps_the_pre_roll_so_the_first_syllable_survives(self):
        stream = FakeStream(
            [silence(ears.BLOCK_S)] * 12
            + [pcm(8000, ears.BLOCK_S)] * 6
            + [silence(ears.BLOCK_S)] * 30
        )
        hearing = ears.listen(reader=stream, max_s=10.0)
        self.assertTrue(hearing.ok, hearing.error)
        # 6 loud blocks plus up to pre_roll + hang of quiet around them
        self.assertGreater(hearing.seconds, 0.3)
        self.assertLess(hearing.seconds, 6 * ears.BLOCK_S + ears.BLOCK_S + 1.0)

    def test_trim_does_not_eat_the_utterance(self):
        loud = 1.0
        stream = FakeStream(
            [silence(ears.BLOCK_S)] * 12
            + [pcm(8000, ears.BLOCK_S)] * int(loud / ears.BLOCK_S)
            + [silence(ears.BLOCK_S)] * 30
        )
        hearing = ears.listen(reader=stream, max_s=10.0)
        self.assertGreater(hearing.seconds, loud * 0.8)

    def test_a_room_tone_alone_is_not_speech(self):
        stream = FakeStream([pcm(120, ears.BLOCK_S)] * 60)
        hearing = ears.listen(reader=stream, max_s=10.0)
        self.assertFalse(hearing.ok)
        self.assertIn("no speech", hearing.error)
        self.assertGreater(hearing.gate, 120)

    def test_a_loud_room_raises_the_gate(self):
        stream = FakeStream(
            [pcm(400, ears.BLOCK_S)] * 12 + [pcm(1200, ears.BLOCK_S)] * 4
            + [pcm(400, ears.BLOCK_S)] * 30
        )
        hearing = ears.listen(reader=stream, max_s=10.0)
        self.assertFalse(hearing.ok, "1200 is under 4x a 400 floor")
        self.assertGreater(hearing.gate, 1200)

    def test_truncated_stream_still_returns_what_it_had(self):
        stream = FakeStream(
            [silence(ears.BLOCK_S)] * 12 + [pcm(8000, ears.BLOCK_S)] * 6
        )
        hearing = ears.listen(reader=stream, max_s=10.0)
        self.assertTrue(hearing.ok, hearing.error)
        self.assertGreater(hearing.seconds, 0.05)

    def test_max_s_bounds_the_wait(self):
        stream = FakeStream([silence(ears.BLOCK_S)] * 10_000)
        hearing = ears.listen(reader=stream, max_s=0.0)
        self.assertFalse(hearing.ok)
        self.assertLess(len(stream.read_sizes), 5)


class RecordTest(unittest.TestCase):
    def test_fixed_length_capture(self):
        fake = mock.Mock()
        fake.stdout = io.BytesIO(pcm(900, 1.0))
        fake.stderr = io.BytesIO(b"")
        fake.poll.return_value = None
        fake.wait.return_value = 0
        with mock.patch.object(ears, "_spawn", return_value=fake), \
                mock.patch.object(ears, "game_node", return_value=(161, "JobSimulator.exe")):
            hearing = ears.record(0.5)
        self.assertTrue(math.isclose(hearing.seconds, 0.5, abs_tol=0.06), hearing.seconds)
        self.assertEqual(hearing.peak, 900)
        self.assertEqual(hearing.node, 161)

    def test_a_dead_stream_is_an_error_not_a_crash(self):
        fake = mock.Mock()
        fake.stdout = io.BytesIO(b"")
        fake.stderr = io.BytesIO(b"connection failed")
        fake.poll.return_value = 1
        fake.wait.return_value = 1
        with mock.patch.object(ears, "_spawn", return_value=fake), \
                mock.patch.object(ears, "game_node", return_value=(161, "JobSimulator.exe")):
            hearing = ears.record(0.5)
        self.assertFalse(hearing.ok)
        self.assertIn("connection failed", hearing.error)


class CliTest(unittest.TestCase):
    def test_list_prints_the_playback_nodes(self):
        dump = [node(161, "JobSimulator.exe"), node(268, "Zen")]
        out = io.StringIO()
        with mock.patch.object(ears, "audio_dump", return_value=dump), \
                mock.patch("sys.stdout", out):
            self.assertEqual(ears.main(["--list"]), 0)
        self.assertIn("JobSimulator.exe", out.getvalue())
        self.assertNotIn("alsa_input", out.getvalue())

    def test_missing_node_exits_two_with_a_reason(self):
        err = io.StringIO()
        with mock.patch.object(ears, "audio_dump", return_value=[]), \
                mock.patch("sys.stderr", err), mock.patch("sys.stdout", io.StringIO()):
            self.assertEqual(ears.main([]), 2)
        self.assertIn("no playback stream", err.getvalue())


if __name__ == "__main__":
    unittest.main()
