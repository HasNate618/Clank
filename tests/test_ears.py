"""The ear: routing the game into a capture sink, and is there speech in it.

No audio device needed. `pactl` and the recorder are mocked, and `listen` takes a reader, so a
fake stream of PCM blocks exercises the gate, the pre-roll and the trim.

The measurements that shaped this module are in research/hearing.md: `pw-record --target`
does not link on this machine and silently records the muted microphone, so the only capture
path here is `parecord --device=<capture sink>.monitor`.
"""

import array
import io
import json
import math
import pathlib
import tempfile
import time
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
    """What a capture looks like: fixed-size blocks then EOF."""

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


def sink_input(index: int, app: str, sink: str) -> dict:
    return {"index": index, "sink": sink, "properties": {"application.name": app}}


def sink_entry(index: int, name: str) -> dict:
    return {"index": index, "name": name}


class RoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def fake_pactl(self, *args: str, timeout: float = 10.0) -> str:
        self.calls.append(args)
        joined = " ".join(args)
        if joined == "-f json list sinks":
            return json.dumps([sink_entry(5280, "bluez_output.headphones"),
                               sink_entry(7, ears.SINK_NAME)])
        if joined == "-f json list sink-inputs":
            return json.dumps([sink_input(11, "JobSimulator.exe", "5280"),
                               sink_input(12, "Zen", "5280")])
        if joined == "get-default-sink":
            return "bluez_output.headphones\n"
        if joined == "list short modules":
            return ""
        if joined.startswith("move-sink-input"):
            return ""
        if joined.startswith("load-module"):
            return "536870917"
        raise AssertionError(f"unexpected pactl call: {joined}")

    def test_sink_names_come_from_pactl(self):
        with mock.patch.object(ears, "_pactl", side_effect=self.fake_pactl):
            self.assertEqual(ears.sink_names(), ["bluez_output.headphones", ears.SINK_NAME])

    def test_game_streams_match_the_application_name(self):
        with mock.patch.object(ears, "_pactl", side_effect=self.fake_pactl):
            found = ears.game_streams("JobSimulator")
        self.assertEqual([s["index"] for s in found], [11])

    def test_a_microphone_or_another_app_is_not_the_game(self):
        with mock.patch.object(ears, "_pactl", side_effect=self.fake_pactl):
            self.assertEqual(ears.game_streams("JobSimulator"), ears.game_streams("jobsimulator.exe")[:1])
            self.assertEqual(ears.game_streams("Zen")[0]["index"], 12)

    def test_an_empty_pattern_matches_nothing(self):
        with mock.patch.object(ears, "_pactl", side_effect=self.fake_pactl):
            self.assertEqual(ears.game_streams("   "), [])

    def test_ensure_sink_creates_it_once(self):
        with mock.patch.object(ears, "_pactl", side_effect=self.fake_pactl):
            self.assertFalse(ears.ensure_sink())  # already in the fake sink list
        with mock.patch.object(ears, "_pactl", side_effect=self.fake_pactl):
            self.assertTrue(ears.ensure_sink("clank-missing"))

    def test_ensure_loopback_is_skipped_when_one_exists(self):
        modules = "5\tmodule-loopback\tsource=clank-game.monitor sink=x"
        with mock.patch.object(ears, "_pactl", side_effect=self.fake_pactl) as pactl:
            pactl.side_effect = lambda *a, **k: (modules if a[:2] == ("list", "short")
                                                else self.fake_pactl(*a, **k))
            self.assertFalse(ears.ensure_loopback())

    def test_ensure_loopback_restores_the_operators_audio(self):
        with mock.patch.object(ears, "_pactl", side_effect=self.fake_pactl):
            self.assertTrue(ears.ensure_loopback())
        self.assertIn(("load-module", "module-loopback", "source=clank-game.monitor",
                       "sink=bluez_output.headphones", "latency_msec=20"), self.calls)

    def test_move_skips_a_stream_already_behind_the_capture_sink(self):
        backend = {"-f json list sinks": json.dumps([sink_entry(7, ears.SINK_NAME)]),
                   "-f json list sink-inputs": json.dumps(
                       [sink_input(11, "JobSimulator.exe", "7")])}
        with mock.patch.object(ears, "_pactl", side_effect=lambda *a, **k: backend[" ".join(a)]):
            self.assertEqual(ears.move_game_streams(), [])

    def test_move_routes_the_game_and_leaves_the_desktop_alone(self):
        with mock.patch.object(ears, "_pactl", side_effect=self.fake_pactl):
            moved = ears.move_game_streams()
        self.assertEqual(moved, [11])
        self.assertIn(("move-sink-input", "11", ears.SINK_NAME), self.calls)

    def test_route_reports_what_it_changed(self):
        with mock.patch.object(ears, "_pactl", side_effect=self.fake_pactl):
            report = ears.route()
        self.assertEqual(report["device"], "clank-game.monitor")
        self.assertEqual(report["moved"], [11])
        self.assertFalse(report["created"])

    def test_route_refuses_when_the_game_is_not_playing(self):
        with mock.patch.object(ears, "_pactl", side_effect=self.fake_pactl):
            with self.assertRaises(ears.EarsError) as caught:
                ears.route("Factorio")
        self.assertIn("no playback stream", str(caught.exception))
        self.assertIn("Zen", str(caught.exception))

    def test_pactl_failure_is_an_ears_error(self):
        boom = mock.Mock(returncode=1, stderr=b"no such sink", stdout=b"")
        with mock.patch.object(ears.subprocess, "run", return_value=boom):
            with self.assertRaises(ears.EarsError) as caught:
                ears._pactl("list", "sinks")
        self.assertIn("no such sink", str(caught.exception))

    def test_missing_pactl_is_an_ears_error(self):
        with mock.patch.object(ears.subprocess, "run", side_effect=FileNotFoundError), \
                self.assertRaises(ears.EarsError):
            ears._pactl("list", "sinks")


class PcmOffsetTest(unittest.TestCase):
    def test_finds_the_data_chunk(self):
        head = b"RIFF....WAVEfmt " + b"\x00" * 16 + b"data" + b"\xff" * 4
        self.assertEqual(ears.pcm_offset(head), head.index(b"data") + 8)

    def test_missing_chunk_is_minus_one(self):
        self.assertEqual(ears.pcm_offset(b"RIFF-only-header"), -1)


class TailReaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.tmp.name) / "capture.wav"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_wav(self, payload: bytes) -> None:
        header = b"RIFF" + b"\x00" * 4 + b"WAVEfmt " + b"\x10\x00\x00\x00" + b"\x00" * 16
        self.path.write_bytes(header + b"data" + b"\xff\xff\xff\xff" + payload)

    def test_skips_the_header_and_returns_samples(self):
        self.write_wav(pcm(500, 0.05))
        reader = ears.TailReader(self.path, wait_s=0.05)
        self.assertEqual(len(reader.read(BLOCK)), BLOCK)

    def test_waits_for_a_file_that_does_not_exist_yet(self):
        reader = ears.TailReader(self.path, wait_s=0.05)
        self.assertEqual(reader.read(BLOCK), b"")
        self.write_wav(pcm(500, 0.05))
        self.assertEqual(len(reader.read(BLOCK)), BLOCK)

    def test_it_reads_only_what_is_there(self):
        self.write_wav(pcm(500, 0.05))
        reader = ears.TailReader(self.path, wait_s=0.05)
        chunk = reader.read(BLOCK * 4)
        self.assertEqual(len(chunk), BLOCK)
        self.assertEqual(reader.read(BLOCK), b"")

    def test_a_headerless_stream_is_read_as_raw(self):
        self.path.write_bytes(b"\x01" * 5000)
        reader = ears.TailReader(self.path, wait_s=0.05)
        self.assertEqual(len(reader.read(BLOCK)), BLOCK)


class LevelTest(unittest.TestCase):
    def test_silence_is_zero(self):
        self.assertEqual(ears.level(pcm(0, 0.1)), (0, 0.0))

    def test_peak_and_rms_of_a_square_wave_agree(self):
        peak, rms = ears.level(pcm(1000, 0.1))
        self.assertEqual(peak, 1000)
        self.assertTrue(math.isclose(rms, 1000.0, rel_tol=1e-6))

    def test_odd_length_is_tolerated(self):
        peak, _ = ears.level(pcm(500, 0.05) + b"\x01")
        self.assertEqual(peak, 500)

    def test_empty_is_zero(self):
        self.assertEqual(ears.level(b""), (0, 0.0))


class WavTest(unittest.TestCase):
    def test_round_trip(self):
        raw = pcm(250, 0.2)
        back, rate, channels = ears.wav_pcm(ears.wav_bytes(raw))
        self.assertEqual((rate, channels), (ears.RATE, 1))
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

    def test_digital_silence_then_speech_is_heard(self):
        """The game is exactly zero between lines; a zero floor must not mean deaf."""
        stream = FakeStream(
            [silence(ears.BLOCK_S)] * 20 + [pcm(6000, ears.BLOCK_S)] * 8
            + [silence(ears.BLOCK_S)] * 40
        )
        hearing = ears.listen(reader=stream, max_s=10.0)
        self.assertTrue(hearing.ok, hearing.error)
        self.assertEqual(hearing.floor, 0.0)

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

    def test_continuous_loud_audio_is_not_an_utterance(self):
        """The gate is contrast: a floor measured on speech is a floor of speech.

        Worth stating because it is the limit of the design - a game that plays music
        continuously at a constant level would never open the gate.
        """
        stream = FakeStream([pcm(9000, ears.BLOCK_S)] * 40)
        hearing = ears.listen(reader=stream, max_s=1.0)
        self.assertFalse(hearing.ok)
        self.assertGreater(hearing.gate, 9000)

    def test_a_dead_reader_does_not_spin_to_the_deadline(self):
        stream = FakeStream([silence(ears.BLOCK_S)] * 12)
        started = time.monotonic()
        hearing = ears.listen(reader=stream, max_s=30.0)
        self.assertFalse(hearing.ok)
        self.assertLess(time.monotonic() - started, 5.0, "listen waited out max_s")


class CaptureStub:
    """Stands in for `ears.capture`: a reader, a finished process, and routing info."""

    def __init__(self, blocks: list[bytes], device: str = "clank-game.monitor"):
        self.reader = FakeStream(blocks)
        self.proc = mock.Mock()
        self.proc.poll.return_value = 0
        self.info = {"device": device}
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        return self.reader, self.proc, self.info

    def __exit__(self, *exc):
        self.exited += 1
        return False


class RecordTest(unittest.TestCase):
    def test_fixed_length_capture(self):
        stub = CaptureStub([pcm(900, 0.5)])
        with mock.patch.object(ears, "capture", return_value=stub):
            hearing = ears.record(0.5)
        self.assertTrue(math.isclose(hearing.seconds, 0.5, abs_tol=0.06), hearing.seconds)
        self.assertEqual(hearing.peak, 900)
        self.assertEqual(hearing.device, "clank-game.monitor")
        self.assertEqual((stub.entered, stub.exited), (1, 1))

    def test_a_dead_capture_is_an_error_not_a_crash(self):
        stub = CaptureStub([])
        stub.info["stderr"] = "connection failed"
        with mock.patch.object(ears, "capture", return_value=stub):
            hearing = ears.record(0.5)
        self.assertFalse(hearing.ok)
        self.assertIn("no audio received", hearing.error)

    def test_listen_uses_the_capture_device_and_cleans_up(self):
        stub = CaptureStub([silence(ears.BLOCK_S)] * 12
                           + [pcm(8000, ears.BLOCK_S)] * 6
                           + [silence(ears.BLOCK_S)] * 20)
        with mock.patch.object(ears, "capture", return_value=stub):
            hearing = ears.listen(max_s=10.0)
        self.assertTrue(hearing.ok, hearing.error)
        self.assertEqual(hearing.device, "clank-game.monitor")
        self.assertEqual((stub.entered, stub.exited), (1, 1))


class CliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = {
            "-f json list sinks": json.dumps([sink_entry(7, ears.SINK_NAME)]),
            "-f json list sink-inputs": json.dumps(
                [sink_input(11, "JobSimulator.exe", "7")]),
            "get-default-sink": "bluez_output.headphones\n",
            "list short modules": "",
        }

    def fake_pactl(self, *args: str, timeout: float = 10.0) -> str:
        try:
            return self.backend[" ".join(args)]
        except KeyError:
            return "0"

    def test_list_prints_the_sinks_and_the_device(self):
        out = io.StringIO()
        with mock.patch.object(ears, "_pactl", side_effect=self.fake_pactl), \
                mock.patch("sys.stdout", out):
            self.assertEqual(ears.main(["--list"]), 0)
        text = out.getvalue()
        self.assertIn(ears.SINK_NAME, text)
        self.assertIn("JobSimulator.exe", text)
        self.assertIn("clank-game.monitor", text)

    def test_route_only_reports_the_routing(self):
        out = io.StringIO()
        with mock.patch.object(ears, "_pactl", side_effect=self.fake_pactl), \
                mock.patch("sys.stdout", out):
            self.assertEqual(ears.main(["--route-only", "--json"]), 0)
        report = json.loads(out.getvalue())
        self.assertEqual(report["device"], "clank-game.monitor")

    def test_no_game_stream_exits_two_with_a_reason(self):
        self.backend["-f json list sink-inputs"] = json.dumps([sink_input(12, "Zen", "7")])
        err = io.StringIO()
        with mock.patch.object(ears, "_pactl", side_effect=self.fake_pactl), \
                mock.patch("sys.stderr", err), mock.patch("sys.stdout", io.StringIO()):
            self.assertEqual(ears.main([]), 2)
        self.assertIn("no playback stream", err.getvalue())


if __name__ == "__main__":
    unittest.main()
