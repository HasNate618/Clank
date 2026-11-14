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
        with mock.patch.object(ears, "_pactl", side_effect=self.fake_pactl), \
                self.assertRaises(ears.EarsError) as caught:
            ears.route("Factorio")
        self.assertIn("no playback stream", str(caught.exception))
        self.assertIn("Zen", str(caught.exception))

    def test_pactl_failure_is_an_ears_error(self):
        boom = mock.Mock(returncode=1, stderr=b"no such sink", stdout=b"")
        with mock.patch.object(ears.subprocess, "run", return_value=boom), \
                self.assertRaises(ears.EarsError) as caught:
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
        blocks = (
            [silence(ears.BLOCK_S)] * 12
            + [pcm(8000, ears.BLOCK_S)] * 10
            + [silence(ears.BLOCK_S)] * 60
        )
        stream = FakeStream(blocks)
        hearing = ears.listen(reader=stream, max_s=10.0)
        self.assertTrue(hearing.ok, hearing.error)
        self.assertGreater(hearing.speech_s, 0.4)
        self.assertLess(hearing.speech_s, 0.7)
        # the hang ended it: the remaining quiet blocks were never read
        self.assertLess(len(stream.read_sizes), len(blocks))
        self.assertEqual(hearing.ended_by, "hang")
        self.assertTrue(hearing.wav.startswith(b"RIFF"))

    def test_a_1s_pause_is_the_same_utterance(self):
        # JobBot: "...museum." + ~1.2 s + "Here's my card." One line, not two: the
        # pause is shorter than the hang, and capturing it as two utterances loses the
        # offer that names the task. MEASURED 2026-11-13, run 190131-679de2.
        gap = int(1.2 / ears.BLOCK_S)
        stream = FakeStream(
            [silence(ears.BLOCK_S)] * 12
            + [pcm(8000, ears.BLOCK_S)] * 8
            + [silence(ears.BLOCK_S)] * gap
            + [pcm(8000, ears.BLOCK_S)] * 8
            + [silence(ears.BLOCK_S)] * 80
        )
        hearing = ears.listen(reader=stream, max_s=10.0)
        self.assertTrue(hearing.ok, hearing.error)
        self.assertGreater(hearing.speech_s, 0.6)
        self.assertEqual(hearing.ended_by, "hang")

    def test_a_2_5s_pause_is_two_utterances(self):
        gap = int(2.5 / ears.BLOCK_S)
        stream = FakeStream(
            [silence(ears.BLOCK_S)] * 12
            + [pcm(8000, ears.BLOCK_S)] * 8
            + [silence(ears.BLOCK_S)] * gap
            + [pcm(8000, ears.BLOCK_S)] * 8
        )
        hearing = ears.listen(reader=stream, max_s=10.0)
        self.assertTrue(hearing.ok, hearing.error)
        self.assertLess(hearing.speech_s, 0.6)

    def test_a_loud_opening_is_heard_once_a_quiet_fifth_arrives(self):
        # The floor is the 20th percentile of every pre-start block, so a loud opening needs a
        # quiet fifth before the gate can open. This pins the limit honestly: the utterance is
        # the SECOND burst, not the first - the loud opening is lost, and that is the price of
        # a percentile floor rather than a rolling median (which would never open at all).
        stream = FakeStream(
            [pcm(8000, ears.BLOCK_S)] * 20
            + [silence(ears.BLOCK_S)] * 10
            + [pcm(8000, ears.BLOCK_S)] * 12
            + [silence(ears.BLOCK_S)] * 80
        )
        hearing = ears.listen(reader=stream, max_s=10.0)
        self.assertTrue(hearing.ok, hearing.error)
        self.assertLess(hearing.speech_s, 1.0, "the loud opening is not part of the utterance")

    def test_a_capture_that_is_loud_throughout_never_opens(self):
        # No quiet fifth, no floor, no gate: the documented limit, not a bug to fix here.
        stream = FakeStream([pcm(8000, ears.BLOCK_S)] * 200)
        hearing = ears.listen(reader=stream, max_s=2.0)
        self.assertFalse(hearing.ok)
        self.assertEqual(hearing.ended_by, "never-started")

    def test_a_deadline_ending_keeps_the_whole_line(self):
        # The trim is `hang_s - tail_s` of quiet, and a line cut by max_s has no quiet to trim.
        # At hang_s = 2.0 the old unconditional trim deleted 1.8 s of live speech, which is the
        # next instruction once confirm() feeds the transcript forward.
        stream = FakeStream(
            [silence(ears.BLOCK_S)] * 12          # a quiet floor, so the gate can open at all
            + [pcm(8000, ears.BLOCK_S)] * 400     # one long line, never a 2 s pause
        )
        hearing = ears.listen(reader=stream, max_s=3.0)
        self.assertTrue(hearing.ok, hearing.error)
        self.assertEqual(hearing.ended_by, "deadline")
        self.assertGreater(hearing.seconds, 2.5, "the trim ate the end of the line")

    def test_a_loud_opening_does_not_set_the_floor(self):
        # The confirm window of run 190131-679de2 in miniature: the loudest part of the line
        # is at its start, so a median of the first half second is the wrong floor for the
        # rest of it (measured there: 2445 -> gate 9781 against a 4415 peak). On this stream
        # the rolling median re-medians on the 2000 blocks, sits at a gate of 8000 and never
        # opens; the cumulative 20th percentile keeps the line's own quiet, and hears it.
        stream = FakeStream(
            [pcm(6000, ears.BLOCK_S)] * 10       # the loud opening
            + [silence(ears.BLOCK_S)] * 4        # the pause between words
            + [pcm(2000, ears.BLOCK_S)] * 40     # the rest of the line, quieter
            + [silence(ears.BLOCK_S)] * 80
        )
        hearing = ears.listen(reader=stream, max_s=10.0)
        self.assertTrue(hearing.ok, hearing.error)
        self.assertLess(hearing.gate, 6000, "the loud opening set the gate")
        self.assertEqual(hearing.ended_by, "hang")

    def test_keeps_the_pre_roll_so_the_first_syllable_survives(self):
        stream = FakeStream(
            [silence(ears.BLOCK_S)] * 12
            + [pcm(8000, ears.BLOCK_S)] * 6
            # longer than the hang, so the hang ends it: at 30 blocks (1.5 s) the stream ran to
            # EOF instead and the test measured the EOF path, not the pre-roll
            + [silence(ears.BLOCK_S)] * int(3.0 / ears.BLOCK_S)
        )
        hearing = ears.listen(reader=stream, max_s=10.0)
        self.assertTrue(hearing.ok, hearing.error)
        self.assertEqual(hearing.ended_by, "hang")
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
        self.assertEqual(hearing.ended_by, "never-started")

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


class RingTests(unittest.TestCase):
    """One recorder for the run, read at any offset.

    The reason this exists is measured, run 181738-ee1d6d: the recording's continuous capture
    held 12.8 s of JobBot's greeting (spoken while the body was resetting) and 14.9 s of his
    next instruction (spoken during a generated reach), and the brain heard one line of it,
    because `listen` records only while it listens and `confirm` opens its window after the act.
    """

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = pathlib.Path(self.dir.name) / "ring.raw"

    def write(self, blocks):
        with self.path.open("ab") as handle:
            for block in blocks:
                handle.write(block)
        return self.path.stat().st_size

    def ring(self):
        return ears.Ring(self.path, device="test.monitor")

    def test_a_reader_returns_whole_blocks_from_an_offset(self):
        size = self.write([silence(0.05)] * 4)
        reader = self.ring().reader(0)
        first = reader.read(int(ears.BLOCK_S * ears.RATE) * 2)
        self.assertEqual(len(first), int(ears.BLOCK_S * ears.RATE) * 2)
        self.assertEqual(reader.offset, len(first))
        self.assertLess(reader.offset, size)

    def test_a_partial_block_is_not_handed_out(self):
        """A read boundary in the middle of a block would misalign every RMS after it."""
        block = int(ears.BLOCK_S * ears.RATE) * 2
        with self.path.open("ab") as handle:
            handle.write(silence(0.05) + b"\x01\x02\x03")
        reader = self.ring().reader(0)
        self.assertEqual(len(reader.read(block * 4)), block)
        self.assertEqual(reader.read(block * 4), b"")

    def test_read_back_is_the_tail_and_read_from_is_the_past(self):
        self.write([silence(0.05)] * 20 + [pcm(6000, 0.05)] * 10)
        ring = self.ring()
        block = int(ears.BLOCK_S * ears.RATE) * 2
        self.assertEqual(len(ring.read_back(0.5)), 10 * block)
        self.assertEqual(len(ring.read_from(20 * block)), 10 * block)
        self.assertEqual(len(ring.read_from(20 * block, max_s=0.25)), 5 * block)
        self.assertEqual(ring.read_from(10_000_000), b"")

    def test_a_mark_is_block_aligned(self):
        self.write([silence(0.05)] * 3 + [b"\x01\x02\x03"])
        ring = self.ring()
        self.assertEqual(ring.mark() % ring.block, 0)

    def test_listen_replays_from_the_beginning(self):
        """Turn one asks the ring for the greeting, however long ago it was spoken."""
        self.write([silence(0.05)] * 20 + [pcm(6000, 0.05)] * 8 + [silence(0.05)] * 60)
        hearing = self.ring().listen(start=0, max_s=5.0)
        self.assertTrue(hearing.ok, hearing.error)
        self.assertEqual(hearing.ended_by, "hang")
        self.assertGreaterEqual(hearing.speech_s, 0.35)

    def test_a_replayed_utterance_is_not_heard_twice(self):
        self.write([silence(0.05)] * 20 + [pcm(6000, 0.05)] * 8 + [silence(0.05)] * 60)
        ring = self.ring()
        self.assertTrue(ring.listen(start=0, max_s=5.0).ok)
        self.assertFalse(ring.listen(max_s=1.0).ok, "the cursor must resume after the utterance")
        self.assertEqual(ring.cursor, ring.mark())

    def test_silence_never_opens_the_gate_and_still_advances(self):
        self.write([silence(0.05)] * 100)
        ring = self.ring()
        hearing = ring.listen(start=0, max_s=1.0)
        self.assertFalse(hearing.ok)
        self.assertEqual(hearing.ended_by, "never-started")
        self.assertGreater(ring.cursor, 0)
