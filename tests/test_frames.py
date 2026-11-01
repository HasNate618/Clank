"""Tests for the frame sources: the container crop and the replay fixture.

`crop_jpeg` shells out to ffmpeg, so `subprocess.run` is mocked throughout: the
ffmpeg *output* is not what is under test, the argv it is handed is. That is the
half that can be wrong in a way nobody notices - a swapped crop argument still
returns a plausible JPEG.
"""

import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

from clank import frames
from clank.frames import Frame

#: The hand-drawn tray box from the 2026-11-01 measurement.
TRAY_BOX = (0.270, 0.325, 0.548, 0.566)


def completed(stdout: bytes = b"", stderr: bytes = b"") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["ffmpeg"], returncode=0,
                                       stdout=stdout, stderr=stderr)


class CropJpeg(unittest.TestCase):
    jpeg = b"\xff\xd8\xff\xe0in\xff\xd9"
    cropped = b"\xff\xd8\xff\xe0out\xff\xd9"

    def run_crop(self, **kwargs):
        """Run `crop_jpeg` with ffmpeg mocked; return (result, mock)."""
        with mock.patch.object(frames.subprocess, "run",
                               return_value=completed(self.cropped)) as run:
            result = frames.crop_jpeg(self.jpeg, TRAY_BOX, **kwargs)
        return result, run

    def test_the_filter_crops_the_box_fractions_then_scales(self) -> None:
        _result, run = self.run_crop()
        argv = run.call_args.args[0]
        self.assertEqual(
            argv[argv.index("-vf") + 1],
            "crop=iw*0.278:ih*0.241:iw*0.27:ih*0.325,scale=1024:-1",
        )

    def test_the_jpeg_is_piped_in_and_the_crop_is_piped_back(self) -> None:
        result, run = self.run_crop()
        self.assertEqual(result, self.cropped)
        argv = run.call_args.args[0]
        self.assertEqual(argv[0], "ffmpeg")
        self.assertEqual(argv[argv.index("-i") + 1], "pipe:0")
        self.assertEqual(argv[argv.index("-f") + 1], "mjpeg")
        self.assertEqual(argv[-1], "pipe:1")
        self.assertEqual(run.call_args.kwargs["input"], self.jpeg)

    def test_the_crop_failure_is_checked(self) -> None:
        _result, run = self.run_crop()
        self.assertTrue(run.call_args.kwargs["check"])

    def test_width_sets_the_scale(self) -> None:
        _result, run = self.run_crop(width=1600)
        argv = run.call_args.args[0]
        self.assertIn("scale=1600:-1", argv[argv.index("-vf") + 1])

    def test_a_non_zero_exit_raises_runtime_error_with_the_stderr(self) -> None:
        failed = subprocess.CalledProcessError(
            1, ["ffmpeg"], output=b"", stderr=b"Invalid crop width",
        )
        with mock.patch.object(frames.subprocess, "run", side_effect=failed):
            with self.assertRaises(RuntimeError) as caught:
                frames.crop_jpeg(self.jpeg, TRAY_BOX)
        self.assertIn("Invalid crop width", str(caught.exception))

    def test_a_missing_ffmpeg_raises_runtime_error(self) -> None:
        with mock.patch.object(frames.subprocess, "run",
                               side_effect=FileNotFoundError("ffmpeg")):
            with self.assertRaises(RuntimeError) as caught:
                frames.crop_jpeg(self.jpeg, TRAY_BOX)
        self.assertIn("ffmpeg", str(caught.exception))


class ReplayPrepare(unittest.TestCase):
    def test_a_clip_that_is_not_there_is_a_file_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = frames.ReplaySource(
                pathlib.Path(tmp) / "missing.mp4", pathlib.Path(tmp) / "cache"
            )
            with self.assertRaises(FileNotFoundError):
                source.prepare()

    def test_ffmpeg_producing_no_frames_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = pathlib.Path(tmp) / "clip.mp4"
            video.write_bytes(b"\x00")
            source = frames.ReplaySource(video, pathlib.Path(tmp) / "cache")
            with mock.patch.object(frames.subprocess, "run",
                                   return_value=completed()):
                with self.assertRaises(RuntimeError) as caught:
                    source.prepare()
        self.assertIn("no frames", str(caught.exception))


class ReplayFrames(unittest.TestCase):
    """A cache that is already extracted, so no ffmpeg is needed at all."""

    stills = [b"\xff\xd8one\xff\xd9", b"\xff\xd8two\xff\xd9"]

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = pathlib.Path(tmp.name)
        self.video = root / "clip.mp4"
        self.video.write_bytes(b"\x00")
        self.cache = root / "frames"
        self.cache.mkdir()
        for index, jpeg in enumerate(self.stills, start=1):
            (self.cache / f"f_{index:05d}.jpg").write_bytes(jpeg)
        self.source = frames.ReplaySource(self.video, self.cache, fps=2.0)

    def test_frame_reads_the_cached_still_and_its_time_from_the_fps(self) -> None:
        frame = self.source.frame(1)
        self.assertEqual(frame.index, 1)
        self.assertEqual(frame.t_s, 0.5)
        self.assertEqual(frame.jpeg, self.stills[1])
        self.assertEqual(frame.source, "replay:clip.mp4")

    def test_frame_out_of_range_raises_index_error(self) -> None:
        self.assertEqual(len(self.source), 2)
        for index in (2, -1):
            with self.assertRaises(IndexError):
                self.source.frame(index)

    def test_an_extracted_cache_is_reused_without_another_ffmpeg_run(self) -> None:
        with mock.patch.object(frames.subprocess, "run") as run:
            self.assertEqual(self.source.prepare(), 2)
        run.assert_not_called()


class DataUrl(unittest.TestCase):
    def test_a_jpeg_frame_serialises_to_a_data_url(self) -> None:
        frame = Frame(index=0, t_s=0.0, jpeg=b"\xff\xd8\xff\xd9", source="test")
        self.assertEqual(frame.data_url(), "data:image/jpeg;base64,/9j/2Q==")


if __name__ == "__main__":
    unittest.main()
