from __future__ import annotations

import struct
import tempfile
import time
import unittest
from pathlib import Path

from clank import depth_stream


def write_stream(
    path: Path,
    *,
    width: int = 3,
    height: int = 2,
    depth=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    color: bytes = b"",
    seq: int = 2,
    publish_ns: int | None = None,
) -> None:
    color_w = width if color else 0
    color_h = height if color else 0
    publish_ns = time.monotonic_ns() if publish_ns is None else publish_ns
    header = depth_stream.HEADER.pack(
        depth_stream.MAGIC,
        1,
        width,
        height,
        0,
        0,
        color_w,
        color_h,
        0,
        0,
        seq,
        0,
        7,
        publish_ns - 5_000_000,
        publish_ns,
        0.01,
        0.0,
        *([0] * 16),
    )
    path.write_bytes(
        header + struct.pack("<%df" % len(depth), *depth) + color
    )


class ReadStream(unittest.TestCase):
    def test_reads_depth_and_colour(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream.bin"
            color = bytes(range(3 * 2 * 4))
            write_stream(path, color=color)
            with depth_stream.Stream(path) as stream:
                frame = stream.read(timeout=0.5)
            self.assertIsNotNone(frame)
            assert frame is not None
            self.assertEqual(frame["frame_id"], 7)
            self.assertEqual(list(frame["depth"]), [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
            self.assertEqual(frame["color"], color)

    def test_latest_ignores_a_stale_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream.bin"
            write_stream(path, publish_ns=time.monotonic_ns() - 10_000_000_000)
            self.assertIsNone(depth_stream.latest(max_age=2.0, path=path))

    def test_latest_accepts_a_fresh_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream.bin"
            write_stream(path)
            self.assertIsNotNone(depth_stream.latest(max_age=2.0, path=path))


class Encode(unittest.TestCase):
    def test_bmp_is_bottom_up_bgr(self) -> None:
        rgba = bytes([255, 0, 0, 255, 0, 0, 255, 255])  # red, blue
        data = depth_stream.bmp(rgba, 2, 1)
        self.assertEqual(data[:2], b"BM")
        width, height = struct.unpack("<ii", data[18:26])
        self.assertEqual((width, height), (2, 1))
        self.assertEqual(data[54:60], bytes([0, 0, 255, 255, 0, 0]))

    def test_npy_header_names_shape(self) -> None:
        values = depth_stream.array.array("f", [1.0, 2.0, 3.0, 4.0])
        data = depth_stream._npy_f32(values, 2, 2)
        self.assertTrue(data.startswith(b"\x93NUMPY"))
        header_len = struct.unpack("<H", data[8:10])[0]
        header = data[10 : 10 + header_len].decode("ascii")
        self.assertIn("'shape': (2, 2)", header)
        self.assertIn("'descr': '<f4'", header)
