#!/usr/bin/env python3
"""Grab the Job Simulator window straight out of Xwayland, with no VNC in the path.

The game is an X11 client on the `-rootless` Xwayland display the session's Weston
started (`:1` in session `20261101T223603Z-27abb1`). Xwayland keeps the window's
pixels, so `XGetImage` on the window returns the current view **upright and at the
game's own resolution** - unlike `weston-screenshooter`, whose VNC-backend frames are
vertically mirrored and only arrive while an authenticated VNC client is attached.

    spike/e3/.venv/bin/python spike/wbc_game/x11_grab.py \
        --display :1 --out /tmp/game.ppm --check

Measured 2026-11-01: window `0x4a00001 "Job Simulator"` at 1920x1080, 32 bpp BGRX,
~1.4 s end to end (most of it the pure-Python pixel decode), and a mean absolute
difference of 1.45 levels against a flipped `weston-screenshooter` frame of the same
scene - i.e. the same pixels, but already the right way up.

This stays a spike on purpose. Two things are needed before it belongs in `clank/`:
an explicit `libx11` path in config (NixOS does not put it on the default loader
path, so this script globs the store), and an Xlib error handler plus `XSync` so a
`BadMatch` on an unmapped window raises instead of calling `exit(1)` in a long-lived
process. The decode and PPM writing are the easy part.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import time
from pathlib import Path

# Xlib constants from X.h / Xlib.h.
Z_PIXMAP = 2
ALL_PLANES = ctypes.c_ulong(-1).value
LSB_FIRST = 0


class XImage(ctypes.Structure):
    # Layout of Xlib.h's `struct _XImage`, in order. Only the leading numeric
    # fields and `data` are read; `funcs` is opaque but must occupy its space.
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("xoffset", ctypes.c_int),
        ("format", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_ubyte)),
        ("byte_order", ctypes.c_int),
        ("bitmap_unit", ctypes.c_int),
        ("bitmap_bit_order", ctypes.c_int),
        ("bitmap_pad", ctypes.c_int),
        ("depth", ctypes.c_int),
        ("bytes_per_line", ctypes.c_int),
        ("bits_per_pixel", ctypes.c_int),
        ("red_mask", ctypes.c_ulong),
        ("green_mask", ctypes.c_ulong),
        ("blue_mask", ctypes.c_ulong),
        ("obdata", ctypes.c_void_p),
        ("funcs", ctypes.c_void_p * 6),
    ]


def find_libx11(explicit: str | None) -> str:
    """Locate libX11. NixOS does not put it on the default loader path."""
    if explicit:
        return explicit
    for pattern in ("/nix/store/*-libx11-*/lib/libX11.so.6", "/usr/lib/libX11.so.6", "/usr/lib/x86_64-linux-gnu/libX11.so.6"):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    raise SystemExit("libX11.so.6 not found; pass --libx11 /path/to/libX11.so.6")


class X11:
    def __init__(self, path: str) -> None:
        self.lib = ctypes.CDLL(path)

        self.lib.XOpenDisplay.restype = ctypes.c_void_p
        self.lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self.lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
        self.lib.XDefaultRootWindow.restype = ctypes.c_ulong
        self.lib.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        self.lib.XQueryTree.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
            ctypes.POINTER(ctypes.c_uint),
        ]
        self.lib.XFetchName.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.XGetGeometry.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint),
        ]
        self.lib.XGetImage.restype = ctypes.POINTER(XImage)
        self.lib.XGetImage.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_ulong,
            ctypes.c_int,
        ]
        self.lib.XDestroyImage.argtypes = [ctypes.POINTER(XImage)]
        self.lib.XFree.argtypes = [ctypes.c_void_p]
        self.lib.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.XSetErrorHandler.argtypes = [ctypes.c_void_p]
        self.lib.XSetErrorHandler.restype = ctypes.c_void_p

        # Xlib's default handler prints and calls exit(1). For a measurement tool
        # that is the wrong failure mode, so errors are counted instead. This is a
        # process-global hook, which is one more reason this lives in a spike.
        self.errors = 0
        callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)

        def on_error(_display, _event):
            self.errors += 1
            return 0

        self._on_error = callback_type(on_error)
        self.lib.XSetErrorHandler(self._on_error)

        self.display = self.lib.XOpenDisplay(None)
        if not self.display:
            raise SystemExit("cannot open the X display (is DISPLAY set?)")

    def close(self) -> None:
        if self.display:
            self.lib.XCloseDisplay(self.display)
            self.display = None

    def names(self, window: int) -> str | None:
        name = ctypes.c_char_p()
        if not self.lib.XFetchName(self.display, window, ctypes.byref(name)):
            return None
        if not name.value:
            return None
        try:
            return name.value.decode("utf-8", "replace")
        finally:
            self.lib.XFree(name)

    def find(self, title: str, window: int | None = None, depth: int = 0) -> int | None:
        current = self.lib.XDefaultRootWindow(self.display) if window is None else window
        if depth:
            name = self.names(current)
            if name and title.lower() in name.lower():
                root = ctypes.c_ulong()
                x = ctypes.c_int()
                y = ctypes.c_int()
                width = ctypes.c_uint()
                height = ctypes.c_uint()
                border = ctypes.c_uint()
                bits = ctypes.c_uint()
                if self.lib.XGetGeometry(
                    self.display,
                    current,
                    ctypes.byref(root),
                    ctypes.byref(x),
                    ctypes.byref(y),
                    ctypes.byref(width),
                    ctypes.byref(height),
                    ctypes.byref(border),
                    ctypes.byref(bits),
                ) and width.value > 100 and height.value > 100:
                    return current
        root_return = ctypes.c_ulong()
        parent_return = ctypes.c_ulong()
        children = ctypes.POINTER(ctypes.c_ulong)()
        count = ctypes.c_uint()
        if not self.lib.XQueryTree(
            self.display,
            current,
            ctypes.byref(root_return),
            ctypes.byref(parent_return),
            ctypes.byref(children),
            ctypes.byref(count),
        ):
            return None
        found = None
        try:
            for index in range(count.value):
                found = self.find(title, children[index], depth + 1)
                if found is not None:
                    break
        finally:
            if children:
                self.lib.XFree(children)
        return found

    def geometry(self, window: int) -> tuple[int, int]:
        root = ctypes.c_ulong()
        x = ctypes.c_int()
        y = ctypes.c_int()
        width = ctypes.c_uint()
        height = ctypes.c_uint()
        border = ctypes.c_uint()
        bits = ctypes.c_uint()
        if not self.lib.XGetGeometry(
            self.display,
            window,
            ctypes.byref(root),
            ctypes.byref(x),
            ctypes.byref(y),
            ctypes.byref(width),
            ctypes.byref(height),
            ctypes.byref(border),
            ctypes.byref(bits),
        ):
            raise SystemExit(f"cannot read geometry for window {window:#x}")
        return width.value, height.value

    def grab_rgb(self, window: int) -> tuple[int, int, bytearray]:
        width, height = self.geometry(window)
        image = self.lib.XGetImage(
            self.display, window, 0, 0, width, height, ALL_PLANES, Z_PIXMAP
        )
        self.lib.XSync(self.display, 0)
        if not image:
            raise SystemExit(
                f"XGetImage failed for window {window:#x} "
                f"(X errors: {self.errors}; is the window mapped?)"
            )
        try:
            header = image.contents
            if header.bits_per_pixel not in (24, 32):
                raise SystemExit(f"unsupported XImage depth: {header.bits_per_pixel} bpp")
            stride = header.bytes_per_line
            raw = ctypes.string_at(header.data, stride * header.height)
            rgb = decode_ximage(
                raw,
                header.width,
                header.height,
                stride,
                header.bits_per_pixel // 8,
                header.byte_order,
                header.red_mask,
                header.green_mask,
                header.blue_mask,
            )
            return header.width, header.height, rgb
        finally:
            self.lib.XDestroyImage(image)


def channel(pixel: int, mask: int) -> int:
    if not mask:
        return 0
    shift = (mask & -mask).bit_length() - 1
    maximum = mask >> shift
    return ((pixel & mask) >> shift) * 255 // maximum


def decode_ximage(
    raw: bytes,
    width: int,
    height: int,
    stride: int,
    bytes_per_pixel: int,
    byte_order: int,
    red_mask: int,
    green_mask: int,
    blue_mask: int,
) -> bytearray:
    """Turn an XImage's packed pixels into RGB bytes.

    Masks and byte order come from the server, not from an assumption: Xwayland
    reports 32 bpp, LSBFirst, `0xff0000/0xff00/0xff` here, but a 16 bpp visual would
    be scaled rather than misread.
    """
    order = "little" if byte_order == LSB_FIRST else "big"
    out = bytearray(width * height * 3)
    cursor = 0
    for y in range(height):
        row = y * stride
        for x in range(width):
            offset = row + x * bytes_per_pixel
            pixel = int.from_bytes(raw[offset : offset + bytes_per_pixel], order)
            out[cursor] = channel(pixel, red_mask)
            out[cursor + 1] = channel(pixel, green_mask)
            out[cursor + 2] = channel(pixel, blue_mask)
            cursor += 3
    return out


def write_ppm(path: Path, width: int, height: int, rgb: bytearray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"P6\n%d %d\n255\n" % (width, height) + rgb)


def analyse(path: Path) -> dict:
    """Numbers that separate a rendered scene from a blank window."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        pixels = np.asarray(rgb, dtype=np.uint8)
    flat = pixels.reshape(-1, 3)
    luma = flat.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    grid = luma.reshape(height, width)
    return {
        "width": width,
        "height": height,
        "bytes": path.stat().st_size,
        "mean_rgb": [round(float(value), 1) for value in flat.mean(axis=0)],
        "luma_std": round(float(luma.std()), 1),
        "distinct_colours": int(len(np.unique(flat, axis=0))),
        "corr_horizontal": round(float(np.corrcoef(grid[:, :-1].ravel(), grid[:, 1:].ravel())[0, 1]), 4),
        "corr_vertical": round(float(np.corrcoef(grid[:-1, :].ravel(), grid[1:, :].ravel())[0, 1]), 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grab the game window from Xwayland.")
    parser.add_argument("--display", default=":1")
    parser.add_argument("--title", default="Job Simulator")
    parser.add_argument("--libx11")
    parser.add_argument("--out", required=True, help=".ppm (always written) or .png (needs Pillow)")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--min-std", type=float, default=10.0)
    parser.add_argument("--min-colours", type=int, default=1000)
    args = parser.parse_args(argv)

    # XOpenDisplay reads DISPLAY, so set it before the library call.
    import os

    os.environ["DISPLAY"] = args.display

    started = time.monotonic()
    x11 = X11(find_libx11(args.libx11))
    try:
        window = x11.find(args.title)
        if window is None:
            raise SystemExit(f"no window matching {args.title!r} on {args.display}")
        title = x11.names(window)
        width, height, rgb = x11.grab_rgb(window)
    finally:
        x11.close()
    elapsed = time.monotonic() - started

    destination = Path(args.out)
    ppm = destination if destination.suffix.lower() == ".ppm" else destination.with_suffix(".ppm")
    write_ppm(ppm, width, height, rgb)
    if destination.suffix.lower() == ".png":
        from PIL import Image

        Image.frombytes("RGB", (width, height), bytes(rgb)).save(destination)

    report = {
        "display": args.display,
        "window": hex(window),
        "title": title,
        "width": width,
        "height": height,
        "path": str(destination.resolve()),
        "ppm": str(ppm.resolve()),
        "seconds": round(elapsed, 3),
    }
    if destination.suffix.lower() == ".png":
        report.update({"bytes": destination.stat().st_size, **analyse(destination)})
        report["rendered"] = (
            report["luma_std"] >= args.min_std
            and report["distinct_colours"] >= args.min_colours
        )
        report["verdict"] = "rendered scene" if report["rendered"] else "blank window"
    print(json.dumps(report, indent=2))
    if args.check and report.get("rendered") is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
