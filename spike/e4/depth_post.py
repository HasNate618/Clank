#!/usr/bin/env python3
"""Turn RenderDoc's raw depth/colour dumps into metric depth and normals.

This runs under the dev-shell Python (it needs numpy); rd_extract.py runs under
qrenderdoc and writes the raw files. The split keeps the qrenderdoc side tiny and
the numerical code testable without a GPU or a capture.

Depth convention: the game's Vulkan depth is D32S8 written by a reversed-Z
projection, so ``d = near / Z`` and ``Z = near / d``. ``near`` is measured from
the stereo pair itself: for a feature at disparity ``p`` pixels,
``Z = f_px * IPD / p``, and the same pixel's depth value gives
``near = Z * d``. The median over high-texture patches is the near plane. On the
Job Simulator capture this came out at 9.92 mm, which is a plausible VR near
clip and consequently gives metric distances of 0.28--8.6 m.

The depth is from the eye buffer, which is stored upside down; both depth and
colour are flipped together so the saved images are upright.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

DEFAULT_FOV_H_DEG = 140.0
DEFAULT_IPD_M = 0.063
DEFAULT_NEAR_M = 0.01
DEPTH_VISUAL_MAX_M = 10.0


def read_pgm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        assert handle.readline().strip() == b"P5"
        line = handle.readline()
        while line.startswith(b"#"):
            line = handle.readline()
        width, height = map(int, line.split())
        assert int(handle.readline()) == 65535
        data = np.frombuffer(handle.read(), dtype=">u2")
    return data.reshape(height, width).astype(np.float32) / 65535.0


def write_pgm(array: np.ndarray, path: Path) -> None:
    height, width = array.shape
    with path.open("wb") as handle:
        handle.write(f"P5\n{width} {height}\n65535\n".encode())
        handle.write((np.clip(array, 0.0, 1.0) * 65535).astype(">u2").tobytes())


def write_ppm(array: np.ndarray, path: Path) -> None:
    height, width, _ = array.shape
    with path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode())
        handle.write(np.clip(array, 0, 255).astype(np.uint8).tobytes())


def _png_tool() -> str | None:
    return shutil.which("magick") or shutil.which("convert")


def to_png(source: Path) -> Path | None:
    tool = _png_tool()
    if tool is None:
        return None
    target = source.with_suffix(".png")
    subprocess.run([tool, str(source), str(target)], check=True, capture_output=True)
    return target


def decode_depth(raw: bytes, width: int, height: int) -> np.ndarray:
    """Decode a tightly packed D32S8 image: float depth in the low four bytes."""
    expected = width * height * 8
    if len(raw) != expected:
        raise ValueError(f"depth dump is {len(raw)} bytes, expected {expected}")
    return (
        np.frombuffer(raw, dtype=np.uint8)
        .reshape(height, width, 8)[:, :, :4]
        .copy()
        .view("<f4")
        .reshape(height, width)
    )


def decode_color(raw: bytes, width: int, height: int, fmt: str) -> np.ndarray | None:
    if len(raw) != width * height * 4:
        return None
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4).copy()
    if fmt.startswith("B8G8R8A8"):
        pixels = pixels[..., [2, 1, 0, 3]]
    return pixels


def estimate_near(
    color: np.ndarray,
    depth: np.ndarray,
    f_px: float,
    ipd_m: float,
    *,
    seed: int = 0,
) -> tuple[float, list[dict]]:
    """Measure the near plane from stereo disparity. Returns (near_m, samples)."""
    height, width = depth.shape
    half = width // 2
    left_rgb = color[:, :half]
    right_rgb = color[:, half:]
    left_gray = left_rgb.astype(np.float32).mean(axis=2)
    right_gray = right_rgb.astype(np.float32).mean(axis=2)
    left_depth = depth[:, :half]

    rng = np.random.default_rng(seed)
    patch = 24
    max_disparity = 140
    candidates: list[tuple[int, int, float]] = []
    for _ in range(600):
        x = int(rng.integers(patch + max_disparity + 20, half - patch - 20))
        y = int(rng.integers(80, height - 80))
        value = float(left_depth[y, x])
        if not (0.0008 < value < 0.2):
            continue
        region = left_gray[y - patch : y + patch, x - patch : x + patch]
        variance = float(region.var())
        if variance < 80.0:
            continue
        candidates.append((x, y, variance))
    candidates.sort(key=lambda item: -item[2])

    samples: list[dict] = []
    near_values: list[float] = []
    for x, y, variance in candidates[:40]:
        reference = left_gray[y - patch : y + patch, x - patch : x + patch].astype(np.float32)
        errors: list[tuple[int, float]] = []
        for disparity in range(4, max_disparity):
            xr = x - disparity
            if xr - patch < 0:
                break
            target = right_gray[y - patch : y + patch, xr - patch : xr + patch].astype(np.float32)
            errors.append((disparity, float(np.abs(reference - target).mean())))
        if len(errors) < 2:
            continue
        errors.sort(key=lambda item: item[1])
        if errors[1][0] == errors[0][0]:
            continue
        if errors[0][1] > 0.7 * errors[1][1]:
            continue
        disparity = errors[0][0]
        z = f_px * ipd_m / disparity
        d = float(left_depth[y, x])
        near = z * d
        near_values.append(near)
        samples.append(
            {
                "x": x,
                "y": y,
                "disparity_px": disparity,
                "z_m": z,
                "depth": d,
                "near_m": near,
                "variance": variance,
                "sad": errors[0][1],
                "second_sad": errors[1][1],
            }
        )

    if len(near_values) < 3:
        return DEFAULT_NEAR_M, samples
    return float(np.median(near_values)), samples


def normals_for_eye(
    z: np.ndarray, valid: np.ndarray, f_px: float
) -> np.ndarray:
    height, width = z.shape
    u = np.arange(width, dtype=np.float32)
    v = np.arange(height, dtype=np.float32)
    grid_u, grid_v = np.meshgrid(u, v)
    x = (grid_u - width / 2.0) * z / f_px
    y = -(grid_v - height / 2.0) * z / f_px
    points = np.stack([x, y, z], axis=-1)

    d_du = np.zeros_like(points)
    d_dv = np.zeros_like(points)
    d_du[:, 1:-1] = (points[:, 2:] - points[:, :-2]) * 0.5
    d_dv[1:-1, :] = (points[2:, :] - points[:-2, :]) * 0.5
    normals = np.cross(d_du, d_dv)
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.divide(normals, length, out=np.zeros_like(normals), where=length > 1e-12)
    normals[normals[..., 2] < 0] *= -1
    normals[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return normals


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]
    with path.open("wb") as handle:
        handle.write(b"ply\nformat binary_little_endian 1.0\n")
        handle.write(f"element vertex {len(points)}\n".encode())
        handle.write(
            b"property float x\nproperty float y\nproperty float z\n"
            b"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        )
        handle.write(b"end_header\n")
        handle.write(vertices.tobytes())


def postprocess(
    workdir: Path,
    outdir: Path,
    *,
    near_override: float | None = None,
    fov_h_deg: float = DEFAULT_FOV_H_DEG,
    ipd_m: float = DEFAULT_IPD_M,
    point_step: int = 3,
) -> dict:
    meta = json.loads((workdir / "meta.json").read_text())
    depth_meta = meta["depth"]
    width = int(depth_meta["width"])
    height = int(depth_meta["height"])
    raw_depth = (workdir / "depth.raw").read_bytes()
    depth = decode_depth(raw_depth, width, height)
    depth = depth[::-1, :]

    color = None
    color_meta = meta.get("color")
    color_path = workdir / "color.raw"
    if color_meta and color_path.exists():
        color = decode_color(
            color_path.read_bytes(), width, height, str(color_meta.get("format", ""))
        )
        if color is not None:
            color = color[::-1, :]
            write_ppm(color[..., :3].astype(np.float32), outdir / "capture-color-stereo.ppm")

    per_eye_width = width // 2
    f_px = (per_eye_width / 2.0) / math.tan(math.radians(fov_h_deg) / 2.0)

    if near_override is not None:
        near_m = float(near_override)
        near_samples: list[dict] = []
        near_source = "override"
    elif color is not None:
        near_m, near_samples = estimate_near(color[..., :3], depth, f_px, ipd_m)
        near_source = "stereo disparity"
    else:
        near_m, near_samples = DEFAULT_NEAR_M, []
        near_source = "default (no colour dump)"

    valid = (depth > 1e-6) & (depth < 0.5)
    z = np.full_like(depth, np.nan, dtype=np.float32)
    z[valid] = near_m / depth[valid]

    visual = np.where(valid, np.clip(z, 0.0, DEPTH_VISUAL_MAX_M) / DEPTH_VISUAL_MAX_M, 1.0)
    write_pgm(visual, outdir / "depth-metric-stereo.pgm")
    write_pgm(visual[:, :per_eye_width], outdir / "depth-metric-left.pgm")
    write_pgm(visual[:, per_eye_width:], outdir / "depth-metric-right.pgm")

    normal_images = []
    depth_arrays = {}
    normal_arrays = {}
    for side, x0 in (("left", 0), ("right", per_eye_width)):
        z_eye = z[:, x0 : x0 + per_eye_width]
        valid_eye = valid[:, x0 : x0 + per_eye_width]
        normals = normals_for_eye(z_eye, valid_eye, f_px)
        depth_arrays[side] = z_eye
        normal_arrays[side] = normals
        image = (normals * 0.5 + 0.5) * 255.0
        write_ppm(image, outdir / f"normal-{side}.ppm")
        normal_images.append(outdir / f"normal-{side}.ppm")
        np.save(outdir / f"depth-metric-{side}.npy", z_eye)
        np.save(outdir / f"normal-{side}.npy", normals)

    if color is not None:
        xs = np.arange(0, per_eye_width, point_step)
        ys = np.arange(0, height, point_step)
        grid_u, grid_v = np.meshgrid(xs, ys)
        z_points = z[::point_step, 0:per_eye_width:point_step]
        color_points = color[::point_step, 0:per_eye_width:point_step, :3]
        ok = np.isfinite(z_points) & (z_points > 0.05) & (z_points < 30.0)
        x_points = (grid_u - per_eye_width / 2.0) * z_points / f_px
        y_points = -(grid_v - height / 2.0) * z_points / f_px
        points = np.stack([x_points[ok], y_points[ok], z_points[ok]], axis=-1)
        write_ply(outdir / "pointcloud-left.ply", points, color_points[ok])
    else:
        points = np.empty((0, 3), dtype=np.float32)

    # ImageMagick is in the dev shell; if it is missing the .pgm/.ppm files are
    # still complete and the PNG names simply do not appear.
    for source in [
        outdir / "depth-metric-stereo.pgm",
        outdir / "depth-metric-left.pgm",
        outdir / "depth-metric-right.pgm",
        *normal_images,
        outdir / "capture-color-stereo.ppm",
    ]:
        if source.exists():
            to_png(source)
    if all((outdir / f"normal-{side}.png").exists() for side in ("left", "right")):
        tool = _png_tool()
        if tool:
            subprocess.run(
                [
                    tool,
                    str(outdir / "normal-left.png"),
                    str(outdir / "normal-right.png"),
                    "+append",
                    str(outdir / "normal-stereo.png"),
                ],
                check=True,
                capture_output=True,
            )

    finite = z[np.isfinite(z)]
    calibration = {
        "capture": meta.get("capture"),
        "frameNumber": meta.get("frameNumber"),
        "event": meta.get("event"),
        "depth": depth_meta,
        "color": color_meta,
        "near_m": near_m,
        "near_source": near_source,
        "near_samples": near_samples,
        "f_px": f_px,
        "fov_h_deg": fov_h_deg,
        "ipd_m": ipd_m,
        "depth_convention": "reversed-Z, Z = near / depth, far = infinity",
        "valid_fraction": float(valid.mean()),
        "metric_min_m": float(finite.min()) if finite.size else None,
        "metric_max_m": float(finite.max()) if finite.size else None,
        "metric_median_m": float(np.median(finite)) if finite.size else None,
        "point_count": int(len(points)),
        "flip_y": True,
        "artifacts": sorted(path.name for path in outdir.iterdir() if path.is_file()),
    }
    (outdir / "depth-calibration.json").write_text(
        json.dumps(calibration, indent=2) + "\n"
    )
    return calibration
