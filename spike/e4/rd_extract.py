#!/usr/bin/env python3
"""Extract the game's scene depth (and the colour written beside it) from an .rdc.

Run as::

    qrenderdoc --python spike/e4/rd_extract.py

Inputs from the environment:

    RD_CAPTURE       host path to the .rdc
    RD_WORKDIR       directory for depth.raw, color.raw, depth.json, meta.json
    RD_STEREO_WIDTH  width of the stereo eye buffer (default 3840)

The capture is a DXVK title, so the depth attachment is a Vulkan D32S8 image.
There are usually several depth-like textures (shadow maps, a 1920-wide mirror,
and one or two 3840-wide stereo buffers). The one we want is the widest depth
texture that is bound as a depth target on the latest draw of the frame: the
game's final eye pass. The old SteamVR background texel at 0.991 is far above
the scene's reversed-Z values, so the post-processing step masks it out.

Only stdlib, os, json, and renderdoc are used here. Numpy lives in
depth_post.py, which runs outside qrenderdoc.
"""

from __future__ import annotations

import json
import os
import time
import traceback

import renderdoc as rd


def _descriptor_id(descriptor) -> str:
    return str(descriptor.resource)


def _find_scene_depth(control, stereo_width: int):
    """Return (texture, event_id, output_resource_id, last_event_by_texture).

    A DXVK trigger can over-capture dozens of frames. Walking every draw in
    every frame makes replay very slow, so scan backwards and stop at the first
    draw that binds a wide D32 depth target. In the last frame that is the final
    eye pass.
    """
    textures = {str(t.resourceId): t for t in control.GetTextures()}
    wide = {
        resource_id
        for resource_id, texture in textures.items()
        if "D32" in texture.format.Name()
        and (not stereo_width or texture.width == stereo_width)
    }
    any_depth = {
        resource_id
        for resource_id, texture in textures.items()
        if "D32" in texture.format.Name()
    }

    draws = [
        action
        for action in control.GetRootActions()
        if int(action.flags) & int(rd.ActionFlags.Drawcall)
    ]
    last: dict[str, int] = {}
    fallback = None
    for action in reversed(draws):
        control.SetFrameEvent(action.eventId, True)
        state = control.GetPipelineState()
        depth_id = _descriptor_id(state.GetDepthTarget())
        if depth_id == "ResourceId::0":
            continue
        last[depth_id] = action.eventId
        if depth_id in wide:
            outputs = [_descriptor_id(out) for out in state.GetOutputTargets()]
            output_id = next((item for item in outputs if item != "ResourceId::0"), None)
            return textures[depth_id], action.eventId, output_id, last
        if fallback is None and depth_id in any_depth:
            fallback = (textures[depth_id], action.eventId, depth_id)

    if fallback is not None:
        texture, event, resource_id = fallback
        control.SetFrameEvent(event, True)
        state = control.GetPipelineState()
        outputs = [_descriptor_id(out) for out in state.GetOutputTargets()]
        output_id = next((item for item in outputs if item != "ResourceId::0"), None)
        return texture, event, output_id, last

    raise RuntimeError("no D32 depth texture was bound by a draw call")


def main() -> int:
    capture_path = os.environ["RD_CAPTURE"]
    workdir = os.environ["RD_WORKDIR"]
    stereo_width = int(os.environ.get("RD_STEREO_WIDTH", "3840"))
    error_path = os.path.join(workdir, "error.txt")
    timings: dict[str, object] = {}
    started = time.perf_counter()

    def stamp(name: str) -> None:
        timings[name] = round(time.perf_counter() - started, 3)

    try:
        capture = rd.OpenCaptureFile()
        result = capture.OpenFile(capture_path, "", None)
        if not result.OK():
            raise RuntimeError(f"OpenFile: {result}")
        stamp("open_file")
        result, control = capture.OpenCapture(rd.ReplayOptions(), None)
        if control is None:
            raise RuntimeError(f"OpenCapture: {result}")
        stamp("open_capture")

        frame = control.GetFrameInfo()
        depth_texture, event, color_id, last = _find_scene_depth(control, stereo_width)
        stamp("find_depth")
        control.SetFrameEvent(event, True)
        state = control.GetPipelineState()

        subresource = rd.Subresource()
        subresource.mip = 0
        subresource.slice = 0
        subresource.sample = 0

        depth_bytes = control.GetTextureData(depth_texture.resourceId, subresource)
        stamp("read_depth")
        with open(os.path.join(workdir, "depth.raw"), "wb") as handle:
            handle.write(depth_bytes)

        depth_png = os.path.join(workdir, "depth.png")
        save = rd.TextureSave()
        save.resourceId = depth_texture.resourceId
        save.mip = 0
        save.destType = rd.FileType.PNG
        control.SaveTexture(save, depth_png)
        stamp("save_depth_png")

        color_meta = None
        if color_id:
            color_texture = next(
                (t for t in control.GetTextures() if str(t.resourceId) == color_id),
                None,
            )
            if color_texture is not None:
                color_bytes = control.GetTextureData(color_texture.resourceId, subresource)
                stamp("read_color")
                with open(os.path.join(workdir, "color.raw"), "wb") as handle:
                    handle.write(color_bytes)
                color_save = rd.TextureSave()
                color_save.resourceId = color_texture.resourceId
                color_save.mip = 0
                color_save.destType = rd.FileType.PNG
                control.SaveTexture(color_save, os.path.join(workdir, "color.png"))
                stamp("save_color_png")
                color_meta = {
                    "resourceId": color_id,
                    "width": color_texture.width,
                    "height": color_texture.height,
                    "format": color_texture.format.Name(),
                    "bytes": len(color_bytes),
                }

        meta = {
            "capture": capture_path,
            "frameNumber": int(frame.frameNumber),
            "event": int(event),
            "stereo_width": stereo_width,
            "depth": {
                "resourceId": str(depth_texture.resourceId),
                "width": depth_texture.width,
                "height": depth_texture.height,
                "format": depth_texture.format.Name(),
                "arraysize": int(depth_texture.arraysize),
                "msSamp": int(depth_texture.msSamp),
                "bytes": len(depth_bytes),
            },
            "color": color_meta,
            "draws": len(last),
        }
        with open(os.path.join(workdir, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
            handle.write("\n")
        stamp("write_meta")
        with open(os.path.join(workdir, "timings.json"), "w", encoding="utf-8") as handle:
            json.dump(timings, handle, indent=2)
            handle.write("\n")
        control.Shutdown()
        os._exit(0)
    except BaseException as exc:  # noqa: BLE001 - qrenderdoc hides stderr
        with open(error_path, "w", encoding="utf-8") as handle:
            handle.write(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}\n")
        with open(os.path.join(workdir, "timings.json"), "w", encoding="utf-8") as handle:
            json.dump(timings, handle, indent=2)
            handle.write("\n")
        os._exit(1)


if __name__ == "__main__":
    raise SystemExit(main())
