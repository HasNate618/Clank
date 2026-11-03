#!/usr/bin/env python3
"""Regenerate depth_stream_spv.h from the committed SPIR-V.

The Vulkan layer embeds the shader as a uint32_t array, so changing
depth_stream.comp needs two steps:

    glslangValidator -V depth_stream.comp -o depth_stream.comp.spv
    python3 gen_header.py
"""

from __future__ import annotations

import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPV = HERE / "depth_stream.comp.spv"
HEADER = HERE / "depth_stream_spv.h"


def main() -> None:
    data = SPV.read_bytes()
    if len(data) % 4:
        raise SystemExit(f"{SPV} is not a whole number of SPIR-V words")
    words = struct.unpack(f"<{len(data) // 4}I", data)
    lines = [
        "// Generated from depth_stream.comp.spv; do not edit by hand.",
        "#pragma once",
        "#include <stdint.h>",
        f"static const uint32_t depth_stream_spv[{len(words)}] = {{",
    ]
    for i in range(0, len(words), 4):
        lines.append("    " + ", ".join(f"0x{w:08x}" for w in words[i : i + 4]) + ",")
    lines.append("};")
    HEADER.write_text("\n".join(lines) + "\n")
    print(f"wrote {HEADER} ({len(words)} words)")


if __name__ == "__main__":
    main()
