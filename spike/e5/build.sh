#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
inc="${VULKAN_INCLUDE:-}"
if [[ -z "$inc" ]]; then
  for d in /nix/store/*vulkan-headers*/include "$(pkg-config --variable=includedir vulkan 2>/dev/null || true)"; do
    if [[ -f "$d/vulkan/vulkan.h" ]]; then inc="$d"; break; fi
  done
fi
if [[ -z "$inc" ]]; then
  echo "vulkan headers not found; set VULKAN_INCLUDE" >&2
  exit 1
fi
mkdir -p build
cc -shared -fPIC -O2 -std=c11 -Wall -Wextra -I"$inc" \
  -o build/depth_stream_layer.so depth_stream_layer.c -lpthread
echo "built build/depth_stream_layer.so"
