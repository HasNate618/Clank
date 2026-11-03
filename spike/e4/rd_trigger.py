#!/usr/bin/env python3
"""Trigger one RenderDoc capture on a running process, from inside qrenderdoc.

Run as::

    qrenderdoc --python spike/e4/rd_trigger.py

Inputs come from the environment so the same file works for every session:

    RD_PORT          target-control port (one is opened per Wine process)
    RD_TRIGGER_OUT   JSON file to write {path, captureId, pid, target, ...}

RenderDoc's Python module only exists inside qrenderdoc's interpreter, which is
why this is a separate process rather than an import in depth_capture.py.

The layer writes captures into the *game's* mount namespace, so the path in the
JSON is a path on the target side, not a host path. depth_capture.py copies it
through /proc/<pid>/root.
"""

from __future__ import annotations

import json
import os
import time
import traceback

import renderdoc as rd


def _write(path: str, document: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")


def main() -> int:
    out = os.environ["RD_TRIGGER_OUT"]
    port = int(os.environ["RD_PORT"])
    timeout = float(os.environ.get("RD_TIMEOUT_SEC", "120"))

    try:
        # forceConnection=True can stop the target on this DXVK layer. False is
        # enough for a local connection and keeps the game alive.
        control = rd.CreateTargetControl(
            f"localhost:{port}", port, "clank-depth", False
        )
        if control is None or not control.Connected():
            raise RuntimeError(f"no target control connection on port {port}")

        # Drop any NewCapture left over from an earlier trigger. A stale file is
        # still a complete capture, but it is not the frame the caller asked for.
        for _ in range(20):
            message = control.ReceiveMessage(50)
            if message.type == rd.TargetControlMessageType.Noop:
                break

        # Trigger and return. The NewCapture message is not reliable on this
        # DXVK title: a trigger can write several files, and the message can be
        # missed. depth_capture.py watches the target's /tmp/RenderDoc for the
        # newest completed file instead.
        control.TriggerCapture(1)
        _write(
            out,
            {
                "triggered": True,
                "pid": int(control.GetPID()),
                "target": control.GetTarget(),
                "port": port,
            },
        )
        time.sleep(1.0)
        control.Shutdown()
        os._exit(0)
    except BaseException as exc:  # noqa: BLE001 - qrenderdoc hides stderr
        _write(out, {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        os._exit(1)


if __name__ == "__main__":
    raise SystemExit(main())
