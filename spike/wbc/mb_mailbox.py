#!/usr/bin/env python3
"""The session mailbox: small json files the 50 Hz body process and the adapter share.

Two processes talk here and neither may block the other: the body process ticks MuJoCo
and SONIC every 20 ms and cannot wait for an adapter that occasionally spends
milliseconds generating motion. So there is no socket and no queue - each direction is
one file, replaced atomically, latest wins.

    pose36.json    body process -> adapter   the last RING achieved 36-float poses,
                                             oldest first, used as generator context
    mb_play.json   adapter -> body process   "load and play clip seq N"
    mb_status.json body process -> adapter   acknowledgement and progress
    mb_clip.npz    adapter -> body process   the resampled clip; the mailbox only ever
                                             stores its path, never writes it

The play payload is deliberately only `seq` and a path: this channel carries joint-space
motion, not world coordinates, object positions or clip bytes. Poses over the xr channel
stay on `control.sock`.

Every read treats a missing, empty, torn or wrongly shaped file as `None` instead of
raising, because "the other side has not written yet" is the normal state at startup.

Standard library only: `tests/` imports this module and must stay stdlib.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

POSE36_NAME = "pose36.json"
CLIP_NAME = "mb_clip.npz"
PLAY_NAME = "mb_play.json"
STATUS_NAME = "mb_status.json"

#: Everything one body process leaves behind for the next one. The directory is not
#: per-session - `clank/dashboard.py` passes `session_root/body`, which is persistent and
#: never removed - so a finished run's files are still there when the next body starts.
STALE_NAMES = (PLAY_NAME, CLIP_NAME, STATUS_NAME, POSE36_NAME)

#: Poses of context handed to the generator: 32 ticks, 0.64 s at BODY_HZ.
RING = 32
#: Floats per pose: root xyz + root quat xyzw + 29 hinge angles (`g1_pack.pose36_from_qpos`).
POSE_LEN = 36
#: The body process's tick rate, recorded in `pose36.json` so the adapter can resample.
BODY_HZ = 50
#: The only statuses the adapter understands.
STATUSES = ("playing", "done", "failed")


def write_atomic(path: Path, payload: bytes) -> None:
    """Replace `path` with `payload`, so a reader never sees a half-written file.

    The temp name carries the pid for the same reason `body_view.write_atomic`'s does:
    two writers sharing one `.tmp` path meant each one's `replace()` could find the
    other's tmp already renamed away, and the slow one raised FileNotFoundError.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _read_json(path: Path) -> dict | None:
    """Parse a mailbox file, or `None` if it is missing, empty, torn or not an object."""
    try:
        data = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _is_pose_row(row: object) -> bool:
    if not isinstance(row, list) or len(row) != POSE_LEN:
        return False
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in row)


def clear_stale(session_dir: Path) -> list[str]:
    """Remove the mailbox files a previous run left in `session_dir`; return any that stay.

    The body process owns the mailbox for as long as it runs, so at construction every file
    in it belongs to someone else. A leftover `mb_play.json` is the dangerous one: a new
    process has loaded no seq yet, so the request looks fresh and its clip plays on the
    first tick with no adapter running. `mb_status.json` goes for the same reason - a stale
    `done`/`seq_ack` reads as this process's own first acknowledgement.

    A missing file, or a missing directory, is the normal state and stays silent. A file
    that cannot be removed is returned instead of raised: the body has to start either way,
    and a caller can only log it.
    """
    left = []
    for name in STALE_NAMES:
        try:
            (session_dir / name).unlink(missing_ok=True)
        except OSError:
            left.append(name)
    return left


def write_pose36(
    session_dir: Path,
    pose: Sequence[float],
    seq: int,
    t_mono_ns: int,
    prev: list | None = None,
) -> list:
    """Append one achieved pose to the context ring and return the new ring.

    `prev` is the caller's in-memory ring; `None` means read the file instead. The body
    process passes its own list because it already holds it, and a 50 Hz read is a read
    spent for nothing.
    """
    row = [float(v) for v in pose]
    if len(row) != POSE_LEN:
        raise ValueError(f"pose36 needs {POSE_LEN} floats, got {len(row)}")
    if prev is not None:
        ring = list(prev)
    else:
        previous = read_pose36(session_dir)
        ring = list(previous["poses"]) if previous is not None else []
    ring.append(row)
    ring = ring[-RING:]
    payload = {"seq": int(seq), "t_mono_ns": int(t_mono_ns), "hz": BODY_HZ, "poses": ring}
    write_atomic(session_dir / POSE36_NAME, json.dumps(payload).encode("utf-8"))
    return ring


def read_pose36(session_dir: Path) -> dict | None:
    """The context ring as written, `{"seq", "t_mono_ns", "hz", "poses"}`, or `None`."""
    data = _read_json(session_dir / POSE36_NAME)
    if data is None:
        return None
    poses = data.get("poses")
    if not isinstance(poses, list) or not poses:
        return None
    if not all(_is_pose_row(row) for row in poses):
        return None
    if not isinstance(data.get("seq"), int) or not isinstance(data.get("t_mono_ns"), int):
        return None
    return data


def write_play(session_dir: Path, seq: int, path: str = CLIP_NAME) -> None:
    """Ask the body process to load and play `path` (relative to `session_dir`) now.

    Latest wins: a request the way process has not picked up yet is overwritten by the
    next one rather than queued.
    """
    write_atomic(
        session_dir / PLAY_NAME,
        json.dumps({"seq": int(seq), "path": str(path)}).encode("utf-8"),
    )


def read_play(session_dir: Path) -> dict | None:
    """The pending request, `{"seq", "path"}`, or `None` when there is nothing to play."""
    data = _read_json(session_dir / PLAY_NAME)
    if data is None:
        return None
    if not isinstance(data.get("seq"), int) or not isinstance(data.get("path"), str):
        return None
    return data


def write_status(session_dir: Path, seq_ack: int, status: str, frame: int, n: int) -> None:
    """Acknowledge request `seq_ack`: `playing`, `done`, or `failed`."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    payload = {"seq_ack": int(seq_ack), "status": status, "frame": int(frame), "n": int(n)}
    write_atomic(session_dir / STATUS_NAME, json.dumps(payload).encode("utf-8"))


def read_status(session_dir: Path) -> dict | None:
    """The body process's progress, `{"seq_ack", "status", "frame", "n"}`, or `None`."""
    data = _read_json(session_dir / STATUS_NAME)
    if data is None:
        return None
    if data.get("status") not in STATUSES:
        return None
    keys = ("seq_ack", "frame", "n")
    if not all(isinstance(data.get(key), int) for key in keys):
        return None
    return data
