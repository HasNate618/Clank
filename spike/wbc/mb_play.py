#!/usr/bin/env python3
"""The executor's half of a deposited clip: load it once, report progress, hold the end.

`sonic_executor.py` ticks MuJoCo and ONNX at 50 Hz and cannot be imported by `tests/`; the
three decisions a deposited clip needs are pure, so they live here instead:

* `on_play_seq_change` - a request is new when its `seq` differs from the one already
  loaded. The mailbox is a latest-wins file, so without the seq a clip would be re-read and
  restarted on every tick.
* `after_clip` - the status to report for the frame just played.
* `lookahead_rows` - the ten clip rows the mode-0 encoder reads from a frame, **every one of
  them clamped to the clip's last row**. This is where a generated clip is held, and it is
  deliberately not a rewrite of the clip: the rows are played in full, so the body follows
  the motion to its end, and only the rows *beyond* the end are the final pose. As playback
  approaches the end the far rows saturate, and on the last frame the window is constant -
  the hold that cannot walk (`research/c3m-results.md`, H1f: freezing the whole horizon held
  a walking clip in place at 0.0085 m/s, against 0.2005 m/s with the anchor alone).

`sonic_runner.motion_frames` applies the same clamp internally when the executor hands it a
clip whose `n` is the true row count, so `lookahead_rows` is the statement of the contract
the observation depends on, not a second policy.

Standard library only: `tests/` imports this module and must stay stdlib.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The sibling modules are imported as top-level names from their own directory, the same
# way every other spike under spike/wbc/ reaches them.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import clip_from_gen  # noqa: E402
import mb_mailbox  # noqa: E402

#: The two statuses this module can report. `mb_mailbox.STATUSES` also allows `failed`,
#: which the executor writes itself when a deposited clip cannot be read.
PLAYING = "playing"
DONE = "done"


def on_play_seq_change(old_seq: int | None, play: dict | None) -> bool:
    """Whether `play` is a request the executor has not loaded yet.

    `play` is what `mb_mailbox.read_play` returns: `{"seq", "path"}`, or `None` when
    nothing was deposited. Only the seq decides, so the same request is not reloaded and an
    older one cannot replay a clip that was replaced.
    """
    if play is None:
        return False
    return int(play["seq"]) != old_seq


def after_clip(frame: int, n: int) -> str:
    """The status for a clip at `frame` of `n`: `playing`, or `done` on the last row.

    A frame past the end stays `done` rather than raising, because the executor may still
    be asked for an observation on the tick after it stopped advancing.
    """
    return DONE if frame >= n - 1 else PLAYING


class Playback:
    """Which deposited clip is playing, on which row, and which request was answered.

    Two seqs, and they are not the same one. `requested_seq` is the last request taken out
    of the mailbox - the mailbox is latest-wins, so it is what stops a payload that cannot
    be loaded being re-attempted every 20 ms, and it is the seq a `failed` status names.
    `clip_seq` is the request the clip *currently playing* came from, and `playing`/`done`
    are only ever reported against it: a reload that fails while an older clip still runs
    must not make that clip acknowledge a request that never ran.
    """

    def __init__(self) -> None:
        self.requested_seq: int | None = None
        self.clip_seq: int | None = None
        self.frame = 0
        self.n = 0

    def pending(self, play: dict | None) -> dict | None:
        """`play` if it is a request this instance has not taken yet, else `None`."""
        return play if on_play_seq_change(self.requested_seq, play) else None

    def rejected(self, seq: int) -> None:
        """Record request `seq` as taken and unplayable. Any playing clip is untouched."""
        self.requested_seq = int(seq)

    def started(self, seq: int, n: int) -> None:
        """Record request `seq` as the clip now playing, from its first row."""
        self.requested_seq = int(seq)
        self.clip_seq = int(seq)
        self.frame = 0
        self.n = int(n)

    def advance(self) -> str:
        """Report the status of the row just played, then move to the next one.

        The status is decided **before** the index moves, so the last row is played by the
        tick that reports `done` and only that tick leaves mode 0. Deciding after the move
        skipped it: frames `0..n-2` played and mode 1 took the body back before `n-1` ever
        reached an observation. A one-row clip still plays its one row, then finishes.
        Frame stays on the last row once `done`, so a further tick cannot move it.
        """
        status = after_clip(self.frame, self.n)
        if status != DONE:
            self.frame, _ = clip_from_gen.clip_tick(self.frame, self.n)
        return status


def write_progress(session_dir: Path, seq: int, status: str, frame: int, n: int) -> str | None:
    """Report progress for `seq`; return why it failed, or `None` when it landed.

    This runs inside the 50 Hz body tick, which owns the physics: a mailbox is never worth
    the loop. A session directory rotated or removed under a running body makes the atomic
    replace raise `OSError`, and an unknown status raises `ValueError`; both come back as a
    string instead. The caller warns once, because a directory that has gone away does not
    come back: this converts a crash into silence, and the adapter is expected to notice
    through the age of its clip rather than through a `failed` status it will never get.
    """
    try:
        mb_mailbox.write_status(session_dir, seq, status, frame, n)
    except (OSError, ValueError) as exc:
        return str(exc) or type(exc).__name__
    return None


def lookahead_rows(frame: int, n: int) -> tuple[int, ...]:
    """The ten clip rows the mode-0 encoder reads from `frame`, clamped to `n - 1`.

    Clamping is per row, not only at the tail: at `frame = 0` of a clip longer than the
    window nothing is clamped, and on the last frame all ten rows are that frame. Rows are
    never rewritten, so the clip is played to its end; what changes is that the encoder is
    never shown a row the clip does not have.
    """
    if n < 1:
        raise ValueError("clip has no frames")
    return clip_from_gen.lookahead_idx(min(int(frame), n - 1), n, flattened=False)
