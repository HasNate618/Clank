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
