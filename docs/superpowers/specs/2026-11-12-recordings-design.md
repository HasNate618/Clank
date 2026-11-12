# Per-run recordings: the game view and the robot view, watchable

2026-11-12. Status: design, waiting on approval.

## Intent

A run is now repeatable, but it leaves only text behind: `run.json`,
`brain.jsonl`, `brain.log`, and the stills the brain happened to look at. When a
run does something strange — a reach that misses, a grab that fires at 78 mm,
JobBot ignored — the diary says what the brain *decided* and says nothing about
what was on screen. Evaluation is guesswork, and a demo video has to be shot by
hand.

A run should leave a recording of its two pictures: what the agent saw (the
game's colour view) and what the robot did (the body view). The dashboard
should have a page that lists past runs and plays any of them, with that run's
diary beside it.

Purpose, in the operator's words: **evaluation now, demo video later.**

## Goals

- Every run that reaches its brain phase leaves a watchable video in its own
  directory, without anybody pressing record.
- The video carries the game's own audio, so the recording contains what the
  brain heard as well as what it saw. The diary's `heard` rows are transcripts of
  exactly this audio.
- One page in the dashboard lists the recordings (newest first, with duration
  and size) and plays one, with `brain.jsonl` beside the picture.
- Clicking a diary row jumps the video to that moment.
- Nothing about a run changes when recording is impossible (no ffmpeg, no depth
  stream). The run is never failed, delayed, or blocked by the recorder.
- The composite is a normal video file: `mp4`, H.264, playable in any player,
  shareable as the demo without a second step.

## Non-goals

- **Any audio but the game's.** `clank/ears.py` captures the `clank-game` sink,
  which only the game feeds; the desktop mix and the operator's microphone never
  enter the capture. That is deliberate and stays true.
- **Depth as video.** The depth plane is 480×270 float; a depth *video* is a
  debugging aid for one class of question (distance error) and a distraction for
  the two purposes here. The stills stay.
- **Burned-in overlays and subtitles.** No font dependency, no re-encode to
  change a label. The diary is HTML beside the video and seeks it.
- **Full-resolution or stereo.** The 3840×1080 view is reachable only through
  RenderDoc at ~16 s per frame. The agent's eye is what a recording of a run
  means.
- **Live streaming into the browser.** `/api/preview/live.rgba` already does
  that; a recording is for afterwards.
- **Editing, trimming, titles.** The file is evidence, not a film.

## Constraints (measured, 2026-11-12)

| Fact | Number |
| --- | --- |
| `clank/` is standard library only | no numpy, no PIL, no requests |
| ffmpeg on this machine | 8.1.2, `/etc/profiles/per-user/nate/bin/ffmpeg` |
| ffmpeg in the flake | **not** provided; recording must degrade to nothing |
| Game colour tile | 960×540 RGBA8, 30 fps, left eye, rows bottom-up |
| Depth plane | 480×270 float32 metres (not recorded) |
| Robot view | three JPEGs (560×420) rewritten atomically at 10 fps |
| Python read rate, `flip=True` | 58 fps (64 % of a core at 30 fps) |
| Python read rate, `flip=False` | **210 fps** — let ffmpeg `vflip` |
| Python push rate into ffmpeg | 259 fps at 960×540 RGBA |
| Encoded size, 960×540/30, `ultrafast` | **24.1 MB/min** (3.2 Mbit/s measured on live content) |
| Robot tile encoded | 30 KB / 10 s — noise beside the game |
| Game audio | `parecord` on `clank-game.monitor`, 16 kHz mono, `ears.capture()` |
| Game audio encoded | AAC ~0.5 MB/min — noise beside the video |

Disk is therefore the constraint that decides defaults: at 15 fps and a calmer
preset a 15-minute run should land near 100–150 MB, and a day of experimentation
is gigabytes. The page shows sizes so the cost is visible; pruning is a knob,
not a surprise.

## Approaches

**A. One composite file per run (chosen).** Record both pictures while the run
happens, mux them side by side into `recording.mp4` at run end. The dashboard
plays that one file. Demo-ready by construction.

**B. Two files plus an HTML player.** Record `game.mp4` and `robot.mp4`, and let
the page tile them with CSS against one clock; produce the composite only on
demand. Keeps full information and defers the layout, but the operator ends up
with two files that must be kept in step, and "the video" is a page rather than
a file.

**C. Frames only, no encoder.** JPEGs plus a stdlib viewer: no ffmpeg
dependency, exact frames, ~1.5 GB per 15 minutes.

Chosen: **A**, integrating into the dashboard as one page. The operator wants
videos and the brain log, not an archive of frames, and wants the demo to be a
file. B's flexibility is real but is worth adding later only if a question turns
up that the composite cannot answer.

## The how, as decided

**`clank/recording.py`, standard library.** One `Recording` object per run:

- `start()` spawns one ffmpeg per source, both writing MP4s into the run
  directory: the game from a rawvideo pipe (RGBA, `flip=False`, `vflip` in the
  filter), the robot from an `image2pipe` MJPEG feed at 10 fps. Two encoders
  avoid the impossible part: composing 960×540 pixels per py-thread frame in
  pure Python.
- A reader thread pulls `depth_stream.Stream.read(flip=False)` at `fps` and
  writes to the game pipe; a second thread copies the newest robot JPEG into the
  image2pipe at `robot_fps`, only when its mtime moved. **A frame the reader
  could not produce is written again from the last one**, at the tick the wall
  clock says it is due: a rawvideo pipe is constant-frame-rate, so a drop would
  otherwise shorten the file and slide every later frame away from the diary
  event that names it. Padding keeps click-to-seek honest to within one frame.
- **Geometry is fixed at 1680×540**: the game is 960 wide, the robot tile is
  scaled to 540 high (720 wide), and a missing source is black padding rather
  than a narrower file. One geometry keeps the composite argv a pure function
  and stops the page's aspect ratio changing between runs.
- A third encoder takes the game's audio: `ears.capture()` held open for the
  run, its PCM piped to `audio.m4a`, and its peak remembered for the manifest.
  Two `parecord` clients on one monitor (the brain's listener and this one) is
  the assumption to verify in the first live run; if one starves the other, the
  brain's listener and the recorder share a single capture instead.
- `stop()` closes all three pipes and waits (about a second), which is all the run
  pays. The compositing pass is **detached**: a background job runs it after the
  run has gone idle, so a fifteen-minute re-encode can never hold
  `run.wait_idle` or refuse the next Start run. `recording.json` is written
  twice, so its `state` says where it is: `recording` while the run is live,
  `ready` once the composite exists, `failed` with the reason. The page shows
  "composing…" for the middle state.
- The compositing job runs **one** ffmpeg
  (`hstack` after scaling the robot tile to the game height, both normalised to
  the same rate, plus `-c:a aac -shortest` for the audio, `-movflags +faststart`
  so a browser can play before the tail arrives) into `recording.mp4`, and
  deletes the parts unless `keep_parts`. Then it rewrites `recording.json` as
  `ready`.
- Every failure is a manifest, never an exception into the run: no ffmpeg, no
  depth stream, a dead pipe, a failed composite — `recording.json` records
  `available: false` and the reason, and the run continues.

**Ownership.** `RunManager` owns it, exactly like the brain child: created before
the brain starts (phase `running`, when the game and the body are both up),
stopped in `_finish` before the run directory is closed. A recording therefore
covers the part of a run that is worth watching, and a run refused at the model
gate leaves none.

**Clocks.** `recording.json`'s `started_at` is when the *recorder* started, not
`run.started_at`: the run's stamp is taken before the model gate, the game restart
and the body reset, which is ~25 s earlier measured. The page seeks to
`event_time − recording.started_at`, and `duration_s` is wall clock from the same
monotonic anchor, so a seek cannot run past the end of a file whose frames were
dropped.

**Manifest `recording.json`** (in the run directory, beside `run.json`):

```json
{
  "available": true,
  "file": "recording.mp4",
  "started_at": "...Z", "ended_at": "...Z",
  "duration_s": 214.6,
  "state": "ready",
  "width": 1680, "height": 540, "fps": 15,
  "bytes": 184320000,
  "sources": {"game": "ok", "robot": "ok", "audio": "ok"},
  "audio_peak": 19822,
  "encoder": "ffmpeg 8.1.2",
  "error": null
}
```

**Dashboard.**

- `GET /api/recordings` — every run with a `recording.json`, newest first:
  run id, session id, started/ended, duration, size, whether the file is
  playable, whether a diary exists, and whether the audio track was silent
  (`audio_peak`).
- `GET /recordings/<run_id>/video` — the file, **with Range support**, because
  `<video>` seeking without `206 Partial Content` does not work. This is the
  one piece of HTTP the dashboard does not have today. The server stays
  HTTP/1.0: Range over one connection per request is fine on localhost, and
  changing `protocol_version` would touch the framing of every other route.
- `GET /recordings` — a page: a list on the left (newest first, size shown), the
  video on the right, the run's `brain.jsonl` beneath it. A row clicked seeks the
  video to `t - run.started_at`.
- Reachable from the topbar and from the Brain panel ("watch this run"), so it is
  not a page you have to remember.

**Config**, a new `[recording]` section: `enabled = true`, `fps = 15`,
`robot_fps = 10`, `crf = 28`, `preset = "veryfast"`, `ffmpeg = "ffmpeg"`,
`keep_parts = false`, `max_total_mb = 20480` (0 = never prune). Unknown `CLANK_*`
overrides stay rejected.

**Retention: 20 GB.** `max_total_mb = 20480` by default. After a composite
finishes, the recordings are summed and the oldest are deleted until the total
fits — never the run just recorded, never a run whose composite has not finished,
and every deletion named in `session.log` (run id, bytes). `0` disables pruning.

**A stale sweep at dashboard start.** If the dashboard dies rather than a run
ending, its encoders keep writing into the run directory with nobody to close
their pipes. On startup, any `*.mp4.part` or part file with no `ready` manifest
is deleted and its run marked `failed: dashboard exited`. Their bytes count
toward the 20 GB until then.

## Tests

- The recorder against a **fake encoder**: a shell script that copies stdin to a
  file, so start/stop/manifest/parts are asserted without ffmpeg and without a
  GPU. The composite command is asserted as argv (it is a pure function).
- Absent encoder and absent stream: `available: false` with the reason, and a run
  that still reaches `running`.
- `RunManager` starts and stops the recorder, and a run refused at the gate has
  no `recording.json`.
- The listing route: a directory of runs, one with a manifest, one without, one
  broken manifest.
- The Range handler: `bytes=0-1` returns 206 with the right slice, a nonsense
  range returns 416, and a full GET still works.
- The page's shipped assets: the list container, the video element, the diary
  container, and the seek expression.
- Audio: a fake capture yielding known PCM gives the peak in the manifest; a
  capture that yields nothing is `available: true` with `audio_peak: 0` and a
  `silent track` note, not a failed recording.

## Open risks

- **Frame pacing.** Pulling 15 fps from a 30 fps seqlock stream keeps the newest
  frame; a stall of the reader (Python, GC) is padded to the wall clock, so it
  costs a still moment, not sync.
- **Sync.** Both encoders start within milliseconds of each other, but there is
  no shared clock stamp on the frames. Drift is bounded by the composite's
  `fps` normalisation (one frame at 15 fps = 67 ms). If a future question needs
  exact alignment, stamp frames in the instant they are read and record the
  stamps.
- **Disk.** 100–150 MB per typical run against a 20 GB cap: roughly 130 runs, or
  a fortnight of heavy experimentation, before the oldest start disappearing. A
  demo-grade export is worth copying out of the state directory first.
- **The game can be muted from outside.** `module-stream-restore` once remembered
  `Mute: yes` for `JobSimulator.exe`, and the capture is then silent while
  everything reports healthy. `audio_peak` in the manifest and the page's "silent
  track" note are the defence; the fix is a person typing `pactl`, not code.
- **ffmpeg absence is silent.** A run on a machine without ffmpeg records
  nothing and says so only in `recording.json` and `session.log`. The dashboard
  should show "recording off" on the recordings page rather than an empty list.
