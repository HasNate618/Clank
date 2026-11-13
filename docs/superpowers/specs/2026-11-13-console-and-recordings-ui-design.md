# One button, one panel, one picture — design, 2026-11-13

Status: awaiting approval. Written after four decisions were made by the operator
(see "Decisions"). Supersedes nothing; refines the runs and recordings specs of
2026-11-12 in the places named below.

## The problem, in the operator's words

> i run run.sh then open dashboard. the body is started already through run.sh?
> then i press "Start" which starts up everything, then i press "Start run" which
> restarts stuff (like game) and actually starts the run … theres like 3 steps to
> start the run. pressing start then start run which restarts game feels redundant.
> also having the 4 buttons for start/stop feels too bloated?
>
> switching between recordings and console feels like clicking a link instead of
> button. and the location of it is weired being in between mute and light/dark
> toggle. the "menu" button also feels like it relates to some menu on the dashboard
> when it actually sends controller input to the game … (i dont think these buttons
> even work)?
>
> for the recordings … there is only one body view on the side of the video with
> lots of whitespace for what should be portrait aspect ratio for body. also having
> the diary on the side makes the video smaller and there is so much whitespace
> below the video.

What the code says, checked before designing:

| Claim | What is true |
| --- | --- |
| "the body is started already through run.sh?" | Yes — `run_dashboard` starts `BodyPreview` when `body.autostart` (`clank/dashboard.py:775`). |
| "i dont think these buttons even work" | They work (`/api/body/start`, `/api/body/stop`), but they are **disabled whenever the body is already running / already stopped**, which is nearly always. They read as dead because they are grey exactly when you would try them. |
| 3 steps to a run | Structural, not cosmetic: `RunManager.start()` raises `ConflictError("session is not running")` (`clank/run.py:252`). A run cannot boot the stack today. |
| "Menu" = dashboard menu? | No — `/api/menu` makes the pose relay press the game's XR menu button. It is a game input sitting among page chrome, one button away from the theme toggle. |
| robot whitespace is padding | No. The body view renders 560×420 and 4:3 scaled to 540 is exactly 720×540: nothing is letterboxed. The empty space is the **render's own framing** — a small figure in a wide, distant shot. Cropping 4:3 to 9:16 would cut the arms off. |

## Decisions (operator, 2026-11-13)

1. **One button.** `Start run` boots the stack itself when it is down, then does
   what it does today. Shut down stays, demoted.
2. **A Controls panel.** Everything that steers the body or the game (menu press,
   look pads, manual override, body start/stop, mute) leaves the topbar for one
   panel in the right rail.
3. **Video-first recordings page.** One wide video at its native aspect; a diary
   strip below it; the run list as a slim rail.
4. **Portrait robot tile**, 304×540 inside a 1264×540 composite.

## Design

### 1. One button

`RunManager.start()` gains the stack: it calls `manager.start()` when the session
is not running, and waits for `running` inside the existing worker (never on the
HTTP thread), bounded by `cfg.timeouts.session_start_sec`.

```text
press Start run
  checking    model gate        (first, because it is cheapest: 0.03 s to refuse)
  stack       monado → weston → websockify → pose → steam → game
  game        restart_game()    (wipe save, relaunch, wait for the new client)
  body        reset
  brain       spike/wbc_game/auto.py
  running
```

- The gate stays first. A refusal must still cost nothing — and now it is the only
  thing that can refuse before minutes are spent.
- A **new phase** `stack` joins `RunState`/`PHASE_DETAILS`: "starting the stack…".
  The topbar spinner and the preview wait text use it like any other phase.
- `stop()` during `stack` cancels the boot with the same generation counter the
  other phases use, and leaves the stack as it found it: a run that is cancelled
  while booting does not half-boot and does not shut anything down.
- A failure anywhere in `stack` is a failed run with `run.json` holding the reason
  and a directory of evidence — the same contract as today. The stack is *not*
  rolled back: the operator may want to see the log.
- `POST /api/session/start` and `POST /api/run/start` both stay (the dashboard's
  own start is still the documented scripted path, and `run.sh` still exists). The
  page stops offering "Start": the topbar shows **Start run** when idle and
  **Stop run** while a run lives.

`Shut down` moves behind a small overflow in the topbar (`▾`) or into Controls:
it is the rare, destructive-ish action and should not sit shoulder to shoulder with
the one you press ten times a session. It stops the run first, then the stack, as
it does today.

### 2. A Controls panel

```text
topbar:  [brand]  ●session · run state · uptime    [Console|Recordings]  [Start run] [Stop run]  ▾
rail:    Brain
         Controls
           menu press                 (the game's XR menu button)
           look left / right
           manual override
           body ● running   [start|stop]
           🔊 game audio
         Logs
         Telemetry
```

- The tab pair is a segmented control on **both** pages, in the same slot, so the
  page you are not on is one click away and looks like a button, not a footnote.
- `Menu` becomes "Menu press" with a title that says what it does to the game.
- Mute moves to Controls: it is a property of the game's audio, not of the page.
- Body start/stop stays, but **the buttons reflect the body's own state** and stay
  enabled whenever an action is possible; when the body autostarted, the panel says
  `running` rather than offering two grey buttons.
- Panel order in the rail keeps the stage-first rule: Brain (what it thought),
  Controls (what you can do), Logs, Telemetry.

### 3. Video-first recordings page

```text
┌──────────────────────────────────────────────────────────────┐
│ runs: 21:18 ✓ · 20:44 ✓ · 19:02 ⌛                            │  slim strip
├──────────────────────────────────────────────────────────────┤
│  ▶   0:42 ──────●──────────── 1:39   1×                      │  transport
│                  [ 1264×540 video ]                          │  native aspect
│           ═══════ diary ticks on the rail ═══════            │
├──────────────────────────────────────────────────────────────┤
│  0:41  act      reach for the card                           │  diary, wide,
│  0:43  result   hand 24.7 mm from the glove                  │  scrolls with
└──────────────────────────────────────────────────────────────┘
```

- The video takes the page width and keeps its own aspect from the manifest
  (`width`/`height`), so **old 1680×540 recordings still play** without a re-encode.
- The diary becomes a wide log under the video, 2–3 columns of `time · kind · text`,
  still revealing with playback and still seeking on row click. Nothing about the
  synced-diary contract changes; only its place on the page does.
- The run list becomes a horizontal strip of chips (time, duration, state), so it
  never competes with the picture for width.
- `?run=` and `?t=` keep working, since they are what a demo link uses.

### 4. Portrait robot tile

- `[body]` gains a portrait render (405×720), i.e. the body view itself renders
  tall instead of 4:3, and the composite scales it to **304×540** (9:16).
  Cropping a 4:3 render to 9:16 would cut the arms, so the render changes, not the
  crop. The Robot panel gets the same portrait tiles for free, which is the aspect
  the operator says a body should have.
- Composite geometry becomes **1264×540** (game 960×540 + robot 304×540), 2.34:1
  instead of 3.11:1 — less like a letterbox strip, more like a frame.
- The camera preset may need a closer follow distance for the taller frame; that is
  a measurement, not a decision, and it is checked with the game off.
- Manifests record the geometry they were made with, and the page reads it, so the
  two geometries coexist in the list.

## Amendment, same day (operator)

Decision 2 is withdrawn. After seeing the panel built, the operator asked for the
controls back where they were: **mute returns to the topbar beside Shut down and the
theme toggle**, and the look pads, the manual/robot switch, the body's own Start/Stop
and the game's menu press stay **in the Robot panel**, whose whole job is the body.
There is no Controls panel and no overflow `▾` menu: Shut down goes back to being a
plain button in the bar. What survives from decision 2 is the honesty the panel was
there to fix — the body's buttons are no longer greyed out just because the body is
already in that state, and a `body-state` line says `running` or `stopped` in words.

Decisions 1, 3 and 4 stand.

## Non-goals

- No change to what a run *does* (save wipe, body reset, brain loop, model gate).
- No change to the recorder's three-encoder architecture or the composite-resume
  behaviour.
- No new `CLANK_*` env vars, no new dashboard framework, `clank/` stays
  standard-library only.
- No re-encoding of existing recordings.

## Risks

- **Booting the stack inside a run makes a slow failure slower.** Mitigated by the
  gate first, by `session_start_sec`, and by the fact that a refusal still costs
  nothing.
- **A run that boots the stack leaves it up when the run ends.** That is the point
  (stop a run, not the machine) and is stated on screen: after Stop run the topbar
  says the stack is still up.
- **Portrait body render is a live-tested change** (MuJoCo + a real session). If the
  camera cannot frame the body well in 9:16, we fall back to 405×720 padded to
  304×540 and say so in the manifest note rather than shipping a cropped arm.

## Verification

1. `python3 -m unittest discover -s tests -t .` and
   `nix build .#checks.x86_64-linux.default`.
2. Headless screenshots of both pages (`chromium --headless=new …`), dark and light.
3. One live run: press **Start run** only, from a stopped dashboard, and watch the
   phases reach `running`; press **Stop run** and see the recording indicator go
   `saving…` → `saved` with its link.
4. One offline composite at 1264×540, probed for size and duration.
