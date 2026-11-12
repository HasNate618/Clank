# The generated reach, called the way the loop calls it

2026-11-12, 18:19. The reach adapter has been merged into `main` and `auto.py`'s act routes the
right hand to it (`act: generated reach`), the left to the cruise (`fe26579`). This run is the
first time that routing was exercised with the game, the depth layer and the body view all
live.

**Every number below is from the act function itself** — `auto.grab_pixel(840, 445, "right",
camera, grab=True)` under `spike/e3/.venv` — not from a probe, and not from a hand-assembled
argv. The camera is the head pose the frame was seen through, so the pixel and the camera agree
by construction.

## What it printed

```text
[brain] 18:19:54 act: generated reach
depth samples 3  spread 0 mm
object world (-0.45, -0.25, 0.988)  pelvis (0.258, -0.463, 0.199)  classified step
object -61.0 deg off the body's heading; the torso starts +64.0 deg off it (gate 20)
root goal would travel 82 mm (+0.04 m forward) and turn -61 deg: one step, worth a generation
gen 1: 52 frames in 0.313 s, glove residual 16.1 mm (wrist 23.5 mm), vram 0.94 GB
clasp  right glove 60 mm from the object (<= 80 mm), closed mid-clip, body reports closure (0.0, 1.0)
gen 1: achieved glove residual 12.2 mm (wrist 175.8 mm)
achieved hand residual 12.2 mm after 1 generation(s)
grab_pixel -> 0
```

- **0.313 s and 0.94 GB with the game up.** Risk R5 (the generator co-resident with the game and
  the Vulkan depth layer) holds under the real load, not just in a probe.
- **16.1 mm commanded, 12.2 mm achieved, on the glove.** The wrist link reads 175.8 mm for the
  same pose: that 0.18 m is `VR_LEVER`, and it is the arithmetic that makes a wrist-pinned
  constraint unable to clasp.
- **The clasp fired at 60 mm** against `CLASP_START_M = 0.08`, which is still uncalibrated. Two
  live samples now (60 mm here, 78 mm on 2026-11-11), both inside a threshold that has never
  been swept. A tighter threshold is untested either way.

`before-card-in-view.png` and `after-card-taken.png` are the game's own view either side: the
card is in JobBot's hand, then it is gone and the console reads `INSERT CARTRIDGE`.

## What the loop did on its own first, and could not

The run before this one reached `running`, heard JobBot's opening line (331 ms, peak 10295),
and then never acted on the card:

```text
think: plan: reach(the Job Simulator display/interface (central teal 'JOB SIMULATOR' sign and console))
act:   locate(...)  -> not_visible  at 0.45 m
think: ... is not visible at 0.45 m
think: plan: reach(the Job Simulator display/interface (...))      <- the same phrase
act:   locate(...)  -> not_visible  at 0.46 m
...                                                                 x3
act:   hold
```

`what-the-brain-saw.png` is the frame from that loop (`frame-181801.png`): the card sits at
`(840, 445)`, 0.281 m away, visible the whole time. The target it named is not a thing the
locator can confirm — stage 3 matches a description against this job's vocabulary
(card, cartridge, lever), so an unknown noun refuses even when a box was found. The refusal is
correct; the target was wrong. (design: `research/locate.md`)

So today the reach works when it is asked, and the loop does not ask for a reachable thing. That
is the open defect, and it is in the plan step, not in the generator.

## Two operational facts from the same session

- **A cold Steam start can swallow `rungameid`.** The first `POST /api/session/start` failed with
  `game failed: timed out waiting for process Z:\home\nate\Games\Job Simulator\JobSimulator.exe`.
  Steam had started (13.13 s to ready) and then asserted at post-logon — `Tool 4183110 "Steam
  Linux Runtime 4.0" unsupported version 0`, minidump uploaded — after which it logged nothing
  for three minutes and never processed the URL. A straight retry reached `running`. The install
  move itself is fine: Steam launched from `/home/nate/Games/Job Simulator/` at 16:18.
- The job's install now lives at `/home/nate/Games/Job Simulator`; `config.toml` and the
  `shortcuts.vdf` `Exe`/`StartDir` agree, and this run wiped the save and relaunched from there.
