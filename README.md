<p align="center">
  <img src="docs/media/title-card.svg" width="480" alt="Clank - probationary unit">
</p>

# Clank

A vision model decides, and a simulated humanoid body does it, in a commercial VR game that was never meant to be driven by either. It sees the game's own rendered frame and depth buffer, chooses one action per turn, and a generated motion prior carries the body through it. No headset, no controllers, no game files touched.

![Run 190131-679de2 on the recordings page: the game's view on the left, the simulated body on the right, and the model's diary below. JobBot says "Hey, Gilman, take my card", and the diary shows the plan, the search and the generated reach.](docs/media/grab-190131.webp)

*Run `190131-679de2`, 12 November 2026. The game's own view, the body that is driving it, and the diary underneath. JobBot says "Hey, Gilman, take my card"; the entries that follow are `plan: grab(card)`, the search for the card, `locate(card)` and `generated reach`. Nothing in that column was typed.*

## Why

The question is how much of a task a small model can carry when it has no privileged view of the world. It gets one picture, one sentence from the instructor, and what happened last time. Then it has to name the next action itself, turn after turn, until the job ends or it gets stuck. There is no task list to follow and no goal written by me: a fixed seed names the world, and every steering decision after that is the model's.

A VR game makes that a fair test rather than a toy. It cannot be scripted, it has real physics, and it has an instructor with an actual script. It also answers back. Either the card ends up in the hand or it does not, and the difference is visible in one frame.

The hard part is geometry, because the picture is all the model has. The head will not hold a commanded pose: held against one target for ten seconds it wobbled **11.7° of yaw**, so a pixel marked once goes stale inside a single reach. Measured, the card moved 1009 → 1122 → 865 → 1007 px during one four-command reach ([`research/reach-placement.md`](research/reach-placement.md)). The fix was to stop tracking the pixel and re-derive it: unproject the target through the measured head into metres, then project it back every tick.

## Three models, and nothing else

| Model | Its job | Where it comes from |
|---|---|---|
| **LFM3-Omni-2B** (Q8_0) | plans every turn, transcribes JobBot, grounds and verifies a crop | Liquid AI, served by llama.cpp on `:11434` |
| **GEAR-SONIC** | tracks a reference motion at 50 Hz, 500 Hz PD | NVIDIA NVlabs, ONNX Runtime |
| **ARDY-G1-RP** | turns one constraint into 52 frames at 25 Hz | NVIDIA, PyTorch, CUDA |

The loop uses three models and no others. I did not train, fine-tune or quantise any of them: the work here is the loop, the geometry, and the plumbing that makes three published models cooperate on one 8 GB card. Python 3.12 (standard library only) for the application in `clank/`; MuJoCo 3.17 in the spikes, where one 20 ms control tick costs 17.66 ms of encoder, decoder and physics ([`research/wbc-policy-proposal.md`](research/wbc-policy-proposal.md)); NixOS, Steam and Proton; Monado with its `remote` driver; a Vulkan layer in C and GLSL.

Solo project. `pi` was the coding agent and DeepSeek V4.1 Flash the model that typed most of the code. The perception chain — the depth buffer, unprojected to metres, projected back to a pixel — and the shape of the loop are mine, and every claim below is backed by a test or a measurement in `research/`.

## Architecture

```mermaid
flowchart LR
    G["Job Simulator<br/>Proton · OpenXR"]
    L["Vulkan layer<br/>480×270 depth, 960×540 colour"]
    V["LFM3-Omni-2B<br/>one action per turn"]
    P["target in metres<br/>unprojected through the head"]
    A["ARDY-G1<br/>52 frames from one constraint"]
    S["SONIC · MuJoCo<br/>50 Hz policy, 500 Hz PD"]
    R["pose_relay<br/>376 B at 50 Hz"]
    M["Monado remote driver<br/>:4242"]

    G -->|"injected after the eye pass"| L
    L -->|"13 ms"| V
    V -->|"locate"| P
    P -->|"hand here, feet planted"| A
    A --> S --> R --> M
    M -->|"head and two hands"| G
```

Read it as one turn of a loop rather than a pipeline. The game is both the sensor and the actuator: the layer reads the frame it has already rendered, and the pose relay writes the next one. There is one simulation. The achieved `mjData` poses go out as 376-byte packets, and the same poses draw the body tile in the image above, so the game and the operator are looking at the same robot rather than two lookalikes.

**Seeing.** The layer is injected after the final eye pass, reads the game's 3840×1080 reversed-Z depth and its colour target, and writes a 480×270 depth map in metres plus a 960×540 colour image into a seqlock-protected mmap. Same frame, twice:

![One frame of Job Simulator, 2026-11-12: the colour the layer reads on the left, the depth it reads on the right, where near surfaces are dark and everything past 10 m is white.](docs/media/seeing-frame.webp)

**Moving.** An action becomes a constraint, right hand at the object with feet planted, and ARDY-G1 writes 52 frames at 25 Hz. SONIC tracks them at 50 Hz with a 500 Hz PD loop and ten frames of look-ahead. The clasp starts closing at 80 mm and the hand is pinned there afterwards. A root waypoint is designed for the out-of-reach case and has never been exercised.

## The brain

One still, one sentence, one action. That is the entire decision surface, and this is what a turn actually looks like:

```text
seed:   The game is Job Simulator. The things you can pick up are a card, a job
        cartridge and a knob. The console is not a thing to pick up.
        The voice is a job instructor.
heard:  Hey, Gilman, take my card.
self:   left=open right=open yaw=0.02 glow=unknown ghost=unknown
last:   done
image:  960×540 colour, from the layer
```

The constitution it answers to is short enough to quote. It states one action, the honesty rules that were measured to be necessary, and the schema as a requirement rather than an example:

```text
You are a robot with two hands. You see one image from your eyes and the speech
since your last action.

You may do exactly one action: hold, look, find, reach, grab, release ...
After find you are told present or not_visible. After a hand action you are told
done, saturated, or timeout, and you see a new image.

Do what the voice asked. Do not invent extra chores. Find a thing before you
reach for it. If find said it is not there, look. If you are unsure, hold.

Grab when glow is yes or unknown; do not grab when glow is no. A yellow ghost
means the held thing will go into the slot if you release.

Reply with one JSON object and nothing else. All three keys are always present:
{"skill": ..., "side": "none"|"left"|"right", "target": <noun or null>}
```

It is not given object positions, a task list, a current goal, or the plan it made last turn beyond the outcome of its last action. It answers with one of six verbs because those are all the adapter can compile: the intent is validated, and an illegal one is rejected rather than clamped. It never writes a joint, a pose or a pixel.

Three calls, not one. **PLAN** picks the verb, side and noun from the still. **GROUND** asks where that noun is, as one quadrant question about a crop the caller supplies. **VERIFY** asks what the cropped patch actually is before the hand closes. The split is measured: this model collapses any image-global grid toward the centre (3/12 on thirds, 0/12 on a nine-cell pick, 0/12 on a 4×4), while the same question about a hand-supplied crop scored 44/44 on the tray.

Four things measuring it changed:

- **The seed was the largest single cause of a wrong verb.** A seed carrying the walkthrough choreography answered `release … cartridge` on three different stills when JobBot had just said "take my card". The seed now names the world and no errand, and the same call became `grab card`.
- **It copies its own constitution.** Put `release` first in the list and it answers `release`, pipe character included. The schema is therefore written as a requirement, and the parser treats a reply with the verb in the key position as the same decision rather than throwing it away over punctuation.
- **Refusing does not teach it.** Told `reach right card -> not_located`, it re-issued `reach` on 3 of 3 stills, and adding the rule in words changed nothing. So for any action that needs a pixel, the search is done in code instead of trusted to the model's next turn.
- **It reads the noun out of the sentence.** On all six motor replies in the same sweep it named the object JobBot had actually said: `card`.

The nouns the model may use are the ones the locator can confirm, and a test fails if the locator's vocabulary and the seed's list drift apart. Every turn is written to the diary as one record: what was heard, what was seen, what was said, what it was told back. The panel in the image at the top is that file, not a reconstruction.

## Decisions and tradeoffs

**Inject into the render pass, not a capture.** RenderDoc replays a frame in about 16 s; the layer costs 13 ms. The price is C and GLSL inside the game's render pass, and knowing that the target is written twice per frame, once by a clear pass and once by a load pass that adds the other eye and its glow. Only the load pass may be injected. Inject both and the preview alternates between dim and bright frames.

**Generate the reach instead of hand-writing and tracking it.** Hand-writing a reference and asking the policy to follow it left residuals of 49–79 mm, because the policy answers a reference with something of its own. A constraint into a motion prior gets 16.1 mm commanded and 12.2 mm achieved. The cost is a CUDA dependency, a right hand only, and a preflight that can refuse, so the scripted path still exists as the fallback.

**Relay the state file the viewer already writes.** A second pose client would have been easy and would have meant two simulations: the tile showing one body and the game another, and every screenshot a quiet lie. The relay is standard-library, reads `body_state.json`, and holds the last pose with a `stale` flag instead of freezing silently.

**A crop per turn, not a bigger locator per turn.** On a full frame, the grounded VL model found the card to 22 px where the 2B model was 360 px off, but the same crop took it 19.3 s against 0.73 s. The loop needs an answer per turn, so the 2B model stays on a crop it can read, and absence is decided in code because neither model can be trusted to refuse.

## Results

| Hop | Result | Conditions | Source |
|---|---|---|---|
| See | layer **13.9 ms** median, p95 24.8; transport 1.4 ms; 29.8 fps | 480×270 depth, 960×540 colour | `research/evidence/depth-stream/latency.txt` |
| Hear | **238 ms** prefill + 310 ms decode | one utterance, 2B model, llama.cpp | `research/hearing.md` |
| Plan | **210 ms** prompt + 203 ms decode | one still plus the sentence | `research/brain-tools.md` |
| Locate | card **21.0 px** (21 mm at 0.332 m), lever 5.9 px | one live frame, five box reads | `research/locate.md` |
| Generate | **16.1 mm** commanded, **12.2 mm** achieved on the glove, 0.313 s, 0.94 GB | game and depth layer live, 52 frames | `research/evidence/ardy-in-the-loop/README.md` |
| Drive the game | `look_left` **−28.6°**, `look_right` **+54.3°**, `hold` returns to ±0.8° | mode `body`, 50 Hz relay | `research/r6-co-residency.md` |
| Walk | **39.79 m** in 60 s, no fall | SONIC default, headless CPU | `research/sonic-gate-results.md` |
| Co-residency | **44.2 Hz** beside the game against 44.5 Hz alone, +177 MiB | one 8 GB card, two views | `research/r6-co-residency.md` |

## Limitations

- **Nobody has played it.** The body took a card and held it; no level has been cleared, and the run above ended with "no word from JobBot after the grab".
- **The body does not walk in-game.** It is clip-anchored: one generated window reaches, it does not step, so anything out of arm's reach is not attempted.
- **It runs at 44.2 Hz against a 50 Hz design point** with the game up. The tick advances exactly 20 ms of simulated time, so the shortfall is real: the game sees 0.89× real time.
- **The torso does not turn to look.** It started 63.9° off the card and the reach did not turn it, so the body reaches sideways at a target it is facing away from.
- **The clasp threshold is a guess.** 80 mm has fired twice, at 60 and 78 mm, and has never been swept either way.
- **The model is not reliable about its own tools.** One run planned `reach(the display interface)` again and again, located nothing, and never acted on the card; absence is therefore decided in code rather than by the model.
- **Controller orientation is unverified.** Positions are exact, but the wrist frame is not an OpenXR grip, so the in-game aim may be rotated by a fixed offset.

## Quick start

You need NixOS with Steam, your own copy of *Job Simulator* (not included, never will be), Monado built with the `remote` driver, an NVIDIA card with 8 GB, and llama.cpp serving the 2B model on port 11434. The model's flags matter and are pinned in [`AGENTS.md`](AGENTS.md); a server started with defaults answers differently.

```bash
git clone <this repo> && cd Clank
cp config.example.toml config.toml    # steam_bin, game_exe, openxr_active_runtime, launch ids
nix develop --command spike/e5/build.sh   # the Vulkan layer
./run.sh                              # dashboard on http://127.0.0.1:8765
```

Then press **Start run** in the dashboard. It boots the stack if it is down (Monado, Weston, websockify, Steam, the game), refuses before spending anything if the model server is not up, wipes the game's save so the job starts at its first frame, stands the body up, and starts the loop. One run costs about 25 s to reach `running`, and leaves a directory with the diary, the frames and a video.

Setup, ports, VNC and the full control table are in [`docs/setup.md`](docs/setup.md).

## Tests

```bash
XDG_RUNTIME_DIR=/tmp/clank-xdg python3 -m unittest discover -s tests -t .
```

**1378 tests in about 120 s**, standard library only: no network, no GPU, no game. They cover the command bus and its validator, the wire format, the run state machine, the recording manifest, the depth reader, the dashboard routes and the transcription boundary. They do not cover the parts that need hardware: the layer's latency, the policy's tracking, and anything that touches the game are measured by hand and recorded in `research/`.

Without a GPU or a copy of the game, the tests and `research/evidence/` are the offline path.

## Where things live

| Path | What |
|---|---|
| `clank/` | The application. Config, session orchestrator, run state machine, dashboard, pose relay, recording, depth reader, and `brain.py`, the prompt assembly and the three model calls. Standard library only. |
| `spike/e5/` | The Vulkan depth layer: C, GLSL, the launcher, the calibration. |
| `spike/wbc_game/` | The loop (`auto.py`), the locator, the reach adapters, the pixel picker. |
| `spike/ardy/`, `spike/e3/` | The motion generator and the MuJoCo body. Each has its own virtualenv. |
| `spike/c0/` | The scripted pose clients, still the fastest way to tell a body fault from a pipe fault. |
| `research/` | The measurements: one document per spike, with conditions, negative results and evidence images. Every number above comes from here. |
| `docs/` | Designs and their archive. `docs/README.md` says which is current. |
| `tests/` | The suite. |

## License and credits

MIT for this repository, see [`LICENSE`](LICENSE). *Job Simulator* is proprietary and not distributed here. GEAR-SONIC and ARDY-G1 are NVIDIA's, under the NVIDIA Open Model License; LFM3-Omni-2B is Liquid AI's. All three are wrapped, not modified. Monado is patched for a full visibility mask and a 60 Hz null compositor (`spike/e5/monado-no-visibility-mask.patch`).
