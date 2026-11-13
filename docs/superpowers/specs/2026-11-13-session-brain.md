# Session brain — Design

- Date: 2026-11-13
- Status: accepted as the live-path spec for this slice; implementation follows `docs/superpowers/plans/2026-11-13-session-brain.md`
- Parent: `docs/superpowers/specs/2026-10-31-brain.md` (instruction following). This document does not replace that parent. It restores two claims the parent already made and that the live loop never kept: **speech is an append-only transcript the model re-reads**, and **cues the body cannot sense are not offered as if they were live**.
- Replaces, for the live chooser only: `docs/superpowers/specs/2026-11-10-brain-tools.md` (archived: the live path asks for a goal in words, not a skill enum; code still locates when a motor needs a pixel).
- Implements against: `clank/brain.py`, `clank/ears.py`, `clank/brainlog.py`, `spike/wbc_game/auto.py`
- Does not implement: ARDY generation, the recording composite, glow/ghost detectors, a HubFSM

Claim tags: `[MEASURED]` / `[OPERATOR]` / `[ASSUMPTION]`.

## 1. Problem

Run `190131-679de2` (session `20261113T000018Z-30a1b5`) grabbed the card and JobBot spoke. The diary and the prompt both lied about what happened.

Measured on that recording:

| What | What was true | What the brain did |
| --- | --- | --- |
| JobBot's offer | video 12.4–18.4 s transcribes **"Here's my card."** `[MEASURED]` | `hang_s = 0.8` ended the intro at "…fine job museum."; `ready()` then spent 5.8 s not listening |
| JobBot's next line | video 42–46 s transcribes **"You grab one of the job cartridges off the table and let's get started."** `[MEASURED]` | `confirm()` ran `listen()`; the floor was the median of a loud first 0.5 s (2445 → gate 9781 vs peak 4415); diary wrote `no word from JobBot` |
| First plan | greeting only, no task | `place(console)` because the prompt names the cartridge and says the console is where things go `[MEASURED: replay at temperature 0, identical goal string]` |
| `object` field | goal `"take the card from JobBot"` | `"JobBot"`; locate would search for the summoner if the noun came from that field `[MEASURED]` |
| Model I/O | 9 calls (2 plan + 7 locate) | nothing stored; llama-server logs timings only |
| Memory | parent spec §5.3 "the model re-reads the transcript" | `think()` is one user message; `history=[]`; `WorkingMemory()` is constructed inside `one_turn()` |

The body reset (`ready()`: kill MuJoCo, respawn, `rest` × 8) ran **twice** in 132 s, including after the body was already standing. ARDY generates from the live body `[MEASURED: clip frame 0 within 1 mm / 0.8°]`; the reset is unverified since that change, and a reset **after** a grab would open the hand.

Glow and ghost are named in `CONSTITUTION` and in every `self:` line. No live caller computes either. `clank/glow.py` is a green detector used by `calibrate_view.py`. There is no ghost detector.

## 2. Claim

The live chooser is a **session**, not a turn. Every planner call sees every utterance, every previous planner reply, and every act/result from this run, plus the current still. It does not see a menu of grabable props, unimplemented cues, or stale pixels.

JobBot's opening, `[OPERATOR 2026-11-13]`:

1. "Hello human! Welcome to the Job Simulator. My name is Jobbot and I'm the curator of this fine job museum. Here's my card."
2. If the card is not taken: "Hey human, take my card."
3. After the card is taken: "Alright, why don't you grab one of the job cartridges off the table and let's get started."

(1) is one speech with a ~1.0–1.5 s pause. Capturing it as two utterances is a gate bug, not a game design.

## 3. Decisions

### 3.1 Hearing

- Default `hang_s` is **2.0 s** (was 0.8). JobBot's pause is shorter; a true end-of-turn is longer.
- Floor is the **20th percentile of every pre-start block**, never recomputed from a sliding 0.5 s window. The 0.5 s median, reset every half second, is what made a listen that opened on speech deaf to that same speech `[MEASURED: confirm window of run 190131-679de2]`.
- `Hearing` records `ended_by`: `hang` | `deadline` | `eof` | `never-started`.
- `confirm()` does **not** use the speech gate. It records a fixed window (`ears.record`, 8 s) and transcribes it. The gate's job is "wait for the next line"; confirm's job is "what did the game say just then", and the game talks over the grab.
- The confirm transcript is the **next instruction**, not only an oracle. It is appended to session memory and becomes the next turn's heard line if nothing newer has arrived.

### 3.2 Prompt

The planner sees, as the latest user text:

```text
You are a robot playing a VR game. This is what was heard: "<utterance or (none)>"
Obey the newest thing that was heard. Earlier speech is context, not an order.
self: left=<empty|holding> right=<empty|holding>
Decide what you are trying to do next, and where, from what you can see.
Reply with JSON only: {"goal": "...", "object": "...", "destination": "...", "side": "left|right"}
```

Banned in the planner prompt, seed, and constitution (the live path):

- any list of pick-up-able props (card / cartridge / lever / knob)
- "the console is where they go" / "the console is not a thing to pick up"
- glow, ghost, or any cue the live loop does not compute
- skill enums (`hold|look|find|…`)
- object positions, pixel numbers, metres

`locate._MATCH_SYNONYMS` stays. It is how stage 3 confirms a noun the model already chose. It is not shown to the planner. A model that names "name tag" will be refused by locate; that refusal is visible in session memory on the next call.

`DEFAULT_SEED` / `PROP_NOUNS` are not sent on the live path. `think()` does not take a seed.

If `task_intent(heard)` finds no verb and no offer (`"here's my card"`), **do not call the model**. Return a hold with note `nothing asked yet`. This is a gate on whether a task exists, not a parse of what the task is. Measured: the greeting plus a still produced `place(console)` because the prompt demanded a goal `[MEASURED]`.

### 3.3 Session memory

One `SessionMemory` per run, owned by `auto.main()`, passed into every `think()` and every `one_turn()`.

Carried into **future planner calls** (the `messages` array `_chat` sends):

| Carried | Form |
| --- | --- |
| every transcribed utterance | user text, in order |
| every planner prompt text | user text, no image |
| every planner raw reply | assistant text |
| every act/result the loop logged | user text (`result: locate(card) present 0.4 m`, `result: grab(card) reached code=0`, `result: refused place: no side`) |

Not carried:

| Dropped | Why |
| --- | --- |
| previous stills | the head moves; a pixel from last turn is a guess `[MEASURED: card 1009→1122→865 px inside one reach]`; images also dominate the context window |
| locate's quadrant / box / describe calls | a different question; they would drown the planner in crop JSON |
| WorkingMemory pixels | same staleness; a motor still locates on a fresh still |
| glow / ghost | no producer |

The current still is the only image, always on the latest user message.

`WorkingMemory` stays **per turn** and still holds pixels. Session memory is linguistic; working memory is geometric.

Locate calls stay one-shot. They do not read `SessionMemory`.

Cap: if `messages` would exceed 80 turns of (user+assistant) pairs, drop the oldest **result** lines first, then the oldest planner pairs, never the heard utterances. `[ASSUMPTION: 80 pairs × ~250 tokens ≪ 65536]`. A test pins that heard lines survive the cap.

**Override, measured 2026-11-10:** text in the prompt overrode speech when it named an errand. The line "Obey the newest thing that was heard" is the mitigation. After this ships, replay the four card stills *with* a prior `place(console)` in history: the new utterance "take my card" must still become `grab`. If it does not, the prior assistant reply is stored but the prior user prompt is truncated to `heard: …` only (drop the model's own previous goal from what it can copy).

### 3.4 Diary

The diary stays the demo. It also has to make the next run diagnosable without an audio autopsy.

- One `heard` row per listen, including quiet: `floor`, `gate`, `peak`, `speech_s`, `ended_by`. Quiet rows are allowed when they carry those numbers; a quiet row with no numbers is still banned.
- Refusals are `result`, not `act: hold (reason)`.
- `ready()` logs `reset body` with duration and is an `act`, not a `think`.
- `llm.jsonl` next to `brain.jsonl` (env `CLANK_LLM_LOG`). One object per `_chat` call: `{t, purpose, model, max_tokens, prompt_text, image_sha256, raw_reply, prompt_ms, decode_ms, prompt_n, predicted_n}`. No base64. Locate's seven calls go here too, because `_chat` is the one writer.

Recordings stay as they are: both views, game audio, a manifest. Verbosity goes in the diary, not the video.

### 3.5 Body reset

`ready()` is recovery, not a schedule.

- Once at run start (`main`), so the first still is from a standing body.
- Again only when the last motor failed (`code != 0`), the body socket is missing, or the body process is not running.
- Never after a successful grab or place. `rest` opens the hands.
- Never at the start of every `one_turn`.

`reach_vr.py`'s own `rest` before a cruise reach stays; that is the generated/cruise path's settle, not a session concern. Do not double it from `auto.py` on a success path.

Whether a reset still improves ARDY's achieved glove residual is **unmeasured** since the generator landed. Log duration and root yaw before/after every reset so a later measurement has the numbers. Do not block this slice on that experiment.

### 3.6 Place

`place` is already a `skill_from_goal` output and `spike/wbc_game/place.py` already runs it. The loop currently logs `place is not wired yet`. After a successful card grab, JobBot's next line is a grab of a cartridge, not a place — but a later "put it in" will be a `place`. Wire it: locate the destination, `approach_point`, reach, open. Empty hand → the loop already has to grab first only if the model said place; **do not** reintroduce `sequence()`. If the model says place with empty hands, the refusal is `result: place refused: hands empty`, and the next think sees that.

## 4. What a planner call looks like

```
POST /v1/chat/completions
temperature 0, max_tokens 160
messages:
  [prior user/assistant/result texts from SessionMemory]
  {role: user, content: [current JPEG q90 960-wide, prompt in §3.2]}
```

No system message. No audio (transcribe is a separate call). No seed.

The raw reply is JSON `{goal, object, destination, side}`. `skill_from_goal(goal)` plus `task_intent(goal)` for the noun (not the `object` field) remain. That mapping is code, measured: the model is good at a goal in words and bad at an enum `[MEASURED 2026-11-10: 0/4 grabs from an enum, 4/4 "take the JobBot card"]`.

## 5. Non-goals

- Feeding previous stills back. If a later measurement shows the model needs the last frame to judge "is the card gone", add **one** previous still, not the run.
- A glow or ghost detector. Do not mention them until one exists.
- Putting `_MATCH_SYNONYMS` in the planner prompt.
- Resetting the body after every successful act.
- Changing the recording composite.
- Making locate itself conversational.
