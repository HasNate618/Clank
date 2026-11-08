# Hearing the game, and only the game

**STATUS: MEASURED, 2026-11-08.** LFM3-Omni-2B, PipeWire 1.6.8, the live Job Simulator
session.

The artifact sentence is "hears JobBot, does the task", and until now nothing in the repo
recorded audio. This is the ear: `clank/ears.py` for capture, `clank/brain.transcribe()` for
words, `clank/brainlog.py` for the sentences the brain says about itself.

## Capture by node, not by device

The game is a PipeWire client, so its output is a node of its own. While a session runs,
`pw-dump` lists it as

```text
150  Stream/Output/Audio  application.name = JobSimulator.exe
```

and `pw-record --target 150 --raw --rate 16000 --channels 1 --format s16 -` streams that
node's PCM. Recording the node means a notification, a video, or the operator talking on
the microphone can never be mistaken for the game - the desktop is not in the graph path.

The node id changes on every launch (161 before a restart, 150 after), so `game_node()`
resolves it per capture by matching `application.name`. It refuses rather than guessing:
`no playback stream matching 'JobSimulator' (playing: Zen)`.

## Two reasons the node is silent, both real

1. **The sink-input was muted.** `pactl` showed `Sink Input #7076 ... Mute: yes` for
   `JobSimulator.exe` - PipeWire's `module-stream-restore` had remembered a muted state for
   that application. Every capture was `peak 0`. This is worth knowing because it looks
   exactly like a broken recorder: the node exists, `pw-record` connects, and the samples
   are all zero. Check `pactl list sink-inputs | grep -A2 JobSimulator` before believing
   anything about the gate.
2. **The game is silent at the menu.** After a session restart the game sits on
   `PULL LEVER TO JOB` with no speech at all. JobBot's first lines come *after* the job
   starts, so a restart alone does not produce audio.

Digital silence is therefore the normal state between lines, and it broke the first gate
design: gating on `floor > 0` made the ear deaf to a digitally silent room, which is
precisely this game. The gate keys on `floor_known`, not on the floor being non-zero. A
test caught that before the game did (`test_a_room_tone_alone_is_not_speech` and friends).

## The gate

Blocks of 50 ms, RMS, floor taken as the median of the first 0.5 s, gate at
`max(4x floor, 250)`. A block above the gate starts the recording and keeps the 0.45 s
pre-roll, so the first syllable is not clipped; it ends after 0.8 s of quiet, less 0.2 s
kept so a final consonant survives. Measured on synthetic blocks with the fake reader, and
on the live node: silence in, `no speech heard (stream ended)` out, never a hang.

## What the model hears

`brain.transcribe()` sends the clip as `input_audio` (base64 WAV) with one instruction:
write down the words verbatim, `(silence)` if there are none. It deliberately does **not**
also decide what to do - measured 2026-11-04, one call doing both produced a goal that
ignored what was said. Hear first, decide second; `sequence()` and the calibrated prompts
then work unchanged.

On the first 8.5 s of the walkthrough ("NO COMMENTARY", so it is the game's own audio):

```text
Hello human, welcome to the job simulator. My name is JobBot, and I'm the curator of this fine
                                            238 ms prefill + 310 ms decode
```

which is SRT lines 1-2 with the audio cut mid-sentence, and the reply was flagged as a cut
line (`...` / trailing dash) rather than silently accepted. The punctuation is tidied and
"JobBot" is capitalised, so a transcript is not a citable quote - it is input.

Silence comes back as an empty transcript, not an error: `(silence)`, `silence`,
`[silence]`, `no speech detected.` and `...` all parse to `""`.

## The brain's own sentences

`clank/brainlog.py` appends one JSON object per message to
`~/.local/state/Clank/brain/messages.jsonl` and echoes the same line to stderr:

```text
[brain] 11:27:32 heard: sounding out the words...
[brain] 11:27:33 heard: Alright, why don't you grab one of the job cartridges
[brain] 11:27:33 think: I will grab the job cartridge (object=job cartridge, side=right)
[brain] 11:27:33 think: next: grab(the job cartridge) -> place(the job cartridge)
[brain] 11:27:33 act: grab 'the job cartridge' (skill=grab)
[brain] 11:27:34 result: grab ok (skill=grab, code=0)
```

That is the demo, so it is written as it happens rather than reconstructed. The dashboard is
being rebuilt, so nothing reads the file yet: it is a diary, not state, and
`CLANK_BRAIN_LOG` redirects it for tests and probes.

## Commands

```bash
# what can be recorded, and is the game among them
python3 -m clank.ears --list

# one utterance from the game, with the words
python3 -m clank.ears --transcribe

# keep listening and print each utterance as it happens
spike/e3/.venv/bin/python spike/wbc_game/listen.py --for 120

# a whole turn: hear, look, decide, act (--dry-run to stop after deciding)
spike/e3/.venv/bin/python spike/wbc_game/do_task.py --listen --dry-run

# read the diary back
python3 -m clank.brainlog --tail 20
```
