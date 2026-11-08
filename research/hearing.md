# Hearing the game, and only the game

**STATUS: MEASURED, 2026-11-08.** LFM3-Omni-2B, PipeWire 1.6.8, the live Job Simulator
session.

The artifact sentence is "hears JobBot, does the task", and until now nothing in the repo
recorded audio. This is the ear: `clank/ears.py` for capture, `clank/brain.transcribe()` for
words, `clank/brainlog.py` for the sentences the brain says about itself.

## Capture by node, not by device - and the tool matters

The first design tapped the game's own PipeWire node:

```text
59  Stream/Output/Audio  application.name = JobSimulator.exe
```

with `pw-record --target 59 ...`. **That does not work on this machine, and it fails
silently.** Every capture was digital silence - peak 0, exit status 0, no stderr - while the
game was audible. Two positive controls proved it was the tool and not the game:

| control | recorded |
| --- | --- |
| `pw-record --target <bluez monitor>` while a 440 Hz tone played to that sink | peak **0** |
| `pw-record --target <node of the pw-play stream>` while that tone played | peak **0** |
| `parecord --device=<sink>.monitor` while the same tone played | peak **9000** |

So `pw-record --target` links to nothing here and quietly records the default *source*, which
is a muted microphone. `parecord` - the PulseAudio-compatible front end - links properly.
Monitor sources also exist only for the PulseAudio API: `pactl list short sinks` shows them,
`pw-dump` lists none of them, which is why they look absent.

The route that works, and is what `clank/ears.py` does:

1. `pactl load-module module-null-sink sink_name=clank-game` - a sink nothing else feeds.
2. `pactl load-module module-loopback source=clank-game.monitor sink=<the operator's sink>` -
   so moving the game does not silence it for the human watching.
3. `pactl move-sink-input <game> clank-game` - only the game's stream moves.
4. `parecord --device=clank-game.monitor --rate=16000 --channels=1 --format=s16le <file>`.

Measured after routing: a 5 s capture of the game at rest is `peak 2374, rms 529` - real
audio, where every earlier attempt was zeros. The desktop mix never enters the capture, so a
notification, a video, or the operator talking cannot be mistaken for JobBot.

`parecord` writes a WAV file and there is no piped mode here, so `TailReader` tails the file
and finds the `data` chunk while the writer is still appending to it.

## Two more ways the samples are silent, both real

1. **The sink-input was muted.** `pactl` showed `Sink Input #7076 ... Mute: yes` for
   `JobSimulator.exe` - `module-stream-restore` had remembered a muted state for that
   application. Worth checking first: `pactl list sink-inputs | grep -A2 JobSimulator`.
2. **The game is silent at the menu.** After a session restart the game sits on
   `PULL LEVER TO JOB` with no speech. JobBot's first lines come *after* the job starts, so a
   restart alone does not produce speech.

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
