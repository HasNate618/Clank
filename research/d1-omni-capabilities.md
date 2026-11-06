# d1-omni — capabilities & formatting

Model on `localhost:11434`: **`d1-Omni-2B-Q8_0.gguf`**, architecture **`lfm3-d1-omni`**
(Liquid AI LFM3, "decoder-one"). Served by `llama-server` with
`mmproj-d1-Omni-2B-Q8_0.gguf`, ctx 16384, 4 slots.

Everything below was verified live against the server, the GGUF metadata, and
`tools/server/server-decision.cpp` (`COMMON_DECISION_TYPE_LFM3_D1_OMNI`).

## 1. What it is

A **decision model, not a chatbot**. It cannot generate text:

```
POST /v1/chat/completions → 501 "This model is a decision model without text generation, use /v1/systemone"
POST /v1/completions      → 501 (same)
POST /completion          → 400 (same message via the legacy path)
```

The only productive endpoint is **`POST /v1/systemone`**. You send a `state`
plus `questions`; the server renders one prompt, runs **one forward pass, zero
generated tokens**, and reads the answer off the next-token logits at the
assistant position. `usage.output_tokens` is always `0`. Typical latency on
this box: ~100–300 ms text, ~200 ms image, ~80 ms audio.

Think Kahneman: this is System 1 — fast reflexive judgment over a closed set
of options. There is no `/v1/systemtwo`.

## 2. Request format

```json
{
  "state": "a cat sits on a mat",
  "questions": {
    "animal":  { "type": "choice", "instructions": "What animal is present?",
                 "criteria": { "cat": "a feline", "dog": "a canine" } },
    "present": { "type": "noul",   "instructions": "Is an animal present?" },
    "rain":    { "type": "score",  "instructions": "How likely is rain?",
                 "criteria": ["impossible", "unlikely", "possible", "likely", "certain"] }
  }
}
```

| field | rule |
|---|---|
| `state` | **required**, but may be `null` on this model (e.g. asking about an image only). Text, a JSON value, or an array of chat content parts. It is *context only* — the model is never asked anything about it except through `instructions`. |
| `questions` | **non-empty object**. Keys are your ids, echoed back in `answers`. Any number per request; they share one evaluated state prefix. |
| `questions.<id>` | must be an **object** (a bare string is a 400). |
| `instructions` | **required**. The actual question. May also be an `input_audio` part (spoken question, §5). |
| `type` | exactly `choice`, `score`, or `noul`. Anything else is a 400. |

### choice — pick one of N named options

- `criteria`: **non-empty object** `{key: description}`. An array is a 400.
- A key maps to one option; the description may be a string, JSON, or `null`
  (null/empty ⇒ the key itself, underscores → spaces, is shown).
- Option order is request order (this model does **not** sort keys).
- Up to 255 options. Labels in the prompt: your keys if *all* are single
  letters, else `A`–`Z` (≤26 options), else `AA, AB, …`.

### score — rate on an ordinal scale

- `criteria`: **array of 2–10** ordered levels. 1 level or 11+ is a 400.
- Option keys become `"0"` … `"n-1"`; the model must reply with a single digit.
- Returns the **median level**: first level whose cumulative probability ≥ ½
  (not the argmax, not the mean — verified: `{0:0.48, 1:0.44, …}` → `1.0`).

### noul — yes/no, one number

- `criteria`: optional object, only `true`/`false` keys
  (`{"true": "…", "false": "…"}`). Anything else is a 400.
- Options are always ordered `[true, false]`; the answer is **P(true)**.
- With descriptions the prompt shows `Yes: …` / `No: …` lines; without them it
  just says "Reply with yes or no only."

## 3. Response format

```json
{
  "model": "/home/nate/AI/models/d1-Omni-2B-Q8_0.gguf",
  "answers": {
    "animal":  { "type": "choice", "choice": "cat",
                 "probabilities": { "cat": 0.9656, "dog": 0.0344 },
                 "confidence": 0.9656 },
    "rain":    { "type": "score", "score": 1.0,
                 "legend": { "0": "impossible", "1": "unlikely", "2": "possible",
                             "3": "likely", "4": "certain" },
                 "probabilities": { "0": 0.48, "1": 0.44, "2": 0.07, "3": 0.01 },
                 "confidence": 0.4798 },
    "present": { "type": "noul", "noul": 0.9090 }
  },
  "usage": { "input_tokens": 43, "output_tokens": 0 }
}
```

- `choice` → winner (`argmax`), full distribution, `confidence` = max probability.
- `score` → median level (float), `legend` (index → your level text), full
  distribution, `confidence` = max probability.
- `noul` → single number, P(yes). No confidence field — the number *is* the confidence.
- `probabilities` keys are in your option order.
- Calibration: softmax `p = exp((logit − max) / T)` with **`T = 3.382983…` baked
  into the GGUF** (`lfm3.decision.temperature.{noul,choice,score}`), identical
  for all three types. There is no per-request temperature; you cannot tune it.
- `usage.input_tokens` is real (prompt + media tokens); `output_tokens` is 0.

## 4. Prompt format (what the model actually reads)

The template ships inside the GGUF (`tokenizer.chat_template.systemone`).
Reconstructed rendering — byte-verified (re-tokenizes to the server's own
`input_tokens`):

```
<|startoftext|><|im_start|>user
[media tokens, if files[] were sent]
a cat sits on a mat


QUESTION:
What animal is present?

Options:
A a feline
B a canine

Reply with the option code only.<|im_end|>
<|im_start|>assistant
<think></think>█
```

Per-type endings:

| type | options block | reply rule |
|---|---|---|
| `choice` | `A a feline` / `B a canine` (label + description) | `Reply with the option code only.` |
| `score` | `0 impossible` … `4 certain` (digit + level) | `Reply with a single digit 0-4 only.` |
| `noul` | `Yes: …` / `No: …` (only if criteria given) | `Reply with yes or no only.` |

`█` is the answer position: the server never samples that token, it reads the
logits there. Answer-token groups: choice → label tokens; score → the digit
tokens; noul → `yes/Yes/YES` vs `no/No/NO` (first single-token form wins).

Anti-injection: control-token spellings in request text get a zero-width space
injected (`d1_break`), so user text can't smuggle special tokens. Non-string
state/instructions are serialized as indented JSON into the prompt.

Budgets (silent truncation past these): text ≤ 16384 tokens total; with media,
text is capped at 896 (image) / 15360 (audio) tokens minus media tokens;
options share `max(96, min(24·n + 32, half))` tokens evenly; the question gets
≥ 16 tokens. Keep states short and options shorter.

## 5. Multimodal capabilities

| modality | status on this server |
|---|---|
| text | full — state, instructions, criteria |
| image | **works via `files[]`** (512px test: 295 tok, correct at 0.978) |
| audio | **works via `files[]`** (1 s 16 kHz WAV: 52 tok, 200 Hz → "low" 0.725) |
| spoken question | supported: `instructions` as `{"type":"input_audio","input_audio":{"data","format"}}` with the clip in `files[]` (200 on test) |
| video | declared (`modalities.video: true`) but impractical here — frames would overflow the 16384 batch |

**Media format — the one rule that matters:**

```json
{ "state": null, "files": ["data:image/png;base64,iVBOR…"], "questions": { … } }
{ "state": null, "files": ["data:audio/wav;base64,UklGR…"], "questions": { … } }
```

- `files` entries must be **data URLs**. Objects (`{"data","format"}`) are a 400.
- **One audio clip max, or images — never both** in one request (400 otherwise).
- `state: null` is the idiom for media-only questions.
- ⚠️ The sibling content-part forms are broken on this build: an `image_url`
  part cost 1357 tokens and answered *wrong* (0.557 for the wrong color vs
  0.978 correct via `files[]`); an `audio_url` part overflows the batch (500).
  **Use `files[]`.**

## 6. Batching

One request, many questions, one shared state evaluation (prefix cache):

```json
{ "state": "You have 22 HP. Enemy has 60 HP and is attacking.",
  "questions": { "move": {…choice…}, "danger": {…noul…},
                 "urgency": {…score…}, "odds": {…score 0–9…} } }
```

→ all four answered, `usage: {input_tokens: 232, output_tokens: 0}`.
Each question renders its own prompt variant off the shared state, so batching
is the cheap way to interrogate one situation from several angles. (Only one
ordering variant per question on this model — no position-bias averaging.)

## 7. Error catalog (all observed live)

| trigger | status | message |
|---|---|---|
| no `state` | 400 | `"state" must be provided` |
| `questions: {}` | 400 | `"questions" must be a non-empty object` |
| question as string/array | 400 | `questions.q1: must be an object` |
| no `instructions` | 400 | `questions.q1: "instructions" must be provided` |
| `type: "vibe"` | 400 | `questions.q1: "type" must be one of: choice, score, noul` |
| choice criteria as array | 400 | `questions.q1: "criteria" must be a non-empty object` |
| score with 1 / 11 levels | 400 | `questions.q1: "criteria" must be an array of 2 to 10 levels` |
| noul criteria as array | 400 | `questions.q1: "criteria" must be an object` |
| `files` as objects | 400 | `images must be data URLs (data:image/...;base64,... or data:audio/...;base64,...)` |
| audio + image together | 400 | `a request has images or one audio clip, not both` |
| chat/completions/completions | 501 | `This model is a decision model without text generation, use /v1/systemone` |
| audio via content part | 500 | `input (N tokens) is too large to process…` |

## 8. Model facts

- LFM3 decoder, **24 blocks, 2048-d, 16 heads, GQA kv=4 on alternating layers**,
  short-conv layers interleaved, **sliding window 256 every 4th layer**,
  **non-causal attention**, RoPE base 1e6, 128k trained context (server: 16384).
- 1.73B params, Q8_0; vocab 128k (BPE, `lfm2` pre-tokenizer); bos 124894,
  eos 124900, pad 124893.
- GGUF tags: `liquid, lfm3, edge, decision, classification, extraction,
  calibration, system-one, multimodal, audio, audio-text-to-text`;
  license `lfm1.0`; 16 languages (en ar zh fr de it ja ko pt es vi th id hi ru pl).
- Server: llama.cpp `llama-server`, `--ctx-size 16384 --mmproj … --flash-attn`,
  12 threads. Sampling settings in `/props` (temp 0.8 etc.) are vestigial —
  nothing is ever sampled.

## 9. curl

```bash
curl -s localhost:11434/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "a cat sits on a mat",
  "questions": {
    "animal":  { "type": "choice", "instructions": "What animal is present?",
                 "criteria": { "cat": "a feline", "dog": "a canine" } },
    "present": { "type": "noul",   "instructions": "Is an animal present?" },
    "rain":    { "type": "score",  "instructions": "How likely is rain?",
                 "criteria": ["impossible", "unlikely", "possible", "likely", "certain"] }
  }
}' | python3 -m json.tool

# image (state null, file as data URL):
curl -s localhost:11434/v1/systemone -H 'Content-Type: application/json' -d "{
  \"state\": null,
  \"files\": [\"data:image/png;base64,$(base64 -w0 pic.png)\"],
  \"questions\": { \"c\": { \"type\": \"choice\",
    \"instructions\": \"Which colour is on the left half?\",
    \"criteria\": { \"red\": \"red\", \"blue\": \"blue\" } } } }"
```

Interactive version of all of the above: `python3 serve.py` → http://127.0.0.1:8777/
