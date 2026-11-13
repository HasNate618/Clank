"""The brain: hear a sentence, look at one still, emit one legal intent.

This is the writer described by `docs/superpowers/specs/2026-10-31-brain.md`. It
owns three things and no others:

1. **Prompt assembly** - a short constitution (body facts and an output schema),
   one injected seed, and a per-turn live block. There is no walkthrough here.
2. **Three model calls** - PLAN (which skill, which side, which noun), GROUND
   (where in the image is that noun) and VERIFY (does that crop actually name the
   noun, once the hand is at the container). They are separate calls on purpose;
   see `ground()` and `verify()` for the measurements that forced the split.
3. **The trace** - one `TurnRecord` per turn saying what it heard, saw, said, was
   told, and whether the intent survived `validate_intent`.

It does **not** move the body. It produces `Intent`s; the adapter compiles them.

Two measurements decided the shape of this file, both taken against the live
`LFM3-Omni-2B` endpoint on 2026-11-01:

- Every image-global grid collapses toward the image centre on this model: 3x3
  thirds 3/12, a 9-cell single pick 0/12, a 4x4 grid 0/12. The same question asked
  about a hand-drawn crop of the container scored 44/44 on the tray, and the
  identical prompt on the full frame 9/12. So GROUND asks one quadrant question
  about a container box the caller supplies; see `ground()`.
- Without an explicit "all three keys always present" line the model omits `side`
  entirely and emits `{"skill": "look", "target": "human"}`, which the intent
  schema rejects. The constitution therefore states the schema as a requirement,
  not an example.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import pathlib
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from typing import Any

from .body import (  # noqa: F401  (re-exported: outcome_from_flags and callers use these)
    FLAG_FALL_RISK,
    FLAG_SATURATED,
    FLAG_TTL_EXPIRED,
    FLAG_UNSUPPORTED,
)
from .frames import Frame, crop_jpeg
from .intent import (
    LOOK_ARC_RAD,
    Intent,
    validate_intent,
)

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "LFM3-Omni-2B-Q8_0.gguf"

#: The planner chooses from two kinds of tool, and only one kind reaches the adapter.
#:
#: MEASURED 2026-11-10: the live loop never asked the model to choose. `task_intent()` read the
#: verb out of JobBot's sentence, and `sequence()` then always ran locate and always reached. The
#: model was a parser. Restoring the chooser means the model may name `locate`, which is a *sense*
#: tool: code runs it and the pixel goes into working memory, never into an `Intent` and never
#: onto the socket, because `clank.intent.SKILLS` has no such skill and the adapter cannot compile
#: one. See `docs/superpowers/specs/2026-11-10-brain-tools.md` §3.
SENSE_SKILLS = ("locate",)
#: The word the model reaches for. MEASURED 2026-11-10: offered as `locate` it was chosen 0 of 3
#: stills and every hand action came back `release`; offered as `find` it was chosen on the frame
#: where the card was clearest. Same tool, so the alias costs nothing and hides no decision.
SENSE_ALIASES = {"find": "locate", "search": "locate"}
MOTOR_SKILLS = ("hold", "look", "reach", "grab", "release", "menu")
PLANNER_SKILLS = MOTOR_SKILLS + SENSE_SKILLS

#: The words the constitution promises the model will be told after each action.
#: Kept to exactly these three: a fourth would break the sentence in
#: `CONSTITUTION`, and the model was measured to answer *around* a promise it was
#: given, not to honour one it was not.
OUTCOMES = ("done", "saturated", "timeout")


def outcome_from_flags(flags: int) -> str:
    """The outcome word for a turn, from the flags the executor computed.

    The executor sets `FLAG_SATURATED` on every tick where the achieved pose is
    further from the target than `tolerance_m`, and writes it into
    `body_state.json`. Nothing read it, so a saturated reach looked like a clean
    one to the next PLAN turn and the model would retry the identical action.

    `FLAG_SATURATED` wins over `FLAG_TTL_EXPIRED` on purpose: both can be set on
    a reach that never arrived, and "saturated" is the fact worth acting on —
    it says the body could not get there, so the retry has to differ. "timeout"
    invites the same action again.

    `FLAG_FALL_RISK` and `FLAG_UNSUPPORTED` map to "done" only because the
    constitution offers three words and no more. They are recorded in the trace,
    not invented into the model-facing vocabulary.

    **Warning, measured 2026-11-01** (`research/sonic-reach-envelope.md`): at the
    contract's default `tolerance_m` of 0.02 m this flag is set on *every* reach,
    because ordinary SONIC tracking error is 0.035–0.18 m. So the word is
    uninformative for a reach — every turn reads "saturated". Use
    `reach_outcome()` instead when a glow cue is available.
    """
    if flags & FLAG_SATURATED:
        return "saturated"
    if flags & FLAG_TTL_EXPIRED:
        return "timeout"
    return "done"


def reach_outcome(glow: str, flags: int) -> str:
    """The outcome word for a **reach**, preferring the game's grab-range highlight.

    Measured 2026-11-01: at the default 0.02 m tolerance `FLAG_SATURATED` fires on
    every reach, including ones the body tracked well, because ordinary SONIC
    tracking error is 0.035–0.18 m. Job Simulator paints the object blue when the
    hand is in grab range, which answers "did the hand arrive" directly instead of
    inferring it from tracking error that cannot carry the claim.

    `glow` is the live-block cue: `"yes"`, `"no"`, or anything else for "we do not
    know". Only the two measured values are honoured; an unrecognised string falls
    back to the flags rather than being read as a `yes`, because a silently
    mis-parsed cue would invent an arrival that never happened.
    """
    if glow == "yes":
        return "done"
    if glow == "no":
        return "saturated"
    return outcome_from_flags(flags)


def last_line(
    skill: str, side: str, target: str | None, outcome: str
) -> str:
    """One `last:` line for `live_block`, in the shape §7.3 specifies.

    `none`/`null` rather than omission: the model is told what it asked for, and
    a missing field reads as a schema it can leave out next time.
    """
    return f"{skill} {side} target={target or 'null'} → {outcome}"


#: Skills that address one hand. `menu` belongs here: the controller's system button exists per
#: hand, and MEASURED 2026-11-10 the model answers `{"skill":"menu","target":"card"}` with no side,
#: which the side rule then refused while the loop ran it anyway. Defaulting the hand is the same
#: call this parser already makes for a reach.
PLANNER_SIDE_SKILLS = ("reach", "grab", "release", "menu")

#: Judged on the recording: a 30 s intro at 1 frame/s leaves ~30 turns of slack.
DEFAULT_TIMEOUT_S = 30.0

CONSTITUTION = """\
You are a robot with two hands. You see one image from your eyes and the speech \
since your last action.

You may do exactly one action:
- hold    keep still
- look    turn to look; name a target to look toward it, or omit the target to \
look a step to the side
- find    find a named thing in what you see; do this before you reach for it or \
grab it
- reach   move one hand toward a named thing you have located
- grab    close one hand
- release open one hand

While holding something, reach moves it with that hand.

After find you are told present or not_visible. After a hand action you are \
told done, saturated, or timeout, and you see a new image.

Reply with one JSON object and nothing else. All three keys are always present:
{"skill":"hold"|"look"|"find"|"reach"|"grab"|"release",\
"side":"none"|"left"|"right","target": <noun string or null>}

side is "none" for hold, look and find, and is "left" or "right" for reach, \
grab and release.

Do what the voice asked. Do not invent extra chores. Find a thing before you reach for it. If find said it is not there, look. If you are unsure, hold.

A full hand cannot grab; use the other hand or release.\
"""

#: Identity and nothing else. MEASURED 2026-11-10: the seed that carried the walkthrough
#: ("use the store clerk cartridge ... pull the lever") overrode what JobBot actually said - asked
#: to take his card, the model answered `release ... cartridge` on three different stills and under
#: every constitution wording tried, and dropping the choreography turned the same call into
#: `grab card` / `find card`. The seed is the largest single cause of a wrong verb measured here,
#: so it now names the world and no errand. See `research/brain-tools.md`.
#: Since 2026-11-13 it names no prop either: `think()` does not send a seed at all, and the planner
#: prompt names no object it could be picked up from (spec §3.2). The constant stays because the
#: replay and evaluation tools (`spike/brainview`, `spike/wbc_game/eval_plan.py`) still default to
#: it. `PROP_NOUNS` went with the menu: `locate._MATCH_SYNONYMS` is the only prop list left.
DEFAULT_SEED = "The game is Job Simulator. The voice is a job instructor."

#: The four answers GROUND accepts. A tray quadrant is the smallest set that
#: measured above chance on this model; see `ground()`.
QUADRANTS = ("TOP-LEFT", "TOP-RIGHT", "BOTTOM-LEFT", "BOTTOM-RIGHT")

#: Words a reply uses to refuse. All three become `error="not visible"`.
_ABSENT_WORDS = ("NONE", "ABSENT", "NOT VISIBLE")


@dataclass(frozen=True)
class BrainConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_s: float = DEFAULT_TIMEOUT_S
    temperature: float = 0.0
    max_tokens: int = 160


# --------------------------------------------------------------------------- #
# model transport
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelReply:
    text: str
    prompt_ms: float = 0.0
    decode_ms: float = 0.0
    error: str = ""
    #: llama.cpp's own token counts, for the `llm.jsonl` trace. 0 when the call failed.
    prompt_n: int = 0
    predicted_n: int = 0


def llm_log_path() -> pathlib.Path | None:
    """Where `llm.jsonl` goes, or None for no trace.

    `CLANK_LLM_LOG` names the file outright, and an empty value disables the trace, so a
    test or a probe can opt out of the one env var the dashboard exports per run.
    Without it the file is the diary's sibling: a run that sets `CLANK_BRAIN_LOG` gets
    `llm.jsonl` in the same directory for free.

    Set it for the **brain process only**. `clank/config.py` rejects any `CLANK_*` that is not
    `<section>_<key>`, so exporting it before `run.sh` stops the dashboard at startup with
    "unknown environment override". `CLANK_BRAIN_LOG` is safe for the same reason: the run
    writes it into the child's environment and never into the shell's.
    """
    explicit = os.environ.get("CLANK_LLM_LOG")
    if explicit is not None:
        return pathlib.Path(explicit) if explicit else None
    diary = os.environ.get("CLANK_BRAIN_LOG", "")
    return pathlib.Path(diary).with_name("llm.jsonl") if diary else None


def _image_digest(part: dict[str, Any]) -> str:
    """sha256 of the JPEG inside an `image_url` part, so a still can be cited by digest."""
    url = str((part.get("image_url") or {}).get("url") or "")
    if not url.startswith("data:") or "," not in url:
        return ""
    try:
        return hashlib.sha256(base64.b64decode(url.split(",", 1)[1])).hexdigest()
    except ValueError:
        return ""


def _trace_llm(messages: list[dict[str, Any]], reply: ModelReply) -> None:
    """Append one line per call: what was asked, what came back, and the timings.

    The still is recorded by digest and never by content - a base64 JPEG is neither
    readable nor cheap, and this file is read by a human diagnosing a run. The text of
    the last user message is the prompt, so a call that carries a still is still
    greppable. A trace that cannot be written (a read-only state dir, a full disk) must
    never fail the call it describes.
    """
    path = llm_log_path()
    if path is None:
        return
    prompt_text = ""
    image_sha256 = ""
    try:
        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                prompt_text, image_sha256 = content, ""
                continue
            texts = []
            image_sha256 = ""
            for part in content or []:
                if part.get("type") == "text":
                    texts.append(str(part.get("text") or ""))
                elif part.get("type") == "image_url":
                    image_sha256 = _image_digest(part)
            prompt_text = "\n".join(texts)
    except (AttributeError, TypeError, ValueError):
        # A malformed message is not worth failing the call over, and half a row is worse than
        # a short one: keep whatever was read before the bad part.
        pass
    # One row must fit one `write()` or a parallel reader (locate's box reads) can interleave
    # halves of two rows. CPython writes in 8 KB text chunks, so this keeps a row atomic.
    prompt_text = prompt_text[-4000:]
    row = {
        "t": round(time.time(), 3),
        "prompt_text": prompt_text,
        "image_sha256": image_sha256,
        "raw_reply": reply.text,
        "prompt_ms": reply.prompt_ms,
        "decode_ms": reply.decode_ms,
        "prompt_n": reply.prompt_n,
        "predicted_n": reply.predicted_n,
        "error": reply.error,
    }
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    except OSError:
        return


@dataclass(frozen=True)
class ServerStatus:
    """What the local llama.cpp server says about itself."""

    up: bool = False
    #: "ok" once weights are loaded, "loading" while llama.cpp is still loading
    #: them, "down" when nothing is listening.
    status: str = "down"
    model: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "up": self.up,
            "status": self.status,
            "model": self.model,
            "error": self.error,
        }


def short_model(name: str) -> str:
    """`/home/x/models/LFM3-Omni-2B-Q8_0.gguf` -> `LFM3-Omni-2B-Q8_0`."""
    leaf = name.rsplit("/", 1)[-1]
    return leaf[:-5] if leaf.endswith(".gguf") else leaf


def server_status(base_url: str = DEFAULT_BASE_URL, timeout_s: float = 0.4) -> ServerStatus:
    """Ask the planner's server whether it is serving, and which weights.

    `/health` is llama.cpp's own report: 200 `{"status":"ok"}` once a model is
    loaded, 503 while it is still loading. The distinction matters to an
    operator - "down" and "loading" look identical from a timer otherwise, and
    one of them just needs waiting. The weights come from `/v1/models`, asked
    only when the server is up, so a stopped server costs one refused
    connection and nothing else.
    """
    base = base_url.rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=timeout_s) as response:  # noqa: S310
            body = json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            return ServerStatus(up=True, status="loading")
        return ServerStatus(error=f"http {exc.code}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return ServerStatus(error=f"transport: {exc}")
    except json.JSONDecodeError as exc:
        return ServerStatus(error=f"health is not json: {exc}")

    model = ""
    try:
        with urllib.request.urlopen(f"{base}/v1/models", timeout=timeout_s) as response:  # noqa: S310
            models = json.loads(response.read().decode("utf-8") or "{}").get("models") or []
        if models:
            model = short_model(str(models[0].get("name") or models[0].get("model") or ""))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, AttributeError):
        # The name is a nicety; the health answer is the answer.
        model = ""
    return ServerStatus(up=True, status=str(body.get("status") or "ok"), model=model)


def _chat(
    config: BrainConfig,
    messages: list[dict[str, Any]],
    max_tokens: int | None = None,
) -> ModelReply:
    """One POST to the OpenAI-compatible endpoint, traced to `llm.jsonl`. Never raises."""
    reply = _chat_request(config, messages, max_tokens)
    _trace_llm(messages, reply)
    return reply


def _chat_request(
    config: BrainConfig,
    messages: list[dict[str, Any]],
    max_tokens: int | None = None,
) -> ModelReply:
    """The POST itself, so `_chat` is the one writer of the trace."""
    payload = {
        "messages": messages,
        "max_tokens": max_tokens if max_tokens is not None else config.max_tokens,
        "temperature": config.temperature,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    # The endpoint is the operator's local server, never a URL from a frame or a model.
    request = urllib.request.Request(  # noqa: S310
        f"{config.base_url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_s) as response:  # noqa: S310
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        return ModelReply("", error=f"http {exc.code}: {detail}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return ModelReply("", error=f"transport: {exc}")
    except json.JSONDecodeError as exc:
        return ModelReply("", error=f"response is not json: {exc}")

    elapsed_ms = (time.monotonic() - started) * 1000.0
    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ModelReply("", error="response had no message content", decode_ms=elapsed_ms)

    timings = body.get("timings") or {}
    return ModelReply(
        text=str(text),
        prompt_ms=_timing(timings, "prompt_ms", 0.0),
        decode_ms=_timing(timings, "predicted_ms", elapsed_ms),
        prompt_n=int(_timing(timings, "prompt_n", 0.0)),
        predicted_n=int(_timing(timings, "predicted_n", 0.0)),
    )


def _timing(timings: dict[str, Any], key: str, default: float) -> float:
    """One number out of llama.cpp's `timings`, or `default`.

    A server that answers `"prompt_n": null` must not turn a successful reply into an
    exception: this function is inside `_chat_request`, which promises never to raise, and an
    exception here would also skip the trace line for a call that did happen.
    """
    try:
        return float(timings.get(key, default))
    except (TypeError, ValueError):
        return default


def _image_part(frame: Frame) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": frame.data_url()}}


def _audio_part(wav: bytes) -> dict[str, Any]:
    encoded = base64.b64encode(wav).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}}


# --------------------------------------------------------------------------- #
# TRANSCRIBE
# --------------------------------------------------------------------------- #

#: The audio path asks for words, not a decision. Measured 2026-11-04: asking one call to
#: both transcribe and choose produced a goal that ignored what was said. The calibrated
#: prompts take a `situation` string, so hear first and decide second.
TRANSCRIBE_PROMPT = (
    "You are listening to a video game character through the game's own audio. "
    "Write down what they say, verbatim, as one line of plain text. No quotes, no speaker "
    "name, no commentary. If you hear no speech, reply with exactly: (silence)"
)

#: What the model says when the clip was silence, in the shapes it has actually used.
_SILENCE_REPLIES = frozenset({
    "", "silence", "(silence)", "no speech", "(no speech)", "no speech detected",
    "(no speech detected)", "[silence]", "...",
})


@dataclass(frozen=True)
class Transcript:
    """One utterance, as heard by the model."""

    text: str = ""
    raw: str = ""
    prompt_ms: float = 0.0
    decode_ms: float = 0.0
    error: str = ""
    ellipsis: bool = False

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def empty(self) -> bool:
        return not self.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "raw": self.raw,
            "ellipsis": self.ellipsis,
            "error": self.error,
            "prompt_ms": round(self.prompt_ms, 1),
            "decode_ms": round(self.decode_ms, 1),
        }


def clean_transcript(text: str) -> str:
    """One line of speech from whatever the model wrapped it in.

    A live capture tends to come back as a quoted sentence, and an unfinished one as a
    trailing ellipsis or a bare dash - both are kept, because a half-heard instruction is
    still the instruction and the caller can see it was cut.
    """
    line = " ".join(text.strip().split())
    for opener, closer in (('"', '"'), ("'", "'"), ("`", "`"), ("\u201c", "\u201d")):
        if len(line) > 1 and line.startswith(opener) and line.endswith(closer):
            line = line[len(opener) : len(line) - len(closer)].strip()
    if line.lower().rstrip(".!").strip() in _SILENCE_REPLIES:
        return ""
    return line


def transcribe(config: BrainConfig, wav: bytes) -> Transcript:
    """Hear one clip. Never raises; an error is a field."""
    if not wav:
        return Transcript(error="no audio")
    reply = _chat(config, [
        {"role": "user", "content": [
            _audio_part(wav),
            {"type": "text", "text": TRANSCRIBE_PROMPT},
        ]},
    ])
    if reply.error:
        return Transcript(raw=reply.text, error=reply.error)
    cleaned = clean_transcript(reply.text)
    return Transcript(
        text=cleaned,
        raw=reply.text.strip(),
        prompt_ms=reply.prompt_ms,
        decode_ms=reply.decode_ms,
        ellipsis=cleaned.endswith(("...", "\u2026", "-")),
    )


# --------------------------------------------------------------------------- #
# PLAN
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Plan:
    """The model's proposed action, before it becomes an `Intent`."""

    skill: str | None = None
    side: str | None = None
    target: str | None = None
    raw: str = ""
    error: str = ""
    note: str = ""
    prompt_ms: float = 0.0
    decode_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.error and self.skill is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "side": self.side,
            "target": self.target,
            "raw": self.raw,
            "error": self.error,
            "note": self.note,
            "prompt_ms": round(self.prompt_ms, 1),
            "decode_ms": round(self.decode_ms, 1),
        }


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a reply, tolerating fences and chatter."""
    candidates: list[str] = []
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    candidates.append(text)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate.strip())
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def live_block(
    left: str,
    right: str,
    yaw_rad: float,
    last: str,
) -> str:
    """The hands and the last outcome. No task list, no goal, no cues.

    `heard` is not a parameter: the situation line says what was heard, and repeating it here was
    the duplicate this block used to emit. Glow and ghost went with the live loop's cue detectors,
    which never had a producer.
    """
    return (
        f"self:  left={left} right={right} yaw={yaw_rad:.2f}\n"
        f"last:  {last}"
    )


def plan(
    config: BrainConfig,
    seed: str,
    frame: Frame,
    live: str,
    transcript: list[str] | None = None,
    wav: bytes | None = None,
    default_side: str = "right",
    system: str | None = None,
) -> Plan:
    """Ask for one action given one still and the recent speech.

    `default_side` exists because the model's key names are not stable across
    modalities: measured 2026-11-01, the same prompt answered `{"skill":
    "release","side":"right",...}` from text and `{"action":"reach",
    "target":"Job Card"}` with the clip attached. The audio path is the one that
    chooses the right *skill*, so the parser accepts `action` as a synonym for
    `skill` and fills a missing side from the free hand, recording that it did.
    """
    content: list[dict[str, Any]] = []
    if wav:
        # Sentinel-only audio is not allowed: if the caller passes a clip, the
        # model hears it. This is the path measured at 197 ms prefill.
        content.append(_audio_part(wav))
    content.append(_image_part(frame))
    blocks = [f"seed: {seed}"] if seed else []
    if transcript:
        blocks.append("heard so far:\n" + "\n".join(f"- {line}" for line in transcript))
    blocks.append(live)
    content.append({"type": "text", "text": "\n\n".join(blocks)})

    reply = _chat(config, [
        {"role": "system", "content": system if system is not None else CONSTITUTION},
        {"role": "user", "content": content},
    ])
    if reply.error:
        return Plan(raw=reply.text, error=reply.error)

    parsed = extract_json_object(reply.text)
    if parsed is None:
        return Plan(raw=reply.text, error="reply contained no JSON object",
                    prompt_ms=reply.prompt_ms, decode_ms=reply.decode_ms)

    raw_skill = parsed.get("skill", parsed.get("action"))
    skill = raw_skill if isinstance(raw_skill, str) else None
    note_alias = ""
    if isinstance(skill, str) and skill.strip().lower() in SENSE_ALIASES:
        canonical = SENSE_ALIASES[skill.strip().lower()]
        note_alias = f"skill {skill!r} read as {canonical!r}"
        skill = canonical
    side = parsed.get("side")
    target = parsed.get("target")

    note = note_alias
    # The model's key names are not stable, but its content is. Three shapes came
    # back for the same request on 2026-11-01:
    #   {"skill":"release","side":"right","target":"Job Bot card"}
    #   {"action":"reach","target":"Job Card"}
    #   {"reach":"right","target":"Job Bot card"}
    # The last puts the skill in the key position. Rejecting it would throw away a
    # correct decision over punctuation, so it is folded in and the fold is
    # recorded - visible in the trace, not hidden.
    if skill is None:
        for key, value in parsed.items():
            if key in PLANNER_SKILLS and isinstance(value, str) and value in (
                "none", "left", "right"
            ):
                skill = key
                side = side or value
                note = f"skill expressed as a key: {key!r}"
                break

    if side is None and skill in SENSE_SKILLS:
        # MEASURED 2026-11-10: asked for a sense tool, the model omits `side` entirely -
        # `{"skill":"find","target":"card"}` - which the side rule below would reject for a tool
        # that has no hand. A sense tool never addresses a side, so its side is "none".
        side = "none"
    if side is None and skill in PLANNER_SIDE_SKILLS:
        side = default_side
        suffix = f"side missing; defaulted to {default_side}"
        note = f"{note}; {suffix}" if note else suffix

    error = ""
    if skill is None:
        error = f"no skill key in {sorted(parsed)!r}"
    elif skill not in PLANNER_SKILLS:
        error = f"skill {skill!r} is not one of {PLANNER_SKILLS}"
    elif not isinstance(side, str) or side not in ("none", "left", "right"):
        error = f"side {side!r} is not none/left/right"
    elif skill in PLANNER_SIDE_SKILLS and side == "none":
        error = f"{skill} needs a side"
    elif skill not in PLANNER_SIDE_SKILLS and side != "none":
        # MEASURED 2026-11-10: asked what to do about JobBot's card, the model answered
        # `{"skill":"hold","side":"right","target":"card"}`. The decision is right and the field
        # is noise, so it is coerced rather than rejected - the same call this parser already
        # makes for a skill expressed as a key. Rejecting it turned a correct hold into an error
        # and the whole turn into nothing.
        side = "none"
        coerced = f"side ignored for {skill}"
        note = f"{note}; {coerced}" if note else coerced

    if target is not None and not isinstance(target, str):
        error = error or f"target {target!r} is not a string or null"
    elif skill in SENSE_SKILLS and not (isinstance(target, str) and target.strip()):
        # A sense tool with nothing to sense is not an action, it is a shrug. Measured on the
        # frozen-still arm: the reply `{"skill":"locate","side":"none","target":null}` only
        # appears when the model has lost the noun, and running it would locate nothing and
        # store nothing, leaving the next think no better informed.
        error = error or f"{skill} needs a target"

    return Plan(
        skill=skill if isinstance(skill, str) and skill in PLANNER_SKILLS else None,
        side=side if isinstance(side, str) else None,
        target=target.strip() if isinstance(target, str) and target.strip() else None,
        raw=reply.text,
        error=error,
        note=note,
        prompt_ms=reply.prompt_ms,
        decode_ms=reply.decode_ms,
    )



# --------------------------------------------------------------------------- #
# NEXT STEP
# --------------------------------------------------------------------------- #

#: Words that mean each skill, in the order they are checked. The model is asked
#: for a *goal in words*, and this table is what turns that into the closed
#: vocabulary `clank.intent` accepts. Keeping the mapping here rather than in the
#: prompt is deliberate: measured 2026-11-04, offering `grab|place|reach|release`
#: as an enum turned "you are holding a cartridge" into `{"skill": "grab"}`
#: every time, while asking for the goal in words produced "insert the cartridge
#: into the game console" with the destination named. The model is good at
#: saying what it wants and bad at picking from our enum, so we let it talk and
#: do the picking ourselves.
_SKILL_WORDS = (
    # `menu` first: "press the menu button" contains "press", and a goal that names the controller
    # button is not a reach. MEASURED 2026-11-10 against the goal prompt.
    ("menu", ("menu button", "press menu", "the menu")),
    ("place", ("insert", "place", "put", "slot", "load", "seat")),
    ("grab", ("take", "grab", "pick", "lift")),
    ("release", ("release", "let go", "drop")),
)

NEXT_STEP_PROMPT = (
    "You are a robot playing a VR game. {situation}\n"
    "Obey the newest thing that was heard. Earlier speech is context, not an order.\n"
    "Decide what you are trying to do next, and where, from what you can see.\n"
    'Reply with JSON only: {{"goal": "...", "object": "...", "destination": "...", '
    '"side": "left|right"}}'
)
#: No prop list, no cue, no skill enum in the line above, and that is the measurement: a prompt
#: that named the props and the console turned JobBot's greeting into `place(console)` on a still
#: where nothing had been asked for (spec §3.2). The model names the noun in its own words and
#: `locate.py` - not this prompt - decides whether a noun can be confirmed.


#: Words that name the act, and the filler that sits between the verb and the noun. The
#: line JobBot says is an instruction, so the verb and the object are both in it; asking the
#: model for them is what produced "insert the card into the machine" from "take my card".
_TASK_VERBS = (
    ("pick up", "grab"), ("hand me", "grab"), ("take", "grab"), ("grab", "grab"),
    ("get", "grab"), ("lift", "grab"),
    ("insert", "place"), ("put", "place"), ("place", "place"), ("slot", "place"),
    # Last, so no earlier match changes meaning. `press` is the one verb JobBot uses for the
    # controller button ("press the menu button" before the job starts), and without it the
    # greeting gate in `think()` would swallow a line that does name a task.
    ("press", "menu"),
    # A direction is a task. Measured 2026-11-13: "pull the lever to job" matched no verb here,
    # so `task_intent` returned ("", "") and `think()`'s gate held - the lever was never attempted
    # on any run. They map to `reach`, not to skills of their own: the direction and its metres
    # come from `move_from_goal`, and `clank.intent.SKILLS` gains nothing.
    ("pull", "reach"), ("push", "reach"), ("drag", "reach"), ("slide", "reach"),
)
_TASK_FILLER = {
    "a", "an", "the", "my", "your", "that", "this", "one", "some", "of", "off", "from",
    "here", "there", "me", "it", "please", "now",
}
_TASK_STOP = {
    "on", "in", "into", "onto", "to", "and", "then", "for", "with", "at", "so", "which",
    "before", "after", "off", "from", "that",
}
#: Words that end the object of a task because they are the *motion*, not the thing.
#: "pull it back" names no object; "pull the lever back" names the lever and stops. `left` and
#: `right` are deliberately absent - "grab the right cartridge" is a thing to grab, not a side.
_TASK_DIRECTION = frozenset({
    "back", "backward", "backwards", "forward", "forwards", "up", "down", "away",
})
#: "let's get started" is not an object. A thing the game names rarely ends in -ed or -ing.
_TASK_PARTICIPLE = re.compile(r"^[a-z]{3,}(?:ed|ing)$")
#: JobBot does not only give orders - he holds the thing out and says what it is.
_TASK_OFFER = re.compile(r"\bhere'?s my ([a-z][a-z'-]*)")


def _singular(word: str) -> str:
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def task_intent(text: str) -> tuple[str, str]:
    """The skill and the object noun the instruction itself names - not the model's guess.

    "take my card" -> ("grab", "card"); "grab one of the job cartridges off the table" ->
    ("grab", "job cartridge"); "here's my card" -> ("grab", "card"), because that is an
    offer, not a statement.

    Measured 2026-11-08: handed that first sentence and one still, the model answered
    `{"goal": "insert the card into the machine", "object": "INSERT CARTRIDGE machine"}` -
    a later step, with an invented object. Every word needed was in the instruction, so the
    verb and the noun are read here and the model is left the part it is good at: naming the
    destination when the task needs one.
    """
    tokens = re.findall(r"[a-z][a-z'\-]*", text.lower())
    for i in range(len(tokens)):
        for key, skill in _TASK_VERBS:
            parts = key.split()
            if tokens[i:i + len(parts)] != parts:
                continue
            noun: list[str] = []
            skipped_participle = False
            for word in tokens[i + len(parts):]:
                if word in _TASK_STOP or word in _TASK_DIRECTION:
                    break
                if word in _TASK_FILLER:
                    continue
                if _TASK_PARTICIPLE.match(word):
                    skipped_participle = True
                    continue
                noun.append(word)
                if len(noun) >= 3:
                    break
            if not noun:
                # "let's get started" is a verb whose object is a participle: nothing is named,
                # so there is no task and the caller's gate must see ("", ""). "take it" is
                # different - the object is a pronoun the sentence already carries - and stays a
                # grab with no noun.
                return ("", "") if skipped_participle else (skill, "")
            return skill, " ".join(noun[:-1] + [_singular(noun[-1])])
    offer = _TASK_OFFER.search(text.lower())
    if offer:
        return "grab", _singular(offer.group(1))
    return "", ""


#: How far a directional move travels, metres. **Not measured on this game's lever**: it is
#: inside the arm's measured travel (0.41-0.51 m, `research/sonic-reach-envelope.md`) and about
#: a lever's own swing, and it is the distance the model itself proposed when offered `move` as
#: a typed tool (`research/brain-tools.md` 2026-11-13: `{"direction": "back", "metres": 0.5}`).
#: Halved from that proposal, because the box's forward floor is 0.079 m and a hand already
#: reaching forward has less than half a metre of pull before it is against the torso.
MOVE_M = 0.25

#: Positional words, and the axis each names in the pelvis frame `(forward, up, lateral)`,
#: lateral positive toward the robot's own left - the order `clank.intent.Intent.offset` uses.
#: A positional word wins over the verb that carried it: "pull the lever down" travels down.
_MOVE_PLACES = (
    (("towards", "me"), (-1.0, 0.0, 0.0)),
    (("toward", "me"), (-1.0, 0.0, 0.0)),
    (("backwards",), (-1.0, 0.0, 0.0)),
    (("backward",), (-1.0, 0.0, 0.0)),
    (("back",), (-1.0, 0.0, 0.0)),
    (("forwards",), (1.0, 0.0, 0.0)),
    (("forward",), (1.0, 0.0, 0.0)),
    (("away",), (1.0, 0.0, 0.0)),
    (("down",), (0.0, -1.0, 0.0)),
    (("up",), (0.0, 1.0, 0.0)),
    (("left",), (0.0, 0.0, 1.0)),
    (("right",), (0.0, 0.0, -1.0)),
)

#: Verbs that are a direction on their own. `lift` is deliberately absent: it is a *grab* in
#: `_TASK_VERBS`, and "lift the card" must stay a grab rather than become a hand moved upward.
#: A verb with no direction word, as a unit direction. The verb alone does not say which way a
#: thing moves - a drawer comes back, a lever goes down - so these are the readings this game
#: needs, and an explicit direction word (`_MOVE_PLACES`) always wins over them.
#:
#: `pull` is **down** because the operator said so: Job Simulator's "PULL LEVER TO JOB" handle
#: swings down toward the player, and a straight-back pull on it slides the grip off the end of
#: its travel `[OPERATOR 2026-11-13]`. `drag` stays backward: dragging is horizontal by the word.
_MOVE_VERBS = (
    ("pull", (0.0, -1.0, 0.0)),
    ("drag", (-1.0, 0.0, 0.0)),
    ("push", (1.0, 0.0, 0.0)),
    ("slide", (1.0, 0.0, 0.0)),
)

#: Prepositions that make a goal a *destination* rather than a direction. Kept to exactly the
#: ones `_TASK_STOP` already treats as ending a noun phrase, so the two readings of a sentence
#: cannot disagree about where the noun stopped.
_MOVE_DESTINATION = frozenset({"on", "in", "into", "onto", "to"})


def _phrase_at(tokens: list[str], phrase: tuple[str, ...]) -> int | None:
    """Where `phrase` starts in `tokens`, or None."""
    width = len(phrase)
    for start in range(len(tokens) - width + 1):
        if tuple(tokens[start:start + width]) == phrase:
            return start
    return None


def move_from_goal(goal: str) -> tuple[float, float, float] | None:
    """The displacement a goal in words asks for, or None when it asks for no movement.

    `(forward, up, lateral)` metres, the `Intent.offset` order, already scaled by `MOVE_M`.

    This is what makes a pull a `reach` instead of a skill. Measured 2026-11-13
    (`research/brain-tools.md`): offered `move(direction, metres)` as a typed tool, this model
    answered "pull the lever to job" with `{"direction": "back", "metres": 0.5}` 3/3, so the
    vocabulary is in the weights. The prose prompt does not surface it - asked for a goal in
    words the same line became `{"goal": "job simulator"}`, with the lever gone - so the
    direction is read here, from the sentence the model did produce, and the metres stay here
    rather than in a prompt that has never been able to hold a number.

    A destination wins over a direction. "insert the cartridge into the slot" contains `in`,
    and it is a `place`: the hand goes above a surface and opens, which is a different motion
    from one that travels. The comparison is by position, so "push the cartridge in" - verb
    first, preposition later - stays a push, which is the same sentence shape read correctly.
    """
    tokens = re.findall(r"[a-z][a-z'\-]*", goal.lower())
    if not tokens:
        return None
    place: tuple[int, tuple[float, float, float]] | None = None
    for phrase, direction in _MOVE_PLACES:
        at = _phrase_at(tokens, phrase)
        if at is not None and (place is None or at < place[0]):
            place = (at, direction)
    verb: tuple[int, tuple[float, float, float]] | None = None
    for phrase, direction in _MOVE_VERBS:
        at = _phrase_at(tokens, (phrase,))
        if at is not None and (verb is None or at < verb[0]):
            verb = (at, direction)
    chosen = place if place is not None else verb
    if chosen is None:
        return None
    destination = next(
        (index for index, word in enumerate(tokens) if word in _MOVE_DESTINATION), None
    )
    if destination is not None:
        earliest = min(candidate[0] for candidate in (place, verb) if candidate is not None)
        if destination < earliest:
            return None
    forward, up, lateral = chosen[1]
    return (forward * MOVE_M, up * MOVE_M, lateral * MOVE_M)


def skill_from_goal(goal: str) -> str:
    """The intent skill a goal in words implies. `reach` when nothing matches."""
    words = goal.lower()
    for skill, keys in _SKILL_WORDS:
        if any(key in words for key in keys):
            return skill
    return "reach"


@dataclass(frozen=True)
class NextStep:
    """What the body should do next, decided from the scene rather than parsed.

    `target` is the noun the aimer will resolve: the **destination** for `place`,
    the object for everything else. That split is the whole point - `place` given
    "cartridge" cannot move anywhere, while `place` given "the cartridge slot" can.
    """

    goal: str = ""
    skill: str | None = None
    target: str | None = None
    side: str | None = None
    object: str | None = None
    destination: str | None = None
    raw: str = ""
    error: str = ""
    prompt_ms: float = 0.0
    decode_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.error and self.skill is not None and bool(self.target)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "skill": self.skill,
            "target": self.target,
            "side": self.side,
            "object": self.object,
            "destination": self.destination,
            "raw": self.raw,
            "error": self.error,
            "prompt_ms": round(self.prompt_ms, 1),
            "decode_ms": round(self.decode_ms, 1),
        }


def next_step(
    config: BrainConfig,
    frame: Frame,
    situation: str,
    default_side: str = "right",
    *,
    session: SessionMemory | None = None,
    heard: str = "",
) -> NextStep:
    """Ask what to do next, given the situation in words and one still.

    This is the autonomous path: no instruction is parsed, because in this game
    there is often none - JobBot says "take a cartridge" and the rest is expected
    to be understood. Measured 2026-11-04: given that situation and a still, the
    model answers `{"goal": "insert the cartridge into the game console",
    "destination": "INSERT CARTRIDGE screen on the central console"}`, which is a
    destination `locate` can resolve. The same question with a skill enum in the
    prompt degenerated to `{"skill": "grab"}` - hence the split.

    `session` is the run's transcript. When it is given the call carries every earlier utterance,
    prompt, reply and result, and this exchange is appended to it - one owner for the prompt, the
    call and the memory, so the trace and the history cannot disagree. `None` is the single user
    message this has always sent.
    """
    prompt = NEXT_STEP_PROMPT.format(situation=situation)
    if session is None:
        messages = [{"role": "user", "content": [
            _image_part(frame), {"type": "text", "text": prompt},
        ]}]
    else:
        messages = session.for_chat(frame, prompt)
    reply = _chat(config, messages)
    if session is not None and reply.text:
        # A failed call has no reply to remember, and an empty assistant message is not a turn.
        # The SITUATION, not the whole prompt: the instruction block is ~72 of the ~123 tokens a
        # stored pair costs, and repeating "You are a robot..." and "Obey the newest thing" forty
        # times is the class of text that overrode the newest utterance once already. The exact
        # prompt of every call is in `llm.jsonl`, so nothing is lost from the record.
        session.remember_turn(situation, reply.text)
    elif session is not None and heard:
        # A failed call still happened to a sentence: keep the sentence, so the next call reads
        # it instead of the run losing an instruction to one bad reply.
        session.note_heard(heard)
    if reply.error:
        return NextStep(raw=reply.text, error=reply.error)

    parsed = extract_json_object(reply.text)
    if parsed is None:
        return NextStep(raw=reply.text, error="reply contained no JSON object",
                        prompt_ms=reply.prompt_ms, decode_ms=reply.decode_ms)

    goal = str(parsed.get("goal") or "").strip()
    obj = str(parsed.get("object") or "").strip()
    dest = str(parsed.get("destination") or "").strip()
    if dest.lower() in ("left", "right"):
        # MEASURED 2026-11-13: the model fills `destination` with a hand on a grab
        # (`"destination": "right"`). Harmless there, but a `place` would aim at a hand, so a side
        # word is no destination at all and falls into the existing "place with no destination".
        dest = ""
    side = str(parsed.get("side") or "").strip().lower()
    if side not in ("left", "right"):
        # Measured once: the reply said "center". A hand must be chosen, and the
        # right hand is the one every reach today used.
        side = default_side

    skill = skill_from_goal(goal)
    if skill == "place":
        if dest.lower() in ("", "none", "null"):
            return NextStep(goal=goal, skill=None, side=side, destination=dest,
                            raw=reply.text, error="place with no destination named",
                            prompt_ms=reply.prompt_ms, decode_ms=reply.decode_ms)
        target = dest
    else:
        target = obj or dest
    if not target:
        return NextStep(goal=goal, skill=None, side=side,
                        raw=reply.text, error="no target named",
                        prompt_ms=reply.prompt_ms, decode_ms=reply.decode_ms)
    return NextStep(goal=goal, skill=skill, target=target, side=side, object=obj or None,
                    destination=dest, raw=reply.text,
                    prompt_ms=reply.prompt_ms, decode_ms=reply.decode_ms)


def sequence(
    step: NextStep, holding: bool, object_noun: str = ""
) -> list[tuple[str, str]]:
    """The ordered steps the executor must run to realise `step`.

    **Measured 2026-11-04: the model plans the goal, never the next step.** Told
    "the cartridge tray is on the counter, your hands are empty", it answered
    `{"goal": "insert cartridge", "destination": "INSERT CARTRIDGE"}` - a `place`
    with nothing in the hand - and it did so under every phrasing tried, including
    one that said "decide the immediate next action only, one step". Asking a model
    to respect a precondition it cannot see is not something to fix with words.

    So ordering is code: a `place` with an empty hand is a `grab` of the object the
    goal names, then the `place`. Whether the hand is empty is the caller's to know
    - from the grasp check, not from the model's opinion.

    **`object_noun` comes from the task, not from the model.** Measured the same
    session: with empty hands and a prompt asking what it must pick up first, it
    answered "empty hands", then echoed the field's own description back, then
    invented "the white cartridge pusher/insert tool". A model that cannot see a
    hand full cannot say what belongs in it. The task line already names the object
    ("take a cartridge"), and this function takes it from there.
    """
    if step.skill is None:
        return []
    if step.skill == "place" and not holding:
        obj = object_noun or step.object or step.target or ""
        if not obj:
            # A place with neither an object nor a destination cannot be sequenced.
            return []
        return [("grab", obj), ("place", step.target or "")]
    return [(step.skill, step.target or "")]


# --------------------------------------------------------------------------- #
# GROUND
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Ground:
    """Where the named noun is: one quadrant of the container box, and a pixel.

    `pixel` is the geometric centre of that quadrant inside `box`, in full-frame
    `[0, 1]`. It is not a per-cartridge calibrated point: spec §10 records that an
    uncalibrated quadrant centre misses GOURMET CHEF by 0.097, and the precision
    path is the wrist-crop VERIFY (§11), not a game-specific offset table here.
    """

    noun: str
    quadrant: str | None = None
    box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    pixel: tuple[float, float] | None = None
    raw: str = ""
    error: str = ""
    methods: str = "quad"
    prompt_ms: float = 0.0
    decode_ms: float = 0.0

    @property
    def cell(self) -> str | None:
        """Wire alias for `quadrant`; the `cell` key stays in `to_dict()` and is
        pinned by tests so older traces still read."""
        return self.quadrant

    def to_dict(self) -> dict[str, Any]:
        return {
            "noun": self.noun,
            "quadrant": self.quadrant,
            "cell": self.cell,
            "box": list(self.box),
            "pixel": list(self.pixel) if self.pixel else None,
            "raw": self.raw,
            "error": self.error,
            "methods": self.methods,
            "prompt_ms": round(self.prompt_ms, 1),
            "decode_ms": round(self.decode_ms, 1),
        }


def compose_pixel(
    box: tuple[float, float, float, float], quadrant: str
) -> tuple[float, float]:
    """Geometric centre of `quadrant` inside `box`, in full-frame `[0, 1]`.

    A pure function of the box and the answer: no per-cartridge offsets, because
    an offset table is game layout inside the brain and spec §10 budgets the
    wrist-crop VERIFY as the precision path instead.
    """
    u0, v0, u1, v1 = box
    u = u0 + (u1 - u0) * ((1 if "RIGHT" in quadrant else 0) + 0.5) / 2
    v = v0 + (v1 - v0) * ((1 if "BOTTOM" in quadrant else 0) + 0.5) / 2
    return (u, v)


def _match_quadrant(text: str) -> str | None:
    """The first `QUADRANTS` answer in a reply, or None for absent/unparseable.

    The reply is stripped to capitals and single spaces and matched as whole
    words, so `TOP-RIGHT`, `TOP RIGHT` and `top, right!` are the same answer.
    `NONE`, `NOT VISIBLE`, `ABSENT` and plain chatter all return None: presence
    detection is unbuilt, so the caller treats an unreadable answer the same way
    it treats a refusal, never as a guessed pixel.
    """
    words = " ".join(re.sub(r"[^A-Z ]", " ", text.upper()).split())
    padded = f" {words} "
    if any(f" {absent} " in padded for absent in _ABSENT_WORDS):
        return None
    for quadrant in QUADRANTS:
        if f" {quadrant.replace('-', ' ')} " in padded:
            return quadrant
    return None


def ground(
    config: BrainConfig,
    frame: Frame,
    noun: str,
    box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    container: str = "tray",
    transcript: list[str] | None = None,
) -> Ground:
    """Ask which quadrant of `container` holds `noun`, then compose a pixel.

    Spec §10. The 3x3 image-global method this replaced is retired: measured
    2026-11-01, thirds scored 3/12, a 9-cell single pick 0/12 and a 4x4 grid 0/12,
    all collapsing toward the image centre. One name->quadrant question asked
    about a crop of the tray scored 44/44 on the same box, and the identical
    prompt on the full frame 9/12 - the crop is the whole difference.

    `box` is that crop, normalised `(u0, v0, u1, v1)`. **`frame` must be the crop
    of `box`** - make it with `frames.crop_jpeg`; `box` only composes the answer
    back into full-frame coordinates. Handing a `box` to the whole still is wrong
    by construction and nothing here can catch it: the pixel lands inside the box
    but was derived from the entire image, which is the 9/12 measurement, not the
    44/44 one. Finding the box automatically is unbuilt; it is the caller's to
    supply.

    `transcript` is accepted so the call sites do not fork, but it is deliberately
    **not** in the prompt: the 44/44 measurement is the bare question, and speech
    is a second context channel nobody measured for this question. The prompt also
    never lists the other cartridges or their positions - measured, the model
    follows a supplied layout table 12/12 even when the table is deliberately
    false.
    """
    prompt = (
        f"Which position in this {container} does the {noun} occupy? "
        "Answer with one of: TOP-LEFT, TOP-RIGHT, BOTTOM-LEFT, BOTTOM-RIGHT."
    )
    reply = _chat(config, [{"role": "user", "content": [
        _image_part(frame), {"type": "text", "text": prompt},
    ]}], max_tokens=60)
    if reply.error:
        return Ground(noun=noun, box=box, raw=reply.text, error=reply.error)

    quadrant = _match_quadrant(reply.text)
    if quadrant is None:
        # NONE, ABSENT, or chatter we cannot read. Presence detection is unbuilt,
        # so an unreadable answer is treated as absence, never as a guessed pixel.
        return Ground(noun=noun, box=box, raw=reply.text.strip(), error="not visible",
                      prompt_ms=reply.prompt_ms, decode_ms=reply.decode_ms)

    return Ground(
        noun=noun,
        quadrant=quadrant,
        box=box,
        pixel=compose_pixel(box, quadrant),
        raw=reply.text.strip(),
        prompt_ms=reply.prompt_ms,
        decode_ms=reply.decode_ms,
    )


# --------------------------------------------------------------------------- #
# VERIFY
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Verify:
    """Does the crop the game just highlighted actually show `noun`?

    `match` is a word-subset test with a refusal guard, not string equality: the
    crop question comes back as a sentence ("the store clerk cartridge") and the
    planner's noun is a label ("STORE CLERK"). Extra words are allowed; a missing
    noun word is a mismatch, and so is any negation word, so "not visible" and
    "no card here" cannot pass as agreement.

    This is the grab gate, not a hint (spec §11): a wrong name must block the
    grab rather than be averaged away, and a false match closes the hand on the
    wrong object while a false mismatch only costs a turn - so the gate fails
    closed. It is never an exception: `NONE`, an empty reply, chatter and a
    transport error all arrive as `match=False`, with the transport message kept
    in `error`.

    `heard` is the stripped reply the gate compared and `raw` is that same
    stripped reply, the form `Ground` keeps.
    """

    noun: str
    heard: str = ""
    match: bool = False
    raw: str = ""
    error: str = ""
    prompt_ms: float = 0.0
    decode_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "noun": self.noun,
            "heard": self.heard,
            "match": self.match,
            "raw": self.raw,
            "error": self.error,
            "prompt_ms": round(self.prompt_ms, 1),
            "decode_ms": round(self.decode_ms, 1),
        }


#: Words that turn a reply into a refusal. `_names_the_noun` fails closed on any
#: of them, the same way `_match_quadrant` treats `_ABSENT_WORDS`.
_NEGATION_WORDS = frozenset({"not", "no", "none", "absent", "never"})


def _names_the_noun(noun: str, heard: str) -> bool:
    """True when `heard` carries every alphabetic word of `noun`, and no refusal.

    Case and punctuation are stripped on both sides, so `"store clerk"` matches
    `"It's the Store-Clerk cartridge."`. Words are compared whole: `"card"` is
    not found in `"discard"`, and `"art"` is not found in `"cartridge"`.

    Any negation word makes the reply a refusal rather than a name, so a reply
    that holds every noun word still mismatches: `"the store clerk is not
    visible"`, `"no card here"`. The gate decides a grab, so it fails closed. A
    noun with no alphabetic word cannot be confirmed against anything, so it
    never matches.
    """
    wanted = set(re.findall(r"[a-z]+", noun.lower()))
    if not wanted:
        return False
    words = set(re.findall(r"[a-z]+", heard.lower()))
    return wanted <= words and not (words & _NEGATION_WORDS)


def verify(config: BrainConfig, frame: Frame, noun: str) -> Verify:
    """Ask what one crop shows, and say whether it is `noun`.

    Spec §11. GROUND's quadrant says *roughly where*: the tray's four cartridges
    sit 0.073 apart normalised and the quadrant centres are only ~0.13 apart, so
    the quadrant cannot separate them. After the game highlights one object
    (`glow=yes`), this asks the model to name that one object instead of a
    position, and the caller grabs only on `match`.

    `frame` is the crop, not the full still: the same crop-is-the-difference
    measurement that made GROUND ask about a box applies here. Cropping is the
    caller's arithmetic (`crop_jpeg`), not image work in the brain.
    """
    prompt = "This is a close-up of one object. What is it? Reply with the name, or NONE."
    reply = _chat(config, [{"role": "user", "content": [
        _image_part(frame), {"type": "text", "text": prompt},
    ]}], max_tokens=40)
    if reply.error:
        return Verify(noun=noun, raw=reply.text.strip(), error=reply.error,
                      prompt_ms=reply.prompt_ms, decode_ms=reply.decode_ms)

    heard = reply.text.strip()
    return Verify(
        noun=noun,
        heard=heard,
        match=_names_the_noun(noun, heard),
        raw=heard,
        prompt_ms=reply.prompt_ms,
        decode_ms=reply.decode_ms,
    )


# --------------------------------------------------------------------------- #
# WORKING MEMORY
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Sighting:
    """One thing the sense tools have found, in code, for this turn only.

    The pixel lives here and nowhere else. Measured 2026-11-01 and again on 2026-11-08: a number in
    a prompt is something the model will repeat back as if it had measured it, so the live block is
    told `present` / `not_visible` / `too_far` and the coordinates stay in Python until an Intent
    needs them.
    """

    noun: str
    present: bool
    pixel: tuple[float, float] | None = None      # full-frame [0, 1], as Intent wants
    pixel_px: tuple[float, float] | None = None   # the frame's own pixels, as locate reports
    metres: float | None = None
    outcome: str = ""                             # present | not_visible | too_far | error


@dataclass
class WorkingMemory:
    """What this turn has looked at, keyed by the noun the planner used.

    Lifetime is one turn on purpose: the head moves between turns, so a pixel from the last turn is
    a guess. Matching is deliberately loose in one direction only - `card` finds `JobBot card`,
    because the planner says "card" and locate needs something to search for - and a miss is
    `None` rather than a nearest key, because a near miss here means the arm reaches for the wrong
    object.
    """

    sightings: dict[str, Sighting] = field(default_factory=dict)

    def remember(self, noun: str, sighting: Sighting) -> None:
        self.sightings[noun.strip().lower()] = sighting

    def find(self, noun: str) -> Sighting | None:
        key = noun.strip().lower()
        if not key:
            return None
        if key in self.sightings:
            return self.sightings[key]
        for stored, sighting in self.sightings.items():
            if key in stored or stored in key:
                return sighting
        return None


# --------------------------------------------------------------------------- #
# SESSION MEMORY
# --------------------------------------------------------------------------- #

#: What a heard line starts with. `SessionMemory.cap` reads it to keep every utterance: the run's
#: speech is the one thing that must survive a trim, because it is the record of what was asked.
HEARD_PREFIX = "This is what was heard: "

#: What a result line starts with. Results are trimmed first: they are the most replaceable, and
#: the next `last:` line says the same thing.
RESULT_PREFIX = "result: "

#: What the run may spend on history, in estimated text tokens.
#:
#: The binding limit is the server's, not ours: `/props` reports `total_slots: 4` for a 65536
#: context, so a single request gets ~16384 tokens, and an oversize request is refused outright -
#: a blind planner rather than a slow one. A stored pair is ~400 chars of situation and ~90 of
#: reply, ~123 tokens by the estimate below, so 80 pairs is ~10k estimated (~13k real for
#: JSON-heavy text) against a ~16k slot with the current prompt and output still to fit.
#:
#: 8000 of history keeps a request near 9k: a safe margin, and ~1.5 s of prefill at the ~5400
#: prompt tokens/s measured against this endpoint (`research/brain-tools.md`). The pair count is
#: the loose bound and the budget is the one that trips.
MEMORY_TOKEN_BUDGET = 8000


@dataclass
class SessionMemory:
    """The run as one conversation: what was heard, asked, answered and done.

    Spec 2026-11-13 §3.3. The lifetime is the run, not the turn: every planner call sends this
    whole list, so the newest utterance is read against everything said before it. The current
    still is the only image and it always rides the newest user message - a pixel from last turn is
    a guess (spec §3.3; `WorkingMemory` is the per-turn, geometric half of the same idea).

    Text only, in the OpenAI chat-message form `_chat` posts unchanged. Locate calls are one-shot
    and never read this: they answer a different question and would drown the planner in crops.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    heard: list[str] = field(default_factory=list)

    def note_heard(self, text: str) -> None:
        """Record one transcribed utterance. An empty transcription is not an utterance."""
        text = text.strip()
        # Two callers can report the same utterance - `think` remembers a greeting it did not act
        # on, and the loop remembers what it heard - so the repeat is not a second thing said.
        if not text or (self.heard and self.heard[-1] == text):
            return
        self.heard.append(text)
        self.messages.append({"role": "user", "content": f'{HEARD_PREFIX}"{text}"'})

    def note_result(self, text: str) -> None:
        """Record what the loop did: `result: locate(card) present 0.4 m`."""
        text = text.strip()
        if text:
            self.messages.append({"role": "user", "content": f"{RESULT_PREFIX}{text}"})

    def remember_turn(self, user_text: str, assistant_text: str) -> None:
        """Keep one planner exchange: the prompt as it was sent, and the raw reply."""
        self.messages.append({"role": "user", "content": user_text})
        self.messages.append({"role": "assistant", "content": assistant_text})

    def for_chat(self, frame: Frame, prompt_text: str) -> list[dict[str, Any]]:
        """`messages` as the next planner call sends them: history, then the still and question.

        `cap()` runs here, at the one place every send goes through. A caller that forgets to trim
        would otherwise grow the history until the server refuses the request - and a refused
        request is a blind planner, not a slow one.
        """
        self.cap()
        return [
            *self.messages,
            {"role": "user", "content": [
                _image_part(frame), {"type": "text", "text": prompt_text},
            ]},
        ]

    def cap(self, max_pairs: int = 80, max_tokens: int = MEMORY_TOKEN_BUDGET) -> None:
        """Trim to `max_pairs` exchanges and `max_tokens` of prompt, newest kept, heard kept.

        Results go first: they are the most replaceable (the next `last:` line says the same
        thing), and a planner pair goes only when no result is left. A heard line is never a
        candidate - losing the run's speech is exactly the failure this class prevents.

        The token budget is the one that binds in practice. A planner prompt measures 692-731
        tokens on the live endpoint, so 80 pairs is ~59k tokens of prefill: at the measured 5400
        prompt tokens/s that is 11 s per call, every call, which would make the loop slower than
        the reach it is asking for. `MEMORY_TOKEN_BUDGET` is sized to keep prefill near 1.5 s.
        """
        while self._pairs() > max_pairs or self._tokens() > max_tokens:
            index = self._oldest_result()
            if index is not None:
                del self.messages[index]
                continue
            index = self._oldest_planner_prompt()
            if index is None:
                break
            del self.messages[index:index + 2]

    def _tokens(self) -> int:
        """A deliberately cheap estimate: 4 characters per token, no tokeniser in `clank/`.

        It measures the stored history only - not the current prompt and not the still, which are
        another ~700 tokens - and 4 chars/token is optimistic for the JSON-heavy replies, where a
        real tokeniser lands nearer 3. Both errors point the same way, which is why the budget
        leaves a wide margin rather than sitting just under the slot.
        """
        return sum(len(self._text(message)) for message in self.messages) // 4

    def _pairs(self) -> int:
        return sum(1 for message in self.messages if message["role"] == "assistant")

    @staticmethod
    def _text(message: dict[str, Any]) -> str:
        content = message.get("content")
        return content if isinstance(content, str) else ""

    def _oldest_result(self) -> int | None:
        for index, message in enumerate(self.messages):
            if message["role"] != "user":
                continue
            if self._text(message).startswith(RESULT_PREFIX):
                return index
        return None

    def _oldest_planner_prompt(self) -> int | None:
        """The oldest user message with an assistant reply right after it.

        Structural, not prefix-based: a stored prompt is the situation (`heard: ... / self: ... /
        last: ...`), which has no fixed opening, and matching on prefixes is how a trimmer ends up
        unable to delete anything. Nothing else appends an assistant message, so "user followed by
        assistant" is exactly a planner exchange.
        """
        for index, message in enumerate(self.messages[:-1]):
            if message["role"] == "user" and self.messages[index + 1]["role"] == "assistant":
                return index
        return None


# --------------------------------------------------------------------------- #
# ASSEMBLE
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Decision:
    """The intent to send, plus why the planner's own action did not survive.

    `intent` is **always** the intent to send, including the `hold` or `look`
    fallback chosen when the plan is refused: a refusal is still an action, and
    dropping it would leave the body without a command. `accepted` records only
    whether the planner's action survived unmodified, and is **never** a gate on
    whether `intent` goes to the adapter.
    """

    intent: Intent
    reason: str = ""
    note: str = ""

    @property
    def accepted(self) -> bool:
        return not self.reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.to_dict(),
            "reason": self.reason,
            "note": self.note,
            "accepted": self.accepted,
        }


def assemble(
    plan_result: Plan,
    ground_result: Ground | None,
    seq: int,
    t_mono_ns: int,
    look_step: int = 1,
    left: str = "empty",
    right: str = "empty",
    glow: str = "unknown",
    ghost: str = "unknown",
    memory: WorkingMemory | None = None,
    offset: tuple[float, float, float] | None = None,
) -> Decision:
    """Turn a plan plus its grounding into an `Intent`, or explain the refusal.

    Refusals are values, not exceptions: the caller holds and shows the reason.

    The occupancy guards are the one place this writer checks a plan against the
    body, and they earn their keep: measured 2026-11-01, the text path answered
    `release right` to "take my card" while the right hand was empty. That is not
    a schema error, so `validate_intent` cannot catch it, and executing it would
    burn a turn to change nothing.

    The cue gates add the same kind of check for Job Simulator's two lights:
    `glow` is blue proximity at the wrist and gates `grab`, `ghost` is the yellow
    insert cue at the slot and gates a `release` whose target names a slot. Both
    are context a planner reads, never a trigger - `glow=yes` alone never emits
    a grab - and `unknown` never refuses, because the replay writer carries no
    game cues and an uncalibrated detector must not freeze the hands.
    """
    if not plan_result.ok:
        return Decision(Intent(skill="hold", side="none", source="vlm", seq=seq,
                               t_mono_ns=t_mono_ns),
                        reason=plan_result.error or "no plan")

    skill = plan_result.skill or "hold"
    side = plan_result.side or "none"
    target = plan_result.target
    pixel: tuple[float, float] | None = None
    note = plan_result.note
    yaw_rad = None

    # `locate` is a sense tool: the caller runs it and stores the Sighting. Emitting an Intent for
    # it would be a lie in two directions - `clank.intent.SKILLS` has no such skill, and the body
    # would be told to hold while the caller measures. The freeze-hold below is honest: nothing
    # moves this think, and `note` says why.
    if skill in SENSE_SKILLS:
        return Decision(Intent(skill="hold", side="none", source="vlm", seq=seq,
                               t_mono_ns=t_mono_ns),
                        note="locate")

    # A reach or grab at a thing nobody has located is refused, not silently located: the model is
    # told `not_located` in its last line and is expected to choose `locate` next. Auto-locating
    # here would put the choreography back in Python, which is the bug this whole slice fixes.
    sighting = memory.find(target) if (memory is not None and target) else None
    # A travel is not a look-and-find: the thing being acted on is already in the hand, so there
    # is nothing to ground and no pixel to wait for. Without this the pull is refused as
    # `not_located` and the lever is never pulled - the exact failure that kept `reach` from
    # being usable on a held object.
    if (memory is not None and target and skill in ("reach", "grab") and offset is None
            and (sighting is None or not sighting.present)):
        why = "never looked at it" if sighting is None else f"last look said {sighting.outcome or 'absent'}"
        return Decision(Intent(skill="hold", side="none", source="vlm", seq=seq,
                               t_mono_ns=t_mono_ns),
                        reason=f"not_located: {target!r} - {why}", note=note)

    hand = {"left": left, "right": right}.get(side)
    if skill == "release" and hand == "empty":
        return Decision(Intent(skill="hold", side="none", source="vlm", seq=seq,
                               t_mono_ns=t_mono_ns),
                        reason=f"release {side} refused: that hand is empty", note=note)
    if skill == "grab" and hand == "holding":
        return Decision(Intent(skill="hold", side="none", source="vlm", seq=seq,
                               t_mono_ns=t_mono_ns),
                        reason=f"grab {side} refused: that hand is already holding",
                        note=note)

    if skill == "grab" and glow == "no":
        return Decision(Intent(skill="hold", side="none", source="vlm", seq=seq,
                               t_mono_ns=t_mono_ns),
                        reason=f"grab {side} refused: glow is no", note=note)
    if skill == "release" and "slot" in (target or "").lower() and ghost == "no":
        return Decision(Intent(skill="hold", side="none", source="vlm", seq=seq,
                               t_mono_ns=t_mono_ns),
                        reason=f"release {side} refused: ghost is no", note=note)

    needs_where = skill in ("reach", "look") and target is not None and offset is None
    if needs_where:
        if sighting is not None and sighting.present and sighting.pixel is not None:
            pixel = sighting.pixel
            note = f"located {sighting.noun!r}"
        elif ground_result is None or ground_result.pixel is None:
            reason = (ground_result.error if ground_result and ground_result.error
                      else "target not grounded")
            # A named thing we cannot see is not a reason to guess a pixel; the
            # constitution already told the model to look when it cannot see.
            # Accepted deviation, carried to the socket work: a GROUND *transport*
            # error lands here too and yaws instead of freeze-holding (spec
            # §11/§12). There is no motion to gate in this slice, and rewiring
            # this path late risks the guard the whole design leans on.
            fallback = Intent(skill="look", side="none", source="vlm", seq=seq,
                              t_mono_ns=t_mono_ns,
                              yaw_rad=LOOK_ARC_RAD * (1 if look_step % 2 else -1))
            return Decision(fallback, reason=f"{reason}: {target!r}", note="downgraded to scan")
        else:
            pixel = ground_result.pixel

    if skill == "grab" and target is not None:
        if sighting is not None and sighting.present:
            pixel = sighting.pixel
            note = f"located {sighting.noun!r}"
        elif memory is not None:
            # Unreachable in practice: the not_located guard above already refused. Kept so the
            # legacy path below cannot quietly take over when a memory is passed.
            return Decision(Intent(skill="hold", side="none", source="vlm", seq=seq,
                                   t_mono_ns=t_mono_ns),
                            reason=f"grab {target!r} but it is not located")
        elif ground_result is None or ground_result.pixel is None:
            return Decision(Intent(skill="hold", side="none", source="vlm", seq=seq,
                                   t_mono_ns=t_mono_ns),
                            reason=f"grab {target!r} but it is not visible")
        else:
            pixel = ground_result.pixel
            note = "grab does not need a pixel; target kept for verify"

    if skill == "look" and target is None:
        yaw_rad = LOOK_ARC_RAD * (1 if look_step % 2 else -1)
        note = "scan"

    intent = Intent(
        skill=skill,
        side=side,
        source="vlm",
        seq=seq,
        t_mono_ns=t_mono_ns,
        pixel=pixel,
        yaw_rad=yaw_rad,
        offset=offset,
    )
    reason = validate_intent(intent)
    if reason and skill in ("look", "reach", "grab", "release"):
        return Decision(Intent(skill="hold", side="none", source="vlm", seq=seq,
                               t_mono_ns=t_mono_ns),
                        reason=reason, note="held after rejection")
    return Decision(intent, reason=reason, note=note)


# --------------------------------------------------------------------------- #
# THINK
# --------------------------------------------------------------------------- #


@dataclass
class Thought:
    """One chooser tick: what the model said, what that became, and what to run.

    `sense` is the only place a sense tool survives assembly, because a sense tool has no Intent.
    The caller looks at it and runs `locate.py`; the model never sees a pixel and `clank/` never
    imports the locator, which keeps this package standard-library only.
    """

    plan: Plan
    decision: Decision
    sense: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "decision": self.decision.to_dict(),
            "sense": self.sense,
        }


def think(
    config: BrainConfig,
    frame: Frame,
    heard: str,
    memory: WorkingMemory,
    *,
    session: SessionMemory | None = None,
    seq: int = 0,
    t_mono_ns: int = 0,
    left: str = "empty",
    right: str = "empty",
    yaw_rad: float = 0.0,
    last: str = "(none)",
    default_side: str = "right",
    look_step: int = 1,
) -> Thought:
    """One chooser tick: PLAN, then ASSEMBLE. Never grounds, never locates.

    Transcribe stays a separate call. Measured twice: one call that asked for the words and a
    decision together produced a decision that ignored the words, so the audio never reaches PLAN
    on the live path.

    No GROUND either. The live path's grounding is `locate.py` in the spike, chosen by the model as
    its own tool - which is the whole point of the sense/motor split: if this function grounded the
    target for it, the model's `locate` would be decoration.

    No seed as well. The seed was the largest measured cause of a wrong verb (see `DEFAULT_SEED`),
    and nothing here reads it.

    `session` is the run's memory (spec 2026-11-13 §3.3): when given, the planner call carries the
    whole transcript and the exchange is appended to it. `None` keeps the old one-shot behaviour
    for callers that do not own a run.
    """
    # Whether a task exists is not the same question as what the task is. MEASURED 2026-11-13:
    # the greeting plus a still produced `place(console)` because the prompt demanded a goal, so an
    # utterance that names no verb and holds nothing out stops here - before the model is paid for.
    # Nothing heard at all is a different fact and gets its own note: a deaf run (a muted sink
    # input, a game sitting at its menu) must not look like a greeting in the diary.
    if not heard.strip() or task_intent(heard) == ("", ""):
        if session is not None:
            # A greeting is still something JobBot said, and the next call must read it. Nothing
            # heard has nothing to remember: `note_heard` ignores an empty utterance.
            session.note_heard(heard)
        idle = Plan(skill="hold", side="none",
                    note="nothing asked yet" if heard.strip() else "nothing heard yet")
        decision = assemble(idle, None, seq, t_mono_ns, left=left, right=right,
                            memory=memory, look_step=look_step)
        return Thought(plan=idle, decision=decision)

    live = live_block(left, right, yaw_rad, last)
    # The question shape is the measurement. MEASURED 2026-11-10, four live stills, the same
    # instruction: asked to pick from the action list this model answered `hold` or `release` on
    # 4 of 4 and never `grab`, while asked what it is trying to do it answered "take the JobBot
    # card" on 4 of 4. So the model states the goal in its own words and `skill_from_goal` reads
    # the body's vocabulary out of *the model's* sentence - not out of JobBot's, which is what
    # made the old loop a parser.
    said = f'This is what was heard: "{heard.strip()}"'
    # The live block travels with the situation: which hands are holding, and what the last action
    # was told, are exactly the context a choice needs, and NEXT_STEP_PROMPT has a slot for it.
    # Without this the model reasons about the scene with no memory of the last outcome.
    situation = f"{said}\n{live}"
    step = next_step(config, frame, situation, default_side=default_side,
                     session=session, heard=heard)
    if step.error:
        plan_result = Plan(raw=step.raw, error=step.error, prompt_ms=step.prompt_ms,
                           decode_ms=step.decode_ms)
    else:
        # The object noun comes from the model's own sentence, not from its `object` field:
        # MEASURED 2026-11-10 the goal was "take the JobBot card" while `object` said "JobBot",
        # and locate searched for a summoner rather than for the thing he is holding. `place`
        # keeps the destination, because a cartridge cannot be put into an object.
        _, noun = task_intent(step.goal)
        target = step.target
        if step.skill != "place" and noun:
            target = noun
        plan_result = Plan(
            skill=step.skill,
            side=step.side if step.skill in PLANNER_SIDE_SKILLS else "none",
            target=target,
            raw=step.raw,
            note=f"goal: {step.goal}",
            prompt_ms=step.prompt_ms,
            decode_ms=step.decode_ms,
        )
    # A direction in the instruction, with the acting hand already closed on something, is a
    # travel - the one motion `clank.intent/1` could not express. The occupancy half is not
    # decoration: "pull the lever to job" arrives with an empty hand on the first turn and must
    # *reach* the lever then, not move away from it. Read from `heard` rather than from the
    # model's sentence, because measured 2026-11-13 the same line through the prose prompt came
    # back as `{"goal": "job simulator"}` - the verb and its noun were both lost.
    holding = right if plan_result.side == "right" else left
    travel = (
        move_from_goal(heard)
        if holding == "holding" and (plan_result.skill or "") in ("reach", "grab", "place")
        else None
    )
    decision = assemble(plan_result, None, seq, t_mono_ns, left=left, right=right,
                        memory=memory, look_step=look_step, offset=travel)
    sense = None
    if plan_result.ok and plan_result.skill in SENSE_SKILLS:
        sense = plan_result.skill
    return Thought(plan=plan_result, decision=decision, sense=sense)


# --------------------------------------------------------------------------- #
# trace
# --------------------------------------------------------------------------- #


@dataclass
class TurnRecord:
    """Everything one turn saw, said, was told, and decided.

    This exists so a human can audit the brain instead of trusting it. The
    dashboard renders it; `to_dict` is the wire form.
    """

    turn: int
    frame_index: int
    t_s: float
    heard: str
    seed: str
    live: str
    plan: Plan
    ground: Ground | None
    decision: Decision
    verify: Verify | None = None
    self_state: dict[str, Any] = field(default_factory=dict)
    outcome: str = ""
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "frame_index": self.frame_index,
            "t_s": round(self.t_s, 2),
            "heard": self.heard,
            "seed": self.seed,
            "live": self.live,
            "self": self.self_state,
            "plan": self.plan.to_dict(),
            "ground": self.ground.to_dict() if self.ground else None,
            "verify": self.verify.to_dict() if self.verify else None,
            "decision": self.decision.to_dict(),
            "outcome": self.outcome,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


def run_turn(
    config: BrainConfig,
    seed: str,
    frame: Frame,
    heard: str,
    history: list[str],
    *,
    seq: int = 0,
    left: str = "empty",
    right: str = "empty",
    yaw_rad: float = 0.0,
    glow: str = "unknown",
    ghost: str = "unknown",
    box: tuple[float, float, float, float] | None = None,
    container: str = "tray",
    last: str = "(none)",
    last_action: tuple[str, str, str | None] | None = None,
    state_flags: int = 0,
    wav: bytes | None = None,
    turn: int = 0,
    look_step: int = 1,
    elapsed_ms: float = 0.0,
) -> TurnRecord:
    """One turn: PLAN on the whole still, then crop, GROUND, VERIFY, ASSEMBLE.

    PLAN sees the full frame - choosing a skill is a scene question and nothing
    measured says a crop helps it. GROUND is asked about the crop of `box`: the
    same name->quadrant question scored 44/44 on a tray crop and 9/12 on the full
    frame (§10), so `box` is the difference between the live 9/12 and the measured
    crop number. With `box=None` the whole frame goes out and the answer composes
    over `(0, 0, 1, 1)` - the 9/12 path, unchanged.

    A crop that cannot be cut is **not** a failed turn. `crop_jpeg` raising (no
    ffmpeg, a malformed box) falls back to grounding the whole frame and records
    the reason in `self_state["crop_error"]`, because losing the crop costs
    accuracy while raising would cost the turn.

    **Known simplification (spec §11):** VERIFY is specified as a crop around the
    projected wrist, and what goes out here is the *container* crop. Projecting the
    wrist needs adapter state - forward kinematics and the frozen head pose - which
    `clank/` does not have. So the grab gate asks about the container: at
    `glow=yes` in grab range the container is what fills that crop (§10 step 4),
    and the check is still a name check against `target`, never a position. When
    the adapter exposes the wrist, this is the call that changes. A *container*
    crop always contains the target, so a `match` proves "the tray holds STORE
    CLERK", not "the hand is on STORE CLERK": spec §11's wrist crop remains
    unproven and this is the gate's one false-allow path.

    VERIFY runs only when the plan is a grab, `glow` says the hand is in range, and
    a `target` exists to match (§11). On a mismatch the ground keeps its quadrant,
    `raw` and timings; only its pixel is cleared and its `error` is marked, so the
    existing not-visible path refuses the grab: a wrong name must block the hand,
    never be averaged into a position. The verdict itself, including the reply the
    gate saw and any transport error, is recorded on the turn's `Verify` on every
    path where `verify()` ran, so a dead model server stays distinguishable from a
    wrong cartridge.
    """
    started = time.monotonic()
    if last_action is not None:
        # Flags come from the executor (body_state.json); the words are the
        # brain's. Composing them here is what makes a saturated reach visible
        # to the next PLAN turn - see `outcome_from_flags`.
        skill, side, target = last_action
        # Only a reach asks the glow. It means "the hand is in grab range", which
        # says nothing about whether a look arrived.
        outcome = (reach_outcome(glow, state_flags) if skill == "reach"
                   else outcome_from_flags(state_flags))
        last = last_line(skill, side, target, outcome)
    live = live_block(left, right, yaw_rad, last)
    plan_result = plan(config, seed, frame, live, transcript=history, wav=wav,
                       default_side="right" if right == "empty" else "left")

    crop_error = ""
    ground_frame = frame
    ground_box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    ground_result: Ground | None = None
    target = plan_result.target
    if plan_result.ok and target and plan_result.skill in ("reach", "look", "grab"):
        if box is not None:
            try:
                ground_frame = Frame(
                    index=frame.index,
                    t_s=frame.t_s,
                    jpeg=crop_jpeg(frame.jpeg, box),
                    source=f"{frame.source}:crop",
                )
                ground_box = box
            except RuntimeError as exc:
                crop_error = str(exc)
        ground_result = ground(config, ground_frame, target, box=ground_box,
                               container=container, transcript=history)

    verify_result: Verify | None = None
    if plan_result.ok and target and plan_result.skill == "grab" and glow == "yes":
        check = verify(config, ground_frame, target)
        verify_result = check
        if not check.match:
            # No pixel on purpose: the refusal has to travel the path the brain
            # already has for "I cannot see it", not a new one invented here. The
            # ground keeps its quadrant, raw and timings - only the pixel goes.
            error = (f"verify mismatch: {check.error}" if check.error
                     else "verify mismatch")
            kept = ground_result or Ground(noun=target, box=ground_box)
            ground_result = replace(kept, pixel=None, error=error)

    decision = assemble(
        plan_result, ground_result, seq=seq,
        t_mono_ns=time.monotonic_ns(), look_step=look_step,
        left=left, right=right, glow=glow, ghost=ghost,
    )
    return TurnRecord(
        turn=turn,
        frame_index=frame.index,
        t_s=frame.t_s,
        heard=heard,
        seed=seed,
        live=live,
        plan=plan_result,
        ground=ground_result,
        decision=decision,
        verify=verify_result,
        self_state={
            "left": left,
            "right": right,
            "yaw_rad": yaw_rad,
            "glow": glow,
            "ghost": ghost,
            "box": list(box) if box is not None else None,
            "crop_error": crop_error,
        },
        elapsed_ms=(time.monotonic() - started) * 1000.0 + elapsed_ms,
    )


__all__ = [
    "BrainConfig",
    "CONSTITUTION",
    "DEFAULT_SEED",
    "Decision",
    "Ground",
    "PLANNER_SKILLS",
    "Plan",
    "OUTCOMES",
    "QUADRANTS",
    "TurnRecord",
    "Verify",
    "assemble",
    "compose_pixel",
    "extract_json_object",
    "ground",
    "last_line",
    "live_block",
    "outcome_from_flags",
    "plan",
    "reach_outcome",
    "run_turn",
    "server_status",
    "verify",
]
