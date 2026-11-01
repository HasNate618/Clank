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
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from typing import Any

from .body import (
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

#: The planner may only ask for skills the adapter can honour in this writer. The
#: schema is wider (`fire`, `strike`, ...) so the adapter does not fork, but the
#: constitution never offers them, and a model that names one is rejected here.
PLANNER_SKILLS = ("hold", "look", "reach", "grab", "release")

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
    """
    if flags & FLAG_SATURATED:
        return "saturated"
    if flags & FLAG_TTL_EXPIRED:
        return "timeout"
    return "done"


def last_line(
    skill: str, side: str, target: str | None, outcome: str
) -> str:
    """One `last:` line for `live_block`, in the shape §7.3 specifies.

    `none`/`null` rather than omission: the model is told what it asked for, and
    a missing field reads as a schema it can leave out next time.
    """
    return f"{skill} {side} target={target or 'null'} → {outcome}"

#: Skills that address one hand.
PLANNER_SIDE_SKILLS = ("reach", "grab", "release")

#: Judged on the recording: a 30 s intro at 1 frame/s leaves ~30 turns of slack.
DEFAULT_TIMEOUT_S = 30.0

CONSTITUTION = """\
You are a robot with two hands. You see one image from your eyes and the speech \
since your last action.

You may do exactly one action:
- hold    keep still
- look    turn to look; name a target to look toward it, or omit the target to \
look a step to the side
- reach   move one hand toward a named thing
- grab    close one hand
- release open one hand

While holding something, reach moves it with that hand.

After each action you are told done, saturated, or timeout, and you see a new \
image.

Reply with one JSON object and nothing else. All three keys are always present:
{"skill":"hold"|"look"|"reach"|"grab"|"release",\
"side":"none"|"left"|"right","target": <noun string or null>}

side is "none" for hold and look, and is "left" or "right" for reach, grab and \
release.

Do what the voice asked. Do not invent extra chores. If you cannot see the \
thing, look. If you are unsure, hold.

Grab only when glow is yes. A yellow ghost means the held thing will go into \
the slot if you release. A full hand cannot grab; use the other hand or \
release.\
"""

#: Asked once, verbatim. One instruction, no choreography.
DEFAULT_SEED = (
    "use the store clerk cartridge (green label).\n"
    "a cartridge goes in the glowing slot; a yellow ghost means you may release.\n"
    "pull the lever down to start the job."
)

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


def _chat(
    config: BrainConfig,
    messages: list[dict[str, Any]],
    max_tokens: int | None = None,
) -> ModelReply:
    """One POST to the OpenAI-compatible endpoint. Never raises into the caller."""
    payload = {
        "messages": messages,
        "max_tokens": max_tokens if max_tokens is not None else config.max_tokens,
        "temperature": config.temperature,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{config.base_url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
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
        prompt_ms=float(timings.get("prompt_ms", 0.0)),
        decode_ms=float(timings.get("predicted_ms", elapsed_ms)),
    )


def _image_part(frame: Frame) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": frame.data_url()}}


def _audio_part(wav: bytes) -> dict[str, Any]:
    encoded = base64.b64encode(wav).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}}


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
    heard: str,
    left: str,
    right: str,
    yaw_rad: float,
    glow: str,
    last: str,
    ghost: str = "unknown",
) -> str:
    """The only per-turn context. No task list, no current-goal line."""
    return (
        f"heard: {heard.strip() if heard.strip() else '(none)'}\n"
        f"self:  left={left} right={right} yaw={yaw_rad:.2f} glow={glow} ghost={ghost}\n"
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
        {"role": "system", "content": CONSTITUTION},
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
    side = parsed.get("side")
    target = parsed.get("target")

    note = ""
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
        error = f"{skill} does not take a side"

    if target is not None and not isinstance(target, str):
        error = error or f"target {target!r} is not a string or null"

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

    needs_where = skill in ("reach", "look") and target is not None
    if needs_where:
        if ground_result is None or ground_result.pixel is None:
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
        pixel = ground_result.pixel

    if skill == "grab" and target is not None:
        if ground_result is None or ground_result.pixel is None:
            return Decision(Intent(skill="hold", side="none", source="vlm", seq=seq,
                                   t_mono_ns=t_mono_ns),
                            reason=f"grab {target!r} but it is not visible")
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
    )
    reason = validate_intent(intent)
    if reason and skill in ("look", "reach", "grab", "release"):
        return Decision(Intent(skill="hold", side="none", source="vlm", seq=seq,
                               t_mono_ns=t_mono_ns),
                        reason=reason, note="held after rejection")
    return Decision(intent, reason=reason, note=note)


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
        last = last_line(skill, side, target, outcome_from_flags(state_flags))
    live = live_block(heard, left, right, yaw_rad, glow, last, ghost=ghost)
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
    "run_turn",
    "verify",
]
