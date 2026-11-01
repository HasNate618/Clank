"""The brain: hear a sentence, look at one still, emit one legal intent.

This is the writer described by `docs/superpowers/specs/2026-10-31-brain.md`. It
owns three things and no others:

1. **Prompt assembly** - a short constitution (body facts and an output schema),
   one injected seed, and a per-turn live block. There is no walkthrough here.
2. **Two model calls** - PLAN (which skill, which side, which noun) and GROUND
   (where in the image is that noun). They are separate calls on purpose; see
   `ground()` for the measurement that forced the split.
3. **The trace** - one `TurnRecord` per turn saying what it heard, saw, said, was
   told, and whether the intent survived `validate_intent`.

It does **not** move the body. It produces `Intent`s; the adapter compiles them.

Two measurements decided the shape of this file, both taken against the live
`LFM3-Omni-2B` endpoint on 2026-11-01:

- A single "which of these 9 cells contains X?" question answered `E` (centre) for
  an object that was in `H` (bottom-centre). A 9-way pick has a centre bias. Two
  decomposed thirds questions answered `BOTTOM` correctly for the same image, so
  grounding is decomposed rather than picked.
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
from dataclasses import dataclass, field
from typing import Any

from .frames import Frame
from .intent import (
    LOOK_ARC_RAD,
    SKILLS,
    Intent,
    validate_intent,
)

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "LFM3-Omni-2B-Q8_0.gguf"

#: The planner may only ask for skills the adapter can honour in this writer. The
#: schema is wider (`fire`, `strike`, ...) so the adapter does not fork, but the
#: constitution never offers them, and a model that names one is rejected here.
PLANNER_SKILLS = ("hold", "look", "reach", "grab", "release")

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

THIRDS_ROWS = ("TOP", "MIDDLE", "BOTTOM")
THIRDS_COLS = ("LEFT", "MIDDLE", "RIGHT")
_CELL_LETTERS = "ABCDEFGHI"


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
    """Where the named noun is, as a 3x3 cell and a normalised pixel."""

    noun: str
    row: int | None = None
    col: int | None = None
    pixel: tuple[float, float] | None = None
    raw_row: str = ""
    raw_col: str = ""
    error: str = ""
    methods: str = "thirds"
    prompt_ms: float = 0.0
    decode_ms: float = 0.0

    @property
    def cell(self) -> str | None:
        if self.row is None or self.col is None:
            return None
        return _CELL_LETTERS[self.row * 3 + self.col]

    def to_dict(self) -> dict[str, Any]:
        return {
            "noun": self.noun,
            "cell": self.cell,
            "row": self.row,
            "col": self.col,
            "pixel": list(self.pixel) if self.pixel else None,
            "raw_row": self.raw_row,
            "raw_col": self.raw_col,
            "error": self.error,
            "methods": self.methods,
            "prompt_ms": round(self.prompt_ms, 1),
            "decode_ms": round(self.decode_ms, 1),
        }


def _match_word(text: str, options: tuple[str, ...], allow_none: bool) -> int | None:
    """Return an index into `options`, -1 for NONE, or None when unparseable."""
    upper = re.sub(r"[^A-Z ]", " ", text.upper())
    tokens = upper.split()
    if allow_none and ("NONE" in tokens or "NOT VISIBLE" in upper or "ABSENT" in tokens):
        return -1
    for index, option in enumerate(options):
        if option in tokens:
            return index
    return None


def ground(
    config: BrainConfig,
    frame: Frame,
    noun: str,
    transcript: list[str] | None = None,
) -> Ground:
    """Locate `noun` by two decomposed questions rather than one 9-way pick.

    Measured 2026-11-01 on the business card (truth: bottom third, centre column,
    `v ~ 0.73`):

    - one 9-way "which cell" question answered `E`, the centre - a centre bias;
    - the vertical third question answered `MIDDLE` for all five phrasings tried,
      so it is stable but biased toward the middle band;
    - a binary "is it in the lower half?" answered `YES`, correctly;
    - recursive bisection with binary questions broke down entirely: past the
      first question the 2B model narrates ("LOOKING AT THE IMAGE, ...") and it
      answers `YES` on a frame that contains no card at all.

    So the coarse 3x3 form is kept because it is the only shape that answers
    consistently, and its known centre bias is recorded rather than papered over.
    A grounder that is honestly coarse is usable - the adapter servos and Job
    Simulator's grab volumes are generous - whereas one that hallucinates
    presence is not. The consequence is in the spec's E1 gate: pixel skills stay
    off until grounding is measured against labelled cells, and the live answer
    to precision is active perception (reach, let the game highlight what is
    grabbable, then close), not a better question.
    """
    context = ""
    if transcript:
        context = "Context:\n" + "\n".join(f"- {line}" for line in transcript) + "\n\n"

    row_prompt = (
        f"{context}Which horizontal band of the image contains the {noun}: "
        "TOP, MIDDLE, or BOTTOM? Reply with exactly one of those words, "
        "or NONE if the object is not visible."
    )
    col_prompt = (
        f"{context}Which vertical band of the image contains the {noun}: "
        "LEFT, MIDDLE, or RIGHT? Reply with exactly one of those words, "
        "or NONE if the object is not visible."
    )

    row_reply = _chat(config, [{"role": "user", "content": [
        _image_part(frame), {"type": "text", "text": row_prompt},
    ]}], max_tokens=12)
    if row_reply.error:
        return Ground(noun=noun, error=f"row: {row_reply.error}")

    row = _match_word(row_reply.text, THIRDS_ROWS, allow_none=True)
    if row is None:
        return Ground(noun=noun, raw_row=row_reply.text,
                      error=f"row reply unparseable: {row_reply.text!r}")
    if row == -1:
        return Ground(noun=noun, raw_row=row_reply.text, error="not visible")

    col_reply = _chat(config, [{"role": "user", "content": [
        _image_part(frame), {"type": "text", "text": col_prompt},
    ]}], max_tokens=12)
    if col_reply.error:
        return Ground(noun=noun, row=row, raw_row=row_reply.text,
                      error=f"col: {col_reply.error}")

    col = _match_word(col_reply.text, THIRDS_COLS, allow_none=True)
    if col is None:
        return Ground(noun=noun, row=row, raw_row=row_reply.text, raw_col=col_reply.text,
                      error=f"col reply unparseable: {col_reply.text!r}")
    if col == -1:
        return Ground(noun=noun, row=row, raw_row=row_reply.text,
                      raw_col=col_reply.text, error="not visible")

    pixel = ((col + 0.5) / 3.0, (row + 0.5) / 3.0)
    return Ground(
        noun=noun,
        row=row,
        col=col,
        pixel=pixel,
        raw_row=row_reply.text.strip(),
        raw_col=col_reply.text.strip(),
        prompt_ms=row_reply.prompt_ms + col_reply.prompt_ms,
        decode_ms=row_reply.decode_ms + col_reply.decode_ms,
    )


# --------------------------------------------------------------------------- #
# ASSEMBLE
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Decision:
    """What survives validation, plus the reason when nothing does."""

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
) -> Decision:
    """Turn a plan plus its grounding into an `Intent`, or explain the refusal.

    Refusals are values, not exceptions: the caller holds and shows the reason.

    The occupancy guards are the one place this writer checks a plan against the
    body, and they earn their keep: measured 2026-11-01, the text path answered
    `release right` to "take my card" while the right hand was empty. That is not
    a schema error, so `validate_intent` cannot catch it, and executing it would
    burn a turn to change nothing.
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

    needs_where = skill in ("reach", "look") and target is not None
    if needs_where:
        if ground_result is None or ground_result.pixel is None:
            reason = (ground_result.error if ground_result and ground_result.error
                      else "target not grounded")
            # A named thing we cannot see is not a reason to guess a pixel; the
            # constitution already told the model to look when it cannot see.
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
    last: str = "(none)",
    wav: bytes | None = None,
    turn: int = 0,
    look_step: int = 1,
    elapsed_ms: float = 0.0,
) -> TurnRecord:
    """One complete turn: PLAN, then GROUND only if the plan needs a location."""
    started = time.monotonic()
    live = live_block(heard, left, right, yaw_rad, glow, last, ghost=ghost)
    plan_result = plan(config, seed, frame, live, transcript=history, wav=wav,
                       default_side="right" if right == "empty" else "left")

    ground_result: Ground | None = None
    if plan_result.ok and plan_result.target and plan_result.skill in (
        "reach", "look", "grab"
    ):
        ground_result = ground(config, frame, plan_result.target, transcript=history)

    decision = assemble(
        plan_result, ground_result, seq=seq,
        t_mono_ns=time.monotonic_ns(), look_step=look_step,
        left=left, right=right,
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
        self_state={
            "left": left,
            "right": right,
            "yaw_rad": yaw_rad,
            "glow": glow,
            "ghost": ghost,
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
    "TurnRecord",
    "assemble",
    "extract_json_object",
    "ground",
    "live_block",
    "plan",
    "run_turn",
]

# `SKILLS` is imported for the schema assertion in tests, not used here.
_ = SKILLS
