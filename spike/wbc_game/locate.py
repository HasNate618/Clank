"""Find a named object's pixel with no human in the loop.

Everything here was measured separately before it was joined:

  * one name -> quadrant question about a container crop scored 44/44;
  * the same question about the whole frame scored 9/12;
  * one call about a *right-half* crop returned an exact pixel 39 px from truth.

So the plan is those three in order: ask the whole frame for a quadrant (4-way,
the only question measured correct at full scale), cut that quadrant, and ask the
crop for an exact pixel once. No grid, no offset table, nothing hand-marked.

The failure mode to watch: if stage 1 names the wrong quadrant, stage 2 is looking
at a crop without the object in it. That is not silent - the absence test measured
`found: false` on a tight crop - so a wrong quadrant should surface as a refusal
rather than a confident pixel. Stage 2's answer is therefore also the check on
stage 1, and `locate()` reports both.

    python3 spike/wbc_game/locate.py --image frame.png --noun "JobBot ID card" \
        --truth 740 415
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import pathlib
import re
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from clank import brain, frames  # noqa: E402
from clank.brain import QUADRANTS, BrainConfig, Frame  # noqa: E402

#: How much bigger than the quadrant the stage-2 look is. 1.0 means the crop is
#: exactly the quadrant; 1.15 keeps a margin so an object sitting on the seam is
#: not cut in half by stage 1's answer.
MARGIN = 1.15

#: ffmpeg resize width for the stage-2 crop. The still is 960 wide, a quadrant is
#: 480, and the model's pixel error is a resolution limit - so the crop goes up to
#: the same 960 the full frame gets, which is 2x for a quadrant.
CROP_WIDTH = 960

#: Stage 2 asks for a **box**, several times, at these `(sent width, crop padding)` pairs.
#: The padding is a fraction of the box's own size, and it jitters the framing so the reads
#: are independent. Measured 2026-11-09 on a live Job Simulator frame: the model is bimodal -
#: the same question at a different send width flipped between 9 px and 384 px - so one read
#: is a coin flip. Taking the median of each of the four **edges** over five reads located the
#: JobBot badge to 29 px where a single read of the same image gave 235 px.
BOX_READS = ((640, 0.0), (720, 0.03), (800, 0.0), (880, 0.04), (960, 0.02))

BOX_PROMPT = (
    "This image is a crop of the game view. Give the bounding box of the {noun} in this "
    'image, as JSON only: {{"x0": <int>, "y0": <int>, "x1": <int>, "y1": <int>}}. '
    "Coordinates are a 0 to 1000 grid: 0 is the left/top edge, 1000 the right/bottom edge. "
    "Do not answer with a different object. If the {noun} is not in this image, reply "
    'exactly {{"none": true}}.'
)

#: Stage 3, as a description rather than a question about the model's own answer.
#: Measured 2026-11-09: shown a crop of JobBot's ID badge the model describes it as "Job Bot
#: card" or "Job Bot Museum Curator card", and the lever knob as "Blue mushroom" - accurate
#: and specific, on two different frames. Asked instead "is there a job cartridge here?"
#: about that same badge crop it answered YES, and asked to choose between the badge and a
#: cartridge it answered "A" for a lever. The model can *say* what it sees and cannot *judge*
#: whether that is what was asked for, so the judgement comes back into code.
DESCRIBE_PROMPT = (
    "What object is in the middle of this image? Name it in at most six words. "
    "No punctuation, no JSON, just the name."
)

#: Words that say nothing about identity: the summoner's own name, and filler.
_MATCH_STOP = frozenset({
    "job", "bot", "jobbot", "simulator", "museum", "the", "a", "an", "of", "in",
    "on", "it", "this", "image", "middle", "object", "thing",
})

#: Colour and light say nothing about identity. They are excluded from the words a match
#: may rest on, because measured 2026-11-09 on a second frame: a box described as "green
#: robot computer" shared its colour with nothing in "job cartridge" and was accepted as a
#: weak match, which is exactly the false accept this check exists to stop. A colour may
#: appear in a description; it may not carry a verdict.
_MATCH_WEAK = frozenset({
    "blue", "green", "red", "white", "black", "yellow", "cyan", "orange", "grey",
    "gray", "pink", "purple", "brown", "light", "dark",
})

#: This job's vocabulary, not a general ontology: what callers ask for, and what the model
#: has actually called those things. "mushroom" is in the lever set because that is the
#: measured description of the blue knob on the stick.
_MATCH_SYNONYMS = {
    "card": frozenset({"card", "badge", "id", "curator", "barcode"}),
    "badge": frozenset({"card", "badge", "id", "curator", "barcode"}),
    "cartridge": frozenset({"cartridge", "cart", "case", "cassette"}),
    "lever": frozenset({"lever", "knob", "handle", "ball", "sphere", "stick", "mushroom"}),
    "knob": frozenset({"lever", "knob", "handle", "ball", "sphere", "stick", "mushroom"}),
}

#: How tightly stage 3 looks around stage 2's answer, as a fraction of the frame.
#: The absence measurement is the reason this exists: one call about a *tight*
#: crop said `found: false` for an object that was not there, while a quadrant
#: crop said a confident pixel. Refusal is a property of the crop, so the check
#: needs its own, small crop.
#:
#: 0.09 was too small. Measured 2026-11-04: a pixel dead centre on the GOURMET
#: CHEF label, verified with a +/-0.09 crop, was refused - a crop that tight holds
#: one word of a label and reads as a fragment rather than as a cartridge. The same
#: refusal, blamed on the verifier, then got verification switched off, and the
#: next locate produced a confident error 0.83 m from the tray. A verifier that is
#: wrong in both directions is worse than none, so the crop is now big enough to
#: contain a whole object.
VERIFY_HALF = 0.16

_PIXEL = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:px)?\s*[, ]\s*(-?\d+(?:\.\d+)?)")

#: A box reply, by key name so the "0"/"1" inside `x0`/`x1` cannot be read as a value.
_BOX_KEYS = re.compile(r'([xy][01])"?\s*:\s*(-?\d+(?:\.\d+)?)')


def parse_box(text: str) -> tuple[float, float, float, float] | None:
    """The four numbers of a box reply, in the order `x0, y0, x1, y1`.

    Read by key name first. A bare four-number reply is accepted too, but only after the
    keyed form fails: measured 2026-11-09, a parser that just takes the first four numbers
    off `{"x0": 105, ...}` reads the `0` in the key and returns a box at negative zero.
    """
    keyed = {name: float(value) for name, value in _BOX_KEYS.findall(text)}
    if len(keyed) == 4:
        return keyed["x0"], keyed["y0"], keyed["x1"], keyed["y1"]
    numbers = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", text)]
    if len(numbers) >= 4:
        return numbers[0], numbers[1], numbers[2], numbers[3]
    return None


def _norm_word(word: str) -> str:
    """Singularise just enough for "cards" and "cartridges"."""
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def describe_matches(noun: str, description: str) -> tuple[bool, str]:
    """Does `description` name the object `noun` asked for? Returns (matched, why).

    The point of this function is that it lives in code. The model was measured to describe
    a crop correctly and to answer YES to "is there a job cartridge here?" about a badge, so
    the judgement cannot stay on the other side of the call.
    """

    def words(text: str) -> list[str]:
        return re.findall(r"[a-z][a-z'\-]*", text.lower())

    described = [_norm_word(w) for w in words(description) if w not in _MATCH_STOP]
    if not described:
        return False, "nothing described"
    asked = [w for w in words(noun) if w not in _MATCH_STOP and w not in _MATCH_WEAK]
    synonyms: set[str] = set()
    for word in words(noun):
        synonyms.update(_norm_word(s) for s in _MATCH_SYNONYMS.get(word, ()))
    described_set = set(described)
    for word in asked:
        stem = _norm_word(word)
        if stem in described_set:
            return True, f"described as {stem!r}"
    shared = sorted(synonyms & described_set)
    if shared:
        return True, f"described as {shared[0]!r}, a synonym of the {noun}"
    # No weak tier: a shared colour is not evidence, and accepting one turned a box
    # described as "green robot computer" into a job cartridge.
    return False, f"described as {described[0]!r}, which is not the {noun}"


def describe_crop(
    config: BrainConfig, frame: Frame, box: tuple[float, float, float, float]
) -> str:
    """Stage 3: what is actually in `box`? One call, no judgement asked of the model."""
    jpeg = frames.crop_jpeg(frame.jpeg, box, width=CROP_WIDTH)
    cropped = Frame(index=frame.index, t_s=frame.t_s, jpeg=jpeg, source=f"describe:{box}")
    reply = brain._chat(config, [{"role": "user", "content": [
        brain._image_part(cropped), {"type": "text", "text": DESCRIBE_PROMPT},
    ]}], max_tokens=24)
    return reply.text.strip()


def _grow(box: tuple[float, float, float, float], pad: float) -> tuple[float, float, float, float]:
    """`box` widened by `pad` of its own size, clipped to the frame."""
    w, h = box[2] - box[0], box[3] - box[1]
    return (
        max(0.0, box[0] - pad * w),
        max(0.0, box[1] - pad * h),
        min(1.0, box[2] + pad * w),
        min(1.0, box[3] + pad * h),
    )


def quadrant_box(quadrant: str, margin: float = MARGIN) -> tuple[float, float, float, float]:
    """The quadrant as `(u0, v0, u1, v1)`, grown by `margin` and clipped to the frame."""
    u0 = 0.0 if "LEFT" in quadrant else 0.5
    v0 = 0.0 if "TOP" in quadrant else 0.5
    cu, cv = u0 + 0.25, v0 + 0.25
    half_u, half_v = 0.25 * margin, 0.25 * margin
    return (
        max(0.0, cu - half_u),
        max(0.0, cv - half_v),
        min(1.0, cu + half_u),
        min(1.0, cv + half_v),
    )


def ask_quadrant(config: BrainConfig, frame: Frame, noun: str) -> brain.Ground:
    """Stage 1: which quadrant of the whole frame holds the noun."""
    return brain.ground(config, frame, noun, box=(0.0, 0.0, 1.0, 1.0), container="scene")


def ask_pixel(
    config: BrainConfig,
    frame: Frame,
    noun: str,
    box: tuple[float, float, float, float],
    size: tuple[int, int],
) -> tuple[tuple[float, float] | None, str, float, float]:
    """Stage 2: one call about the crop, asking for an exact pixel.

    The prompt asks for coordinates *of this image* and says the object is there,
    because the measured 39 px call did exactly that - no grid, no quadrant, no
    list of candidates. A refusal (`NONE`) is returned as None, never as a guess.
    """
    prompt = (
        f"This image is a crop of the game view. Where is the {noun} in this image? "
        "Reply with two numbers only: the x and y pixel of its centre, "
        "in this image's own coordinates."
    )
    jpeg = frames.crop_jpeg(frame.jpeg, box, width=CROP_WIDTH)
    cropped = Frame(index=frame.index, t_s=frame.t_s, jpeg=jpeg, source=f"crop:{box}")
    reply = brain._chat(config, [{"role": "user", "content": [
        brain._image_part(cropped), {"type": "text", "text": prompt},
    ]}], max_tokens=40)
    if reply.error:
        return None, reply.text, reply.prompt_ms, reply.decode_ms
    words = reply.text.upper()
    if any(word in words for word in ("NONE", "ABSENT", "NOT VISIBLE", "CANNOT")):
        return None, reply.text, reply.prompt_ms, reply.decode_ms
    match = _PIXEL.search(reply.text)
    if match is None:
        return None, reply.text, reply.prompt_ms, reply.decode_ms
    # **Both** axes divide by CROP_WIDTH, and that is measured, not assumed.
    # The crop is resized to CROP_WIDTH wide with its height following the aspect
    # ratio (a quadrant of a 960x540 frame becomes 960x483), so the "correct"
    # thing looked like dividing y by 483. Measured against three truth pixels:
    # dividing both by 960 gave 4.7 px and 17.4 px; dividing y by the crop's own
    # height gave 136 px and 163 px. The model answers on a width-normalised,
    # aspect-ignoring grid - its y for a known point (fraction 0.57 of the crop)
    # came back as 551 and 620, giving an implied divisor of 968 and 1089, i.e.
    # ~960, not 483. Trust the measurement over the arithmetic.
    return (
        float(match.group(1)) / CROP_WIDTH,
        float(match.group(2)) / CROP_WIDTH,
    ), reply.text, reply.prompt_ms, reply.decode_ms


def ask_box(
    config: BrainConfig,
    frame: Frame,
    noun: str,
    box: tuple[float, float, float, float],
    width: int,
    pad: float,
) -> tuple[tuple[float, float, float, float] | None, str]:
    """One stage-2 read: a box for `noun`, composed back into `box`'s coordinates.

    The reply is a 0-1000 grid over the image that was sent, and that image is `box`
    grown by `pad`, so the grid point maps straight back without a second offset table.
    """
    shown = _grow(box, pad)
    jpeg = frames.crop_jpeg(frame.jpeg, shown, width=width)
    cropped = Frame(index=frame.index, t_s=frame.t_s, jpeg=jpeg, source=f"box:{shown}")
    reply = brain._chat(
        config,
        [{"role": "user", "content": [
            brain._image_part(cropped),
            {"type": "text", "text": BOX_PROMPT.format(noun=noun)},
        ]}],
        max_tokens=60,
    )
    if reply.error:
        return None, reply.text
    got = parse_box(reply.text)
    if got is None:
        return None, reply.text
    span_x, span_y = shown[2] - shown[0], shown[3] - shown[1]
    x0 = shown[0] + got[0] / 1000.0 * span_x
    y0 = shown[1] + got[1] / 1000.0 * span_y
    x1 = shown[0] + got[2] / 1000.0 * span_x
    y1 = shown[1] + got[3] / 1000.0 * span_y
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return (x0, y0, x1, y1), reply.text


def read_box(
    config: BrainConfig,
    frame: Frame,
    noun: str,
    box: tuple[float, float, float, float],
    reads: tuple[tuple[int, float], ...] = BOX_READS,
    times: list[float] | None = None,
    parallel: int = 4,
) -> tuple[
    tuple[float, float, float, float] | None,
    list[tuple[float, float, float, float]],
    list[str],
    bool,
]:
    """Stage 2: the box, read several times at different framings.

    Returns the median-edge box, every individual box, every raw reply, and whether the
    model refused. The median is computed **per edge**, not per centre, so one wild read
    cannot drag the answer the way it drags a mean: on the measured frame four reads
    bracketed the badge and the fifth bracketed the console behind it.

    Refusal is a majority of reads replying `{"none": true}`, which `BOX_PROMPT` now asks
    for. Measured 2026-11-09: without that clause the model never refused a box - asked for
    a banana it returned the brightest object in the frame - and with it, 5 of 5 reads
    refused. A refusal is worth more than a box here, because a box it cannot have is
    exactly what the old pipeline invented.

    Per-read durations are appended to `times` when given. They are needed because a slow
    read is one the server queued or the network retried, and a caller deciding whether it
    can afford the grid has to know the real cost of one call. With `parallel` > 1 they are
    wall time per read as well, but the reads share the wall clock, so five entries sum to
    more than the time the call took - `locate()` times the whole stage separately.

    The reads are independent questions about different crops on a server that serves four
    at once (`/props` reports `total_slots: 4`), so they run concurrently; `executor.map`
    keeps them in `reads` order, which is what makes the median and the raw-reply list
    identical to the serial loop. Pass `parallel=1` for that loop.
    """

    def read_one(
        framing: tuple[int, float],
    ) -> tuple[tuple[float, float, float, float] | None, str, float]:
        width, pad = framing
        started = time.perf_counter()
        got, raw = ask_box(config, frame, noun, box, width, pad)
        return got, raw, time.perf_counter() - started

    if parallel > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
            answers = list(pool.map(read_one, reads))
    else:
        answers = [read_one(framing) for framing in reads]

    boxes: list[tuple[float, float, float, float]] = []
    raws: list[str] = []
    refusals = 0
    for one, raw, seconds in answers:
        if times is not None:
            times.append(seconds)
        raws.append(raw.strip())
        if '"none"' in raw.lower():
            refusals += 1
            continue
        if one is not None:
            boxes.append(one)
    refused = refusals * 2 > len(reads)
    if not boxes:
        return None, [], raws, refused
    median = tuple(statistics.median(edge) for edge in zip(*boxes, strict=True))
    return median, boxes, raws, refused


def verify_present(
    config: BrainConfig,
    frame: Frame,
    noun: str,
    pixel_px: tuple[float, float],
    size: tuple[int, int],
    box: tuple[float, float, float, float] | None = None,
) -> tuple[bool | None, str]:
    """Stage 3: is the noun actually in a tight crop around stage 2's answer?

    Returns (True, raw) / (False, raw) / (None, raw) when the reply is unreadable.
    A `False` is not a guess: it is the measured behaviour of a tight crop, and it
    is what stops a wrong stage-1 quadrant from becoming a confident pixel.
    """
    width, height = size
    if box is not None:
        crop_box = box
    else:
        u, v = pixel_px[0] / width, pixel_px[1] / height
        crop_box = (
            max(0.0, u - VERIFY_HALF),
            max(0.0, v - VERIFY_HALF),
            min(1.0, u + VERIFY_HALF),
            min(1.0, v + VERIFY_HALF),
        )
    jpeg = frames.crop_jpeg(frame.jpeg, crop_box, width=CROP_WIDTH)
    cropped = Frame(index=frame.index, t_s=frame.t_s, jpeg=jpeg, source=f"verify:{crop_box}")
    prompt = f"Is there a {noun} in this image? Answer YES or NO."
    reply = brain._chat(config, [{"role": "user", "content": [
        brain._image_part(cropped), {"type": "text", "text": prompt},
    ]}], max_tokens=12)
    words = reply.text.upper()
    if "NO" in words.split() or words.strip().startswith("NO"):
        return False, reply.text
    if "YES" in words:
        return True, reply.text
    return None, reply.text


def locate(
    config: BrainConfig,
    frame: Frame,
    noun: str,
    size: tuple[int, int],
    verify: bool = True,
    reads: int = 5,
    verify_mode: str = "describe",
    parallel: int = 4,
) -> dict:
    """Quadrant, then box reads. Returns a dict with a full-frame pixel in px.

    `timings` reports where the wall clock went, in ms, and how many model calls were made.
    It is emitted rather than estimated because the cost moved by more than 2x between the
    first measurement of this pipeline and the second, and a stale estimate is worse than no
    estimate: an older doc claimed 0.72 s per call when the live server answers in 0.40 s.
    """
    width, height = size
    started = time.perf_counter()
    timings: dict = {"quadrant_ms": 0.0, "reads_ms": 0.0, "verify_ms": 0.0}
    calls = {"n": 0}

    def done() -> dict:
        """`out` with its clock stopped, so every early return still reports a total."""
        timings["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
        timings["calls"] = calls["n"]
        out["timings"] = timings
        return out

    stage1_started = time.perf_counter()
    stage1 = ask_quadrant(config, frame, noun)
    calls["n"] += 1
    timings["quadrant_ms"] = round((time.perf_counter() - stage1_started) * 1000, 1)
    out: dict = {
        "noun": noun,
        "quadrant": stage1.quadrant,
        "stage1_raw": stage1.raw,
        "error": stage1.error,
        "pixel_px": None,
        "box": None,
        "stage2_raw": "",
    }
    if stage1.quadrant is None:
        return done()

    box = quadrant_box(stage1.quadrant)
    out["box"] = [round(v, 4) for v in box]
    if reads <= 0:
        # The single point read this module was measured on. Kept so the two can be
        # compared on the same frame; the default is the box ensemble.
        reads_started = time.perf_counter()
        uv, raw, p_ms, d_ms = ask_pixel(config, frame, noun, box, size)
        calls["n"] += 1
        timings["reads_ms"] = round((time.perf_counter() - reads_started) * 1000, 1)
        out["stage2_raw"] = raw.strip()
        out["stage2_prompt_ms"] = round(p_ms, 1)
        out["stage2_decode_ms"] = round(d_ms, 1)
        if uv is None:
            out["error"] = "stage 2 did not name a pixel"
            return done()
        u = box[0] + uv[0] * (box[2] - box[0])
        v = box[1] + uv[1] * (box[3] - box[1])
        pixel = (round(u * width, 1), round(v * height, 1))
        out["pixel_px"] = list(pixel)
        if not verify:
            out["verify_raw"] = "skipped"
            out["present"] = None
            return done()
        verify_started = time.perf_counter()
        present, raw3 = verify_present(config, frame, noun, pixel, size)
        calls["n"] += 1
        timings["verify_ms"] = round((time.perf_counter() - verify_started) * 1000, 1)
        out["verify_raw"] = raw3.strip()
        out["present"] = present
        if present is False:
            out["error"] = "not visible"
        return done()

    plan = tuple(BOX_READS[i % len(BOX_READS)] for i in range(reads))
    per_read: list[float] = []
    reads_started = time.perf_counter()
    median, boxes, raws, refused = read_box(config, frame, noun, box, plan,
                                            times=per_read, parallel=parallel)
    timings["reads_ms"] = round((time.perf_counter() - reads_started) * 1000, 1)
    # Per-read wall time. Under `parallel` these overlap, so they sum to more than
    # `reads_ms` - that is contention, not double counting.
    timings["read_ms"] = [round(t * 1000, 1) for t in per_read]
    calls["n"] += len(raws)
    out["stage2_raw"] = raws
    out["stage2_reads"] = len(raws)
    refusals = sum(1 for raw in raws if '"none"' in raw.lower())
    out["none_votes"] = refusals
    if median is None or refused:
        # A majority refusing is the answer, not a failure to get one: the noun is not in
        # this frame, which is something the pipeline could not previously be told.
        out["error"] = "not visible" if refused else "stage 2 did not name a box"
        out["present"] = False if refused else None
        return done()
    out["found_box"] = [round(v, 4) for v in median]
    out["stage2_boxes"] = [[round(v, 4) for v in b] for b in boxes]
    centre = ((median[0] + median[2]) / 2, (median[1] + median[3]) / 2)
    out["pixel_px"] = [round(centre[0] * width, 1), round(centre[1] * height, 1)]
    # How far the individual reads disagreed with the median, in px of this frame. The
    # caller's confidence signal: measured, the reads that landed 380 px away are visible
    # here as a large spread long before anyone acts on them.
    out["read_spread_px"] = round(
        max(
            (
                (((b[0] + b[2]) / 2 - centre[0]) * width) ** 2
                + (((b[1] + b[3]) / 2 - centre[1]) * height) ** 2
            )
            ** 0.5
            for b in boxes
        ),
        1,
    )
    out["box_reads"] = len(boxes)
    if not verify:
        out["verify_raw"] = "skipped"
        out["present"] = None
        return done()
    # Verify on the model's own extent, but never on a crop smaller than the measured safe
    # size. A lever is a thin box: measured 2026-11-09, stage 2 placed it 5.9 px from truth
    # and the verifier still refused it, because a 33x7 px crop holds a grey stick with no
    # knob in it. The floor is `VERIFY_HALF`, the size that measurement says reads as an
    # object rather than as a fragment.
    half_w = max((median[2] - median[0]) / 2, VERIFY_HALF)
    half_h = max((median[3] - median[1]) / 2, VERIFY_HALF)
    verify_box = (
        max(0.0, centre[0] - half_w),
        max(0.0, centre[1] - half_h),
        min(1.0, centre[0] + half_w),
        min(1.0, centre[1] + half_h),
    )
    verify_started = time.perf_counter()
    if verify_mode == "describe":
        # Ask what is *in* the box and decide here. The model describes accurately and
        # judges its own answer badly - see `DESCRIBE_PROMPT` - so the comparison is code.
        description = describe_crop(config, frame, verify_box)
        present, why = describe_matches(noun, description)
        out["match"] = why
        out["verify_raw"] = description
    else:
        present, raw3 = verify_present(
            config, frame, noun, (centre[0] * width, centre[1] * height), size, box=verify_box
        )
        out["verify_raw"] = raw3.strip()
    calls["n"] += 1
    timings["verify_ms"] = round((time.perf_counter() - verify_started) * 1000, 1)
    out["present"] = present
    if present is False:
        out["error"] = "not visible"
    return done()


def _faded(
    image: pathlib.Path, depth: pathlib.Path, percentile: float, near: float, far: float
) -> tuple[bytes, tuple[int, int], float]:
    """The still with everything behind the working volume faded, its size, and the seed.

    numpy and Pillow are imported here rather than at module scope: `locate.py` itself
    stays standard-library only, like the rest of the surface it sits on, and the fade is
    an optional input to the same pipeline rather than a dependency of it.
    """
    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import fade as fade_module

    rgb = np.asarray(Image.open(image).convert("RGB"))
    depth_m = np.load(depth)
    target = fade_module.foreground_depth(depth_m, percentile)
    out = fade_module.fade_foreground(rgb, depth_m, percentile, near, far)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    return buf.getvalue(), (int(out.shape[1]), int(out.shape[0])), target


def _post_overlay(url: str, out: dict, size: tuple[int, int]) -> bool:
    """Tell the dashboard where we looked, so it can draw it on the picture.

    Best effort on purpose: a demo must not fail because a dashboard is not running, so
    every transport problem is a ``False`` rather than an exception.
    """
    import urllib.parse
    import urllib.request

    addressed = urllib.parse.urlsplit(url)
    # This posts to a dashboard on this machine and nothing else, so anything that is not
    # loopback HTTP is refused before a socket is opened. `file:` in a URL that came from
    # the command line is a read primitive, not a dashboard.
    if addressed.scheme not in ("http", "https"):
        return False
    if addressed.hostname not in ("127.0.0.1", "localhost", "::1"):
        return False

    payload = json.dumps({
        "noun": out.get("noun") or "object",
        "box": out.get("found_box"),
        "pixel": out.get("pixel_px"),
        "size": [size[0], size[1]],
        "spread": out.get("read_spread_px"),
        "present": out.get("present"),
    }).encode()
    request = urllib.request.Request(  # noqa: S310 - scheme and host checked just above
        url, data=payload, method="POST", headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(  # noqa: S310 - scheme and host checked just above
            request, timeout=2
        ) as response:
            return 200 <= response.status < 300
    except Exception:  # noqa: BLE001 - no dashboard is the normal case for a CLI run
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--image", type=pathlib.Path, required=True)
    ap.add_argument("--noun", default="JobBot ID card")
    ap.add_argument("--truth", nargs=2, type=float, metavar=("X", "Y"))
    ap.add_argument("--model")
    ap.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="how many box reads to have in flight at once (1 for the serial loop)",
    )
    ap.add_argument(
        "--reads",
        type=int,
        default=len(BOX_READS),
        help="how many jittered box reads stage 2 makes; 0 falls back to the single point "
        "read this module was originally measured on, so the two can be compared",
    )
    ap.add_argument(
        "--depth",
        type=pathlib.Path,
        help="float32 .npy depth for the same view. Fades the background before grounding, "
        "using a band derived from the depth map, so no object depth is needed first",
    )
    ap.add_argument("--no-fade", action="store_true", help="ignore --depth")
    ap.add_argument("--fade-percentile", type=float, default=5.0)
    ap.add_argument("--fade-near", type=float, default=1.06)
    ap.add_argument("--fade-far", type=float, default=1.45)
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the tight-crop YES/NO. Measured 2026-11-04: on a cartridge label the "
        "verifier refused a pixel that was dead centre on the object, because a tight crop "
        "of a label reads as a fragment rather than as a cartridge. For a grab the grasp "
        "itself is the stronger post-condition.",
    )
    ap.add_argument(
        "--verify-mode",
        choices=("describe", "yesno"),
        default="describe",
        help="how stage 3 decides: 'describe' asks what is in the box and compares the "
        "answer here, 'yesno' asks the model whether the noun is there. Measured "
        "2026-11-09: yes/no answered YES for a job cartridge about a crop of JobBot's ID "
        "badge, and the description of that same crop was 'Job Bot card'.",
    )
    ap.add_argument("--jsonl", type=pathlib.Path)
    ap.add_argument(
        "--post",
        metavar="URL",
        help="POST the result to a dashboard so the preview draws the box, e.g. "
        "http://127.0.0.1:8765/api/overlay",
    )
    args = ap.parse_args(argv)

    config = BrainConfig()
    if args.model:
        config = BrainConfig(model=args.model)
    from PIL import Image  # only for the size; ffmpeg does the crops

    faded: dict = {}
    fade_ms = 0.0
    if args.depth and not args.no_fade:
        fade_started = time.perf_counter()
        jpeg, size, target = _faded(
            args.image, args.depth, args.fade_percentile, args.fade_near, args.fade_far
        )
        fade_ms = round((time.perf_counter() - fade_started) * 1000, 1)
        frame = Frame(index=0, t_s=0.0, jpeg=jpeg, source=f"{args.image}+fade")
        faded = {
            "target_m": round(target, 3),
            "cut_m": [round(target * args.fade_near, 3), round(target * args.fade_far, 3)],
        }
    else:
        with Image.open(args.image) as img:
            size = img.size
        frame = Frame(index=0, t_s=0.0, jpeg=args.image.read_bytes(), source=str(args.image))

    out = locate(
        config, frame, args.noun, size, verify=not args.no_verify, reads=args.reads,
        verify_mode=args.verify_mode,
    )
    if faded:
        out["fade"] = faded
    if fade_ms:
        # Separate from `total_ms`: the fade is paid once for the still, not once per call,
        # and a caller comparing the faded and unfaded paths needs to see it on its own.
        out.setdefault("timings", {})["fade_ms"] = fade_ms
    if args.post and out.get("found_box"):
        out["posted"] = _post_overlay(args.post, out, size)
    if args.truth and out["pixel_px"]:
        dx = out["pixel_px"][0] - args.truth[0]
        dy = out["pixel_px"][1] - args.truth[1]
        out["error_px"] = round((dx * dx + dy * dy) ** 0.5, 1)
    print(json.dumps(out, indent=2))
    if args.jsonl:
        with args.jsonl.open("a") as fh:
            fh.write(json.dumps(out) + "\n")
    return 0 if out["pixel_px"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
