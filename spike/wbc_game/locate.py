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
import io
import json
import pathlib
import re
import sys

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


def verify_present(
    config: BrainConfig,
    frame: Frame,
    noun: str,
    pixel_px: tuple[float, float],
    size: tuple[int, int],
) -> tuple[bool | None, str]:
    """Stage 3: is the noun actually in a tight crop around stage 2's answer?

    Returns (True, raw) / (False, raw) / (None, raw) when the reply is unreadable.
    A `False` is not a guess: it is the measured behaviour of a tight crop, and it
    is what stops a wrong stage-1 quadrant from becoming a confident pixel.
    """
    width, height = size
    u, v = pixel_px[0] / width, pixel_px[1] / height
    box = (
        max(0.0, u - VERIFY_HALF),
        max(0.0, v - VERIFY_HALF),
        min(1.0, u + VERIFY_HALF),
        min(1.0, v + VERIFY_HALF),
    )
    jpeg = frames.crop_jpeg(frame.jpeg, box, width=CROP_WIDTH)
    cropped = Frame(index=frame.index, t_s=frame.t_s, jpeg=jpeg, source=f"verify:{box}")
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
    config: BrainConfig, frame: Frame, noun: str, size: tuple[int, int], verify: bool = True
) -> dict:
    """Quadrant, then one crop call. Returns a dict with a full-frame pixel in px."""
    width, height = size
    stage1 = ask_quadrant(config, frame, noun)
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
        return out

    box = quadrant_box(stage1.quadrant)
    out["box"] = [round(v, 4) for v in box]
    uv, raw, p_ms, d_ms = ask_pixel(config, frame, noun, box, size)
    out["stage2_raw"] = raw.strip()
    out["stage2_prompt_ms"] = round(p_ms, 1)
    out["stage2_decode_ms"] = round(d_ms, 1)
    if uv is None:
        out["error"] = "stage 2 did not name a pixel"
        return out
    u = box[0] + uv[0] * (box[2] - box[0])
    v = box[1] + uv[1] * (box[3] - box[1])
    out["pixel_px"] = [round(u * width, 1), round(v * height, 1)]
    if not verify:
        out["verify_raw"] = "skipped"
        out["present"] = None
        return out
    present, raw3 = verify_present(config, frame, noun, out["pixel_px"], size)
    out["verify_raw"] = raw3.strip()
    out["present"] = present
    if present is False:
        out["error"] = "not visible"
        out["quadrant"] = stage1.quadrant
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", type=pathlib.Path, required=True)
    ap.add_argument("--noun", default="JobBot ID card")
    ap.add_argument("--truth", nargs=2, type=float, metavar=("X", "Y"))
    ap.add_argument("--model")
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the tight-crop YES/NO. Measured 2026-11-04: on a cartridge label the "
        "verifier refused a pixel that was dead centre on the object, because a tight crop "
        "of a label reads as a fragment rather than as a cartridge. For a grab the grasp "
        "itself is the stronger post-condition.",
    )
    ap.add_argument("--jsonl", type=pathlib.Path)
    args = ap.parse_args(argv)

    config = BrainConfig()
    if args.model:
        config = BrainConfig(model=args.model)
    from PIL import Image  # only for the size; ffmpeg does the crops

    with Image.open(args.image) as img:
        size = img.size
    frame = Frame(index=0, t_s=0.0, jpeg=args.image.read_bytes(), source=str(args.image))

    out = locate(config, frame, args.noun, size, verify=not args.no_verify)
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
