#!/usr/bin/env python3
"""
Grounding harness for LFM3-Omni-2B on the Job Simulator "Take my card." task.

STATUS: NOT YET RUN. Every number this script would produce is UNMEASURED.
Nothing in this file is a result; it is the instrument.

Design constraints (from the task owner):
  * client only. POSTs to http://127.0.0.1:11434/v1/chat/completions. It never
    binds, listens on, or opens a port.
  * one CLI subcommand per configuration so each experiment is independently
    runnable and its raw output cleanly separable.
  * prints the reply VERBATIM, then a single JSON line with mode, prompt hash,
    wall-clock seconds and whatever usage/timings the server returns.

Run with the venv python (PIL is needed for crop modes):
    spike/e3/.venv/bin/python research/grounding/ground.py --help
    spike/e3/.venv/bin/python research/grounding/ground.py --mode all --dry-run

Modes are deliberately METHOD-varied rather than repeated: at temperature 0 the
model is byte-identical across repeats, so repeating one call measures nothing.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

ENDPOINT = "http://127.0.0.1:11434/v1/chat/completions"
MODEL = "/home/nate/AI/models/LFM3-Omni-2B-Q8_0.gguf"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_IMAGE = os.path.join(REPO, "research", "evidence", "grounding-seed-view.png")

# Ground truth, in full-frame 960x540 pixels of DEFAULT_IMAGE.
TRUTH_PX = (705, 420)
FRAME_W, FRAME_H = 960, 540
TRUTH_NAME = "JobBot ID card"

# Gold description, used for isolated GROUND measurement. It paraphrases the
# human's own description of the object and is deliberately discriminative so
# that a GROUND failure cannot be blamed on a vague input.
GOLD_DESC = (
    "the JobBot ID card: a small white badge with a barcode and small printed "
    "text reading JobBot, leaning against the lower left of the green monitor "
    "that has a cartoon face"
)

# Default crop boxes, full-frame coords (x0, y0, x1, y1).
CROP_RIGHT_BOX = (480, 0, 960, 540)         # right half of the frame
CROP_CONTAINER_BOX = (600, 240, 960, 500)   # the green monitor + the card
NO_BOX = (0, 0, 0, 0)

GAME_CONTEXT = """\
You are the planner for a humanoid robot working a shift at the Job Simulator
museum counter.

Scene: you are facing a museum counter. In the middle is a white retro terminal.
To your right sits a green desktop computer with a cartoon face; this machine is
the NPC "JobBot", your co-worker. JobBot hands out tasks by speaking to you.
"""

JOBBOT_LINE = "Take my card."
ABSENT_LINE = "Fetch the red fire extinguisher."


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------

def plan_prompt(line: str = JOBBOT_LINE) -> str:
    return (
        GAME_CONTEXT
        + '\nJobBot says: "' + line + '"\n\n'
        "Work out the single best low-level action. The action vocabulary is "
        "skill + side + target.\n"
        "- skill: one of grab, press, push, turn, place\n"
        "- side: which hand to use, left or right\n"
        "- target: the object to act on\n\n"
        "Reply with JSON only and nothing else:\n"
        '{"skill": "<skill>", "side": "<left|right>", "target": "<the object, '
        "described precisely enough that someone who never heard JobBot speak "
        'can find it in a photo of this scene>"}'
    )


def ground_prompt(desc: str, w: int, h: int) -> str:
    return (
        "This is a photo of the robot's view of a Job Simulator museum counter.\n\n"
        "Find this object: " + desc + "\n\n"
        "Reply with JSON only and nothing else:\n"
        '{"found": true, "x": <integer pixel x>, "y": <integer pixel y>}\n\n'
        "x and y are pixel coordinates in the image you were given, origin at the "
        "TOP-LEFT corner, x increasing to the right. The image is "
        + str(w) + " pixels wide and " + str(h) + " pixels high. "
        'If the object is not in the image, reply {"found": false}.'
    )


def grid3_prompt(desc: str, w: int, h: int) -> str:
    return (
        "This is a photo of the robot's view of a Job Simulator museum counter.\n\n"
        "Find this object: " + desc + "\n\n"
        "The image is divided into a 3 by 3 grid. Reply with JSON only and nothing "
        "else:\n"
        '{"found": true, "cell": "<row>,<col>"}\n\n'
        "row is 0 for the top third, 1 for the middle third, 2 for the bottom "
        "third. col is 0 for the left third, 1 for the middle third, 2 for the "
        "right third. The image is " + str(w) + " pixels wide and " + str(h) +
        " pixels high. "
        'If the object is not in the image, reply {"found": false}.'
    )


def quadrant_prompt(desc: str, w: int, h: int) -> str:
    return (
        "This is a photo of the robot's view of a Job Simulator museum counter.\n\n"
        "Find this object: " + desc + "\n\n"
        "Reply with JSON only and nothing else:\n"
        '{"found": true, "quadrant": "<top-left|top-right|bottom-left|bottom-right>"}\n\n'
        "The quadrant refers to the image you were given: top-left is the upper "
        "left corner. The image is " + str(w) + " pixels wide and " + str(h) +
        " pixels high. "
        'If the object is not in the image, reply {"found": false}.'
    )


def one_call_prompt(line: str = JOBBOT_LINE) -> str:
    return (
        GAME_CONTEXT
        + '\nJobBot says: "' + line + '"\n\n'
        "Decide the single best low-level action AND locate its target in the "
        "image you were given.\n"
        "- skill: one of grab, press, push, turn, place\n"
        "- side: which hand to use, left or right\n"
        "- target: the object to act on\n\n"
        "Reply with JSON only and nothing else:\n"
        '{"skill": "<skill>", "side": "<left|right>", "target": "<the object>", '
        '"found": true, "x": <integer pixel x>, "y": <integer pixel y>}\n\n'
        "x and y are pixel coordinates in the image you were given, origin at the "
        "TOP-LEFT corner, x increasing to the right. "
        'If the target is not in the image, reply {"found": false}.'
    )


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------

def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def data_url(path: str) -> str:
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode("ascii")


def image_size(path: str, fallback=(FRAME_W, FRAME_H)):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        return fallback


def make_crop(src: str, box, scale: int) -> tuple[str, int, int]:
    """Crop src to box and optionally upscale. Returns (path, w, h)."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        sys.exit("crop modes need PIL: run with spike/e3/.venv/bin/python")
    im = Image.open(src).convert("RGB")
    im = im.crop(box)
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
    out = os.path.join(
        HERE, "crop_%d_%d_%d_%d_x%d.png" % (box[0], box[1], box[2], box[3], scale)
    )
    im.save(out)
    return out, im.width, im.height


def post(prompt: str, image_path: str | None, temperature: float, max_tokens: int):
    content = [{"type": "text", "text": prompt}]
    if image_path:
        content.append({"type": "image_url", "image_url": {"url": data_url(image_path)}})
    body = {
        "model": MODEL,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            raw = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        status = exc.code
    except Exception as exc:  # connection refused, timeout, ...
        return "", time.perf_counter() - t0, {"error": repr(exc)}, 0
    wall = time.perf_counter() - t0
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw, wall, {"error": "non-json response", "body": raw[:2000]}, status
    text = ""
    try:
        text = payload["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        text = ""
    meta = {"usage": payload.get("usage"), "timings": payload.get("timings")}
    if not text:
        meta["raw_body"] = raw[:2000]
    return text, wall, meta, status


FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json(text: str):
    """Best-effort JSON object out of a model reply. Returns (obj|None, note)."""
    if not text:
        return None, "empty reply"
    candidates = FENCE.findall(text)
    candidates.append(text)
    for cand in candidates:
        cand = cand.strip()
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj, "ok"
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj, "ok (substring)"
        except json.JSONDecodeError:
            pass
    return None, "unparseable"


def quadrant_of(px, py, w, h) -> str:
    return ("top" if py < h / 2 else "bottom") + "-" + ("left" if px < w / 2 else "right")


def cell_centre(row: int, col: int, w: int, h: int):
    """Centre, in the coordinates of the image the model was shown."""
    return ((col + 0.5) * w / 3.0, (row + 0.5) * h / 3.0)


def to_full(box, scale, x, y):
    """Map a point in the shown image back to full-frame pixels."""
    return (box[0] + x / float(scale), box[1] + y / float(scale))


def to_shown(box, scale, x, y):
    """Map a full-frame point into the coordinates of the shown image."""
    return ((x - box[0]) * scale, (y - box[1]) * scale)


def err(px, py):
    return ((px - TRUTH_PX[0]) ** 2 + (py - TRUTH_PX[1]) ** 2) ** 0.5


def truth_cell(box, scale, w, h):
    tx, ty = to_shown(box, scale, TRUTH_PX[0], TRUTH_PX[1])
    return [int(max(0, min(2, ty / (h / 3.0)))), int(max(0, min(2, tx / (w / 3.0))))]


def truth_quadrant(box, scale, w, h):
    tx, ty = to_shown(box, scale, TRUTH_PX[0], TRUTH_PX[1])
    return quadrant_of(tx, ty, w, h)


# --------------------------------------------------------------------------
# run one configuration
# --------------------------------------------------------------------------

def run(args, mode, prompt, image_path, img_w, img_h, box, scale, expects):
    """Run one call and print verbatim reply + one JSON record line."""
    box = NO_BOX if box is None else tuple(box)
    if args.dry_run:
        print("\n" + "=" * 78)
        print("MODE: %s  (dry run, no request sent)" % mode)
        print("IMAGE: %s  %dx%d  box=%s scale=%d"
              % (image_path or "<none>", img_w, img_h, box, scale))
        print("-" * 78)
        print(prompt)
        print("=" * 78)
        return

    text, wall, meta, status = post(prompt, image_path, args.temperature, args.max_tokens)
    obj, note = parse_json(text)
    obj = obj or {}

    rec = {
        "mode": mode,
        "temperature": args.temperature,
        "prompt_sha256": sha(prompt),
        "wall_seconds": round(wall, 3),
        "http_status": status,
        "image": os.path.basename(image_path) if image_path else None,
        "image_wh": [img_w, img_h],
        "crop_box": list(box),
        "crop_scale": scale,
        "truth_px": list(TRUTH_PX),
        "truth_quadrant": truth_quadrant(box, scale, img_w, img_h)
        if expects == "quadrant" else None,
        "truth_cell": truth_cell(box, scale, img_w, img_h)
        if expects == "grid3" else None,
        "parse": note,
        "meta": meta,
    }

    found = obj.get("found", None)
    if found is False:
        rec["found"] = False
        rec["hallucinated"] = False

    if expects in ("px", "one-call") and obj and found is not False:
        try:
            x, y = float(obj["x"]), float(obj["y"])
            mx, my = to_full(box, scale, x, y)
            rec["returned_px"] = [x, y]
            rec["mapped_px"] = [round(mx, 1), round(my, 1)]
            rec["error_px"] = round(err(mx, my), 1)
            rec["in_frame"] = bool(0 <= mx <= FRAME_W and 0 <= my <= FRAME_H)
            rec["found"] = True
            rec["hallucinated"] = bool(args.absent)
        except (KeyError, TypeError, ValueError):
            rec["parse"] = "no numeric x/y"

    if expects == "grid3" and obj and found is not False:
        try:
            row, col = [int(v) for v in str(obj["cell"]).replace(" ", "").split(",")]
            cx, cy = cell_centre(row, col, img_w, img_h)
            mx, my = to_full(box, scale, cx, cy)
            rec["returned_cell"] = [row, col]
            rec["mapped_px"] = [round(mx, 1), round(my, 1)]
            rec["error_px"] = round(err(mx, my), 1)
            rec["cell_correct"] = ([row, col] == rec["truth_cell"])
        except (KeyError, TypeError, ValueError):
            rec["parse"] = "no usable cell"

    if expects == "quadrant" and obj and found is not False:
        q = str(obj.get("quadrant", "")).lower().strip()
        rec["returned_quadrant"] = q
        rec["quadrant_correct"] = (q == rec["truth_quadrant"])
        rec["mapped_px"] = "n/a (quadrant only)"

    if "skill" in obj:
        rec["skill"] = obj.get("skill")
        rec["side"] = obj.get("side")
    if "target" in obj:
        tgt = str(obj.get("target", "")).strip()
        rec["target"] = tgt
        low = tgt.lower()
        names_object = any(k in low for k in ("card", "badge", "id", "1011", "barcode"))
        is_specific = any(k in low for k in ("white", "monitor", "green", "screen",
                                             "barcode", "curator", "muse", "jobbot"))
        rec["target_names_object"] = names_object
        rec["target_discriminative"] = bool(names_object and is_specific)

    print("\n" + "=" * 78)
    print("MODE: %s   temp=%s   image=%s (%dx%d)%s"
          % (mode, args.temperature,
             os.path.basename(image_path) if image_path else "<none text-only>",
             img_w, img_h,
             ("   box=%s scale=%d" % (box, scale)) if box != NO_BOX else ""))
    print("-" * 78)
    print("PROMPT:")
    print(prompt)
    print("-" * 78)
    print("REPLY (verbatim):")
    print(text if text else "<empty>")
    print("-" * 78)
    print("RECORD: " + json.dumps(rec, sort_keys=True))
    print("=" * 78)

    if args.jsonl:
        rec["reply"] = text
        rec["prompt"] = prompt
        with open(args.jsonl, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")


# --------------------------------------------------------------------------
# configurations
# --------------------------------------------------------------------------

def cfg_plain(args):
    yield ("ground-full", ground_prompt(args.description, FRAME_W, FRAME_H),
           args.image, None, 1, "px")
    yield ("ground-grid3", grid3_prompt(args.description, FRAME_W, FRAME_H),
           args.image, None, 1, "grid3")


def cfg_crop_right(args):
    path, w, h = make_crop(args.image, CROP_RIGHT_BOX, args.crop_scale)
    print("# crop written: %s (%dx%d)" % (path, w, h))
    yield ("ground-crop-right", ground_prompt(args.description, w, h), path,
           CROP_RIGHT_BOX, args.crop_scale, "px")
    yield ("ground-crop-right-grid3", grid3_prompt(args.description, w, h), path,
           CROP_RIGHT_BOX, args.crop_scale, "grid3")


def cfg_quadrant(args):
    path, w, h = make_crop(args.image, CROP_CONTAINER_BOX, args.crop_scale)
    print("# crop written: %s (%dx%d)" % (path, w, h))
    yield ("ground-quadrant", quadrant_prompt(args.description, w, h), path,
           CROP_CONTAINER_BOX, args.crop_scale, "quadrant")


def cfg_one_call(args):
    yield ("one-call", one_call_prompt(), args.image, None, 1, "one-call")
    path, w, h = make_crop(args.image, CROP_RIGHT_BOX, args.crop_scale)
    print("# crop written: %s (%dx%d)" % (path, w, h))
    yield ("one-call-crop-right", one_call_prompt(), path, CROP_RIGHT_BOX,
           args.crop_scale, "one-call")


def cfg_absent_one_call(args):
    # The line names an object that is not in the scene. A correctly grounded
    # model returns found=false. This model always answers, so this is the test.
    yield ("absent-one-call", one_call_prompt(ABSENT_LINE), args.image, None, 1,
           "one-call")


def cfg_absent_two_call(args):
    desc = "the red fire extinguisher"
    yield ("absent-plan", plan_prompt(ABSENT_LINE), None, None, 1, "plan")
    yield ("absent-ground-full", ground_prompt(desc, FRAME_W, FRAME_H), args.image,
           None, 1, "px")
    path, w, h = make_crop(args.image, CROP_RIGHT_BOX, args.crop_scale)
    print("# crop written: %s (%dx%d)" % (path, w, h))
    yield ("absent-ground-crop-right", ground_prompt(desc, w, h), path,
           CROP_RIGHT_BOX, args.crop_scale, "px")


MODES = {
    "plan": lambda a: [("plan", plan_prompt(), None, None, 1, "plan")],
    "plan-vision": lambda a: [("plan-vision", plan_prompt(), a.image, None, 1, "plan")],
    "ground-full": lambda a: [m for m in cfg_plain(a) if m[0] == "ground-full"],
    "ground-grid3": lambda a: [m for m in cfg_plain(a) if m[0] == "ground-grid3"],
    "ground-crop-right": lambda a: [m for m in cfg_crop_right(a)
                                    if m[0] == "ground-crop-right"],
    "ground-crop-right-grid3": lambda a: [m for m in cfg_crop_right(a)
                                          if m[0] == "ground-crop-right-grid3"],
    "ground-quadrant": lambda a: list(cfg_quadrant(a)),
    "one-call": lambda a: list(cfg_one_call(a)),
    "absent-one-call": lambda a: list(cfg_absent_one_call(a)),
    "absent-two-call": lambda a: list(cfg_absent_two_call(a)),
    "ground": lambda a: (list(cfg_plain(a)) + list(cfg_crop_right(a))
                         + list(cfg_quadrant(a))),
    "all": lambda a: (list(cfg_plain(a)) + list(cfg_crop_right(a))
                      + list(cfg_quadrant(a)) + list(cfg_one_call(a))
                      + list(cfg_absent_one_call(a)) + list(cfg_absent_two_call(a))),
}


def main(argv=None):
    p = argparse.ArgumentParser(
        description="NOT-YET-RUN grounding harness for LFM3-Omni-2B. "
                    "Client only; never binds a port.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="modes: " + ", ".join(sorted(MODES)),
    )
    p.add_argument("--mode", default="plan", choices=sorted(MODES))
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--description", default=GOLD_DESC,
                   help="object description fed to GROUND (default: gold description)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--repeat", type=int, default=1,
                   help="only meaningful at temperature > 0; temp 0 is byte-identical")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--crop-scale", type=int, default=2,
                   help="integer upscale applied to crops (default 2)")
    p.add_argument("--jsonl", default=None,
                   help="append full records (prompt+reply) as JSONL to this path")
    p.add_argument("--dry-run", action="store_true",
                   help="print the exact prompts without sending anything")
    p.add_argument("--vision", action="store_true",
                   help="alias for --mode plan-vision")
    args = p.parse_args(argv)

    if args.vision:
        args.mode = "plan-vision"
    args.absent = args.mode.startswith("absent")
    if not os.path.exists(args.image):
        sys.exit("image not found: " + args.image)
    if args.repeat > 1 and args.temperature == 0.0:
        print("# note: temperature 0 is byte-identical across repeats; repeating "
              "measures nothing. Vary --mode instead.", file=sys.stderr)

    for i in range(max(1, args.repeat)):
        if args.repeat > 1:
            print("# repeat %d/%d" % (i + 1, args.repeat))
        for mode, prompt, image, box, scale, expects in MODES[args.mode](args):
            size = image_size(image) if image else (FRAME_W, FRAME_H)
            run(args, mode, prompt, image, size[0], size[1], box, scale, expects)
    return 0


if __name__ == "__main__":
    sys.exit(main())
