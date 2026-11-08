"""The loop: the game speaks, the brain answers, the body acts. Nothing typed, nothing clicked.

One process, because two measurements say so: the hand must be closed inside the reach loop
(a separate process left ~2 s in which the hand drifted 0.09 m, which is more than the grab
radius), and every step of one turn has to share the same frame, since a pixel marked once
goes stale within a tick as the head wobbles.

    spike/e3/.venv/bin/python spike/wbc_game/auto.py --for 900
    spike/e3/.venv/bin/python spike/wbc_game/auto.py --for 0 --say "take my card"   # no ear

What it does per turn:

    ears.listen       -> the game's own audio, gated and trimmed
    brain.transcribe  -> the words JobBot said
    brain.next_step   -> {goal, object, destination, side} from one still
    brain.sequence    -> the ordered skills; ordering is code, never the prompt
    locate            -> the pixel the noun is at, verified present
    reach_vr --grab   -> one continuous stream plus the close, in the same process

Every decision goes to clank.brainlog, so the dashboard's brain panel is the transcript of
what happened rather than a reconstruction. A quiet game writes nothing at all.
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import do_task  # noqa: E402
import locate  # noqa: E402
import reach_vr  # noqa: E402

import math  # noqa: E402

from clank import brain, brainlog, ears  # noqa: E402
from clank.body import Pose  # noqa: E402
from clank.brain import BrainConfig, Frame  # noqa: E402
from clank.depth_stream import Stream  # noqa: E402
from clank.pixel3d import point_from_pixel  # noqa: E402
from clank.projection import Fov  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
VENV = do_task.VENV
DASHBOARD = "http://127.0.0.1:8765"

#: Where the frames the brain looked at are kept, so a wrong grounding can be checked
#: after the fact instead of argued about.
STATE = pathlib.Path.home() / ".local/state/Clank/auto"

#: A session start takes about a minute; stop this long before the deadline.
MARGIN_S = 25.0

#: How many frames to try before saying the thing is not there. `reach_vr` exits 1 when
#: the marked point is more than `MAX_PLAUSIBLE_M` from the resting hand - measured
#: 2026-11-08, locate("card") answered the JOB SIMULATOR sign at 4.9 m - so a fresh frame
#: and another look is what that code asks for. Exit 2 is different: the point is real and
#: out of reach, and looking again will not help.
LOOK_TRIES = 4

#: The same frustum `place.py` uses: the OpenXR FOV the game renders with.
FOV = Fov.from_degrees(70.0, 70.0, 57.11, 57.11)
#: Head yaw offsets to try, in order, before giving up on seeing the thing. Measured: a
#: commanded yaw RATE does not turn the body (+0.2 deg where 69 was asked), but the head is
#: pinned by every reach command, so the *look* is the one aimable part of the robot.
LOOK_DEG = (0.0, 25.0, -25.0, 45.0, -45.0)
#: A grounded point outside this distance from the *resting hand* is not the object being
#: reached for. The bound is reach_vr's own: it refuses past MAX_PLAUSIBLE_M from the hand,
#: and measuring from the camera instead let a 0.9 m-away pixel pass here and be refused
#: there. Nothing is grabbable behind you either.
MIN_OBJ_M = 0.10
MAX_OBJ_M = 1.0


def _post(path: str, payload: dict, timeout: float = 30.0) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(  # noqa: S310 - fixed http://127.0.0.1 dashboard url
        DASHBOARD + path, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read() or b"{}")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"error": str(exc)}


def ready(mode: str = "body") -> bool:
    """Put the session in `mode` and reset the body. A reach from wherever the policy
    wandered measured 49-79 mm of residual; from a reset root it is 37 mm."""
    if mode:
        _post("/api/session/mode", {"mode": mode})
    brainlog.log("think", "standing up before I look", mode=mode or "unchanged")
    for command in ("stop", "start"):
        out = _post(f"/api/body/{command}", {})
        if out.get("error"):
            brainlog.log("error", f"body {command}: {out['error']}")
            return False
        time.sleep(2.0 if command == "stop" else 1.0)
    return True


def head_pose(st: dict) -> Pose:
    return Pose(p=tuple(st["head"]["p"]), q_xyzw=tuple(st["head"]["q_xyzw"]))


def look(dyaw_deg: float, side: str = "right") -> None:
    """Point the head without moving the arm: pin the head to its own pose, yawed.

    The arm target is the wrist's current position, so the reach verb's only effect here is
    the head pin. Repeated by the caller, because a command's TTL is short and a single
    look can expire before the frame is captured.
    """
    st = reach_vr.state()
    half = math.radians(dyaw_deg) / 2.0
    turn = (math.cos(half), 0.0, 0.0, math.sin(half))  # wxyz, about +z
    q_new = reach_vr._wxyz_to_xyzw(  # noqa: SLF001 - the shared quaternion helpers
        reach_vr._quat_mul(turn, reach_vr._xyzw_to_wxyz(st["head"]["q_xyzw"]))  # noqa: SLF001
    )
    reach_vr.send(verb="reach", side=side, yaw_deg=0.0,
                  target=[round(v, 4) for v in reach_vr.link(st, side)],
                  head=[round(v, 5) for v in (*st["head"]["p"], *q_new)])


def frame_and_jpeg(width: int = 960) -> tuple[bytes, tuple[int, int], bytes, dict, Pose]:
    """One still the brain can look at, plus the bytes to mark for evidence."""
    from PIL import Image

    with Stream() as stream:
        f = stream.read(timeout=2.0)
    if f is None:
        raise SystemExit("no frame from the depth stream")
    image = Image.frombytes(
        "RGBA", (f["color_width"], f["color_height"]), bytes(f["color"])
    ).convert("RGB")
    height = round(width * image.height / image.width)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    png = io.BytesIO()
    image.save(png, format="PNG")
    return buf.getvalue(), (width, height), png.getvalue(), f, head_pose(reach_vr.state())


def mark(png: bytes, pixel: tuple[float, float], out: pathlib.Path) -> None:
    """Draw where the brain said the object is, so the claim can be checked by eye."""
    from PIL import Image, ImageDraw

    image = Image.open(io.BytesIO(png)).convert("RGB")
    draw = ImageDraw.Draw(image)
    x, y = pixel
    draw.line([(x - 18, y), (x + 18, y)], fill=(0, 255, 0), width=2)
    draw.line([(x, y - 18), (x, y + 18)], fill=(0, 255, 0), width=2)
    draw.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(0, 255, 0), width=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)


def grab_pixel(x: float, y: float, side: str) -> int:
    """Reach the pixel and close, in one process with no gap."""
    return subprocess.call([
        VENV, str(HERE / "reach_vr.py"), f"{x:.1f}", f"{y:.1f}", "--side", side,
        "--no-rest", "--no-clamp", "--tol", "0.02", "--corrections", "4",
        "--gain", "0.6", "--guard", "0.25", "--grab",
    ])


def one_turn(say: str = "", pattern: str = ears.DEFAULT_PATTERN,
             max_s: float = 25.0, plan_only: bool = False) -> str:
    """Hear, interpret, ground, act. Returns the words heard ("" if the game was quiet)."""
    if say:
        spoken = say
        brainlog.log("heard", spoken, source="typed")
    else:
        try:
            hearing = ears.listen(pattern=pattern, max_s=max_s)
        except ears.EarsError as exc:
            return f"waiting: {exc}"
        if not hearing.ok:
            return ""
        heard = brain.transcribe(BrainConfig(), hearing.wav)
        if heard.error:
            brainlog.log("error", f"transcribe failed: {heard.error}")
            return ""
        if heard.empty:
            return ""
        spoken = heard.text
        brainlog.log("heard", spoken, ms=heard.decode_ms, speech_s=hearing.speech_s,
                     peak=hearing.peak, cut=heard.ellipsis)

    # Stand the robot up *before* looking, not after. Measured 2026-11-08: the frame was
    # captured from wherever the previous run's failed reach had left the body, so the card
    # was off-screen and locate answered the signage instead. The default pose is the one
    # the card is visible from; the reset is what puts the robot back in it.
    if not plan_only and not ready():
        return spoken

    jpeg, size, png, depth_frame, camera = frame_and_jpeg()
    stamp = time.strftime("%H%M%S")
    sight = STATE / f"frame-{stamp}.png"
    sight.parent.mkdir(parents=True, exist_ok=True)
    sight.write_bytes(png)

    step = brain.next_step(
        BrainConfig(max_tokens=160), Frame(0, 0.0, jpeg, "live"),
        f'JobBot said: "{spoken}"', default_side="right",
    )
    if step.error:
        brainlog.log("error", f"the model did not answer: {step.error}")
        return spoken
    brainlog.log("think", f"I will {step.goal}", object=step.object,
                 destination=step.destination, side=step.side)

    side = step.side or "right"
    # The instruction names the verb and the object; the model was measured answering a
    # later step with an invented object ("insert the card into the machine" from "take my
    # card"). So the task line wins for both, and the model keeps the destination.
    skill, noun = brain.task_intent(spoken)
    if noun:
        if step.object and noun not in (step.object or "").lower():
            brainlog.log("think", f"the task says {noun!r}, the model said {step.object!r}"
                                  " - the task wins")
        steps = [("grab", noun)]
        if skill == "place" and step.destination:
            steps.append(("place", step.destination))
    else:
        steps = brain.sequence(step, holding=False, object_noun=step.object or "")
    brainlog.log("think", "plan: " + (" -> ".join(f"{s}({n})" for s, n in steps) or "nothing"),
                 hand="empty", frame=sight.name)
    if plan_only or not steps:
        return spoken

    for skill, target in steps:
        if skill != "grab":
            brainlog.log("think", f"skipping {skill}({target}) - only grab is wired")
            continue
        # Look before reaching. The robot cannot turn its body, so the head is swept until
        # the noun grounds to a point the arm could actually get to; a pixel that unprojects
        # to metres away is signage, not the card (measured: locate answered the JOB
        # SIMULATOR sign at 4.9 m).
        seen = None
        # The hand nearest the object reaches for it. The model's `side` is a guess from a
        # still and it guessed left for a card on the right, which crosses the body and
        # saturates; the grounded point's own y says which hand has it.
        rest_left = reach_vr.wrist("left")
        rest_right = reach_vr.wrist("right")
        for dyaw in LOOK_DEG:
            if dyaw:
                for _ in range(4):
                    look(dyaw, side)
                    time.sleep(0.18)
            else:
                look(0.0, side)
            jpeg, size, png, depth_frame, camera = frame_and_jpeg()
            found = locate.locate(BrainConfig(), Frame(0, 0.0, jpeg, "live"), target, size)
            pixel = found.get("pixel_px")
            metres = 0.0
            point = None
            if pixel:
                point = point_from_pixel(depth_frame, pixel[0] / size[0],
                                         pixel[1] / size[1], camera, FOV)
                if point is not None:
                    nearest = min(math.dist(point, rest_left),
                                  math.dist(point, rest_right))
                    metres = nearest
                    side = "left" if point[1] > 0.0 else "right"
            brainlog.log(
                "act", f"locate({target})", pixel=pixel, present=found.get("present"),
                metres=round(metres, 2), head_yaw_off=dyaw,
                quadrant=found.get("quadrant"), error=found.get("error") or "",
            )
            if pixel and MIN_OBJ_M <= metres <= MAX_OBJ_M:
                seen = (pixel, png)
                brainlog.log("think", f"the {target} is {metres:.2f} m from my hand; "
                                      f"reaching with the {side}", side=side)
                break
            if pixel:
                brainlog.log("think", f"that pixel is {metres:.2f} m away - not the "
                                      f"{target}; looking elsewhere")
        if seen is None:
            brainlog.log("result", f"cannot see {target} within reach", ok=False)
            break
        pixel, png = seen
        mark(png, pixel, STATE / f"marked-{stamp}.png")
        started = time.monotonic()
        code = grab_pixel(pixel[0], pixel[1], side)
        seconds = round(time.monotonic() - started, 1)
        brainlog.log(
            "result", f"grab({target}) {'reached' if code == 0 else 'failed'}",
            ok=code == 0, code=code, seconds=seconds, pixel=pixel,
            marked=f"marked-{stamp}.png",
        )
        if code == 2:
            brainlog.log("think", f"the {target} is out of reach - the body cannot turn")
        break
    return spoken


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--for", type=float, default=900.0, dest="duration",
                    help="how long to keep listening, seconds")
    ap.add_argument("--max-s", type=float, default=25.0, dest="max_s")
    ap.add_argument("--pattern", default=ears.DEFAULT_PATTERN)
    ap.add_argument("--say", default="",
                    help="a line to pretend JobBot said, instead of listening")
    ap.add_argument("--plan-only", action="store_true",
                    help="hear and decide, but do not move the body")
    args = ap.parse_args(argv)

    started = time.monotonic()
    deadline = started + args.duration
    turns = 0
    quiet = 0
    waiting = False
    while True:
        budget = max(1.0, deadline - time.monotonic())
        spoken = one_turn(args.say, args.pattern, min(args.max_s, budget),
                          plan_only=args.plan_only)
        args.say = ""  # a typed line happens once
        if spoken.startswith("waiting"):
            if not waiting:
                brainlog.log("heard", spoken)
                waiting = True
            time.sleep(1.0)
        elif spoken:
            waiting = False
            turns += 1
            if args.plan_only:
                return 0
        else:
            quiet += 1
        if time.monotonic() + MARGIN_S >= deadline:
            break
    brainlog.log("result", f"{turns} turn(s), {quiet} quiet window(s), "
                          f"{time.monotonic() - started:.0f}s listening")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
