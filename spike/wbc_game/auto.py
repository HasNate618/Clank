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
    ready             -> the body stands up and goes back to its clip
    brain.think       -> the model chooses one act, from the still and the speech
    sense             -> locate plus the depth gate, when the act needs a pixel
    reach_vr          -> one continuous stream, closing inside it when the act is a grab

The ordering is the measurement: the model was never asked to choose, so nothing here reads the
sentence for a verb. Design and its measurements: `docs/superpowers/specs/2026-11-10-brain-tools.md`
and `research/brain-tools.md`. On this model the act is chosen correctly (object 3 of 3 stills);
the sense tool is chosen only sometimes, so the body guarantees the search for an act that needs
one rather than trusting a refusal to teach the model anything.

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

from clank import brain, brainlog, ears, pose_relay  # noqa: E402
from clank.body import Pose  # noqa: E402
from clank.brain import BrainConfig, Frame  # noqa: E402
from clank.depth_stream import Stream  # noqa: E402
from clank.player_scale import read as read_player_scale  # noqa: E402
from clank.projection import Fov  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
VENV = do_task.VENV
DASHBOARD = "http://127.0.0.1:8765"

#: Where the frames the brain looked at are kept, so a wrong grounding can be checked
#: after the fact instead of argued about.
STATE = pathlib.Path.home() / ".local/state/Clank/auto"

#: A session start takes about a minute; stop this long before the deadline.
MARGIN_S = 25.0

#: How long to ask for the rest pose, and how often. The clip answers in under a second and
#: the command's TTL is short, so it has to be asked for repeatedly.
REST_TICKS = 8
REST_PERIOD_S = 0.25

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
#: How many times one turn may look for a thing before it gives up on it. Two is enough for
#: "find it, then find it from another angle"; the third is a loop an operator has to watch, and
#: MEASURED 2026-11-10 the model does not learn from a refusal, so more ticks do not help it.
SENSE_BUDGET = 2
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
    # A restart alone leaves the body in whatever pose the last command pinned: measured
    # 2026-11-08, arms straight out in the hold clip's T-pose. `rest` is the only command
    # that means "back to the clip", so send it until the body is actually standing.
    for _ in range(REST_TICKS):
        reach_vr.send(verb="rest")
        time.sleep(REST_PERIOD_S)
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


def confirm(pattern: str, max_s: float = 20.0) -> str:
    """The game is the oracle, and it is free: when the player takes the card, JobBot stops
    asking for it and tells you to take a cartridge. Exit codes cannot say that - the reach
    closed 82 mm short today and reported SATURATED either way - but the script can.
    """
    try:
        hearing = ears.listen(pattern=pattern, max_s=max_s)
    except ears.EarsError as exc:
        brainlog.log("error", f"could not listen for the answer: {exc}")
        return ""
    if not hearing.ok:
        return ""
    heard = brain.transcribe(BrainConfig(), hearing.wav)
    if heard.error:
        brainlog.log("error", f"transcribe failed: {heard.error}")
        return ""
    if heard.empty:
        return ""
    brainlog.log("heard", heard.text, ms=heard.decode_ms, speech_s=hearing.speech_s,
                 cut=heard.ellipsis)
    return heard.text


def grab_pixel(x: float, y: float, side: str, camera: Pose | None = None,
               grab: bool = True) -> int:
    """Reach the pixel, and close if asked, in one process with no gap.

    `camera` is the pose the pixel was marked from. Without it the reach unprojects through
    whatever the head has drifted to by then, and the same pixel means a different ray.

    `grab=False` is the model's `reach` - a hand moved to a thing without closing. The close
    never happens in a second process: measured 2026-11-09, a 2 s gap let the hand drift 0.09 m,
    which is more than the grab radius, so a brain that decides to grab must decide it here.
    """
    argv = [
        VENV, str(HERE / "reach_vr.py"), f"{x:.1f}", f"{y:.1f}", "--side", side,
        "--no-rest", "--no-clamp", "--tol", "0.02", "--corrections", "4",
        "--gain", "0.6", "--guard", "0.25",
    ]
    if grab:
        argv.append("--grab")
    if camera is not None:
        argv += ["--camera", *[f"{v:.5f}" for v in (*camera.p, *camera.q_xyzw)]]
    return subprocess.call(argv)


def sense(target: str, memory: brain.WorkingMemory, stamp: str):
    """Run locate on a fresh still, gate it against depth, and remember what it found.

    This is the body's own sense step, and it runs when a motor needs a pixel - design A. The
    model chooses *what* to act on and the body still has to find out *where*. The model does
    choose `find` sometimes (1 of 3 stills measured), which is why the constitution offers it,
    but on this model it will not choose it when asked and re-issues a refused reach instead of
    correcting itself (3 of 3), so the loop cannot depend on the choice. `research/brain-tools.md`.

    Returns the sighting and, when it is usable, `(pixel, png, camera, side)` for the motor.
    """
    jpeg, size, png, depth_frame, camera = frame_and_jpeg()
    found = locate.locate(BrainConfig(), Frame(0, 0.0, jpeg, "live"), target, size)
    pixel = found.get("pixel_px")
    metres = 0.0
    side = "right"
    if pixel:
        # Two questions, two samples. The *median* around the pixel says whether this is an
        # object at all (a sign on a far wall measures metres and is refused); the *nearest*
        # is the card itself, a thin plane standing in front of that wall.
        # Both go through the declared player scale: a session told to report 1.43x would
        # otherwise place every sighting 1.43x past the object.
        declared = read_player_scale()
        centre = declared.pixel_point(depth_frame, pixel[0] / size[0], pixel[1] / size[1],
                                      camera, FOV)
        if centre is not None:
            metres = min(math.dist(centre, reach_vr.wrist("left")),
                         math.dist(centre, reach_vr.wrist("right")))
        point = declared.pixel_point(depth_frame, pixel[0] / size[0], pixel[1] / size[1],
                                    camera, FOV, nearest=True,
                                    min_m=reach_vr.MIN_SURFACE_M)
        if point is not None:
            side = "left" if point[1] > 0.0 else "right"
    outcome = "present"
    if not pixel or found.get("present") is False:
        # A refusal is a refusal: stage 3 now asks what is in the box and judges the answer in
        # code, so `present: False` means the box described itself as something else.
        outcome = "not_visible"
    elif metres < reach_vr.MIN_SURFACE_M:
        outcome = "not_visible"          # the robot's own glove sits at about 0.2 m
    elif metres > MAX_OBJ_M:
        outcome = "too_far"
    sighting = brain.Sighting(
        noun=target,
        present=outcome == "present",
        pixel=(pixel[0] / size[0], pixel[1] / size[1]) if pixel else None,
        pixel_px=(pixel[0], pixel[1]) if pixel else None,
        metres=round(metres, 3),
        outcome=outcome,
    )
    memory.remember(target, sighting)
    brainlog.log("act", f"locate({target})", pixel=sighting.pixel_px, outcome=outcome,
                 metres=round(metres, 2), side=side,
                 quadrant=found.get("quadrant"), error=found.get("error") or "")
    if outcome != "present":
        brainlog.log("think", f"the {target} is {outcome.replace('_', ' ')}"
                              + (f" at {metres:.2f} m" if metres else ""))
        return sighting, None
    assert pixel is not None      # "present" means locate gave a pixel and the gate passed it
    mark(png, pixel, STATE / f"marked-{stamp}.png")
    return sighting, ((float(pixel[0]), float(pixel[1])), png, camera, side)


def hands_holding() -> tuple[str, str]:
    """Which hands are closed, from the body's own state rather than from a guess."""
    closure = reach_vr.state().get("closure") or [0.0, 0.0]
    return tuple("holding" if float(c) >= 0.5 else "empty" for c in closure)  # type: ignore[return-value]


def one_turn(say: str = "", pattern: str = ears.DEFAULT_PATTERN,
             max_s: float = 25.0, plan_only: bool = False) -> str:
    """Hear, think, find, act. Returns the words heard ("" if the game was quiet).

    The model chooses the act: which skill, on which thing, with which hand. Nothing here reads
    the sentence for a verb - that was `task_intent`, and it made the brain a parser. What this
    function owns is the body: reset, the still, the depth gate, and the one measured way to
    find a pixel. See `docs/superpowers/specs/2026-11-10-brain-tools.md`.
    """
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

    if not plan_only and not ready():
        return spoken

    stamp = time.strftime("%H%M%S")
    memory = brain.WorkingMemory()
    last = "(none)"
    seen = None
    thought = None
    intent = None
    for _ in range(SENSE_BUDGET + 1):
        jpeg, size, png, _depth, _camera = frame_and_jpeg()
        sight = STATE / f"frame-{stamp}.png"
        sight.parent.mkdir(parents=True, exist_ok=True)
        sight.write_bytes(png)
        left, right = hands_holding()
        thought = brain.think(
            BrainConfig(max_tokens=160), brain.DEFAULT_SEED, Frame(0, 0.0, jpeg, "live"),
            spoken, [], memory, left=left, right=right, last=last,
        )
        if thought.plan.error:
            brainlog.log("error", f"the model did not answer: {thought.plan.error}")
            return spoken
        target = thought.plan.target or ""
        brainlog.log("think", f"plan: {thought.plan.skill}({target or '-'})",
                     side=thought.plan.side, frame=sight.name, note=thought.plan.note)
        if plan_only:
            return spoken
        if thought.sense == "locate":
            found, seen = sense(target, memory, stamp)
            last = brain.last_line("locate", "none", target, found.outcome)
            continue
        intent = thought.decision.intent
        if intent.skill in ("reach", "grab") and target:
            known = memory.find(target)
            if known is None or not known.present:
                # The model chose the act; the body has to find the thing before it can do it.
                # This is not a hidden skill: `locate` here is what "reach the card" means when
                # nothing has measured where the card is.
                brainlog.log("think", f"the {target} is not found yet - finding it")
                found, seen = sense(target, memory, stamp)
                last = brain.last_line("locate", "none", target, found.outcome)
                thought.decision = brain.assemble(
                    thought.plan, None, 0, 0, memory=memory, left=left, right=right)
                intent = thought.decision.intent
                if found.outcome != "present":
                    continue
        if not thought.decision.accepted:
            brainlog.log("think", f"holding: {thought.decision.reason}")
        break

    if intent is None or thought is None:
        return spoken
    skill = intent.skill
    # `thought` cannot be None here, and saying so once keeps every later `thought.plan` honest
    # instead of a chain of "is not a known attribute of None" warnings that hide a real one.
    assert thought is not None
    if skill in ("hold", "rest"):
        brainlog.log("act", "hold")
        return spoken
    if skill == "look":
        yaw_deg = math.degrees(intent.yaw_rad or 0.0)
        look(yaw_deg, "right")
        brainlog.log("act", f"look({yaw_deg:+.0f} deg)")
        return spoken
    if skill == "menu":
        # JobBot asks a player to press the controller's menu button, and no joint of the
        # simulated body can press it: the press goes out on the wire as a short system_click
        # from the relay, which is the same path the dashboard button uses.
        request = pose_relay.request_click(intent.side)
        brainlog.log("act", f"menu({intent.side})", seq=request.get("seq"))
        return spoken
    if skill == "release":
        reach_vr.send(verb=f"open_{intent.side}")
        brainlog.log("act", f"release({intent.side})",
                     closed=reach_vr.state().get("closure"))
        return spoken
    if skill not in ("reach", "grab"):
        brainlog.log("think", f"{skill} is not wired yet")
        return spoken
    if seen is None:
        brainlog.log("result", f"cannot {skill} the {thought.plan.target or 'thing'} - "
                               "nothing was found", ok=False)
        return spoken

    pixel, png, camera, side = seen
    started = time.monotonic()
    code = grab_pixel(pixel[0], pixel[1], side, camera, grab=skill == "grab")
    seconds = round(time.monotonic() - started, 1)
    brainlog.log("result", f"{skill}({thought.plan.target or '-'}) "
                           f"{'reached' if code == 0 else 'failed'}",
                 ok=code == 0, code=code, seconds=seconds, pixel=pixel, hand=side,
                 marked=f"marked-{stamp}.png")
    if skill != "grab":
        return spoken
    # The game says whether it worked. JobBot moves on to the cartridges once the card is
    # taken, and repeats himself if it is not - the only oracle that sees the grab itself.
    if not say:
        spoken_after = confirm(pattern)
        if spoken_after:
            moved_on = "cartridge" in spoken_after.lower()
            brainlog.log(
                "result",
                "the card is in my hand - JobBot moved on to the cartridges"
                if moved_on else f"JobBot said something else: {spoken_after[:90]}",
                ok=moved_on,
            )
        else:
            brainlog.log("result", f"no word from JobBot after the grab; reach said code {code}",
                         ok=False)
    if code == 2:
        brainlog.log("think", f"the {thought.plan.target or 'thing'} is out of reach - "
                              "the body cannot turn")
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
