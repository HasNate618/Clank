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
    reach_ardy        -> a generated clip: the hand goes to the mark, closing inside it on a grab
                      (reach_vr carries the left hand, and the act when the generator refuses)

The generated path is the session's path now: `reach_ardy` aims a constraint at the glove, ARDY
writes 52 frames, and SONIC tracks them. `reach_vr` is the fallback, for three measured reasons -
the generator pins the right hand only, it needs torch and a CUDA device, and its preflight can
refuse. All three are decided before it moves, so falling back costs a reach, not a half-reach.

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
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass, replace
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "wbc"))

import ardy_client  # noqa: E402  # pyright: ignore[reportMissingImports]
import do_task  # noqa: E402
import fade  # noqa: E402
import locate  # noqa: E402
import place  # noqa: E402
import reach_vr  # noqa: E402

import math  # noqa: E402

from clank import brain, brainlog, ears, pose_relay, workspace  # noqa: E402
from clank.body import Pose  # noqa: E402
from clank.brain import BrainConfig, Frame  # noqa: E402
from clank.depth_stream import Stream  # noqa: E402
from clank.player_scale import read as read_player_scale  # noqa: E402
from clank.projection import Fov  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
VENV = do_task.VENV
DASHBOARD = "http://127.0.0.1:8765"

#: The generated path needs torch and CUDA, so it does not run under the body's venv.
#: On NixOS `libcuda.so.1` is off the loader path and torch then reports `cuda False` on a
#: working driver, which is why the interpreter comes with an environment and not alone.
ARDY_PY = HERE.parent / "ardy/.venv/bin/python"
ARDY_LD = "/run/opengl-driver/lib"
#: `reach_ardy` exits 3 when it cannot generate at all - no clip ring, no right-hand support,
#: or a preflight refusal - and every one of those happens before it moves the body.
ARDY_UNSUPPORTED = 3

#: Where the frames the brain looked at are kept, so a wrong grounding can be checked
#: after the fact instead of argued about. The dashboard gives each run its own
#: directory (`CLANK_AUTO_STATE`), so run N's frames cannot be mistaken for run N+1's;
#: a hand-run keeps the shared default.
STATE = pathlib.Path(
    os.environ.get("CLANK_AUTO_STATE")
    or pathlib.Path.home() / ".local/state/Clank/auto"
)

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
#: The game opens its audio stream a second or two after its window appears, and `route()`
#: refuses to run against nothing, so the loop waits for it: 10 s in 1 s steps.
ROUTE_TRIES = 10
ROUTE_WAIT_S = 1.0

#: The run's one recorder. Started before the body reset, because that is when JobBot speaks
#: his opening line: measured, 12.8 s of it played while the body was being reset and the loop
#: was not listening. None means there is no ring (no parecord, or a plan-only run) and every
#: question falls back to opening its own window, as it used to.
RING: "ears.Ring | None" = None

#: Where a noun was looked for and refused, per run. The next search for the same noun skips
#: those places. MEASURED, run 193516-ca1fde: seven attempts, one picture, one question, the same
#: console every time - because nothing carried the refusal forward.
REFUSED: dict[str, list[tuple[float, float, float, float]]] = {}

#: What the last turn failed to find, and the act it was for. The next quiet window spends itself
#: on one re-search for it instead of doing nothing. MEASURED, run 204320-0213a5: locate refused
#: the cartridge at 20:44:07 and the run then took no action at all, because `one_turn` returns an
#: empty Turn whenever the gate is quiet. A refusal is not the end of the initiative.
PENDING_SEARCH: "tuple[str, str] | None" = None

SENSE_BUDGET = 2
#: A grounded point outside this distance from the *resting hand* is not the object being
#: reached for. The bound is reach_vr's own: it refuses past MAX_PLAUSIBLE_M from the hand,
#: and measuring from the camera instead let a 0.9 m-away pixel pass here and be refused
#: there. Nothing is grabbable behind you either.
MIN_OBJ_M = 0.10
MAX_OBJ_M = 1.0

#: What the grounder's still is faded to: the same band the depth gate accepts, so the model
#: cannot be shown a world the body then judges by different numbers.
#:
#: MEASURED 2026-11-13 on the run that could not find JobBot's card. Raw still: quadrant
#: BOTTOM-LEFT, refused, 3/3. The fade module's own default (1.06-1.45x the nearest clutter =
#: 0.35-0.48 m here) blackened everything past 0.48 m, which is where the card was: BOTTOM-LEFT,
#: refused, 3/3. This band (0.28-1.00 m) blackens the robot's own glove and the room but keeps
#: the card: BOTTOM-RIGHT, 3/3. Same model, same frame, same noun - the band was the whole
#: difference.
GROUNDING_BAND_M = (reach_vr.MIN_SURFACE_M, MAX_OBJ_M)


@dataclass
class Turn:
    """What one pass through the loop produced, including what the next pass must not re-hear."""

    spoken: str = ""
    #: JobBot's reply after a grab. It is the next instruction, already transcribed, so the next
    #: turn plans from it instead of listening for it again - and instead of missing it, which is
    #: what happened to "Here's my card" when a capture dropout ate the pause.
    pending: str = ""
    #: The grab's exit code, or None if nothing was grabbed. It decides whether the next turn
    #: pays 5.8 s for a body reset: a reset after a successful grab opens the hands.
    code: int | None = None


def _body_needs_reset(code: int | None) -> bool:
    """Reset is recovery, not a schedule.

    MEASURED 2026-11-13: `ready()` costs 5.8 s per turn, moves the root yaw (~+13 deg, varying
    run to run) and clears the freeze latch, so a reset before every turn changes the goal it is
    about to aim at. It is worth paying after a failed motor or a dead socket, and not after a
    grab that worked - a reset there re-opens the hand that is holding the card.
    """
    if code is not None and code != 0:
        return True
    return not reach_vr.SOCK.exists()


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
    started = time.monotonic()
    if mode:
        _post("/api/session/mode", {"mode": mode})
    brainlog.log("act", "reset body", mode=mode or "unchanged")
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
    # "done", because this is the same reset the row above announced: two rows reading
    # "reset body" five seconds apart look like two resets, and were read as two.
    brainlog.log("act", "reset body done", seconds=round(time.monotonic() - started, 1))
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


def faded_for_grounding(png: bytes, depth_frame: dict) -> bytes:
    """The still the grounder sees: everything outside `GROUNDING_BAND_M` replaced by a backdrop.

    The planner keeps the raw still - it has to see the whole room to name a destination - and
    the grounder gets this, because finding one thing in a white-on-white museum is a search
    through clutter, and the clutter is what pulled the model onto JobBot's body instead of the
    card in his hand.

    Returns the input unchanged when there is no depth: a frame that cannot be faded is still
    groundable, just worse, and refusing to look is worse than that.
    """
    import numpy as np
    from PIL import Image

    near_m, far_m = GROUNDING_BAND_M
    depth = depth_frame.get("depth")
    if not depth:
        return png
    try:
        rgb = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
        depth_m = np.asarray(depth, dtype=np.float32).reshape(
            int(depth_frame.get("height") or 270), int(depth_frame.get("width") or 480))
        # `fade` takes near/far as multiples of a target; with target 1.0 they are metres.
        out = fade.fade(rgb, depth_m, 1.0, near=near_m, far=far_m)
    except (ValueError, TypeError) as exc:
        brainlog.log("think", f"no fade for grounding: {exc}")
        return png
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


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


def confirm(pattern: str, max_s: float = 20.0, after: int | None = None) -> str:
    """The game is the oracle, and it is free: when the player takes the card, JobBot stops
    asking for it and tells you to take a cartridge. Exit codes cannot say that - the reach
    closed 82 mm short today and reported SATURATED either way - but the script can.

    A fixed window through `ears.record`, not `ears.listen`: the gate's job is "wait for the
    next line", and this question is "what did the game just say". The game talks over the
    grab, so a listen that opens on JobBot already talking measures its floor on the speech
    and gates itself deaf. MEASURED 2026-11-13, run 190131-679de2: floor 2445, gate 9781, peak
    4415, and the diary wrote "no word from JobBot" over an audible
    "You grab one of the job cartridges off the table and let's get started."
    """
    if after is not None and RING is not None:
        # The act's own seconds, read back out of the ring. This is the fix for the measured
        # miss: JobBot's reply ("...grab one of the job cartridges off the table...") ran for
        # 14.9 s *during* a generated reach, and this window opened one second after it ended.
        # Not `read_from`: that stops at the clock, which is mid-sentence. MEASURED 2026-11-14,
        # run 025412-cf7309: the reply ran 12.6 s and this read at 5 s into it, so the tail
        # became the next turn's instruction ("Google and let's get started").
        pcm = RING.read_utterance(after, max_s=max_s)
        # Reading is consuming: the utterance just transcribed must not be heard again by the
        # next `listen`. MEASURED, run 193516-ca1fde: without this the same sentence arrived
        # three times - once as the act's reply, then again whenever a turn resumed listening -
        # and every copy went into the session memory and back into the planner's prompt.
        RING.cursor = RING.mark()
        if not pcm:
            brainlog.log("result", "nothing in the ring across the act", ok=False)
            return ""
        wav = ears.wav_bytes(pcm)
        heard = brain.transcribe(BrainConfig(), wav)
        if heard.error:
            brainlog.log("error", f"transcribe failed: {heard.error}")
            return ""
        if heard.empty:
            return ""
        brainlog.log("heard", heard.text, ms=heard.decode_ms,
                     seconds=len(pcm) / (ears.RATE * 2), source="act", cut=heard.ellipsis)
        return heard.text
    try:
        hearing = ears.record(max_s, pattern=pattern)
    except ears.EarsError as exc:
        brainlog.log("error", f"could not record the answer: {exc}")
        return ""
    if not hearing.wav:
        brainlog.log("result", "no audio at all when JobBot was meant to answer", ok=False)
        return ""
    heard = brain.transcribe(BrainConfig(), hearing.wav)
    if heard.error:
        brainlog.log("error", f"transcribe failed: {heard.error}")
        return ""
    if heard.empty:
        return ""
    brainlog.log("heard", heard.text, ms=heard.decode_ms, seconds=hearing.seconds,
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

    The right hand goes to the generated path and the left to the cruise: `ardy_gen` pins the
    right hand only (`RightHandConstraintSet`), so a left reach would be refused anyway. A
    generator that refuses before moving - no CUDA, no clip ring - falls back to the cruise
    rather than failing the act, because on this machine the GPU is often off and the game does
    not care how the hand arrived.
    """
    if side == "right" and ARDY_PY.exists():
        brainlog.log("act", "generated reach")
        code = subprocess.call(_generated_argv(x, y, side, camera, grab), env=_ardy_env())
        if code != ARDY_UNSUPPORTED:
            return code
        brainlog.log("act", "no generated reach here; the cruise carries it", code=code)
    brainlog.log("act", f"cruise reach ({side})")
    return subprocess.call(_cruise_argv(x, y, side, camera, grab))


def warm_generator() -> None:
    """Start the resident generator now, so the first reach is not the one that pays for it.

    Best effort, like everything else about the generator: if it cannot come up, each reach
    loads it in its own process and says so, which is what happened before this existed. A run
    must never be delayed by a nicety, nor failed by one.
    """
    sock = reach_vr.SOCK.parent / ardy_client.SOCKET_NAME
    spawn = [
        str(ARDY_PY), str(HERE.parent / "wbc" / "ardy_daemon.py"), "--socket", str(sock),
    ]
    try:
        # With `_ardy_env()`, and that is the whole point of this being one place: measured, a
        # daemon spawned without `LD_LIBRARY_PATH=/run/opengl-driver/lib` reports `cuda False`,
        # generates on the CPU, and hands every later reach a `peak_vram_gb` of `None`. The
        # reaches each carried the path; the resident generator - the one that does the work -
        # did not.
        up = ardy_client.ensure(sock, spawn, wait_s=30.0, env=_ardy_env())
    except OSError as exc:
        brainlog.log("act", f"generator not resident: {exc}")
        return
    if up:
        brainlog.log("act", "generator resident")
    else:
        brainlog.log("act", "generator not resident; each reach will load it")


def _start_ring(pattern: str) -> "ears.Ring | None":
    """Begin the run's one recorder, or return None so every window falls back to its own.

    A ring that will not start is a degradation, not a failure: the loop then records per
    question exactly as it did before, which hears less but still hears.
    """
    # The run opens the recorder before it relaunches the game, because the relaunch is what
    # makes the game talk: MEASURED 2026-11-14, run 025412-cf7309, a ring started here began at
    # 02:54:28 and the greeting's first words were never in the file. Reading it is this loop's
    # whole part in it - one writer, one reader.
    given = os.environ.get("AUDIO_RING_PATH", "").strip()
    if given and pathlib.Path(given).exists():
        attached = ears.Ring(pathlib.Path(given), pattern=pattern)
        brainlog.log("act", f"audio ring {attached.path.name} from the run "
                            f"({attached.available_s():.0f} s already)")
        return attached
    # A second recorder must not open the run's file: parecord truncates, and MEASURED
    # 2026-11-14, run 031829-b1a009, that is how the greeting lost its first two words a second
    # time - the run's ring was opened, then this one opened the same path over it.
    path = (STATE.parent / "audio-ring-loop.raw" if given
            else STATE.parent / "audio-ring.raw")
    ring = ears.Ring(path, pattern=pattern)
    if not ring.start():
        brainlog.log("error", f"no audio ring: {ring.error}; each window records its own")
        return None
    brainlog.log("act", f"audio ring {path.name} from {ring.device}")
    return ring


def _ardy_env() -> dict[str, str]:
    """`LD_LIBRARY_PATH` carrying `libcuda.so.1`, prepended rather than overwritten."""
    env = dict(os.environ)
    carried = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = f"{ARDY_LD}:{carried}" if carried else ARDY_LD
    return env


def _marked_camera(camera: Pose | None) -> list[str]:
    return [] if camera is None else [
        "--camera", *[f"{v:.5f}" for v in (*camera.p, *camera.q_xyzw)]
    ]


def _generated_argv(x: float, y: float, side: str, camera: Pose | None,
                    grab: bool) -> list[str]:
    """No `--tol`: the default is the policy's own `TRACK_SUCCESS_M`.

    The cruise call's tighter 0.02 would have reported the live grab (24.7 mm achieved) as a
    failure, and a generated reach is graded on the glove, which the script measures itself.
    """
    argv = [
        str(ARDY_PY), str(HERE / "reach_ardy.py"), f"{x:.1f}", f"{y:.1f}", "--side", side,
    ]
    if grab:
        argv.append("--grab")
    return argv + _marked_camera(camera)


def _cruise_argv(x: float, y: float, side: str, camera: Pose | None,
                 grab: bool) -> list[str]:
    argv = [
        VENV, str(HERE / "reach_vr.py"), f"{x:.1f}", f"{y:.1f}", "--side", side,
        "--no-rest", "--no-clamp", "--tol", "0.02", "--corrections", "4",
        "--gain", "0.6", "--guard", "0.25",
    ]
    if grab:
        argv.append("--grab")
    return argv + _marked_camera(camera)


def _move_argv(offset: tuple[float, float, float], side: str,
               camera: Pose | None) -> list[str]:
    """A travel: the hand moves from where it is by `offset`, metres in the pelvis frame.

    **The cruise carries it, and only the cruise.** Measured 2026-11-13 on a live body, back to
    back, same hand and same 0.20 m request:

    | path | travel | 10 s after the process exits |
    | --- | --- | --- |
    | generated clip (`reach_ardy --move`) | 0.20 m, 67 mm residual | **back where it started** |
    | cruise (`reach_vr --move`) | 0.174 m, 7.4 mm residual | **still there** |

    The clip is tracked while it plays - the arm really does go back - and then the body returns
    to the *previous* reach's target: `CommandBus` retires its freeze-hold snapshot only on a real
    command, a deposited clip is not one, so the hold that follows a clip bakes a stale pose and
    the arm walks home. Re-sending the hold, pinning the achieved state into the verb, and giving
    the hold its own keepalive were each tried and each lost the arm within a second. A cruise
    re-commands an absolute target every keepalive, which is why it survives its own exit, and a
    travel needs nothing a generator adds: no object to plan around, no root step.

    `--grab` is passed: a pull that let go halfway would drag nothing.
    """
    forward, up, lateral = (float(value) for value in offset)
    metres = [f"{forward:.3f}", f"{up:.3f}", f"{lateral:.3f}"]
    return [VENV, str(HERE / "reach_vr.py"), "--move", *metres,
            "--side", side, "--no-rest", "--no-clamp", "--grab"] + _marked_camera(camera)


def _number(value: object, default: float = 0.0) -> float:
    """A float from a value another process wrote, or `default`.

    The body state and locate's reply are both written by code we control, but a half-written
    packet must not kill the loop that is watching the game - and a NaN read as a distance would
    put a reach somewhere nonsense.
    """
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def sense(target: str, memory: brain.WorkingMemory, stamp: str,
          avoid: "tuple[tuple[float, float, float, float], ...]" = ()):
    """Run locate on a fresh still, gate it against depth, and remember what it found.

    This is the body's own sense step, and it runs when a motor needs a pixel - design A. The
    model chooses *what* to act on and the body still has to find out *where*. The model does
    choose `find` sometimes (1 of 3 stills measured), which is why the constitution offers it,
    but on this model it will not choose it when asked and re-issues a refused reach instead of
    correcting itself (3 of 3), so the loop cannot depend on the choice. `research/brain-tools.md`.

    Returns the sighting and, when it is usable, `(pixel, png, camera, side, point)` for the
    motor. The point is the destination's own 3-D position, which is what `place` hovers above -
    a reach to the pixel would put the hand inside the console.
    """
    jpeg, size, png, depth_frame, camera = frame_and_jpeg()
    # The grounder looks at the faded still; the planner looked at the raw one, and the evidence
    # png stays raw so the marked image is what the game actually drew.
    found = locate.locate(BrainConfig(),
                          Frame(0, 0.0, faded_for_grounding(png, depth_frame), "live"),
                          target, size, avoid=avoid)
    pixel = found.get("pixel_px")
    metres = 0.0
    point = None
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
        # Its own name, because "not visible at 0.19 m" reads like a grounding failure and is
        # not one: measured, the thing at 0.19 m was the console the robot was standing at,
        # named correctly by stage 3 as "Job cartridge" off its INSERT CARTRIDGE panel.
        outcome = "too_close"
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
    if outcome != "present":
        refused_box = found.get("found_box")
        if refused_box:
            REFUSED.setdefault(target, []).append(tuple(refused_box))
    brainlog.log("act", f"locate({target})", pixel=sighting.pixel_px, outcome=outcome,
                 metres=round(metres, 2), side=side,
                 quadrant=found.get("quadrant"), error=found.get("error") or "")
    if outcome != "present":
        brainlog.log("think", f"the {target} is {outcome.replace('_', ' ')}"
                              + (f" at {metres:.2f} m" if metres else ""))
        return sighting, None
    assert pixel is not None      # "present" means locate gave a pixel and the gate passed it
    mark(png, pixel, STATE / f"marked-{stamp}.png")
    return sighting, ((_number(pixel[0]), _number(pixel[1])), png, camera, side, point)


def hands_holding() -> tuple[str, str]:
    """Which hands are closed, from the body's own state rather than from a guess."""
    closure = reach_vr.state().get("closure") or [0.0, 0.0]
    held = ["holding" if _number(c) >= 0.5 else "empty" for c in closure][:2]
    return (held + ["empty", "empty"])[:2]  # type: ignore[return-value]


def _proactive_retry(pattern: str, session) -> "Turn | None":
    """One re-search for a target the last turn was refused, before going back to listening.

    The constitution already says what to do with a refusal - "If find said it is not there, look"
    - and the body knows where it already looked: `REFUSED` holds the box, so `sense` sends the
    grounder to the quadrant beside it. One attempt only; `PENDING_SEARCH` is cleared before the
    search so a refusal here cannot become a loop.
    """
    global PENDING_SEARCH
    if PENDING_SEARCH is None:
        return None
    target, skill = PENDING_SEARCH
    PENDING_SEARCH = None
    stamp = time.strftime("%H%M%S")
    found, seen = sense(target, brain.WorkingMemory(), stamp,
                        avoid=tuple(REFUSED.get(target, ())))
    if found.outcome != "present" or seen is None:
        brainlog.log("result", f"{target} is still not there: {found.outcome}", ok=False)
        return Turn()
    pixel, _png, camera, side, _point = seen
    started = time.monotonic()
    code = grab_pixel(pixel[0], pixel[1], side, camera, grab=skill == "grab")
    brainlog.log("result", f"{skill}({target}) "
                           f"{'reached' if code == 0 else 'failed'} (asked again)",
                 ok=code == 0, code=code, seconds=round(time.monotonic() - started, 1),
                 pixel=pixel, hand=side)
    if session is not None:
        session.note_result(f"{skill}({target}) "
                            f"{'reached' if code == 0 else 'failed'} (asked again)")
    return Turn(code=code)


def _place(thought, seen, memory: brain.WorkingMemory, stamp: str,
           spoken: str, session: brain.SessionMemory | None) -> Turn:
    """`place` is a reach to a point that has nothing under it, then an open hand.

    Not a new motion: `research/reach-placement.md` measured a hover in free air landing 15-65 mm
    from the goal, and the release itself. The destination is located the same way the object is,
    because the model names it in words and the body still has to find it.

    A `place` with nothing in the hand is refused before it moves. That refusal is the one worth
    writing down: a place that quietly does nothing looks exactly like a place that worked.
    """
    plan = thought.plan
    held = [side for side, state in zip(("left", "right"), hands_holding()) if state == "holding"]
    if not held:
        brainlog.log("result", "place refused: hands empty", ok=False)
        if session is not None:
            session.note_result("place refused: hands empty")
        return Turn(spoken=spoken)
    destination = (plan.destination or plan.target or "").strip()
    if not destination:
        brainlog.log("result", "place refused: no destination", ok=False)
        if session is not None:
            session.note_result("place refused: no destination")
        return Turn(spoken=spoken)
    if seen is None or seen[4] is None:
        found, seen = sense(destination, memory, stamp)
        if session is not None:
            session.note_result(f"locate({destination}) {found.outcome}")
        if seen is None or seen[4] is None:
            brainlog.log("result", f"place refused: {destination} not found", ok=False)
            if session is not None:
                session.note_result(f"place refused: {destination} not found")
            return Turn(spoken=spoken)
    _pixel, _png, camera, _side, point = seen
    assert point is not None     # the guard above refuses a destination with no depth point
    side = held[0]          # the hand that is actually holding something
    goal = (point[0], point[1], point[2] + workspace.APPROACH_M)
    started = time.monotonic()
    wrist, gap = place.hover(goal, side, camera)
    reach_vr.send(verb=f"open_{side}")
    brainlog.log("result", f"place({destination}) above it, hand open", ok=True,
                 gap_m=round(gap, 3), seconds=round(time.monotonic() - started, 1),
                 hand=side, wrist=[round(v, 3) for v in wrist],
                 marked=f"marked-{stamp}.png")
    if session is not None:
        session.note_result(f"place({destination}) released above it, gap {gap:.3f} m")
    return Turn(spoken=spoken)


def one_turn(say: str = "", pattern: str = ears.DEFAULT_PATTERN,
             max_s: float = 25.0, plan_only: bool = False, *,
             session: brain.SessionMemory | None = None,
             pending_heard: str = "",
             last_code: int | None = None) -> Turn:
    """Hear, think, find, act. Returns what happened, including what the next turn inherits.

    The model chooses the act: which skill, on which thing, with which hand. Nothing here reads
    the sentence for a verb - that was `task_intent`, and it made the brain a parser. What this
    function owns is the body: the still, the depth gate, and the one measured way to find a
    pixel. See `docs/superpowers/specs/2026-11-10-brain-tools.md`.

    `session` is the run's conversation, and it is the reason a lost capture no longer loses the
    instruction: everything heard, asked, answered and done goes into it, and every planner call
    reads all of it. `pending_heard` is the previous turn's confirmed reply - JobBot's answer to
    the grab, already transcribed, which is the next instruction. The body is reset only when
    `last_code` says something went wrong; see `_body_needs_reset`.
    """
    if say:
        spoken = say
        brainlog.log("heard", spoken, source="typed")
    elif pending_heard:
        # JobBot's answer to the last act, captured as part of the confirmation rather than
        # listened for again. MEASURED 2026-11-13: "Here's my card" fell in a 0.6 s gap inside
        # the opening, so the pause rule cannot be the only way an instruction arrives.
        spoken = pending_heard
        # Nothing is written here. `confirm` already put this sentence in the diary beside the act
        # it answered, and a second row for it says the game spoke twice when it spoke once -
        # MEASURED, run 193516-ca1fde: one sentence, three `heard` rows. The instruction is
        # already visible; repeating it in another colour is noise.
    else:
        try:
            if RING is not None:
                hearing = RING.listen(max_s=max_s)
            else:
                hearing = ears.listen(pattern=pattern, max_s=max_s)
        except ears.EarsError as exc:
            return Turn(spoken=f"waiting: {exc}")
        if not hearing.ok:
            # The gate never opened: nothing was said, so there is no event to write. The
            # numbers go to the caller's quiet count, not to the diary. But a quiet window after
            # a refusal is not idle time: it is the only time the body has to look again.
            retry = _proactive_retry(pattern, session)
            return Turn() if retry is None else retry
        heard = brain.transcribe(BrainConfig(), hearing.wav)
        if heard.error:
            brainlog.log("error", f"transcribe failed: {heard.error}")
            return Turn()
        if heard.empty:
            # Speech over the gate that transcribed to nothing IS an event: it is the shape of
            # a silent miss, and the numbers are what turn it into a one-line diagnosis.
            brainlog.log("heard", "", ms=heard.decode_ms, speech_s=hearing.speech_s,
                         peak=hearing.peak, floor=round(hearing.floor, 1),
                         gate=round(hearing.gate, 1), ended_by=hearing.ended_by)
            return Turn()
        spoken = heard.text
        brainlog.log("heard", spoken, ms=heard.decode_ms, speech_s=hearing.speech_s,
                     peak=hearing.peak, cut=heard.ellipsis)

    # Everything from here to the end of the act is JobBot's answer to it, and `confirm` reads
    # it back out of the ring instead of opening a window once the act is over. MEASURED, run
    # 181738-ee1d6d: his reply ran 14.9 s during the reach and the old window opened after it.
    act_mark = RING.mark() if RING is not None else None

    if not plan_only and _body_needs_reset(last_code) and not ready():
        return Turn(spoken=spoken)

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
            BrainConfig(max_tokens=160), Frame(0, 0.0, jpeg, "live"),
            spoken, memory, left=left, right=right, last=last, session=session,
        )
        if thought.plan.error:
            brainlog.log("error", f"the model did not answer: {thought.plan.error}")
            return Turn(spoken=spoken)
        # Disambiguate in code before the body looks: "job cartridge" names both the rack and
        # the INSERT CARTRIDGE panel on the console 0.19 m in front of the robot. The planner
        # prompt cannot carry the label - measured 3/3, any prop named there outranks the
        # sentence the model was told to obey - so `clank.brain.refine_target` does it.
        planned_target = thought.plan.target or ""
        target = brain.refine_target(planned_target)
        if target != planned_target:
            brainlog.log("think", f"{planned_target} -> {target} (the label is in code)")
        # The plan carries the refined noun too, because `brain.assemble` reads it back out to ask
        # whether the thing has been seen - and a memory holding "STORE CLERK cartridge" cannot
        # answer a question about "job cartridge". MEASURED 2026-11-14, run 032621-8fa203: locate
        # found the cartridge at 0.52 m and the same turn held, refused as
        # "not_located: 'job cartridge' - never looked at it".
        thought.plan = replace(thought.plan, target=target)
        brainlog.log("think", f"plan: {thought.plan.skill}({target or '-'})",
                     side=thought.plan.side, frame=sight.name, note=thought.plan.note)
        if plan_only:
            return Turn(spoken=spoken)
        if thought.sense == "locate":
            found, seen = sense(target, memory, stamp, avoid=tuple(REFUSED.get(target, ())))
            last = brain.last_line("locate", "none", target, found.outcome)
            if session is not None:
                session.note_result(f"locate({target}) {found.outcome}"
                                    + (f" at {found.metres:.2f} m" if found.metres else ""))
            if found.outcome != "present":
                # A refusal is an answer. The old loop retried the identical question on an
                # identical frame at temperature 0 and got the identical answer, three times a
                # turn and again every turn: measured, seven attempts at one console. The
                # refusal is now in the session, where the planner reads it, and the place it
                # was found is in `REFUSED`, so the next search asks a different question.
                brainlog.log("result", f"{target} is not there: {found.outcome}"
                                       + (f" at {found.metres:.2f} m" if found.metres else ""),
                             ok=False)
                global PENDING_SEARCH
                PENDING_SEARCH = (target, "grab")
                return Turn(spoken=spoken)
            continue
        intent = thought.decision.intent
        # What the model ASKED for, not what the refusal handed back: a plan for an unlocated
        # object assembles into a hold, so reading the intent here would skip the search and
        # quietly turn "grab the card" into "do nothing". MEASURED 2026-11-10.
        asked = thought.plan.skill or ""
        if asked in ("reach", "grab") and target:
            known = memory.find(target)
            if known is None or not known.present:
                # The model chose the act; the body has to find the thing before it can do it.
                # This is not a hidden skill: `locate` here is what "reach the card" means when
                # nothing has measured where the card is.
                brainlog.log("think", f"the {target} is not found yet - finding it")
                found, seen = sense(target, memory, stamp,
                                    avoid=tuple(REFUSED.get(target, ())))
                last = brain.last_line("locate", "none", target, found.outcome)
                if session is not None:
                    session.note_result(f"locate({target}) {found.outcome}"
                                        + (f" at {found.metres:.2f} m" if found.metres else ""))
                thought.decision = brain.assemble(
                    thought.plan, None, 0, 0, memory=memory, left=left, right=right)
                intent = thought.decision.intent
                if found.outcome != "present":
                    # This `continue` was the repeat: it re-planned, re-located the same noun on
                    # the same picture and got the same answer, three times a turn, and then the
                    # next turn did it again. Measured, run 193516-ca1fde: seven attempts, one
                    # console. The refusal ends the turn; the place it was refused is remembered
                    # in `REFUSED`, so the next turn's search excludes it.
                    brainlog.log("result", f"{target} is not there: {found.outcome}"
                                           + (f" at {found.metres:.2f} m" if found.metres else ""),
                                 ok=False)
                    PENDING_SEARCH = (target, asked)
                    return Turn(spoken=spoken)
        if not thought.decision.accepted:
            brainlog.log("think", f"holding: {thought.decision.reason}")
        break

    if intent is None or thought is None:
        return Turn(spoken=spoken)
    if not thought.decision.accepted and intent.skill not in ("hold", "rest"):
        # A refused plan is not an action. MEASURED 2026-11-10: a menu press with no hand was
        # refused by assemble and then executed here anyway, which is the loop doing something
        # the brain did not validly ask for - the one class of bug this design cannot tolerate.
        brainlog.log("result", f"refused {intent.skill}: {thought.decision.reason}", ok=False)
        if session is not None:
            session.note_result(f"refused {intent.skill}: {thought.decision.reason}")
        return Turn(spoken=spoken)
    skill = intent.skill
    # `thought` cannot be None here, and saying so once keeps every later `thought.plan` honest
    # instead of a chain of "is not a known attribute of None" warnings that hide a real one.
    assert thought is not None
    if skill in ("hold", "rest"):
        brainlog.log("act", "hold")
        return Turn(spoken=spoken)
    if skill == "look":
        yaw_deg = math.degrees(intent.yaw_rad or 0.0)
        look(yaw_deg, "right")
        brainlog.log("act", f"look({yaw_deg:+.0f} deg)")
        return Turn(spoken=spoken)
    if skill == "menu":
        # JobBot asks a player to press the controller's menu button, and no joint of the
        # simulated body can press it: the press goes out on the wire as a short system_click
        # from the relay, which is the same path the dashboard button uses.
        request = pose_relay.request_click(intent.side)
        brainlog.log("act", f"menu({intent.side})", seq=request.get("seq"))
        return Turn(spoken=spoken)
    if skill == "release":
        reach_vr.send(verb=f"open_{intent.side}")
        brainlog.log("act", f"release({intent.side})",
                     closed=reach_vr.state().get("closure"))
        return Turn(spoken=spoken)
    if intent.offset is not None:
        # A travel needs nothing found: the hand is already on the thing and moves. Measured
        # 2026-11-13 (`research/brain-tools.md`): offered `move` as a typed tool, this model
        # answered "pull the lever to job" with `{"direction": "back", "metres": 0.5}` 3/3, so
        # the vocabulary is there - what was missing was a way for `clank.intent/1` to say it.
        started = time.monotonic()
        # Unpacked rather than sliced: `intent.head` is a 7-float tuple and `Pose` takes exactly
        # three and four, which a slice cannot promise a type checker.
        camera = None
        if intent.head is not None and len(intent.head) == 7:
            hx, hy, hz, qx, qy, qz, qw = intent.head
            camera = Pose(p=(hx, hy, hz), q_xyzw=(qx, qy, qz, qw))
        code = subprocess.call(_move_argv(intent.offset, intent.side or "right", camera))
        seconds = round(time.monotonic() - started, 1)
        where = tuple(round(float(value), 2) for value in intent.offset)
        brainlog.log("result", f"move{where} {'reached' if code == 0 else 'failed'}",
                     ok=code == 0, code=code, seconds=seconds, hand=intent.side)
        if session is not None:
            session.note_result(f"move{where} code={code}")
        return Turn(spoken=spoken, code=code)
    if skill == "place":
        return _place(thought, seen, memory, stamp, spoken, session)
    if skill not in ("reach", "grab"):
        brainlog.log("think", f"{skill} is not wired yet")
        return Turn(spoken=spoken)
    if seen is None:
        brainlog.log("result", f"cannot {skill} the {thought.plan.target or 'thing'} - "
                               "nothing was found", ok=False)
        return Turn(spoken=spoken)

    pixel, png, camera, side, _point = seen
    started = time.monotonic()
    code = grab_pixel(pixel[0], pixel[1], side, camera, grab=skill == "grab")
    seconds = round(time.monotonic() - started, 1)
    brainlog.log("result", f"{skill}({thought.plan.target or '-'}) "
                           f"{'reached' if code == 0 else 'failed'}",
                 ok=code == 0, code=code, seconds=seconds, pixel=pixel, hand=side,
                 marked=f"marked-{stamp}.png")
    if session is not None:
        session.note_result(f"{skill}({thought.plan.target or '-'}) "
                            f"{'reached' if code == 0 else 'failed'} code={code}")
    if skill != "grab":
        return Turn(spoken=spoken, code=code)
    # The game says whether it worked, and it is asked every time - including on a typed line,
    # because JobBot reacts to the *game* and not to how the instruction arrived. MEASURED
    # 2026-11-10: with the confirmation skipped under --say, the diary wrote "grab failed
    # (code=2)" for a grab the game had accepted and the operator watched succeed.
    spoken_after = confirm(pattern, after=act_mark)
    if session is not None and spoken_after:
        session.note_result(f"JobBot said: {spoken_after[:120]}")
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
    # What JobBot says after the act is the next instruction, already in hand.
    return Turn(spoken=spoken, code=code, pending=spoken_after)


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
    STATE.mkdir(parents=True, exist_ok=True)
    if not args.plan_only:
        warm_generator()

    # One session for the whole run: everything heard, asked, answered and done. One reset
    # here, before the first turn, and afterwards only when something went wrong.
    session = brain.SessionMemory()
    pending = ""
    last_code: int | None = None
    # Put the game behind the capture sink before listening to it, every run. Measured: nothing
    # else calls `ears.route()`, so a restarted game played to the operator's speakers with a
    # remembered `Mute: yes` on its new stream, and the loop sat deaf for six minutes while the
    # game talked. `route()` also clears that mute, which `module-stream-restore` reapplies to
    # every new stream. Idempotent. A game with no stream *yet* is a queue, not a failure - the
    # stream appears a second or two after the window - so this waits for it rather than dying
    # deaf, and a game that never routes is reported and the loop still runs.
    for attempt in range(ROUTE_TRIES):
        try:
            brainlog.log("act", f"route audio {ears.route(pattern=args.pattern)}")
            break
        except Exception as exc:  # noqa: BLE001 - no stream yet is a state, not a crash
            if attempt == ROUTE_TRIES - 1:
                brainlog.log("error", f"could not route the game's audio: {exc}")
            else:
                time.sleep(ROUTE_WAIT_S)

    # One recorder for the whole run, started here - before the reset, not after it. JobBot's
    # greeting arrives within seconds of the game's window and the body reset costs ~6 s, so a
    # recorder that starts when listening starts has already missed the opening line: measured,
    # 12.8 s of it played during the reset. `Ring` also takes the greeting out of the equation
    # for turn one, which replays the ring from its first byte.
    global RING
    ring = _start_ring(args.pattern)
    RING = ring
    if not args.plan_only and not ready():
        brainlog.log("error", "the body did not come back up; a turn may still find it")
    started = time.monotonic()
    deadline = started + args.duration
    turns = 0
    quiet = 0
    waiting = False
    while True:
        budget = max(1.0, deadline - time.monotonic())
        turn = one_turn(args.say, args.pattern, min(args.max_s, budget),
                        plan_only=args.plan_only, session=session,
                        pending_heard=pending, last_code=last_code)
        spoken = turn.spoken
        pending = turn.pending
        if turn.code is not None:
            last_code = turn.code
        args.say = ""  # a typed line happens once
        session.cap()
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
    if ring is not None:
        detail = ring.stop()
        if detail:
            brainlog.log("error", f"the audio ring said: {detail}")
    brainlog.log("result", f"{turns} turn(s), {quiet} quiet window(s), "
                          f"{time.monotonic() - started:.0f}s listening")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
