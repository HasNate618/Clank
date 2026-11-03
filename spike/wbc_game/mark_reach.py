"""Mark a pixel in a captured frame, then drive the right wrist onto it.

The operator names a pixel; this turns it into a point and moves the arm there. Two
facts shape the design:

* A pixel is a ray, not a point. `unproject` needs a depth, and *any* depth along the
  ray projects back to the same pixel - so the hand will look like it is on the object
  at every depth it tries. Depth is therefore not solvable from one image, and this
  tool takes it as an argument rather than pretending to measure it.
* The body moves the camera. Every `reach` shifts the head, and the root drifts, so a
  root-relative point does not stay on a world object. The anchor is therefore stored
  in **world** coordinates and re-derived each step from the current root.

Each step captures a frame and draws both crosshairs: the anchor (where the object
should be) and the wrist (where the hand is). If the anchoring is right, the anchor
crosshair stays on the object while the camera moves.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from clank.body import Pose  # noqa: E402
from clank.projection import Fov, _rotate, _rotate_inverse, project_point, unproject  # noqa: E402
from clank.workspace import clamp_wrist, in_workspace  # noqa: E402

SOCK = Path.home() / ".local/state/Clank/sessions/body/control.sock"
STATE = Path.home() / ".local/state/Clank/sessions/body/body_state.json"
FOV = Fov.from_degrees(70.0, 70.0, 57.11, 57.11)


def send(**kw: object) -> str:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(2.0)
        s.connect(str(SOCK))
        s.sendall((json.dumps(kw) + "\n").encode())
        try:
            return s.recv(255).decode().strip()
        except TimeoutError:
            return "-"


def state() -> dict:
    return json.loads(STATE.read_text())


def root_xyzw(st: dict) -> tuple[float, float, float, float]:
    """body_state stores root_q as (w, x, y, z); `_rotate` wants (x, y, z, w)."""
    w, x, y, z = st["root_q"]
    return (x, y, z, w)


def grab(path: Path) -> Path:
    subprocess.run(
        [str(REPO / "spike/e3/.venv/bin/python"), str(REPO / "spike/wbc_game/x11_grab.py"),
         "--out", str(path)],
        cwd=REPO, capture_output=True, timeout=120, check=True,
    )
    return path


def overlay(src: Path, dst: Path, marks: list[tuple[str, tuple[float, float], str]]) -> None:
    from PIL import Image, ImageDraw

    im = Image.open(src).convert("RGB")
    d = ImageDraw.Draw(im)
    w, h = im.size
    for label, (u, v), colour in marks:
        x, y = u * w, v * h
        d.line([(x - 34, y), (x + 34, y)], fill=colour, width=4)
        d.line([(x, y - 34), (x, y + 34)], fill=colour, width=4)
        d.ellipse([x - 8, y - 8, x + 8, y + 8], outline=colour, width=3)
        d.text((x + 12, y - 26), label, fill=colour)
    im.save(dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--u", type=float, required=True, help="normalised x, 0=left")
    ap.add_argument("--v", type=float, required=True, help="normalised y, 0=top")
    ap.add_argument("--depth", type=float, default=0.45, help="forward distance along gaze (m)")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--tag", default="target")
    ap.add_argument("--yaw", type=float, default=0.0,
                    help="head yaw the pixel was marked at (deg); pinned on every reach")
    args = ap.parse_args()

    # Go to the marking camera first. `look` rebuilds the head from the clip, so this
    # is the same view every run - a pixel is only valid from its own orientation.
    send(verb="look", yaw_deg=args.yaw)
    time.sleep(3.0)
    st = state()
    head = Pose(p=tuple(st["head"]["p"]), q_xyzw=tuple(st["head"]["q_xyzw"]))
    p_root = unproject((args.u, args.v), head, FOV, args.depth)
    if p_root is None:
        print("pixel is at or behind the image plane - no answer")
        return 2
    inside = in_workspace(p_root)
    p_root = clamp_wrist(p_root)
    w_point = [st["root_p"][k] + _rotate(root_xyzw(st), p_root)[k] for k in range(3)]
    print(f"{args.tag}: pixel ({args.u:.3f},{args.v:.3f}) at {args.depth:.2f} m "
          f"-> root ({p_root[0]:+.3f},{p_root[1]:+.3f},{p_root[2]:+.3f})"
          f"{'' if inside else ' CLAMPED'}")

    offset = [0.0, 0.0, 0.0]
    frames: list[Path] = []
    for i in range(args.steps):
        st = state()
        head = Pose(p=tuple(st["head"]["p"]), q_xyzw=tuple(st["head"]["q_xyzw"]))
        wrist = tuple(st["right_hand"]["p"])
        # Re-derive the world anchor in the robot's current root frame.
        back = _rotate_inverse(root_xyzw(st), [w_point[k] - st["root_p"][k] for k in range(3)])
        target = clamp_wrist(back)
        err = [target[k] - wrist[k] for k in range(3)]
        dist = sum(e * e for e in err) ** 0.5
        anchor_uv = project_point(head, target, FOV)
        wrist_uv = project_point(head, wrist, FOV)
        raw = grab(REPO / f"/tmp/mr_{i}.png")
        marks = [(f"{args.tag}", anchor_uv, (255, 230, 0))] if anchor_uv else []
        if wrist_uv:
            marks.append(("wrist", wrist_uv, (0, 255, 90)))
        out = REPO / f"/tmp/mr_{i}_ov.png"
        overlay(raw, out, marks)
        frames.append(out)
        dpx = (None if (anchor_uv is None or wrist_uv is None)
               else ((anchor_uv[0] - wrist_uv[0]) * 1920, (anchor_uv[1] - wrist_uv[1]) * 1080))
        print(f"  {i}: err {dist:.3f} m  wrist ({wrist[0]:+.3f},{wrist[1]:+.3f},{wrist[2]:+.3f})"
              f"  pixel gap {None if dpx is None else (round(dpx[0]), round(dpx[1]))}")
        if dist < 0.04:
            print("  arrived")
            break
        offset = [offset[k] + err[k] for k in range(3)]
        # look_yaw_deg pins the head, so the body cannot drag the camera off the
        # object while the arm moves. Without it each reach moved the head 0.101 m.
        send(verb="reach", side="right", forward_m=round(offset[0], 3),
             lateral_m=round(offset[1], 3), up_m=round(offset[2], 3),
             look_yaw_deg=args.yaw)
        time.sleep(1.6)

    if frames:
        from PIL import Image

        sheet = Image.new("RGB", (640 * len(frames), 360), (12, 12, 12))
        for i, f in enumerate(frames):
            sheet.paste(Image.open(f).resize((640, 360)), (640 * i, 0))
        sheet.save(f"/tmp/mr_sheet_{args.tag}.jpg", quality=90)
        print(f"  montage /tmp/mr_sheet_{args.tag}.jpg")
    st = state()
    print("  closure", st.get("closure"), "flags", st.get("flags"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
