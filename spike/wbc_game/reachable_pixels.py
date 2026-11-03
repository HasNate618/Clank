"""Which pixels can the right wrist actually reach, from here?

A marked pixel is a ray, and the arm covers some rays and not others. Rather than
servo at a mark and discover afterwards that it sat outside the arm's cone (measured
2026-11-01: a lever knob 0.4 m of lateral travel away, with the wrist projecting to
(11.4, 6.2) - off the image, so the servo was blind), project a grid of candidate
wrist poses and draw the region they cover.

The mask is honest about its own limits: it is the **workspace clamp**, not the arm.
The clamp is a stability limit (measured residuals 0.05-0.19 m inside it), so green
means "the command will be accepted here", not "the hand will arrive exactly".
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from clank.body import Pose  # noqa: E402
from clank.projection import Fov, project_point  # noqa: E402
from clank.workspace import WRIST_BOX  # noqa: E402

STATE = Path.home() / ".local/state/Clank/sessions/body/body_state.json"
FOV = Fov.from_degrees(70.0, 70.0, 57.11, 57.11)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--u", type=float, default=None, help="mark to test, if any")
    ap.add_argument("--v", type=float, default=None)
    ap.add_argument("--grid", type=int, default=28, help="samples per axis")
    ap.add_argument("--out", default="/tmp/reachable.jpg")
    args = ap.parse_args()

    st = json.loads(STATE.read_text())
    head = Pose(p=tuple(st["head"]["p"]), q_xyzw=tuple(st["head"]["q_xyzw"]))
    wrist = tuple(st["right_hand"]["p"])

    n = args.grid
    xs = [WRIST_BOX.x[0] + (WRIST_BOX.x[1] - WRIST_BOX.x[0]) * i / (n - 1) for i in range(n)]
    ys = [WRIST_BOX.y[0] + (WRIST_BOX.y[1] - WRIST_BOX.y[0]) * i / (n - 1) for i in range(n)]
    zs = [WRIST_BOX.z[0] + (WRIST_BOX.z[1] - WRIST_BOX.z[0]) * i / (n - 1) for i in range(n)]

    mark = None if args.u is None or args.v is None else (args.u, args.v)
    best = None
    covered = 0
    points: list[tuple[float, float]] = []
    for x in xs:
        for y in ys:
            for z in zs:
                uv = project_point(head, (x, y, z), FOV)
                if uv is None or not (0.0 <= uv[0] <= 1.0 and 0.0 <= uv[1] <= 1.0):
                    continue
                covered += 1
                points.append(uv)
                if mark is not None:
                    d = (((uv[0] - mark[0]) * 1920) ** 2 + ((uv[1] - mark[1]) * 1080) ** 2)
                    if best is None or d < best[0]:
                        best = (d, (x, y, z), uv)

    print(f"wrist now ({wrist[0]:+.3f},{wrist[1]:+.3f},{wrist[2]:+.3f})  "
          f"grid points in view {covered}/{n ** 3}")
    if best is not None:
        d, p, uv = best
        print(f"nearest reachable wrist to the mark: ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})"
              f" -> pixel ({uv[0]:.3f},{uv[1]:.3f}), {d ** 0.5:.0f} px away")
        travel = sum((p[k] - wrist[k]) ** 2 for k in range(3)) ** 0.5
        print(f"  that is {travel:.3f} m of travel from where the wrist is")

    frame = REPO / "/tmp/reach_mask_src.png"
    subprocess.run([str(REPO / "spike/e3/.venv/bin/python"),
                    str(REPO / "spike/wbc_game/x11_grab.py"), "--out", str(frame)],
                   cwd=REPO, capture_output=True, timeout=120, check=True)

    from PIL import Image, ImageDraw

    im = Image.open(frame).convert("RGB")
    w, h = im.size
    mask = Image.new("RGB", (w // 8, h // 8), (0, 0, 0))
    md = ImageDraw.Draw(mask)
    seen = set()
    for u, v in points:
        cell = (int(u * (w // 8)), int(v * (h // 8)))
        if cell in seen:
            continue
        seen.add(cell)
        md.ellipse([cell[0] - 2, cell[1] - 2, cell[0] + 2, cell[1] + 2], fill=(0, 255, 90))
    im = Image.blend(im, mask.resize((w, h)).filter(__import__("PIL.ImageFilter", fromlist=["x"]).GaussianBlur(9)), 0.45)
    d = ImageDraw.Draw(im)
    uv = project_point(head, wrist, FOV)
    if uv is not None and 0.0 <= uv[0] <= 1.0 and 0.0 <= uv[1] <= 1.0:
        d.line([(uv[0] * w - 30, uv[1] * h), (uv[0] * w + 30, uv[1] * h)], fill=(0, 200, 255), width=4)
        d.line([(uv[0] * w, uv[1] * h - 30), (uv[0] * w, uv[1] * h + 30)], fill=(0, 200, 255), width=4)
        d.text((uv[0] * w + 10, uv[1] * h - 24), "wrist", fill=(0, 200, 255))
    else:
        d.text((20, 20), "wrist is outside this view", fill=(0, 200, 255))
    if mark is not None:
        mx, my = mark[0] * w, mark[1] * h
        d.line([(mx - 34, my), (mx + 34, my)], fill=(255, 230, 0), width=4)
        d.line([(mx, my - 34), (mx, my + 34)], fill=(255, 230, 0), width=4)
    im.save(args.out, quality=90)
    im.resize((1100, int(1100 * h / w))).save(str(args.out).replace(".jpg", "_small.jpg"), quality=90)
    print(f"  mask -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
