#!/usr/bin/env python3
"""Click the live game frame to name a pixel, instead of describing where something is.

The operator can see the game and the agent cannot, so the expensive part of every reach so far
has been guessing which pixels are the object: a depth local-minimum scan found the console's
corner, the vision model found a card 0.81 m away, and projecting candidates onto the handle found
a queue stanchion. All three were wrong and each cost a reach attempt.

This serves one frame on http://127.0.0.1:8771 with a labelled 100 px grid. Click the object, and
the pixel is written to /tmp/clank_pixel.json and printed. The agent then unprojects that pixel.

Run with the spike venv (PIL):  spike/e3/.venv/bin/python spike/wbc_game/pick_pixel.py
Optional: --port, --timeout, --no-reset
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "/home/nate/Projects/Clank")
sys.path.insert(0, "/home/nate/Projects/Clank/spike/wbc_game")

#: Outputs live under the app's own state directory, not /tmp: the file is handed between two
#: processes and /tmp is world-writable.
STATE = pathlib.Path.home() / ".local" / "state" / "Clank" / "pick-pixel"
OUT = STATE / "pixel.json"
FRAME = STATE / "frame.jpg"

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>pick a pixel</title>
<style>
 body{{margin:0;background:#111;color:#eee;font:14px/1.4 monospace}}
 #wrap{{position:relative;display:inline-block;margin:8px}}
 img{{display:block;width:960px;height:540px;cursor:crosshair}}
 canvas{{position:absolute;left:0;top:0;pointer-events:none}}
 #hud{{padding:8px 12px}}
</style></head><body>
<div id="hud">click the object. pixel = <span id="px">-</span>
 &nbsp;|&nbsp; grid is 100 px &nbsp;|&nbsp; <span id="st">waiting</span></div>
<div id="wrap">
 <img id="im" src="data:image/jpeg;base64,{b64}">
 <canvas id="cv" width="960" height="540"></canvas>
</div>
<script>
const im=document.getElementById('im'), cv=document.getElementById('cv'),
      ctx=cv.getContext('2d'), px=document.getElementById('px'), st=document.getElementById('st');
function grid(){{
  ctx.clearRect(0,0,960,540);
  ctx.strokeStyle='rgba(255,60,60,0.35)'; ctx.lineWidth=1; ctx.font='11px monospace';
  ctx.fillStyle='rgba(255,60,60,0.85)';
  for(let x=0;x<960;x+=100){{ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,540);ctx.stroke();
     ctx.fillText(x, x+2, 12);}}
  for(let y=0;y<540;y+=100){{ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(960,y);ctx.stroke();
     ctx.fillText(y, 2, y-3);}}
  ctx.strokeStyle='rgba(255,200,0,0.9)'; ctx.lineWidth=2;
  ctx.beginPath();ctx.moveTo(480,0);ctx.lineTo(480,540);ctx.moveTo(0,270);ctx.lineTo(960,270);ctx.stroke();
  ctx.fillStyle='rgba(255,200,0,0.9)';ctx.fillText('480,270',486,266);
}}
grid();
im.addEventListener('click', async e=>{{
  const r=im.getBoundingClientRect();
  const x=Math.round((e.clientX-r.left)*960/r.width), y=Math.round((e.clientY-r.top)*540/r.height);
  px.textContent=x+', '+y;
  ctx.strokeStyle='#0f0'; ctx.lineWidth=3;
  ctx.beginPath();ctx.moveTo(x-18,y);ctx.lineTo(x+18,y);ctx.moveTo(x,y-18);ctx.lineTo(x,y+18);ctx.stroke();
  ctx.font='bold 16px monospace'; ctx.fillStyle='#0f0'; ctx.fillText(x+', '+y, x+22, y-6);
  const resp=await fetch('/click',{{method:'POST',headers:{{'Content-Type':'application/json'}},
                                   body:JSON.stringify({{x,y}})}});
  st.textContent = resp.ok ? 'sent: '+x+', '+y : 'failed';
}});
</script></body></html>"""


def reset_body(wait_s: float = 3.0) -> None:
    """Stop and start the body, which is what actually resets the root.

    `rest` only resets the arm; the root needs a stop/start, and the dashboard endpoints are the
    supported way to ask. Done inline so the tool does not depend on an ad-hoc script.
    """
    import urllib.request

    for path in ("/api/body/stop", "/api/body/start"):
        req = urllib.request.Request(
            f"http://127.0.0.1:8765{path}",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as r:  # noqa: S310 - loopback only
                r.read()
        except OSError as exc:
            print(f"reset: {path} failed: {exc}", flush=True)
    time.sleep(wait_s)


def grab_frame(reset: bool) -> bytes:
    if reset:
        reset_body()
    import find_and_reach as fr

    jpeg, size = fr.capture(width=960)
    FRAME.write_bytes(jpeg)
    print(f"frame {size[0]}x{size[1]} -> {FRAME}", flush=True)
    return jpeg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--no-reset", action="store_true")
    args = ap.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)

    jpeg = grab_frame(not args.no_reset)
    b64 = base64.b64encode(jpeg).decode()
    page = PAGE.format(b64=b64).encode()
    clicked: list[tuple[int, int]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a) -> None:  # quiet
            pass

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length") or 0)
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
                x, y = int(data["x"]), int(data["y"])
            except Exception:
                self.send_response(400)
                self.end_headers()
                return
            clicked.append((x, y))
            OUT.write_text(json.dumps({"x": x, "y": y, "frame": str(FRAME)}))
            print(f"CLICK {x} {y}", flush=True)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"open http://127.0.0.1:{args.port}/ and click the object", flush=True)
    t0 = time.monotonic()
    while not clicked and time.monotonic() - t0 < args.timeout:
        srv.handle_request()
    if clicked:
        x, y = clicked[-1]
        print(f"PIXEL {x} {y}  (also in {OUT})", flush=True)
        time.sleep(1.0)
    else:
        print("no click", flush=True)
    return 0 if clicked else 1


if __name__ == "__main__":
    raise SystemExit(main())
