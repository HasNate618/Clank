#!/usr/bin/env python3
"""brainview - a throwaway dashboard for watching the brain think on a recording.

This is a development instrument, not product UI. It exists to answer one
question by eye: *does the brain's story match the picture it was looking at?*

Why it is separate from `clank/dashboard.py`: that dashboard reports the body. It
has no room for a per-turn reasoning trace, and it should not grow one for a
fixture that will be deleted once the live session is wired.

The pause the operator asked for is structural, not a playback control. Each turn
binds one frozen still to one trace record, and the clip never advances on its
own. That reproduces the live invariant by other means: in the session the robot
holds its head still while the model thinks, so the frame it reasoned about is
still the frame it grounds on. Here the still simply does not move until the next
step is requested.

    python3 spike/brainview/app.py \
        --video "Job Simulator VR FULL WALKTHROUGH ... .mp4" \
        --heard-file 0549396b6562.txt \
        --port 8770
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import re
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SRT_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")


def slice_wav(wav_bytes: bytes, start_s: float, end_s: float) -> bytes:
    """Cut one utterance out of the clip, as a standalone 16 kHz mono WAV.

    Without this the model hears the entire intro on every turn, so it answers
    "reach the card" at t=3 s when the card has not been mentioned yet. That is
    not a model error to tune away with prompting - it is the harness feeding the
    wrong audio. One cue in, one cue out.
    """
    try:
        with wave.open(io.BytesIO(wav_bytes)) as source:
            rate = source.getframerate()
            channels = source.getnchannels()
            width = source.getsampwidth()
            total = source.getnframes()
            first = max(0, int(start_s * rate))
            last = min(total, int(end_s * rate))
            if last <= first:
                return wav_bytes
            source.setpos(first)
            frames = source.readframes(last - first)
    except (wave.Error, EOFError):
        return wav_bytes
    out = io.BytesIO()
    with wave.open(out, "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(width)
        target.setframerate(rate)
        target.writeframes(frames)
    return out.getvalue()


def parse_srt(text: str) -> list[tuple[float, float, str]]:
    """Parse an SRT into (start_s, end_s, line) cues.

    The clip ships real subtitle timings, so the utterance timeline is read
    rather than inferred. An earlier attempt to derive speech boundaries from
    `ffmpeg silencedetect` produced 21 fragments for 4 subtitle blocks and no
    reliable mapping onto the text; the subtitle file is the ground truth.
    """
    cues: list[tuple[float, float, str]] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line for line in block.splitlines() if line.strip()]
        arrow = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if arrow is None:
            continue
        stamps = _SRT_TIME.findall(lines[arrow])
        if len(stamps) < 2:
            continue

        def seconds(parts: tuple[str, str, str, str]) -> float:
            h, m, s, ms = parts
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

        body = " ".join(lines[arrow + 1 :]).strip()
        if body:
            cues.append((seconds(stamps[0]), seconds(stamps[1]), body))
    return cues

from clank.brain import (  # noqa: E402
    DEFAULT_SEED,
    BrainConfig,
    TurnRecord,
    run_turn,
)
from clank.frames import ReplaySource  # noqa: E402

#: The tray bounding box measured by hand on 2026-11-01, normalised
#: `(u0, v0, u1, v1)`. A development fixture default, not a brain constant:
#: automatic container detection is unbuilt (spec §10), so the box is supplied by
#: the caller. A detector replaces this line, never `clank/brain.py`.
TRAY_BOX = (0.270, 0.325, 0.548, 0.566)


class Session:
    """Shared state: the frozen clip, the trace, and the cursor."""

    #: How many earlier utterances the brain may still see. Matches the spec's
    #: "seed + last K transcript lines"; older lines drop so a stale instruction
    #: cannot haunt a later turn.
    HISTORY_K = 4

    def __init__(
        self,
        source: ReplaySource,
        config: BrainConfig,
        seed: str,
        heard: str,
        wav: bytes | None,
        cues: list[tuple[float, float, str]] | None = None,
    ) -> None:
        self.source = source
        self.config = config
        self.seed = seed
        self.heard = heard
        self.wav = wav
        self.cues = cues or []
        self.index = 0
        self.records: list[TurnRecord] = []
        self.look_step = 1
        self._lock = threading.Lock()

    def heard_for(self, t_s: float) -> tuple[str, list[str]]:
        """Which utterance is audible at `t_s`, and what was said before it.

        Timings come from the clip's subtitle file, not from an estimate. Without
        a cue list the whole transcript is audible from the start, and that is
        what produced the wrong plan earlier: one `heard` blob containing both
        "take my card" and "grab a job cartridge" makes the model choose, and it
        chose the cartridge. Real hearing delivers one line at a time and the
        most recent instruction wins, which is what this reproduces.
        """
        if not self.cues:
            return self.heard, []
        active = ""
        history: list[str] = []
        for start, _end, text in self.cues:
            if start <= t_s:
                if active:
                    history.append(active)
                active = text
            else:
                break
        return (active or self.heard), history[-self.HISTORY_K :]

    @property
    def frames(self) -> int:
        return len(self.source)

    def wav_for(self, t_s: float) -> bytes | None:
        """The audio window the brain should hear at `t_s`, or the whole clip."""
        if self.wav is None:
            return None
        for start, end, _text in self.cues:
            if start <= t_s < end:
                return slice_wav(self.wav, start, end)
        if self.cues and t_s >= self.cues[-1][1]:
            start, end, _ = self.cues[-1]
            return slice_wav(self.wav, start, end)
        return self.wav

    def step(self) -> TurnRecord:
        with self._lock:
            if self.index >= self.frames:
                raise IndexError("no frames left")
            frame = self.source.frame(self.index)
            heard, history = self.heard_for(frame.t_s)
            # Stub: no hand occupancy is passed, so both hands are always
            # "empty" here. Every `release` is therefore refused and this
            # instrument cannot show an insert; a refused release chip is this
            # stub, not a brain error.
            record = run_turn(
                self.config,
                self.seed,
                frame,
                heard=heard,
                history=history,
                seq=len(self.records),
                box=TRAY_BOX,
                wav=self.wav_for(frame.t_s),
                turn=len(self.records) + 1,
                look_step=self.look_step,
            )
            self.records.append(record)
            self.look_step += 1
            self.index = min(self.index + 1, self.frames)
            return record

    def goto(self, index: int) -> None:
        with self._lock:
            self.index = max(0, min(index, self.frames - 1))

    def reset(self) -> None:
        with self._lock:
            self.records.clear()
            self.index = 0
            self.look_step = 1

    def state(self) -> dict:
        return {
            "frames": self.frames,
            "index": self.index,
            "seed": self.seed,
            "heard": self.heard,
            "cues": [
                {"start": round(s, 2), "end": round(e, 2), "text": t}
                for s, e, t in self.cues
            ],
            "model": self.config.model,
            "audio": self.wav is not None,
            "records": [r.to_dict() for r in self.records],
        }


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>brainview</title>
<style>
:root{--bg:#0d1117;--fg:#c9d1d9;--dim:#6e7681;--ok:#3fb950;--bad:#f85149;
      --accent:#58a6ff;--panel:#161b22;--line:#30363d}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
header{display:flex;gap:16px;align-items:baseline;padding:10px 16px;
        border-bottom:1px solid var(--line);flex-wrap:wrap}
h1{font-size:14px;margin:0;letter-spacing:.08em;text-transform:uppercase}
.tag{color:var(--dim)} .val{color:var(--accent)}
main{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:16px;padding:16px}
.stage{position:relative;background:#000;border:1px solid var(--line);border-radius:6px;
       overflow:hidden;line-height:0}
.stage img{width:100%;display:block}
.ov{position:absolute;inset:0;pointer-events:none}
.frozen{position:absolute;left:8px;top:8px;background:#000a;padding:2px 8px;
        border-radius:4px;font-size:11px;color:var(--ok);letter-spacing:.05em}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:6px;
       padding:10px 12px;margin-bottom:12px}
.panel h2{font-size:11px;margin:0 0 6px;color:var(--dim);letter-spacing:.1em;
          text-transform:uppercase;font-weight:600}
.row{display:flex;gap:8px;margin:2px 0}
.row b{color:var(--dim);min-width:74px;font-weight:400}
pre{margin:0;white-space:pre-wrap;word-break:break-word;font:inherit}
.accepted{color:var(--ok)} .rejected{color:var(--bad)}
.timeline{display:flex;flex-wrap:wrap;gap:6px;padding:0 16px 16px}
.chip{border:1px solid var(--line);border-radius:4px;padding:3px 8px;
      cursor:pointer;font-size:11px;background:var(--panel)}
.chip:hover{border-color:var(--accent)}
.chip.rej{border-color:var(--bad)}
.chip.on{background:var(--accent);color:#000;font-weight:700}
button{background:var(--panel);color:var(--fg);border:1px solid var(--line);
       border-radius:5px;padding:6px 14px;font:inherit;cursor:pointer}
button:hover{border-color:var(--accent)}
button.primary{border-color:var(--accent);color:var(--accent)}
button:disabled{opacity:.4;cursor:default}
.controls{display:flex;gap:8px;padding:0 16px 16px;flex-wrap:wrap}
.empty{color:var(--dim);font-style:italic}
</style></head><body>
<header>
  <h1>brainview</h1>
  <span class="tag">model <span class="val" id="model">-</span></span>
  <span class="tag">frame <span class="val" id="idx">0</span>/<span class="val" id="frames">0</span></span>
  <span class="tag">turns <span class="val" id="turns">0</span></span>
  <span class="tag">audio <span class="val" id="audio">-</span></span>
  <span class="tag">seed <span class="val" id="seed">-</span></span>
</header>
<main>
  <div>
    <div class="stage">
      <img id="shot" alt="frozen frame">
      <span class="frozen" id="frozen">FROZEN</span>
      <svg class="ov" id="ov" viewBox="0 0 100 100" preserveAspectRatio="none"></svg>
    </div>
    <div class="panel" style="margin-top:12px">
      <h2>heard</h2>
      <pre id="heard" class="empty">-</pre>
    </div>
  </div>
  <div>
    <div class="panel"><h2>plan</h2><pre id="plan" class="empty">no turn yet</pre></div>
    <div class="panel"><h2>ground</h2><pre id="ground" class="empty">-</pre></div>
    <div class="panel"><h2>decision</h2><pre id="decision" class="empty">-</pre></div>
    <div class="panel"><h2>self</h2><pre id="self" class="empty">-</pre></div>
  </div>
</main>
<div class="controls">
  <button class="primary" id="step">Step (run brain)</button>
  <button id="auto">Auto run</button>
  <button id="back">Back</button>
  <button id="reset">Reset</button>
</div>
<div class="timeline" id="timeline"></div>
<script>
const $ = (id) => document.getElementById(id);
let view = -1, auto = null, lastLen = 0;

async function post(path, body) {
  const r = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'},
                               body: JSON.stringify(body||{})});
  return r.json();
}

function drawOv(rec) {
  // The viewBox is 0 0 100 100 stretched over the image, which is exactly the
  // normalised [0, 1] space GROUND reports in: multiply by 100, no arithmetic.
  const ov = $('ov');
  ov.innerHTML = '';
  const g = rec && rec.ground;
  if (!g || !g.box) { return; }

  const boxAttrs = (extra) => Object.assign({
    fill: 'none', stroke: '#6e7681', 'stroke-width': '1',
    'vector-effect': 'non-scaling-stroke'}, extra || {});
  const el = (tag, attrs) => {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs)) { node.setAttribute(k, v); }
    ov.appendChild(node);
  };

  const [u0, v0, u1, v1] = g.box;
  const x = u0 * 100, y = v0 * 100, w = (u1 - u0) * 100, h = (v1 - v0) * 100;
  el('rect', boxAttrs({x: x, y: y, width: w, height: h}));

  // The 2x2 quarters of the box are the answer vocabulary: GROUND names one of
  // them, so the grid is drawn inside the container box, not across the image.
  el('line', boxAttrs({x1: x + w/2, y1: y, x2: x + w/2, y2: y + h,
                       'stroke-width': '0.5'}));
  el('line', boxAttrs({x1: x, y1: y + h/2, x2: x + w, y2: y + h/2,
                       'stroke-width': '0.5'}));

  if (g.quadrant) {
    const right = g.quadrant.indexOf('RIGHT') >= 0;
    const bottom = g.quadrant.indexOf('BOTTOM') >= 0;
    el('rect', boxAttrs({
      x: x + (right ? w/2 : 0), y: y + (bottom ? h/2 : 0),
      width: w/2, height: h/2,
      fill: 'rgba(88,166,255,0.14)', stroke: '#58a6ff'}));
  }

  if (g.pixel) {
    const cx = g.pixel[0] * 100, cy = g.pixel[1] * 100;
    for (const [x1, y1, x2, y2] of [[cx-2, cy, cx+2, cy], [cx, cy-2, cx, cy+2]]) {
      el('line', {x1: x1, y1: y1, x2: x2, y2: y2, stroke: '#3fb950',
                  'stroke-width': '1.5',
                  'vector-effect': 'non-scaling-stroke'});
    }
  }
}

function render(st) {
  $('model').textContent = st.model;
  $('frames').textContent = st.frames;
  $('idx').textContent = st.index;
  $('turns').textContent = st.records.length;
  $('audio').textContent = st.audio ? 'on (model hears clip)' : 'off (text only)';
  $('seed').textContent = st.seed;

  if (st.records.length !== lastLen) { lastLen = st.records.length; view = st.records.length - 1; }
  const rec = view >= 0 && view < st.records.length ? st.records[view] : null;

  const frameIdx = rec ? rec.frame_index : st.index;
  $('shot').src = '/api/frame?i=' + frameIdx + '&t=' + Date.now();
  $('frozen').textContent = rec ? ('FROZEN @ ' + rec.t_s + 's') : 'FROZEN';

  drawOv(rec);

  $('heard').textContent = rec ? rec.heard : (st.heard || '-');
  $('heard').className = rec ? '' : 'empty';

  if (!rec) {
    for (const id of ['plan','ground','decision','self']) {
      $(id).textContent = id === 'plan' ? 'no turn yet' : '-';
      $(id).className = 'empty';
    }
  } else {
    const p = rec.plan;
    $('plan').className = '';
    $('plan').innerHTML =
      '<span class="row"><b>parsed</b><span>' +
        esc(JSON.stringify({skill:p.skill, side:p.side, target:p.target})) + '</span></span>' +
      (p.error ? '<span class="row"><b>error</b><span class="rejected">'+esc(p.error)+'</span></span>' : '') +
      '<span class="row"><b>raw</b><span>' + esc(p.raw.trim() || '(empty)') + '</span></span>' +
      '<span class="row"><b>latency</b><span class="tag">prefill ' + p.prompt_ms +
        ' ms / decode ' + p.decode_ms + ' ms</span></span>';

    const g = rec.ground;
    $('ground').className = '';
    $('ground').innerHTML = !g ? '<span class="tag">not needed</span>' :
      '<span class="row"><b>noun</b><span>' + esc(g.noun) + '</span></span>' +
      '<span class="row"><b>quadrant</b><span class="val">' + (g.quadrant || '-') + '</span>' +
        '<span class="tag">box ' + (g.box ? g.box.map(v => v.toFixed(3)).join(', ') : '-') +
        '</span></span>' +
      '<span class="row"><b>pixel</b><span>' + (g.pixel ? g.pixel.map(v=>v.toFixed(3)).join(', ') : 'none') + '</span></span>' +
      (g.raw ? '<span class="row"><b>raw</b><span>' + esc(g.raw.trim() || '(empty)') + '</span></span>' : '') +
      (g.error ? '<span class="row"><b>error</b><span class="rejected">'+esc(g.error)+'</span></span>' : '');

    const d = rec.decision;
    $('decision').className = '';
    $('decision').innerHTML =
      '<span class="row"><b>intent</b><span>' +
        esc(JSON.stringify({skill:d.intent.skill, side:d.intent.side,
             pixel:d.intent.pixel, yaw_rad:d.intent.yaw_rad})) + '</span></span>' +
      '<span class="row"><b>verdict</b><span class="' +
        (d.accepted ? 'accepted">PLAN KEPT' : 'rejected">PLAN OVERRIDDEN') + '</span></span>' +
      (d.reason ? '<span class="row"><b>reason</b><span class="rejected">'+esc(d.reason)+'</span></span>' : '') +
      (d.note ? '<span class="row"><b>note</b><span class="tag">'+esc(d.note)+'</span></span>' : '') +
      '<span class="row"><b>turn ms</b><span class="tag">' + rec.elapsed_ms + '</span></span>';

    const s = rec.self;
    $('self').className = '';
    $('self').textContent = 'left=' + s.left + '  right=' + s.right +
      '  yaw=' + s.yaw_rad + '  glow=' + s.glow + '  ghost=' + s.ghost +
      '  box=' + (s.box ? s.box.map(v => v.toFixed(3)).join(',') : 'whole frame') +
      (s.crop_error ? '\nCROP FAILED: ' + s.crop_error : '');
  }

  const tl = $('timeline');
  tl.innerHTML = '';
  st.records.forEach((r, i) => {
    const c = document.createElement('div');
    c.className = 'chip' + (r.decision.accepted ? '' : ' rej') + (i === view ? ' on' : '');
    c.textContent = r.turn + ':' + (r.plan.skill || '?');
    c.title = JSON.stringify(r.plan) + '\\n' + (r.decision.reason || 'accepted');
    c.onclick = () => { view = i; refresh(); };
    tl.appendChild(c);
  });
}

function esc(s) {
  return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}

async function refresh() {
  const r = await fetch('/api/state');
  render(await r.json());
}

$('step').onclick = async () => { await post('/api/step'); view = -1; refresh(); };
$('back').onclick = async () => {
  const st = await (await fetch('/api/state')).json();
  view = Math.max(0, (view < 0 ? st.records.length - 1 : view) - 1);
  refresh();
};
$('reset').onclick = async () => { await post('/api/reset'); view = -1; lastLen = 0; refresh(); };
$('auto').onclick = () => {
  if (auto) { clearInterval(auto); auto = null; $('auto').textContent = 'Auto run'; return; }
  $('auto').textContent = 'Stop';
  auto = setInterval(async () => {
    const st = await (await fetch('/api/state')).json();
    if (st.index >= st.frames) { clearInterval(auto); auto = null; $('auto').textContent='Auto run'; return; }
    await post('/api/step'); view = -1; refresh();
  }, 1400);
};

refresh();
setInterval(() => { if (!auto) refresh(); }, 2000);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    session: Session

    def log_message(self, fmt: str, *args) -> None:  # keep the pane readable
        if "/api/state" not in (args[0] if args else ""):
            sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif parsed.path == "/api/state":
            self._json(self.session.state())
        elif parsed.path == "/api/frame":
            query = parse_qs(parsed.query)
            try:
                index = int(query.get("i", ["0"])[0])
                frame = self.session.source.frame(index)
            except (ValueError, IndexError) as exc:
                self._json({"error": str(exc)}, 400)
                return
            self._send(200, frame.jpeg, "image/jpeg")
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            payload = {}

        parsed = urlparse(self.path)
        if parsed.path == "/api/step":
            try:
                record = self.session.step()
            except IndexError as exc:
                self._json({"error": str(exc)}, 400)
                return
            self._json({"ok": True, "record": record.to_dict()})
        elif parsed.path == "/api/goto":
            self.session.goto(int(payload.get("i", 0)))
            self._json({"ok": True})
        elif parsed.path == "/api/reset":
            self.session.reset()
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="clip to decompose into stills")
    parser.add_argument("--cache", default="/tmp/brainview-frames")
    parser.add_argument("--heard-file", help="ASR transcript; its body becomes `heard`")
    parser.add_argument("--srt", help="subtitle file: real utterance timings per frame")
    parser.add_argument("--heard", help="override the heard text")
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--audio", help="wav clip the model hears (default: none)")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--base-url", default=BrainConfig.base_url)
    parser.add_argument("--model", default=BrainConfig.model)
    args = parser.parse_args()

    source = ReplaySource(args.video, args.cache, fps=args.fps)
    count = source.prepare()

    heard = args.heard or ""
    if not heard and args.heard_file:
        text = pathlib.Path(args.heard_file).read_text(encoding="utf-8")
        # The ASR file is a small header (Language/Duration/Model) then a ruler of
        # '=' then the body. Split on the ruler itself, not a fixed 20 chars, or
        # the remainder of the ruler survives into `heard`.
        body = re.split(r"={10,}", text, maxsplit=1)[-1].strip()
        heard = " ".join(body.split())

    wav = pathlib.Path(args.audio).read_bytes() if args.audio else None
    config = BrainConfig(base_url=args.base_url, model=args.model)
    cues = parse_srt(pathlib.Path(args.srt).read_text(encoding="utf-8")) if args.srt else []

    Handler.session = Session(source, config, args.seed, heard, wav, cues=cues)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"brainview: {count} frames at {args.fps} fps -> http://127.0.0.1:{args.port}")
    print(f"  model {config.model} at {config.base_url}")
    print(f"  audio {'on' if wav else 'off'}")
    if cues:
        for start, end, text in cues:
            print(f"  cue {start:6.2f}-{end:6.2f}s  {text[:60]}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
