#!/usr/bin/env python3
"""Drive a running body_view over control.sock and read the real body_state.json.

Run: `python3 spike/wbc_game/drive_live_body.py`

This is the adapter's runtime, not a probe: body_view owns a CommandBus, so the
freeze-hold latch is exercised through bus.take() exactly as in production.
Results are in `research/live-body-runtime.md`.
"""
"""Drive the live body_view over control.sock and read the real body_state.json.

This is the runtime the adapter will use, not a probe harness: body_view owns a
CommandBus, so the latch fix is exercised through bus.take() exactly as it will be.
"""
import json, socket, time, pathlib
SOCK = "/home/nate/.local/state/Clank/sessions/body/control.sock"
STATE = pathlib.Path("/home/nate/.local/state/Clank/sessions/body/body_state.json")
FLAG_SATURATED = 1

def send(payload):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(2.0)
        s.connect(SOCK)
        s.sendall((json.dumps(payload) + "\n").encode())
        try:
            return s.recv(4096).decode().strip()
        except TimeoutError:
            return "(no reply)"

def read():
    return json.loads(STATE.read_text())

def line(tag, d):
    print(f"{tag:>10} seq_ack={d['seq_ack']:<4} flags={d['flags']} "
          f"res_r={d['residual_right_m']:.3f} wrist=({d['right_hand']['p'][0]:+.3f},"
          f"{d['right_hand']['p'][1]:+.3f},{d['right_hand']['p'][2]:+.3f})", flush=True)

print("baseline:", flush=True); line("idle", read())
print("\nreach right forward 0.55 ->", send({"verb": "reach", "side": "right",
                                            "forward_m": 0.55, "up_m": 0.0, "lateral_m": 0.0}), flush=True)
for i in range(12):
    time.sleep(0.5); line(f"+{(i+1)*0.5:.1f}s", read())

print("\nidle soak 20 s (latch):", flush=True)
start = None; pts = []
for i in range(20):
    time.sleep(1.0)
    d = read(); p = d["right_hand"]["p"]
    if start is None:
        start = p
    pts.append(p)
    if i % 4 == 0:
        line(f"+{i+1}s", d)
drift = max(sum((a-b)**2 for a, b in zip(p, start)) ** 0.5 for p in pts)
print(f"\nlatch drift over 20 s: {drift:.4f} m", flush=True)
