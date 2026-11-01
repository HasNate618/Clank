"""Measure the compositor's horizontal FOV from images.

We emit the HMD pose, so a head yaw of theta is a camera rotation of exactly theta.
A pure rotation shifts every pixel by an amount that depends only on its angle, not
its depth - so a background band gives a clean horizontal shift. Then

    u   = 0.5 + tan(alpha) / (2 tan(hfov/2))
    dpx = W * dtheta / (2 tan(hfov/2))     (small angles, near the axis)

Sweep two yaws and fit, so one bad correlation cannot set the answer.
"""
import json, socket, sys, time, subprocess
import numpy as np
from PIL import Image, ImageOps
sys.path.insert(0, "/home/nate/Projects/Clank")
from clank.capture import capture_frame

SOCK = "/home/nate/.local/state/Clank/sessions/body/control.sock"
SS = "/nix/store/gdwnms70r3rnpl3q3qxqicl3413yhi1m-weston-16.0.0/bin/weston-screenshooter"

def look(yaw_deg):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(2.0); s.connect(SOCK)
        s.sendall((json.dumps({"verb": "look", "yaw_deg": yaw_deg}) + "\n").encode())
        try: return s.recv(512).decode().strip()
        except TimeoutError: return "(no reply)"

def shot(tag):
    p = capture_frame(screenshooter=SS, runtime_dir="/run/user/1000",
                      wayland_display="clank-27abb1", work_dir="/tmp/clank-shots",
                      destination=f"/tmp/fov_{tag}.png", timeout=20.0)
    im = ImageOps.flip(Image.open(p).convert("L"))
    return np.asarray(im, dtype=np.float64)

def shift_px(a, b, band=(0.05, 0.42)):
    """Horizontal shift of b relative to a, from the column profile of a band."""
    h = a.shape[0]
    lo, hi = int(band[0] * h), int(band[1] * h)
    pa = a[lo:hi].sum(axis=0); pb = b[lo:hi].sum(axis=0)
    pa = pa - pa.mean(); pb = pb - pb.mean()
    n = len(pa)
    size = 1 << int(np.ceil(np.log2(2 * n)))
    fa = np.fft.rfft(pa, size); fb = np.fft.rfft(pb, size)
    cc = np.fft.irfft(fb * np.conj(fa), size)
    idx = int(np.argmax(cc))
    if idx > size // 2:
        idx -= size
    # Sub-pixel refinement on the peak's neighbours.
    left, right = cc[(idx - 1) % size], cc[(idx + 1) % size]
    denom = (left - 2 * cc[idx % size] + right)
    frac = 0.5 * (left - right) / denom if denom else 0.0
    return idx + (frac if abs(frac) < 1 else 0.0)

print("look(0) ->", look(0)); time.sleep(2.5)
base = shot("0")
W = base.shape[1]
print(f"frame {base.shape[1]}x{base.shape[0]}")

for yaw in (-30.0, 30.0):
    print(f"look({yaw:+.0f}) ->", look(yaw)); time.sleep(2.5)
    moved = shot(f"{int(yaw):+d}")
    dpx = shift_px(base, moved)
    theta = abs(np.radians(yaw))
    if dpx == 0:
        print(f"  yaw {yaw:+.0f}: no shift found"); continue
    half = np.arctan(W * theta / (2 * abs(dpx)))
    print(f"  yaw {yaw:+.0f}deg  shift {dpx:+.1f} px  -> hfov {np.degrees(2*half):.2f} deg")
