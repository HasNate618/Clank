"""FOV from a central strip. Features near u=0.5 have alpha~0, so
    u(alpha=0 -> theta) = 0.5 - tan(theta)/(2 tan(h/2))
is exact for them and needs no template match and no assumed depth."""
import json, socket, sys, time
import numpy as np
from PIL import Image, ImageOps
sys.path.insert(0, "/home/nate/Projects/Clank")
from clank.capture import capture_frame

SOCK = "/home/nate/.local/state/Clank/sessions/body/control.sock"
SS = "/nix/store/gdwnms70r3rnpl3q3qxqicl3413yhi1m-weston-16.0.0/bin/weston-screenshooter"

def look(y):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(2.0); s.connect(SOCK)
        s.sendall((json.dumps({"verb":"look","yaw_deg":y})+"\n").encode())
        try: return s.recv(256).decode().strip()
        except TimeoutError: return "-"

def shot(tag):
    p = capture_frame(screenshooter=SS, runtime_dir="/run/user/1000",
                      wayland_display="clank-27abb1", work_dir="/tmp/clank-shots",
                      destination=f"/tmp/c_{tag}.png", timeout=20.0)
    return np.asarray(ImageOps.flip(Image.open(p).convert("L")), dtype=np.float64)

def shift(a, b, x0=760, x1=1160, y0=140, y1=760):
    pa = a[y0:y1, x0:x1].sum(axis=0); pa -= pa.mean()
    pb = b[y0:y1, x0:x1].sum(axis=0); pb -= pb.mean()
    n = len(pa); size = 1 << int(np.ceil(np.log2(2 * n)))
    cc = np.fft.irfft(np.fft.rfft(pb, size) * np.conj(np.fft.rfft(pa, size)), size)
    k = int(np.argmax(cc))
    if k > size // 2: k -= size
    l, c, r = cc[(k-1) % size], cc[k % size], cc[(k+1) % size]
    d = l - 2*c + r
    frac = 0.5*(l-r)/d if d else 0.0
    return k + (frac if abs(frac) < 1 else 0.0)

look(0); time.sleep(2.5)
ref = shot("0")
W = ref.shape[1]
print(f"frame {ref.shape[1]}x{ref.shape[0]}; central strip x 760..1160")
est = []
for yaw in (-15.0, -10.0, 10.0, 15.0):
    look(yaw); time.sleep(2.5)
    d = shift(ref, shot(f"{int(yaw):+d}"))
    theta = np.radians(abs(yaw))
    if d == 0:
        print(f"  {yaw:+5.0f}  no shift"); continue
    h = np.degrees(2 * np.arctan(W * np.tan(theta) / (2 * abs(d))))
    est.append(h)
    print(f"  yaw {yaw:+5.0f}  shift {d:+8.1f} px  -> hfov {h:6.2f} deg")
if est:
    print(f"\nmean hfov {np.mean(est):.2f} deg   spread {max(est)-min(est):.2f} deg")
