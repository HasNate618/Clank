import json, math, socket, sys, time
import numpy as np
from PIL import Image, ImageOps
sys.path.insert(0, "/home/nate/Projects/Clank")
from clank.capture import capture_frame
SOCK = "/home/nate/.local/state/Clank/sessions/body/control.sock"
SS = "/nix/store/gdwnms70r3rnpl3q3qxqicl3413yhi1m-weston-16.0.0/bin/weston-screenshooter"
W, H = 480, 270

def send(**kw):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(2.0); s.connect(SOCK); s.sendall((json.dumps(kw)+"\n").encode())
        try: return s.recv(256).decode().strip()
        except TimeoutError: return "-"
def rmat(q):
    x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def shot(tag):
    p = capture_frame(screenshooter=SS, runtime_dir="/run/user/1000", wayland_display="clank-27abb1",
                      work_dir="/tmp/clank-shots", destination=f"/tmp/h_{tag}.png", timeout=20.0)
    im = ImageOps.flip(Image.open(p).convert("L")).resize((W,H), Image.LANCZOS)
    return im
def warp(src, R, f):
    K = np.array([[f,0,W/2],[0,f,H/2],[0,0,1.]])
    Hm = K @ R @ np.linalg.inv(K); Hm /= Hm[2,2]
    Hi = np.linalg.inv(Hm); Hi /= Hi[2,2]
    return src.transform((W,H), Image.PERSPECTIVE, tuple(Hi.flatten()[:8]), Image.BILINEAR)

def search(a, b, R, min_frac=0.03):
    best, curve = None, []
    for f in range(150, 2401, 25):
        w = np.asarray(warp(a, R, f), dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        m = w > 0
        if m.sum() < min_frac*W*H: continue
        e = float(np.abs(w[m]-bb[m]).mean())
        curve.append((f, e))
        if best is None or e < best[1]: best = (f, e)
    return best, curve

import importlib
def rot_y(d):
    t = math.radians(d)
    return np.array([[math.cos(t),0,math.sin(t)],[0,1,0],[-math.sin(t),0,math.cos(t)]])

# ---- self-test with the corrected threshold ----
A = Image.open("/tmp/n_0.png").convert("L").resize((W,H), Image.LANCZOS)
print("self-test (synthetic, known focal length):")
for f_true in (500, 800, 1400):
    B = warp(A, rot_y(29.), f_true)
    (f_hat, e), _ = search(A, B, rot_y(29.))
    print(f"  f_true={f_true:5d} -> f_hat={f_hat:5d}  err {e:6.3f}  "
          f"{'OK' if abs(f_hat-f_true)<=50 else 'MISS'}")

# ---- real pair ----
send(verb="look", yaw_deg=0.0); time.sleep(3.0)
s1 = json.load(open("/home/nate/.local/state/Clank/sessions/body/body_state.json")); i1 = shot("1")
send(verb="look", yaw_deg=25.0); time.sleep(3.0)
s2 = json.load(open("/home/nate/.local/state/Clank/sessions/body/body_state.json")); i2 = shot("2")
C = np.array([[0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.]])
Rw = lambda s: rmat(s["root_q"]) @ rmat(s["head"]["q_xyzw"])
Rrel = C @ (Rw(s2).T @ Rw(s1)) @ C.T
ang = math.degrees(math.acos(max(-1,min(1,(np.trace(Rw(s2).T@Rw(s1))-1)/2))))
print(f"\nreal pair: world head rotation {ang:.2f} deg")
(f, e), curve = search(i1, i2, Rrel)
hf = lambda f: 2*math.degrees(math.atan((W/2)/f))
print(f"  best f={f} -> hfov {hf(f):.1f} deg, err {e:.3f}")
print("  errors across f:", " ".join(f"{ff}:{ee:.0f}" for ff, ee in curve[::10]))
