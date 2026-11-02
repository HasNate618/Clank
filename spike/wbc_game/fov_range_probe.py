"""Was the boundary minimum my search range, or the camera?

Searches f from 40 (hfov 161 deg) upward, reports the animation floor measured
with no commands, and prints the whole curve so a flat fit is visible as flat.
"""
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
                      work_dir="/tmp/clank-shots", destination=f"/tmp/r_{tag}.png", timeout=20.0)
    return ImageOps.flip(Image.open(p).convert("L")).resize((W,H), Image.LANCZOS)
def grab(tag, yaw):
    if yaw is not None: send(verb="look", yaw_deg=yaw)
    time.sleep(3.0)
    st = json.load(open("/home/nate/.local/state/Clank/sessions/body/body_state.json"))
    return st, shot(tag)
def warp(src, R, f):
    K = np.array([[f,0,W/2],[0,f,H/2],[0,0,1.]])
    Hm = K @ R @ np.linalg.inv(K); Hm /= Hm[2,2]
    Hi = np.linalg.inv(Hm); Hi /= Hi[2,2]
    return src.transform((W,H), Image.PERSPECTIVE, tuple(Hi.flatten()[:8]), Image.BILINEAR)
def err(a, b, R, f, min_frac=0.02):
    w = np.asarray(warp(a, R, f), dtype=np.float64); bb = np.asarray(b, dtype=np.float64)
    m = w > 0
    return float(np.abs(w[m]-bb[m]).mean()) if m.sum() > min_frac*W*H else None

sA, iA = grab("A", -35.0)
sC, iC = grab("C", None)            # no command: animation floor
sB, iB = grab("B", 35.0)
floor = float(np.abs(np.asarray(iA,dtype=np.float64)-np.asarray(iC,dtype=np.float64)).mean())
C = np.array([[0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.]])
Rw = lambda s: rmat(s["root_q"]) @ rmat(s["head"]["q_xyzw"])
R = C @ (Rw(sB).T @ Rw(sA)) @ C.T
ang = math.degrees(math.acos(max(-1,min(1,(np.trace(Rw(sB).T@Rw(sA))-1)/2))))
print(f"animation floor (no command, 3 s apart): {floor:.2f}")
print(f"world head rotation A->B: {ang:.2f} deg")
print(f"raw A vs B (no warp): {float(np.abs(np.asarray(iA,dtype=np.float64)-np.asarray(iB,dtype=np.float64)).mean()):.2f}")
best=None; rows=[]
for f in range(40, 421, 5):
    e = err(iA, iB, R, f)
    if e is None: continue
    hf = 2*math.degrees(math.atan((W/2)/f))
    rows.append((f,hf,e))
    if best is None or e < best[2]: best=(f,hf,e)
print("\n  f_px   hfov    err")
for f,hf,e in rows:
    if f % 40 == 0 or (best and abs(f-best[0])<=5): print(f"  {f:5d} {hf:7.2f}  {e:6.2f}")
print(f"\nBEST f={best[0]} hfov={best[1]:.2f} err={best[2]:.2f}  (animation floor {floor:.2f})")
