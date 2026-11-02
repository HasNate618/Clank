"""Is the fitted focal length consistent across rotation magnitudes?

A pure-rotation homography should recover the same f at 10, 20 and 35 deg. If the
three disagree, the model is wrong and no single number is a measurement.
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
                      work_dir="/tmp/clank-shots", destination=f"/tmp/mf_{tag}.png", timeout=20.0)
    return ImageOps.flip(Image.open(p).convert("L")).resize((W,H), Image.LANCZOS)
def grab(tag, yaw):
    send(verb="look", yaw_deg=yaw); time.sleep(3.0)
    st = json.load(open("/home/nate/.local/state/Clank/sessions/body/body_state.json"))
    return st, shot(tag)
def fit(a, b, R):
    best = None
    for f in range(40, 401, 2):
        K = np.array([[f,0,W/2],[0,f,H/2],[0,0,1.]])
        Hm = K @ R @ np.linalg.inv(K); Hm /= Hm[2,2]
        Hi = np.linalg.inv(Hm); Hi /= Hi[2,2]
        w = np.asarray(a.transform((W,H), Image.PERSPECTIVE, tuple(Hi.flatten()[:8]), Image.BILINEAR), np.float64)
        bb = np.asarray(b, np.float64); m = w > 0
        if m.sum() < 0.02*W*H: continue
        e = float(np.abs(w[m]-bb[m]).mean())
        if best is None or e < best[2]: best = (f, 2*math.degrees(math.atan((W/2)/f)), e)
    return best
C = np.array([[0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.]])
Rw = lambda s: rmat(s["root_q"]) @ rmat(s["head"]["q_xyzw"])
s0, i0 = grab("0", 0.0)
print(f"reference head p {[round(v,3) for v in s0['head']['p']]}")
for yaw in (10.0, 20.0, 35.0):
    s1, i1 = grab(f"{int(yaw)}", yaw)
    R = C @ (Rw(s1).T @ Rw(s0)) @ C.T
    ang = math.degrees(math.acos(max(-1,min(1,(np.trace(Rw(s1).T@Rw(s0))-1)/2))))
    dp = np.linalg.norm(np.array(s1["head"]["p"]) - np.array(s0["head"]["p"]))
    b = fit(i0, i1, R)
    print(f"yaw {yaw:4.0f} -> world rot {ang:5.2f} deg, head moved {dp*1000:5.1f} mm | "
          f"best f={b[0]:3d} (hfov {b[1]:6.2f}) err {b[2]:6.2f}")
