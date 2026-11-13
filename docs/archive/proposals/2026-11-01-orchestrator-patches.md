# Three patches to `clank/orchestrator.py` (proposed, not applied)

> **Archived 2026-11-13.** Proposed diffs, never applied. Kept as the record of what was considered.
> Current behaviour: [`README.md`](../../../README.md) · agent rules: [`AGENTS.md`](../../../AGENTS.md).

`clank/orchestrator.py` is another agent's file, so these are written as proposed
diffs. Each one is small, each unblocks something measured, and each has a check that
tells you whether it worked.

---

## 1. Stereo: stop forcing the null compositor

**Unblocks:** real depth, which is the cheapest route to the reachability question
(`research/live-pixel-servo.md` section 8), plus an absolute `f_px` for the field of
view instead of a three-angle fit.

**Evidence it is forced.** `/proc/<monado>/environ` contains `XRT_COMPOSITOR_NULL=1`
and `monado.log` says `Null compositor starting`. The captured buffer is one eye:
left-vs-right half NCC is -0.085 against a 1.000 self-control. Yet a presenting
compositor is known to work on this GPU - `spike/e3/evidence/hello_look_left.png` is a
genuine side-by-side pair, left/right NCC 0.970.

**The diff** (`_start_monado`, currently lines 785-793):

```diff
         env.update(
             {
-                "XRT_COMPOSITOR_NULL": "1",
+                # A null compositor renders nothing, so the game is given a
+                # single eye and there is no depth to be had. The GPU presents
+                # fine (spike/e3/evidence/hello_look_left.png is a real pair).
+                "XRT_COMPOSITOR_NULL": "0",
+                "XRT_COMPOSITOR_COMPUTE": "false",
+                "VK_DRIVER_FILES": "/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json",
+                "XRT_COMPOSITOR_SCALE_PERCENTAGE": "50",
                 "P_OVERRIDE_ACTIVE_CONFIG": "remote",
```

**Why each variable.** `NULL=0` stops the escape hatch. `COMPUTE=false` keeps the
compositor on the real display path. `VK_DRIVER_FILES` pins the NVIDIA ICD, because a
NixOS session does not always have it on the default search path and the failure is a
silent fallback to lavapipe. `SCALE_PERCENTAGE=50` is headroom on an 8 GB laptop:
3840x1080 at full scale plus the game is more than this card wants, and the depth
answer does not care about resolution.

**The check, pre-registered.** With the hand at Z ~ 0.42 m and the eye separation from
the driver (`0.065 m`, 1920-wide eyes), disparity should be about **52 px** per eye. If
the captured buffer is still one eye wide (aspect 1.78 rather than 3.56), the patch
did not take.

**Risk.** This changes what the *engine* sees, so if it goes wrong the game may fail to
present at all. Keep `XRT_COMPOSITOR_NULL: "1"` behind a config flag rather than
deleting it.

---

## 2. Stop the relay from freezing the game on a deadline

**Unblocks:** nothing new, but it removes a silent, severe failure.

**Evidence.** The relay ran with `--duration 3600`, exited cleanly at the limit, and the
game held a frozen pose for **17 minutes** while the dashboard reported
`state: running, last_error: None, pose: False, game: True`. Restarting only the relay
restored live poses immediately. Measured 2026-11-01, and again this session: the game
was frozen and the first sign was two captures differing by 1.77% of pixels.

**The diff** (the `body` branch of the join argv, currently lines 909-916):

```diff
                 str(state),
-                "--duration",
-                "3600",
+                # 0 = until killed. A relay that expires holds the game on its
+                # last pose forever and the dashboard cannot tell.
+                "--duration",
+                "0",
```

`clank/pose_relay.py` now treats `0` as unlimited, warns one minute before a bounded
run ends, and prints `EXPIRY_NOTE` when it stops (commit `0b229f9`), so both halves of
this behave.

**Better, if you want it:** surface child death in `/api/state` - the worst combination
is `pose: False` alongside `state: running, last_error: None`. Restarting the relay on
exit would be better still; that is a supervisor decision, not a relay one.

**The check.** Kill the relay by hand and confirm `/api/state` stops claiming `running`
with no error, or that nothing claims it in the first place because `--duration 0`
never expires.

---

## 3. An explicit VNC password for unattended capture

**Unblocks:** screenshots without a human keeping a VNC client attached.

**Evidence.** `weston-screenshooter` only completes while an authenticated VNC client
is attached: **6 of 6** captures timed out with nobody connected. Every frame in this
project was taken with a client attached, which is fine at a terminal and fatal for an
unattended demo.

**The diff** (the weston argv, currently around lines 816-832): add

```diff
             "--backend=vnc",
+            *([f"--password={self.cfg.preview.vnc_password}"]
+              if getattr(self.cfg.preview, "vnc_password", None) else []),
```

with `vnc_password: str | None = None` added to the preview config, and the password
supplied in `config.toml` (which is git-ignored) or an environment variable.

**Never hardcode a password, and never handle a local-account credential** - Weston's
VNC authenticates a local account, and that belongs to the operator. A demo-only
password in the ignored config is the intended shape.

**The check.** Detach every VNC client, then run one capture. It should return in
under a second, as it already does when a client is attached (measured 378 ms,
1,425,351 bytes, 1920x1080).

---

## Summary

| patch | unblocks | risk |
| --- | --- | --- |
| `XRT_COMPOSITOR_NULL` 1 -> 0 | depth, absolute `f_px`, reachability | game may fail to present |
| `--duration 0` | removes a silent freeze | none measured |
| VNC password | unattended capture | credential handling |
