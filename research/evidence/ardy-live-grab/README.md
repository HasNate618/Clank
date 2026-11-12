# The intro card, grabbed live on the generated path

2026-11-11/12, Job Simulator running, `scale 1.2`, floor pivot, head scaled, client camera.

This is the first live grab where the reach was **generated** by ARDY and played by SONIC mode 0,
rather than streamed by `reach_vr`. It follows two defects fixed the same evening (see the ledger):
the clasp sending a bare closure, and the freeze-hold re-latching onto the pre-reach pose.

## The run

```
LD_LIBRARY_PATH=/run/opengl-driver/lib spike/ardy/.venv/bin/python \
    spike/wbc_game/reach_ardy.py 840 445 --grab
```

| | |
| --- | --- |
| card pixel | (840, 445) in the 960x540 colour frame |
| depth there | 0.275 m nearest / 0.282 m median; the console at the same frame reads 0.467 m, so the pixel is unambiguous |
| object point | world (-0.451, -0.255, 0.993), 59.6 deg off the body's heading |
| classification | step; root goal 76 mm forward and -60 deg, so one generation was worth it |
| generation | 52 frames in 0.198 s, 0.94 GB VRAM |
| generator residual | **15.9 mm** on the glove (24.3 mm on the wrist link) |
| clasp | fired mid-clip at **78 mm** against the 80 mm threshold; the body reported `closure (0.0, 1.0)` |
| achieved residual | **24.7 mm** on the glove (180.9 mm on the wrist — the 0.18 m lever, visible in the difference) |
| hold afterwards | `grip_r` and `curl_r` pinned at 1.0 across 300 ticks, `flags 0`, no fall risk |

## What it proves, and what it does not

Proves: the generated reach, the mode-0 playback, the mid-clip clasp and the freeze-hold all work
end to end with the game up and the depth layer live, and the aim at 1.2x scale lands inside the
game's own grab radius.

Does not prove: the look. The torso started 63.9 deg off the object and the reach did not turn it
(`research/reach-architecture-critique.md` §8.5 records why that is a stance problem, not a
constraint one). Does not prove the out-of-reach case, which is still unexercised, nor the clasp
threshold, which remains the uncalibrated 80 mm guess.
