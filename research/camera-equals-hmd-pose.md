# Is the rendered camera actually at the HMD pose?

The pixel servo, the FOV, and every "look at the object then reach for its pixel"
plan rest on one assumption: **the game renders from the pose the relay sends as
the HMD.** It has never been tested. This note tests it and does not confirm it.

`[MEASURED 2026-11-01]` unless marked otherwise.

## 0. CORRECTION, same day — sections 4 and 5 are withdrawn

The searches behind "a pure rotation does not explain the two frames" ran in a
**480x270 downscaled space with a focal-length floor of 150 px**. At 480 px wide
that floor *is* hfov 116 deg, so a wide camera sat outside the search space: for a
140 deg horizontal field, `f_px = 240/tan(70) = 87 px`, unreachable at 150. Every
search therefore bottomed out at its own boundary. "No interior minimum" was
reporting a property of the grid, not of the camera.

Re-run with the floor at 40 px, same method, same session, a -35 to +35 deg head
sweep (69.55 deg of measured world rotation):

| f_px (480 space) | hfov | mean abs diff after warp |
| ---: | ---: | ---: |
| 40 | 161.1 | 59.20 |
| 120 | 126.9 | 49.44 |
| **140** | **119.5** | **35.67** |
| 200 | 100.4 | 38.85 |
| 320 | 73.7 | 59.44 |

There is now an interior minimum, at 480-space f=140 (hfov ~119.5 deg), against an
**animation floor of 20.91** measured with no command and a 3 s gap, and a raw
uncorrected A-vs-B difference of 70.14. Poses also demonstrably still drive the
view, which does not support a concurrent report that the game's XR link had died.

**Withdrawn: "a pure camera rotation does not explain the two frames" and "the
camera-equals-HMD-pose assumption is in doubt".** That was a wrong search range, not
a wrong camera. The fit is still not clean (35.7 against a floor of 20.9), so treat
~119.5 deg as indicative, not measured.

### 0.1 What the driver is configured for

CITED from the patched service binary's display block: `w_pixels` 3840,
`h_pixels` 1080, `fov = 140 deg`, lens separation 0.065 m, giving per-eye
**1920x1080** at **140.0 H x 114.22 V** and **f_px = 349.4** in 1920-px space.
(960/tan70 = 349.4 and 540/tan57.11 = 349.1 agree to 0.1 %, so square pixels.)
MEASURED alongside it: `monado.log` reports `view count: 2`, and the game's Unity log
reports `Created eye textures with a "double wide" layout`.

That leaves an **open discrepancy: the fit says ~119.5 deg, the declared geometry
says 140.0 deg.** The fit's residual is too high to settle which is right.

Also flagged, unverified: `clank/projection.py::project_point` maps angle to pixel
*linearly* (`u = (fov.angle_left - yaw)/fov.horizontal`) rather than rectilinearly.
At 33 deg off-axis those differ substantially, and the pixel servo is built on that
function — so if the field really is 140 deg, edge-of-view error is large.

### 0.2 Why stereo cannot rescue the measurement from this session

Both eyes are rendered and neither is presented: the service runs
`XRT_COMPOSITOR_NULL=1`, and that compositor allocates both eye swapchains and
discards them, so nothing downstream can see an eye at all. The window we screenshot
is the game's own and shows one eye. A presenting compositor was shown to work on
this GPU earlier (`spike/e3/evidence/hello_look_left.png` is a genuine side-by-side
pair, left/right NCC 0.970), so a restart path exists but was not taken.

The default IPD is CITED at 63 mm (`OXR_DEBUG_IPD_MM`), split ±31.5 mm. If a live
stereo pair is ever captured, the hand at Z ~ 0.42 m should show a disparity of
`349.4 * 0.063 / 0.42 = 52 px` per 1920-wide eye — a pre-registered number to test
against, not a result.

## 1. The hand is rendered, and can be brought into frame

Earlier work concluded "the wrist is out of view" from the hold clip. That is a
property of *that clip*, not of the game. On a live session the right hand becomes
visible when it is raised — `{"verb":"reach","side":"right","forward_m":0.42,
"up_m":0.40,"lateral_m":0.20}` puts the wrist 0.445 m forward and 9.3 deg off the
camera axis, and a pale cartoon hand appears in frame. So VR hands **are** rendered
and **are** driven by our poses. Useful: it gives a movable, self-luminous marker
for future calibration.

## 2. The wrist method fails, and it has a known confound

From the pose in that same capture the wrist sat at, in the head's own frame,
`forward 0.352, left 0.229, up -0.007` — a yaw of **+33.07 deg**, pitch -1.14.
`project_point` then predicts:

| assumed hfov | predicted u |
| ---: | ---: |
| 60 | -0.05 (off frame left) |
| 90 | 0.13 |
| 110 | 0.20 |

The hand is observed near **u ~ 0.35**. Solving hfov = yaw_w / (0.5 - u) gives
**~200 deg**, which is not a headset. So either the FOV is nonsense, or the rendered
hand is not at the wrist joint — and the latter is likely: Job Simulator's hand model
is a large cartoon hand centred on the palm, not the wrist, so it sits ~0.1 m out
along the arm, which at 0.4 m range is ~15 deg of error. **The wrist method cannot
separate those two explanations.** Do not use it without a model-origin correction.

## 3. The rotation-homography method, and why its first failure did not count

For a camera rotating about its centre, two frames are related by
`H = K C R(q2)^T R(q1) C^T K^-1`, with `K` built from one unknown: the focal length
in pixels. Warp one frame onto the other and pick the `f` that minimises the
difference. The head quaternions are exact, and the root must be composed first
(`root_q * head_q`), because that is what the relay sends.

The first searches all bottomed out at the widest allowed FOV. The cause was in the
estimator, not the camera: the validity guard required **>40 % of pixels** to survive
the warp, and a large rotation at a *narrow* FOV leaves little overlap — so exactly
the correct narrow candidates were discarded, biasing every search toward wide.

**Self-test after the fix** (synthesise a pair at a known focal length, then recover
it with the same code; this is the mutation check for an estimator):

| f_true | recovered | error |
| ---: | ---: | ---: |
| 500 | 500 | 0.000 |
| 800 | 800 | 0.000 |
| 1400 | 850 | 94.5 |

So the estimator is sound in the range that matters; 1400 is unrecoverable because at
29 deg the overlap falls below the floor, and that is reported rather than hidden.

## 4. The result: the real pair does not fit, at any focal length

`look 0` then `look 25` on a live session, head settled for 3 s each:

- world head rotation between frames: **28.51 deg** (root composed)
- head translation: 45.4 mm

| f_px | hfov | mean abs diff after warp |
| ---: | ---: | ---: |
| 150 | 116.0 | **43.6** (best, boundary) |
| 400 | 60.4 | 53 |
| 650 | 40.4 | 57 |
| 900 | 29.9 | 53 |

The curve is shallow and monotone toward wide — no interior minimum. For comparison,
the animation floor was measured with **no commands sent and the body static to
within 5 mm**: mean abs diff 4.4 then 11.7 then 11.0 over successive 5 s intervals,
concentrated in moving props and light beams (ceiling blocks ranged 0.3 to 6.0, so
the camera itself was static during that test).

A true rotation fit gives ~0 on synthetic data; the animation floor is ~11; the real
pair gives ~44 at best. **A pure camera rotation does not explain the two frames.**

## 5. What this does and does not establish

ESTABLISHED: the rendered view between two HMD yaws is not reproduced by a
pure-rotation homography built from the head poses, with a margin far outside the
animation noise. Hands are rendered and follow our poses.

NOT ESTABLISHED, and now in doubt: that the camera equals the HMD pose. This is the
assumption the pixel servo, `servo_step`, `unproject`, and `reach_outcome` all sit on.

Leading explanations, none tested:

1. The game smooths or clamps head yaw, so the rendered rotation is smaller than the
   commanded one. Predicts a consistent under-rotation; testable by searching a
   rotation *scale* alongside `f`.
2. The game translates the player during the sequence. Predicts a rotation fit that
   improves when observed from a nearer part of the scene; testable with a static
   wall filling the frame.
3. The pose the relay sends is ahead of the rendered frame by more than the settle
   time. Predicts a *latency*-dependent error; testable by comparing captures taken
   at the same pose but different settle intervals.
4. The camera is not the head centre but an eye offset with its own transform.

## 6. Next test, and it is cheap

Add a **rotation-scale parameter** to the search: fit `(f, s)` where the rendered
rotation is `s * Rrel`. If `s` lands near 0.5, explanation 1 is confirmed and the
pixel servo needs a scale factor that nothing in the codebase currently has. If the
fit still fails, point the camera at a **flat static wall** so the scene has no depth
variation and no animation, and repeat — that removes explanations 2 and the
animation floor in one move.

Until then: **do not build pixel-to-motion control on the assumption that the
rendered camera is the HMD pose.** The FOV is not measured either, and the two
independent attempts above disagree by more than 2x.

Probes kept: `spike/wbc_game/fov_rotation_homography.py` (self-test included),
`fov_yaw_probe.py`, `fov_shift_probe.py`, `fov_range_probe.py`.

## 7. RESOLVED: the field is ~140 deg, and the projection model cannot represent it

Multi-angle fit with the corrected range, same reference frame, head settled each
time, 480x270 space:

| yaw | measured world rotation | best f | hfov | error |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 9.46 | 100 | 134.8 | 21.0 |
| 20 | 21.32 | 78 | 144.0 | 42.0 |
| 35 | 36.25 | 80 | 143.1 | 43.1 |

The two larger rotations agree at 143-144 deg; the 10 deg case is the weakest
constraint (least signal) and sits at 134.8. All three bracket the driver's declared
140.0 deg horizontal, and the measured f (78-80 in 480 px) is within ~10 % of the
declared 87.35. The earlier 119.5 deg figure came from a 69.55 deg pair, where the
overlap is smallest and the fit is least constrained.

So: **hfov ~ 140 deg, vfov ~ 114.2 deg**, indicative and three-angle consistent,
not a precision number (residuals sit near the animation floor of ~21 because props
and light beams move between frames).

### 7.1 The consequence: `project_point` is a linear map, not a projection

`clank/projection.py::project_point` computes

```
u = (fov.angle_left - yaw) / fov.horizontal
```

which is **linear in angle**. An OpenXR frustum is rectilinear: the tangent-space map
is

```
u = (tan(yaw) + tan(angle_left)) / (tan(angle_left) + tan(angle_right))
```

At a 140 deg field the difference is not a rounding error:

| off-axis angle | linear u | rectilinear u | error (fraction of frame) |
| ---: | ---: | ---: | ---: |
| 22.5 | 0.750 | 0.707 | 0.043 |
| 35 | 0.750 | 0.627 | **0.123** |
| 50 | 0.857 | 0.720 | 0.137 |

At 35 deg that is **~236 px of 1920**. Every pixel consumer inherits it: `unproject`,
`servo_step`, and the `reach_outcome` glow test. The pixel servo therefore mis-aims by
a large fraction of the frame anywhere away from the centre - which is consistent with
its measured failure to converge in `research/pixel-servo-sim.md` (it "lands then
oscillates", and damping made it strictly worse).

**Fixed 2026-11-01.** `project_point` and `unproject` now use the tangent map and its
exact inverse (`_angle_to_pixel` / `_pixel_to_angle`), with five discriminating tests
in `tests/test_projection.py:RectilinearProjection`. The existing projection tests
could not tell the two models apart - every *edge* angle lands on the same pixel
either way, and the centre of a symmetric frustum is 0.5 either way - so the new
tests were mutation-checked three ways: restoring the linear map fails 4 of them,
flipping the sign fails 3, and dropping the second tangent from the denominator fails
2.

Whether the servo's failure to converge was arithmetic or mechanical has an answer,
and it is *not* arithmetic in the sim. Size of the artefact, in metres of hand
placement at 0.45 m depth:

| field | pixel 0.10 | pixel 0.25 | pixel 0.40 |
| ---: | ---: | ---: | ---: |
| 90 deg (the servo sim fixture) | 0.033 m | 0.039 m | 0.019 m |
| 140 deg (measured on the live game) | **0.322 m** | **0.303 m** | 0.135 m |

At 90 deg the error is a few centimetres, an order of magnitude below the 0.19-0.32 m
shortfall the sim measures, and re-running `servo_sim_probe.py` after the fix
reproduces that shortfall essentially unchanged. So the arm's failure to arrive is
mechanical (`research/pixel-servo-sim.md`).

At 140 deg it is a different story: 0.13-0.32 m, the same magnitude as the shortfall.
**On the live game the linear model alone would have put the hand about a third of a
metre from where it was asked to go**, so the fix matters roughly ten times more in
the game than in the sim fixture the servo was tuned against.

## 8. Live validation: draw the prediction on the frame and look

The sim fixture is 90 deg, so it cannot test the corrected projection or the field of
view. The game can. Method: raise the hand, capture, project the **known** wrist
position from `body_state` through `project_point` under two fields of view, and draw
both crosshairs on the real frame (`spike/wbc_game` machinery, images under
`research/evidence/`).

**Frame 1, hand near the axis** (`hand-projection-near-axis.jpg`). Head
(-0.017, 0.183, 0.355), wrist (0.467, 0.231, 0.234), only ~5.7 deg off-axis. The two
predictions differ by 40 px and **both land on the hand**, so this frame validates
the projection but cannot discriminate the FOV.

**Frame 2, hand far off-axis** (`hand-projection-140deg.jpg`). Head
(-0.079, 0.158, 0.358), wrist (0.237, 0.526, 0.304). The predictions now differ by
**156 px**:

| assumed field | predicted pixel |
| ---: | ---: |
| 140 deg | (871, 503) |
| 90 deg | (715, 483) |

The 140 deg crosshair lands **on the wrist**, where the rendered hand meets the arm.
The 90 deg crosshair lands about 150 px to the left of the hand, in empty space.

So the live render agrees with a ~140 deg field and excludes 90 deg, independently of
the homography fit and of the CITED driver geometry. MEASURED.

Caveat, and it is why this validates rather than measures: `project_point` names the
**wrist joint**, while the game draws a palm-centred cartoon hand, so a residual of
about one hand-width is expected and is not counted as error. The method cannot pin
the field to a degree; it can only kill a hypothesis, which is all it was asked to do
here.
