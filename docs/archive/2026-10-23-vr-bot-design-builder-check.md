# Builder check — open questions before planning

> **Archived 2026-11-13.** The open-questions audit taken before implementation began. The questions were answered by building; kept as the audit trail for `2026-10-23-clank-design.md`.
> Current behaviour: [`README.md`](../../README.md) · agent rules: [`AGENTS.md`](../../AGENTS.md).

**Date:** 2026-10-23
**Inputs:** `docs/2026-10-23-clank-design.md`, E1–E3 spike results, `research/humanoid-command-space.md`, `RESEARCH.md` §5/§12, `stubborn-gamer/SEED.md`.
**Scope:** Only questions where a wrong answer changes what gets planned. §15 author calls and §16 builder choices are treated as closed. Ordered by blast radius: #1–#2 can invalidate the milestone; #3–#4 change the architecture; #5 changes C5's scope.

## 1. Is the game-side path known to work, or is E4 the first test?

**Question.** Has any actual OpenVR title — specifically Superhot — been shown to accept the remote-injected HMD through xrizer, with working trigger/grip input and grim-capturable pixels? If not, should a minimal game-side smoke test be added before C1, or is E4 allowed to be the first go/no-go?

**Why it matters.** E1–E3 prove the `mndrmt3` path for `hello_xr` / `xr_view_probe`. They do not prove that a Proton/OpenVR title accepts the synthetic HMD, that xrizer maps the remote driver's `trigger_value` / `squeeze_value` / buttons, or that `grim` captures the game's window rather than only the `hello_xr` `openxr` window. The design treats "OpenVR titles reach Monado via xrizer" as a fact and puts the first real game at E4, after C1–C5. If the game refuses the synthetic HMD, the input path is dead, or the window is not capturable, the failure is integration, not body control—and it lands after most of the work. §16 leaves the *launch command* to the builder; this is not that question. It is whether the go/no-go evidence exists before committing to the body-first order.

**What settles it.** Either add a C0/E4a smoke test (launch Superhot or any OpenVR title under the existing Proton/xrizer/Monado stack, confirm the rendered view follows an injected pose, confirm trigger/grip reaches the game, confirm `grim` captures the game window), or explicitly accept E4 as a late go/no-go and state that the body work is worth doing even if the game path fails.

## 2. What must the demo show to justify "balance-constrained whole-body control"—and is C3 on the critical path?

**Question.** Is the portfolio claim satisfied by C1–C2 (E3-style stand PD, arms/waist moving, base excursion logged), or does it require C3 (Playground ONNX legs) and/or a deliberate perturbation/recovery test?

**Why it matters.** The intent names **balance-constrained whole-body control** as the claim that must be true on camera. E3 is a floating-base G1 held by Menagerie position actuators while scripted arms move; base excursion was ≤3.08 cm, but there was no balance controller, no perturbation, and no locomotion policy. C3 is in the work order but its allowed fail is "station-keeping still bad," and E4 is gated only on C1–C2. The portfolio success bar (§17) does not require walking. If the claim needs a real balance controller or a perturbation test, C3 and a new test are critical; if a standing body whose base reacts to arm motion is enough, the plan can drop C3 from v1 and say so in the README. Right now the doc can be read either way, and the difference is substantial work.

**What settles it.** Pick the minimum evidence: (a) C1–C2 stand PD + arm motion + logged base excursion, (b) C3 ONNX walking under velocity command, or (c) a deliberate perturbation/recovery test. Then state the exact claim the video and README are allowed to make for that choice.

## 3. What do `hold` and `look` actually do in v1, and what happens to unsupported authority bits?

**Question.** Is `hold` a world-frame freeze of the achieved head/hands, a root-relative pose hold plus base-velocity hold, or stand PD during think-time? Which controller owns it once the ONNX is loaded, and what does the executor do with a known authority bit it reports as `UNSUPPORTED`? Related: what happens to `look` when the requested yaw exceeds the ±0.6 rad waist share and the base-yaw remainder is `UNSUPPORTED`?

**Why it matters.** `hold` is load-bearing: the VLM only thinks while hold holds, and Superhot advances time when the tracked HMD/controllers move. The design says hold freezes "base + wrists in game space," but `BodyCommand` poses are root-relative and `hold` reuses last achieved poses; if the base drifts, root-relative hands move in game space. That is not hypothetical once the ONNX is loaded: the Playground policy drifts **0.475 m in 5 s** and **~1.18 m at p95 think time**. The doc offers three fixes (bias the velocity command, add a documented residual, keep stand PD for think-time) but does not choose. The frame story is also inconsistent: `BodyState` achieved poses are called "same Pose type" and "MuJoCo world for state," while command poses are root-relative. `AUTH_ROOT_VEL` is `UNSUPPORTED` in v1 even though `hold` and `look` set it; whether an unsupported bit rejects the command, is ignored, or must be masked by the adapter is undefined. The same ambiguity sets the v1 look arc: waist yaw ±0.6 rad plus an unsupported base-yaw remainder means the robot can only look within roughly ±35°, and the doc does not say what happens when the request exceeds it. Superhot expects the player to turn. A wrong choice changes C1's validator, C2's hold, C3's mode switching, and whether `look` is usable at E4.

**What settles it.** Define: (i) the root frame (pelvis body frame vs gravity-aligned yaw frame) and the frame of `BodyState` head/hands; (ii) whether `hold` is world-frame achieved poses, root-relative poses plus base-velocity hold, or stand PD; (iii) whether unsupported authority bits reject, ignore, or must be masked; (iv) whether `look` beyond the waist share clamps, returns `UNSUPPORTED`, or turns the base—and whether the v1 arc is acceptable for Superhot. Add the hold drift budget that keeps Superhot paused during a 20 s think.

## 4. Which OpenXR reference space and HMD height/scale does Superhot get?

**Question.** Does the HMD packet carry the G1's actual `head_link` position (~1.221 m), a documented fixed eye offset, or a scaled height—and does the game see it in `LOCAL` or `STAGE` with which floor origin?

**Why it matters.** The design never picks `LOCAL` vs `STAGE` and does not mention Monado's tracking-origin `initial_offset`. E3 sends the achieved `head_link` position (~1.221 m) as the HMD; E1's test used 1.6 m. `LOCAL` and `STAGE` differ by a floor offset and yaw, and Monado applies a tracking-origin offset; a wrong choice produces persistent height/yaw bias. Superhot is a room-scale title built around human eye height. At 1.22 m the camera is child-height; with `LOCAL` the origin may be at the initial head pose. A fixed eye offset or scale would fix playability but must be documented as an offset, not smuggled into the "achieved pose" claim. This choice also affects aiming, the recorded view, and whether the G1 can reach game objects.

**What settles it.** Decide: raw `head_link` (honest, low), fixed eye offset (documented), or scale; and specify `STAGE` with a floor origin or `LOCAL` with an explicit offset. Check the game's camera height and floor in Superhot before E4.

## 5. Does the VLM choose a target, or only a skill/side—and if a target, what is the 2D→3D contract?

**Question.** For `reach` / `strike`, does the VLM emit a target point, a 2D bbox/point that something converts to 3D, or only the skill and side with the adapter supplying E3 local offsets?

**Why it matters.** The adapter signature is `reach(side, p)` / `strike(side, p)` with a 3D root-relative `p`, but the VLM sees one monocular still. §8 allows "scripted/root-relative offsets (E3 REACH / PUNCH local vectors)" if no tracker, and §16 leaves tracker implementation to the builder. That makes the simple v1 path legal: VLM names `reach` / `strike` + side, adapter supplies a fixed offset. But the intent says the VLM "selects goals," and §9 mentions an optional bbox in the digest. If the VLM is supposed to pick *which* object or point, the design is missing the 2D→3D rule (fixed depth plane, bbox unprojection, object class) and the owner of that conversion. The difference is whether C5 is a skill classifier or a goal selector, and whether the README's claim is "named a skill" or "chose a target."

**What settles it.** Pick one: (a) v1 VLM emits skill+side only; adapter uses E3 offsets; README says "named a skill"; no 3D target work; or (b) VLM emits a 2D bbox/point and a specified rule converts it to root-relative 3D at a fixed depth/plane; tracker/owner defined. Do not leave the adapter's `p` undefined.
