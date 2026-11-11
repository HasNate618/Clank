# Which generator drives the body: MotionBricks, Kimodo, or ARDY

**Date:** 2026-11-11
**Status:** decision recorded. Code read directly from the three repositories (shallow clones, all `LICENSE`-checked); nothing was run.
**Question:** the reach plan (`docs/superpowers/specs/2026-11-11-ardy-reach-design.md`, originally MotionBricks-first) bet on MotionBricks. Three NVIDIA motion generators now exist in the same family. Which one should produce the body's reference motion, and what does picking it cost?

**Answer.** Pursue **ARDY**; fall back to **MotionBricks + Kimodo-authored still poses**; do not train a motion model.

The reason is one asymmetric fact: ARDY and Kimodo accept **end-effector constraints** (`right-hand` at a 3D point, feet planted, root waypoint), and MotionBricks does not — its targets are full 36-float poses only. That single difference is why our spec needs a hand-built IK still frame (Task 5) and a four-generation step chain (Task 8), and why its two biggest risks exist at all.

Claim tags: `[REPO]` read in the named file; `[CITED]` external source; `[MEASURED]` local run; `[ASSUMPTION]` untested.

---

## 1. What each one is

| | MotionBricks | Kimodo | ARDY |
|---|---|---|---|
| Method | masked in-between over motion tokens | kinematic motion diffusion | autoregressive diffusion ("interactive Kimodo") |
| Regime | real-time, chunked | offline | streaming, replans per window |
| G1 checkpoint | `mmm_v6_g1` | `Kimodo-G1-RP-v1` | `ARDY-G1-RP-25FPS-Horizon52` |
| Rate | 30 Hz `[REPO: docs/unitree_g1_model.md]` | 30 Hz `[REPO: docs/source/user_guide/motion_convert.md:7]` | 25 Hz, horizon 52 frames `[REPO: README.md checkpoints table]` |
| Chunk | 4 context + ≤4 target poses → 18–58 frames (0.6–1.9 s) `[REPO: docs/unitree_g1_model.md]` | whole clip | 2.08 s window, receding |
| Speed | 2.67 ms FP32 / 1.63 ms FP16, N=1, 4090 `[REPO: docs/onnx_inference.md]` | 100 denoise steps default `[REPO: docs/source/user_guide/cli.md:76]` | ~33 ms/step, 4 steps, 4090 `[CITED: arXiv:2607.08741]` |
| Runtime | ONNX / TensorRT, no Python ML stack | PyTorch | PyTorch ≥2.4, C++ ext, optional TRT |
| Licence | Apache-2.0 code, NVIDIA Open Model weights | same | same |

All three target the **same 34-joint G1 skeleton** — ARDY's is `G1Skeleton34` `[REPO: ardy/skeleton/definitions.py:286]`, MotionBricks' is the identical `*_skel` naming, and Kimodo's G1 CSV is *exactly* our 36-float pose (root xyz + quat + 29 angles) `[REPO: docs/source/user_guide/motion_convert.md:39]`. So the packing and resampling layers are portable across all three.

## 2. The constraint channel is the whole decision

MotionBricks takes poses and nothing else. Its hand/foot pins are documented **SOMA only** `[REPO: docs/runtime_api.md, TargetPoseInferenceParams.ik_pin_mask]` — and `TargetPoseIKCorrector` exists precisely because the generated motion only approximately reaches a target.

Both Kimodo and ARDY take sparse, arbitrary-frame, long-horizon constraints:

```
root2d          root waypoints / paths on the ground plane
fullbody        full pose keyframe (joint rotations + root)
left-hand / right-hand / left-foot / right-foot   end-effector positions + global rotations
end-effector    arbitrary joint sets
```

`[REPO: kimodo docs/source/user_guide/constraints.md; ardy constraints.py:56,138,257,415-439]`

ARDY names the G1 end-effectors explicitly: hands are `right_wrist_yaw_skel` / `right_hand_roll_skel`, feet are `right_ankle_roll_skel` / `right_toe_base` `[REPO: ardy/skeleton/definitions.py, G1Skeleton34]`. That is the wrist we already report to the game.

Consequence, in our plan's terms:

| Our plan's part | Why it exists | Under ARDY |
|---|---|---|
| Task 5: mink invents the still frame | MotionBricks cannot be told "hand here" | **gone** — hand is a constraint |
| Task 5's OOD risk (spec §7.2, §10 risk 1) | a hand-IK pose is outside the mocap distribution | **gone** — pose comes from the motion prior |
| Task 8's step chain, ≤4 generations / 8 s | no long-horizon goals, 1.9 s chunks | **gone** — one `root2d` waypoint |
| Task 2's 35° yaw gate + 5 cm residual | decides plant vs construct-a-stepped-pose | becomes "add a root waypoint"; mink demotes to a reachability check |
| Task 1 packing, Task 3 mailbox, Task 4 resample, Task 7 playback | model-agnostic | **unchanged** |

## 3. What actually takes VRAM in ARDY

~90% of it is one module that does not have to be on the card, and does not run per frame.

| Consumer | Size | Notes |
|---|---|---|
| **Text encoder** — LLM2Vec over `Llama-3-8B-Instruct` + PEFT adapter, `llm_dim: 4096` `[REPO: ardy/model/load_model.py:20-30]` | **~14 GB** bf16 `[REPO: README.md:192]`, "the ~16GB local encoder" `[REPO: scripts/interactive_demo/loading.py:17]` | runs once per prompt; the repo already provides three ways off the card: `TextEncoderAPI` HTTP service on :9550, `cpu/bfloat16`, and a per-prompt embedding cache `[REPO: load_model.py:88-111; scripts/run_text_encoder_server.py; scripts/interactive_demo/embedding_cache.py]` |
| **Denoiser**, ×3 when CFG is `separated` | `cfg_type="separated"` concat-batches B=1 → **B=3** every step (text-only / constraint-only / unconditional) `[REPO: ardy/model/cfg.py:75-77]` | `CFG_TYPES = ["nocfg", "regular", "separated"]` `[REPO: cfg.py:9]` — this is a config flag, not a fixed cost |
| **Context length** | history crop (≤ `FPS × 10 s − horizon`) and future crop (constraints past the horizon) `[REPO: README.md model tab]` | attention memory, the one cost we control per call |
| **Body autoencoder / decoder** | small | tokenizer in, motion out |
| **TensorRT** | build reserves a 4 GB workspace `[REPO: scripts/export_onnx.py:304]` | build-time only |
| **Demo scaffolding** | viser viewport, skinned meshes, CUDA context, allocator cache | not run by us |

So the question "does ARDY fit on an 8 GB card with the game holding 5.2 GB" is really "does the denoiser fit at batch 1, horizon 52, bf16, with CFG dialled down" — plausibly ~1–3 GB, **unmeasured**, and the parameters live in the HF checkpoint config, not the repo.

## 4. Text: nothing drops in, but there are three cheap exits

`text_feat is None` means "encode the strings for me", not "no text" `[REPO: ardy/model/ardy_model.py:557]`. Text is required by the API — but the guidance wrapper already runs a zero-text pass, so an all-zero tensor is a shape the model was trained against.

| Path | What it costs | What it buys |
|---|---|---|
| **0. Zero text** | nothing; one experiment | no encoder, no 14 GB, no gated HF token. Risk: blander motion without semantics — may not matter, our constraints carry the task |
| **1. Distil the tower** | one cloud afternoon; teacher generates the labels | ~300 MB ModernBERT/GTE-class encoder that reproduces the 4096-d space; frozen motion model, no dataset licence involved |
| **2. Small LLM2Vec + projection** | same training, bigger student | 2.6 GB (`Meta-Llama-3.2-1B`) or 6 GB (3B) `[CITED: McGill-NLP/LLM2Vec collection]` |
| **3. Retrain the motion model** | hundreds of GPU-hours + a corpus | not this project |

Nothing off the shelf is drop-in: the 4096-d LLM2Vec space is baked into the trained cross-attention, so any other encoder is a space change that must be learned. Distillation is the small, well-posed version of "swap the tower".

## 5. Datasets, and the licence that bites

| Corpus | Size | G1 native | Licence |
|---|---|---|---|
| **BONES-SEED** | 142,220 motions, ~288 h, 522 actors, G1 CSV at 120 fps, up to 6 NL captions + temporal segmentation `[CITED: hf.co/datasets/bones-studio/seed]` | yes | **gated**: academic or sub-\$1M-startup only; no redistribution; forbids Results that are a commercially substitutable motion generator, while **explicitly permitting** control policies and tracking models `[CITED: bones.studio/info/seed-license]` |
| Bones Rigplay (RP1, 700 h) | what every model above was trained on | yes | proprietary, unavailable |
| HumanML3D | 14,616 motions / 44,970 captions / 28.6 h, SMPL-H 20 fps | no | non-commercial, AMASS + SMPL registration |
| LaFAN1-G1 | 2.4k clips / 4.6 h | retargeted | non-commercial |
| KIT-ML | 3,911 / 6,278 | no | non-commercial |

Two things follow. The only G1-native corpus at scale is licence-gated, and its terms are aimed at exactly "train a generative motion model", while permitting trackers and policies. And the corpus we actually need — constraint→motion pairs — can be *mined* from any motion source by masking joints, which is how ARDY trains its constraint channel; that includes mining from MotionBricks or Kimodo outputs, whose weights are under the Open Model Agreement and whose outputs carry no dataset licence.

## 6. Decision

**Pursue ARDY for live reach and grab.** Constraints in, motion out, SONIC tracks it:

```
right hand at the object (+ facing)     ← the constraint MotionBricks cannot take
both feet planted, or a root2d waypoint when genuinely out of reach
head/chest toward it
history = last N achieved poses
   → ARDY-G1 (25 Hz, horizon 52) → resample 25→50 → SONIC mode 0 → MuJoCo → relay
```

**Fallback ladder**, each rung losing only look quality, never capability:

1. ARDY live, constraints, text off the card.
2. **MotionBricks live + Kimodo-authored still poses.** Kimodo generates the still frames once from `right-hand` + foot constraints (so they are in-distribution, which removes the plan's worst risk), MotionBricks in-betweens them at 2.67 ms with no Torch in the loop. This is a *strict improvement* to the current plan even if ARDY never runs.
3. The current `reach_vr` cruise.

**Not doing:** training a motion model. Compute, licences, and the fact that a 4060 has to run the result all say no.

## 7. Open questions, and the experiment that settles each

1. **Does ARDY's denoiser fit beside the game?** Weights are known: `nvidia/ARDY-G1-RP-25FPS-Horizon52` is 775 MB total (`denoiser.safetensors` 632 MB, `tokenizer.safetensors` 143 MB), public and ungated `[REPO: HF API, 2026-11-11]`. So the open part is only activations at batch 1 / horizon 52 / bf16, `separated` vs `nocfg`. *(game-down spike, one session)*
2. **Is constraint-only generation usable?** Same constrained reach with text vs zero text; compare hand-constraint adherence and how human the path looks. *(same spike)*
3. **Is ARDY's G1 output in our qpos layout?** **Answered before the spike.** `MujocoQposConverter` exports "root (pelvis, 7 = translation + rotation) + 29 dof joints (29)", z-up, +X forward, right-handed `[REPO: ardy/exports/mujoco.py:21-32]` — our pose36 exactly. ARDY's *internal* space is y-up/z-forward over 34 joints, and the converter owns that change, so Tasks 1 and 4 plug in unchanged.
6. **Is the still-frame path needed at all?** `scripts/generate.py --model g1 --duration N --constraints x.json` writes both ARDY npz and a MuJoCo qpos CSV `[REPO: scripts/generate.py:303-316]`, so the first constrained generated reach is a CLI call, not a bridge script. The CLI has no text-free flag, which is why the text-free arm needs a short library call instead.
4. **25 → 50 Hz resampling:** our Task 4 already parameterises `src_hz`; verify the velocities, not just the positions, because SONIC's look-ahead compares both.
5. **Does SONIC track ARDY motion as well as MotionBricks motion?** Unchanged from the spec's kill criterion: >15 cm means stop, not fall back to the cruise.

## 8. What this does to the in-flight plan

Tasks 1–4 and 7 are unaffected and worth finishing as written. The decision point is **Task 5**:

- If ARDY is pursued: Task 5 becomes "author constraints, and keep mink only as a reachability check"; Task 6 becomes an ARDY adapter; Task 8 loses the step chain.
- If ARDY fails its spike: Task 5 changes to "Kimodo authors the still poses" instead of mink, and the rest of the plan stands as written.

---

## 9. ARDY on this machine — first measurements (2026-11-11)

Run on the RTX 4060 Laptop (8 GB, driver 610.57.04) with the GPU to itself, `torch 2.11.0+cu128`,
`ARDY-G1-RP-25FPS-Horizon52`, text tower absent. Probe: `spike/ardy/probe_g1.py`.

NixOS prerequisite, and it is easy to lose an hour to: `libcuda.so.1` lives under
`/run/opengl-driver/lib` and is not on the loader path, so torch reports `cuda False` with a
perfectly good driver until `LD_LIBRARY_PATH=/run/opengl-driver/lib`.

| Quantity | Measured |
|---|---|
| Weights resident | **0.93 GB** |
| Peak VRAM, one 52-frame window | **0.94 GB** allocated / 0.96 GB reserved |
| Generate, 52 frames (2.08 s of motion), 10 steps | **0.07–0.20 s** |
| Hand residual, `cfg_weight=(1, 2)`, zero text | **60.3 mm** (45.5 mm in the factored adapter, 33–80 mm across unseeded runs) |
| Hand residual, `cfg_weight=(1, 4)` | 68.3 mm |
| Hand residual, `cfg_weight=(2, 0)` (scalar default) | 869.8 mm |
| Root waypoint adherence, target −0.80 m in z | **−0.790 m**; 0.60 m requested → 0.566 m travelled |
| qpos export | `(1, 52, 36)` — our pose36 exactly |

So §7.1 is answered: **0.94 GB peak, comfortably inside the ~2.6 GB the game leaves free**, and
generation is 10–30× faster than the motion it produces. §7.2 is answered in the sense that matters
most: constraints are followed, and 60 mm of adherence is the same order as SONIC's own tracking
error, so the residual a plan has to close is the tracker's, not the generator's.

One trap, and it is silent: with `cfg_type="separated"`, a scalar `cfg_weight` sets the **constraint**
guidance weight to **zero** (`ardy_model.py:274-279`; `w_text, w_cstr = cfg_weight, 0.0`). A
constrained run under the default therefore ignores its constraints and returns 870 mm. Pass
`cfg_weight=(text, constraint)` — two values — or constraints are decoration.

`num_denoising_steps` must be ≤ `num_base_steps` (10 for this checkpoint, `config.yaml`); 25 or 50
trip a device-side assert rather than a clean error.
