# Language to action on a Unitree G1 — METATRON and the off-the-shelf intent stacks

**Question.** What turns a sentence like "pick up the salt shaker and shake over the food" into G1 motion? Is METATRON relevant, and are there already built solutions for turning intent into action on the G1?

**Evidence tags.** `CITED` = a source link; `MEASURED` = a local measurement (none in this document); `ASSUMPTION` = a design hypothesis; `SECONDARY` = a source that reports someone else's numbers and was not checked against the primary artifact.

**Answer.** The sentence is two problems, not one. **Intent** is language + perception turning a request into a skill and an object target. **Execution** is a fast controller turning that skill into whole-body motion. METATRON is a released, directly relevant **execution layer** for the G1: one frozen promptable policy plus a skill layer written in ordinary code. It is not a language layer, and it does not perceive the salt shaker. Separately, several complete intent-to-action stacks now exist for the G1 — Unitree's UnifoLM-VLA-0, NVIDIA's GR00T N1.7 reference workflow, `unitree_lerobot`, and LeRobot — but none ships the salt-shaker behavior. The realistic build is a fine-tuned VLA for intent plus a whole-body controller for execution, with 100–300 teleoperated demonstrations of the exact task.

---

## 1. The decomposition

```text
"pick up the salt shaker and shake over the food"
  │
  ├─ intent + grounding        language model / VLM + object detector
  │     → skill sequence + object poses
  │       [find, reach, grasp, lift, move_over, shake, release]
  │
  ├─ skill layer               ordinary code or a VLA action head
  │     → typed targets at 10–30 Hz
  │
  └─ execution layer           whole-body policy + PD
        → 29 joint commands at 50 Hz, PD at 500–1000 Hz
```

The split matters because the pieces are independently replaceable. METATRON and SONIC live in the bottom box. UnifoLM-VLA-0 and GR00T live mostly in the top two. `unitree_lerobot` and LeRobot are the plumbing that connects them. No published G1 stack does all three well out of the box.

---

## 2. METATRON — what it is, and why it is relevant

METATRON is NVIDIA's promptable whole-body execution layer, released 2026-10-28 with weights, a native G1 package, and an ONNX browser demo [CITED](https://github.com/NVlabs/ProtoMotions), [CITED](https://huggingface.co/nvidia/Metatron-v1), [CITED](https://research.nvidia.com/labs/par/projects/metatron/). The paper is arXiv 2610.38027 [CITED](https://searcharxiv.org/abs/2610.38027).

### 2.1 The interface

The policy takes a **partial prompt** `g_t`, not a verb and not joint angles. A prompt combines four optional goal streams plus scene context [CITED: paper §III-B]:

| Stream | What it specifies |
|---|---|
| Spatio-temporal goals | body positions or orientations in world coordinates, each with a time-to-arrival; position and orientation are independently optional |
| Canonicalized pose goals | a desired pose independent of world placement, relative to the target torso and heading |
| Velocity | planar travel velocity in the heading frame, no endpoint |
| Style cue | an example clip whose pooled frames bias the character of the motion |
| Scene | bounding boxes; the G1 accepts one object represented by up to two boxes |

Masks identify which bodies, properties, and timesteps are constrained. An omitted goal stays unspecified and the policy generates a completion. The paper's own summary: "Goals specify what to do. Metatron coordinates how to move." [CITED](https://research.nvidia.com/labs/par/projects/metatron/)

Above the policy sits a **skill layer**: stateful programs that construct prompts at each control step from task intent. Skills are ordinary code, authored by humans or coding agents, and refined offline against simulation without retraining the policy. The paper's examples are steering, sitting, recovery, interactive compliance, and teleoperated pickup. The steering skill, for instance, fits a periodic foot-target rule from locomotion clips; sitting reduces to three static targets derived from the seat box; recovery constrains torso height [CITED: paper §IV].

### 2.2 What is proven on the G1

- One frozen checkpoint per embodiment. On the physical G1 it supports walking, recovery, teleoperated object pickup (an apple), and sitting on a sofa [CITED: paper abstract, §VI].
- The G1 policy distills four specialist trackers (BONES, AMASS+LAFAN, recovery, SAMP) and runs at 50 Hz control with 200 Hz simulation, four physics steps per action [CITED: paper §V, appendix].
- Deployment is a C++ TensorRT/CUDA-graphs stack on a laptop with an RTX 5090; measured sensor-to-command processing is 1.3 ms, not the control period. An Xbox controller and PICO headset supply runtime commands [CITED: paper §VI, appendix H].
- Whole-body reaching: higher reached rate than ARDY+SONIC and a GRU+IK+SONIC pipeline across 54 wrist targets; the gap is largest where the robot must step or counterbalance [CITED: paper §VI].
- Push robustness: higher observed success than MaskedMimic+ and HOVER+ at every nonzero force, though intervals overlap and all methods fall below 10% at 600–800 N [CITED: paper §VI].

### 2.3 What it does not do

This is the load-bearing limitation for the salt-shaker question:

> "The task system remains responsible for goal selection and long-horizon planning. We place the G1 near a chair or sofa and supply a nearby-object cue; autonomous approach followed by sitting is not demonstrated. Localization and perception could enable automatic goal generation, while LLM-generated programs could sequence goals for longer-horizon tasks." [CITED: paper §VII]

Also stated: the repertoire depends on training coverage; abruptly commanding 2 m/s from rest often triggers a front roll; unspecified goals or incidental style cues can produce unintended motion [CITED: paper §VII].

So METATRON is the best-fit **body** for "shake over the food" once something else has found the shaker, decided the grasp, and written the shake skill. It is not the thing that understands the sentence, and the hardware demonstrations are teleoperated rather than autonomous.

---

## 3. Built intent-to-action stacks for the G1

| Stack | Intent input | Action space | Status | Gap for the salt shaker |
|---|---|---|---|---|
| **Unitree UnifoLM-VLA-0** | Text instruction + image; Qwen2.5-VL-7B backbone | Reported as 31-D upper body: two 7-DoF arms + 3-DoF waist + two 7-DoF Dex3 hands `[CITED, SECONDARY]` | Open code, weights, and datasets, 2026-01-29; 12 task categories on one policy incl. pick/place, pour, stack, fold [CITED](https://github.com/unitreerobotics/unifolm-vla) | Upper body only; legs need a separate controller; "shake over food" is not one of the 12 categories, so it needs fine-tuning data |
| **Unitree UnifoLM-WLA-1.0** | Multimodal instruction | Whole-body manipulation, 6B parameters | Announced on Unitree's open-source page: ~2,500 h real data, 64 tasks, two-finger and five-finger hands [CITED](https://www.unitree.com/opensource/) | Weights, license, and reproducibility were not verified in this pass |
| **NVIDIA GR00T N1.7 + Isaac ROS / Isaac Lab** | Language instruction + head RGB + proprio | Per-embodiment; G1 apple-to-plate uses arm/hand/waist plus optional locomotion | Full reference workflow: teleop → MCAP → LeRobot → fine-tune (40+ GB VRAM) → LEAPP export → deploy on Jetson Thor [CITED](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/index.html) | Task-specific; the provided apple-to-plate artifact is a tutorial, not a general manipulator |
| **GR00T-WholeBodyControl + SONIC** | VLA predicts **64-D SONIC latent motion tokens** | Latent tokens decoded by SONIC into full-body joints at 50 Hz | Documented VLA workflow on G1 with SONIC [CITED](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html) | Same data requirement; the latent space is a learned command space, so skills are not directly editable |
| **unitree_lerobot** | LeRobot datasets from AVP teleoperation | ACT, Diffusion Policy, pi0, pi0.5, GR00T | Actively maintained Unitree repo; record → convert → train → deploy [CITED](https://github.com/unitreerobotics/unitree_lerobot) | Training framework, not a behavior; you supply the demos |
| **LeRobot native G1** | Task string + cameras | VLA policies over G1 state/action; sim and real | 29/23-DoF support, teleoperate, record, train, rollout, Holosoma/Groot locomotion controllers [CITED](https://huggingface.co/docs/lerobot/unitree_g1) | Same; good default plumbing |
| **METATRON** | Code-generated prompts (body targets, velocity, style, boxes) | 29 joint commands at 50 Hz | Released weights + native G1 package + ProtoMotions code [CITED](https://huggingface.co/nvidia/Metatron-v1) | No language, no perception, no grasp synthesis; teleoperated pickup only |
| **Research: hierarchical VLM + imitation skills** | GPT-4o planner + Gemini skill monitor | Learned skill policies over a whole-body RL controller | 73% full-sequence success over 40 real G1 trials on a non-prehensile pick-and-place [CITED](https://arxiv.org/html/2506.22827v2) | One task, two tables, controlled scene |
| **Research: VLM-RL in Isaac Lab** | Qwen3-VL decomposes language into skills | Locomotion + arm + finger policy cascade | Open repo, simulated; 4/7 multi-env walk-to success, 7/8 standing [CITED](https://github.com/mturan33/isaac-g1-vlm) | Simulation only; grasp is scripted/magnetic |

### 3.1 Reality check on success rates

The published numbers are early:

- A 2026 thesis fine-tuned GR00T and pi0.5 on a physical G1 with Inspire hands for an industrial pick-and-place. The best online success rate was **42.5%**, described as a proof of concept rather than a mature solution [CITED](http://urn.kb.se/resolve?urn=urn%3Anbn%3Ase%3Amdh%3Adiva-77603).
- The hierarchical VLM + imitation system reached **73%** over 40 real trials, with most failures in the grasp skill when object placement left the training distribution [CITED](https://arxiv.org/html/2506.22827v2).
- NVIDIA's `g1_locomanip_finetune` checkpoint is explicitly **simulation-only** and "not intended for deployment on physical robots" [CITED](https://huggingface.co/nvidia/g1_locomanip_finetune).
- A GR00T N1.7 piston pick-and-place fine-tune on 102 episodes is validated in Isaac Lab; real-robot behavior is untested [CITED](https://huggingface.co/jren313/g1-inspire-piston-starvla-gr00t-n1d7).

The pattern is consistent: a single-task policy trained on 50–350 demonstrations can work in the demonstrated scene, and generalization is the open problem.

---

## 4. The salt-shaker sentence, mapped

"Pick up the salt shaker and shake over the food" is harder than a pick-and-place benchmark for four reasons:

1. **Grasp.** The bare 29-DoF G1 has no fingers. A Dex3-1 or a five-finger hand is required. The grasp must survive acceleration, so grip force or slip detection matters; open-loop finger position will drop a full shaker.
2. **Object grounding.** "The salt shaker" and "the food" are open-vocabulary targets. A detector or VLM must return their 3D positions or masks from the head RGB-D camera. METATRON can consume a box pose; it cannot produce one.
3. **Contact-rich shaking.** Shaking is a periodic wrist/arm motion under a maintained grasp. It is not a locomotion or reaching primitive, and it is not in UnifoLM's 12 categories. "Pour" is the closest published category.
4. **No object state.** The robot cannot see how much salt remains. The command has to be "shake N cycles," not "shake until enough."

A workable decomposition:

```text
find(salt_shaker)      VLM/detector → 3D box or mask
find(food)             VLM/detector → 3D point
reach(shaker)          VLA or IK skill
grasp(shaker, force)   hand policy / force control
lift(0.15 m)           VLA or skill
move_over(food)        VLA or skill
shake(amplitude, n)    scripted periodic target, grasp maintained
release()              open hand
verify()               shaker lifted? inverted over target? retry
```

METATRON fits the middle rows: it can take hand targets, torso targets, and a scene box, and it coordinates balance while the arm moves. It cannot do `find`, `grasp`, or `shake` by itself. UnifoLM-VLA-0 or a fine-tuned GR00T can do the whole upper-body sequence, but only after training on demonstrations of that sequence.

---

## 5. Recommended composition

**For a tabletop demo (fixed base, one object):**
1. Upper-body VLA: UnifoLM-VLA-0 or GR00T N1.7 fine-tuned on 100–300 teleoperated episodes of the exact task.
2. A Dex3-1 or five-finger hand.
3. A separate locomotion/balance controller, or a fixed stance.
4. A scripted shake skill layered on top if the VLA does not learn the oscillation cleanly.
5. Verification by a cheap visual check, not by trusting the policy.

**For whole-body competence (walk to the table, then manipulate):**
1. METATRON or SONIC as the execution layer.
2. A thin skill layer in code for reach/grasp/shake, with the object box supplied by perception.
3. A VLM/LLM planner at 0.5–1 Hz that selects skills and object targets. Do not put the language model in the 50 Hz loop.
4. Teleoperated demonstrations for the grasp and shake, because neither METATRON nor the locomotion policy supplies them.

**What not to expect:** a base GR00T or UnifoLM checkpoint to accept "pick up the salt shaker and shake over the food" in a new kitchen. Every credible result above is task-specific and scene-specific, and the best real-world number found is 42.5% on a constrained industrial task.

---

## 6. Relation to Clank

METATRON is a candidate **executor** behind the existing intent layer, not a replacement for it. Its prompt interface is a different command space from `BodyCommand`: it accepts timed body goals, velocity, style, and scene boxes, while `BodyCommand` is a masked task-space command with TTL and authority bits. Adopting METATRON would mean writing an adapter from `BodyCommand`/skills to METATRON prompts, or replacing the SONIC executor outright. That is a design decision, not a conclusion of this note.

The useful overlap is the architecture: METATRON independently arrives at the same split this repo already uses — a slow writer that names goals, a fast frozen body that owns feasibility, and skills as ordinary code. The difference is that METATRON ships a stronger whole-body policy and an explicit scene-box channel, while Clank's current executor is SONIC with a masked task-space contract.

---

## 7. Uncertainty

- **METATRON hardware results are teleoperated.** The paper demonstrates object pickup with PICO hand targets, not a language instruction. Do not read the abstract as autonomous language-conditioned manipulation.
- **The METATRON paper is a preprint.** Numbers above are from the paper text; they were not reproduced. The arXiv identifier comes from a search index, not a DOI.
- **UnifoLM-VLA-0 action dimension is secondary.** The 31-D upper-body layout comes from a third-party write-up [CITED, SECONDARY](https://vnrobo.com/en/blog/research-unifolm-vla-g1); the GitHub README does not state the action width.
- **UnifoLM-WLA-1.0 was not verified.** The 6B / 64-task / 2,500-hour figures are from Unitree's open-source page; weights and license were not checked.
- **GR00T workflow requirements are from NVIDIA docs.** The 40+ GB VRAM figure and Jetson Thor deployment path were not run here.
- **No salt-shaker benchmark exists.** The task decomposition in §4 is an `[ASSUMPTION]` assembled from adjacent systems, not a published result.
- **Success rates are not comparable across papers.** Different objects, scenes, success definitions, and trial counts. The 42.5% and 73% figures should be read as orders of magnitude, not a ranking.

---

## 8. Sources

- METATRON paper — [arXiv 2610.38027](https://searcharxiv.org/abs/2610.38027); local copy at `/home/nate/Downloads/metatron-paper.md`
- METATRON project page — [research.nvidia.com/labs/par/projects/metatron](https://research.nvidia.com/labs/par/projects/metatron/)
- METATRON weights — [nvidia/Metatron-v1](https://huggingface.co/nvidia/Metatron-v1)
- ProtoMotions / METATRON release — [NVlabs/ProtoMotions](https://github.com/nvlabs/protomotions/)
- Unitree UnifoLM-VLA-0 — [GitHub](https://github.com/unitreerobotics/unifolm-vla); [Unitree open source](https://www.unitree.com/opensource/)
- Unitree `unitree_lerobot` — [GitHub](https://github.com/unitreerobotics/unitree_lerobot)
- LeRobot Unitree G1 — [Hugging Face docs](https://huggingface.co/docs/lerobot/unitree_g1)
- NVIDIA GR00T G1 reference workflow — [course](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/index.html); [Isaac ROS tutorial](https://nvidia-isaac-ros.github.io/v/release-4.5/reference_workflows/isaac_for_physical_ai/tutorials/tutorials.html)
- GR00T-WholeBodyControl VLA workflow — [NVlabs docs](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html)
- Hierarchical VLM + imitation on G1 — [arXiv 2506.22827](https://arxiv.org/html/2506.22827v2)
- VLM-RL G1 pick-and-place — [mturan33/isaac-g1-vlm](https://github.com/mturan33/isaac-g1-vlm)
- Industrial G1 VLA thesis — [URN](http://urn.kb.se/resolve?urn=urn%3Anbn%3Ase%3Amdh%3Adiva-77603)
- Simulation-only G1 locomanip checkpoint — [nvidia/g1_locomanip_finetune](https://huggingface.co/nvidia/g1_locomanip_finetune)
- G1 piston GR00T N1.7 fine-tune — [jren313/g1-inspire-piston-starvla-gr00t-n1d7](https://huggingface.co/jren313/g1-inspire-piston-starvla-gr00t-n1d7)
- UnifoLM-VLA-0 secondary summary — [VnRobo](https://vnrobo.com/en/blog/research-unifolm-vla-g1)
