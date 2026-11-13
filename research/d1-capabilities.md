# d1 decision models — capabilities, runtime parity, and whether d1 is Laya again

**Date:** 2026-10-28. **Scope:** Liquid AI open d1 (`d1-2B`, `d1-Omni-2B`, and the runtime support around them), measured against the job in `../docs/2026-10-23-clank-design.md` §9: name a skill and a side from one frame plus a ≤10-line digest, on CPU, in place of MiniCPM5-2B (20 s p50 / 31 s p95) and LFM2.5-3B (0.21 s, degenerate).

**Claim tags.** `[MEASURED]` = a local spike on this machine, file named. `[CITED: url]` = a source I read today. `[VENDOR]` after a citation means the number was produced by Liquid AI or by a model's own authors. `[ASSUMPTION]` = my reasoning, untested. **The desk study below is `[CITED]`/`[VENDOR]` only — it was written without touching the model.** Local CPU latency was then measured on this machine and is recorded in **§6b**, which supersedes the three places below that say no CPU image number exists (§0, §6, §9.1).

---

## 0. Verdict

**d1 is not Laya, and the difference is structural rather than incremental.** Laya 421M was a 395M ModernBERT encoder with a 26M head, 512-token context, no vision, no first-party runtime, no paper, and a vendor card that admitted near-chance zero-shot accuracy. d1-2B is a 1.92B LFM3 multimodal backbone (1.73B LM + 164M vision encoder + 25M projector) with a 128K context, an in-tree llama.cpp implementation merged upstream into `ggml-org/llama.cpp`, official GGUF/ONNX/MLX exports, and vendor numbers reported against six independently-maintained public decision boards. `[CITED: https://huggingface.co/LiquidAI/d1-2B]` `[CITED: https://huggingface.co/LiquidAI/LFM3-VL-2B]` `[CITED: https://github.com/ggml-org/llama.cpp/pull/31275]`

**But it is not yet a brain, and the reason is the vision path, not the model.** The only independent d1 number anywhere is **text-only**: a third party measured `d1-2B` Q4_K_M at **1.7 s p50 at 4 CPU threads, 1.85 GB RSS**, on 231 text cases. That is ~12× faster than MiniCPM5-2B's 20 s p50 and it fits beside Superhot + Monado + MuJoCo with room to spare. `[CITED: https://github.com/prometheion/nerve-protocol/issues/178]` **No *third-party* CPU number existed for a call including a picture** when this was written; it has since been measured here — **7.0 s at 116 image tokens, 36–39 s at 591** (§6b). The picture is the part Clank needs, and the image budget is 4096 tokens per picture, which on CPU is a multi-second prefill on its own. `[CITED: https://huggingface.co/LiquidAI/d1-2B-GGUF]` `[ASSUMPTION]` So d1's headline "millisecond" advantage is a GPU claim that may not survive the CPU translation, exactly like every other Liquid number we have looked at.

**Recommended posture: worth one cheap test, and the cheap test is text-first, not pixels.** Ask d1-2B `choice` over the ≤10-line digest *without* an image (1.7 s, measured by a third party, no vision encoder at all) and compare to a majority-class baseline on held-out Superhot frames. Liquid's own game demo does exactly this — "The AI does not receive the entire game screen", it gets a text summary of lane occupancy — and it runs the *hosted* `d1.1`, not local `d1-2B`. `[CITED: https://docs.liquid.ai/examples/laptop-examples/build-a-game-with-decision-models]` If d1 cannot beat a frequency table on the digest, pixels will not rescue it and we stop. If it can, the image path is worth 1.9 GB and one ONNX download.

**Two hard blockers to know before anyone builds on this.** (1) llama.cpp's `/v1/systemone` supports **only `noul`, `choice` and `score`** — no multiple-choice, no extraction, no estimates, no boxes — and as of 2026-10-25 the endpoint has open, unfixed concurrency bugs. `[CITED: https://huggingface.co/LiquidAI/d1-2B-GGUF]` `[CITED: https://github.com/ggml-org/llama.cpp/issues/31150]` (2) **No calibration metric is published for d1 at all** — no ECE, no Brier — while the card instructs "Calibrate any decision threshold on data from your own use." `[CITED: https://huggingface.co/LiquidAI/d1-2B]` For a confidence-gated brain that is the single most important missing number, because confidently-wrong is the failure class that killed Laya.

---

## 1. What d1 is

| | d1-2B | d1-Omni-2B | d1-omni-600M | d1-3B |
|---|---|---|---|---|
| Params | **1.92B** (1.73B LM + 164M vision + 25M projector) | **2.17B** | 587M | ~3B (LFM2.5-VL-3B based) |
| Built on | LFM3-VL-2B | LFM3-Omni-2B | LFM2.5-Encoder-350M (bidirectional) | LFM2.5-VL-3B |
| Modalities | text/JSON + image + video | text/JSON + image + video + audio | text + (image **or** audio) | text + image |
| Context | **128K** tokens | 128K | 16K | (card not read) |
| Vocab | 128,000 | 128,000 | — | — |
| Licence | LFM Open License v1.0 | same | same | same |
| Released | HF repo created 2026-10-24 | 2026-10-24 | 2026-10-07 | 2026-10-07 |

`[CITED: https://huggingface.co/LiquidAI/d1-2B]` `[CITED: https://huggingface.co/LiquidAI/d1-Omni-2B]` `[CITED: https://huggingface.co/api/models?search=d1-2B]` `[CITED: https://www.liquid.ai/blog/open-d1]`

- Architecture: LFM3 backbone, 24 layers — **13 gated short convolutions, 5 sliding-window attention, 6 global attention**. Hybrid conv/attention, decoder-only. `[CITED: d1-2B card]`
- 16 languages: en, ar, zh, fr, de, it, ja, ko, pt, es, vi, th, id, hi, ru, pl. `[CITED: d1-2B card]`
- "It is not a chat model." Output has **zero generated tokens** — answers are read off logits at designated answer slots in one forward pass. `[CITED: d1-2B card, ONNX card]`
- **Licence terms:** LFM Open License v1.0 is Apache-2.0-shaped (copyright + patent grant, redistribution conditions) with a **Commercial Use Limitation: free commercial use only below $10,000,000 annual revenue**, above which commercial use "is not licensed under this Agreement." Non-profits exempt for research. `[CITED: https://huggingface.co/LiquidAI/d1-2B/raw/main/LICENSE]` For a portfolio project this is a non-issue; it is not Apache-2.0 and must not be described as such.
- **Vendor's own caveats, quoted:** "Knowledge-heavy questions favor larger models, and answers get less reliable past about 64K tokens of state. Estimates are reliable on counts and scores, less so on free quantities." `[CITED: d1-2B card]` Also: "Without audio, use d1-2B: it is stronger on text and single boxes, reports missing values more often, and is better calibrated" — i.e. **d1-2B is the better-calibrated of the two**, stated by the vendor. `[CITED: d1-Omni-2B card]`

---

## 2. Typed-answer matrix

Every question is `{type, instructions, ...}`; every answer repeats its `type`. `[CITED: https://docs.liquid.ai/lfm/models/decision-models]`

| `type` | Question fields | Exact output shape | Notes / gotchas |
|---|---|---|---|
| `noul` | — | `noul`: **P(yes)**, a float 0–1 | A probability, not a boolean. Threshold it yourself. |
| `choice` | `criteria: {name: description}` | `choice` (winner), `confidence`, `probabilities: {name: p}` | Full distribution over options — this is the gate material. |
| `choice` + `multiple: true` | same | `choices` (all options with **P(yes) ≥ 0.5**), `probabilities` (**each option's own independent P(yes)**, not softmaxed) | Semantics differ from `choice`: per-option binary heads, no normalisation. |
| `score` | `criteria: [level, ...]`, **2 to 10 ordered levels, lowest first** | `score` (**the median level**, continuous), `confidence`, `probabilities` per level, `legend` | Levels index from 0. A score of 2.999 means all weight on level 3. `[CITED: https://docs.liquid.ai/guides/decision-model-guide]` |
| `text` | — | `text`, `span`, `found` | Span offsets into the state. |
| `number` | `unit` (opt) | `number`, `unit`, `text` (span as written), `span`, `found` | Read, not converted: `4 %` → 4 with unit `%`. |
| `date` | — | `date` (ISO 8601), `text`, `span`, `found` | |
| `number` + `range` | `range:[low,high]`, `step` (must divide `high-low`), `unit` | `number` (**the median**), `interval` = **[p10, p90]**, `unit` | **Estimate.** At most **128 values** between low and high. **No `found`.** |
| `box` | — | `box: {top, bottom, left, right}` as **fractions of the picture**, `found`, `confidence` | Requires **exactly one** picture in `media`. |
| `boxes` | `max` (1–16, default **16**) | `boxes[]` (each with box/found/confidence, reading order), `count`, `at_least` | `at_least: true` when every instance read was present, so the picture may hold more than `count`. |

- **Missing values:** a value or box is missing when `found < 0.5`, and is then `None`. For `boxes`, instances under 0.5 are dropped. `[CITED: decision-models docs]`
- **JSON states:** when the state is JSON, a found value also carries `path`, an RFC 6901 pointer. `[CITED: decision-models docs]` — directly useful for a digest-as-JSON brain.
- **Max questions per call:** the ONNX decoder "answers a state and **up to 16 questions**, packed into one row without padding, in a single forward pass with no cache." `[CITED: https://huggingface.co/LiquidAI/d1-2B-ONNX]` The Transformers path exposes `set_size=`, "the default is the model's **trained size (16)**. Larger is faster and costs some accuracy on text." `[CITED: d1-2B card]` The card's marketing line "any number of questions" should be read as "16 in distribution; more is allowed and degrades". **Our contract needs 3–5.**
- **Image token budget:** "A picture takes at most **half of `-ub`**: 8192 keeps the full budget of **4096 tokens per picture**, smaller values lower it." `[CITED: d1-2B-GGUF card]` Video: sampled at **2 fps**, consecutive frames paired, each frame prefixed with its timestamp. `[CITED: https://docs.liquid.ai/lfm/key-concepts/vision-capabilities]`
- **No sampling parameters exist.** "It reads its answers from a single forward pass, so there are no sampling parameters to set." `[CITED: d1-2B card]` This is the whole point for us: no grammar to get wrong, no repetition loop, no truncation.

**Relevance to §9 of the design.** The brain's output is "skill + side only", no 3D point. That is `choice` over ~5 skills plus a `choice`/`noul` for side — **entirely inside the llama.cpp subset**. A `box` for target bearing would be the one genuinely useful addition, and it is exactly the one llama.cpp does not support (§3).

---

## 3. Runtime parity — the table that decides this

| Runtime | Minimum version | `noul` | `choice` | `choice multiple` | `score` | `text`/`number`/`date` | `number`+range (estimate) | `box`/`boxes` | Image | Video | Verdict for Clank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **llama.cpp** | **b11960 / v0.8.0+** | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ (≤4096 tok) | ✅ (needs `ffmpeg` in PATH) | **Usable today for our exact contract.** |
| **ONNX Runtime** | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (per temporal patch) | Full typed answers **via the repo's Python checkpoint code**; q4/q8 CPU builds. |
| **Transformers** | `transformers>=5.19`, `trust_remote_code=True` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Full support. CPU dtype must be **float32** (`mps`/`cuda` use bf16). |
| ggmlc / other | — | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Not offered for d1 (contrast: this was Laya's only GGUF path). |

`[CITED: https://huggingface.co/LiquidAI/d1-2B-GGUF]` `[CITED: https://huggingface.co/LiquidAI/d1-2B-ONNX]` `[CITED: https://huggingface.co/LiquidAI/d1-2B]` `[CITED: https://github.com/ggml-org/llama.cpp/pull/31283]`

**llama.cpp detail, and it matters.**
- Upstream support is **4 days old**: PR #31275 "model : add LiquidAI/d1-2B and LiquidAI/d1-omni-2B" merged 2026-10-27T22:36Z; version bump to 0.8.0 in PR #31309 on 2026-10-28. The PR says: "At this point, support is **partial**. More features (like bounding boxes) will be added in a follow-up PR… Video input support will be added later. Correctness and conformance verified against the reference HF implementations. F32/Q8_0/Q4_0 quantization smoke-tested on Mac and CUDA." `[CITED: https://github.com/ggml-org/llama.cpp/pull/31275]` Video *is* enabled per PR #31283's capability matrix (d1-2B ✅ video, ✗ soundtrack). `[CITED: https://github.com/ggml-org/llama.cpp/pull/31283]`
- Server invocation from the vendor: `llama-server -hf LiquidAI/d1-2B-GGUF:Q8_0 -b 8192 -ub 8192`, then `POST /v1/systemone`. **A question is read in one batch**, so `-b`/`-ub` must cover the state plus media. `[CITED: d1-2B-GGUF card]`
- **Open bugs in `/v1/systemone`, all unfixed as of the dates shown — none are d1-specific, all are the shared endpoint:**
  - #31150 (2026-10-25, open): in **router mode**, 2–4 **concurrent** requests kill the model instance (exit 1, no error line). Sequential is reliable. `[CITED: https://github.com/ggml-org/llama.cpp/issues/31150]`
  - #30482 (2026-10-13, open): if two questions in one request share a prompt prefix, `-np ≥ 2` returns **HTTP 500 "the question and its options must fit in one batch"**; `-np 1` is fine. `[CITED: https://github.com/ggml-org/llama.cpp/issues/30482]`
  - #30064 (2026-10-06, **closed 10-07**): a Q8_0 GGUF's `systemone` **probabilities collapsed toward uniform** where BF16 was correct (macro-bench: a 0.99 BF16 answer became 0.54 at Q8_0). Fixed, but it is a warning that **the probability readout is quantization-sensitive** — verify any confidence gate on your own artefact and quant level. `[CITED: https://github.com/ggml-org/llama.cpp/issues/30064]`
  - The API surface is still churning: open PRs #29832 (model-agnostic `systemone`), #30113 (`--decision-type` fallback), #30710, #30779; open issue #30161 ("Decision 2.0 GGUF and systemone support"). `[CITED: https://api.github.com/search/issues?q=systemone+repo:ggml-org/llama.cpp]`
  - **Operational rule that follows:** run `-np 1` with prefixed-question-free prompts, or use ONNX. Do not put this behind a router.
- **Docs inconsistency to ignore:** the complete-library matrix lists `d1-2B` GGUF as "Coming soon" and ONNX as ✗, while `LiquidAI/d1-2B-GGUF`, `-ONNX` and `-MLX-{4,8,fp16}bit` all exist and the GGUF card documents b11960 support. The matrix is stale. `[CITED: https://docs.liquid.ai/lfm/models/complete-library]` vs `[CITED: https://huggingface.co/api/models?search=d1-2B]`

**ONNX detail.** Six precisions, each pairing a decoder with a vision encoder: `q4` 1.94 GB and `q8` 2.51 GB are labelled **CPU**; `q4f16` 1.95 GB, `q8f16` 2.34 GB, `fp16` 3.90 GB are **GPU** (fp16 activations); `fp32` 7.80 GB is the reference. "q4 and q4f16 keep only the MLP gate and up projections in int4 and the rest in int8." The CUDA EP has **no fp32 GroupQueryAttention kernel**, so the fp32 graphs run sliding-window layers on CPU — do not pick fp32 on GPU. Budget notes: keep each packed row under **8,192 tokens**, or **4,096 with `decoder_q4`**. `[CITED: https://huggingface.co/LiquidAI/d1-2B-ONNX]`

---

## 4. Sizes, memory, licence

| Model | Params | Weights on disk | Runtime RSS | Licence | Notes |
|---|---|---|---|---|---|
| **d1-2B** | 1.92B | GGUF Q4_K_M **1.05 GB**; ONNX q4 **1.94 GB**, q8 **2.51 GB**, fp32 7.80 GB; MLX 4/8/fp16 | **1.85 GB** (Q4_K_M, CPU, third party) | LFM Open v1.0, <$10M revenue | Fits beside Superhot + Monado + MuJoCo in 8 GB. Contrast MiMo-9B 7,346 MiB. |
| **d1-Omni-2B** | 2.17B | GGUF exists; MLX fp16/8bit; ONNX exists | not measured | LFM Open v1.0 | Adds audio; vendor says worse than d1-2B on text, single boxes, calibration. Not needed by us. |
| **d1-omni-600M** | 587M | GGUF Q8_0/BF16/F16 only | not measured | LFM Open v1.0 | 16K ctx, **experimental**, scores 15.95 on Decision Index v0.2.1 vs d1-3B's 48.57 `[VENDOR]`. Weak on every published row (58.7 DecisionBench vs 77.1). |
| **LFM3-VL-2B** | 1.92B (1.73B LM + 164M vision + 25M projector) | GGUF + ONNX + MLX | — | LFM Open v1.0 | The backbone d1-2B is post-trained from. 128K ctx. Generative — the thing we are trying to avoid. |
| **LFM3-1.7B** | 1.73B | GGUF + ONNX + MLX | — | LFM Open v1.0 | **Text-only**, 24 layers (same hybrid layout), **256K** context. Not a vision model. |

`[CITED: https://huggingface.co/LiquidAI/d1-2B-ONNX]` `[CITED: https://huggingface.co/LiquidAI/LFM3-VL-2B]` `[CITED: https://huggingface.co/LiquidAI/LFM3-1.7B]` `[CITED: https://docs.liquid.ai/lfm/models/complete-library]` `[CITED: https://github.com/prometheion/nerve-protocol/issues/178]`

RAM figure for MiniCPM5-2B (3,932 MiB) and MiMo (7,346 MiB) are from `../docs/archive/2026-10-23-clank-state-of-findings.md` §6 — `[MEASURED]`, ours.

---

## 5. Evidence table — vendor vs independent

### 5.1 VENDOR claims (`[VENDOR]`, read from primary cards today)

| Claim | Number | Source |
|---|---|---|
| Decision Index v0.3 public suite | **51.1** (d1-2B) vs 49.0 (d1-Omni-2B) vs 49.0 (d1-3B) vs 18.3 (omni-600M) | d1-2B card |
| DecisionBench (Hanno Labs) v1 | 77.1 | d1-2B card |
| decisionbench.ai v4 | 85.2 | d1-2B card |
| Fast Decisions | 58.8 (d1-3B edges it at 59.3) | d1-2B card |
| JevBench v1.4.1 | 77.9 | d1-2B card |
| ImajevBench v2.0-lite | 75.5 | d1-2B card |
| WorkflowEvals (evals.typesafe.ai) | 81.6 | d1-2B card |
| Typed Decisions | 61.3 (d1-3B 65.0, d1-Omni-2B 63.1) | d1-2B card |
| Own board, >100 public+private benchmarks | 63.2 | d1-2B card |
| Vision average (unweighted mean of 18 rows) | 70.2 | d1-2B card |
| Notable vision rows | MMBench 87.9, CV-Bench 86.1, MMVP 82.7, MME 85.4, BLINK 66.2, HallusionBench 72.1, **CountQA 51.4**, **ERQA 45.8**, **VisualPuzzles 29.9**, **MMMU-Pro 26.4** | d1-2B card |
| Latency | 5.7 ms 1q / 7.6 ms 3q / 12.3 ms 384px image / 16.1 ms box / 24.4 ms boxes / 52 ms 5K-token state / 1,132 states-s on **H100 PCIe**; 3.8 / 5.6 / 8.7 / 11.2 / 16.5 / 44 ms / 1,699 s⁻¹ on **MI325X** | d1-2B card |
| Apple M5 Pro (`mps`) | 18.6 ms 1q, 26.5 ms 3q, 45.2 ms 384px image, 753 ms 5K-token state, 109/s packed | d1-2B card |

Two things to be strict about. **The benchmark names are third-party boards but the numbers were produced by Liquid running them** — that is stronger than a home-grown eval and much stronger than Laya's single 400-case internal benchmark, but it is still vendor-executed and should never be quoted as an independent result. **The "over a hundred benchmarks" board is undisclosed**, so 63.2 is unauditable. All speed figures are "warm calls after `model.compile()` in bf16, median of 20 calls, best of three runs" — a compiled, GPU-resident, single-question regime. Zero of these are CPU.

### 5.2 INDEPENDENT measurements — what exists

| What | Number | Source | Caveat |
|---|---|---|---|
| d1-2B Q4_K_M, CPU, text only | **p50 1.7 s @ 4 threads**, **5.3 s @ 2 threads**, 1.05 GB file, **1.85 GB RAM** | `[CITED: https://github.com/prometheion/nerve-protocol/issues/178]` (2026-10-28, 231 cases) | Single third party, one day old, self-described busy machine ("only the ratios count"), **no images**, and it is their own safety-classification task, not ours. |
| d1-3B Q4_K_M, CPU, text only | p50 5.1 s @ 4 threads / 8.9 s @ 2 threads, 3.40 GB RAM | same | Same caveat. |
| d1-2B vs d1-3B accuracy on their task | 55.4 / 54.5 catch % (d1-2B marginally ahead), 1.5 % false-refusal each, injection 73.3 % each | same | Their task; n=231. Confirms d1-2B ≈ d1-3B, matching the vendor's claim. |
| **Laya full fine-tune, CPU, same harness** | **99.9 % catch / 2.0 % false-refusal at p50 0.71 s @ 8 threads** — *better than any d1* | `[CITED: https://github.com/prometheion/nerve-protocol/issues/141]` | The sharpest result in this report: **a fine-tuned 421M Laya beat zero-shot 2B d1** on a narrow closed-vocabulary task. Parameters are not the axis; task-specific training is. Exactly the Laya report's conclusion, reproduced by a different party. |
| HF community reception | **2 discussions on `d1-2B`, both administrative** (a base-model metadata PR and a bot "Add evaluation results" PR). No user reports, no bug threads, no quantisations by third parties. Downloads: 591 (d1-2B), 1,101 (GGUF). | `[CITED: https://huggingface.co/api/models/LiquidAI/d1-2B/discussions]` `[CITED: https://huggingface.co/api/models?search=d1-2B]` | **Negative result.** Four days old, essentially unexamined by the community. No d1 finetunes, no third-party quants. Compare Laya at one week: 34 quantisations, 57 finetunes. |
| Third-party *integrations* (not evals) | `prometheion/nerve-protocol` (d1-2B became its default), `kevinmichaelchen/effect-coffee-shop` (d1-2B vs frequency table, deferred), `pennel-app/pennel` (wants a Dutch/English d1 benchmark), `madxmike/gallant`, `jundot/omlx` | `[CITED: https://api.github.com/search/issues?q=%22d1-2B%22]` | Signals interest, proves nothing about accuracy. |
| Any independent evaluation of d1 on vision | **None found.** | — | Negative result: see §9. |

### 5.3 The evidence I specifically looked for and did not find

- **No ECE, no Brier, no reliability diagram, no calibration table for d1 anywhere.** The word "calibrated" appears as a claim in the tagline, the blog and the docs; no number backs it. `[VENDOR]` unquantified.
- **No accuracy number on non-photographic imagery.** Every vision row is natural-image VQA or a web-UI benchmark. There is **no game-screenshot, pixel-art, flat-shaded, or synthetic-render benchmark**. Closest proxies: **VisualWebBench 96.2** (web page screenshots — text-heavy, nothing like Superhot) and **VisualPuzzles 29.9** (abstract visual reasoning — *near chance-ish and the worst row on the card*). `[VENDOR]` `[CITED: d1-2B card]`
- **No confidently-wrong disclosure** of the Laya Khmer-0.000-at-0.952 class. Nothing about failure modes, no known-bugs section, no "honest limits" section. d1's card is *less* candid than Laya's was.
- **No CPU latency, vendor or independent, for any image-bearing call.**
- **No independent reproduction of a single vendor number.**

---

## 6. CPU latency — plainly

- **Vendor CPU figures: none.** Every speed table is H100, MI325X and M5 Pro (`mps`, i.e. Metal, not CPU). The d1-3B blog's edge table (Jetson Orin Nano 50 ms) is accelerator-only. `[CITED: d1-2B card]` `[CITED: https://www.liquid.ai/blog/open-d1]`
- **Independent CPU figures: exactly one cluster, text-only, from one third party, one day old.** 1.7 s p50 @ 4 threads for a Q4_K_M text call (§5.2).
- **Independent CPU figure for an image call: none existed at writing — measured since, see §6b, which confirms this estimate was the right order of magnitude at high detail and too pessimistic at low detail.** `[ASSUMPTION]` Estimate: a 4096-token picture at a plausible 60–150 tok/s CPU prefill for a 1.9B Q4 model is **27–68 s of prefill**, i.e. *at or worse than* MiniCPM5-2B's 20 s p50. This model has *no KV cache and no decode loop*, so a call is one prefill and nothing else — which is the best possible shape for CPU — but a 4096-token prefill is still a 4096-token prefill. If the image path is the binding cost, d1 buys us latency *only* on the text/digest path, where it is 1.7 s.
- Mitigations that would need measuring, not assuming: a smaller `-ub` reduces the per-picture budget proportionally (fewer tokens, less detail); a 384 px frame is the regime the vendor benchmarks; `decoder_q4` caps rows at 4096 tokens.

---

## 6b. CPU latency measured on this machine — 2026-10-28 `[MEASURED]`

Taken after the desk study, on the same box (`/v1/systemone`, `POST`). This closes §6 and §9.1, which were the file's most decision-relevant gap.

**Configuration.** `llama-server` (llama.cpp upstream, commit `8ca4060`, the build that first carries `/v1/systemone`), model **`d1-Omni-2B-Q8_0.gguf`** + `mmproj-d1-Omni-2B-Q8_0.gguf`, **CPU only**, `-t 12 -tb 12 -b 16384 -ub 16384 -ctk q8_0 -ctv q8_0 --flash-attn on`, port 11434. Note these are *not* the recommended settings: this is the **Omni** model (vendor says d1-2B is better calibrated and stronger on text), at **Q8_0** rather than Q4_K_M, with a **16K `-ub`** that grants each picture an 8192-token budget instead of 4096. Two questions per call (`noul` + `choice`).

| Condition | Input tokens | Latency | Notes |
|---|---:|---:|---|
| text-only state, warm | 111 | **5.45 s / 5.61 s** | identical probabilities both runs |
| image 160×100 | 116 | **7.68 s** | 116 is the floor — same token count as 320 px |
| image 320×200 | 116 | **6.95 s** | floor again; resolution did not change the token count |
| image 640×400 | 591 | **38.90 s / 35.89 s** | usable detail costs 5× the tokens |

**What the numbers say.**

- **The vision encode is the whole cost.** Marginal prefill is roughly **15 tokens/s** on the image path — 591 tokens costs ~31 s more than the 111-token text call, while 5 extra tokens at the floor cost ~1.5 s. Cost tracks *image tokens*, not pixels, which is why 160 px and 320 px are indistinguishable and 640 px is catastrophic.
- **At usable detail, d1 is slower than the model it was meant to replace.** 36–39 s against MiniCPM5-2B's **20 s p50**. The pixel path does not win on this CPU.
- **The text path does win, and by a lot.** 5.5 s here on Q8_0/2.2B/12 threads, versus a third party's **1.7 s** on Q4_K_M/d1-2B/4 threads (§5.2). The recommended model and quant should land in the **~1.5–3 s** range on this box — 7–13× faster than MiniCPM5-2B, and inside the window where a think no longer needs a clean stop.
- **Answers were deterministic** (byte-identical probabilities across repeated calls), as a zero-token readout must be. On a tutorial frame it answered `noul` enemy **0.07–0.08** and `choice` **none** at every resolution — consistent, and *unverified*, because the frame has not been hand-labelled.
- **A 404 trap, recorded so nobody repeats it:** `/v1/systemone` is registered as **POST only**; a `GET` returns the generic 404 and looks exactly like "this build does not support d1". It did cost one wrong diagnosis here.

**What this closes.** §8's four ordered requirements now have their third one answered: *image call fast enough on CPU* — **no, not at useful detail**. Requirements (1) competence on the digest and (2) calibration remain the only live questions, and both are cheap: ~100 digest-only calls at ~5 s is under ten minutes of compute. That is the test to run, and it is the same conclusion §0 reached from the desk study alone: **d1 is a decision model over a digest, not a VLM over frames.**

**Caveat on the settings.** `-ub 8192` would halve the per-picture token budget (§2), and d1-2B Q4_K_M is both smaller and better calibrated. A re-run under the recommended configuration would move the image number down somewhat, but the 5× token penalty for usable detail is structural, not a flag.

## 7. d1 vs Laya — what is structurally different

| Axis | Laya (prior art, deleted) | d1-2B | Material? |
|---|---|---|---|
| Backbone | ModernBERT-large **395M, bidirectional encoder**, English root checkpoint 512 ctx | **LFM3 1.73B decoder-only**, hybrid 13 conv + 11 attention layers, **128K ctx** | **Yes.** 4.5× the parameters in a modern hybrid backbone; 256× the context. |
| Head | +26M option-marker scorer, 2 head layers, act/escalate head → 421M | Answer-slot readout over logits at designated slots; total params identical to the backbone (1.92B), so **no separate head budget** `[ASSUMPTION]` (param counts match exactly; whether the LM head is reused or replaced is not stated) | Yes, qualitatively simpler. |
| Vision | **None.** Text-only. | 164M vision encoder + 25M projector; images, video at 2 fps | **Decisive for us.** Laya could never see a Superhot frame; only a hand-written text adapter could. |
| Answer types | `choice` / `score` / `noul` | `noul` / `choice` / `choice multiple` / `score` / `text` / `number` / `date` / **estimate with p10–p90 interval** / `box` / `boxes` | Yes — d1 answers *where*, Laya could not. |
| Calibration apparatus | Released temperatures; independent run still got **Brier 0.565, worst of four**; vendor's own ECE table self-contradictory; **0.000 on Khmer at 0.952 confidence** | `found < 0.5` gating, `confidence` + full distribution on every categorical answer, p10/p90 on estimates — **but no published calibration metric at all** | **Mixed, and this is the risk.** d1 ships better *machinery* for uncertainty and worse *evidence* that the machinery is honest. |
| Accuracy evidence, zero-shot | 56.1 % WANLI (8th of 9), 1.25 kills vs random 1.88, 6/12 direction checks — **independently measured, and bad** | 51.1–85.2 across seven named third-party boards, 63.2 on an undisclosed hundred-benchmark board — **vendor-executed, no independent reproduction** | **Yes on volume, no on independence.** d1 has much more evidence and none of it is third-party. |
| Training disclosure | No paper, one dev.to post; vendor admitted base checkpoints near chance | No paper, but a detailed blog: weight averaging of two backbones, per-seed/data-mixture fine-tunes merged, "training on long inputs, shuffling answer options, and fixing shortcuts in the data made a bigger difference than more advanced techniques" | Yes — "fixing shortcuts in the data" is the specific failure Laya died of. |
| Tooling | GGUF re-uploads were **ggmlc, explicitly not llama.cpp**; ONNX only third-party | **llama.cpp upstream** (`ggml-org/llama.cpp` #31275, v0.8.0), official ONNX, official MLX, first-party docs with runnable curl | **Decisive.** Laya was unavailable to a llama.cpp-only build with zero workaround. d1 is available today with a release bump. |
| Ecosystem | 34 quants, 57 finetunes, NixOS module, PyPI package | Official exports only; **zero third-party quants or finetunes at 4 days old** | **No, worse.** d1 is 4 days old; assume breakage. |
| Latency | 15.7 ms in-process (DGX Spark GPU); 39.5 ms T4 vendor; 421M on CPU ≈ 0.71 s @ 8 threads (third-party, fine-tuned) | 5.7 ms H100 (compiled, bf16); **1.7 s CPU @ 4 threads, text** | On CPU, d1 is ~2.4× slower than Laya and ~12× faster than MiniCPM5-2B. |

**Bottom line on the comparison.** d1 is a *different class of artefact*: a first-party multimodal model with first-party runtimes and a public eval posture, versus a hobbyist open reproduction of a closed API. The structural risk that killed Laya — an encoder too weak to turn a structured scene into a correct action, with confidence that cannot detect its own failure — is **not disproven for d1**. It is simply untested. The one piece of evidence pointing the wrong way is §5.2's fine-tuned Laya row: on a narrow closed-vocabulary task, a *trained* 421M beat *zero-shot* 2B d1, which says the axis that decides our outcome is task-specific training, not vendor benchmark rank.

---

## 8. What this means for Clank

**Does d1 plausibly beat a majority-class baseline on Superhot frames?** `[ASSUMPTION]` **Plausibly, on the digest; unknowably, on pixels.** The digest is exactly d1's designed input: a short text or JSON state with named questions and 3–16 declared options. `hold` will be the majority class and the model will likely over-predict it, which is the same degeneracy as LFM2.5-3B's 73 % `navigate_to` — but visible: we get `probabilities` over all skills, so "collapsed" is a *measurement* (`p(hold) ≈ 1.0` on frames where it should not be) rather than an inference from token counts. The pixel path has no evidence for or against, because no benchmark in existence covers flat-shaded game frames, and the closest abstract-reasoning row (VisualPuzzles 29.9) is bad.

**The four things that would have to be true, in order.**
1. **Competence on the digest** — beats majority class on held-out frames with the closed skill vocabulary. *Cheap to test, 1.7 s per call, no vision.*
2. **Calibration is usable** — `confidence` separates right from wrong, so `confidence < τ → hold` is a real gate. *Not published; must be measured on our frames.*
3. **Image call is fast enough on CPU** — the 4096-token picture budget must not cost 30 s. *Completely unmeasured; the number that decides whether d1 replaces MiniCPM or merely complements it.*
4. **Endpoint stability** — `-np 1`, no router, pinned llama.cpp ≥ b11960. *Known workaround, known cost.*

**The cheapest test that would settle (1) and (2), ~1 hour, no GPU, no game:**

1. Take **60–100 grim frames** already captured in `spike/c0` (or any Superhot footage); write the ≤10-line digest per frame **by hand** from the picture (skill set, `supported` bits, residuals, drift, last outcome). Hand-writing the digest is the point: it removes the vision encoder and the tracker from the experiment.
2. Label each frame with the action the *scripted* E3 commander would issue (`hold`, `look`+direction, `reach`+side, `punch`+side, `trigger`). This is the ground truth a scripted policy already achieves — it is the honest bar, per the Laya report's "a model that can be below a scripted policy is not a gate".
3. Ask d1-2B Q4_K_M three questions in **one** call: `choice` over the 5 skills; `choice` over `left`/`right`; `noul` "is a target visible". Record `probabilities` and `confidence`.
4. Report, against the **majority-class baseline** and the **scripted label**: accuracy, confusion matrix, per-class recall, and the calibration of `confidence` (bucket accuracy vs mean confidence — a 5-bucket reliability table is enough). ~100 calls at 1.7 s ≈ 3 minutes of compute.
5. Only if (1) passes: repeat with `media=[frame]`, `-ub 8192`, and time the same 100 calls. That single number — **image-call p50 on our CPU** — is the fact the whole decision turns on and nobody has.

Practical notes for whoever runs it: do **not** reuse port 11434 (the user is rebuilding it); run a separate `llama-server -np 1` on a free port, or go straight to ONNX `q4` + the repo's Python checkpoint code, which also gives `box` and `multiple` if the test wants them. Pin llama.cpp ≥ b11960. Do not gate on confidence until you have reproduced the probability readout at your quant level (§3, #30064).

---

## 9. Still unknown

1. ~~**CPU latency with an image.** No number exists.~~ **Answered 2026-10-28 — §6b.** 7.0 s at 116 image tokens, 36–39 s at 591, CPU-only, on the machine this project runs on.
2. **Whether d1's `confidence` is calibrated.** No ECE, Brier or reliability data published by anyone. The card's instruction to calibrate your own threshold implies "no".
3. **d1 on game frames specifically.** No pixel-art / flat-shaded / synthetic-render evaluation exists, vendor or independent. VisualWebBench 96.2 is web pages; VisualPuzzles 29.9 is the only abstract-reasoning row.
4. **Whether the `box` output is any good.** Vendor benchmarks boxes for *latency* (16.1 ms) and never for **localisation accuracy**; no mAP/IoU appears anywhere on the card. `[VENDOR]`
5. **Whether the 4096-token picture budget is enough for a Superhot frame at our capture resolution** — the docs give the budget, not the preprocessing geometry for this checkpoint (`processor_config.json` not read). Wasting 4096 tokens on a flat-shaded scene that a tracker could summarise in 20 is a plausible own-goal.
6. **Independent reproduction of any vendor number.** None exists at 4 days old. HF has 2 administrative discussions and zero user reports.
7. **Endpoint stability under our usage.** The open bugs are on other decision models; nobody has reported on d1-2B specifically because it is 4 days old. The concurrency crash (#31150) and prefix-collision 500 (#30482) are in shared code and should be assumed live.
8. **`Sparse` source health.** `www.liquid.ai/blog/open-d1` covers **d1-3B and d1-omni-600M (Oct 7)**, not d1-2B — the d1-2B card links to a post that does not mention d1-2B, and its citation block is the d1-3B post. There is no dedicated d1-2B announcement post on the blog index. `[CITED: https://www.liquid.ai/blog]`
9. **Sources I could not reach (recorded as negative results, not guessed):** `html.duckduckgo.com` returned a bot challenge (67 "anomaly" markers); `reddit.com/search.json` returned non-JSON. So I have **no** Reddit/HN/forum sentiment for d1 — absence of evidence, not evidence of absence.

## 10. Sources read

Primary: `huggingface.co/LiquidAI/d1-2B`, `/d1-2B-GGUF`, `/d1-2B-ONNX`, `/d1-Omni-2B`, `/LFM3-VL-2B`, `/LFM3-1.7B` (raw READMEs + `LICENSE`), `docs.liquid.ai/lfm/models/decision-models`, `/lfm/models/complete-library`, `/guides/decision-model-guide`, `/examples/laptop-examples/build-a-game-with-decision-models`, `/lfm/key-concepts/vision-capabilities`, `/deployment/on-device/onnx`, `liquid.ai/blog/open-d1`, `liquid.ai/blog`. GitHub: `ggml-org/llama.cpp` PRs #31275, #31283, #31309, #29818, #29321; issues #31150, #30482, #30064, #30161. Third-party: `prometheion/nerve-protocol` issues #178 and #141, `kevinmichaelchen/effect-coffee-shop` #123/#126, `pennel-app/pennel` #53. Hugging Face discussion and model-search APIs. Prior art: `../stubborn-gamer/research/squads/wave78-laya-capabilities.md`. Local context: `../docs/archive/2026-10-23-clank-state-of-findings.md` §6–§7, `../docs/2026-10-23-clank-design.md` §9, `research/system1-game-agents.md` §4.2/§7.
