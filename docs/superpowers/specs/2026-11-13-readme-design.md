# README rewrite — design

**Date:** 2026-11-13. **Status:** draft for approval.
**Replaces:** the 2026-11-04-era `README.md` (347 lines), which describes Superhot, 323 tests and an
architecture with no eyes.

## Intent

A hiring reviewer gives this repo 30–60 seconds, then maybe four minutes if it survives the first
screen. The README must answer, in that order: what this is, that it runs, what was hard, and which
part is mine. The story is the **closed loop**: game pixels and speech in, one act chosen, a
generated motion, a hand closed on a real object, all measured.

Audience: a technical hiring manager or engineer, on a portfolio pass. Not a user installing a
library; not a research reader (they get `research/`).

## Goals

- First screen carries the summary, the proof, the hard problem, and the one number.
- Architecture is the argument, not an appendix: the round trip drawn once, then each hop with its
  cost and its failure.
- Eight measured rows, each linking the document that holds the conditions.
- Limitations specific enough to be quoted in an interview.
- Quick start that works from a fresh clone.
- Passes the portfolio checklist: no placeholders, no badge wall, no adjective without a number.

## Non-goals

- Not a setup manual: ports, PAM, VNC, config keys and the control table move to `docs/setup.md`.
- Not a build log: dated session narrative stays in `AGENTS.md` and `research/`.
- Not the video yet. The still ships now; the video replaces it when it exists, and the still becomes
  its poster frame and backup. No "coming soon" line, no placeholder panel.

## Figures

Every figure is additive. A section must read correctly with its figure removed, so a figure that is
not ready changes nothing, and adding one later never requires rewriting the section. No mock panels,
no greyed-out "TBD" boxes, and every caption names the run, the date, and what the reader is looking at.

| # | Where | Panels | State |
|---|---|---|---|
| F1 hero | top | game view, robot view, diary: `190131-679de2`, the card grab | have — ship now; the video replaces it later |
| F2 how it sees | Architecture / Seeing | the same frame four ways: colour, depth map, depth-faded, and the still the model was given | buildable now from `research/evidence/`; GIF later |
| F3 how it moves | Architecture / Motion | target, generated clip, achieved glove | the `ardy-live-grab` pair exists; needs the achieved panel |
| F4 the body | Stack / body | MuJoCo view beside the game's robot tile | `recordings-frame.png` is usable; optional |

GIF rules, when they come: 10–20 s, 800–1000 px wide, 10–15 fps, palette-optimised, under a few MB,
with alt text. Each GIF replaces its figure's panels rather than adding to the page.

## Anti-goals

Anything that would fail the "how do you know?" test: a number without a source, a claim of play,
a claim to have trained GEAR-SONIC or ARDY-G1, "production-ready", a badge, an unchecked roadmap.

## Constraints from the code, not from the old README

- Target is **Job Simulator**, not Superhot. `config.toml`, `spike/wbc_game/`, `clank/run.py`.
- **1378 tests** (measured 2026-11-13), stdlib only, no network, no GPU.
- Perception is the **Vulkan layer** (`spike/e5`): 480×270 depth + colour, ~13 ms.
- The loop is `spike/wbc_game/auto.py`: listen → transcribe → think → locate → `reach_ardy` →
  `pose_relay`. The reach is generated (ARDY-G1) and tracked by SONIC.
- NixOS, one RTX 4060 8 GB, a Monado `remote` driver, Steam + Proton.

## Approaches considered

**A. The loop at the top, exploded beneath.** One diagram of the round trip; each hop a section with
its number and its failure. **Picked.** The AI contribution and the systems work are legible in the
same glance, which is what the reviewer is asked to judge.

**B. Exploded stack, bottom-up.** Five chapters (Vulkan → Wine/Monado → ARDY/SONIC → relay →
orchestration). Strongest breadth signal; the closed loop becomes something the reader assembles,
and the headline was the loop.

**C. Claim / mechanism / measurement cards.** Twelve three-line cards. Most scannable, no narrative,
and the AI loop becomes one card among twelve.

## Decisions taken in conversation

1. **Audience:** hiring / portfolio review.
2. **Headline:** the loop closes. The number is the grasp, not the walk.
3. **Hero image:** `2026-11-12_22-18-57.png` — run `190131-679de2`, 1:22, game view, robot view, and
   the diary showing `plan: grab(card)` → `locate(card)` → `generated reach`. Ship at 1600 px wide.
4. **Results:** eight rows in the README, each linking its research document.
5. **Role:** solo; GEAR-SONIC and ARDY-G1 run unmodified; the coding agent and its model named in one
   clause (`pi`, DeepSeek V4.1 Flash) because a cheap worker model strengthens the claim that the
   pipeline design is the human contribution.
6. **Archive:** done in `bf918b6`; `docs/README.md` is now the document map.

## Structure

| Section | Budget | Contents |
|---|---|---|
| Title card + name | 6 | existing `docs/media/title-card.svg` |
| Summary | 6 | the model first: one 2B VLM hears, decides, finds and commands; the round trip in one line; the question it answers |
| Hero still | 4 | the run above, captioned with what the picture cannot say (run id, nine model calls, the diary's last line and that it is wrong) |
| The brain: one 2B model, every call in the loop | 60 | subsections **sees / hears / decides / finds / commands**; the live prompt and the JSON it answers; why the verb is read out of the goal in code; the head cannot hold a pose, so the pixel is re-derived; identity decided in code; three measured surprises; the diary |
| Three models, and nothing else | 8 | the 2B model is the brain, SONIC and ARDY-G1 are the hands; "none trained here" |
| Role | 4 | solo; `pi` + DeepSeek V4.1 Flash in one clause; the perception chain and the loop are the author's |
| Architecture | 12 | mermaid round trip, one paragraph on the single simulation, slot for F2 |
| The brain | 40 | (moved above; this row is retained only in the archive) |
| Decisions and tradeoffs | 22 | four: inject at the layer, generate rather than hand-write the reference, one body relayed rather than two clients, a crop per turn rather than a bigger locator |
| Results | 14 | 8 rows, source column links |
| Limitations | 12 | nobody has played it; clip-anchored; 44.2 Hz against 50; torso does not turn; the clasp margin; the model and its tools; the unverified controller frame |
| Quick start | 14 | Prerequisites (NixOS, Steam, your copy of the game, Monado with `remote`, an 8 GB card), four commands, what you should see |
| Tests | 5 | command, count, what is not covered |
| Where things live | 10 | repo map, 8 rows |
| Setup and controls | 2 | pointer to `docs/setup.md` |
| License and credits | 5 | MIT; NVIDIA and Liquid AI model licences; the game |

Target: **~180 lines, under 16 KB**, first four sections inside one screen at 1280×800.

## What moves out

- `BodyCommand` field table → link `docs/2026-10-23-clank-design.md` §6.
- The 25-row measurement table → stays where it already is, in `research/`; the README keeps eight.
- Ports, VNC/PAM, `config.toml` keys, the full control table → new `docs/setup.md` (~50 lines),
  indexed in `docs/README.md`.
- Superhot-era prose, the 323 number, the PNG-preview dashboard description → gone.

## Left to the builder

- The hero sentence's exact wording; every number re-checked against its research document.
- Whether the second worked example in the role clause is the LOAD-pass injection decision or
  something else — only the operator can confirm which calls were his.
- F2's panels must be the same frame or the same moment. If the existing captures do not align,
  drop the figure rather than compose four pictures taken at four times and caption them as one.
- Image downscale and check that the caption's run id matches `run.json`.
- Mermaid render check on GitHub, and first-screen height on a 1280×800 viewport.
- A humanizer pass over the prose before commit.

## Verification

1. `python3 -m unittest discover -s tests -t .` (unchanged by a README, run anyway).
2. Every relative link and every backticked path resolves.
3. Every number in the README appears in the cited research document with the same value.
4. Quick start followed verbatim from a fresh clone into `/tmp`.
5. Screenshot the rendered README at 1280×800: the first four sections fit.
6. Guide checklist for the About panel (description, topics, pin order, social preview) — the
   operator's to fill, listed at the end of the work.
