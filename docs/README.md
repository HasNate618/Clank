# docs

Where each document stands, so a reader can tell live from historical without opening anything.

**The rule this file exists for: when you change behaviour, change the document that describes it
in the same commit.** If you are not going to keep it true, archive it and say so.

## Current

These describe code that runs today. A claim here that disagrees with the code is a bug.

| Doc | What it is |
| --- | --- |
| `2026-10-23-clank-design.md` | The command contract (`BodyCommand` / `BodyState`), the authority mask, the executor, the game-side findings. §6 is still the authority `clank/body.py` cites; §18's controller survey is history. |
| `superpowers/specs/2026-10-31-brain.md` | The brain: one loop, one act, the tools it may call. |
| `superpowers/specs/2026-11-13-session-brain.md` | The live chooser: what it hears, what it remembers for the whole run, and what it writes down. |
| `superpowers/specs/2026-11-11-ardy-reach-design.md` | Constraint into a motion prior into SONIC, and what it does not do (the look, the out-of-reach case). |
| `superpowers/specs/2026-11-12-runs-design.md` | Stack versus run: what costs a boot and what is repeatable. |
| `superpowers/specs/2026-11-12-recordings-design.md` | The video pipeline and its evidence-not-dependency contract. |
| `superpowers/specs/2026-11-13-console-and-recordings-ui-design.md` | The console page and the recordings page. |
| `setup.md` | Install list, `config.toml` keys, ports, controls, safety. The README's five commands get you a run; this is the rest. |
| `superpowers/specs/2026-11-13-readme-design.md` | Why the README is shaped the way it is, and the figure plan for the GIFs still to come. |
| `media/title-card.svg` | The image at the top of the README. |

## Archived

`docs/archive/` holds what was superseded. Every file there opens with a banner saying what replaced
it. Nothing in it is a claim about current behaviour, and its numbers are dated to its own week.

| Path | What it was |
| --- | --- |
| `archive/specs/2026-11-10-brain-tools.md` | How an act was chosen from a still, and why the sense tool was not trusted to be chosen. Superseded by the session spec. |
| `archive/2026-10-11-feasibility-study.md` | The feasibility study, written when no code existed here. Prior art, not a plan. |
| `archive/2026-10-23-clank-state-of-findings.md` | The Superhot-era ground-truth file. |
| `archive/2026-10-23-vr-bot-design-builder-check.md` | The open-questions audit taken before implementation. |
| `archive/plans/` | 13 implementation plans, all carried out. Kept for the task breakdown and the gates. |
| `archive/specs/` | 4 designs that were superseded or never accepted. |
| `archive/proposals/` | Proposed diffs, never applied. |

## Not design documents

`research/` is the measurement record: per-spike results with their numbers, their conditions, and
their negative results. It is not rewritten when the project changes target — the date on each file
is the era it measures. `research/motion-generators.md` and `research/g1-scale.md` are where the
reach and the scale numbers are argued.
