# FAILFORGE Visual Benchmark Card

## Benchmark name

**FAILFORGE-Visual Synthetic Desktop**

## Purpose

This benchmark evaluates controlled failure modes in visual computer-use agents:
visual grounding, dynamic UI recovery, verification, shortcut exploitation, OOD robustness,
and long-horizon compounding error.

It is designed for mechanism analysis, not as a replacement for OSWorld, WeaveBench,
BrowserGym, or MiniWoB.

## Observation space

- 32x32 RGB screenshot-like visual state
- 4x4 spatial interaction grid
- eight visual widget types
- multimodal goal token
- optional pop-up/verification/shortcut states

## Action space

- 16 spatial click actions
- VERIFY
- DISMISS

## Task families

1. Normal visual grounding
2. Layout shift
3. Visual noise
4. Occlusion
5. Theme/grayscale shift
6. Combined OOD
7. Pop-up recovery
8. Verification required
9. Shortcut trap
10. Long-horizon composition

## Core metrics

- Overall action accuracy
- Combined OOD accuracy
- Checkpoint accuracy
- Exact completion
- ≥80% completion
- Genuine accuracy
- Shortcut rate
- Verifier score
- Failure recurrence

## Current executed result

Main visual OOD comparison:

- UniformFactor-RL: **43.30%**
- FAILFORGE-HardMine: **49.19%**
- Difference: **+5.89 pp**
- Positive seeds: **4/4**
- Paired p-value: **0.0334**

Verifier stress test:

- Full verifier shortcut rate: **0.1%**
- No-verifier shortcut rate: **27.4%**

## Intended use

Use this benchmark to test:
- whether a method improves visual robustness under controlled OOD shifts;
- whether a method resists shortcut exploitation;
- whether local action competence transfers to long-horizon completion;
- whether failure-conditioned sampling helps beyond uniform synthetic sampling.

## Out-of-scope use

Do not use this benchmark to claim:
- OSWorld SOTA;
- WeaveBench SOTA;
- real browser agent superiority;
- production readiness;
- general CUA performance.

## Known limitations

- Simplified synthetic UI.
- Lightweight visual policy.
- Four main visual seeds.
- No real browser DOM.
- No natural-language web pages.
- Does not include real files, spreadsheets, or application APIs.
- Exact completion remains weak at long horizons.

## Dataset generation

All observations are procedurally generated. No personal data, scraped user data,
or proprietary model outputs are used.

## Safety considerations

The benchmark intentionally includes shortcut traps. These are used to test whether
agents exploit evaluator loopholes. Evaluation should always distinguish genuine
success from shortcut success.

## Reproducibility

Raw CSVs, summaries, figures, scripts, and seed protocols are included in the final package.
