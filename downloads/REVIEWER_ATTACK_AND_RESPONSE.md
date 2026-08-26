# FAILFORGE Reviewer Attack Test and Response Plan

This document plays the role of a harsh NeurIPS/ICLR-style reviewer and pre-writes the response.

## Verdict risk

Current project status: **strong workshop / early conference paper**, not yet a full public-benchmark SOTA paper.

The strongest honest positioning is:

> A controlled experimental study of failure-conditioned synthetic RL for visual computer-use agents, with a public-benchmark bridge implemented but not executed.

## Major concern 1 — Synthetic benchmark may be too simple

**Attack:** The visual benchmark is not real computer use. It uses 32x32 synthetic screenshots and simplified actions.

**Response:** Correct. The paper does not claim real-browser SOTA. The benchmark is designed for mechanism isolation: verifier ablation, shortcut stress, hard-example mining, and long-horizon compounding error. The BrowserGym/MiniWoB bridge is included for the next external validation step.

**Fix in paper:** Put this limitation in Abstract/Limitations, not only appendix.

## Major concern 2 — Small seed count

**Attack:** Main visual OOD result uses 4 seeds.

**Response:** We report matched seed comparisons, effect size, and p-value, but avoid overclaiming. The claim is bounded: evidence that example-level failure mining helps in this controlled benchmark.

**Fix:** Add confidence intervals and mark as controlled evidence.

## Major concern 3 — Uniform synthetic is very strong

**Attack:** FAILFORGE only improves over uniform by +5.89 pp.

**Response:** That is the point: uniform synthetic is a serious baseline. Naïve failure curricula failed. The contribution is not "failure always wins"; it is that failure conditioning must operate at the example level and be mixed with exploration/replay.

## Major concern 4 — Failure-only did not work

**Attack:** Early failure-only variants failed.

**Response:** Retained as negative result. This improves the paper: it shows why closed-loop distribution design matters.

## Major concern 5 — No public benchmark score

**Attack:** Without MiniWoB/OSWorld/WeaveBench scores, the paper is not SOTA.

**Response:** Agree. We do not claim public benchmark SOTA. The contribution is a controlled mechanism study plus a benchmark-ready bridge.

## Major concern 6 — Visual warm start reduces RL purity

**Attack:** Warm-starting makes it less like pure RL.

**Response:** Pure sparse visual RL collapsed. Warm-starting reflects practical CUA training, where perception must be established before long-horizon RL can be meaningful.

## Major concern 7 — Verifier may encode the answer

**Attack:** The verifier could leak ground truth.

**Response:** The verifier is not used to choose actions at inference. It corrects training reward and audits trajectories. The no-verifier ablation shows why this matters.

## Major concern 8 — Hard mining might just select easier examples

**Attack:** The hard-mining procedure may accidentally change distribution difficulty.

**Response:** UniformFactor-RL trains on the same factor pool. FAILFORGE changes selection based on current policy difficulty. Future work should add matched distribution controls.

## Major concern 9 — Long-horizon exact completion is poor

**Attack:** Exact completion collapses.

**Response:** Yes. This is a finding, not a hidden flaw. Local competence does not solve long-horizon compounding error.

## Major concern 10 — Does this scale to frontier models?

**Attack:** Lightweight agents may not represent Claude/GPT-style CUAs.

**Response:** The benchmark tests mechanisms, not frontier-model absolute performance. The BrowserGym adapter is the bridge to external agents.

## Required paper edits before submission

- [x] Do not claim SOTA on public CUA benchmarks.
- [x] Put negative results in main text.
- [x] Separate controlled evidence from external validation.
- [x] Report verifier pathology prominently.
- [x] Include benchmark card.
- [x] Include reproducibility statement.
- [x] State BrowserGym limitation clearly.
- [ ] Run MiniWoB public benchmark in a networked/browser-enabled environment.
- [ ] Increase main visual seeds from 4 to 8 or 12 if compute allows.
- [ ] Add stronger visual policy or pretrained encoder.
- [ ] Add memory/planning module for long-horizon exact completion.
