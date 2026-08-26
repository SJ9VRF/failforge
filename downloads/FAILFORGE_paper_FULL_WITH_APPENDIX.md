# FAILFORGE: Failure-Conditioned Synthetic Reinforcement Learning for Computer-Use Agents

**Aura Yavary**  
UC Davis  
Draft version: 2026-08-26

## Abstract

Computer-use agents increasingly operate through visual interfaces, browser environments, filesystems, forms, dialogs, and multi-step software workflows. Yet current agents remain brittle under long-horizon interaction, hidden state, dynamic interface changes, and trajectory-level verification requirements. This paper studies a simple hypothesis: failures should not merely score an agent; they should determine the computer worlds the agent experiences next.

We introduce **FAILFORGE**, a failure-conditioned synthetic reinforcement-learning framework for computer-use agents. FAILFORGE verifies trajectories, diagnoses failures into interpretable categories, synthesizes or selects counterfactual environments around those failures, and mixes those hard examples with broad exploration and competence replay. We implement and evaluate the method in two controlled settings: a symbolic computer-use MDP for mechanism analysis and a screenshot-based visual computer-use benchmark with desktop-like widgets, pop-ups, verification states, shortcut traps, and visual OOD shifts.

Across 4 visual seeds, example-level failure mining improves combined visual OOD accuracy from **43.3%** for UniformFactor-RL to **49.2%**, a **+5.89 percentage-point** matched improvement with 4/4 positive seeds, paired p=0.0334, and Cohen's dz=1.869. Trajectory-aware verification is critical: in a visual shortcut stress test, removing verifier correction increases shortcut selection from **0.1%** to **27.4%** and reduces genuine task accuracy from **80.9%** to **57.8%**. Long-horizon experiments show that local checkpoint accuracy can remain stable while exact task completion collapses, isolating compounding error as a distinct failure mechanism.

The results support a bounded claim: failure-conditioned training can improve visual computer-use robustness when it operates at the example/environment level, but failure-only curricula are insufficient. Robust computer-use RL requires a calibrated mixture of failure frontier mining, broad exploration, replay, and trajectory-aware verification.

---

## 1. Introduction

The central challenge in computer use is no longer only whether an agent can locate a button or type into a field. Real workflows are long, stateful, partially observed, and filled with dynamic interface events. An agent may correctly execute many local actions while still failing the task because it lost a constraint, skipped verification, relied on stale state, or exploited the evaluator.

This project asks whether synthetic reinforcement learning can be made more useful by changing what is synthesized. Instead of generating more random tasks, we generate or select tasks from the agent's own failure frontier.

The core loop is:

\[
\textbf{Act} \rightarrow \textbf{Verify} \rightarrow \textbf{Diagnose} \rightarrow \textbf{Forge} \rightarrow \textbf{Mix} \rightarrow \textbf{Train} \rightarrow \textbf{Replay}.
\]

This matters because public computer-use benchmarks increasingly reveal long-horizon and verification gaps. OSWorld2.0 reports professional-scale workflows with hundreds of tool calls and low completion rates for frontier agents. WeaveBench similarly emphasizes trajectory-aware judging and shortcut detection. Synthetic Computers at Scale shows that synthetic computer environments can be a plausible substrate for long-horizon agent learning. FAILFORGE connects these themes by turning verified failures into the next training distribution.

### Contributions

1. **Failure-conditioned synthetic RL.** We formulate a closed-loop training distribution in which recent trajectory failures shape future computer-use environments.
2. **Trajectory-aware verification.** We separate terminal reward from verified execution quality and explicitly penalize shortcut behavior.
3. **Two executable benchmarks.** We evaluate on a symbolic computer-use MDP and a screenshot-based visual computer-use benchmark.
4. **Negative results.** We show that failure-only curricula and task-category reweighting are insufficient; effective failure conditioning must operate at the example/environment level and retain exploration.
5. **Public-benchmark bridge.** We provide a BrowserGym/MiniWoB-ready adapter and protocol without fabricating public benchmark scores.

---

## 2. Related Work

### Computer-use benchmarks

OSWorld2.0 evaluates long-horizon real-world computer-use workflows and highlights failures such as missing hidden state, losing constraints, and skipping verification. WeaveBench evaluates hybrid GUI/CLI workflows and introduces trajectory-aware judging with shortcut detection. These benchmarks motivate our focus on long-horizon robustness and trajectory verification.

### Synthetic computer environments

Synthetic Computers at Scale creates user-specific synthetic computers and long-horizon simulations for productivity tasks. FAILFORGE differs in that the synthetic distribution is not static: failures alter future environment selection and generation.

### Reinforcement learning from evaluators

Evaluator-driven RL is increasingly important for agents whose success cannot be captured by a simple terminal signal. FAILFORGE uses the verifier both as an auditing tool and as a reward-correction mechanism, but its main novelty is that failure diagnosis also changes future training environments.

### Curriculum learning and hard-example mining

FAILFORGE is related to curriculum learning and hard-example mining, but differs in its computer-use focus and trajectory-level failure taxonomy. Our experiments show that hard examples alone are not enough; they must be mixed with broad exploration and replay.

---

## 3. Method

Let a computer-use environment be

\[
\mathcal{E}=(S,A,T,O,R,\mathcal{G}),
\]

where \(S\) is latent state, \(A\) is an action space such as click/type/verify/dismiss, \(T\) is the transition function, \(O\) is a visual or symbolic observation space, \(R\) is a reward function, and \(\mathcal{G}\) is a goal distribution.

An agent produces a trajectory

\[
	au=(o_1,a_1,r_1,\ldots,o_T,a_T,r_T).
\]

A trajectory verifier computes:

\[
V(	au)=R_{terminal} + \alpha R_{milestone} + \beta R_{verify} - \gamma R_{shortcut} - \delta R_{invalid}.
\]

A failure classifier maps trajectories into a failure vector:

\[
f(	au)=[f_{ground},f_{state},f_{verify},f_{recover},f_{horizon},f_{shortcut}].
\]

The next training distribution is:

\[
p_{train,t}(E)=
\lambda_f(t)p(E\mid f(	au),\theta_t)
+
\lambda_u(t)p_{uniform}(E)
+
\lambda_r(t)p_{replay}(E),
\]

where \(	heta_t\) is the current policy and \(\lambda_f+\lambda_u+\lambda_r=1\).

### Algorithm 1: FAILFORGE

```text
Input: initial policy πθ, environment family E, verifier V, failure classifier C
Initialize replay buffer B and failure memory F

for training round t = 1 ... T:
    Sample environments from mixture:
        ptrain = λf pfrontier(F) + λu puniform + λr preplay(B)

    Roll out πθ and collect trajectories τ
    For each τ:
        Compute verified score V(τ)
        Diagnose failures C(τ)
        Add successful/partial trajectories to replay buffer B
        Update failure memory F

    Forge or select counterfactual environments from high-severity failures
    Update policy πθ using reward corrected by V(τ)
```

The key design choice is that failure conditioning is not an isolated data filter. It is a closed-loop distributional intervention.

---

## 4. Benchmarks

### 4.1 Symbolic computer-use MDP

The symbolic MDP isolates state tracking, verification, recovery, horizon, sparse reward, and shortcut traps. It is useful for mechanism-level ablations because the ground-truth failure type is known.

### 4.2 Visual synthetic desktop benchmark

The visual benchmark renders 32x32 RGB screenshot-like desktop states with a 4x4 click grid, eight widget types, a multimodal goal token, and global actions VERIFY and DISMISS. Evaluation includes:

- IID visual grounding
- layout shift
- visual noise
- occlusion
- grayscale/theme shift
- combined OOD
- pop-up recovery
- verification states
- shortcut traps
- long-horizon composition

Training uses a visual-grounding warm start followed by reward-based fine-tuning. Pure sparse visual RL from scratch collapsed toward easy global actions, so it is retained as a negative baseline rather than used as the main comparison.

### 4.3 BrowserGym/MiniWoB bridge

We implement a BrowserGym/MiniWoB-ready adapter, task discovery, task-family mapping, adaptive sampler, verifier, failure classifier, seed protocol, and logging harness. Because this execution environment lacks network access for installing BrowserGym/MiniWoB and Chromium, we do not report public benchmark scores. The bridge is benchmark-ready but not benchmark-executed.

---

## 5. Experimental Setup

### Methods

- **UniformFactor-RL:** samples visual perturbations uniformly.
- **FAILFORGE-HardMine:** samples a larger candidate pool, estimates current policy difficulty, and trains on low-correct-probability examples mixed with random examples.
- **No-verifier:** removes trajectory-aware reward correction in a shortcut-heavy setting.
- **Hard-ratio ablations:** sweep hard-example mixture fractions.
- **Long-horizon evaluation:** compose tasks across horizons 4, 8, 16, 32, 64, and 128.

### Metrics

- Overall action accuracy
- Combined visual OOD accuracy
- Genuine task accuracy
- Shortcut rate
- Checkpoint accuracy
- Exact completion
- ≥80% completion
- Failure recurrence
- Verifier score

---

## 6. Results

### 6.1 Main visual OOD result

Per-seed combined OOD accuracy:

|   seed |   FAILFORGE-HardMine |   UniformFactor-RL |
|-------:|---------------------:|-------------------:|
|      0 |               0.495  |             0.435  |
|      1 |               0.4835 |             0.3815 |
|      2 |               0.57   |             0.525  |
|      3 |               0.419  |             0.3905 |

Mean FAILFORGE-HardMine accuracy: **49.19%**  
Mean UniformFactor-RL accuracy: **43.30%**  
Matched improvement: **+5.89 pp**  
Positive seeds: **4/4**  
Paired p-value: **0.0334**  
Cohen's dz: **1.869**

Interpretation: example-level failure mining improves visual OOD robustness in this controlled benchmark. The claim is not that FAILFORGE universally dominates uniform synthetic training, but that failure-conditioned selection becomes useful when it targets concrete visual examples rather than only broad task categories.

### 6.2 Verifier and shortcut exploitation

Visual verifier stress test:

| variant     |   genuine_accuracy |   shortcut_rate |
|:------------|-------------------:|----------------:|
| full        |             0.8094 |          0.0006 |
| no_verifier |             0.5781 |          0.2744 |

Removing verifier correction increases shortcut rate from **0.1%** to **27.4%**. This is the clearest safety/evaluation result in the project.

### 6.3 Long-horizon scaling

Long-horizon results:

|   horizon |   checkpoint_accuracy |   exact_completion |   ge80_completion |
|----------:|----------------------:|-------------------:|------------------:|
|         4 |                0.4604 |             0.0472 |            0.0472 |
|         8 |                0.458  |             0.0056 |            0.0194 |
|        16 |                0.461  |             0      |            0.0069 |
|        32 |                0.4611 |             0      |            0.0014 |
|        64 |                0.4629 |             0      |            0      |
|       128 |                0.4629 |             0      |            0      |

Checkpoint accuracy remains relatively stable while exact completion collapses. This shows that local action competence is not sufficient for long-horizon computer-use success.

### 6.4 Hard-mining ablation

| variant   |   mean |    std |
|:----------|-------:|-------:|
| hard100   | 0.3638 | 0.0686 |
| hard50    | 0.3605 | 0.0149 |
| hard75    | 0.3283 | 0.051  |
| uniform   | 0.3319 | 0.0668 |

Hard-example mining is non-monotonic. A fixed hard-example ratio should not be treated as universally optimal.

---

## 7. Failure Analysis

### F1: Sparse visual RL collapse

From random initialization, sparse RL learned easy global actions and did not learn reliable visual grounding. This motivated visual warm-starting.

### F2: Category-level failure reweighting is too weak

Task-type reweighting did not reliably beat uniform synthetic training. Failure conditioning must operate at the level of concrete examples or generated counterfactual environments.

### F3: Shortcut exploitation without verification

Outcome-only reward in shortcut-heavy states leads the agent to exploit shortcut affordances. This validates the need for trajectory-aware reward correction.

### F4: Long-horizon compounding error

Even when local checkpoint accuracy is stable, exact completion collapses at longer horizons. This suggests future work should include memory, plan repair, and hierarchical verification.

### F5: Hardness/exploration tradeoff

Too much hard mining can reduce coverage; too much uniform sampling can under-train the failure frontier. The mixture is a real hyperparameter, not a cosmetic detail.

---

## 8. Limitations

1. The visual benchmark is synthetic and does not replace OSWorld, WeaveBench, BrowserGym, or MiniWoB public benchmark scores.
2. The visual policy is lightweight; frontier multimodal agents may behave differently.
3. The hard-mining effect is measured across 4 visual seeds; larger seed counts would improve confidence.
4. Exact long-horizon completion remains weak.
5. The BrowserGym bridge is implemented but not executed against real MiniWoB due to runtime network/browser limitations.
6. The synthetic UI is simplified relative to real desktop applications.

These limitations are intentionally retained to avoid overstating the result.

---

## 9. Safety and Evaluation Implications

The verifier ablation suggests that computer-use agents trained with incomplete outcome signals may learn behavior that appears successful while violating task intent. For safety-critical computer use, evaluation should inspect trajectories, intermediate states, and shortcut patterns rather than relying only on final deliverables.

FAILFORGE treats failure diagnosis as both a safety tool and a training signal. This is especially important when synthetic environments create many opportunities for reward hacking.

---

## 10. Conclusion

FAILFORGE studies a closed-loop synthetic RL regime for computer-use agents: verified failures determine future training environments. The central empirical lesson is nuanced. Failure-only training is not enough, and uniform synthetic data is a strong baseline. However, when failure conditioning is applied at the example/environment level and combined with exploration, replay, and trajectory-aware verification, it improves visual OOD robustness and suppresses shortcut exploitation.

The next step is external validation on BrowserGym/MiniWoB and, eventually, OSWorld/WeaveBench-style long-horizon public benchmarks.

## References

[1] M. Yuan et al. **OSWorld2.0: Benchmarking Computer Use Agents on Long-Horizon Real-World Tasks.** arXiv:2606.29537, 2026.  
[2] W. Li et al. **WeaveBench: A Long-Horizon, Real-World Benchmark for Hybrid Computer-Use Agent Evaluation.** arXiv:2606.09426, 2026.  
[3] T. Ge, B. Peng, H. Cheng, J. Gao. **Synthetic Computers at Scale for Long-Horizon Productivity Simulation.** arXiv:2604.28181, 2026.  
[4] ServiceNow Research. **BrowserGym: a Gym environment for web task automation.** GitHub repository, accessed Aug. 26, 2026.  
[5] Farama Foundation. **MiniWoB++: Mini World of Bits++ web interaction environments.** GitHub repository, accessed Aug. 26, 2026.


# Appendix

This appendix is intentionally comprehensive. It includes all executed experimental families,
negative results, ablations, raw aggregate tables, implementation notes, and the public-benchmark
bridge status. No public benchmark score is fabricated.

## Appendix A — Complete Experimental Inventory

The project contains three evidence layers:

1. **Symbolic computer-use benchmark** — executed.
2. **Visual synthetic computer-use benchmark** — executed.
3. **BrowserGym/MiniWoB bridge** — implemented and mock-tested; real public benchmark execution
   not completed in this runtime because required browser/network dependencies could not be installed.

The executed experiment families are:

- sparse RL baseline;
- shaped RL baseline;
- uniform synthetic RL;
- fixed curriculum;
- failure-only curriculum;
- replay-anchored curriculum;
- adaptive FAILFORGE;
- verifier ablation;
- reward-hacking stress test;
- replay-mixture sweep;
- long-horizon scaling;
- synthetic training-budget scaling;
- failure-targeting ablations;
- visual OOD evaluation;
- visual example-level hard mining;
- hard-example mixture ablation;
- visual verifier stress test;
- visual long-horizon scaling;
- mock BrowserGym bridge validation.



## Appendix B — Symbolic Benchmark: Main Results


### B.1 Main symbolic sweep summary

| strategy       | scenario   |   checkpoint_mean |   checkpoint_sd |   highcomp_mean |   exact_mean |   verify_mean |   recovery_mean |   shortcut_mean |
|:---------------|:-----------|------------------:|----------------:|----------------:|-------------:|--------------:|----------------:|----------------:|
| fail_no_replay | combined   |          0.688724 |       0.0345417 |     0.0341667   |   0          |      0.73606  |        0.641336 |       0         |
| fail_no_replay | dynamic    |          0.703056 |       0.0439435 |     0.133333    |   0          |      0.751713 |        0.659301 |       0         |
| fail_no_replay | hack       |          0.703194 |       0.042415  |     0.146667    |   0          |      0.798522 |        0.660632 |       0         |
| fail_no_replay | iid        |          0.754583 |       0.0675834 |     0.429167    |   0.0375     |      0.807949 |        0.786265 |       0         |
| fail_no_replay | long       |          0.727695 |       0.0519998 |     0.130833    |   0          |      0.811079 |        0.723987 |       0         |
| fail_replay    | combined   |          0.668555 |       0.0419484 |     0.0166667   |   0          |      0.713937 |        0.63153  |       0         |
| fail_replay    | dynamic    |          0.679479 |       0.0529775 |     0.0991667   |   0          |      0.721833 |        0.64688  |       0         |
| fail_replay    | hack       |          0.680417 |       0.0531344 |     0.108333    |   0          |      0.75831  |        0.647154 |       0         |
| fail_replay    | iid        |          0.722708 |       0.0763171 |     0.324167    |   0.0308333  |      0.736478 |        0.759756 |       0         |
| fail_replay    | long       |          0.699544 |       0.0616737 |     0.0808333   |   0          |      0.764687 |        0.705154 |       0         |
| fixed          | combined   |          0.588581 |       0.0510451 |     0.000833333 |   0          |      0.623862 |        0.561676 |       0         |
| fixed          | dynamic    |          0.6      |       0.0403564 |     0.0175      |   0          |      0.637381 |        0.577243 |       0         |
| fixed          | hack       |          0.598993 |       0.0463982 |     0.0175      |   0          |      0.67225  |        0.583479 |       0         |
| fixed          | iid        |          0.626806 |       0.0330548 |     0.120833    |   0.0025     |      0.639801 |        0.69506  |       0         |
| fixed          | long       |          0.612969 |       0.0416943 |     0.0025      |   0          |      0.666354 |        0.633782 |       0         |
| shaped         | combined   |          0.275938 |       0.0550919 |     0           |   0          |      0.225049 |        0.232317 |       0.426107  |
| shaped         | dynamic    |          0.304375 |       0.0583429 |     0           |   0          |      0.257363 |        0.267997 |       0.423958  |
| shaped         | hack       |          0.126424 |       0.054736  |     0           |   0          |      0.148027 |        0.107547 |       0.681458  |
| shaped         | iid        |          0.587431 |       0.0904411 |     0.108333    |   0.00416667 |      0.560396 |        0.685081 |       0.0993056 |
| shaped         | long       |          0.47125  |       0.0698184 |     0           |   0          |      0.43835  |        0.50564  |       0.111966  |
| sparse         | combined   |          0.212578 |       0.0838446 |     0           |   0          |      0.208397 |        0.228343 |       0.128945  |
| sparse         | dynamic    |          0.214028 |       0.0747062 |     0           |   0          |      0.20911  |        0.229246 |       0.119896  |
| sparse         | hack       |          0.213194 |       0.0265296 |     0           |   0          |      0.182998 |        0.226521 |       0.175486  |
| sparse         | iid        |          0.204792 |       0.0538638 |     0           |   0          |      0.236943 |        0.261662 |       0.0195139 |
| sparse         | long       |          0.202982 |       0.0494054 |     0           |   0          |      0.221395 |        0.232886 |       0.0259375 |
| uniform        | combined   |          0.636432 |       0.124859  |     0.115       |   0          |      0.649861 |        0.64267  |       0         |
| uniform        | dynamic    |          0.636736 |       0.122506  |     0.109167    |   0.00166667 |      0.649723 |        0.64622  |       0         |
| uniform        | hack       |          0.651944 |       0.0968818 |     0.114167    |   0.00166667 |      0.66582  |        0.675136 |       0         |
| uniform        | iid        |          0.669583 |       0.0814353 |     0.22        |   0.0133333  |      0.59348  |        0.747455 |       0         |
| uniform        | long       |          0.659622 |       0.0901662 |     0.0966667   |   0          |      0.640433 |        0.719255 |       0         |


### B.2 Adaptive FAILFORGE summary

| scenario   |   checkpoint |        sd |   highcomp |   verify |   recovery |   shortcut |
|:-----------|-------------:|----------:|-----------:|---------:|-----------:|-----------:|
| combined   |     0.69054  | 0.0837564 |   0.154167 | 0.726606 |   0.693654 |          0 |
| dynamic    |     0.691753 | 0.0841195 |   0.182083 | 0.725971 |   0.701522 |          0 |
| hack       |     0.687396 | 0.083271  |   0.165    | 0.711157 |   0.701118 |          0 |
| iid        |     0.705347 | 0.066481  |   0.29375  | 0.625224 |   0.760338 |          0 |
| long       |     0.693301 | 0.0773829 |   0.133333 | 0.672496 |   0.728935 |          0 |


### B.3 Extended symbolic baselines

|   seed | strategy       | scenario   |   exact_completion |   high_completion |   checkpoint_acc |   p50_completion |   verify_acc |   recovery_rate |   shortcut_rate |
|-------:|:---------------|:-----------|-------------------:|------------------:|-----------------:|-----------------:|-------------:|----------------:|----------------:|
|      0 | uniform        | iid        |         0          |        0.11       |         0.621667 |         0.666667 |     0.430108 |        0.724211 |               0 |
|      0 | uniform        | long       |         0          |        0          |         0.618073 |         0.625    |     0.540444 |        0.712163 |               0 |
|      0 | uniform        | dynamic    |         0          |        0.0233333  |         0.632778 |         0.625    |     0.661126 |        0.67482  |               0 |
|      0 | uniform        | combined   |         0          |        0.00333333 |         0.635156 |         0.625    |     0.665288 |        0.678725 |               0 |
|      0 | uniform        | hack       |         0          |        0.02       |         0.615972 |         0.625    |     0.611547 |        0.662417 |               0 |
|      0 | fixed          | iid        |         0.00666667 |        0.16       |         0.65     |         0.666667 |     0.617204 |        0.715789 |               0 |
|      0 | fixed          | long       |         0          |        0.00333333 |         0.646563 |         0.640625 |     0.664384 |        0.663442 |               0 |
|      0 | fixed          | dynamic    |         0          |        0.03       |         0.631111 |         0.625    |     0.665875 |        0.622782 |               0 |
|      0 | fixed          | combined   |         0          |        0          |         0.627865 |         0.625    |     0.662471 |        0.614613 |               0 |
|      0 | fixed          | hack       |         0          |        0.0166667  |         0.638056 |         0.625    |     0.685538 |        0.648559 |               0 |
|      0 | fail_no_replay | iid        |         0.0533333  |        0.536667   |         0.791111 |         0.833333 |     0.849462 |        0.814737 |               0 |
|      0 | fail_no_replay | long       |         0          |        0.176667   |         0.753281 |         0.75     |     0.84801  |        0.745335 |               0 |
|      0 | fail_no_replay | dynamic    |         0          |        0.163333   |         0.721806 |         0.708333 |     0.773213 |        0.671463 |               0 |
|      0 | fail_no_replay | combined   |         0          |        0.0366667  |         0.707031 |         0.703125 |     0.759285 |        0.647206 |               0 |
|      0 | fail_no_replay | hack       |         0          |        0.19       |         0.727083 |         0.729167 |     0.84417  |        0.681818 |               0 |
|      1 | uniform        | iid        |         0.0533333  |        0.506667   |         0.785833 |         0.833333 |     0.63913  |        0.837953 |               0 |
|      1 | uniform        | long       |         0          |        0.383333   |         0.789635 |         0.78125  |     0.73496  |        0.843078 |               0 |
|      1 | uniform        | dynamic    |         0.00666667 |        0.39       |         0.790694 |         0.791667 |     0.815144 |        0.809375 |               0 |
|      1 | uniform        | combined   |         0          |        0.453333   |         0.794427 |         0.796875 |     0.826613 |        0.813117 |               0 |
|      1 | uniform        | hack       |         0.00666667 |        0.4        |         0.792083 |         0.791667 |     0.778733 |        0.816397 |               0 |
|      1 | fixed          | iid        |         0.00333333 |        0.166667   |         0.653333 |         0.666667 |     0.621739 |        0.663113 |               0 |
|      1 | fixed          | long       |         0          |        0.00666667 |         0.64974  |         0.640625 |     0.678584 |        0.677419 |               0 |
|      1 | fixed          | dynamic    |         0          |        0.0333333  |         0.6375   |         0.625    |     0.672587 |        0.616827 |               0 |
|      1 | fixed          | combined   |         0          |        0.00333333 |         0.636927 |         0.640625 |     0.673787 |        0.621345 |               0 |
|      1 | fixed          | hack       |         0          |        0.05       |         0.640278 |         0.625    |     0.681525 |        0.617206 |               0 |
|      1 | fail_no_replay | iid        |         0.00333333 |        0.166667   |         0.653333 |         0.666667 |     0.621739 |        0.663113 |               0 |
|      1 | fail_no_replay | long       |         0          |        0.00666667 |         0.64974  |         0.640625 |     0.678584 |        0.677419 |               0 |
|      1 | fail_no_replay | dynamic    |         0          |        0.0333333  |         0.6375   |         0.625    |     0.672587 |        0.616827 |               0 |
|      1 | fail_no_replay | combined   |         0          |        0.00333333 |         0.636927 |         0.640625 |     0.673787 |        0.621345 |               0 |
|      1 | fail_no_replay | hack       |         0          |        0.05       |         0.640278 |         0.625    |     0.681525 |        0.617206 |               0 |
|      2 | uniform        | iid        |         0          |        0.08       |         0.605833 |         0.583333 |     0.632735 |        0.702991 |               0 |
|      2 | uniform        | long       |         0          |        0          |         0.585    |         0.585938 |     0.612259 |        0.665845 |               0 |
|      2 | uniform        | dynamic    |         0          |        0          |         0.490833 |         0.5      |     0.446525 |        0.476068 |               0 |
|      2 | uniform        | combined   |         0          |        0          |         0.489063 |         0.484375 |     0.447872 |        0.46985  |               0 |
|      2 | uniform        | hack       |         0          |        0.00333333 |         0.569861 |         0.583333 |     0.578977 |        0.599461 |               0 |
|      2 | fixed          | iid        |         0          |        0.0666667  |         0.581944 |         0.583333 |     0.566866 |        0.707265 |               0 |
|      2 | fixed          | long       |         0          |        0          |         0.566406 |         0.5625   |     0.600551 |        0.619014 |               0 |
|      2 | fixed          | dynamic    |         0          |        0.00666667 |         0.574583 |         0.583333 |     0.604493 |        0.566425 |               0 |
|      2 | fixed          | combined   |         0          |        0          |         0.551719 |         0.546875 |     0.587089 |        0.539737 |               0 |
|      2 | fixed          | hack       |         0          |        0.00333333 |         0.558472 |         0.583333 |     0.630145 |        0.571429 |               0 |
|      2 | fail_no_replay | iid        |         0.05       |        0.52       |         0.790278 |         0.833333 |     0.878244 |        0.83547  |               0 |
|      2 | fail_no_replay | long       |         0          |        0.17       |         0.751771 |         0.75     |     0.859504 |        0.738028 |               0 |
|      2 | fail_no_replay | dynamic    |         0          |        0.183333   |         0.731389 |         0.75     |     0.782822 |        0.67943  |               0 |
|      2 | fail_no_replay | combined   |         0          |        0.0433333  |         0.704948 |         0.703125 |     0.759299 |        0.65039  |               0 |
|      2 | fail_no_replay | hack       |         0          |        0.19       |         0.73     |         0.708333 |     0.843715 |        0.68248  |               0 |
|      3 | uniform        | iid        |         0          |        0.183333   |         0.665    |         0.666667 |     0.671946 |        0.724665 |               0 |
|      3 | uniform        | long       |         0          |        0.00333333 |         0.645781 |         0.640625 |     0.674069 |        0.655933 |               0 |
|      3 | uniform        | dynamic    |         0          |        0.0233333  |         0.632639 |         0.625    |     0.676097 |        0.624618 |               0 |
|      3 | uniform        | combined   |         0          |        0.00333333 |         0.627083 |         0.625    |     0.659673 |        0.608987 |               0 |
|      3 | uniform        | hack       |         0          |        0.0333333  |         0.629861 |         0.625    |     0.694026 |        0.622271 |               0 |
|      3 | fixed          | iid        |         0          |        0.09       |         0.621944 |         0.666667 |     0.753394 |        0.694073 |               0 |
|      3 | fixed          | long       |         0          |        0          |         0.589167 |         0.59375  |     0.7219   |        0.575253 |               0 |
|      3 | fixed          | dynamic    |         0          |        0          |         0.556806 |         0.541667 |     0.606569 |        0.502936 |               0 |
|      3 | fixed          | combined   |         0          |        0          |         0.537813 |         0.546875 |     0.572102 |        0.471009 |               0 |
|      3 | fixed          | hack       |         0          |        0          |         0.559167 |         0.541667 |     0.691792 |        0.496725 |               0 |
|      3 | fail_no_replay | iid        |         0.0433333  |        0.493333   |         0.783611 |         0.75     |     0.882353 |        0.83174  |               0 |
|      3 | fail_no_replay | long       |         0          |        0.17       |         0.75599  |         0.765625 |     0.858217 |        0.735166 |               0 |
|      3 | fail_no_replay | dynamic    |         0          |        0.153333   |         0.721528 |         0.708333 |     0.778231 |        0.669486 |               0 |
|      3 | fail_no_replay | combined   |         0          |        0.0533333  |         0.70599  |         0.703125 |     0.751867 |        0.646403 |               0 |
|      3 | fail_no_replay | hack       |         0          |        0.156667   |         0.715417 |         0.708333 |     0.824679 |        0.661026 |               0 |
|      4 | uniform        | iid        |         0.0633333  |        0.453333   |         0.785278 |         0.75     |     0.660793 |        0.832579 |               0 |
|      4 | uniform        | long       |         0          |        0.373333   |         0.788281 |         0.796875 |     0.741306 |        0.845855 |               0 |
|      4 | uniform        | dynamic    |         0.00666667 |        0.436667   |         0.795694 |         0.791667 |     0.812263 |        0.824318 |               0 |
|      4 | uniform        | hack       |         0.00333333 |        0.39       |         0.78625  |         0.791667 |     0.784027 |        0.808319 |               0 |
|      4 | fixed          | iid        |         0.03       |        0.503333   |         0.788333 |         0.833333 |     0.903084 |        0.830317 |               0 |
|      4 | fixed          | long       |         0          |        0.166667   |         0.754531 |         0.75     |     0.864443 |        0.744268 |               0 |
|      4 | fixed          | dynamic    |         0          |        0.156667   |         0.721806 |         0.708333 |     0.778646 |        0.677597 |               0 |
|      4 | fixed          | hack       |         0          |        0.173333   |         0.726944 |         0.708333 |     0.835771 |        0.685779 |               0 |
|      4 | fail_no_replay | iid        |         0          |        0.12       |         0.636111 |         0.666667 |     0.781938 |        0.739819 |               0 |
|      4 | fail_no_replay | long       |         0          |        0          |         0.599792 |         0.59375  |     0.726047 |        0.609171 |               0 |
|      4 | fail_no_replay | dynamic    |         0          |        0.00333333 |         0.569028 |         0.583333 |     0.59233  |        0.527286 |               0 |
|      4 | fail_no_replay | hack       |         0          |        0.00333333 |         0.583056 |         0.583333 |     0.678853 |        0.545812 |               0 |
|      5 | uniform        | iid        |         0.00333333 |        0.156667   |         0.659167 |         0.666667 |     0.650104 |        0.702638 |               0 |
|      5 | uniform        | long       |         0          |        0.00333333 |         0.64474  |         0.640625 |     0.678024 |        0.662249 |               0 |
|      5 | uniform        | dynamic    |         0          |        0.0366667  |         0.641944 |         0.625    |     0.666432 |        0.633365 |               0 |
|      5 | uniform        | hack       |         0          |        0.03       |         0.63375  |         0.625    |     0.686799 |        0.624585 |               0 |
|      5 | fixed          | iid        |         0.00666667 |        0.19       |         0.669722 |         0.666667 |     0.68323  |        0.832134 |               0 |
|      5 | fixed          | long       |         0          |        0          |         0.549323 |         0.546875 |     0.564906 |        0.59633  |               0 |
|      5 | fixed          | dynamic    |         0          |        0          |         0.520694 |         0.5      |     0.505036 |        0.496176 |               0 |
|      5 | fixed          | hack       |         0          |        0          |         0.471528 |         0.458333 |     0.484768 |        0.459025 |               0 |
|      5 | fail_no_replay | iid        |         0.05       |        0.506667   |         0.781944 |         0.833333 |     0.867495 |        0.83693  |               0 |
|      5 | fail_no_replay | long       |         0          |        0.163333   |         0.750104 |         0.75     |     0.854855 |        0.7421   |               0 |
|      5 | fail_no_replay | dynamic    |         0          |        0.176667   |         0.727778 |         0.75     |     0.770907 |        0.676864 |               0 |
|      5 | fail_no_replay | hack       |         0          |        0.18       |         0.727222 |         0.75     |     0.825762 |        0.680509 |               0 |
|      6 | uniform        | iid        |         0.00333333 |        0.16       |         0.661944 |         0.666667 |     0.6      |        0.701789 |               0 |
|      6 | uniform        | long       |         0          |        0          |         0.642656 |         0.640625 |     0.647547 |        0.667855 |               0 |
|      6 | uniform        | dynamic    |         0          |        0.0133333  |         0.644444 |         0.666667 |     0.677241 |        0.633701 |               0 |
|      6 | uniform        | hack       |         0          |        0.0233333  |         0.637222 |         0.625    |     0.682621 |        0.629609 |               0 |
|      6 | fixed          | iid        |         0.00333333 |        0.16       |         0.661944 |         0.666667 |     0.6      |        0.701789 |               0 |
|      6 | fixed          | long       |         0          |        0          |         0.642656 |         0.640625 |     0.647547 |        0.667855 |               0 |
|      6 | fixed          | dynamic    |         0          |        0.0133333  |         0.644444 |         0.666667 |     0.677241 |        0.633701 |               0 |
|      6 | fixed          | hack       |         0          |        0.0233333  |         0.637222 |         0.625    |     0.682621 |        0.629609 |               0 |
|      6 | fail_no_replay | iid        |         0          |        0.00666667 |         0.449444 |         0.416667 |     0.464516 |        0.554672 |               0 |
|      6 | fail_no_replay | long       |         0          |        0          |         0.444427 |         0.4375   |     0.50622  |        0.52067  |               0 |
|      6 | fail_no_replay | dynamic    |         0          |        0          |         0.400694 |         0.375    |     0.378515 |        0.403265 |               0 |
|      6 | fail_no_replay | hack       |         0          |        0          |         0.329722 |         0.333333 |     0.34245  |        0.348376 |               0 |
|      7 | uniform        | iid        |         0.0533333  |        0.486667   |         0.788056 |         0.75     |     0.66309  |        0.860606 |               0 |
|      7 | uniform        | long       |         0          |        0.403333   |         0.789271 |         0.796875 |     0.730432 |        0.839493 |               0 |
|      7 | uniform        | dynamic    |         0.00333333 |        0.426667   |         0.795694 |         0.791667 |     0.816512 |        0.825196 |               0 |
|      7 | uniform        | hack       |         0.00666667 |        0.38       |         0.7875   |         0.791667 |     0.781198 |        0.824943 |               0 |
|      7 | fixed          | iid        |         0          |        0.06       |         0.528333 |         0.5      |     0.592275 |        0.719192 |               0 |
|      7 | fixed          | long       |         0          |        0          |         0.537813 |         0.53125  |     0.61039  |        0.665982 |               0 |
|      7 | fixed          | dynamic    |         0          |        0          |         0.459583 |         0.458333 |     0.424004 |        0.467254 |               0 |
|      7 | fixed          | hack       |         0          |        0          |         0.472361 |         0.458333 |     0.518187 |        0.550343 |               0 |
|      7 | fail_no_replay | iid        |         0.0666667  |        0.486667   |         0.782222 |         0.75     |     0.877682 |        0.810101 |               0 |
|      7 | fail_no_replay | long       |         0          |        0.123333   |         0.749323 |         0.75     |     0.849421 |        0.742642 |               0 |
|      7 | fail_no_replay | dynamic    |         0          |        0.166667   |         0.715833 |         0.708333 |     0.770222 |        0.665873 |               0 |
|      7 | fail_no_replay | hack       |         0          |        0.136667   |         0.717639 |         0.708333 |     0.81925  |        0.671053 |               0 |
|      4 | uniform        | combined   |         0          |        0.413333   |         0.790781 |         0.796875 |     0.81995  |        0.809276 |               0 |
|      4 | fixed          | combined   |         0          |        0.0266667  |         0.702812 |         0.703125 |     0.75566  |        0.639993 |               0 |
|      4 | fail_no_replay | combined   |         0          |        0          |         0.555469 |         0.546875 |     0.569685 |        0.491253 |               0 |
|      5 | uniform        | combined   |         0          |        0.00333333 |         0.627292 |         0.625    |     0.660682 |        0.607731 |               0 |
|      5 | fixed          | combined   |         0          |        0          |         0.507708 |         0.515625 |     0.510361 |        0.472125 |               0 |
|      5 | fail_no_replay | combined   |         0          |        0.0366667  |         0.702708 |         0.703125 |     0.748214 |        0.646213 |               0 |
|      6 | uniform        | combined   |         0          |        0.00333333 |         0.633333 |         0.640625 |     0.673575 |        0.618038 |               0 |
|      6 | fixed          | combined   |         0          |        0.00333333 |         0.633333 |         0.640625 |     0.673575 |        0.618038 |               0 |
|      6 | fail_no_replay | combined   |         0          |        0          |         0.389688 |         0.390625 |     0.365374 |        0.381249 |               0 |
|      7 | uniform        | combined   |         0          |        0.393333   |         0.790833 |         0.796875 |     0.825161 |        0.809495 |               0 |
|      7 | fixed          | combined   |         0          |        0          |         0.457813 |         0.453125 |     0.407391 |        0.45644  |               0 |
|      7 | fail_no_replay | combined   |         0          |        0.0266667  |         0.698646 |         0.703125 |     0.745795 |        0.64756  |               0 |


### B.4 Replay-mixture sweep summary

|   lambda_ | scenario   |   checkpoint |   highcomp |   verify |   recovery |
|----------:|:-----------|-------------:|-----------:|---------:|-----------:|
|      0    | combined   |     0.676229 | 0.0146667  | 0.72302  |   0.635114 |
|      0    | iid        |     0.754222 | 0.413333   | 0.814785 |   0.790909 |
|      0.1  | combined   |     0.651771 | 0.00666667 | 0.694962 |   0.622242 |
|      0.1  | iid        |     0.71     | 0.304      | 0.734754 |   0.74767  |
|      0.25 | combined   |     0.650937 | 0.008      | 0.693597 |   0.622499 |
|      0.25 | iid        |     0.711444 | 0.286667   | 0.741593 |   0.749813 |
|      0.4  | combined   |     0.650937 | 0.008      | 0.693597 |   0.622499 |
|      0.4  | iid        |     0.711444 | 0.286667   | 0.741593 |   0.749813 |
|      0.6  | combined   |     0.705917 | 0.137333   | 0.74668  |   0.690223 |
|      0.6  | iid        |     0.748778 | 0.390667   | 0.756584 |   0.792431 |
|      0.8  | combined   |     0.626979 | 0          | 0.666712 |   0.611601 |
|      0.8  | iid        |     0.668778 | 0.178667   | 0.655389 |   0.724077 |


### B.5 Symbolic horizon-scaling summary

|   horizon |   checkpoint |   highcomp |      exact |   verify |   recovery |
|----------:|-------------:|-----------:|-----------:|---------:|-----------:|
|         8 |     0.705208 |   0.256667 | 0.0716667  | 0.75342  |   0.684333 |
|        16 |     0.7075   |   0.301667 | 0.00833333 | 0.754306 |   0.689155 |
|        32 |     0.70375  |   0.2      | 0          | 0.743419 |   0.683556 |
|        64 |     0.708281 |   0.14     | 0          | 0.752584 |   0.692535 |
|        96 |     0.708559 |   0.168333 | 0          | 0.745724 |   0.688463 |
|       128 |     0.707409 |   0.145    | 0          | 0.748787 |   0.689116 |


### B.6 Synthetic training-budget scaling summary

|   train_steps |   approx_train_episodes |   checkpoint |   highcomp |   verify |   recovery |
|--------------:|------------------------:|-------------:|-----------:|---------:|-----------:|
|            75 |                    2400 |     0.337417 |   0        | 0.356166 |   0.361319 |
|           150 |                    4800 |     0.46     |   0        | 0.464973 |   0.484186 |
|           300 |                    9600 |     0.710167 |   0.130667 | 0.749998 |   0.69616  |
|           500 |                   16000 |     0.710167 |   0.130667 | 0.749998 |   0.69616  |


### B.7 Symbolic ablation summary

| variant            | scenario   |   checkpoint |   highcomp |   verify |   recovery |   shortcut |
|:-------------------|:-----------|-------------:|-----------:|---------:|-----------:|-----------:|
| full               | combined   |     0.653458 |  0.0146667 | 0.688052 |   0.629129 |   0        |
| full               | hack       |     0.665056 |  0.092     | 0.727062 |   0.650591 |   0        |
| no_antihack_target | combined   |     0.684354 |  0.0293333 | 0.728633 |   0.657671 |   0        |
| no_antihack_target | hack       |     0.695278 |  0.134667  | 0.76352  |   0.677434 |   0        |
| no_recover_target  | combined   |     0.658937 |  0.0146667 | 0.695049 |   0.629129 |   0        |
| no_recover_target  | hack       |     0.669889 |  0.096     | 0.735742 |   0.650591 |   0        |
| no_state_target    | combined   |     0.659479 |  0.0146667 | 0.695982 |   0.629129 |   0        |
| no_state_target    | hack       |     0.672333 |  0.096     | 0.741384 |   0.650591 |   0        |
| no_verifier        | combined   |     0.409667 |  0         | 0.379618 |   0.403143 |   0.465854 |
| no_verifier        | hack       |     0.1275   |  0         | 0.146821 |   0.102333 |   0.721111 |
| no_verify_target   | combined   |     0.653458 |  0.0146667 | 0.688052 |   0.629129 |   0        |
| no_verify_target   | hack       |     0.665056 |  0.092     | 0.727062 |   0.650591 |   0        |



### B.8 Symbolic negative results and interpretation

The symbolic experiments produced several important negative results that were retained:

- Failure-only curricula did **not** reliably dominate uniform synthetic training.
- Replay exhibited a non-monotonic tradeoff rather than a simple "more replay is better" pattern.
- Some curricula improved long-horizon generalization while degrading easier or dynamic tasks,
  revealing catastrophic-forgetting-like behavior.
- The verifier ablation was substantially stronger than many individual failure-targeting ablations,
  indicating that reward integrity is a separate and critical mechanism.
- Synthetic experience scaling showed early gains followed by saturation, suggesting that data volume
  alone is insufficient after a competence threshold is reached.

These negative results motivated the final adaptive mixture formulation and the move from
task-category reweighting toward example/environment-level failure mining.


## Appendix C — Visual Benchmark: Complete Results


### C.1 Visual OOD summary

| method             | eval      |     mean |       std |
|:-------------------|:----------|---------:|----------:|
| FAILFORGE-HardMine | IID       | 0.803875 | 0.109826  |
| FAILFORGE-HardMine | combined  | 0.491875 | 0.0618983 |
| FAILFORGE-HardMine | noise     | 0.801    | 0.110472  |
| FAILFORGE-HardMine | occlusion | 0.800125 | 0.109987  |
| FAILFORGE-HardMine | shift     | 0.49525  | 0.0596581 |
| UniformFactor-RL   | IID       | 0.69725  | 0.127818  |
| UniformFactor-RL   | combined  | 0.433    | 0.0656417 |
| UniformFactor-RL   | noise     | 0.6935   | 0.127068  |
| UniformFactor-RL   | occlusion | 0.695    | 0.12705   |
| UniformFactor-RL   | shift     | 0.43225  | 0.0559174 |


### C.2 Main failure-conditioned hard-mining results (all seeds/evals)

|   seed | method             | eval      |   overall |   normal_acc |   verify_acc |   popup_acc |   hack_acc |   hack_shortcut |
|-------:|:-------------------|:----------|----------:|-------------:|-------------:|------------:|-----------:|----------------:|
|      0 | UniformFactor-RL   | IID       |    0.6185 |        0.224 |        0.99  |       1     |      0.26  |           0     |
|      0 | UniformFactor-RL   | noise     |    0.617  |        0.222 |        0.988 |       1     |      0.258 |           0     |
|      0 | UniformFactor-RL   | shift     |    0.4385 |        0.034 |        0.99  |       0.64  |      0.09  |           0     |
|      0 | UniformFactor-RL   | occlusion |    0.6155 |        0.218 |        0.99  |       1     |      0.254 |           0     |
|      0 | UniformFactor-RL   | combined  |    0.435  |        0.036 |        0.982 |       0.652 |      0.07  |           0     |
|      0 | FAILFORGE-HardMine | IID       |    0.7925 |        0.6   |        0.884 |       1     |      0.686 |           0     |
|      0 | FAILFORGE-HardMine | noise     |    0.792  |        0.606 |        0.88  |       1     |      0.682 |           0     |
|      0 | FAILFORGE-HardMine | shift     |    0.4955 |        0.114 |        0.874 |       0.834 |      0.16  |           0     |
|      0 | FAILFORGE-HardMine | occlusion |    0.787  |        0.588 |        0.878 |       1     |      0.682 |           0     |
|      0 | FAILFORGE-HardMine | combined  |    0.495  |        0.12  |        0.87  |       0.816 |      0.174 |           0.002 |
|      1 | UniformFactor-RL   | IID       |    0.65   |        0.29  |        0.992 |       1     |      0.318 |           0     |
|      1 | UniformFactor-RL   | noise     |    0.6415 |        0.272 |        0.994 |       1     |      0.3   |           0     |
|      1 | UniformFactor-RL   | shift     |    0.3925 |        0.026 |        0.998 |       0.478 |      0.068 |           0     |
|      1 | UniformFactor-RL   | occlusion |    0.6485 |        0.278 |        0.994 |       1     |      0.322 |           0     |
|      1 | UniformFactor-RL   | combined  |    0.3815 |        0.032 |        1     |       0.46  |      0.034 |           0     |
|      1 | FAILFORGE-HardMine | IID       |    0.792  |        0.754 |        0.614 |       1     |      0.8   |           0     |
|      1 | FAILFORGE-HardMine | noise     |    0.798  |        0.742 |        0.65  |       1     |      0.8   |           0     |
|      1 | FAILFORGE-HardMine | shift     |    0.503  |        0.174 |        0.832 |       0.786 |      0.22  |           0.01  |
|      1 | FAILFORGE-HardMine | occlusion |    0.7945 |        0.754 |        0.63  |       1     |      0.794 |           0     |
|      1 | FAILFORGE-HardMine | combined  |    0.4835 |        0.162 |        0.866 |       0.762 |      0.144 |           0     |
|      2 | UniformFactor-RL   | IID       |    0.888  |        0.762 |        1     |       1     |      0.79  |           0     |
|      2 | UniformFactor-RL   | noise     |    0.8835 |        0.75  |        1     |       1     |      0.784 |           0     |
|      2 | UniformFactor-RL   | shift     |    0.509  |        0.188 |        0.75  |       0.86  |      0.238 |           0.008 |
|      2 | UniformFactor-RL   | occlusion |    0.8845 |        0.762 |        1     |       1     |      0.776 |           0     |
|      2 | UniformFactor-RL   | combined  |    0.525  |        0.196 |        0.81  |       0.892 |      0.202 |           0.006 |
|      2 | FAILFORGE-HardMine | IID       |    0.949  |        0.882 |        1     |       1     |      0.914 |           0     |
|      2 | FAILFORGE-HardMine | noise     |    0.942  |        0.862 |        1     |       1     |      0.906 |           0     |
|      2 | FAILFORGE-HardMine | shift     |    0.564  |        0.262 |        0.776 |       0.912 |      0.306 |           0.022 |
|      2 | FAILFORGE-HardMine | occlusion |    0.9435 |        0.874 |        1     |       1     |      0.9   |           0     |
|      2 | FAILFORGE-HardMine | combined  |    0.57   |        0.266 |        0.818 |       0.93  |      0.266 |           0.01  |
|      3 | UniformFactor-RL   | IID       |    0.6325 |        0.298 |        0.89  |       0.984 |      0.358 |           0     |
|      3 | UniformFactor-RL   | noise     |    0.632  |        0.292 |        0.908 |       0.984 |      0.344 |           0     |
|      3 | UniformFactor-RL   | shift     |    0.389  |        0.032 |        0.982 |       0.488 |      0.054 |           0     |
|      3 | UniformFactor-RL   | occlusion |    0.6315 |        0.284 |        0.904 |       0.986 |      0.352 |           0     |
|      3 | UniformFactor-RL   | combined  |    0.3905 |        0.038 |        0.976 |       0.482 |      0.066 |           0     |
|      3 | FAILFORGE-HardMine | IID       |    0.682  |        0.67  |        0.314 |       0.994 |      0.75  |           0     |
|      3 | FAILFORGE-HardMine | noise     |    0.672  |        0.648 |        0.332 |       0.994 |      0.714 |           0     |
|      3 | FAILFORGE-HardMine | shift     |    0.4185 |        0.122 |        0.594 |       0.834 |      0.124 |           0     |
|      3 | FAILFORGE-HardMine | occlusion |    0.6755 |        0.638 |        0.34  |       0.994 |      0.73  |           0     |
|      3 | FAILFORGE-HardMine | combined  |    0.419  |        0.1   |        0.62  |       0.82  |      0.136 |           0     |


### C.3 Earlier visual task-type reweighting sweep

|   seed | method     | eval      |   overall |   normal_acc |   verify_acc |   popup_acc |   hack_acc |   hack_shortcut |
|-------:|:-----------|:----------|----------:|-------------:|-------------:|------------:|-----------:|----------------:|
|      0 | BC-only    | IID       |  0.594444 |   0.553333   |   0.364444   |   0.928889  |  0.531111  |      0          |
|      0 | BC-only    | noise     |  0.598333 |   0.557778   |   0.397778   |   0.931111  |  0.506667  |      0          |
|      0 | BC-only    | shift     |  0.382222 |   0.1        |   0.728889   |   0.6       |  0.1       |      0          |
|      0 | BC-only    | occlusion |  0.586667 |   0.52       |   0.373333   |   0.937778  |  0.515556  |      0          |
|      0 | BC-only    | combined  |  0.392778 |   0.0888889  |   0.782222   |   0.606667  |  0.0933333 |      0          |
|      0 | Fixed-RL   | IID       |  0.405    |   0.517778   |   0          |   0.622222  |  0.48      |      0.0133333  |
|      0 | Fixed-RL   | noise     |  0.396111 |   0.504444   |   0          |   0.613333  |  0.466667  |      0.00222222 |
|      0 | Fixed-RL   | shift     |  0.175    |   0.2        |   0          |   0.331111  |  0.168889  |      0.02       |
|      0 | Fixed-RL   | occlusion |  0.402222 |   0.511111   |   0          |   0.626667  |  0.471111  |      0.0155556  |
|      0 | Fixed-RL   | combined  |  0.158889 |   0.16       |   0          |   0.326667  |  0.148889  |      0.0222222  |
|      0 | Uniform-RL | IID       |  0.664444 |   0.315556   |   0.988889   |   1         |  0.353333  |      0          |
|      0 | Uniform-RL | noise     |  0.657778 |   0.306667   |   0.997778   |   1         |  0.326667  |      0          |
|      0 | Uniform-RL | shift     |  0.331111 |   0.0333333  |   0.993333   |   0.262222  |  0.0355556 |      0          |
|      0 | Uniform-RL | occlusion |  0.661111 |   0.306667   |   0.988889   |   1         |  0.348889  |      0          |
|      0 | Uniform-RL | combined  |  0.319444 |   0.0244444  |   1          |   0.224444  |  0.0288889 |      0          |
|      0 | FAILFORGE  | IID       |  0.625556 |   0.266667   |   0.906667   |   0.993333  |  0.335556  |      0          |
|      0 | FAILFORGE  | noise     |  0.613889 |   0.24       |   0.924444   |   0.991111  |  0.3       |      0          |
|      0 | FAILFORGE  | shift     |  0.296111 |   0.0422222  |   0.968889   |   0.137778  |  0.0355556 |      0          |
|      0 | FAILFORGE  | occlusion |  0.616111 |   0.251111   |   0.908889   |   0.991111  |  0.313333  |      0          |
|      0 | FAILFORGE  | combined  |  0.285556 |   0.0311111  |   0.986667   |   0.0933333 |  0.0311111 |      0          |
|      1 | BC-only    | IID       |  0.545556 |   0.355556   |   0.52       |   0.982222  |  0.324444  |      0          |
|      1 | BC-only    | noise     |  0.555    |   0.36       |   0.551111   |   0.984444  |  0.324444  |      0          |
|      1 | BC-only    | shift     |  0.360556 |   0.0733333  |   0.631111   |   0.657778  |  0.08      |      0          |
|      1 | BC-only    | occlusion |  0.548333 |   0.351111   |   0.553333   |   0.982222  |  0.306667  |      0          |
|      1 | BC-only    | combined  |  0.352778 |   0.0866667  |   0.622222   |   0.648889  |  0.0533333 |      0.00222222 |
|      1 | Fixed-RL   | IID       |  0.378333 |   0.482222   |   0          |   0.511111  |  0.52      |      0          |
|      1 | Fixed-RL   | noise     |  0.373889 |   0.473333   |   0          |   0.511111  |  0.511111  |      0          |
|      1 | Fixed-RL   | shift     |  0.163889 |   0.193333   |   0.00444444 |   0.246667  |  0.211111  |      0.0222222  |
|      1 | Fixed-RL   | occlusion |  0.372778 |   0.468889   |   0          |   0.511111  |  0.511111  |      0          |
|      1 | Fixed-RL   | combined  |  0.168889 |   0.222222   |   0.0111111  |   0.202222  |  0.24      |      0.0155556  |
|      1 | Uniform-RL | IID       |  0.593889 |   0.211111   |   0.988889   |   1         |  0.175556  |      0          |
|      1 | Uniform-RL | noise     |  0.589444 |   0.197778   |   0.991111   |   1         |  0.168889  |      0          |
|      1 | Uniform-RL | shift     |  0.365    |   0.0422222  |   0.993333   |   0.38      |  0.0444444 |      0          |
|      1 | Uniform-RL | occlusion |  0.592778 |   0.206667   |   0.993333   |   1         |  0.171111  |      0          |
|      1 | Uniform-RL | combined  |  0.353889 |   0.0377778  |   0.995556   |   0.346667  |  0.0355556 |      0          |
|      1 | FAILFORGE  | IID       |  0.568333 |   0.137778   |   0.988889   |   1         |  0.146667  |      0          |
|      1 | FAILFORGE  | noise     |  0.565    |   0.126667   |   0.988889   |   1         |  0.144444  |      0          |
|      1 | FAILFORGE  | shift     |  0.362222 |   0.0177778  |   0.984444   |   0.415556  |  0.0311111 |      0          |
|      1 | FAILFORGE  | occlusion |  0.566111 |   0.137778   |   0.984444   |   1         |  0.142222  |      0          |
|      1 | FAILFORGE  | combined  |  0.352778 |   0.02       |   0.991111   |   0.386667  |  0.0133333 |      0          |
|      2 | BC-only    | IID       |  0.557778 |   0.482222   |   0.731111   |   0.544444  |  0.473333  |      0          |
|      2 | BC-only    | noise     |  0.562778 |   0.464444   |   0.755556   |   0.586667  |  0.444444  |      0          |
|      2 | BC-only    | shift     |  0.336111 |   0.0688889  |   0.664444   |   0.524444  |  0.0866667 |      0          |
|      2 | BC-only    | occlusion |  0.557222 |   0.48       |   0.733333   |   0.562222  |  0.453333  |      0          |
|      2 | BC-only    | combined  |  0.318889 |   0.0555556  |   0.644444   |   0.522222  |  0.0533333 |      0          |
|      2 | Fixed-RL   | IID       |  0.511111 |   0.551111   |   0.493333   |   0.473333  |  0.526667  |      0          |
|      2 | Fixed-RL   | noise     |  0.505556 |   0.537778   |   0.486667   |   0.473333  |  0.524444  |      0          |
|      2 | Fixed-RL   | shift     |  0.25     |   0.12       |   0.426667   |   0.308889  |  0.144444  |      0.00666667 |
|      2 | Fixed-RL   | occlusion |  0.508333 |   0.548889   |   0.491111   |   0.473333  |  0.52      |      0          |
|      2 | Fixed-RL   | combined  |  0.24     |   0.106667   |   0.388889   |   0.317778  |  0.146667  |      0.00222222 |
|      2 | Uniform-RL | IID       |  0.622778 |   0.257778   |   0.982222   |   0.955556  |  0.295556  |      0          |
|      2 | Uniform-RL | noise     |  0.62     |   0.246667   |   0.988889   |   0.964444  |  0.28      |      0          |
|      2 | Uniform-RL | shift     |  0.375    |   0.0488889  |   0.404444   |   0.993333  |  0.0533333 |      0          |
|      2 | Uniform-RL | occlusion |  0.621667 |   0.248889   |   0.984444   |   0.966667  |  0.286667  |      0          |
|      2 | Uniform-RL | combined  |  0.336111 |   0.0177778  |   0.293333   |   0.993333  |  0.04      |      0          |
|      2 | FAILFORGE  | IID       |  0.575    |   0.171111   |   0.948889   |   0.975556  |  0.204444  |      0          |
|      2 | FAILFORGE  | noise     |  0.568889 |   0.157778   |   0.944444   |   0.982222  |  0.191111  |      0          |
|      2 | FAILFORGE  | shift     |  0.292222 |   0.0177778  |   0.133333   |   1         |  0.0177778 |      0          |
|      2 | FAILFORGE  | occlusion |  0.572222 |   0.171111   |   0.942222   |   0.977778  |  0.197778  |      0          |
|      2 | FAILFORGE  | combined  |  0.278333 |   0.00666667 |   0.0844444  |   1         |  0.0222222 |      0          |
|      3 | BC-only    | IID       |  0.743889 |   0.591111   |   0.808889   |   1         |  0.575556  |      0.00222222 |
|      3 | BC-only    | noise     |  0.739444 |   0.586667   |   0.82       |   1         |  0.551111  |      0.00222222 |
|      3 | BC-only    | shift     |  0.280556 |   0.0644444  |   0.826667   |   0.164444  |  0.0666667 |      0.00222222 |
|      3 | BC-only    | occlusion |  0.746667 |   0.577778   |   0.813333   |   1         |  0.595556  |      0.00222222 |
|      3 | BC-only    | combined  |  0.265    |   0.0555556  |   0.842222   |   0.12      |  0.0422222 |      0          |
|      3 | Fixed-RL   | IID       |  0.453333 |   0.4        |   0.475556   |   0.497778  |  0.44      |      0          |
|      3 | Fixed-RL   | noise     |  0.441111 |   0.377778   |   0.484444   |   0.48      |  0.422222  |      0          |
|      3 | Fixed-RL   | shift     |  0.204444 |   0.12       |   0.477778   |   0.113333  |  0.106667  |      0.00666667 |
|      3 | Fixed-RL   | occlusion |  0.447222 |   0.395556   |   0.466667   |   0.493333  |  0.433333  |      0          |
|      3 | Fixed-RL   | combined  |  0.185    |   0.108889   |   0.462222   |   0.108889  |  0.06      |      0.00222222 |
|      3 | Uniform-RL | IID       |  0.862778 |   0.713333   |   0.98       |   1         |  0.757778  |      0.00222222 |
|      3 | Uniform-RL | noise     |  0.85     |   0.713333   |   0.977778   |   1         |  0.708889  |      0.00222222 |
|      3 | Uniform-RL | shift     |  0.285556 |   0.102222   |   0.842222   |   0.106667  |  0.0911111 |      0.00666667 |
|      3 | Uniform-RL | occlusion |  0.855556 |   0.695556   |   0.977778   |   1         |  0.748889  |      0.00222222 |
|      3 | Uniform-RL | combined  |  0.279444 |   0.0888889  |   0.848889   |   0.08      |  0.1       |      0          |
|      3 | FAILFORGE  | IID       |  0.857778 |   0.764444   |   0.884444   |   1         |  0.782222  |      0.00222222 |
|      3 | FAILFORGE  | noise     |  0.854444 |   0.771111   |   0.884444   |   1         |  0.762222  |      0.00222222 |
|      3 | FAILFORGE  | shift     |  0.287778 |   0.157778   |   0.762222   |   0.1       |  0.131111  |      0.00888889 |
|      3 | FAILFORGE  | occlusion |  0.852778 |   0.755556   |   0.877778   |   1         |  0.777778  |      0.00222222 |
|      3 | FAILFORGE  | combined  |  0.278333 |   0.148889   |   0.755556   |   0.0777778 |  0.131111  |      0.00222222 |


### C.4 Visual verifier / reward-hacking stress test

|   seed | variant     |   genuine_accuracy |   shortcut_rate |
|-------:|:------------|-------------------:|----------------:|
|      0 | full        |            0.73875 |         0       |
|      0 | no_verifier |            0.615   |         0.21125 |
|      1 | full        |            0.77625 |         0       |
|      1 | no_verifier |            0.4825  |         0.31625 |
|      2 | full        |            0.815   |         0.0025  |
|      2 | no_verifier |            0.5625  |         0.32    |
|      3 | full        |            0.9075  |         0       |
|      3 | no_verifier |            0.6525  |         0.25    |


### C.5 Visual long-horizon summary

| Unnamed: 0   | checkpoint_accuracy   | checkpoint_accuracy.1   | exact_completion     | exact_completion.1   | ge80_completion      | ge80_completion.1     | ge50_completion     | ge50_completion.1   |
|:-------------|:----------------------|:------------------------|:---------------------|:---------------------|:---------------------|:----------------------|:--------------------|:--------------------|
| nan          | mean                  | std                     | mean                 | std                  | mean                 | std                   | mean                | std                 |
| horizon      | nan                   | nan                     | nan                  | nan                  | nan                  | nan                   | nan                 | nan                 |
| 4            | 0.46041666666666664   | 0.04072232327813242     | 0.04722222222222217  | 0.020538030567424863 | 0.04722222222222217  | 0.020538030567424863  | 0.6347222222222222  | 0.07887780308128539 |
| 8            | 0.4579861111111111    | 0.04382342457125259     | 0.005555555555555525 | 0.00453609211626514  | 0.0194444444444444   | 0.01727292417238665   | 0.55                | 0.0920726140344893  |
| 16           | 0.46102430555555557   | 0.04753891541984599     | 0.0                  | 0.0                  | 0.006944444444444425 | 0.0053190394875352095 | 0.45416666666666666 | 0.15994951621253978 |
| 32           | 0.4610677083333333    | 0.03148745145685314     | 0.0                  | 0.0                  | 0.001388888888888875 | 0.0027777777777777497 | 0.41388888888888886 | 0.1349820899566068  |
| 64           | 0.462890625           | 0.04334866418286781     | 0.0                  | 0.0                  | 0.0                  | 0.0                   | 0.33333333333333326 | 0.2568005102362789  |
| 128          | 0.46287977430555555   | 0.041074217087453645    | 0.0                  | 0.0                  | 0.0                  | 0.0                   | 0.3                 | 0.2733348388997982  |


### C.6 Hard-example mixture ablation summary

| variant   |     mean |       std |
|:----------|---------:|----------:|
| hard100   | 0.36381  | 0.0685913 |
| hard50    | 0.360476 | 0.014869  |
| hard75    | 0.328333 | 0.0510369 |
| uniform   | 0.331905 | 0.0668319 |



### C.7 Main matched visual OOD result

The strongest current visual result compares FAILFORGE-HardMine against UniformFactor-RL under
combined visual distribution shift. FAILFORGE-HardMine is better in all four matched seeds.

The correct interpretation is bounded:

- the result supports example-level failure-conditioned selection in this controlled benchmark;
- it does not establish universal dominance over uniform synthetic data;
- it does not imply OSWorld, WeaveBench, MiniWoB, or BrowserGym SOTA.

### C.8 Visual RL-from-scratch collapse

A pure sparse-reward visual RL run from random initialization collapsed toward easy global actions
such as VERIFY/DISMISS rather than acquiring reliable visual grounding. This experiment is retained
as a negative baseline.

The final visual protocol therefore uses:

1. visual-grounding warm start;
2. reward-based synthetic fine-tuning;
3. failure-conditioned example selection.

This is not hidden preprocessing; it is part of the method required to make the RL experiment stable.

### C.9 Task-category reweighting failure

An earlier version of FAILFORGE only adjusted the frequency of broad task categories after failure.
Across multiple seeds this did not reliably outperform UniformFactor-RL.

This finding motivated a stricter version of the hypothesis:

> Failure conditioning must act on concrete examples or environment configurations,
> not only on high-level task labels.

### C.10 Example-level hard mining

The final visual version generates a candidate pool, evaluates the current policy's confidence on the
correct action, and preferentially trains on low-correct-probability examples while preserving random
exploration.

This is the operational implementation of the "failure → counterfactual environment" idea in the
visual benchmark.

### C.11 Hard-example ratio is non-monotonic

The hard-mining fraction is a genuine hyperparameter. The shorter ablation shows that neither a fixed
75% hard mixture nor any single ratio is universally optimal.

The supported conclusion is therefore:

> Failure-conditioned selection can improve robustness, but the balance between failure frontier and
> broad exploration must be calibrated.




## Appendix D — Verifier and Reward-Hacking Analysis

The trajectory verifier is designed to distinguish genuine completion from shortcuts or incomplete
execution.

The visual stress test demonstrates a large qualitative change when verifier correction is removed:
shortcut exploitation increases sharply and genuine performance falls.

This supports separating two questions:

1. **Did the environment emit reward?**
2. **Did the trajectory genuinely satisfy the intended task?**

For computer-use agents, these quantities can diverge substantially.

The verifier therefore checks:

- required intermediate states;
- explicit verification requirements;
- invalid actions;
- recovery events;
- shortcut markers;
- genuine terminal success.

The verifier is used during training reward correction and during evaluation auditing, but it is not
used as an inference-time oracle that chooses actions for the policy.



## Appendix E — Long-Horizon Analysis

The long-horizon visual experiments deliberately distinguish:

- checkpoint/action-level competence;
- partial trajectory completion;
- exact task completion.

A major finding is that checkpoint accuracy can remain approximately stable while exact completion
collapses as horizon grows.

This reflects multiplicative compounding error. If per-step success probability is \(p\), then a crude
independence approximation gives:

\[
P(\text{exact completion at horizon } H) \approx p^H.
\]

Real trajectories are not independent, but the approximation explains why apparently reasonable
local competence is insufficient for reliable long-horizon computer use.

The implication is that future work should introduce:

- memory;
- plan repair;
- hierarchical verification;
- subgoal-level recovery;
- explicit state summaries;
- adaptive replanning.




## Appendix F — Failure Taxonomy

FAILFORGE uses the following failure categories across symbolic and visual settings:

### F.1 Grounding/action-invalid failure
The agent selects an incorrect interface element or emits an invalid interaction.

### F.2 State-tracking failure
The agent loses information or constraints needed later.

### F.3 Missing verification
The agent assumes success without checking a required state.

### F.4 Recovery failure
The agent cannot recover after a pop-up, unexpected state, or interruption.

### F.5 Long-horizon drift
Local actions remain plausible while the global task objective is lost.

### F.6 Reward hacking / shortcut exploitation
The policy exploits an evaluator loophole or apparent reward signal without genuine task completion.

### F.7 Loop/stagnation
The agent repeats ineffective actions without progress.

### F.8 Unclassified failure
A residual category used when failure is observed but does not match an existing mechanism.




## Appendix G — Reproducibility Protocol

### G.1 Seeds

The executed visual main experiment uses four matched seeds.
The BrowserGym public-benchmark protocol is configured for eight seeds for future validation.

### G.2 Reporting

Matched comparisons should report:

- mean;
- standard deviation;
- paired difference;
- number of positive matched seeds;
- paired significance test;
- effect size;
- bootstrap confidence interval where relevant.

### G.3 Integrity rule

Public benchmark scores may only be reported after the corresponding real benchmark environment has
successfully executed.

Mock-environment tests validate the integration logic but are never treated as benchmark evidence.



## Appendix H — BrowserGym / MiniWoB Public-Benchmark Bridge

A public-benchmark bridge was implemented for BrowserGym/MiniWoB with:

- environment discovery;
- seeded reset wrapper;
- failure classifier;
- trajectory verifier;
- adaptive failure sampler;
- evaluation runner;
- task-family configuration;
- CSV logging;
- mock MiniWoB environment;
- core unit tests.

The bridge passed internal mock/control-flow tests.

A real `browsergym-miniwob` installation was attempted in the current execution runtime. The package
could not be installed because outbound DNS/PyPI network access was unavailable. Consequently:

- no real MiniWoB episode was executed;
- no MiniWoB score is reported;
- no BrowserGym performance claim is made.

This boundary is intentional and is part of the reproducibility record.



## Appendix I — Benchmark Card Summary

The visual synthetic benchmark is intended for mechanism-level experiments involving:

- visual robustness;
- recovery;
- verification;
- shortcut detection;
- long-horizon compounding error;
- synthetic curriculum selection.

It is not intended to replace public computer-use benchmarks.

Out-of-scope claims include:

- real-browser production readiness;
- OSWorld SOTA;
- OSWorld2.0 SOTA;
- WeaveBench SOTA;
- MiniWoB SOTA.




## Appendix J — Complete Claim Ledger

### Claims directly supported by executed experiments

1. Example-level failure-conditioned hard mining improves combined visual OOD accuracy over the
   matched UniformFactor-RL baseline in the executed four-seed visual benchmark.
2. Trajectory-aware reward correction strongly suppresses shortcut exploitation in the executed
   visual stress test.
3. Failure-only/task-category curricula are insufficient in the tested settings.
4. Hard-example mixture effects are non-monotonic.
5. Long-horizon exact completion collapses despite relatively stable local/checkpoint competence.
6. Synthetic experience scaling exhibits gains followed by saturation in the symbolic benchmark.
7. The BrowserGym integration logic is implemented and mock-tested.

### Claims not supported and therefore not made

1. Public benchmark SOTA.
2. Frontier-model SOTA.
3. Production reliability.
4. Universal superiority of failure-conditioned training.
5. Universal optimality of any replay or hard-example ratio.




## Appendix K — File Manifest

The complete package includes:

- `FAILFORGE_paper_submission_draft.md`
- `BENCHMARK_CARD.md`
- `STATISTICAL_ROBUSTNESS_REPORT.md`
- `FAILURE_ANALYSIS_ATLAS.md`
- `ALGORITHM_SECTION.md`
- `REVIEWER_ATTACK_AND_RESPONSE.md`
- `REPRODUCIBILITY_STATEMENT.md`
- `HIRING_FACING_SUMMARY.md`
- `manifest.json`
- `run_all.sh`
- symbolic raw/summary CSVs;
- visual raw/summary CSVs;
- paper-ready plots;
- synthetic benchmark screenshots;
- BrowserGym bridge code and tests;
- public-benchmark checklist.

This appendix is designed so that no executed experimental family is omitted from the paper record.

