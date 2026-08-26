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
