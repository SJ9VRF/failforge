# FAILFORGE Algorithm Section

## Objective

Train computer-use agents whose future training distribution adapts to their verified failures.

## Definitions

- `πθ`: current computer-use policy.
- `E`: environment family.
- `τ`: trajectory.
- `V(τ)`: trajectory verifier score.
- `C(τ)`: failure classifier.
- `B`: replay buffer of mastered or partially mastered examples.
- `F`: failure memory.

## Training distribution

\[
p_{train}(E)=λ_f p(E|F)+λ_u p_{uniform}(E)+λ_r p_{replay}(E|B)
\]

where \(λ_f+λ_u+λ_r=1\).

## Pseudocode

```text
Initialize policy πθ
Initialize replay buffer B
Initialize failure memory F

for round t in 1..T:
    Sample candidate environments C_t
    Score candidate difficulty under πθ
    Select mixture:
        hard examples from failure frontier
        uniform exploration examples
        replay examples from B

    for each environment E in mixture:
        Roll out πθ
        Compute terminal reward
        Compute trajectory-aware verifier score V(τ)
        Diagnose failures C(τ)
        Store success/partial success in B
        Store failures in F

    Update πθ using verified reward
```

## Why each component exists

### Failure frontier

Targets the agent's current weaknesses.

### Uniform exploration

Prevents overfitting to a narrow set of failures.

### Replay

Preserves mastered capabilities and reduces catastrophic forgetting.

### Verifier

Prevents shortcut exploitation and outcome-only reward hacking.

### Failure classifier

Turns raw trajectories into structured training signals.

## Complexity

If each round samples \(N\) candidate environments, executes \(K\) selected rollouts,
and each rollout has horizon \(H\), then environment interaction cost is:

\[
O(KH)
\]

and candidate scoring cost is:

\[
O(N)
\]

for lightweight policies or \(O(NC_\pi)\) when policy inference cost \(C_\pi\) dominates.

## Implementation note

The current visual benchmark implements example-level hard mining:
candidate screenshots are scored by the current policy, and examples with low probability
assigned to the correct action are upweighted.
