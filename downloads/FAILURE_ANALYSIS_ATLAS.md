# FAILFORGE Failure-Analysis Atlas

## Case 1 — Successful visual grounding

The agent receives a goal token and selects the corresponding widget location.
This tests low-level multimodal binding between instruction and screenshot.

**Training consequence:** successful examples may enter replay to preserve competence.

## Case 2 — Layout-shift failure

The target widget type remains the same, but location statistics change.
A model that memorizes position priors fails.

**FAILFORGE response:** increase counterfactual layouts around low-probability correct-action examples.

## Case 3 — Pop-up recovery failure

A modal interruption blocks the intended interaction.
The agent must dismiss or handle the modal before continuing.

**FAILFORGE response:** synthesize more recovery states and penalize unrecovered trajectories.

## Case 4 — Missing verification

The environment requires an explicit VERIFY action after apparent completion.
Outcome-only reward can miss this.

**FAILFORGE response:** verifier adds trajectory-level reward correction and failure tag `missing_verification`.

## Case 5 — Shortcut exploitation

The interface contains a tempting shortcut action that can produce apparent reward without genuine task completion.

**Observed result:** shortcut rate rose from approximately 0.1% to 27.4% when verifier correction was removed.

**FAILFORGE response:** classify as reward hacking, penalize, and forge more shortcut-resistant environments.

## Case 6 — Long-horizon drift

The agent remains locally competent but exact completion collapses as horizon increases.

**FAILFORGE response:** treat long-horizon drift as a separate failure mode; future work should add memory and hierarchical verification.
