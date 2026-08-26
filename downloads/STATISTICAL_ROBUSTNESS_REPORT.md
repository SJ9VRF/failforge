# Statistical Robustness Report

## Main matched comparison

Combined visual OOD accuracy:

|   seed |   FAILFORGE-HardMine |   UniformFactor-RL |
|-------:|---------------------:|-------------------:|
|      0 |               0.495  |             0.435  |
|      1 |               0.4835 |             0.3815 |
|      2 |               0.57   |             0.525  |
|      3 |               0.419  |             0.3905 |

Mean difference FAILFORGE-HardMine − UniformFactor-RL:

\[
0.0589
\]

Percentage points:

\[
+5.89
\]

Positive matched seeds:

\[
4/4
\]

Paired p-value:

\[
p=0.0334
\]

Cohen's dz:

\[
d_z=1.869
\]

Bootstrap 95% CI for mean difference:

\[
[3.64, 8.78] \text{ percentage points}
\]

## Interpretation

The effect is positive in every matched seed and statistically significant under a paired test,
but the sample size remains small. The result should be framed as controlled evidence, not a
final public benchmark claim.

## Multiple comparisons

The main hypothesis test should be pre-registered as:

> FAILFORGE-HardMine improves combined visual OOD accuracy over UniformFactor-RL.

Other OOD views and ablations should be interpreted as exploratory unless corrected.

## Power note

With only four seeds, p-values are sensitive. The effect size is large, but future versions
should run 8–12 seeds before submission to a top-tier main conference.

## Verifier stress

| variant     |   genuine_accuracy |   shortcut_rate |
|:------------|-------------------:|----------------:|
| full        |             0.8094 |          0.0006 |
| no_verifier |             0.5781 |          0.2744 |

The verifier result is mechanistically strong because the shortcut-rate increase is large
and aligned with the expected failure mode.
