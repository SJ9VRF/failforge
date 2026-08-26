# Reproducibility Statement

## What has been executed

1. Symbolic FAILFORGE benchmark.
2. Visual synthetic desktop benchmark.
3. Visual verifier stress test.
4. Long-horizon visual scaling.
5. Hard-mining ablation.
6. BrowserGym bridge mock execution and core unit tests.

## What has not been executed

Real MiniWoB/BrowserGym benchmark scoring has **not** been executed in this runtime because
`browsergym-miniwob` and browser dependencies could not be installed due to unavailable
network/DNS access in the execution container.

## Exact claims allowed

Allowed:

- FAILFORGE-HardMine improves combined visual OOD accuracy over UniformFactor-RL in the executed synthetic visual benchmark.
- Trajectory-aware verification suppresses shortcut exploitation in the executed visual stress test.
- Local checkpoint accuracy does not guarantee exact long-horizon completion.
- BrowserGym/MiniWoB bridge is implemented and mock-tested.

Not allowed:

- OSWorld SOTA.
- OSWorld2.0 SOTA.
- WeaveBench SOTA.
- MiniWoB SOTA.
- Real-browser benchmark performance.
- Production computer-use reliability.

## Main statistics

- UniformFactor-RL combined visual OOD: **43.30%**
- FAILFORGE-HardMine combined visual OOD: **49.19%**
- Difference: **+5.89 pp**
- Positive seeds: **4/4**
- Paired p-value: **0.0334**
- Cohen's dz: **1.869**

Verifier stress:

- Full verifier shortcut rate: **0.1%**
- No-verifier shortcut rate: **27.4%**

## Seed protocol

The main visual experiment used four seeds. BrowserGym protocol file specifies eight seeds
for future public benchmark validation.

## File provenance

All raw CSVs and figures are included under:

- `failforge_visual/`
- `failforge_package/`
- `failforge_browsergym_bridge/`

## Re-run guide

For existing synthetic results, inspect the CSVs and scripts in `failforge_visual/`.

For BrowserGym/MiniWoB:

```bash
pip install browsergym-miniwob
playwright install chromium
export MINIWOB_URL=<your MiniWoB URL>
cd failforge_browsergym_bridge
PYTHONPATH=. python scripts/run_miniwob_failforge.py --episodes 100 --seed 0
```

Only after this command successfully runs should MiniWoB performance be reported.
