# FAILFORGE

**Failure-Conditioned Synthetic Reinforcement Learning for Computer-Use Agents**

Research-first project website and reproducibility package for FAILFORGE.

## Website

The site is fully static. Open `index.html` locally, or deploy the repository directly with GitHub Pages.

### GitHub Pages

1. Create a new GitHub repository.
2. Upload all files from this folder to the repository root.
3. In GitHub, open **Settings → Pages**.
4. Under **Build and deployment**, select **Deploy from a branch**.
5. Select branch `main` and folder `/ (root)`.
6. Save.

The page should publish automatically.

## Visual system

### Fonts
- **Inter** — headings and body text.
- **IBM Plex Mono** — metrics, labels, code, and research metadata.
- System fallbacks are included if Google Fonts are unavailable.

### Colors
- Background: `#F6F8FB`
- Primary ink: `#0B1220`
- Research blue: `#195C8C`
- Warm evidence gold: `#B78322`
- Muted text: `#647086`
- Success: `#267354`
- Failure / warning: `#A44B4B`

The palette is intentionally restrained: research lab / paper-first, not startup/crypto styling.

## Repository structure

```text
.
├── index.html
├── styles.css
├── script.js
├── 404.html
├── site.webmanifest
├── robots.txt
├── .nojekyll
├── .gitignore
├── CITATION.cff
├── README.md
├── DEPLOY.md
├── PROJECT_STRUCTURE.md
├── assets/
│   ├── failforge_demo.gif
│   ├── architecture.svg
│   ├── agent_loop.svg
│   ├── trajectory_case_study.png
│   ├── visual_ood_main.png
│   ├── visual_ood_profile.png
│   ├── visual_long_horizon.png
│   ├── visual_verifier_ablation.png
│   ├── hard_mining_ablation.png
│   ├── visual_benchmark_samples.png
│   ├── og-preview.png
│   └── favicon.png
└── downloads/
    ├── paper
    ├── technical reports
    ├── reproducibility artifacts
    ├── code packages
    └── experiment bundles
```

## Research integrity

The page only reports experiments that were actually executed in the included project artifacts.

It does **not** claim OSWorld, WeaveBench, BrowserGym, or MiniWoB SOTA.

The BrowserGym/MiniWoB integration is implemented and mock-tested, but real public benchmark scores were not produced in the original execution runtime because browser/network dependencies could not be installed there.

## Main executed visual result

- UniformFactor-RL combined visual OOD: **43.3%**
- FAILFORGE-HardMine: **49.2%**
- Matched improvement: **+5.9 percentage points**
- 4/4 matched seeds positive

Verifier stress:
- shortcut rate with verifier: approximately **0.1%**
- shortcut rate without verifier: approximately **27.4%**

See the full paper and statistical report in `downloads/`.

## Local preview

With Python installed:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

## Publishing the code separately

The current repo can be used as both the project page and downloadable research package. If you later create a separate source-code repository, replace the current **Code package** CTA with the public GitHub URL.

## License

No open-source license is asserted by this website package. Add the license you want before redistributing code as an open-source project.
