# Deployment Guide

## Option 1 — GitHub Pages

This repository is already prepared for GitHub Pages.

1. Push all files to `main`.
2. Open **Settings → Pages**.
3. Choose **Deploy from a branch**.
4. Select `main` and `/ (root)`.
5. Save.

`.nojekyll` is included so GitHub Pages serves the static files directly.

## Option 2 — Existing personal site

Copy the whole folder under a route such as:

```text
/projects/failforge/
```

All asset and download links are relative, so the site works from a subdirectory.

## Custom domain

If you later have a domain, add a file named `CNAME` containing only the domain, for example:

```text
failforge.example.com
```

Do not add a placeholder CNAME before the real domain exists.

## Before publishing

- Verify `index.html` loads.
- Click Paper, Code package, Technical report, and Demo.
- Confirm large ZIP files fit the hosting provider's file-size limits.
- If GitHub rejects a large file, move that bundle to GitHub Releases and replace its link in `index.html`.
- Add the final public GitHub URL to the Code button once the repository exists.
