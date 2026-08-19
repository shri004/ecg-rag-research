# Git Workflow — Two-Person Team

## Branch structure
- `main` — always working. Only merged via reviewed pull request (PR).
- `dev` — integration branch. Both of you merge feature branches here first.
- `feature/<short-name>` — one branch per task, e.g. `feature/ptbxl-preprocessing`,
  `feature/resnet-training`, `feature/vector-db`.

Flow: `feature/x` → PR into `dev` → (weekly) `dev` → PR into `main`.

## Daily workflow
1. `git pull origin dev` before starting work each day.
2. Create/switch to your feature branch: `git checkout -b feature/your-task`.
3. Commit small, working chunks — not one giant commit at the end of the day.
4. Push your branch: `git push origin feature/your-task`.
5. Open a PR into `dev`. The other person reviews before merging — even a
   quick "looks fine" comment. This is what makes the repo trustworthy for
   your paper's reproducibility claims later.

## Commit message convention
`<module>: <short description>`
Examples:
- `module1: add PTB-XL preprocessing script`
- `module2: fix chunking overlap bug in guideline splitter`
- `module3: add RAGAS faithfulness scorer`

## Splitting work between two people
Suggested split once Module 1 is done (adjust as you go):
- **Person A**: ECG model training/evaluation (Module 1), later GUI backend (Module 5)
- **Person B**: Knowledge base + RAG + Ollama pipeline (Module 2-3), later faithfulness eval (Module 3-4)
Both: clinician evaluation coordination (Module 4), paper writing (Module 5).

Work in different files/modules when possible to avoid merge conflicts.
If you must edit the same file, communicate first (a quick message beats
a merge conflict).

## Notebooks and merge conflicts
Jupyter notebooks (.ipynb) diff badly in git (output cells cause noise).
Rules:
- Clear all outputs before committing: Kernel → Restart & Clear Output.
- Prefer `.py` scripts for anything that becomes a pipeline component;
  keep notebooks for exploration only.
- Optional but recommended: install `nbstripout` to auto-strip outputs on
  commit (`pip install nbstripout && nbstripout --install`).

## What to commit vs. NOT commit

| Commit | Do NOT commit |
|---|---|
| Source code (`src/`, `rag/`, `eval/`, `gui/`) | Raw/processed PTB-XL data (`data/`) — license terms + size |
| `requirements.txt`, configs (no secrets) | Model checkpoints (`.pt` files) — large, regenerable from code |
| Small result summaries (CSV metrics, plots under ~1MB) | Vector DB files (`vectordb/`) — regenerable from `kb/raw_guidelines/` |
| Documentation (`docs/`, `README.md`) | `.env` files, API keys, any secrets |
| Guideline source citations/metadata (`kb/raw_guidelines/sources.md`) | Full copyrighted guideline PDFs if license restricts redistribution — check each source's terms |
| Paper drafts, figures (`paper/`) | Clinician-provided raw annotations (PHI/privacy risk) — keep local only, share via secure channel your ethics process approves |

**Rule of thumb:** if it's regenerable by running a script, or if it's large/licensed/private data, it doesn't belong in git. If someone cloning the repo needs it to reproduce your results without re-running everything, it's a candidate — but check size and license first.

## Model checkpoints
Don't commit `best_model.pt`. Options if you need to share a trained model:
- Google Drive / shared folder link, referenced in `docs/` or `README.md`.
- If you specifically need versioned large-file tracking, look into Git LFS
  later — not necessary for a 5-week project, adds complexity you don't need yet.
