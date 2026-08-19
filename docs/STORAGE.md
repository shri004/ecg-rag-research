# Storage Strategy

## Overview
- **GitHub** — code, docs, configs, small reproducible results (<1MB). Source of truth for everything here.
- **Google Drive (shared folder, college account)** — working cache for large regenerable artifacts: PTB-XL raw + processed data, model checkpoints, guideline KB embeddings, vector DB. NOT source of truth — everything here should be reproducible by re-running the scripts in this repo against the manifest below.
- **Colab** — GPU training/inference, mounts the shared Drive folder.
- **Local Ollama** — RAG/LLM inference (kept local deliberately — this is part of the research design, not just infra convenience).
- **Zenodo** — permanent, citable archival of *final* artifacts only, done once at project end (or per major milestone if useful sooner). Free, CERN-operated, DOI-issuing, up to 50GB per record by default. This is what goes in the paper's data-availability section — not a Drive link, which can break after account deactivation.

## Shared Drive folder
- **Link:** _[paste shared folder link here once created]_
- **Access:** both team members as Editor, not personal "My Drive" — avoids divergent copies.
- **Structure:** mirror the repo layout, e.g.:
  ```
  ECG-RAG-Project-Drive/
  ├── data/ptbxl/                 (raw + processed/)
  ├── checkpoints/                 (best_model_*.pt)
  ├── kb/embeddings/
  └── vectordb/
  ```

## Artifact manifest
Log every checkpoint/large artifact here — this file lives in git (it's small text), the actual files live on Drive. Update this whenever you save something worth keeping.

| Artifact | Drive path | Git commit | Date | Key metric | Notes |
|---|---|---|---|---|---|
| _example_ `best_model_v1.pt` | `checkpoints/best_model_v1.pt` | `a1b2c3d` | 2026-08-20 | test macro-AUC 0.91 | Module 1 baseline, ResNet1DWang, 30 epochs |

## Colab checkpointing rule
Always save checkpoints to the **mounted Drive path**, not Colab's local disk — free-tier sessions disconnect (~12hr limit, no GPU guarantee), and anything only on local disk is lost on disconnect.

```python
from google.colab import drive
drive.mount('/content/drive')
CHECKPOINT_DIR = '/content/drive/MyDrive/ECG-RAG-Project-Drive/checkpoints/'
```

## End-of-project archival (Zenodo)
Before the project concludes (or before submitting anywhere), archive:
- Final trained model checkpoint(s)
- Guideline KB (chunks + source citations — check each guideline source's redistribution license first; some may only allow citation, not redistribution of full text)
- Vector DB snapshot (optional — regenerable, but convenient for reviewers)

Steps: create a Zenodo account (GitHub/ORCID login works), upload as a new dataset record, fill in metadata (title, authors, description, license), publish to get a DOI. Cite this DOI in the paper's data-availability statement, alongside the standard PTB-XL/PhysioNet citation for the original dataset (do not re-upload PTB-XL itself to Zenodo — link to the official PhysioNet source instead; you're archiving what *you* produced, not redistributing the source dataset).

## What never goes on Drive or anywhere outside your ethics-approved process
Clinician-provided raw annotations / any real patient data — keep local only, per `docs/GIT_WORKFLOW.md` and your institution's data handling requirements. Only aggregated, de-identified summary results go in the repo/Drive.
