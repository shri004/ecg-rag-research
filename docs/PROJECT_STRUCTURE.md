# Project Structure

```
ecg_rag_project/
├── README.md               Project overview, quickstart
├── requirements.txt         Python dependencies (pinned)
├── .gitignore
├── docs/                    This folder — planning + workflow docs
│   ├── GIT_WORKFLOW.md      Git/GitHub collaboration rules
│   ├── PROJECT_PLAN.md      5-week module-by-module plan
│   └── MODULE1_GUIDE.md     Step-by-step Module 1 instructions
│
├── configs/                 Config files (hyperparameters, paths) — no secrets
│
├── data/                    Datasets. NOT committed to git (see .gitignore)
│   └── ptbxl/                PTB-XL downloads here (raw + processed/)
│
├── src/                     Module 1: ECG model code
│   ├── data_prep.py          PTB-XL download + preprocessing
│   ├── model.py              ResNet1DWang architecture
│   └── train.py              Training loop + evaluation
│
├── kb/                      Module 2: medical knowledge base
│   ├── raw_guidelines/        Source guideline documents + sources.md
│   │                          (citation list — check redistribution license
│   │                          per source before committing full PDFs)
│   ├── chunks/                Chunked/cleaned text ready for embedding
│   └── embeddings/            Generated embeddings — NOT committed (regenerable)
│
├── vectordb/                Module 2: vector database files — NOT committed
│                             (regenerable by re-running the embedding script)
│
├── rag/                     Module 3: RAG pipeline
│   ├── retriever.py           Embedding + retrieval logic
│   ├── ollama_client.py       Local LLM calls via Ollama
│   └── pipeline.py            End-to-end: prediction -> retrieval -> explanation
│
├── eval/                    Module 3-4: faithfulness + clinician evaluation
│   ├── automated/             RAGAS-style / automated faithfulness scoring
│   └── clinician/              Clinician evaluation forms, aggregated (non-PHI)
│                               results only — raw annotations stay local,
│                               never committed (see .gitignore)
│
├── results/                 Output metrics, tables, plots per module
│   ├── module1/                Classifier metrics (AUC, confusion matrices)
│   ├── module2/                KB/retrieval quality checks
│   ├── module3/                Faithfulness scores, RAG vs non-RAG numbers
│   └── module4/                Clinician evaluation summary (aggregated)
│
├── gui/                     Module 5: demo interface
│   ├── backend/                FastAPI app
│   └── frontend/               React frontend
│
├── paper/                   Module 5: writeup
│   └── figures/                Final figures for the paper
│
├── notebooks/                Exploration only — not pipeline code (see
│                              docs/GIT_WORKFLOW.md re: notebook diffs)
│
└── tests/                    Basic sanity tests (e.g. shape checks, smoke tests)
```

## Why this shape

- **One folder per module** (`src/` = Module 1, `kb/`+`vectordb/` = Module 2,
  `rag/` = Module 3, `eval/` = Module 3-4, `gui/`+`paper/` = Module 5) so the
  repo structure mirrors your actual project timeline. Easy to point a
  reader (or examiner) at "this is where Module 3 lives."
- **`results/` is separate from code** — keeps generated output away from
  source, and makes it obvious what to screenshot/cite in the paper.
- **Nothing large or licensed lives in git** — data, embeddings, vector DB,
  and model checkpoints are all regenerable by re-running scripts against
  the same inputs. This is what makes the repo *reproducible* rather than
  just a code dump — anyone with PTB-XL access and this repo should be able
  to reproduce your results end to end.
