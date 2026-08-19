# Project Plan — Module by Module (5 weeks)

Research question: Does RAG with local/open LLMs (via Ollama) produce more
faithful, clinician-verifiable ECG explanations than non-RAG generation —
and how faithful are local models at this task, period?

---

## Module 1 — ECG classifier baseline (Week 1)
**Goal:** working, verified classifier — infrastructure, not the paper's contribution.
- Download + preprocess PTB-XL (5-class diagnostic superclass task)
- Train `ResNet1DWang` (faithful reproduction, see `src/model.py`)
- Target: macro-AUC ≈ 0.92-0.93 on the official fold-10 test set
- **Output:** trained checkpoint, `results/module1/metrics.csv`, confusion matrix
- **Done when:** your macro-AUC is within a reasonable range of the published
  0.930 (±0.05) benchmark. Large gap → check preprocessing/split first, not architecture.

## Module 2 — Medical knowledge base + embeddings + vector DB + Ollama (Week 2)
**Goal:** a retrievable, cited store of cardiology guideline knowledge, and Ollama running locally.
- Curate guideline source excerpts (AHA/ESC or similar) per diagnostic superclass — properly cited, not scraped indiscriminately
- Chunk + clean text (`kb/chunks/`)
- Generate embeddings (sentence-transformers or a medical-domain embedding model), store in a vector DB (Chroma recommended — simple, local, matches your stack)
- Install Ollama locally, pull 2-3 open models to compare later (e.g. a general one + a medical-tuned one if available)
- **Output:** working `retriever.py` that returns top-k relevant guideline chunks for a diagnosis
- **Done when:** you can query the vector DB with a diagnosis label and get back sensible, correctly-cited guideline text

## Module 3 — RAG pipeline + explanation generation + faithfulness eval harness (Week 3)
**Goal:** the core of your contribution.
- Wire pipeline: ECG prediction → retrieve relevant guideline chunks → Ollama generates explanation grounded in retrieved evidence
- Build faithfulness eval harness: does the generated explanation's claims actually match the retrieved evidence? (adapt RAGAS-style claim-checking, or build a simpler claim-vs-source overlap checker if RAGAS proves too heavy)
- Run on 2-3 local models via Ollama
- **Output:** per-model, per-case faithfulness scores; example explanations with retrieved evidence shown side by side
- **Done when:** you can generate an explanation for a test case and automatically score whether it's grounded in what was retrieved

## Module 4 — RAG vs non-RAG comparison + clinician evaluation (Week 4)
**Goal:** the actual experimental result.
- Run the same test cases through RAG and non-RAG (direct LLM, no retrieval) pipelines
- Compare faithfulness scores, hallucination rates, across both conditions and across models
- Get 5-15 cases reviewed by a clinician — do they agree with your automated faithfulness metric? This validates (or challenges) your automated scoring, it doesn't need to be a large formal study
- **Output:** `results/module4/rag_vs_nonrag.csv`, clinician agreement summary
- **Done when:** you have a defensible RAG-vs-non-RAG comparison table backed by both automated metrics and a clinician sanity-check

## Module 5 — Analysis, GUI/API, paper writing (Week 5)
**Goal:** package it.
- Analyze results, make final figures/tables for `paper/`
- Build minimal FastAPI backend + simple frontend demo (`gui/`) — presentation only, don't over-invest here
- Write up methods/results — this is where your literature review (30-40 papers) and refined gap statement get used
- **Output:** paper draft, working demo, presentable repo

---

## Time-budget reality check
Weeks 1-2 are infrastructure (necessary, not the contribution).
Weeks 3-4 are where your actual paper lives — protect this time.
If anything slips, cut GUI polish (Module 5) before cutting faithfulness
rigor (Module 3-4).
