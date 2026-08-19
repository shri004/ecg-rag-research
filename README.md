# ECG-RAG Faithfulness Project

Research question: Does retrieval-augmented generation (RAG) with local/open LLMs
(via Ollama) produce more faithful, clinician-verifiable explanations for ECG
diagnoses than non-RAG generation — and how faithful are local models, period?

## Module map
1. `src/data_prep.py` — download + preprocess PTB-XL
2. `src/model.py` — **ResNet1D-Wang**, a faithful reproduction of:
   Wang, Z., Yan, W., Oates, T. (2017). "Time Series Classification from
   Scratch with Deep Neural Networks: A Strong Baseline." IJCNN 2017.
   (3 residual blocks, filters {64,128,128}, kernels [8,5,3] per block —
   verified against the original paper text and an independent
   reproduction citing the same source). This is the architecture
   Strodthoff et al. (2021, IEEE JBHI) benchmark as `resnet1d_wang` on
   PTB-XL.
   **Target baseline: macro-AUC 0.930 (±0.05) on the diagnostic-superclass
   task**, per github.com/helme/ecg_ptbxl_benchmarking leaderboard.
3. `src/train.py` — training loop
4. (Module 2, next) — knowledge base + embeddings + vector DB
5. (Module 3) — Ollama + RAG pipeline
6. (Module 4) — faithfulness eval harness

## Environment setup (run locally — this needs PhysioNet access + a GPU ideally)

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Getting PTB-XL

PTB-XL is open-access on PhysioNet (no credentialing needed, unlike MIMIC-IV-ECG),
but you must still follow the PhysioNet Credentialed/Open Data License and cite:

- Wagner et al. 2020, "PTB-XL, a large publicly available electrocardiography
  dataset," Scientific Data.
- Strodthoff et al. 2021, "Deep Learning for ECG Analysis: Benchmarks and
  Insights from PTB-XL," IEEE JBHI.

```bash
python src/data_prep.py --download --data-dir ./data/ptbxl
```

This downloads ~3GB. If bandwidth/storage is a problem on your machine, use
Google Colab (mount Drive for persistence) — see `notebooks/` once we add one.

## Sanity check before trusting results

Once trained, compare your macro-AUC against the official benchmark leaderboard
here: https://github.com/helme/ecg_ptbxl_benchmarking — if you're wildly off,
something in preprocessing/split is wrong, not necessarily your architecture.
