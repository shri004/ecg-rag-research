from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]

SUMMARY_PATH = (
    ROOT / "results" / "module2" / "runs"
    / "ecg_clean_v1" / "metrics_summary_heldout.json"
)

OUTPUT_DIR = ROOT / "results" / "module2" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load actual experiment results
with SUMMARY_PATH.open("r", encoding="utf-8") as f:
    results = json.load(f)

summary = results["summary"]

# Figure 1: Overall retrieval comparison
arms = ["dense", "dense_clean", "hybrid_clean"]
labels = ["Dense", "Dense (clean)", "Hybrid (clean)"]
metrics = [
    ("p_strict", "Strict Precision@3"),
    ("hit", "Hit@3"),
    ("mrr", "MRR@3"),
    ("ndcg", "nDCG@3"),
]

x = np.arange(len(metrics))
width = 0.24

fig, ax = plt.subplots(figsize=(10, 6))
for i, (arm, label) in enumerate(zip(arms, labels)):
    values = [summary[arm][key] for key, _ in metrics]
    bars = ax.bar(x + (i - 1) * width, values, width, label=label)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

ax.set_title("ECG Knowledge Retrieval: Overall Performance")
ax.set_ylabel("Metric score")
ax.set_xticks(x)
ax.set_xticklabels([name for _, name in metrics])
ax.set_ylim(0, 1.15)
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "figure1_retrieval_metrics.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 2: Strict Precision@3 by ECG superclass
classes = ["NORM", "MI", "STTC", "CD", "HYP"]
class_arms = ["dense_clean", "hybrid_clean"]
class_labels = ["Dense (clean)", "Hybrid (clean)"]

x = np.arange(len(classes))
width = 0.34

fig, ax = plt.subplots(figsize=(9, 5.5))
for i, (arm, label) in enumerate(zip(class_arms, class_labels)):
    values = [
        summary[arm]["by_class"][cls]["p_strict"]
        for cls in classes
    ]
    bars = ax.bar(x + (i - 0.5) * width, values, width, label=label)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

ax.set_title("Strict Precision@3 by ECG Superclass")
ax.set_ylabel("Strict Precision@3")
ax.set_xlabel("ECG superclass")
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylim(0, 1.15)
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "figure2_classwise_precision.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 3: Junk and garbled-text rates
quality_metrics = [
    ("junk", "Junk content"),
    ("garbled", "Garbled text"),
]

x = np.arange(len(quality_metrics))
width = 0.24

fig, ax = plt.subplots(figsize=(8, 5.5))
for i, (arm, label) in enumerate(zip(arms, labels)):
    values = [summary[arm][key] for key, _ in quality_metrics]
    bars = ax.bar(x + (i - 1) * width, values, width, label=label)
    ax.bar_label(
        bars,
        labels=[f"{value:.1%}" for value in values],
        padding=3,
        fontsize=9,
    )

ax.set_title("Retrieved-Content Quality")
ax.set_ylabel("Rate")
ax.set_xticks(x)
ax.set_xticklabels([name for _, name in quality_metrics])
max_rate = max(
    summary[arm][key]
    for arm in arms
    for key, _ in quality_metrics
)
ax.set_ylim(0, max_rate * 1.4 + 0.02)
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "figure3_retrieval_quality.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nFigures saved in: {OUTPUT_DIR}")
for path in sorted(OUTPUT_DIR.glob("figure*.png")):
    print(f" - {path.name}")
