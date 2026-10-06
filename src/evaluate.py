import os
import json
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    multilabel_confusion_matrix,
)

from src.model import ResNet1d


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = "/kaggle/working/processed"
CHECKPOINT = "/kaggle/working/ecg-rag-research/checkpoints/resnet_best.pt"
OUT_DIR = "/kaggle/working/ecg-rag-research/results/module1"

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# Load test data
# --------------------------------------------------

X_test = np.load(
    os.path.join(DATA_DIR, "X_test.npy"),
    mmap_mode="r"
)

y_test_raw = pd.read_pickle(
    os.path.join(DATA_DIR, "y_test.pkl")
)

print("Test ECG shape:", X_test.shape)
print("Number of test records:", len(y_test_raw))


# --------------------------------------------------
# Convert labels
# --------------------------------------------------

mlb = MultiLabelBinarizer(classes=CLASSES)
mlb.fit([CLASSES])

y_test = mlb.transform(y_test_raw).astype(np.float32)

print("Label shape:", y_test.shape)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class ECGTestDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        # Original shape: (T, C)
        # Model expects:   (C, T)

        x = torch.from_numpy(
            np.asarray(self.X[idx], dtype=np.float32)
        ).permute(1, 0)

        y = torch.from_numpy(self.y[idx])

        return x, y


test_dataset = ECGTestDataset(X_test, y_test)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = ResNet1d(
    in_channels=X_test.shape[2],
    num_classes=len(CLASSES)
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

model.load_state_dict(checkpoint)
model.to(device)
model.eval()

print("Checkpoint loaded successfully.")


# --------------------------------------------------
# Test inference
# --------------------------------------------------

all_probs = []
all_targets = []

with torch.no_grad():

    for batch_idx, (xb, yb) in enumerate(test_loader):

        xb = xb.to(device, non_blocking=True)

        logits = model(xb)

        probs = torch.sigmoid(logits)

        all_probs.append(
            probs.cpu().numpy()
        )

        all_targets.append(
            yb.numpy()
        )

        if (batch_idx + 1) % 10 == 0:
            print(
                f"Processed "
                f"{min((batch_idx + 1) * 64, len(test_dataset))}"
                f"/{len(test_dataset)} test records"
            )


y_prob = np.concatenate(all_probs, axis=0)
y_true = np.concatenate(all_targets, axis=0)

y_pred = (y_prob >= 0.5).astype(int)

print("Inference complete.")
print("Predictions shape:", y_prob.shape)


# --------------------------------------------------
# Overall metrics
# --------------------------------------------------

class_auc = {}

for i, class_name in enumerate(CLASSES):

    try:
        class_auc[class_name] = float(
            roc_auc_score(
                y_true[:, i],
                y_prob[:, i]
            )
        )
    except ValueError:
        class_auc[class_name] = None


valid_aucs = [
    value for value in class_auc.values()
    if value is not None
]

macro_auc = float(np.mean(valid_aucs))

macro_f1 = float(
    f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )
)

micro_f1 = float(
    f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )
)

macro_precision = float(
    precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )
)

macro_recall = float(
    recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )
)


# --------------------------------------------------
# Class-wise metrics
# --------------------------------------------------

class_f1 = f1_score(
    y_true,
    y_pred,
    average=None,
    zero_division=0
)

class_precision = precision_score(
    y_true,
    y_pred,
    average=None,
    zero_division=0
)

class_recall = recall_score(
    y_true,
    y_pred,
    average=None,
    zero_division=0
)

class_metrics = {}

for i, class_name in enumerate(CLASSES):

    class_metrics[class_name] = {
        "precision": float(class_precision[i]),
        "recall": float(class_recall[i]),
        "f1": float(class_f1[i]),
        "auroc": class_auc[class_name]
    }


# --------------------------------------------------
# Confusion matrices
# --------------------------------------------------

cm = multilabel_confusion_matrix(
    y_true,
    y_pred
)

confusion_matrices = {}

for i, class_name in enumerate(CLASSES):

    confusion_matrices[class_name] = cm[i].tolist()


# --------------------------------------------------
# Save predictions
# --------------------------------------------------

np.save(
    os.path.join(OUT_DIR, "test_probabilities.npy"),
    y_prob
)

np.save(
    os.path.join(OUT_DIR, "test_predictions.npy"),
    y_pred
)

np.save(
    os.path.join(OUT_DIR, "test_targets.npy"),
    y_true
)


# --------------------------------------------------
# Save evaluation summary
# --------------------------------------------------

results = {
    "dataset": "PTB-XL",
    "split": "official test split",
    "num_test_records": int(len(y_true)),
    "classes": CLASSES,
    "checkpoint": CHECKPOINT,

    "overall": {
        "macro_auroc": macro_auc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall
    },

    "class_wise": class_metrics,

    "confusion_matrices": confusion_matrices
}


summary_path = os.path.join(
    OUT_DIR,
    "evaluation_summary.json"
)

with open(summary_path, "w") as f:
    json.dump(results, f, indent=2)


# --------------------------------------------------
# Print final results
# --------------------------------------------------

print("\n" + "=" * 55)
print("FINAL TEST RESULTS")
print("=" * 55)

print(f"Macro AUROC : {macro_auc:.4f}")
print(f"Macro F1    : {macro_f1:.4f}")
print(f"Micro F1    : {micro_f1:.4f}")
print(f"Macro Prec. : {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")

print("\nClass-wise results:")

for class_name in CLASSES:

    m = class_metrics[class_name]

    print(
        f"{class_name:5s} | "
        f"Precision={m['precision']:.4f} | "
        f"Recall={m['recall']:.4f} | "
        f"F1={m['f1']:.4f} | "
        f"AUROC={m['auroc']:.4f}"
    )

print("\nSaved:")
print(summary_path)