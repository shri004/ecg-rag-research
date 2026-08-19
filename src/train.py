"""
Train ResNet1d on PTB-XL diagnostic superclass task.

Usage:
    python train.py --data-dir ./data/ptbxl/processed --epochs 30
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, TensorDataset

from model import ResNet1d

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_split(data_dir, split):
    X = np.load(os.path.join(data_dir, f"X_{split}.npy"))
    y = __import__("pandas").read_pickle(os.path.join(data_dir, f"y_{split}.pkl"))
    return X, y


def to_tensors(X, y_labels, mlb):
    # X: (N, T, C) from wfdb -> model expects (N, C, T)
    X_t = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
    y_t = torch.tensor(mlb.transform(y_labels), dtype=torch.float32)
    return X_t, y_t


def train(data_dir, epochs, batch_size, lr, device):
    X_train, y_train_raw = load_split(data_dir, "train")
    X_val, y_val_raw = load_split(data_dir, "val")

    mlb = MultiLabelBinarizer(classes=CLASSES)
    mlb.fit([CLASSES])

    X_train_t, y_train_t = to_tensors(X_train, y_train_raw, mlb)
    X_val_t, y_val_t = to_tensors(X_val, y_val_raw, mlb)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

    model = ResNet1d(in_channels=X_train_t.shape[1], num_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        val_auc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} | train_loss={total_loss/len(train_loader.dataset):.4f} | val_macro_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  -> new best, saved checkpoint (val_macro_auc={val_auc:.4f})")

    print(f"Best val macro-AUC: {best_val_auc:.4f}")
    print("Sanity check: compare against github.com/helme/ecg_ptbxl_benchmarking "
          "leaderboard for the diagnostic-superclass task before trusting this number.")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_targets = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).sigmoid().cpu().numpy()
        all_logits.append(logits)
        all_targets.append(yb.numpy())
    y_pred = np.concatenate(all_logits)
    y_true = np.concatenate(all_targets)
    # macro AUC, skipping classes with no positive examples in this split
    aucs = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:
            aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    return float(np.mean(aucs)) if aucs else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    train(args.data_dir, args.epochs, args.batch_size, args.lr, device)
