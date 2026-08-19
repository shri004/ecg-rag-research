"""
PTB-XL download + preprocessing.

PTB-XL structure (after download):
  ptbxl_database.csv   - metadata per record (patient info, scp_codes, fold)
  scp_statements.csv   - maps SCP codes -> diagnostic class / superclass
  records100/           - 100Hz WFDB signal files
  records500/           - 500Hz WFDB signal files

We use the diagnostic superclass task (5 classes: NORM, MI, STTC, CD, HYP),
matching the primary task in Strodthoff et al. 2021 -- a good first target
since it's the most-benchmarked, easiest to sanity-check against published
numbers.

Usage:
    python data_prep.py --download --data-dir ./data/ptbxl
    python data_prep.py --data-dir ./data/ptbxl   # skip download, just preprocess
"""
import argparse
import ast
import os
import subprocess

import numpy as np
import pandas as pd
import wfdb

PTBXL_URL = (
    "https://physionet.org/files/ptb-xl/1.0.3/"
)


def download_ptbxl(data_dir: str):
    """Mirrors PhysioNet's recommended wget approach. Requires no
    credentialing for PTB-XL (unlike MIMIC-IV-ECG)."""
    os.makedirs(data_dir, exist_ok=True)
    cmd = [
        "wget", "-r", "-N", "-c", "-np",
        "-P", data_dir,
        PTBXL_URL,
    ]
    print("Downloading PTB-XL (~3GB) -- this will take a while.")
    subprocess.run(cmd, check=True)


def load_raw_signals(df: pd.DataFrame, sampling_rate: int, records_dir: str) -> np.ndarray:
    """Load raw ECG signals for each record referenced in df."""
    if sampling_rate == 100:
        paths = df.filename_lr
    else:
        paths = df.filename_hr
    signals = [wfdb.rdsamp(os.path.join(records_dir, f))[0] for f in paths]
    return np.array(signals)


def aggregate_diagnostic_superclass(scp_codes: dict, agg_df: pd.DataFrame) -> list:
    """Map a record's scp_codes dict to diagnostic superclasses (NORM/MI/STTC/CD/HYP)."""
    classes = set()
    for code in scp_codes.keys():
        if code in agg_df.index:
            classes.add(agg_df.loc[code].diagnostic_class)
    return list(classes)


def prepare(data_dir: str, sampling_rate: int = 100):
    meta_path = os.path.join(data_dir, "ptbxl_database.csv")
    scp_path = os.path.join(data_dir, "scp_statements.csv")

    df = pd.read_csv(meta_path, index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    agg_df = pd.read_csv(scp_path, index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    df["diagnostic_superclass"] = df.scp_codes.apply(
    lambda codes: aggregate_diagnostic_superclass(codes, agg_df)
)

    # Remove records that have no diagnostic superclass label.
    # These records are not negative examples; they are simply unlabeled
    # for the five-class diagnostic-superclass task.
    before = len(df)
    df = df[df.diagnostic_superclass.apply(len) > 0]
    print(f"Dropped {before - len(df)} records with no diagnostic superclass label")
    print(f"Remaining records: {len(df)}")

    # PTB-XL ships with a recommended 10-fold split for reproducible
    # benchmarking -- fold 10 is the standard held-out test fold.
    X = load_raw_signals(df, sampling_rate, data_dir)
    train_mask = df.strat_fold <= 8
    val_mask = df.strat_fold == 9
    test_mask = df.strat_fold == 10

    splits = {
        "X_train": X[train_mask.values], "y_train": df[train_mask].diagnostic_superclass,
        "X_val": X[val_mask.values], "y_val": df[val_mask].diagnostic_superclass,
        "X_test": X[test_mask.values], "y_test": df[test_mask].diagnostic_superclass,
    }

    out_dir = os.path.join(data_dir, "processed")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_train.npy"), splits["X_train"])
    np.save(os.path.join(out_dir, "X_val.npy"), splits["X_val"])
    np.save(os.path.join(out_dir, "X_test.npy"), splits["X_test"])
    splits["y_train"].to_pickle(os.path.join(out_dir, "y_train.pkl"))
    splits["y_val"].to_pickle(os.path.join(out_dir, "y_val.pkl"))
    splits["y_test"].to_pickle(os.path.join(out_dir, "y_test.pkl"))

    print(f"Train/val/test sizes: {len(splits['X_train'])}/{len(splits['X_val'])}/{len(splits['X_test'])}")
    print(f"Saved processed arrays to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--sampling-rate", type=int, default=100, choices=[100, 500])
    args = parser.parse_args()

    if args.download:
        download_ptbxl(args.data_dir)

    prepare(args.data_dir, args.sampling_rate)
