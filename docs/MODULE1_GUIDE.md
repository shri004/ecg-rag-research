# Module 1 — Step-by-Step Guide (ECG Classifier Baseline)

Follow these steps in order. Each step tells you how to check it worked
before moving on — don't skip the checks.

---

## Step 0: Prerequisites

You need:

- Python 3.10 or 3.11 installed (`python3 --version` to check)
- ~5GB free disk space (PTB-XL is ~3GB, plus processed arrays)
- A GPU is nice but not required for Module 1 — this model is small
  (~500K params); CPU training will work, just slower. If you have access
  to Google Colab (free GPU), that's a fine alternative to your own machine.

---

## Step 1: Clone the repo and set up your environment

```bash
git clone <your-repo-url>
cd ecg_rag_project

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Verify:** run `python3 -c "import torch; print(torch.__version__)"` —
should print a version number with no errors. If it fails, the install
didn't work — check the error message before continuing.

---

## Step 2: Download PTB-XL

```bash
python src/data_prep.py --download --data-dir ./data/ptbxl
```

This downloads from PhysioNet (~3GB, no login/credentialing needed for
PTB-XL). Takes a while depending on your connection — this is normal.

**Verify:** after it finishes, check:

```bash
ls data/ptbxl/
```

You should see `ptbxl_database.csv`, `scp_statements.csv`, and folders
like `records100/`. If any of these are missing, the download didn't
complete — re-run the command (it resumes, won't re-download finished parts).

---

## Step 3: Preprocess the data

The same script also handles preprocessing. If you already downloaded in
Step 2, running it again (without `--download`) just re-does preprocessing:

```bash
python src/data_prep.py --data-dir ./data/ptbxl
```

This builds the 5-class diagnostic-superclass labels and splits the data
into train/val/test using PTB-XL's official fold assignment (folds 1-8
train, fold 9 val, fold 10 test — this is the standard split, don't change
it, or your results won't be comparable to the published benchmark).

**Verify:** check the output:

```bash
ls data/ptbxl/processed/
```

You should see 6 files: `X_train.npy`, `X_val.npy`, `X_test.npy`,
`y_train.pkl`, `y_val.pkl`, `y_test.pkl`. The script also prints train/val/
test set sizes when it finishes — sanity-check these are roughly
17,400 / 2,180 / 2,200 (approximate; exact numbers depend on how many
records have a diagnostic-superclass label).

**Version note:** PTB-XL v1.0.3 (current release, what `data_prep.py`
downloads) contains 21,799 records from 18,869 patients — _not_ the
21,837/18,885 reported in the original Wagner et al. 2020 paper, which
was written against v1.0.0/1.0.1. PhysioNet corrected/removed a small
number of records in later versions. This is expected and not a bug —
just don't be alarmed if your counts don't match the original paper
exactly; they should match v1.0.3's numbers instead.

---

## Step 4: Sanity-check the model architecture (no training yet)

Before running a full training job, confirm the model itself works:

```bash
cd src
python model.py
cd ..
```

**Verify:** should print:

```
Output shape: torch.Size([4, 5])
Total params: 509,573
```

If the shape or param count is different, something changed — stop and
check before training (training on a broken model wastes time).

---

## Step 5: Train the model

```bash
python src/train.py --data-dir ./data/ptbxl/processed --epochs 30
```

This will:

- Print `Using device: cpu` or `Using device: cuda` at the start (cuda =
  GPU available and being used)
- Print one line per epoch showing training loss and validation macro-AUC
- Save the best checkpoint as `best_model.pt` whenever validation AUC improves
- Print a final summary when done

Expect validation macro-AUC to climb over the first ~10-15 epochs, then
plateau. If it's not improving at all after 5-10 epochs, something's
wrong (see Troubleshooting below).

**Verify while it's running:** validation macro-AUC should generally trend
upward, even if noisy epoch-to-epoch. If it stays near 0.5 (random-chance
level) after 10+ epochs, stop and troubleshoot rather than waiting it out.

**Verify when done:** check the final printed "Best val macro-AUC" — this
should land somewhere in the 0.85-0.93 range. (Validation AUC and the
official 0.930 test-set benchmark aren't the exact same number, but they
should be in the same neighborhood.)

---

## Step 6: Evaluate on the test set and compare to the published baseline

The training script currently reports validation AUC per epoch. To get
your final, reportable number, you need test-set AUC using the saved
best checkpoint. Add this as a short follow-up script (or run interactively):

```python
import torch
from src.model import ResNet1DWang
from src.train import load_split, to_tensors, evaluate, CLASSES
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, TensorDataset

mlb = MultiLabelBinarizer(classes=CLASSES)
mlb.fit([CLASSES])

X_test, y_test_raw = load_split("data/ptbxl/processed", "test")
X_test_t, y_test_t = to_tensors(X_test, y_test_raw, mlb)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64)

model = ResNet1DWang(in_channels=12, num_classes=5)
model.load_state_dict(torch.load("best_model.pt"))

test_auc = evaluate(model, test_loader, "cpu")
print(f"Test macro-AUC: {test_auc:.4f}")
```

**Verify — this is the key sanity check for the whole module:** compare
your test macro-AUC against the published benchmark:

- **Target: 0.930 (±0.05)** — from Strodthoff et al. 2021, `resnet1d_wang`,
  PTB-XL diagnostic-superclass task
  (github.com/helme/ecg_ptbxl_benchmarking leaderboard)
- **Within ~0.85-0.95:** good, your pipeline is sound — proceed to Module 2
- **Below ~0.80:** stop and check, in this order:
  1. Confirm you're using the official fold split (train=1-8, val=9, test=10)
  2. Confirm labels are the 5-class diagnostic superclass, not some other task
  3. Confirm signals are 100Hz (not accidentally loading 500Hz)
  4. Check training didn't stop early / loss didn't diverge (NaN)
  5. Only after ruling out 1-4, consider it might be a training-budget
     issue (try more epochs, or a lower learning rate)

---

## Step 7: Save your results

```bash
mkdir -p results/module1
```

Save: your test macro-AUC number, a copy of the per-epoch training log
(copy-paste terminal output or redirect to a file next time:
`python src/train.py ... | tee results/module1/training_log.txt`),
and `best_model.pt` (keep this locally — don't commit it to git, see
`docs/GIT_WORKFLOW.md`).

---

## Step 8: Commit your code (not your data/model)

```bash
git add src/ docs/ requirements.txt results/module1/training_log.txt
git commit -m "module1: train ResNet1DWang baseline on PTB-XL"
git push origin feature/module1-baseline
```

Open a PR into `dev` per `docs/GIT_WORKFLOW.md`. Do NOT `git add` the
`data/` folder or `best_model.pt` — `.gitignore` should already block
these, but double check with `git status` before committing.

---

## Troubleshooting

| Symptom                                      | Likely cause                                                                                                                           |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `pip install` fails on `wfdb` or `torch`     | Check Python version (3.10/3.11 recommended); try `pip install --upgrade pip` first                                                    |
| Download in Step 2 hangs or fails partway    | Re-run the same command — it resumes; check your internet connection                                                                   |
| `FileNotFoundError` in Step 3                | Step 2 didn't complete — re-check `data/ptbxl/` contents                                                                               |
| Training loss is `nan`                       | Learning rate too high — try `--lr 0.0001`; also check for corrupted signal data (rare)                                                |
| Validation AUC stuck near 0.5 for 10+ epochs | Check labels aren't accidentally all one class; check `y_train` isn't empty; check data loaded correctly (print a few label counts)    |
| Training is extremely slow on CPU            | Expected — this model is small but PTB-XL has ~17K training examples; consider Google Colab's free GPU tier if it's impractically slow |
| Test AUC far below 0.85                      | See Step 6's ordered checklist above                                                                                                   |

If you hit something not listed here, bring the exact error message back —
don't guess-fix blindly.
