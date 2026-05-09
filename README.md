## Project Structure
- `models/`: Contains the core mathematical architecture (`unrolled_rc_rpca.py`, `rc_rpca.py`).
- `data/`: Data generation scripts including Gatheral's SVI synthetic surfaces (`synthetic_svi.py`) and empirical data preprocessing (`preprocessing.py`).
- `training/`: Training loops and PyTorch dataset wrappers.
- `evaluation/`: Scripts to reproduce synthetic AUROC metrics and real-data Information Coefficient (IC) statistics.
---

## 1. Environment Setup
```bash
# After clone, create a .venv and install dependencies
pip install -r requirements.txt
```
## 2. Reproducing the Pipeline

The primary specification for the experiments in the paper uses an unrolling depth of K = 12. Follow the steps below to train the model and generate the evaluation metrics.

### Step 2.1: Train the U-RC-RPCA Model (K=12)
```bash
python -m training.train --K 12 --epochs 60 ` --npz data/synthetic.npz --run-name urc_K12 
```

### Step 2.2: Evaluate Synthetic Anomaly Detection (AUROC)
```bash
python -m evaluation.synthetic_auroc --run-dir runs/urc_K12 --npz data/synthetic.npz 
```


### Step 2.3: Evaluate Real-Data Straddle P&L (IC)
```bash
python -m evaluation.real_data_ic \\
        --run-dir runs/urc_K12 \\
        --symbols AAPL NVDA TSLA \\
        --dates 2026-03-03 2026-03-04 2026-03-05 2026-03-06 2026-03-09 2026-03-10 2026-03-11 2026-03-13 2026-03-17 2026-03-18 2026-03-19 2026-03-23 2026-03-25 2026-03-26 2026-03-30 2026-03-31 2026-04-02 2026-04-10 2026-04-13 2026-04-14 2026-04-15 2026-04-17 2026-04-20 2026-04-22 2026-04-23 2026-04-29
        --n-bootstraps 10000
```
