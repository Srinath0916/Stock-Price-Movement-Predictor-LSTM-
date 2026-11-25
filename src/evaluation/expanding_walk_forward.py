"""
Genuine Expanding-Window Walk-Forward Validation Engine for S&P 500 Stock Movement Predictor.

Strict Quantitative Integrity Rules:
1. Chronological expanding training windows (no lookahead, no future data leakage).
2. StandardScaler is fitted strictly and exclusively on each fold's training partition.
3. A separate chronological validation window is used for monitoring loss and checkpoint selection.
4. An untouched future test window is used strictly for final out-of-sample evaluation.
5. Sequences are prevented from crossing ticker boundaries.
6. A separate, freshly initialized StockLSTM is trained per fold.
7. Existing production checkpoint (models/best_lstm_model.pt) is kept untouched.
8. Every fold computes: Accuracy, Precision, Recall, F1, Loss, Majority-Class Baseline, and Confusion Matrix.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_binary_metrics, evaluate_majority_baseline
from src.models.dataset import MODEL_FEATURES, TimeSeriesDataset
from src.models.lstm import StockLSTM
from src.models.train import train_one_epoch, evaluate_model


# 3 Chronological Expanding Folds covering macroeconomic market regimes
EXPANDING_FOLDS = [
    {
        "fold_id": "fold_1",
        "name": "Expanding Fold 1: Post-GFC Expansion (2015–2017 OOS)",
        "train_start": "2000-03-14",
        "train_end": "2012-12-31",
        "val_start": "2013-01-01",
        "val_end": "2014-12-31",
        "test_start": "2015-01-01",
        "test_end": "2017-12-31",
        "description": "Trained on 2000–2012 (Dot-Com bust, 2008 GFC); validated on 2013–2014 QE recovery; evaluated out-of-sample on 2015–2017 late bull regime.",
    },
    {
        "fold_id": "fold_2",
        "name": "Expanding Fold 2: Trade Wars & COVID Shock (2018–2021 OOS)",
        "train_start": "2000-03-14",
        "train_end": "2015-12-31",
        "val_start": "2016-01-01",
        "val_end": "2017-12-31",
        "test_start": "2018-01-01",
        "test_end": "2021-12-31",
        "description": "Expanding history through 2015; validated on 2016–2017 rate lift-off; evaluated out-of-sample on US-China trade tensions and COVID-19 pandemic shock.",
    },
    {
        "fold_id": "fold_3",
        "name": "Expanding Fold 3: Rate Hikes & GenAI Rally (2022–2026 OOS)",
        "train_start": "2000-03-14",
        "train_end": "2019-12-31",
        "val_start": "2020-01-01",
        "val_end": "2021-12-31",
        "test_start": "2022-01-01",
        "test_end": "2026-02-19",
        "description": "Expanding history through 2019; validated on 2020–2021 stimulus volatility; evaluated out-of-sample on 2022 bear market and the GenAI secular rally.",
    },
]


def execute_expanding_walk_forward(
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    batch_size: int = 8192,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.20,
    lr: float = 0.001,
    max_epochs: int = 4,
    lookback: int = 30,
) -> List[Dict]:
    """
    Executes genuine expanding-window training and evaluation across all folds.
    """
    data_path = Path(data_path) if data_path else PROJECT_ROOT / "data" / "processed" / "SP500_Targeted_Data.parquet"
    out_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "models" / "walk_forward"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 80)
    print("GENUINE EXPANDING-WINDOW WALK-FORWARD VALIDATION PIPELINE")
    print("=" * 80)
    print(f"Dataset Path:       {data_path}")
    print(f"Fold Output Dir:    {out_dir}")
    print(f"Device:             {device}")
    print(f"Architecture:       StockLSTM (input=8, hidden={hidden_size}, layers={num_layers}, dropout={dropout})")
    print(f"Lookback Window:    {lookback} trading sessions")
    print(f"Batch Size:         {batch_size}")
    print(f"Max Epochs / Fold:  {max_epochs}")
    print("=" * 80)

    # 1. Load data
    print("Reading targeted dataset...")
    df = pd.read_parquet(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    # Clean NaNs in features and target
    valid_mask = df[MODEL_FEATURES].notna().all(axis=1) & df["Target"].notna()
    df_clean = df[valid_mask].copy().sort_values(by=["Ticker", "Date"]).reset_index(drop=True)

    raw_features = df_clean[MODEL_FEATURES].to_numpy().astype(np.float32)
    raw_targets = df_clean["Target"].to_numpy().astype(np.float32)[:, None]
    tickers = df_clean["Ticker"].to_numpy()
    dates = df_clean["Date"].to_numpy()

    # Vectorized boundary check: sequence must belong to the exact same ticker
    is_valid_seq = tickers[lookback - 1:] == tickers[:-lookback + 1]
    valid_indices = np.where(is_valid_seq)[0] + (lookback - 1)
    seq_dates = pd.to_datetime(dates[valid_indices])

    print(f"Total valid non-crossing sequences: {len(valid_indices):,}")

    fold_results = []
    criterion = nn.BCEWithLogitsLoss()

    for fold_idx, fold in enumerate(EXPANDING_FOLDS, start=1):
        f_id = fold["fold_id"]
        f_name = fold["name"]
        print(f"\n{'='*70}")
        print(f"STARTING FOLD {fold_idx}/{len(EXPANDING_FOLDS)}: {f_name}")
        print(f"Train Period:  {fold['train_start']} -> {fold['train_end']}")
        print(f"Val Period:    {fold['val_start']} -> {fold['val_end']}")
        print(f"Test Period:   {fold['test_start']} -> {fold['test_end']}")
        print(f"{'='*70}")

        dt_tr_end = pd.to_datetime(fold["train_end"])
        dt_val_start = pd.to_datetime(fold["val_start"])
        dt_val_end = pd.to_datetime(fold["val_end"])
        dt_te_start = pd.to_datetime(fold["test_start"])
        dt_te_end = pd.to_datetime(fold["test_end"])

        # Determine indices belonging to train, val, and test partitions
        tr_seq_mask = seq_dates <= dt_tr_end
        val_seq_mask = (seq_dates >= dt_val_start) & (seq_dates <= dt_val_end)
        te_seq_mask = (seq_dates >= dt_te_start) & (seq_dates <= dt_te_end)

        tr_indices = valid_indices[tr_seq_mask]
        val_indices = valid_indices[val_seq_mask]
        te_indices = valid_indices[te_seq_mask]

        print(f"Sequences -> Train: {len(tr_indices):,} | Val: {len(val_indices):,} | Test (OOS): {len(te_indices):,}")

        # 2. Strict Scaler Fitting: FITTED ONLY ON THIS FOLD'S TRAINING DATA
        print("Fitting StandardScaler strictly on fold training data...")
        # Get raw feature rows associated with train
        row_tr_mask = df_clean["Date"] <= dt_tr_end
        fold_scaler = StandardScaler()
        fold_scaler.fit(df_clean.loc[row_tr_mask, MODEL_FEATURES])

        # Transform entire feature matrix using this fold's scaler
        scaled_X = fold_scaler.transform(raw_features).astype(np.float32)

        # Build Datasets and DataLoaders
        train_ds = TimeSeriesDataset(scaled_X, raw_targets, tr_indices, lookback)
        val_ds = TimeSeriesDataset(scaled_X, raw_targets, val_indices, lookback)
        test_ds = TimeSeriesDataset(scaled_X, raw_targets, te_indices, lookback)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # 3. Fresh Model Initialization
        fold_model = StockLSTM(
            input_dim=8,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)

        optimizer = torch.optim.Adam(fold_model.parameters(), lr=lr, weight_decay=1e-5)

        best_val_loss = float("inf")
        best_val_f1 = 0.0
        best_state_dict = None
        best_epoch = 0

        # 4. Training Loop with Validation Monitoring
        fold_start_t = time.time()
        for epoch in range(1, max_epochs + 1):
            ep_start = time.time()
            tr_loss, tr_acc = train_one_epoch(fold_model, train_loader, criterion, optimizer, device, max_norm=1.0)
            val_loss, val_metrics = evaluate_model(fold_model, val_loader, criterion, device)
            ep_duration = time.time() - ep_start

            print(
                f"  Epoch {epoch}/{max_epochs} ({ep_duration:.1f}s) | "
                f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']*100:.2f}%, Val F1: {val_metrics['f1']:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_f1 = val_metrics["f1"]
                best_state_dict = {k: v.cpu().clone() for k, v in fold_model.state_dict().items()}
                best_epoch = epoch

        # Load best weights for final evaluation
        fold_model.load_state_dict(best_state_dict)
        fold_model.to(device)

        # 5. Out-of-Sample Test Evaluation
        print(f"Evaluating Fold {fold_idx} on untouched out-of-sample test partition ({fold['test_start']} to {fold['test_end']})...")
        test_loss, test_metrics = evaluate_model(fold_model, test_loader, criterion, device)

        # Extract test ground truth for baseline comparison
        te_targets_slice = raw_targets[te_indices].ravel()
        baseline = evaluate_majority_baseline(y_train=raw_targets[tr_indices].ravel(), y_eval=te_targets_slice, split_name=f_name)

        delta_acc = test_metrics["accuracy"] - baseline["accuracy"]
        delta_f1 = test_metrics["f1"] - baseline["f1"]

        # 6. Save fold-specific checkpoint and scaler (never touching best_lstm_model.pt)
        ckpt_path = out_dir / f"{f_id}_lstm.pt"
        scaler_file = out_dir / f"{f_id}_scaler.joblib"
        torch.save(best_state_dict, ckpt_path)
        joblib.dump(fold_scaler, scaler_file)

        fold_record = {
            "fold_id": f_id,
            "name": f_name,
            "description": fold["description"],
            "train_period": f"{fold['train_start']} to {fold['train_end']}",
            "val_period": f"{fold['val_start']} to {fold['val_end']}",
            "test_period": f"{fold['test_start']} to {fold['test_end']}",
            "train_sequences": int(len(tr_indices)),
            "val_sequences": int(len(val_indices)),
            "test_sequences": int(len(te_indices)),
            "best_epoch": best_epoch,
            "test_loss": round(float(test_loss), 4),
            "test_accuracy": round(float(test_metrics["accuracy"]), 4),
            "test_precision": round(float(test_metrics["precision"]), 4),
            "test_recall": round(float(test_metrics["recall"]), 4),
            "test_f1": round(float(test_metrics["f1"]), 4),
            "baseline_accuracy": round(float(baseline["accuracy"]), 4),
            "baseline_f1": round(float(baseline["f1"]), 4),
            "delta_accuracy": round(float(delta_acc), 4),
            "delta_f1": round(float(delta_f1), 4),
            "confusion_matrix": test_metrics["confusion_matrix"],
            "tn": test_metrics["tn"],
            "fp": test_metrics["fp"],
            "fn": test_metrics["fn"],
            "tp": test_metrics["tp"],
            "checkpoint_path": str(ckpt_path.relative_to(PROJECT_ROOT)),
            "scaler_path": str(scaler_file.relative_to(PROJECT_ROOT)),
            "elapsed_seconds": round(time.time() - fold_start_t, 1),
        }
        fold_results.append(fold_record)

        print(f"\n--- FOLD {fold_idx} OUT-OF-SAMPLE TEST RESULTS ---")
        print(f"  Test Accuracy:    {test_metrics['accuracy']*100:.2f}% (Baseline: {baseline['accuracy']*100:.2f}%, Delta: {delta_acc*100:+.2f}%)")
        print(f"  Precision:        {test_metrics['precision']*100:.2f}% | Recall: {test_metrics['recall']*100:.2f}% | F1: {test_metrics['f1']:.4f}")
        print(f"  Loss (BCE):       {test_loss:.4f}")
        print(f"  Confusion Matrix: [TN: {test_metrics['tn']:,}, FP: {test_metrics['fp']:,}] | [FN: {test_metrics['fn']:,}, TP: {test_metrics['tp']:,}]")
        print(f"  Saved Checkpoint: {ckpt_path}")
        print(f"  Saved Scaler:     {scaler_file}")

    # 7. Persist complete results to JSON
    json_path = results_dir / "expanding_walk_forward_results.json"
    with open(json_path, "w") as f:
        json.dump(fold_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✓ EXPANDING WALK-FORWARD PIPELINE COMPLETE")
    print(f"Saved complete results across {len(fold_results)} folds to: {json_path}")
    print(f"{'='*80}")

    return fold_results


if __name__ == "__main__":
    execute_expanding_walk_forward()

