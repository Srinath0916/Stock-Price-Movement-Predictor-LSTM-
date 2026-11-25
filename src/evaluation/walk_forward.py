"""
Walk-Forward Chronological Validation Engine for S&P 500 Stock Movement Predictor.

Evaluates the trained LSTM model across multiple distinct chronological market regimes
to measure performance stability, directional accuracy, and F1-score across bull, bear,
and high-volatility environments without lookahead bias or data leakage.
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_binary_metrics, evaluate_majority_baseline
from src.evaluation.evaluate_model import load_model_from_checkpoint
from src.models.dataset import MODEL_FEATURES, TimeSeriesDataset


# Key historical regimes defined by economic & market conditions
REGIME_WINDOWS = [
    {
        "id": "regime_1",
        "name": "Late Bull / Low Volatility",
        "start_date": "2015-01-01",
        "end_date": "2017-12-31",
        "description": "Steady secular expansion, zero-rate policy tapering, low VIX environment.",
    },
    {
        "id": "regime_2",
        "name": "Trade Wars & Volatility Spike",
        "start_date": "2018-01-01",
        "end_date": "2019-12-31",
        "description": "US-China trade tensions, Q4 2018 liquidity crunch, and Fed policy pivot.",
    },
    {
        "id": "regime_3",
        "name": "COVID Crash & Rapid Stimulus Rally",
        "start_date": "2020-01-01",
        "end_date": "2021-12-31",
        "description": "Unprecedented March 2020 crash followed by historic fiscal/monetary stimulus rally.",
    },
    {
        "id": "regime_4",
        "name": "Rate Hikes & 2022 Bear Market",
        "start_date": "2022-01-01",
        "end_date": "2023-12-31",
        "description": "Aggressive 500bps Fed rate hikes, 2022 bear market drawdown, followed by early disinflation recovery.",
    },
    {
        "id": "regime_5",
        "name": "Disinflation & GenAI Secular Rally",
        "start_date": "2024-01-01",
        "end_date": "2026-02-19",
        "description": "Fed rate cuts cycle, corporate earnings resilience, and mega-cap semiconductor/AI rally.",
    },
]


def run_walk_forward_evaluation(
    data_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    scaler_path: Optional[Path] = None,
    output_json_path: Optional[Path] = None,
    batch_size: int = 4096,
    lookback: int = 30,
) -> List[Dict]:
    """
    Executes walk-forward regime evaluation across all defined windows.
    """
    data_path = Path(data_path) if data_path else PROJECT_ROOT / "data" / "processed" / "SP500_Targeted_Data.parquet"
    model_path = Path(model_path) if model_path else PROJECT_ROOT / "models" / "best_lstm_model.pt"
    scaler_path = Path(scaler_path) if scaler_path else PROJECT_ROOT / "models" / "scaler.joblib"
    output_json_path = Path(output_json_path) if output_json_path else PROJECT_ROOT / "results" / "walk_forward_results.json"

    print("=" * 80)
    print("WALK-FORWARD CHRONOLOGICAL REGIME EVALUATION")
    print("=" * 80)
    print(f"Model Checkpoint: {model_path}")
    print(f"Data File:        {data_path}")
    print(f"Scaler:           {scaler_path}")

    device = torch.device("cpu")
    model, hidden_size = load_model_from_checkpoint(model_path, device)
    scaler = joblib.load(scaler_path)

    print(f"Loaded StockLSTM (hidden_size={hidden_size}, lookback={lookback})")
    print("Loading targeted dataset...")
    df = pd.read_parquet(data_path)

    # Filter out warmup NaNs and final session NaNs
    valid_mask = df[MODEL_FEATURES].notna().all(axis=1) & df["Target"].notna()
    df_clean = df[valid_mask].copy().sort_values(by=["Ticker", "Date"]).reset_index(drop=True)
    df_clean["Date"] = pd.to_datetime(df_clean["Date"])

    # Scale features using training-fitted scaler
    X_scaled = scaler.transform(df_clean[MODEL_FEATURES]).astype(np.float32)
    y_all = df_clean["Target"].to_numpy().astype(np.float32)[:, None]
    dates = df_clean["Date"].to_numpy()
    tickers = df_clean["Ticker"].to_numpy()

    # Vectorized sequence boundary check
    is_valid_seq = tickers[lookback - 1 :] == tickers[: -lookback + 1]
    valid_indices = np.where(is_valid_seq)[0] + (lookback - 1)
    seq_dates = pd.to_datetime(dates[valid_indices])

    results = []
    criterion = nn.BCEWithLogitsLoss()

    for window in REGIME_WINDOWS:
        w_id = window["id"]
        w_name = window["name"]
        w_start = pd.to_datetime(window["start_date"])
        w_end = pd.to_datetime(window["end_date"])

        mask = (seq_dates >= w_start) & (seq_dates <= w_end)
        w_indices = valid_indices[mask]

        if len(w_indices) == 0:
            print(f"Warning: No sequences found for {w_name} ({w_start} to {w_end})")
            continue

        dataset = TimeSeriesDataset(X_scaled, y_all, w_indices, lookback)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Evaluate model
        all_preds = []
        all_targets = []
        running_loss = 0.0
        total = 0

        with torch.no_grad():
            for bx, by in loader:
                bx = bx.to(device)
                by = by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)

                batch_len = len(by)
                running_loss += loss.item() * batch_len
                total += batch_len

                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(by.cpu().numpy())

        avg_loss = running_loss / max(1, total)
        y_prob = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)

        metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)

        # Baseline evaluation (majority class in this window)
        baseline = evaluate_majority_baseline(y_train=y_true, y_eval=y_true, split_name=w_name)

        delta_acc = metrics["accuracy"] - baseline["accuracy"]
        delta_f1 = metrics["f1"] - baseline["f1"]

        regime_result = {
            "regime_id": w_id,
            "name": w_name,
            "start_date": window["start_date"],
            "end_date": window["end_date"],
            "description": window["description"],
            "total_sequences": int(metrics["total_samples"]),
            "loss": round(float(avg_loss), 4),
            "accuracy": round(float(metrics["accuracy"]), 4),
            "precision": round(float(metrics["precision"]), 4),
            "recall": round(float(metrics["recall"]), 4),
            "f1": round(float(metrics["f1"]), 4),
            "baseline_accuracy": round(float(baseline["accuracy"]), 4),
            "baseline_f1": round(float(baseline["f1"]), 4),
            "delta_accuracy": round(float(delta_acc), 4),
            "delta_f1": round(float(delta_f1), 4),
            "confusion_matrix": metrics["confusion_matrix"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
            "is_out_of_sample": w_start >= pd.to_datetime("2022-03-25"),
        }
        results.append(regime_result)

        print(f"\n--- {w_name} ({window['start_date']} to {window['end_date']}) ---")
        print(f"  Sequences:    {metrics['total_samples']:,}")
        print(f"  Accuracy:     {metrics['accuracy']*100:.2f}% (Baseline: {baseline['accuracy']*100:.2f}%, Delta: {delta_acc*100:+.2f}%)")
        print(f"  Precision:    {metrics['precision']*100:.2f}% | Recall: {metrics['recall']*100:.2f}% | F1: {metrics['f1']:.4f}")
        print(f"  Loss (BCE):   {avg_loss:.4f}")
        print(f"  Confusion:    [TN: {metrics['tn']:,}, FP: {metrics['fp']:,}] | [FN: {metrics['fn']:,}, TP: {metrics['tp']:,}]")

    # Save to JSON
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved walk-forward validation results to: {output_json_path}")
    return results


if __name__ == "__main__":
    run_walk_forward_evaluation()
