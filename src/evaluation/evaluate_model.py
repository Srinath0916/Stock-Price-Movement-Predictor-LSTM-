"""
Standalone Evaluation Script for S&P 500 Stock Movement Predictor.
Loads a trained PyTorch model checkpoint and evaluates directional accuracy,
precision, recall, F1 score, and confusion matrix on the out-of-sample test or validation set.

Usage from terminal:
    .venv/bin/python src/evaluation/evaluate_model.py
    .venv/bin/python src/evaluation/evaluate_model.py --split val
    .venv/bin/python src/evaluation/evaluate_model.py --model models/best_lstm_model.pt
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Tuple

# Ensure project root is in sys.path
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
from src.models.dataset import DataPipeline, MODEL_FEATURES, TimeSeriesDataset
from src.models.lstm import StockLSTM
from src.models.train import evaluate_model


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[StockLSTM, int]:
    """
    Loads model weights and automatically infers hidden_size from checkpoint tensors.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Detect hidden_size from fc.weight shape: [1, hidden_size]
    hidden_size = ckpt["fc.weight"].shape[1]
    input_dim = ckpt["lstm.weight_ih_l0"].shape[1]
    num_layers = 2 if "lstm.weight_ih_l1" in ckpt else 1

    model = StockLSTM(
        input_dim=input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.2,
    )
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model, hidden_size


def run_evaluation(
    data_path: Path,
    model_path: Path,
    split: str = "test",
    batch_size: int = 4096,
):
    start_time = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 80)
    print("S&P 500 STOCK MOVEMENT PREDICTOR — MODEL ACCURACY EVALUATION")
    print("=" * 80)
    print(f"Model Checkpoint: {model_path}")
    print(f"Data Source:      {data_path}")
    print(f"Evaluation Split: {split.upper()}")
    print(f"Compute Device:   {device}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    # 1. Load model and inspect architecture
    model, hidden_size = load_model_from_checkpoint(model_path, device)
    # Detect lookback: default 30, or 60 if expB in filename
    lookback = 60 if "lb60" in model_path.name else 30
    print(f"Detected Architecture: StockLSTM(input_dim=8, hidden_size={hidden_size}, num_layers=2, lookback={lookback})")

    # 2. Prepare dataset using verified pipeline
    pipeline = DataPipeline(data_path=data_path, lookback=lookback, scaler_save_path=None)
    split_info = pipeline.prepare_data()

    train_loader, val_loader, test_loader = pipeline.get_dataloaders(batch_size=batch_size)

    y_train = pipeline.y_all[pipeline.train_indices].ravel()
    y_val = pipeline.y_all[pipeline.val_indices].ravel()
    y_test = pipeline.y_all[pipeline.test_indices].ravel()

    criterion = nn.BCEWithLogitsLoss()

    splits_to_eval = []
    if split in ["test", "all"]:
        splits_to_eval.append(("Test Set", test_loader, y_test, split_info["dates"]["test_start"], split_info["dates"]["test_end"]))
    if split in ["val", "validation", "all"]:
        splits_to_eval.append(("Validation Set", val_loader, y_val, split_info["dates"]["val_start"], split_info["dates"]["val_end"]))

    for split_label, loader, y_ground_truth, start_d, end_d in splits_to_eval:
        print("\n" + "=" * 80)
        print(f"ACCURACY RESULTS FOR: {split_label.upper()} ({start_d} to {end_d})")
        print("=" * 80)

        # Baseline calculation
        baseline = evaluate_majority_baseline(y_train, y_ground_truth, split_name=split_label)

        # Model evaluation
        loss, metrics = evaluate_model(model=model, dataloader=loader, criterion=criterion, device=device)

        acc = metrics["accuracy"] * 100
        prec = metrics["precision"] * 100
        rec = metrics["recall"] * 100
        f1 = metrics["f1"]
        base_acc = baseline["accuracy"] * 100
        delta_acc = acc - base_acc

        print(f"\nTotal Evaluated Samples: {metrics['total_samples']:,}")
        print(f"Loss (BCE):              {loss:.4f}")
        print("-" * 50)
        print(f"Directional Accuracy:    {acc:6.2f}%   (Naive Baseline: {base_acc:5.2f}%  |  Delta: {delta_acc:+5.2f}%)")
        print(f"Precision:               {prec:6.2f}%   (Positive Prediction Accuracy)")
        print(f"Recall:                  {rec:6.2f}%   (Capture Rate of Actual UP days)")
        print(f"F1 Score:                {f1:6.4f}   (Naive Baseline F1: {baseline['f1']:.4f})")
        print("-" * 50)

        tn, fp, fn, tp = metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]
        total = metrics["total_samples"]

        print("\nConfusion Matrix:")
        print(f"┌─────────────────────────┬─────────────────────────┬─────────────────────────┐")
        print(f"│ Ground Truth / Model    │ Predicted DOWN/FLAT (0) │ Predicted UP (1)        │")
        print(f"├─────────────────────────┼─────────────────────────┼─────────────────────────┤")
        print(f"│ Actual DOWN/FLAT (0)    │ TN: {tn:17,d} │ FP: {fp:17,d} │ Total: {tn+fp:,}")
        print(f"│ Actual UP (1)           │ FN: {fn:17,d} │ TP: {tp:17,d} │ Total: {fn+tp:,}")
        print(f"├─────────────────────────┼─────────────────────────┼─────────────────────────┤")
        print(f"│ Total Predictions       │ {tn+fn:21,d} │ {fp+tp:21,d} │ {total:,}")
        print(f"└─────────────────────────┴─────────────────────────┴─────────────────────────┘")

    print("\n" + "=" * 80)
    print(f"Evaluation finished in {time.time() - start_time:.2f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained LSTM model on stock price movement data.")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "best_lstm_model.pt",
        help="Path to trained PyTorch model checkpoint (.pt)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "SP500_Targeted_Data.parquet",
        help="Path to targeted parquet data",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "all"],
        help="Split to evaluate: 'test' (default), 'val', or 'all'",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Evaluation batch size",
    )

    args = parser.parse_args()
    run_evaluation(
        data_path=args.data,
        model_path=args.model,
        split=args.split,
        batch_size=args.batch_size,
    )
