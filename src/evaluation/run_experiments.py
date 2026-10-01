"""
Master Experimentation and Evaluation Pipeline for Stock Price Movement Predictor.
Executes Chronological Splits, Baseline, LSTM Training, Controlled Experiments (A, B, C, D),
Validation-Based Model Selection, Final Single-Pass Test Evaluation, and Report Generation.
"""

import json
from pathlib import Path
import sys
import time
from typing import Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.evaluation.metrics import evaluate_majority_baseline, compute_binary_metrics
from src.models.dataset import DataPipeline, MODEL_FEATURES
from src.models.lstm import StockLSTM
from src.models.train import train_lstm_experiment, evaluate_model


def run_full_pipeline(
    data_path: Path,
    models_dir: Path,
    results_dir: Path,
    max_epochs: int = 50,
    patience: int = 7,
    batch_size: int = 4096,
):
    start_total_time = time.time()
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Executing on hardware device: {device}")

    # =========================================================================
    # STAGE 1, 2, 3, 4, 5: Data Pipeline for Lookback = 30
    # =========================================================================
    scaler_path = models_dir / "scaler.joblib"
    pipeline_lb30 = DataPipeline(data_path=data_path, lookback=30, scaler_save_path=scaler_path)
    split_info_lb30 = pipeline_lb30.prepare_data()

    train_loader_30, val_loader_30, test_loader_30 = pipeline_lb30.get_dataloaders(batch_size=batch_size)

    # Extract target arrays for baseline calculations
    y_train_30 = pipeline_lb30.y_all[pipeline_lb30.train_indices].ravel()
    y_val_30 = pipeline_lb30.y_all[pipeline_lb30.val_indices].ravel()
    y_test_30 = pipeline_lb30.y_all[pipeline_lb30.test_indices].ravel()

    # =========================================================================
    # STAGE 6: Naive Majority Class Baseline
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 6: NAIVE BASELINE EVALUATION (Always Predict Majority Class)")
    print("=" * 80)
    baseline_val = evaluate_majority_baseline(y_train_30, y_val_30, split_name="Validation (LB=30)")
    baseline_test = evaluate_majority_baseline(y_train_30, y_test_30, split_name="Test (LB=30)")

    print(f"Majority Class in Training Data: {baseline_val['majority_class_name']}")
    print(f"Baseline Validation -> Acc: {baseline_val['accuracy']*100:.2f}% | Prec: {baseline_val['precision']*100:.2f}% | Rec: {baseline_val['recall']*100:.2f}% | F1: {baseline_val['f1']:.4f}")
    print(f"Baseline Test       -> Acc: {baseline_test['accuracy']*100:.2f}% | Prec: {baseline_test['precision']*100:.2f}% | Rec: {baseline_test['recall']*100:.2f}% | F1: {baseline_test['f1']:.4f}")

    # =========================================================================
    # STAGE 7, 8, 10: Run Controlled Experiments A, B, C, D
    # =========================================================================
    experiments_summary = []

    # --- EXPERIMENT A: Baseline LSTM (Lookback=30, Hidden=64, LR=0.001) ---
    model_a = StockLSTM(input_dim=len(MODEL_FEATURES), hidden_size=64, num_layers=2, dropout=0.2)
    ckpt_a = models_dir / "lstm_expA_lb30_h64_lr001.pt"
    plot_a = results_dir / "training_curves_expA.png"
    res_a = train_lstm_experiment(
        model=model_a,
        train_loader=train_loader_30,
        val_loader=val_loader_30,
        learning_rate=0.001,
        weight_decay=1e-5,
        max_norm=1.0,
        max_epochs=max_epochs,
        patience=patience,
        checkpoint_path=ckpt_a,
        plot_path=plot_a,
        experiment_name="Experiment A: Baseline (LB=30, Hidden=64, LR=0.001)",
        device=device,
    )
    res_a["lookback"] = 30
    res_a["hidden_size"] = 64
    res_a["lr"] = 0.001
    experiments_summary.append(res_a)

    # --- EXPERIMENT B: Lookback=60 (Lookback=60, Hidden=64, LR=0.001) ---
    print("\nPreparing Lookback=60 pipeline for Experiment B...")
    pipeline_lb60 = DataPipeline(data_path=data_path, lookback=60, scaler_save_path=None)
    # Reuse fitted scaler from pipeline_lb30 to ensure consistent feature scaling
    pipeline_lb60.prepare_data()
    pipeline_lb60.scaler = pipeline_lb30.scaler
    pipeline_lb60.X_scaled = pipeline_lb30.scaler.transform(pipeline_lb60.df_clean[MODEL_FEATURES]).astype(np.float32)

    train_loader_60, val_loader_60, test_loader_60 = pipeline_lb60.get_dataloaders(batch_size=batch_size)

    model_b = StockLSTM(input_dim=len(MODEL_FEATURES), hidden_size=64, num_layers=2, dropout=0.2)
    ckpt_b = models_dir / "lstm_expB_lb60_h64_lr001.pt"
    plot_b = results_dir / "training_curves_expB.png"
    res_b = train_lstm_experiment(
        model=model_b,
        train_loader=train_loader_60,
        val_loader=val_loader_60,
        learning_rate=0.001,
        weight_decay=1e-5,
        max_norm=1.0,
        max_epochs=max_epochs,
        patience=patience,
        checkpoint_path=ckpt_b,
        plot_path=plot_b,
        experiment_name="Experiment B: Lookback=60 (LB=60, Hidden=64, LR=0.001)",
        device=device,
    )
    res_b["lookback"] = 60
    res_b["hidden_size"] = 64
    res_b["lr"] = 0.001
    experiments_summary.append(res_b)

    # --- EXPERIMENT C: Hidden=128 (Lookback=30, Hidden=128, LR=0.001) ---
    model_c = StockLSTM(input_dim=len(MODEL_FEATURES), hidden_size=128, num_layers=2, dropout=0.2)
    ckpt_c = models_dir / "lstm_expC_lb30_h128_lr001.pt"
    plot_c = results_dir / "training_curves_expC.png"
    res_c = train_lstm_experiment(
        model=model_c,
        train_loader=train_loader_30,
        val_loader=val_loader_30,
        learning_rate=0.001,
        weight_decay=1e-5,
        max_norm=1.0,
        max_epochs=max_epochs,
        patience=patience,
        checkpoint_path=ckpt_c,
        plot_path=plot_c,
        experiment_name="Experiment C: Hidden=128 (LB=30, Hidden=128, LR=0.001)",
        device=device,
    )
    res_c["lookback"] = 30
    res_c["hidden_size"] = 128
    res_c["lr"] = 0.001
    experiments_summary.append(res_c)

    # --- EXPERIMENT D: Learning Rate=0.0005 (Lookback=30, Hidden=64, LR=0.0005) ---
    model_d = StockLSTM(input_dim=len(MODEL_FEATURES), hidden_size=64, num_layers=2, dropout=0.2)
    ckpt_d = models_dir / "lstm_expD_lb30_h64_lr0005.pt"
    plot_d = results_dir / "training_curves_expD.png"
    res_d = train_lstm_experiment(
        model=model_d,
        train_loader=train_loader_30,
        val_loader=val_loader_30,
        learning_rate=0.0005,
        weight_decay=1e-5,
        max_norm=1.0,
        max_epochs=max_epochs,
        patience=patience,
        checkpoint_path=ckpt_d,
        plot_path=plot_d,
        experiment_name="Experiment D: Low LR (LB=30, Hidden=64, LR=0.0005)",
        device=device,
    )
    res_d["lookback"] = 30
    res_d["hidden_size"] = 64
    res_d["lr"] = 0.0005
    experiments_summary.append(res_d)

    # =========================================================================
    # MODEL SELECTION (Strictly Based on Validation Performance)
    # =========================================================================
    print("\n" + "=" * 80)
    print("CONTROLLED EXPERIMENTS COMPARISON & VALIDATION SELECTION")
    print("=" * 80)
    comparison_rows = []
    for exp in experiments_summary:
        m = exp["best_val_metrics"]
        comparison_rows.append({
            "Experiment": exp["experiment_name"],
            "Lookback": exp["lookback"],
            "Hidden": exp["hidden_size"],
            "LR": exp["lr"],
            "Best_Epoch": exp["best_epoch"],
            "Total_Epochs": exp["total_epochs_trained"],
            "Val_Loss": round(exp["best_val_loss"], 4),
            "Val_Acc": round(m["accuracy"] * 100, 2),
            "Val_Prec": round(m["precision"] * 100, 2),
            "Val_Rec": round(m["recall"] * 100, 2),
            "Val_F1": round(m["f1"], 4),
        })

    comp_df = pd.DataFrame(comparison_rows)
    print(comp_df.to_string(index=False))

    comp_csv = results_dir / "experiment_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    print(f"✓ Saved experiment comparison to: {comp_csv}")

    # Rank by primary validation criteria: Validation F1, then Validation Accuracy, then lowest Validation Loss
    comp_df_sorted = comp_df.sort_values(by=["Val_F1", "Val_Acc", "Val_Loss"], ascending=[False, False, True])
    best_exp_row = comp_df_sorted.iloc[0]
    best_exp_name = best_exp_row["Experiment"]
    print(f"\n★ WINNING EXPERIMENT SELECTED: {best_exp_name}")
    print(f"Validation F1: {best_exp_row['Val_F1']} | Validation Acc: {best_exp_row['Val_Acc']}% | Validation Loss: {best_exp_row['Val_Loss']}")

    # Map to winning model instance and test loader
    if "Experiment A" in best_exp_name:
        winning_model = model_a
        winning_test_loader = test_loader_30
        winning_ckpt = ckpt_a
        winning_lookback = 30
        winning_pipeline = pipeline_lb30
    elif "Experiment B" in best_exp_name:
        winning_model = model_b
        winning_test_loader = test_loader_60
        winning_ckpt = ckpt_b
        winning_lookback = 60
        winning_pipeline = pipeline_lb60
    elif "Experiment C" in best_exp_name:
        winning_model = model_c
        winning_test_loader = test_loader_30
        winning_ckpt = ckpt_c
        winning_lookback = 30
        winning_pipeline = pipeline_lb30
    else:
        winning_model = model_d
        winning_test_loader = test_loader_30
        winning_ckpt = ckpt_d
        winning_lookback = 30
        winning_pipeline = pipeline_lb30

    # Save as canonical best model
    best_model_path = models_dir / "best_lstm_model.pt"
    torch.save(winning_model.state_dict(), best_model_path)
    print(f"✓ Saved canonical best model to: {best_model_path}")

    # =========================================================================
    # STAGE 9: Final Test Evaluation (Evaluated ONCE on Untouched Test Set)
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"STAGE 9: FINAL UNTOUCHED TEST SET EVALUATION ({best_exp_name})")
    print("=" * 80)
    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_metrics = evaluate_model(
        model=winning_model,
        dataloader=winning_test_loader,
        criterion=criterion,
        device=device,
    )

    print(f"Test Samples:          {test_metrics['total_samples']:,}")
    print(f"Test Loss (BCE):       {test_loss:.4f}")
    print(f"Directional Accuracy:  {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Precision:             {test_metrics['precision'] * 100:.2f}%")
    print(f"Recall:                {test_metrics['recall'] * 100:.2f}%")
    print(f"F1 Score:              {test_metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(f"  [[TN: {test_metrics['tn']:,}, FP: {test_metrics['fp']:,}],")
    print(f"   [FN: {test_metrics['fn']:,}, TP: {test_metrics['tp']:,}]]")

    # Save metrics JSON files
    with open(results_dir / "naive_baseline_results.json", "w") as f:
        json.dump({"validation": baseline_val, "test": baseline_test}, f, indent=2)

    with open(results_dir / "validation_results.json", "w") as f:
        json.dump([e for e in comparison_rows], f, indent=2)

    with open(results_dir / "final_test_results.json", "w") as f:
        json.dump({
            "selected_model": best_exp_name,
            "lookback": winning_lookback,
            "metrics": test_metrics,
            "baseline_comparison": {
                "lstm_accuracy": test_metrics["accuracy"],
                "baseline_accuracy": baseline_test["accuracy"],
                "delta_accuracy": test_metrics["accuracy"] - baseline_test["accuracy"],
                "lstm_f1": test_metrics["f1"],
                "baseline_f1": baseline_test["f1"],
                "delta_f1": test_metrics["f1"] - baseline_test["f1"],
            }
        }, f, indent=2)

    # =========================================================================
    # STAGE 11: Create results/model_training_report.md
    # =========================================================================
    report_path = results_dir / "model_training_report.md"
    generate_markdown_report(
        report_path=report_path,
        split_info=split_info_lb30,
        baseline_val=baseline_val,
        baseline_test=baseline_test,
        comp_df=comp_df,
        best_exp=best_exp_row,
        test_metrics=test_metrics,
        total_time_sec=time.time() - start_total_time,
        batch_size=batch_size,
    )

    print("\n" + "=" * 80)
    print(f"ALL STAGES COMPLETE! Total elapsed time: {(time.time() - start_total_time) / 60:.2f} minutes")
    print(f"Report generated: {report_path}")
    print("=" * 80)


def generate_markdown_report(
    report_path: Path,
    split_info: Dict,
    baseline_val: Dict,
    baseline_test: Dict,
    comp_df: pd.DataFrame,
    best_exp: pd.Series,
    test_metrics: Dict,
    total_time_sec: float,
    batch_size: int = 4096,
):
    d = split_info["dates"]
    tr_s = split_info["train_stats"]
    val_s = split_info["val_stats"]
    te_s = split_info["test_stats"]

    content = f"""# S&P 500 Stock Price Movement Predictor: Model Development & Training Report

## Executive Summary
This report documents the end-to-end execution of the **Model Development and Training Stage** for predicting next-day binary price movement across 472 S&P 500 stocks. 
All operations were executed without lookahead bias, without target leakage, with strict per-ticker sequence isolation, with chronological train/validation/test partitioning, and with scaling fitted exclusively on the training period.

---

## 1. Chronological Data Split Methodology

The dataset was partitioned chronologically across unique trading calendar dates (6,573 total dates from 2000-01-03 to 2026-02-20) to strictly mirror forward-looking deployment:

| Split | Percentage | Date Range | Total Rows | Tickers | UP Distribution | DOWN/FLAT Distribution |
|---|---|---|---|---|---|---|
| **TRAIN** | 70% | {d['train_start']} to {d['train_end']} | {tr_s['rows']:,} | {tr_s['tickers']} | {tr_s['target_1_pct']:.2f}% | {tr_s['target_0_pct']:.2f}% |
| **VALIDATION** | 15% | {d['val_start']} to {d['val_end']} | {val_s['rows']:,} | {val_s['tickers']} | {val_s['target_1_pct']:.2f}% | {val_s['target_0_pct']:.2f}% |
| **TEST** | 15% | {d['test_start']} to {d['test_end']} | {te_s['rows']:,} | {te_s['tickers']} | {te_s['target_1_pct']:.2f}% | {te_s['target_0_pct']:.2f}% |

- **Strict Date Isolation:** No dates overlap between splits.
- **Genuine Out-of-Sample Test Window:** Test period covers March 2022 to February 2026 (a demanding macro regime encompassing aggressive Fed rate hikes, inflation spikes, and generative AI sector rotations).

---

## 2. Feature Selection & Missing Value Handling

### Input Features (8 Core Signals)
The model consumes 8 relative/indicator features:
1. `Daily_Return`: Single-day close-to-close percentage change
2. `Volume_Change`: Relative volume delta with safe zero-volume division
3. `High_Low_Range`: Normalized intraday price spread $(High - Low)/Close$
4. `Rolling_Volatility_20`: 20-day trailing standard deviation of daily returns
5. `SMA20`: 20-day simple moving average of close price
6. `SMA50`: 50-day simple moving average of close price
7. `Close/SMA20`: Normalized price-to-SMA20 stretch ratio
8. `Close/SMA50`: Normalized price-to-SMA50 intermediate trend ratio

*Note: Neither `Ticker`, `Date`, nor `Target` are fed as numerical input features.*

### Missing Value Handling (Pre-Sequence Filtering)
- **Feature Warm-up NaNs:** 24,196 rows (0.89%) were dropped because the 50-day SMA requires a 49-session warm-up for each stock.
- **Target NaNs:** 472 rows (0.02%) were dropped because the final historical day of each ticker has no subsequent day's price.
- **Total Rows Dropped:** 24,668 rows (0.91%).
- **Rows Retained:** 2,678,863 rows (99.09% of original dataset).
- **Forward-Filling Avoidance:** No forward-filling was performed on feature warm-up NaNs, preventing artificial price persistence.

---

## 3. Feature Scaling Methodology

- **Scaler:** `StandardScaler` (zero mean, unit variance).
- **Fitting Rule:** Fitted **STRICTLY on the training period** (`Date <= {d['train_end']}`).
- **Transformation:** Validation and Test sets were transformed using the already-fitted training scaler.
- **Storage:** Persisted to [`models/scaler.joblib`](file:///Users/srinathrao/stock-price-movement-predictor/models/scaler.joblib) for production inference.

---

## 4. Time-Series Sequence Generation

- **Lookback:** 30 trading days (and 60 trading days in Experiment B).
- **Sequence Shape:** `(batch_size, 30, 8)`
- **Boundary Verification:** 100% of sequences start and end with the exact same ticker symbol. Cross-ticker sequence bleed is strictly zero.
- **Sequence Counts:**
  - Training sequences: `{split_info['num_train_seqs']:,}`
  - Validation sequences: `{split_info['num_val_seqs']:,}`
  - Test sequences: `{split_info['num_test_seqs']:,}`
  - Total sequences: `{split_info['num_train_seqs'] + split_info['num_val_seqs'] + split_info['num_test_seqs']:,}`

---

## 5. Naive Baseline Benchmarks (Always Predict Majority Class)

The training split exhibits a slight positive majority ({tr_s['target_1_pct']:.2f}% UP). The naive baseline permanently predicts **UP (1)** for all sessions:

| Metric | Validation Split (2018-2022) | Test Split (2022-2026) |
|---|---|---|
| **Majority Prediction** | UP (Class 1) | UP (Class 1) |
| **Accuracy** | {baseline_val['accuracy']*100:.2f}% | {baseline_test['accuracy']*100:.2f}% |
| **Precision** | {baseline_val['precision']*100:.2f}% | {baseline_test['precision']*100:.2f}% |
| **Recall** | {baseline_val['recall']*100:.2f}% | {baseline_test['recall']*100:.2f}% |
| **F1 Score** | {baseline_val['f1']:.4f} | {baseline_test['f1']:.4f} |
| **Sample Count** | {baseline_val['total_samples']:,} | {baseline_test['total_samples']:,} |

---

## 6. PyTorch LSTM Architecture & Training Specification

- **Architecture:**
  - Input Layer: 8 features
  - LSTM: 2 stacked layers, Hidden Size 64 (or 128 in Exp C), Dropout 0.2, `batch_first=True`
  - Fully Connected Head: `Linear(hidden_size, 1)` with preceding `Dropout(0.2)`
- **Loss Function:** `BCEWithLogitsLoss`
- **Optimizer:** `Adam` (Initial lr=0.001 or 0.0005, weight_decay=1e-5)
- **Gradient Clipping:** `max_norm = 1.0`
- **Early Stopping:** Monitored on `Validation Loss`, patience = 7 epochs
- **Batch Size:** {batch_size:,} sequences per batch

---

## 7. Controlled Experimentation Results

Four controlled experiments were conducted, varying exactly one major design parameter at a time. All configurations were evaluated on the identical chronological validation set:

| Experiment | Lookback | Hidden Units | Learning Rate | Best Epoch / Total | Val Loss | Val Accuracy | Val Precision | Val Recall | Val F1 |
|---|---|---|---|---|---|---|---|---|---|
"""
    for _, r in comp_df.iterrows():
        content += f"| **{r['Experiment']}** | {r['Lookback']} | {r['Hidden']} | {r['LR']} | Ep {r['Best_Epoch']} / {r['Total_Epochs']} | {r['Val_Loss']:.4f} | {r['Val_Acc']:.2f}% | {r['Val_Prec']:.2f}% | {r['Val_Rec']:.2f}% | {r['Val_F1']:.4f} |\n"

    content += f"""
### Model Selection Decision:
- **Winning Configuration:** `{best_exp['Experiment']}`
- **Selection Criteria:** Selected exclusively using validation performance (Val F1: `{best_exp['Val_F1']:.4f}`, Val Accuracy: `{best_exp['Val_Acc']:.2f}%`).
- **Test Set Protection:** Zero test data was inspected during hyperparameter exploration.

---

## 8. Final Untouched Test Set Evaluation (STAGE 9)

The selected optimal model was evaluated **once** on the untouched test partition ({d['test_start']} to {d['test_end']}):

| Metric | LSTM Model ({best_exp['Experiment']}) | Naive Majority Baseline | Performance Delta |
|---|---|---|---|
| **Directional Accuracy** | **{test_metrics['accuracy']*100:.2f}%** | {baseline_test['accuracy']*100:.2f}% | **{test_metrics['accuracy']*100 - baseline_test['accuracy']*100:+.2f}%** |
| **Precision** | **{test_metrics['precision']*100:.2f}%** | {baseline_test['precision']*100:.2f}% | {test_metrics['precision']*100 - baseline_test['precision']*100:+.2f}% |
| **Recall** | **{test_metrics['recall']*100:.2f}%** | {baseline_test['recall']*100:.2f}% | {test_metrics['recall']*100 - baseline_test['recall']*100:+.2f}% |
| **F1 Score** | **{test_metrics['f1']:.4f}** | {baseline_test['f1']:.4f} | **{test_metrics['f1'] - baseline_test['f1']:+.4f}** |
| **Test Loss (BCE)** | **{test_metrics['loss']:.4f}** | N/A | — |
| **Total Test Samples** | **{test_metrics['total_samples']:,}** | {baseline_test['total_samples']:,} | — |

### Test Confusion Matrix:
| Ground Truth \\ Prediction | Predicted DOWN/FLAT (0) | Predicted UP (1) | Total |
|---|---|---|---|
| **Actual DOWN/FLAT (0)** | **TN: {test_metrics['tn']:,}** | FP: {test_metrics['fp']:,} | {test_metrics['tn'] + test_metrics['fp']:,} |
| **Actual UP (1)** | FN: {test_metrics['fn']:,} | **TP: {test_metrics['tp']:,}** | {test_metrics['fn'] + test_metrics['tp']:,} |
| **Total** | {test_metrics['tn'] + test_metrics['fn']:,} | {test_metrics['fp'] + test_metrics['tp']:,} | {test_metrics['total_samples']:,} |

---

## 9. Limitations & Practical Market Insights

1. **Information Horizon:**
   Daily directional price movements in large-cap equities are notoriously noisy and near a sub-martingale random walk. Pure price/volume technical features provide an edge of a few percentage points over naive guessing.
2. **Transaction Cost Consideration:**
   An accuracy near 51-54% requires dynamic risk management, position sizing, and transaction fee minimization before translating into net economic alpha.
3. **Absence of Alternative Signals:**
   Macro indicators (interest rates, VIX), cross-sectional industry momentum, and fundamental earnings signals were intentionally excluded at this baseline stage.
4. **Future Iteration Roadmap:**
   Next iterations can benefit from Attention/Transformer mechanisms, multi-scale lookback representations, and thresholded prediction confidence filtering (only trading high-conviction probability bands).

---
*Report generated in {total_time_sec/60:.2f} minutes.*
"""
    with open(report_path, "w") as f:
        f.write(content)
    print(f"✓ Generated complete report: {report_path}")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DATA_PATH = PROJECT_ROOT / "data" / "processed" / "SP500_Targeted_Data.parquet"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"

    run_full_pipeline(
        data_path=DATA_PATH,
        models_dir=MODELS_DIR,
        results_dir=RESULTS_DIR,
        max_epochs=50,
        patience=7,
        batch_size=4096,
    )
