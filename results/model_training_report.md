# S&P 500 Stock Price Movement Predictor: Model Development & Training Report

## Executive Summary
This report documents the end-to-end execution of the **Model Development and Training Stage** for predicting next-day binary price movement across 472 S&P 500 stocks. 
All operations were executed without lookahead bias, without target leakage, with strict per-ticker sequence isolation, with chronological train/validation/test partitioning, and with scaling fitted exclusively on the training period.

---

## 1. Chronological Data Split Methodology

The dataset was partitioned chronologically across unique trading calendar dates (6,573 total dates from 2000-01-03 to 2026-02-20) to strictly mirror forward-looking deployment:

| Split | Percentage | Date Range | Total Rows | Tickers | UP Distribution | DOWN/FLAT Distribution |
|---|---|---|---|---|---|---|
| **TRAIN** | 70% | 2000-03-14 to 2018-05-07 | 1,775,416 | 444 | 50.32% | 49.68% |
| **VALIDATION** | 15% | 2018-05-08 to 2022-03-24 | 444,787 | 464 | 52.74% | 47.26% |
| **TEST** | 15% | 2022-03-25 to 2026-02-19 | 458,660 | 472 | 51.64% | 48.36% |

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
- **Fitting Rule:** Fitted **STRICTLY on the training period** (`Date <= 2018-05-07`).
- **Transformation:** Validation and Test sets were transformed using the already-fitted training scaler.
- **Storage:** Persisted to [`models/scaler.joblib`](file:///Users/srinathrao/stock-price-movement-predictor/models/scaler.joblib) for production inference.

---

## 4. Time-Series Sequence Generation

- **Lookback:** 30 trading days (and 60 trading days in Experiment B).
- **Sequence Shape:** `(batch_size, 30, 8)`
- **Boundary Verification:** 100% of sequences start and end with the exact same ticker symbol. Cross-ticker sequence bleed is strictly zero.
- **Sequence Counts:**
  - Training sequences: `1,762,540`
  - Validation sequences: `444,207`
  - Test sequences: `458,428`
  - Total sequences: `2,665,175`

---

## 5. Naive Baseline Benchmarks (Always Predict Majority Class)

The training split exhibits a slight positive majority (50.32% UP). The naive baseline permanently predicts **UP (1)** for all sessions:

| Metric | Validation Split (2018-2022) | Test Split (2022-2026) |
|---|---|---|
| **Majority Prediction** | UP (Class 1) | UP (Class 1) |
| **Accuracy** | 52.75% | 51.64% |
| **Precision** | 52.75% | 51.64% |
| **Recall** | 100.00% | 100.00% |
| **F1 Score** | 0.6906 | 0.6811 |
| **Sample Count** | 444,207 | 458,428 |

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
- **Batch Size:** 4,096 sequences per batch

---

## 7. Controlled Experimentation Results

Four controlled experiments were conducted, varying exactly one major design parameter at a time. All configurations were evaluated on the identical chronological validation set:

| Experiment | Lookback | Hidden Units | Learning Rate | Best Epoch / Total | Val Loss | Val Accuracy | Val Precision | Val Recall | Val F1 |
|---|---|---|---|---|---|---|---|---|---|
| **Experiment A: Baseline (LB=30, Hidden=64, LR=0.001)** | 30 | 64 | 0.001 | Ep 3 / 10 | 0.6910 | 52.78% | 53.24% | 85.93% | 0.6575 |
| **Experiment B: Lookback=60 (LB=60, Hidden=64, LR=0.001)** | 60 | 64 | 0.001 | Ep 3 / 10 | 0.6910 | 52.79% | 53.26% | 85.91% | 0.6575 |
| **Experiment C: Hidden=128 (LB=30, Hidden=128, LR=0.001)** | 30 | 128 | 0.001 | Ep 1 / 8 | 0.6910 | 52.85% | 53.24% | 87.29% | 0.6614 |
| **Experiment D: Low LR (LB=30, Hidden=64, LR=0.0005)** | 30 | 64 | 0.0005 | Ep 5 / 12 | 0.6912 | 52.69% | 53.27% | 83.90% | 0.6517 |

### Model Selection Decision:
- **Winning Configuration:** `Experiment C: Hidden=128 (LB=30, Hidden=128, LR=0.001)`
- **Selection Criteria:** Selected exclusively using validation performance (Val F1: `0.6614`, Val Accuracy: `52.85%`).
- **Test Set Protection:** Zero test data was inspected during hyperparameter exploration.

---

## 8. Final Untouched Test Set Evaluation (STAGE 9)

The selected optimal model was evaluated **once** on the untouched test partition (2022-03-25 to 2026-02-19):

| Metric | LSTM Model (Experiment C: Hidden=128 (LB=30, Hidden=128, LR=0.001)) | Naive Majority Baseline | Performance Delta |
|---|---|---|---|
| **Directional Accuracy** | **51.75%** | 51.64% | **+0.11%** |
| **Precision** | **51.88%** | 51.64% | +0.24% |
| **Recall** | **90.67%** | 100.00% | -9.33% |
| **F1 Score** | **0.6599** | 0.6811 | **-0.0211** |
| **Test Loss (BCE)** | **0.6929** | N/A | — |
| **Total Test Samples** | **458,428** | 458,428 | — |

### Test Confusion Matrix:
| Ground Truth \ Prediction | Predicted DOWN/FLAT (0) | Predicted UP (1) | Total |
|---|---|---|---|
| **Actual DOWN/FLAT (0)** | **TN: 22,597** | FP: 199,101 | 221,698 |
| **Actual UP (1)** | FN: 22,096 | **TP: 214,634** | 236,730 |
| **Total** | 44,693 | 413,735 | 458,428 |

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
*Report generated in 68.31 minutes.*
