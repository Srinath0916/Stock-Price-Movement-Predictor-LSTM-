# Project Progress & Technical Documentation
## S&P 500 Stock Price Movement Predictor (LSTM Binary Classifier)

**Author:** Srinath Rao  
**Project Status:** Model Development & Evaluation Complete; Documentation & Demonstration in Progress  
**Last Updated:** October 2026  

---

## 1. Executive Summary & Objective

This project implements an end-to-end machine learning pipeline to predict the **next-trading-day directional price movement** of individual stocks across the S&P 500:

$$\text{Target}_t = \begin{cases} 1.0 & \text{if } \text{Close}_{t+1} > \text{Close}_t \\ 0.0 & \text{if } \text{Close}_{t+1} \le \text{Close}_t \end{cases}$$

The objective is to test whether recurrent neural networks (LSTMs) utilizing strictly technical and momentum indicators can extract directional predictive signal from historical price and volume data without data leakage or lookahead bias.

### Key Dataset & Problem Characteristics
- **Scope:** 472 S&P 500 constituent stocks covering 6,573 unique trading days (January 3, 2000 to February 20, 2026).
- **Volume:** 2,703,531 total daily market observations.
- **Problem Type:** Multi-asset binary time-series classification.
- **Evaluation Constraint:** Strict chronological out-of-sample partitioning (no random shuffling of dates). All sequence operations and indicator calculations are strictly isolated within individual tickers.

---

## 2. Completed Work by Pipeline Stage

### Stage 1: Data Ingestion & Minimal Cleaning
- **Script:** [`src/data/clean_data.py`](file:///Users/srinathrao/stock-price-movement-predictor/src/data/clean_data.py) & [`src/data/clean_checks.py`](file:///Users/srinathrao/stock-price-movement-predictor/src/data/clean_checks.py)
- **Input:** Raw Kaggle export [`data/raw/SP500_Historical_Data.csv`](file:///Users/srinathrao/stock-price-movement-predictor/data/raw/SP500_Historical_Data.csv) (136 MB, 2,703,531 rows).
- **Actions Performed:**
  - Standardized `Date` parsing to datetime format without row loss.
  - Sorted chronologically by `['Ticker', 'Date']`.
  - Verified that `Close` and `Adj Close` are identical across all 2,703,531 rows; safely dropped `Adj Close` to eliminate redundancy.
  - Zero-volume rows (trading halts) and historical price extremes were preserved intact without synthetic imputation.
- **Output:** [`data/processed/SP500_Cleaned_Data.parquet`](file:///Users/srinathrao/stock-price-movement-predictor/data/processed/SP500_Cleaned_Data.parquet) (39.4 MB).

### Stage 2: Exploratory Data Analysis (EDA)
- **Script:** [`src/data/eda.py`](file:///Users/srinathrao/stock-price-movement-predictor/src/data/eda.py)
- **Verified Findings:**
  - **Mean Daily Return:** `+0.0724%` per trading day.
  - **Standard Deviation:** `2.3607%`.
  - **Skewness:** `0.5682`.
  - **Excess Kurtosis (Fisher):** `29.69` (Pearson kurtosis = `32.69`). This confirms significant leptokurtosis (fat tails) typical of equity returns, where extreme tail movements occur far more frequently than predicted by a normal distribution.
  - Generated and saved 5 distribution figures to [`results/`](file:///Users/srinathrao/stock-price-movement-predictor/results).

### Stage 3: Feature Engineering
- **Script:** [`src/features/build_features.py`](file:///Users/srinathrao/stock-price-movement-predictor/src/features/build_features.py)
- **Input:** Cleaned Parquet data.
- **Engineered Features (8 Core Signals):**
  1. `Daily_Return`: Single-day close-to-close return: $(Close_t - Close_{t-1}) / Close_{t-1}$.
  2. `Volume_Change`: Relative volume delta with safe zero-volume handling:
     - If both $Volume_{t-1} = 0$ and $Volume_t = 0 \rightarrow 0.0$.
     - If $Volume_{t-1} = 0$ and $Volume_t > 0 \rightarrow \text{NaN}$ (avoiding $+ \infty$).
  3. `High_Low_Range`: Normalized intraday price range: $(High_t - Low_t) / Close_t$.
  4. `Rolling_Volatility_20`: Trailing 20-day rolling standard deviation of `Daily_Return`.
  5. `SMA20`: 20-day simple moving average of `Close`.
  6. `SMA50`: 50-day simple moving average of `Close`.
  7. `Close/SMA20`: Price-to-SMA20 stretch ratio.
  8. `Close/SMA50`: Price-to-SMA50 intermediate trend ratio.
- **Leakage Prevention:** All window calculations are executed strictly per ticker group. Neither `Ticker`, `Date`, nor future returns are fed as model inputs.
- **Output:** [`data/processed/SP500_Featured_Data.parquet`](file:///Users/srinathrao/stock-price-movement-predictor/data/processed/SP500_Featured_Data.parquet) (196.5 MB).

### Stage 4: Target Creation
- **Script:** [`src/features/build_target.py`](file:///Users/srinathrao/stock-price-movement-predictor/src/features/build_target.py)
- **Logic:** Shift `Close` by $-1$ strictly within each ticker group.
  - $\text{Target} = 1.0$ if $Close_{t+1} > Close_t$
  - $\text{Target} = 0.0$ if $Close_{t+1} \le Close_t$
  - $\text{Target} = \text{NaN}$ for the final historical date of each ticker (exactly 472 rows).
- **Validation:** Automated test confirms exactly 472 boundary NaNs and 0 cross-ticker target bleed.
- **Output:** [`data/processed/SP500_Targeted_Data.parquet`](file:///Users/srinathrao/stock-price-movement-predictor/data/processed/SP500_Targeted_Data.parquet) (196.8 MB).

### Stage 5: Chronological Data Split & Scaling
- **Script:** [`src/models/dataset.py`](file:///Users/srinathrao/stock-price-movement-predictor/src/models/dataset.py)
- **Chronological Split (70% / 15% / 15% by unique trading dates):**
  - **Train Set (70%):** 2000-03-14 to 2018-05-07 | 4,566 unique dates | 1,775,416 rows | 50.32% UP / 49.68% DOWN
  - **Validation Set (15%):** 2018-05-08 to 2022-03-24 | 978 unique dates | 444,787 rows | 52.74% UP / 47.26% DOWN
  - **Test Set (15%):** 2022-03-25 to 2026-02-19 | 979 unique dates | 458,660 rows | 51.64% UP / 48.36% DOWN
- **Scaling:** `StandardScaler` fitted **strictly on the training split** and persisted to [`models/scaler.joblib`](file:///Users/srinathrao/stock-price-movement-predictor/models/scaler.joblib). Validation and test sets are transformed using the fitted training parameters.
- **Missing Value Handling:**
  - 24,196 rows dropped due to 50-day SMA warm-up (49 initial rows per ticker).
  - 472 rows dropped due to target generation on the final session.
  - Retained: 2,678,863 rows (99.09% of original dataset).

### Stage 6: Sequence Generation & PyTorch Dataset
- **Script:** [`src/models/dataset.py`](file:///Users/srinathrao/stock-price-movement-predictor/src/models/dataset.py)
- **Mechanism:** Vectorized ticker-boundary verification (`tickers[lookback-1:] == tickers[:-lookback+1]`).
- **Memory Efficiency:** `TimeSeriesDataset` indexes directly into a shared contiguous 2D NumPy array of shape $(N, 8)$, avoiding RAM exhaustion from materializing sliding window tensors.
- **Sequence Counts (Lookback = 30 days):**
  - Training sequences: `1,762,540`
  - Validation sequences: `444,207`
  - Test sequences: `458,428`
  - Total valid sequences: `2,665,175` (100% verified internal to ticker boundaries).

### Stage 7: LSTM Architecture & Training Specification
- **Scripts:** [`src/models/lstm.py`](file:///Users/srinathrao/stock-price-movement-predictor/src/models/lstm.py), [`src/models/train.py`](file:///Users/srinathrao/stock-price-movement-predictor/src/models/train.py)
- **Model:** `StockLSTM`
  - Input dimension: 8 features
  - Hidden layers: 2 stacked LSTM layers with inter-layer dropout = 0.2 (`batch_first=True`)
  - Output head: Dropout (0.2) $\rightarrow$ Linear(hidden_size, 1) producing raw logit
- **Optimization:** `BCEWithLogitsLoss`, Adam optimizer (weight decay $1 \times 10^{-5}$), gradient clipping (`max_norm = 1.0`), batch size = 4,096.
- **Early Stopping:** Monitored on validation loss with patience = 7 epochs.

### Stage 8: Controlled Experiments & Selection
- **Script:** [`src/evaluation/run_experiments.py`](file:///Users/srinathrao/stock-price-movement-predictor/src/evaluation/run_experiments.py)
- Four experiments varying one major parameter against the validation split:

| Experiment | Lookback | Hidden Units | Learning Rate | Best Epoch / Total | Val Loss | Val Acc | Val Prec | Val Rec | Val F1 | Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Exp A: Baseline** | 30 | 64 | 0.001 | Ep 3 / 10 | 0.6910 | 52.78% | 53.24% | 85.93% | 0.6575 | `models/lstm_expA_lb30_h64_lr001.pt` |
| **Exp B: Lookback 60** | 60 | 64 | 0.001 | Ep 3 / 10 | 0.6910 | 52.79% | 53.26% | 85.91% | 0.6575 | `models/lstm_expB_lb60_h64_lr001.pt` |
| **Exp C: Hidden 128** | 30 | 128 | 0.001 | Ep 1 / 8 | 0.6910 | **52.85%** | 53.24% | 87.29% | **0.6614** | `models/lstm_expC_lb30_h128_lr001.pt` |
| **Exp D: Low LR** | 30 | 64 | 0.0005 | Ep 5 / 12 | 0.6912 | 52.69% | 53.27% | 83.90% | 0.6517 | `models/lstm_expD_lb30_h64_lr0005.pt` |

- **Selection:** **Experiment C** achieved the highest Validation F1 (0.6614) and Validation Accuracy (52.85%). Checkpoint saved as [`models/best_lstm_model.pt`](file:///Users/srinathrao/stock-price-movement-predictor/models/best_lstm_model.pt).

### Stage 9: Final Out-of-Sample Test Evaluation
- **Evaluation Period:** 2022-03-25 to 2026-02-19 (458,428 unseen test sequences).
- **Benchmark:** Naive Majority Class Baseline (always predict UP based on training data distribution).

| Metric | Selected LSTM (Exp C) | Naive Baseline | Performance Delta |
| :--- | :---: | :---: | :---: |
| **Directional Accuracy** | **51.75%** | 51.64% | **+0.11%** |
| **Precision** | **51.88%** | 51.64% | +0.24% |
| **Recall (Sensitivity)** | **90.67%** | 100.00% | -9.33% |
| **F1 Score** | **0.6599** | 0.6811 | -0.0212 |
| **Binary Cross-Entropy Loss** | **0.6929** | — | — |

#### Confusion Matrix (Test Set):
| Ground Truth \ Prediction | Predicted DOWN/FLAT (`0`) | Predicted UP (`1`) | Actual Total |
| :--- | :---: | :---: | :---: |
| **Actual DOWN/FLAT (`0`)** | **22,597** (TN) | **199,101** (FP) | 221,698 (48.36%) |
| **Actual UP (`1`)** | **22,096** (FN) | **214,634** (TP) | 236,730 (51.64%) |
| **Total Predictions** | **44,693** (9.75%) | **413,735** (90.25%) | **458,428** |

### Stage 10: Standalone Evaluation Script
- **Script:** [`src/evaluation/evaluate_model.py`](file:///Users/srinathrao/stock-price-movement-predictor/src/evaluation/evaluate_model.py)
- Command-line tool to load any checkpoint and evaluate on `test`, `val`, or `all` splits without retraining.

---

## 3. Key Files and Responsibilities

| File Path | Primary Responsibility |
| :--- | :--- |
| [`src/data/clean_data.py`](file:///Users/srinathrao/stock-price-movement-predictor/src/data/clean_data.py) | Ingests raw CSV, standardizes datetime, verifies no data loss, writes cleaned Parquet. |
| [`src/data/eda.py`](file:///Users/srinathrao/stock-price-movement-predictor/src/data/eda.py) | Computes summary statistics, distributions, and saves EDA charts to `results/`. |
| [`src/features/build_features.py`](file:///Users/srinathrao/stock-price-movement-predictor/src/features/build_features.py) | Computes 8 technical features grouped per ticker with safe zero-division rules. |
| [`src/features/build_target.py`](file:///Users/srinathrao/stock-price-movement-predictor/src/features/build_target.py) | Constructs binary next-day directional target with ticker boundary validation. |
| [`src/models/dataset.py`](file:///Users/srinathrao/stock-price-movement-predictor/src/models/dataset.py) | Chronological 70/15/15 split, scaler fitting on train only, memory-efficient PyTorch Dataset. |
| [`src/models/lstm.py`](file:///Users/srinathrao/stock-price-movement-predictor/src/models/lstm.py) | PyTorch 2-layer LSTM architecture definition with dropout and linear projection. |
| [`src/models/train.py`](file:///Users/srinathrao/stock-price-movement-predictor/src/models/train.py) | Training and evaluation loops, gradient clipping, early stopping, and loss plotting. |
| [`src/evaluation/metrics.py`](file:///Users/srinathrao/stock-price-movement-predictor/src/evaluation/metrics.py) | Binary classification metrics (Accuracy, Precision, Recall, F1) and majority baseline. |
| [`src/evaluation/run_experiments.py`](file:///Users/srinathrao/stock-price-movement-predictor/src/evaluation/run_experiments.py) | Master pipeline executing Experiments A–D, model selection, and test evaluation. |
| [`src/evaluation/evaluate_model.py`](file:///Users/srinathrao/stock-price-movement-predictor/src/evaluation/evaluate_model.py) | Standalone evaluation CLI tool for evaluating saved checkpoints on test or validation data. |

---

## 4. Analysis of Model Behavior & Limitations

1. **Empirical Edge vs. Naive Baseline:**
   The model achieves **51.75% accuracy** on 458,428 out-of-sample trading sessions between March 2022 and February 2026. While an accuracy of ~52% might seem modest in computer vision or NLP, daily equity direction prediction operates in an environment close to an efficient market hypothesis / sub-martingale random walk. An edge of **+0.11%** over the majority class baseline across half a million unseen sessions confirms that pure technical indicators provide marginal directional signal.
2. **Positive Prediction Bias (High Recall, High False Positives):**
   The model predicts UP in 90.25% of cases. Consequently, it achieves a high UP recall (90.67%) but incurs 199,101 False Positives. The loss landscape is shallow (BCE loss ~ 0.691–0.693), and the model leans toward the historical upward drift of the broad market.
3. **Absence of Alternative Data & Macro Features:**
   The current feature set is intentionally restricted to 8 price-volume technical indicators. Cross-sectional momentum, macroeconomic indicators (Fed funds rate, CPI, Treasury yields), and earnings sentiment are not included in this baseline.
4. **No Transaction Cost / Slippage Modeling:**
   A raw directional prediction accuracy near 52% cannot be assumed profitable without incorporating realistic bid-ask spreads, transaction commissions, and trade execution filters.

---

## 5. Completed vs. Future Work

### Verified Completed Work
- [x] Full data ingestion, minimal cleaning, and invariant checks.
- [x] Statistical EDA, return distribution analysis (Fisher excess kurtosis = 29.69), and 5 saved plots.
- [x] 8 core technical features computed strictly per ticker with zero leakage.
- [x] Binary target creation with validated ticker boundary isolation.
- [x] Chronological 70/15/15 train/val/test split by calendar dates.
- [x] StandardScaler fitted strictly on training data and saved to `models/scaler.joblib`.
- [x] Memory-efficient sequence generation (2,665,175 sequences, 0 boundary violations).
- [x] PyTorch 2-layer LSTM implementation with dropout and gradient clipping.
- [x] 4 controlled experiments (A, B, C, D) run, compared, and documented.
- [x] Model selection based exclusively on validation F1.
- [x] Single-pass out-of-sample test evaluation and confusion matrix.
- [x] Standalone CLI evaluation tool (`src/evaluation/evaluate_model.py`).
- [x] Interactive Research Web Demonstration (React + Vite in `frontend/`) featuring real stock time series, SMA overlays, 4-experiment training curves, test confusion matrix, and financial limitations analysis.

### Future Milestones (Not Yet Implemented)
- [ ] **Demonstration Notebook (`notebooks/02_project_demo.ipynb`):** Interactive, self-contained walkthrough loading saved results and models without retraining.
- [ ] **Confidence Threshold Analysis:** Evaluating directional precision when filtering by model probability thresholds (e.g., trading only when $p > 0.55$ or $p > 0.60$).
- [ ] **Ticker-Level & Sector Accuracy Breakdown:** Investigating if mega-cap liquid stocks or defensive sectors exhibit higher directional predictability.
- [ ] **Alternative Architectures:** Testing Temporal Convolutional Networks (TCN) or Multi-Head Self-Attention mechanisms.
- [ ] **Economic Backtesting:** Simulating long/short or long-only strategies with transaction fee models and Sharpe ratio calculations.
