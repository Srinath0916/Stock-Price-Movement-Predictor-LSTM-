# Model Card: S&P 500 Stock Price Movement Predictor (StockLSTM)

**Model Version:** `StockLSTM_ExpC_Hidden128`  
**Checkpoint File:** `models/best_lstm_model.pt`  
**Feature Scaler:** `models/scaler.joblib`  
**Architecture:** 2-Layer Long Short-Term Memory (LSTM) with Dropout (0.2) & Linear Classification Head  
**Date Created:** October 2026 (Updated November 2026)  
**Author:** Srinath Rao  
**Framework:** PyTorch 2.14, Scikit-Learn 1.9, FastAPI 0.142  

---

## 1. Model Overview & Intended Use

### Model Description
StockLSTM is a recurrent neural network designed for **binary directional classification of next-trading-day equity price movement** across constituents of the S&P 500. Given a 30-day chronological rolling window of 8 normalized technical and momentum features, the model outputs a raw logit and an estimated probability $P(\text{Close}_{t+1} > \text{Close}_t)$:

$$\text{Target}_t = \begin{cases} 1.0 & \text{if } \text{Close}_{t+1} > \text{Close}_t \\ 0.0 & \text{if } \text{Close}_{t+1} \le \text{Close}_t \end{cases}$$

### Intended Use
- **Primary Use Case:** Serving as a quantitative directional signal and momentum indicator to assist quantitative analysts and traders in screening prospective trades before market open.
- **Decision Support:** Intended solely as one quantitative input alongside human fundamental analysis, risk sizing, and macroeconomic context.
- **Operating Decision Threshold:** $\tau = 0.50$ (customizable up to $0.65$ for higher-conviction screening). Predictions with $P(\text{UP}) \ge \tau$ are classified as `UP`; otherwise `DOWN`.

### Non-Intended Use (Explicit Disclaimers)
- **NOT Financial Advice:** This model is not an automated financial advisor, broker, or financial planner.
- **NOT a Profit Guarantee:** A directional probability above 50% does not imply positive expected return, nor does it quantify the magnitude or duration of price changes.
- **NOT an Exact Price Predictor:** The model predicts directional orientation only; it does not forecast target price levels or volatility bounds.
- **NOT for Execution without Risk Management:** In live equity markets, trading without stop-loss orders, position sizing limits, and transaction cost modeling will lead to capital loss.

---

## 2. Training Data & Preprocessing

- **Dataset Scope:** 472 S&P 500 constituent stocks covering 6,573 unique trading days (January 3, 2000 to February 20, 2026), totaling 2,703,531 daily records.
- **Lossless Ingestion:** Zero-volume sessions (trading halts) and historical price spikes are preserved intact without synthetic smoothing.
- **Chronological Split (Strictly by Calendar Date):**
  - **Training Split (70%):** March 14, 2000 to May 07, 2018 (1,775,416 rows | 50.32% UP / 49.68% DOWN).
  - **Validation Split (15%):** May 08, 2018 to March 24, 2022 (444,787 rows | 52.74% UP / 47.26% DOWN).
  - **Holdout Test Split (15%):** March 25, 2022 to February 19, 2026 (458,660 rows | 51.64% UP / 48.36% DOWN).
- **Leakage Prevention & Scaling:** `StandardScaler` is fitted **strictly on the training partition** and persisted to `models/scaler.joblib`. Validation and test partitions are transformed using the fitted training statistics without re-fitting.
- **Eight Engineered Features (Computed Strictly per Ticker):**
  1. `Daily_Return`: Percentage close-to-close return $(Close_t - Close_{t-1}) / Close_{t-1}$.
  2. `Volume_Change`: Relative volume delta with safe zero-volume denominator handling.
  3. `High_Low_Range`: Normalized intraday volatility spread $(High_t - Low_t) / Close_t$.
  4. `Rolling_Volatility_20`: 20-day rolling standard deviation of `Daily_Return`.
  5. `SMA20`: 20-day Simple Moving Average of `Close`.
  6. `SMA50`: 50-day Simple Moving Average of `Close`.
  7. `Close/SMA20`: Price-to-SMA20 stretch ratio.
  8. `Close/SMA50`: Price-to-SMA50 intermediate trend ratio.

---

## 3. Architecture & Hyperparameters

- **Input Dimension:** 8 features $\times$ 30 time steps (shape: `[batch_size, 30, 8]`).
- **Hidden Size:** 128 units (selected in Experiment C for top validation F1).
- **Number of Layers:** 2 stacked LSTM layers (`batch_first=True`).
- **Inter-layer Dropout:** 0.20.
- **Output Head:** Dropout (0.20) $\rightarrow$ `Linear(128, 1)` producing raw logit.
- **Loss Function:** `nn.BCEWithLogitsLoss`.
- **Optimizer:** Adam ($\alpha = 0.001$, weight decay $= 10^{-5}$).
- **Gradient Clipping:** Max $L_2$ norm $= 1.0$.
- **Batch Size:** 4,096.

---

## 4. Evaluation Results & Benchmarks

### Untouched Out-of-Sample Test Set (March 25, 2022 to February 19, 2026; 458,428 holdout sequences)

| Metric | StockLSTM (Exp C) | Naive Baseline (Always UP) | Delta |
|---|:---:|:---:|:---:|
| **Directional Accuracy** | **51.75%** | 51.64% | **+0.11%** |
| **Precision** | **51.88%** | 51.64% | +0.24% |
| **Recall (Sensitivity)** | **90.67%** | 100.00% | -9.33% |
| **F1 Score** | **0.6599** | 0.6811 | -0.0212 |
| **BCE Loss** | **0.6929** | — | — |

### Holdout Test Confusion Matrix:
- **True Negatives (TN):** 22,597
- **False Positives (FP):** 199,101
- **False Negatives (FN):** 22,096
- **True Positives (TP):** 214,634

---

## 5. Walk-Forward Chronological Regime Stability

Evaluated across 5 distinct chronological market regimes covering 1,270,628 sequences:

| Market Regime | Period | Total Sequences | Accuracy | Baseline Acc | Delta Acc | F1 Score |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Late Bull / Low Volatility** | 2015-01-01 &rarr; 2017-12-31 | 329,476 | **52.47%** | 51.97% | **+0.50%** | 0.6629 |
| **Trade Wars & Volatility Spike** | 2018-01-01 &rarr; 2019-12-31 | 224,752 | **53.25%** | 53.39% | -0.14% | 0.6758 |
| **COVID Crash & Stimulus Rally** | 2020-01-01 &rarr; 2021-12-31 | 231,537 | **52.90%** | 52.53% | **+0.37%** | 0.6519 |
| **Rate Hikes & 2022 Bear Market** | 2022-01-01 &rarr; 2023-12-31 | 233,034 | **50.77%** | 50.69% | **+0.08%** | 0.6484 |
| **Disinflation & GenAI Secular Rally** | 2024-01-01 &rarr; 2026-02-19 | 251,829 | **52.29%** | 52.18% | **+0.11%** | 0.6667 |

---

## 6. Known Limitations & Failure Modes

1. **Secular Upward Drift Bias:**
   Because equity indexes trend upward over multi-decade periods, the model learns a high prior for positive days. On the test set, it predicts `UP` in 90.25% of sessions, producing high recall (90.67%) but a substantial count of false positives (199,101).
2. **Transaction Costs & Slippage Impact:**
   An out-of-sample directional edge of +0.11% is extremely thin. In live trading, broker commissions, exchange fees, and bid-ask spread friction can erode or reverse gross strategy alpha unless trade frequency is constrained.
3. **Absence of Alternative & Macro Signals:**
   The model intentionally uses only 8 price-volume technical indicators. It lacks real-time awareness of macroeconomic indicators (Fed interest rates, CPI inflation, yield curve inversions), corporate earnings announcements, or news sentiment.
4. **Degradation During Structural Regime Shifts:**
   During sharp macro pivots (e.g. initial aggressive Fed rate hikes in early 2022), directional accuracy dropped toward 50.77%, highlighting that historical momentum patterns have lower predictive power during sharp macroeconomic transitions.
