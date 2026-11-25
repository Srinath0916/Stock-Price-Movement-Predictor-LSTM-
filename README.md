# S&P 500 Stock Price Movement Predictor

A production-grade quantitative machine learning system that predicts the **next-trading-day directional price movement** of S&P 500 constituent stocks using a 2-layer Long Short-Term Memory (LSTM) neural network, technical momentum features, a high-performance **FastAPI** backend service, and an interactive **React + Vite** financial dashboard.

$$\text{Target}_t = \begin{cases} 1.0 & \text{if } \text{Close}_{t+1} > \text{Close}_t \\ 0.0 & \text{if } \text{Close}_{t+1} \le \text{Close}_t \end{cases}$$

- **Documentation & Verification:** [`docs/PROJECT_PROGRESS.md`](docs/PROJECT_PROGRESS.md)
- **Formal Model Card:** [`MODEL_CARD.md`](MODEL_CARD.md)

---

## End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Data Pipeline
        A["data/raw/SP500_Historical_Data.csv (472 Tickers, 2.7M Rows)"] --> B["src/data/clean_data.py"]
        B --> C["data/processed/SP500_Cleaned_Data.parquet"]
        C --> D["src/features/build_features.py (8 Core Features)"]
        D --> E["data/processed/SP500_Featured_Data.parquet"]
        E --> F["src/features/build_target.py (Binary Target)"]
        F --> G["data/processed/SP500_Targeted_Data.parquet"]
    end

    subgraph Model Development & Validation
        G --> H["src/models/dataset.py (Chronological 70/15/15 Split)"]
        H --> I["models/scaler.joblib (Train Scaler)"]
        H --> J["src/models/lstm.py (StockLSTM: 2-layer, 128 hidden)"]
        J --> K["src/models/train.py & run_experiments.py"]
        K --> L["models/best_lstm_model.pt (Experiment C Checkpoint)"]
        L --> M["src/evaluation/walk_forward.py (5 Market Regimes)"]
        L --> N["src/evaluation/backtest.py (PnL Simulation & Sharpe)"]
    end

    subgraph FastAPI REST Service (:8000)
        L & I & G --> O["src/api/app.py (FastAPI App)"]
        O --> P["GET/POST /api/predict (Live PyTorch Inference)"]
        O --> Q["GET/POST /api/backtest (Strategy Simulation)"]
        O --> R["GET /api/walk-forward (Regime Breakdown)"]
        O --> S["GET /api/tickers & /api/stock/{ticker}"]
    end

    subgraph React + Vite Financial Dashboard (:5173)
        P & Q & R & S --> T["frontend/src/App.jsx"]
        T --> U["StockChart.jsx (SVG Candlesticks & SMAs)"]
        T --> V["PredictionCard.jsx (Live PyTorch Call)"]
        T --> W["BacktestView.jsx (Equity Curve & Alpha)"]
        T --> X["ModelPerformanceView.jsx (Scorecard)"]
    end

    style A fill:#1e293b,stroke:#475569,color:#f8fafc
    style L fill:#064e3b,stroke:#10b981,color:#f8fafc
    style O fill:#1e1b4b,stroke:#6366f1,color:#f8fafc
    style T fill:#0c4a6e,stroke:#0284c7,color:#f8fafc
```

---

## Dataset & Technical Indicators

- **Dataset Scope:** 472 S&P 500 stocks across 6,573 trading sessions (January 2000 to February 2026), totaling 2,703,531 daily observations.
- **8 Core Engineered Features (Zero Lookahead, Computed Strictly per Ticker):**
  1. `Daily_Return`: Single-day close-to-close return $(Close_t - Close_{t-1}) / Close_{t-1}$.
  2. `Volume_Change`: Relative volume delta with safe zero-volume handling.
  3. `High_Low_Range`: Intraday range normalized by close price $(High_t - Low_t) / Close_t$.
  4. `Rolling_Volatility_20`: Trailing 20-day standard deviation of daily return.
  5. `SMA20`: 20-day simple moving average of close price.
  6. `SMA50`: 50-day simple moving average of close price.
  7. `Close/SMA20`: Price-to-20SMA mean-reversion stretch ratio.
  8. `Close/SMA50`: Price-to-50SMA intermediate trend ratio.
- **Chronological 70/15/15 Data Split:**
  - **Train Set:** 2000-03-14 to 2018-05-07 (1,775,416 rows | 50.32% UP)
  - **Validation Set:** 2018-05-08 to 2022-03-24 (444,787 rows | 52.74% UP)
  - **Holdout Test Set:** 2022-03-25 to 2026-02-19 (458,660 rows | 51.64% UP)
- **Feature Scaling:** `StandardScaler` fitted **strictly on the training split** and persisted to [`models/scaler.joblib`](models/scaler.joblib).
- **Sequence Generation:** 30-day sliding window sequences (2,665,175 total valid sequences; 0 cross-ticker violations).

---

## Out-of-Sample Performance (Untouched Test Set)

Evaluated on the out-of-sample holdout partition (**March 25, 2022 to February 19, 2026**, 458,428 sequences):

| Metric | Selected LSTM (Exp C) | Naive Baseline (Always UP) | Net Difference |
| :--- | :---: | :---: | :---: |
| **Directional Accuracy** | **51.75%** | 51.64% | **+0.11%** |
| **Precision** | **51.88%** | 51.64% | +0.24% |
| **Recall (Sensitivity)** | **90.67%** | 100.00% | -9.33% |
| **F1 Score** | **0.6599** | 0.6811 | -0.0212 |
| **Binary Cross-Entropy Loss** | **0.6929** | — | — |

### Test Confusion Matrix:
- **True Negatives (TN):** 22,597 | **False Positives (FP):** 199,101
- **False Negatives (FN):** 22,096 | **True Positives (TP):** 214,634

---

## Walk-Forward Chronological Regime Stability

Evaluated across 5 distinct chronological market regimes covering 1,270,628 sequences:

| Market Regime | Period | Total Sequences | Accuracy | Baseline Acc | Delta Acc | F1 Score |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Late Bull / Low Volatility** | 2015-01-01 &rarr; 2017-12-31 | 329,476 | **52.47%** | 51.97% | **+0.50%** | 0.6629 |
| **Trade Wars & Volatility Spike** | 2018-01-01 &rarr; 2019-12-31 | 224,752 | **53.25%** | 53.39% | -0.14% | 0.6758 |
| **COVID Crash & Stimulus Rally** | 2020-01-01 &rarr; 2021-12-31 | 231,537 | **52.90%** | 52.53% | **+0.37%** | 0.6519 |
| **Rate Hikes & 2022 Bear Market** | 2022-01-01 &rarr; 2023-12-31 | 233,034 | **50.77%** | 50.69% | **+0.08%** | 0.6484 |
| **Disinflation & GenAI Secular Rally** | 2024-01-01 &rarr; 2026-02-19 | 251,829 | **52.29%** | 52.18% | **+0.11%** | 0.6667 |

---

## Getting Started & Running the Complete Stack

### 1. Environment Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt
```

### 2. Start the FastAPI Backend Service

```bash
# Run FastAPI server on port 8000
.venv/bin/uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

The interactive OpenAPI documentation is available at **`http://localhost:8000/docs`**.

### 3. Start the React + Vite Dashboard

In a separate terminal:

```bash
# Navigate to frontend and start Vite server
cd frontend
npm run dev
```

Open your browser at **`http://localhost:5173/`**.

---

## FastAPI REST API Reference

| Endpoint | Method | Description | Example Query |
|---|:---:|---|---|
| `/api/health` | `GET` | Health check, device, loaded model version | `curl http://localhost:8000/api/health` |
| `/api/predict` | `GET/POST` | Executes PyTorch forward pass on 30-day window | `curl "http://localhost:8000/api/predict?ticker=AAPL"` |
| `/api/backtest` | `GET/POST` | Runs economic backtest with simulated PnL & Sharpe | `curl "http://localhost:8000/api/backtest?ticker=AAPL&strategy_mode=long_only"` |
| `/api/walk-forward` | `GET` | Returns 5 chronological market regime results | `curl http://localhost:8000/api/walk-forward` |
| `/api/tickers` | `GET` | Lists all 472 available S&P 500 tickers with latest prices | `curl http://localhost:8000/api/tickers` |
| `/api/stock/{ticker}` | `GET` | Returns OHLCV trading records + 8 indicators | `curl "http://localhost:8000/api/stock/AAPL?days=252"` |

---

## Running Offline Evaluation & Backtesting CLIs

```bash
# 1. Standalone test set accuracy check
.venv/bin/python src/evaluation/evaluate_model.py

# 2. Walk-forward 5-regime chronological validation
.venv/bin/python src/evaluation/walk_forward.py

# 3. Quantitative economic backtesting CLI (with transaction costs)
.venv/bin/python src/evaluation/backtest.py --ticker AAPL --mode long_only --fee 5.0
```

---

## Project Structure

```text
stock-price-movement-predictor/
├── data/
│   ├── raw/                 # Original dataset (SP500_Historical_Data.csv)
│   └── processed/           # Cleaned, featured, and targeted Parquet datasets
├── docs/
│   └── PROJECT_PROGRESS.md  # Detailed technical documentation and metrics
├── frontend/                # React 19 + Vite financial research dashboard
│   ├── src/
│   │   ├── components/      # UI components (Chart, PredictionCard, BacktestView)
│   │   ├── data/            # Local static fallback cache
│   │   └── App.jsx          # Root application shell
│   └── vite.config.js       # Vite dev proxy configuration
├── models/                  # Saved model checkpoints (.pt) and scaler.joblib
│   ├── best_lstm_model.pt   # Canonical selected Experiment C weights (128 hidden)
│   └── scaler.joblib        # StandardScaler fitted on training split
├── results/                 # Evaluation JSONs, walk-forward records, and PNG plots
├── src/
│   ├── api/                 # FastAPI REST application (app.py, schemas.py, services.py)
│   ├── data/                # Data cleaning and EDA routines
│   ├── features/            # 8 engineered indicators and binary target creation
│   ├── models/              # StockLSTM architecture, dataset pipeline, training loop
│   └── evaluation/          # Metrics, experiments runner, walk-forward, and backtesting
├── MODEL_CARD.md            # Standardized ML model card
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation and architecture
```
