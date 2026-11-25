"""
Exports verified dataset samples, experiment records, and evaluation metrics
from the Python ML pipeline into frontend/src/data/projectData.json.
Provides 3 years of daily trading sessions plus full all-time monthly points
for dynamic date range filtering (1M, 3M, 6M, 1Y, 3Y, ALL).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import numpy as np
import pandas as pd

FRONTEND_DATA_DIR = PROJECT_ROOT / "frontend" / "src" / "data"
FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Parquet Data
targeted_parquet = PROJECT_ROOT / "data" / "processed" / "SP500_Targeted_Data.parquet"
print(f"Reading {targeted_parquet}...")
df = pd.read_parquet(targeted_parquet)

# 2. Target Tickers for Interactive Charting
SAMPLE_TICKERS = {
    "AAPL": {"name": "Apple Inc.", "sector": "Information Technology", "desc": "Global consumer hardware, software ecosystems, and cloud subscription services."},
    "MSFT": {"name": "Microsoft Corp.", "sector": "Information Technology", "desc": "Enterprise cloud infrastructure (Azure), enterprise software, and AI platforms."},
    "NVDA": {"name": "NVIDIA Corp.", "sector": "Semiconductors", "desc": "High-performance GPU acceleration, enterprise data center hardware, and AI silicon."},
    "JPM": {"name": "JPMorgan Chase & Co.", "sector": "Financial Services", "desc": "Global investment banking, commercial treasury services, and asset management."},
    "JNJ": {"name": "Johnson & Johnson", "sector": "Healthcare", "desc": "Diversified pharmaceuticals, medical technologies, and biotechnology research."},
    "CAT": {"name": "Caterpillar Inc.", "sector": "Industrials", "desc": "Global industrial manufacturing of construction, resource extraction, and mining equipment."},
    "AMZN": {"name": "Amazon.com Inc.", "sector": "Consumer Discretionary", "desc": "E-commerce logistics, cloud compute (AWS), and digital media platforms."},
    "XOM": {"name": "Exxon Mobil Corp.", "sector": "Energy", "desc": "Upstream petroleum exploration, natural gas production, and petrochemical refining."},
}

ticker_samples = {}
for ticker, meta in SAMPLE_TICKERS.items():
    sub = df[df["Ticker"] == ticker].sort_values(by="Date").reset_index(drop=True)
    if len(sub) == 0:
        continue

    total_days = len(sub)
    up_pct = float((sub["Target"] == 1.0).sum() / (sub["Target"].notna().sum()) * 100)
    avg_return = float(sub["Daily_Return"].mean() * 100)
    volatility = float(sub["Daily_Return"].std() * 100)
    
    # 52-week stats (last 252 sessions)
    past_year = sub.tail(252)
    high_52w = float(past_year["High"].max())
    low_52w = float(past_year["Low"].min())
    ret_1y = float((past_year["Close"].iloc[-1] - past_year["Close"].iloc[0]) / past_year["Close"].iloc[0] * 100)

    # Take up to 756 sessions (3 full trading years: 2023-2026) for daily resolution
    daily_slice = sub.tail(756).copy()
    
    records = []
    for _, row in daily_slice.iterrows():
        records.append({
            "date": pd.to_datetime(row["Date"]).strftime("%Y-%m-%d"),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
            "dailyReturn": round(float(row["Daily_Return"]) * 100, 2) if pd.notna(row["Daily_Return"]) else None,
            "volumeChange": round(float(row["Volume_Change"]) * 100, 2) if pd.notna(row["Volume_Change"]) else None,
            "highLowRange": round(float(row["High_Low_Range"]) * 100, 2) if pd.notna(row["High_Low_Range"]) else None,
            "rollingVol20": round(float(row["Rolling_Volatility_20"]) * 100, 2) if pd.notna(row["Rolling_Volatility_20"]) else None,
            "sma20": round(float(row["SMA20"]), 2) if pd.notna(row["SMA20"]) else None,
            "sma50": round(float(row["SMA50"]), 2) if pd.notna(row["SMA50"]) else None,
            "closeSma20": round(float(row["Close/SMA20"]), 4) if pd.notna(row["Close/SMA20"]) else None,
            "closeSma50": round(float(row["Close/SMA50"]), 4) if pd.notna(row["Close/SMA50"]) else None,
            "target": int(row["Target"]) if pd.notna(row["Target"]) else None,
        })

    # All-time downsampled history (take monthly end sessions for smooth 2000-2026 macro chart)
    sub_monthly = sub.set_index("Date").resample("ME").last().dropna(subset=["Close"]).reset_index()
    all_time_records = []
    for _, row in sub_monthly.iterrows():
        all_time_records.append({
            "date": pd.to_datetime(row["Date"]).strftime("%Y-%m-%d"),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        })

    ticker_samples[ticker] = {
        "ticker": ticker,
        "name": meta["name"],
        "sector": meta["sector"],
        "description": meta["desc"],
        "stats": {
            "totalSessions": total_days,
            "startDate": pd.to_datetime(sub["Date"].iloc[0]).strftime("%Y-%m-%d"),
            "endDate": pd.to_datetime(sub["Date"].iloc[-1]).strftime("%Y-%m-%d"),
            "upPercent": round(up_pct, 2),
            "avgReturn": round(avg_return, 4),
            "annualizedVol": round(volatility * np.sqrt(252), 2),
            "latestPrice": round(float(sub["Close"].iloc[-1]), 2),
            "high52w": round(high_52w, 2),
            "low52w": round(low_52w, 2),
            "return1y": round(ret_1y, 2),
        },
        "history": records,
        "allTimeHistory": all_time_records,
    }

# 3. Load Scaler Parameters
scaler_path = PROJECT_ROOT / "models" / "scaler.joblib"
scaler = joblib.load(scaler_path)
feature_names = [
    "Daily_Return", "Volume_Change", "High_Low_Range", "Rolling_Volatility_20",
    "SMA20", "SMA50", "Close/SMA20", "Close/SMA50"
]
scaler_info = [
    {
        "feature": name,
        "mean": float(m),
        "std": float(s),
        "description": {
            "Daily_Return": "Single-day percentage return (Close_t - Close_{t-1}) / Close_{t-1}",
            "Volume_Change": "Relative volume delta with zero-volume safe division",
            "High_Low_Range": "Intraday trading spread (High - Low) / Close",
            "Rolling_Volatility_20": "20-day trailing rolling standard deviation of daily return",
            "SMA20": "20-day Simple Moving Average of Close price",
            "SMA50": "50-day Simple Moving Average of Close price",
            "Close/SMA20": "Price to 20-day SMA stretch ratio",
            "Close/SMA50": "Price to 50-day SMA intermediate trend ratio",
        }.get(name, "")
    }
    for name, m, s in zip(feature_names, scaler.mean_, scaler.scale_)
]

# 4. Load Results
comp_df = pd.read_csv(PROJECT_ROOT / "results" / "experiment_comparison.csv")
experiments = comp_df.to_dict(orient="records")

with open(PROJECT_ROOT / "results" / "final_test_results.json") as f:
    final_test = json.load(f)

with open(PROJECT_ROOT / "results" / "naive_baseline_results.json") as f:
    naive_baseline = json.load(f)

# 5. Compute Genuine PyTorch Model Predictions for Latest 30-Day Window
import torch
from src.evaluation.evaluate_model import load_model_from_checkpoint

device = torch.device("cpu")
model_path = PROJECT_ROOT / "models" / "best_lstm_model.pt"
model, hidden_size = load_model_from_checkpoint(model_path, device)

predictions = {}
for ticker, meta in SAMPLE_TICKERS.items():
    sub = df[df["Ticker"] == ticker].sort_values(by="Date").reset_index(drop=True)
    if len(sub) < 30:
        continue
    last30 = sub.tail(30).copy()
    feat_vals = last30[feature_names].values
    scaled = scaler.transform(feat_vals)
    tensor_in = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logit = float(model(tensor_in).item())
        prob = float(torch.sigmoid(torch.tensor(logit)).item())
    pred_dir = "UP" if prob >= 0.5 else "DOWN"
    last_date = pd.to_datetime(last30.iloc[-1]["Date"]).strftime("%Y-%m-%d")
    window_start = pd.to_datetime(last30.iloc[0]["Date"]).strftime("%Y-%m-%d")
    last_close = float(last30.iloc[-1]["Close"])
    conf = prob if prob >= 0.5 else (1.0 - prob)

    predictions[ticker] = {
        "ticker": ticker,
        "companyName": meta["name"],
        "inputDate": last_date,
        "windowStartDate": window_start,
        "windowLength": 30,
        "targetSession": "Next Trading Session (Feb 23, 2026)",
        "lastClose": round(last_close, 2),
        "prediction": pred_dir,
        "probability": round(prob, 4),
        "confidencePct": round(conf * 100, 2),
        "logit": round(logit, 4),
        "modelCheckpoint": "models/best_lstm_model.pt",
        "scalerUsed": "models/scaler.joblib",
        "featureCount": 8,
        "explanation": f"The PyTorch LSTM model evaluated the 30 consecutive trading sessions from {window_start} to {last_date}. Accounting for normalized momentum, volume dynamics, volatility, and SMA stretch, the model outputs a {pred_dir} movement probability of {prob * 100:.1f}%.",
        "caution": "Experimental prediction based on 51.75% out-of-sample directional accuracy. This does not guarantee future results and is not financial advice."
    }

# 5. Assemble Payload
payload = {
    "project": {
        "title": "S&P 500 Stock Movement Predictor",
        "subtitle": "Explore historical stock prices and predict whether a stock may move UP or DOWN in the next trading session using an LSTM model.",
        "author": "Srinath Rao",
        "date": "October 2026",
        "githubRepo": "https://github.com/Srinath0916/Stock-Price-Movement-Predictor-LSTM-",
    },
    "dataset": {
        "rawRows": 2703531,
        "cleanRows": 2678863,
        "droppedRows": 24668,
        "droppedWarmupSma": 24196,
        "droppedFinalTarget": 472,
        "uniqueTickers": 472,
        "totalTradingDays": 6573,
        "startDate": "2000-01-03",
        "endDate": "2026-02-20",
        "splits": {
            "train": {
                "name": "Training",
                "percent": 70,
                "startDate": "2000-03-14",
                "endDate": "2018-05-07",
                "datesCount": 4566,
                "rows": 1775416,
                "sequences": 1762540,
                "upPercent": 50.32,
                "downPercent": 49.68,
                "regime": "Dot-com burst, 2008 GFC, post-crisis expansion, low-rate quantitative easing",
            },
            "val": {
                "name": "Validation",
                "percent": 15,
                "startDate": "2018-05-08",
                "endDate": "2022-03-24",
                "datesCount": 978,
                "rows": 444787,
                "sequences": 444207,
                "upPercent": 52.74,
                "downPercent": 47.26,
                "regime": "Trade wars, COVID-19 crash & swift stimulus recovery, 2021 tech rally",
            },
            "test": {
                "name": "Test (Out-of-Sample)",
                "percent": 15,
                "startDate": "2022-03-25",
                "endDate": "2026-02-19",
                "datesCount": 979,
                "rows": 458660,
                "sequences": 458428,
                "upPercent": 51.64,
                "downPercent": 48.36,
                "regime": "Aggressive Fed rate hikes, 2022 bear market, disinflation, GenAI rally",
            },
        },
    },
    "scalerFeatures": scaler_info,
    "experiments": experiments,
    "finalTestResults": final_test,
    "naiveBaseline": naive_baseline,
    "tickerSamples": ticker_samples,
    "predictions": predictions,
}

output_json = FRONTEND_DATA_DIR / "projectData.json"
with open(output_json, "w") as f:
    json.dump(payload, f, indent=2)

print(f"✓ Successfully exported real project data to {output_json} ({output_json.stat().st_size / 1024:.1f} KB)")
