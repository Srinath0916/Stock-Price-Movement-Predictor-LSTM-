"""
Backend Services for Live Market Data Ingestion, PyTorch Inference, Stock History, and Backtesting.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from pathlib import Path
import sys
import threading
from typing import Dict, List, Optional, Tuple, Any
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
import yfinance as yf

from src.evaluation.evaluate_model import load_model_from_checkpoint
from src.evaluation.backtest import BacktestEngine
from src.models.dataset import MODEL_FEATURES
from src.data.calendar import get_next_trading_session, get_market_status
from src.api.cache import MarketDataCache, compute_rsi

# Representative metadata dictionary for prominent S&P 500 stocks
TICKER_METADATA = {
    "AAPL": {"name": "Apple Inc.", "sector": "Information Technology", "desc": "Consumer electronics, software ecosystems, and subscription services."},
    "MSFT": {"name": "Microsoft Corp.", "sector": "Information Technology", "desc": "Enterprise cloud infrastructure, enterprise software, and AI systems."},
    "NVDA": {"name": "NVIDIA Corp.", "sector": "Semiconductors", "desc": "GPU acceleration, enterprise data center hardware, and AI silicon."},
    "AMZN": {"name": "Amazon.com Inc.", "sector": "Consumer Discretionary", "desc": "E-commerce logistics, AWS cloud infrastructure, and digital platforms."},
    "GOOGL": {"name": "Alphabet Inc.", "sector": "Communication Services", "desc": "Internet search, online advertising, YouTube, and cloud compute."},
    "META": {"name": "Meta Platforms Inc.", "sector": "Communication Services", "desc": "Social computing, advertising networks, and open-source AI infrastructure."},
    "TSLA": {"name": "Tesla Inc.", "sector": "Consumer Discretionary", "desc": "Electric vehicles, battery energy storage, and automated robotics."},
    "JPM": {"name": "JPMorgan Chase & Co.", "sector": "Financial Services", "desc": "Global investment banking, treasury services, and asset management."},
    "JNJ": {"name": "Johnson & Johnson", "sector": "Healthcare", "desc": "Diversified pharmaceuticals, medical technologies, and biotechnology."},
    "CAT": {"name": "Caterpillar Inc.", "sector": "Industrials", "desc": "Heavy industrial manufacturing of construction, mining, and energy machinery."},
    "XOM": {"name": "Exxon Mobil Corp.", "sector": "Energy", "desc": "Upstream petroleum exploration, natural gas production, and petrochemicals."},
    "V": {"name": "Visa Inc.", "sector": "Financial Services", "desc": "Global digital payments processing and financial transaction security."},
    "PG": {"name": "Procter & Gamble Co.", "sector": "Consumer Staples", "desc": "Global consumer packaged goods, personal health, and hygiene products."},
    "UNH": {"name": "UnitedHealth Group", "sector": "Healthcare", "desc": "Managed healthcare coverage, digital health solutions, and pharmacy services."},
    "HD": {"name": "Home Depot Inc.", "sector": "Consumer Discretionary", "desc": "Home improvement retail, tools, construction products, and services."},
    "BAC": {"name": "Bank of America Corp.", "sector": "Financial Services", "desc": "Commercial and consumer banking, wealth management, and securities underwriting."},
    "KO": {"name": "Coca-Cola Co.", "sector": "Consumer Staples", "desc": "Global beverage manufacturing, distribution, and brand marketing."},
    "COST": {"name": "Costco Wholesale Corp.", "sector": "Consumer Staples", "desc": "Membership warehouse retail, grocery distribution, and global supply chain."},
    "LLY": {"name": "Eli Lilly & Co.", "sector": "Healthcare", "desc": "Pharmaceutical research, diabetes therapeutics, oncology, and immunology."},
    "WMT": {"name": "Walmart Inc.", "sector": "Consumer Staples", "desc": "Global omnichannel retail, grocery distribution, and logistics network."},
}


def compute_eight_features_from_ohlcv(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the exact eight engineered features used during training:
    1. Daily Return: (Close_t - Close_{t-1}) / Close_{t-1}
    2. Volume Change: (Volume_t - Volume_{t-1}) / Volume_{t-1} with safe zero handling
    3. High-Low Range: (High_t - Low_t) / Close_t
    4. Rolling Volatility (20-day): Trailing std of Daily_Return
    5. SMA20: 20-day Simple Moving Average of Close
    6. SMA50: 50-day Simple Moving Average of Close
    7. Close/SMA20: Ratio of Close to SMA20
    8. Close/SMA50: Ratio of Close to SMA50
    """
    df = df_raw.copy().sort_values(by="Date").reset_index(drop=True)

    # 1. Daily Return
    df["Daily_Return"] = df["Close"].pct_change()

    # 2. Volume Change with safe zero-volume handling
    prev_vol = df["Volume"].shift(1)
    safe_denom = np.where(prev_vol == 0, np.nan, prev_vol)
    vol_diff = df["Volume"] - prev_vol
    vol_change = vol_diff / safe_denom
    vol_change = np.where((prev_vol == 0) & (df["Volume"] == 0), 0.0, vol_change)
    df["Volume_Change"] = vol_change

    # 3. High-Low Range
    df["High_Low_Range"] = (df["High"] - df["Low"]) / df["Close"]

    # 4. 20-day Rolling Volatility
    df["Rolling_Volatility_20"] = df["Daily_Return"].rolling(window=20, min_periods=20).std()

    # 5. SMA20
    df["SMA20"] = df["Close"].rolling(window=20, min_periods=20).mean()

    # 6. SMA50
    df["SMA50"] = df["Close"].rolling(window=50, min_periods=50).mean()

    # 7. Close/SMA20
    df["Close/SMA20"] = df["Close"] / df["SMA20"]

    # 8. Close/SMA50
    df["Close/SMA50"] = df["Close"] / df["SMA50"]

    return df


class PredictionService:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        data_path: Optional[Path] = None,
    ):
        self.model_path = Path(model_path) if model_path else PROJECT_ROOT / "models" / "best_lstm_model.pt"
        self.scaler_path = Path(scaler_path) if scaler_path else PROJECT_ROOT / "models" / "scaler.joblib"
        self.data_path = Path(data_path) if data_path else PROJECT_ROOT / "data" / "processed" / "SP500_Targeted_Data.parquet"
        self.device = torch.device("cpu")

        print("Initializing PredictionService...")
        self.model, self.hidden_size = load_model_from_checkpoint(self.model_path, self.device)
        self.scaler = joblib.load(self.scaler_path)
        self.model_version = "StockLSTM_ExpC_Hidden128"

        # Load targeted historical dataset
        print(f"Loading historical market dataset from: {self.data_path}")
        self.df = pd.read_parquet(self.data_path)
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df = self.df.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)

        # Build ticker index summary
        self.tickers_list = sorted(self.df["Ticker"].unique().tolist())

        # Initialize Market Data Cache
        self.cache = MarketDataCache()
        self.cache.init_tickers_from_parquet(self.df)

        # Initialize backtest engine
        self.backtest_engine = BacktestEngine(
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            data_path=self.data_path,
            device=self.device,
        )

        # Start non-blocking prefetch of prominent tickers and indices
        threading.Thread(target=self._initial_background_prefetch, daemon=True).start()

        print(f"✓ PredictionService ready: {len(self.tickers_list)} tickers indexed, model hidden={self.hidden_size}")

    def _initial_background_prefetch(self):
        """Pre-fetches indices and active prominent tickers in the background without blocking startup."""
        try:
            print("Prefetching market indices (^GSPC, ^IXIC)...")
            self.cache.get_indices()
            
            # Prefetch top liquid equities in parallel batch
            top_equities = [
                "AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA",
                "JPM", "JNJ", "XOM", "V", "PG", "UNH", "HD", "BAC",
                "KO", "COST", "LLY", "WMT", "AVGO", "AMD", "NFLX", "CRM"
            ]
            print(f"Prefetching live quotes for {len(top_equities)} prominent S&P 500 stocks...")
            self.cache.refresh_tickers_batch(top_equities)
            print("✓ Background prefetch complete.")
        except Exception as e:
            print(f"Background prefetch notice: {e}")

    def get_indices(self) -> Dict[str, Any]:
        """Returns live/latest S&P 500 and NASDAQ-100 index quotes with status."""
        return self.cache.get_indices()

    def get_all_tickers(self) -> List[Dict]:
        """Returns list of supported S&P 500 tickers with latest cached prices and 52-week ranges."""
        return self.cache.get_all_tickers()

    def get_market_overview(self) -> Dict[str, Any]:
        """Returns structured market overview payload including indices, gainers, losers, and breadth."""
        return self.cache.get_market_overview()

    def fetch_live_market_data(self, ticker: str) -> pd.DataFrame:
        """
        Retrieves recent real-market OHLCV data from Yahoo Finance,
        calculates the 8 engineered features in strict accordance with the training pipeline,
        and returns the feature-rich dataframe.
        """
        tk = ticker.upper()
        yf_symbol = tk.replace(".", "-")
        try:
            yf_ticker = yf.Ticker(yf_symbol)
            # Fetch 1 year of daily history to ensure sufficient 50-session SMA warm-up
            hist = yf_ticker.history(period="1y", interval="1d", auto_adjust=False)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to market data provider for '{tk}': {str(e)}")

        if hist is None or len(hist) < 50:
            raise ValueError(f"Insufficient recent market data returned for '{tk}' (received {len(hist) if hist is not None else 0} sessions, minimum 50 required).")

        df = hist.reset_index()
        # Clean timezone
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df["Ticker"] = tk
        df = df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]].copy()

        # Compute the 8 features
        featured_df = compute_eight_features_from_ohlcv(df)

        # Drop initial SMA50 warm-up rows
        valid_df = featured_df.dropna(subset=MODEL_FEATURES).reset_index(drop=True)
        if len(valid_df) < 30:
            raise ValueError(f"Ticker '{tk}' has only {len(valid_df)} valid trading sessions after feature warm-up (30 required).")

        return valid_df

    def get_ticker_history(self, ticker: str, days: int = 756) -> Dict:
        """
        Retrieves combined trading history for a ticker by splicing historical Parquet data
        with newer Yahoo Finance market sessions without duplicates.
        """
        return self.cache.get_merged_history(ticker=ticker, df_parquet=self.df, days=days)

    def predict(self, ticker: str, threshold: float = 0.50, use_live_market_data: bool = True) -> Dict:
        """
        Executes genuine PyTorch model inference on the latest 30-session window.
        Uses real-market data via Yahoo Finance when use_live_market_data=True,
        or historical parquet dataset when use_live_market_data=False.
        """
        tk = ticker.upper()
        meta = TICKER_METADATA.get(tk, {})
        company_name = meta.get("name", f"{tk} Corporation")

        source_name = "Live Market Data (Yahoo Finance)"
        is_live = True

        if use_live_market_data:
            try:
                sub_valid = self.fetch_live_market_data(tk)
            except Exception as e:
                # If provider call fails, raise proper error as required
                raise RuntimeError(f"Live market data fetch failed for '{tk}': {str(e)}")
        else:
            # Historical dataset mode
            source_name = "Historical S&P 500 Dataset"
            is_live = False
            sub = self.df[self.df["Ticker"] == tk].sort_values(by="Date").reset_index(drop=True)
            if len(sub) == 0:
                raise ValueError(f"Ticker '{tk}' not found in historical S&P 500 dataset.")
            sub_valid = sub[sub[MODEL_FEATURES].notna().all(axis=1)].copy().reset_index(drop=True)
            if len(sub_valid) < 30:
                raise ValueError(f"Insufficient history ({len(sub_valid)} valid sessions) for ticker '{tk}'. Minimum 30 required.")

        # Take strictly the latest 30 valid trading sessions
        last30 = sub_valid.tail(30).copy()
        feat_df = last30[MODEL_FEATURES]

        # Apply StandardScaler fitted strictly on training partition
        scaled = self.scaler.transform(feat_df)
        tensor_in = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        # PyTorch forward inference
        with torch.no_grad():
            logit = float(self.model(tensor_in).item())
            prob = float(torch.sigmoid(torch.tensor(logit)).item())

        pred_dir = "UP" if prob >= threshold else "DOWN"
        conf = prob if prob >= 0.5 else (1.0 - prob)

        start_date = pd.to_datetime(last30.iloc[0]["Date"]).strftime("%Y-%m-%d")
        end_date = pd.to_datetime(last30.iloc[-1]["Date"]).strftime("%Y-%m-%d")
        base_close = float(last30.iloc[-1]["Close"])

        # Update cache with the latest live price
        daily_ret = float(last30.iloc[-1]["Daily_Return"] * 100) if pd.notna(last30.iloc[-1]["Daily_Return"]) else 0.0
        self.cache.update_ticker_quote(tk, price=base_close, change_pct=daily_ret, date_str=end_date)

        # Determine target session descriptor using US market holiday calendar
        last_dt = pd.to_datetime(last30.iloc[-1]["Date"])
        next_trading_day = get_next_trading_session(last_dt).strftime("%b %d, %Y")
        target_session = f"Next Trading Session ({next_trading_day})"

        # Extract latest 8 feature values
        last_row = last30.iloc[-1]
        latest_feats = {
            col: round(float(last_row[col]), 4) if pd.notna(last_row[col]) else None
            for col in MODEL_FEATURES
        }

        explanation = (
            f"The PyTorch StockLSTM model evaluated 30 consecutive trading sessions "
            f"from {start_date} to {end_date} for {company_name} ({tk}) sourced from {source_name}. "
            f"Accounting for technical momentum, volume dynamics, volatility, and SMA stretch ratios, "
            f"the model computes an estimated {prob * 100:.1f}% probability of an UP outcome on {target_session}."
        )

        caution = (
            "Caution: In out-of-sample holdout testing across 458,428 sequences, this model achieved "
            "51.75% directional accuracy (+0.11% vs majority baseline). Directional probabilities do "
            "not guarantee profitable trades and must not be used as financial advice."
        )

        return {
            "ticker": tk,
            "companyName": company_name,
            "prediction": pred_dir,
            "probability": round(prob, 4),
            "confidencePct": round(conf * 100, 2),
            "logit": round(logit, 4),
            "decisionThreshold": threshold,
            "modelVersion": self.model_version,
            "modelCheckpoint": "models/best_lstm_model.pt",
            "dataSource": source_name,
            "isLiveMarketData": is_live,
            "latestMarketDate": end_date,
            "inputStartDate": start_date,
            "inputEndDate": end_date,
            "inputSessionsCount": 30,
            "targetSession": target_session,
            "baseClose": round(base_close, 2),
            "latestFeatures": latest_feats,
            "explanation": explanation,
            "caution": caution,
        }

    def run_backtest(
        self,
        ticker: str = "AAPL",
        start_date: str = "2022-03-25",
        end_date: str = "2026-02-19",
        strategy_mode: str = "long_only",
        threshold: float = 0.50,
        initial_capital: float = 10000.0,
        transaction_cost_bps: float = 5.0,
    ) -> Dict:
        return self.backtest_engine.run_backtest(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            strategy_mode=strategy_mode,
            threshold=threshold,
            initial_capital=initial_capital,
            transaction_cost_bps=transaction_cost_bps,
        )

    def get_walk_forward_results(self) -> List[Dict]:
        wf_path = PROJECT_ROOT / "results" / "walk_forward_results.json"
        if wf_path.exists():
            with open(wf_path) as f:
                return json.load(f)
        from src.evaluation.walk_forward import run_walk_forward_evaluation
        return run_walk_forward_evaluation()

    def get_expanding_walk_forward_results(self) -> List[Dict]:
        wf_path = PROJECT_ROOT / "results" / "expanding_walk_forward_results.json"
        if wf_path.exists():
            with open(wf_path) as f:
                return json.load(f)
        from src.evaluation.expanding_walk_forward import execute_expanding_walk_forward
        return execute_expanding_walk_forward()

    def get_model_info(self) -> Dict:
        """Returns comprehensive model metadata, training bounds, and evaluation results."""
        return {
            "model_name": "StockLSTM",
            "model_version": self.model_version,
            "checkpoint_path": str(self.model_path.relative_to(PROJECT_ROOT)),
            "scaler_path": str(self.scaler_path.relative_to(PROJECT_ROOT)),
            "input_dim": 8,
            "hidden_size": self.hidden_size,
            "num_layers": 2,
            "dropout": 0.20,
            "sequence_length": 30,
            "decision_threshold": 0.50,
            "feature_names": MODEL_FEATURES,
            "training_split": {
                "start": "2000-03-14",
                "end": "2018-05-07",
                "rows": 1775416,
                "percentage": 70,
                "regime": "Dot-com bust, 2008 GFC, post-crisis expansion",
            },
            "validation_split": {
                "start": "2018-05-08",
                "end": "2022-03-24",
                "rows": 444787,
                "percentage": 15,
                "regime": "Trade wars, COVID-19 crash & rapid stimulus recovery",
            },
            "test_split": {
                "start": "2022-03-25",
                "end": "2026-02-19",
                "rows": 458660,
                "percentage": 15,
                "regime": "Aggressive Fed rate hikes, 2022 bear market, GenAI rally",
            },
            "test_metrics": {
                "directional_accuracy": 0.5175,
                "baseline_accuracy": 0.5164,
                "delta_accuracy": 0.0011,
                "f1_score": 0.6599,
                "precision": 0.5188,
                "recall": 0.9067,
                "total_sequences": 458428,
            },
            "target_definition": "Binary: 1.0 if Close_{t+1} > Close_t else 0.0",
        }
