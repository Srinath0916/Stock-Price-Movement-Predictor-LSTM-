"""
Verification Script for Market Data & Provider Tests.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yfinance as yf
from src.api.services import PredictionService

def main():
    s = PredictionService()

    tickers_to_test = ["AAPL", "MSFT", "NVDA"]
    print("=== 1. TESTING 3 TICKERS WITH LIVE YAHOO FINANCE DATA ===")
    for tk in tickers_to_test:
        df_live = s.fetch_live_market_data(tk)
        latest_dt = df_live["Date"].iloc[-1].strftime("%Y-%m-%d")
        latest_close = df_live["Close"].iloc[-1]
        pred = s.predict(tk, threshold=0.50, use_live_market_data=True)
        print(f"Ticker: {tk}")
        print(f"  Valid Sessions after 50-day SMA warm-up: {len(df_live)}")
        print(f"  Latest Market Date: {latest_dt}")
        print(f"  Latest Close: ${latest_close:.2f}")
        print(f"  SMA20: {df_live['SMA20'].iloc[-1]:.2f} | SMA50: {df_live['SMA50'].iloc[-1]:.2f}")
        print(f"  Rolling Vol 20: {df_live['Rolling_Volatility_20'].iloc[-1]*100:.2f}%")
        print(f"  Prediction: {pred['prediction']} (Prob: {pred['probability']*100:.2f}%)")
        print(f"  Target Session: {pred['targetSession']}")
        print(f"  Data Source Label: {pred['dataSource']}")
        print(f"  isLiveMarketData: {pred['isLiveMarketData']}")
        assert pred["latestMarketDate"] == latest_dt, f"Mismatch: {pred['latestMarketDate']} vs {latest_dt}"
        assert pred["isLiveMarketData"] is True

    print("\n=== 2. TESTING INVALID TICKER ERROR HANDLING ===")
    try:
        s.fetch_live_market_data("INVALIDTICKERXYZ99")
        print("ERROR: Invalid ticker did not raise exception!")
    except Exception as e:
        print(f"Properly rejected invalid ticker: {type(e).__name__}: {e}")

    try:
        s.predict("INVALIDTICKERXYZ99", threshold=0.50, use_live_market_data=True)
        print("ERROR: Predict did not raise exception for invalid ticker!")
    except Exception as e:
        print(f"Properly raised error on predict: {type(e).__name__}: {e}")

    print("\n=== 3. TESTING HISTORICAL FALLBACK LABELING ===")
    hist_pred = s.predict("AAPL", threshold=0.50, use_live_market_data=False)
    print("Historical prediction data source:", hist_pred["dataSource"])
    print("Historical isLiveMarketData:", hist_pred["isLiveMarketData"])
    print("Historical latest market date:", hist_pred["latestMarketDate"])
    assert hist_pred["isLiveMarketData"] is False
    assert "Historical" in hist_pred["dataSource"]
    print("Historical mode explicitly and accurately labelled!")

if __name__ == "__main__":
    main()

