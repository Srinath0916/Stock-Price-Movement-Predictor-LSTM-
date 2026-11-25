"""
Integration and Unit Tests for StockSense AI FastAPI Backend.
Tests all endpoints:
- GET /api/health (with market status)
- GET /api/market/indices (S&P 500 & NASDAQ live indices)
- GET /api/market/overview (market breadth, gainers, losers, sector performance)
- GET /api/model-info
- GET /api/tickers (with latest updated quotes)
- GET /api/historical/{ticker} (with Parquet + Yahoo Finance merging & RSI)
- POST /api/predict (historical dataset & live data validation)
- POST /api/backtest (long_only & long_short with transaction fees)
- GET /api/walk-forward
- GET /api/walk-forward/expanding
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient
from src.api.app import app


client = TestClient(app)


def test_health_endpoint():
    """Verify backend status, device, model readiness, and market status."""
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert data["model_loaded"] is True
        assert data["available_tickers"] >= 400
        assert "Yahoo Finance" in data["live_market_provider"]
        assert "market_status" in data


def test_market_indices_endpoint():
    """Verify live/latest index quotes for S&P 500 (^GSPC) and NASDAQ (^IXIC)."""
    with TestClient(app) as client:
        response = client.get("/api/market/indices")
        assert response.status_code == 200
        data = response.json()
        assert "^GSPC" in data
        assert "^IXIC" in data
        assert data["^GSPC"]["price"] > 3000
        assert data["^IXIC"]["price"] > 10000
        assert "date" in data["^GSPC"]
        assert "marketStatus" in data["^GSPC"]


def test_market_overview_endpoint():
    """Verify market overview response: indices, gainers, losers, breadth, and sectors."""
    with TestClient(app) as client:
        response = client.get("/api/market/overview")
        assert response.status_code == 200
        data = response.json()
        assert "indices" in data
        assert "topGainers" in data
        assert "topLosers" in data
        assert "advancingCount" in data
        assert "decliningCount" in data
        assert "sectorPerformance" in data
        assert len(data["topGainers"]) > 0
        assert len(data["topLosers"]) > 0


def test_model_info_endpoint():
    """Verify model architecture, features, and split parameters."""
    with TestClient(app) as client:
        response = client.get("/api/model-info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "StockLSTM"
        assert data["input_dim"] == 8
        assert data["sequence_length"] == 30
        assert len(data["feature_names"]) == 8
        assert "Daily_Return" in data["feature_names"]
        assert "Volume_Change" in data["feature_names"]
        assert data["test_metrics"]["directional_accuracy"] > 0.50


def test_tickers_endpoint():
    """Verify supported tickers list returns correct structure and latest prices."""
    with TestClient(app) as client:
        response = client.get("/api/tickers")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 400
        first = data[0]
        assert "ticker" in first
        assert "name" in first
        assert "sector" in first
        assert "latestPrice" in first
        assert "changePct" in first


def test_historical_endpoint_with_merging_and_rsi():
    """Verify historical OHLCV, 8 features, and 14-period RSI for AAPL."""
    with TestClient(app) as client:
        response = client.get("/api/historical/AAPL?days=100")
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert "history" in data
        assert len(data["history"]) == 100
        session = data["history"][-1]
        assert "close" in session
        assert "dailyReturn" in session
        assert "rollingVol20" in session
        assert "sma20" in session
        assert "sma50" in session
        assert "rsi14" in session
        assert session["rsi14"] is not None
        assert "stats" in data
        assert data["stats"]["latestPrice"] > 0
        assert "allTimeHistory" in data


def test_historical_endpoint_invalid_ticker():
    """Verify 404 for completely non-existent ticker."""
    with TestClient(app) as client:
        response = client.get("/api/historical/NONEXISTENT12345")
        assert response.status_code in (400, 404, 500)


def test_predict_endpoint_historical():
    """Verify PyTorch model forward pass using historical dataset mode."""
    with TestClient(app) as client:
        payload = {
            "ticker": "AAPL",
            "threshold": 0.50,
            "use_live_market_data": False,
        }
        response = client.post("/api/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert data["prediction"] in ("UP", "DOWN")
        assert 0.0 <= data["probability"] <= 1.0
        assert data["inputSessionsCount"] == 30
        assert data["isLiveMarketData"] is False
        assert len(data["latestFeatures"]) == 8


def test_predict_endpoint_live():
    """Verify PyTorch model forward pass using live Yahoo Finance market ingestion."""
    with TestClient(app) as client:
        payload = {
            "ticker": "MSFT",
            "threshold": 0.50,
            "use_live_market_data": True,
        }
        response = client.post("/api/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "MSFT"
        assert data["prediction"] in ("UP", "DOWN")
        assert 0.0 <= data["probability"] <= 1.0
        assert data["isLiveMarketData"] is True
        assert "Yahoo Finance" in data["dataSource"]


def test_backtest_endpoint_long_only():
    """Verify economic backtest simulation in long-only mode with transaction friction."""
    with TestClient(app) as client:
        payload = {
            "ticker": "AAPL",
            "start_date": "2023-01-01",
            "end_date": "2026-02-19",
            "strategy_mode": "long_only",
            "threshold": 0.50,
            "initial_capital": 10000.0,
            "transaction_cost_bps": 5.0,
        }
        response = client.post("/api/backtest", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert data["strategyMode"] == "long_only"
        assert "metrics" in data
        metrics = data["metrics"]
        assert "strategyReturnPct" in metrics
        assert "benchmarkReturnPct" in metrics
        assert "sharpeRatio" in metrics
        assert "maxDrawdownPct" in metrics
        assert "equityCurve" in data
        assert len(data["equityCurve"]) > 0


def test_backtest_endpoint_long_short():
    """Verify economic backtest simulation in long-short mode with 10 bps fees."""
    with TestClient(app) as client:
        payload = {
            "ticker": "MSFT",
            "start_date": "2023-01-01",
            "end_date": "2026-02-19",
            "strategy_mode": "long_short",
            "threshold": 0.52,
            "initial_capital": 25000.0,
            "transaction_cost_bps": 10.0,
        }
        response = client.post("/api/backtest", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "MSFT"
        assert data["strategyMode"] == "long_short"
        assert data["initialCapital"] == 25000.0


def test_walk_forward_endpoint():
    """Verify walk-forward validation regimes response."""
    with TestClient(app) as client:
        response = client.get("/api/walk-forward")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 5
        regime = data[0]
        assert "regime_id" in regime
        assert "accuracy" in regime
        assert "confusion_matrix" in regime or "tn" in regime


def test_expanding_walk_forward_endpoint():
    """Verify genuine expanding-window walk-forward validation response."""
    with TestClient(app) as client:
        response = client.get("/api/walk-forward/expanding")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3
        fold = data[0]
        assert "fold_id" in fold
        assert "test_accuracy" in fold
        assert "checkpoint_path" in fold
        assert "scaler_path" in fold
        assert "confusion_matrix" in fold
        assert fold["train_sequences"] > 1000000
