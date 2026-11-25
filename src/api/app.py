"""
FastAPI Application for StockSense AI.
Exposes endpoints for:
- Live/latest PyTorch inference (/api/predict)
- Historical OHLCV combined with live market data (/api/historical/{ticker})
- Market overview & breadth (/api/market/overview)
- Market indices (^GSPC, ^IXIC) (/api/market/indices)
- Live Server-Sent Events stream (/api/stream/market)
- Economic backtesting (/api/backtest)
- Walk-forward validation (/api/walk-forward)
- Model metadata (/api/model-info)
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from src.api.schemas import (
    BacktestRequest,
    BacktestResponse,
    ExpandingWalkForwardFold,
    HealthResponse,
    HistoricalDataResponse,
    IndexItem,
    MarketOverviewResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    TickerItem,
    WalkForwardRegime,
)
from src.api.services import PredictionService
from src.data.calendar import get_market_status


service: Optional[PredictionService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    print("Starting up StockSense AI API Service...")
    service = PredictionService()
    yield
    print("Shutting down StockSense AI API Service...")


app = FastAPI(
    title="StockSense AI — Stock Market Intelligence & Quantitative Research API",
    description="FastAPI service serving PyTorch LSTM directional movement inference, live market data caching, SSE streaming, and quantitative backtesting.",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns backend and model status, compute device, ticker count, live provider, and market status."""
    if service is None:
        raise HTTPException(status_code=503, detail="Prediction service not initialized")
    
    market_st = get_market_status()
    return {
        "status": "online",
        "model_loaded": service.model is not None,
        "model_version": service.model_version,
        "device": str(service.device),
        "available_tickers": len(service.tickers_list),
        "live_market_provider": "Yahoo Finance (yfinance)",
        "market_status": market_st["statusLabel"],
        "indices_cached": len(service.cache._indices_cache) > 0,
    }


@app.get("/api/market/indices", response_model=Dict[str, IndexItem], tags=["Market Data"])
def get_indices():
    """Returns live/latest quotes for major indices: S&P 500 (^GSPC) and NASDAQ-100 (^IXIC)."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return service.get_indices()


@app.get("/api/market/overview", response_model=MarketOverviewResponse, tags=["Market Data"])
def get_market_overview():
    """Returns structured market overview: major indices, top gainers, top losers, breadth, and sector breakdown."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return service.get_market_overview()


@app.get("/api/stream/market", tags=["Streaming"])
async def stream_market_ticks(request: Request):
    """
    Server-Sent Events (SSE) endpoint pushing live market ticks, index quotes,
    and market status updates to connected frontend clients every 5 seconds.
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    async def event_generator():
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break

            try:
                indices = service.get_indices()
                market_st = get_market_status()
                all_tickers = service.get_all_tickers()
                
                # Active sample tickers (AAPL, NVDA, MSFT, TSLA, AMZN)
                sample_syms = ["AAPL", "NVDA", "MSFT", "TSLA", "AMZN"]
                sample_map = {t["ticker"]: t for t in all_tickers if t["ticker"] in sample_syms}

                payload = {
                    "type": "market_tick",
                    "timestamp": market_st["currentTimeET"],
                    "marketStatus": market_st["statusLabel"],
                    "isOpen": market_st["isOpen"],
                    "indices": indices,
                    "activeSample": list(sample_map.values()),
                }

                yield f"event: market_update\ndata: {json.dumps(payload)}\n\n"
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

            # Sleep 5 seconds before next tick
            await asyncio.sleep(5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/model-info", response_model=ModelInfoResponse, tags=["Model Information"])
def get_model_info():
    """Returns model architecture, version, features, training window, and out-of-sample metrics."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return service.get_model_info()


@app.get("/api/tickers", response_model=List[TickerItem], tags=["Market Data"])
def get_tickers():
    """Returns list of supported S&P 500 tickers with latest cached price, 52-week range, and sector."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return service.get_all_tickers()


@app.get("/api/historical/{ticker}", response_model=HistoricalDataResponse, tags=["Market Data"])
def get_historical_data(
    ticker: str,
    days: int = Query(756, ge=30, le=2000, description="Number of trading sessions"),
):
    """Returns combined historical OHLCV records, 8 engineered features, and 14-period RSI."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        return service.get_ticker_history(ticker, days=days)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stock/{ticker}", response_model=HistoricalDataResponse, tags=["Market Data"], include_in_schema=False)
def get_stock_alias(ticker: str, days: int = Query(756, ge=30, le=2000)):
    """Alias for /api/historical/{ticker} to maintain backward compatibility."""
    return get_historical_data(ticker, days=days)


@app.post("/api/predict", response_model=PredictResponse, tags=["Inference"])
def predict_post(req: PredictRequest):
    """
    Executes actual PyTorch LSTM forward pass for requested ticker on trailing 30-session window.
    Defaults to live market data from Yahoo Finance; set use_live_market_data=False for historical dataset.
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        return service.predict(
            ticker=req.ticker,
            threshold=req.threshold,
            use_live_market_data=req.use_live_market_data,
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/predict", response_model=PredictResponse, tags=["Inference"])
def predict_get(
    ticker: str = Query("AAPL", description="Stock ticker symbol"),
    threshold: float = Query(0.50, ge=0.0, le=1.0, description="Decision probability cutoff"),
    live: bool = Query(True, description="Fetch live recent market data via yfinance"),
):
    """GET variant of the predict endpoint."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        return service.predict(
            ticker=ticker,
            threshold=threshold,
            use_live_market_data=live,
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/backtest", response_model=BacktestResponse, tags=["Backtesting"])
def backtest_post(req: BacktestRequest):
    """
    Executes historical economic backtest evaluating simulated trading performance
    against the Buy-and-Hold benchmark with transaction costs and slippage.
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        return service.run_backtest(
            ticker=req.ticker,
            start_date=req.start_date,
            end_date=req.end_date,
            strategy_mode=req.strategy_mode,
            threshold=req.threshold,
            initial_capital=req.initial_capital,
            transaction_cost_bps=req.transaction_cost_bps,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/backtest", response_model=BacktestResponse, tags=["Backtesting"])
def backtest_get(
    ticker: str = Query("AAPL", description="Stock ticker symbol"),
    start_date: str = Query("2022-03-25", description="Start date YYYY-MM-DD"),
    end_date: str = Query("2026-02-19", description="End date YYYY-MM-DD"),
    strategy_mode: str = Query("long_only", description="'long_only' or 'long_short'"),
    threshold: float = Query(0.50, ge=0.0, le=1.0, description="Decision threshold"),
    initial_capital: float = Query(10000.0, ge=100.0, description="Initial investment capital"),
    fee_bps: float = Query(5.0, ge=0.0, le=100.0, description="Transaction fee in basis points"),
):
    """GET variant of the backtest endpoint."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        return service.run_backtest(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            strategy_mode=strategy_mode,
            threshold=threshold,
            initial_capital=initial_capital,
            transaction_cost_bps=fee_bps,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/walk-forward", response_model=List[WalkForwardRegime], tags=["Walk-Forward Validation"])
def get_walk_forward_regimes():
    """Returns chronological walk-forward validation results across 5 distinct market regimes."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return service.get_walk_forward_results()


@app.get("/api/walk-forward/expanding", response_model=List[ExpandingWalkForwardFold], tags=["Walk-Forward Validation"])
def get_expanding_walk_forward():
    """
    Returns genuine expanding-window walk-forward validation results where a separate
    LSTM model and StandardScaler were trained and fitted from scratch per chronological fold.
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return service.get_expanding_walk_forward_results()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
