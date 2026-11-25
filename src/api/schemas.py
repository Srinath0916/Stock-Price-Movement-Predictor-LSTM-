"""
Pydantic Request & Response Schemas for StockSense AI FastAPI Service.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    device: str
    available_tickers: int
    live_market_provider: str
    market_status: Optional[str] = None
    indices_cached: Optional[bool] = None


class TickerItem(BaseModel):
    ticker: str
    name: str
    sector: str
    latestPrice: float
    changePct: float
    high52w: float
    low52w: float
    latestDate: Optional[str] = None
    dataSource: Optional[str] = None


class IndexItem(BaseModel):
    symbol: str
    name: str
    price: float
    change: float
    changePct: float
    high: float
    low: float
    open: float
    date: str
    marketStatus: str
    lastUpdated: str
    dataSource: str


class MarketOverviewResponse(BaseModel):
    indices: Dict[str, IndexItem]
    marketStatus: Dict[str, Any]
    lastUpdated: str
    totalTickers: int
    advancingCount: int
    decliningCount: int
    avgReturnPct: float
    topGainers: List[TickerItem]
    topLosers: List[TickerItem]
    sectorPerformance: Dict[str, Dict[str, Any]]


class PredictRequest(BaseModel):
    ticker: str = Field("AAPL", description="S&P 500 stock ticker symbol")
    threshold: float = Field(0.50, description="Decision probability cutoff for UP classification")
    use_live_market_data: bool = Field(True, description="Whether to fetch fresh recent OHLCV via live provider (yfinance) or use historical dataset")


class PredictResponse(BaseModel):
    ticker: str
    companyName: str
    prediction: str  # "UP" or "DOWN"
    probability: float
    confidencePct: float
    logit: float
    decisionThreshold: float  # 0.50
    modelVersion: str
    modelCheckpoint: str
    dataSource: str  # "Live Market Data (Yahoo Finance)" or "Historical S&P 500 Dataset"
    isLiveMarketData: bool
    latestMarketDate: str
    inputStartDate: str
    inputEndDate: str
    inputSessionsCount: int
    targetSession: str
    baseClose: float
    latestFeatures: Dict[str, Optional[float]]
    explanation: str
    caution: str


class BacktestRequest(BaseModel):
    ticker: str = Field("AAPL", description="Stock ticker symbol")
    start_date: str = Field("2022-03-25", description="Start date YYYY-MM-DD")
    end_date: str = Field("2026-02-19", description="End date YYYY-MM-DD")
    strategy_mode: str = Field("long_only", description="'long_only' or 'long_short'")
    threshold: float = Field(0.50, description="Decision threshold")
    initial_capital: float = Field(10000.0, description="Initial investment capital")
    transaction_cost_bps: float = Field(5.0, description="Transaction fee in basis points (5 bps = 0.05%)")


class EquityPoint(BaseModel):
    date: str
    close: float
    prob: float
    signal: str
    position: int
    strategyReturn: float
    benchmarkReturn: float
    strategyEquity: float
    benchmarkEquity: float


class BacktestMetrics(BaseModel):
    strategyReturnPct: float
    benchmarkReturnPct: float
    alphaPct: float
    cagrStrategy: float
    cagrBenchmark: float
    annualizedVolStrategy: float
    annualizedVolBenchmark: float
    sharpeRatio: float
    benchmarkSharpeRatio: float
    maxDrawdownPct: float
    benchmarkMaxDrawdownPct: float
    winRatePct: float
    profitFactor: float
    directionalAccuracyPct: float
    f1Score: float
    precisionPct: float
    recallPct: float
    transactionCostBps: float
    confusionMatrix: Optional[List[List[int]]] = None
    tn: Optional[int] = None
    fp: Optional[int] = None
    fn: Optional[int] = None
    tp: Optional[int] = None


class BacktestResponse(BaseModel):
    ticker: str
    strategyMode: str
    threshold: float
    startDate: str
    endDate: str
    totalSessions: int
    initialCapital: float
    finalStrategyEquity: float
    finalBenchmarkEquity: float
    metrics: BacktestMetrics
    equityCurve: List[EquityPoint]


class WalkForwardRegime(BaseModel):
    regime_id: str
    name: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    period: Optional[str] = None
    description: str
    total_sequences: int
    accuracy: float
    baseline_accuracy: float
    delta_accuracy: float
    f1: Optional[float] = None
    f1_score: Optional[float] = None
    precision: float
    recall: float
    loss: float
    confusion_matrix: List[List[int]]
    tn: Optional[int] = None
    fp: Optional[int] = None
    fn: Optional[int] = None
    tp: Optional[int] = None
    is_out_of_sample: Optional[bool] = None


class ExpandingWalkForwardFold(BaseModel):
    fold_id: str
    name: str
    description: str
    train_period: str
    val_period: str
    test_period: str
    train_sequences: int
    val_sequences: int
    test_sequences: int
    best_epoch: int
    test_loss: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    baseline_accuracy: float
    baseline_f1: float
    delta_accuracy: float
    delta_f1: float
    confusion_matrix: List[List[int]]
    tn: int
    fp: int
    fn: int
    tp: int
    checkpoint_path: str
    scaler_path: str
    elapsed_seconds: float


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    checkpoint_path: str
    scaler_path: str
    input_dim: int
    hidden_size: int
    num_layers: int
    dropout: float
    sequence_length: int
    decision_threshold: float
    feature_names: List[str]
    training_split: Dict[str, Any]
    validation_split: Dict[str, Any]
    test_split: Dict[str, Any]
    test_metrics: Dict[str, Any]
    target_definition: str


class HistoricalSession(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    dailyReturn: Optional[float] = None
    volumeChange: Optional[float] = None
    highLowRange: Optional[float] = None
    rollingVol20: Optional[float] = None
    sma20: Optional[float] = None
    sma50: Optional[float] = None
    closeSma20: Optional[float] = None
    closeSma50: Optional[float] = None
    rsi14: Optional[float] = None
    target: Optional[int] = None


class HistoricalDataResponse(BaseModel):
    ticker: str
    name: str
    sector: str
    description: str
    dataSource: str
    stats: Dict[str, Any]
    history: List[HistoricalSession]
    allTimeHistory: Optional[List[Dict[str, Any]]] = None
