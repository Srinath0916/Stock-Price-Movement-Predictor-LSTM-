"""
Economic Backtesting Engine for S&P 500 Stock Movement Predictor.

Simulates quantitative trading strategies based on the LSTM directional classifier:
- Replays model predictions across historical holdout periods.
- Supports Long-Only and Long-Short strategy modes.
- Models transaction costs and slippage on position turnovers.
- Tracks equity curves against the Buy-and-Hold benchmark.
- Computes risk-adjusted performance metrics: Sharpe Ratio, Max Drawdown, Win Rate, and Profit Factor.
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import torch

from src.evaluation.evaluate_model import load_model_from_checkpoint
from src.evaluation.metrics import compute_binary_metrics
from src.models.dataset import MODEL_FEATURES


class BacktestEngine:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        data_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ):
        self.model_path = Path(model_path) if model_path else PROJECT_ROOT / "models" / "best_lstm_model.pt"
        self.scaler_path = Path(scaler_path) if scaler_path else PROJECT_ROOT / "models" / "scaler.joblib"
        self.data_path = Path(data_path) if data_path else PROJECT_ROOT / "data" / "processed" / "SP500_Targeted_Data.parquet"
        self.device = device or torch.device("cpu")

        self.model, self.hidden_size = load_model_from_checkpoint(self.model_path, self.device)
        self.scaler = joblib.load(self.scaler_path)
        self.df: Optional[pd.DataFrame] = None

    def _load_data(self):
        if self.df is None:
            self.df = pd.read_parquet(self.data_path)
            self.df["Date"] = pd.to_datetime(self.df["Date"])

    def run_backtest(
        self,
        ticker: str = "AAPL",
        start_date: str = "2022-03-25",
        end_date: str = "2026-02-19",
        strategy_mode: str = "long_only",  # 'long_only' or 'long_short'
        threshold: float = 0.50,
        initial_capital: float = 10000.0,
        transaction_cost_bps: float = 5.0,  # 5 bps = 0.05% per position turnover
        lookback: int = 30,
    ) -> Dict:
        """
        Executes an economic backtest for a specific ticker over the requested date window.
        """
        self._load_data()

        # Extract data for ticker
        sub = self.df[self.df["Ticker"] == ticker].sort_values(by="Date").reset_index(drop=True)
        if len(sub) == 0:
            raise ValueError(f"Ticker '{ticker}' not found in dataset.")

        # Ensure valid features and dates
        sub_valid = sub[sub[MODEL_FEATURES].notna().all(axis=1)].copy().reset_index(drop=True)
        
        # Filter dates (with lookback buffer before start_date so first prediction starts on start_date)
        dt_start = pd.to_datetime(start_date)
        dt_end = pd.to_datetime(end_date)

        target_dates = sub_valid[(sub_valid["Date"] >= dt_start) & (sub_valid["Date"] <= dt_end)]
        if len(target_dates) < 5:
            raise ValueError(f"Insufficient historical data for {ticker} between {start_date} and {end_date}.")

        first_eval_idx = target_dates.index[0]
        if first_eval_idx < lookback:
            # Shift starting index to allow lookback window
            first_eval_idx = lookback

        eval_indices = [idx for idx in target_dates.index if idx >= lookback]
        if len(eval_indices) == 0:
            raise ValueError(f"Cannot form valid sequences of length {lookback} for {ticker} in selected range.")

        # Prepare batch inference tensors
        scaled_features = self.scaler.transform(sub_valid[MODEL_FEATURES]).astype(np.float32)

        # Build 3D array of sequences (num_eval, lookback, 8)
        seqs = np.zeros((len(eval_indices), lookback, len(MODEL_FEATURES)), dtype=np.float32)
        for i, idx in enumerate(eval_indices):
            seqs[i] = scaled_features[idx - lookback + 1 : idx + 1]

        # Model batch inference
        tensor_in = torch.tensor(seqs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor_in).cpu().numpy().ravel()
            probs = torch.sigmoid(torch.tensor(logits)).numpy().ravel()

        # Build trading dataframe
        records = []
        fee_rate = transaction_cost_bps / 10000.0

        current_strat_equity = initial_capital
        current_bench_equity = initial_capital
        prev_pos = 0.0

        daily_strat_returns = []
        daily_bench_returns = []

        total_trades = 0
        winning_trades = 0
        gross_profit = 0.0
        gross_loss = 0.0

        y_true_list = []
        y_pred_prob_list = []

        for i, idx in enumerate(eval_indices):
            row = sub_valid.iloc[idx]
            dt_str = row["Date"].strftime("%Y-%m-%d")
            prob = float(probs[i])
            actual_target = row["Target"] if pd.notna(row["Target"]) else None

            # Determine signal for next trading session
            # If prob >= threshold -> Model calls UP
            # If prob < threshold -> Model calls DOWN
            if prob >= threshold:
                next_pos = 1.0  # Long
            else:
                next_pos = -1.0 if strategy_mode == "long_short" else 0.0  # Short or Cash

            # Check if position changed
            pos_delta = abs(next_pos - prev_pos)
            fee = fee_rate * pos_delta if pos_delta > 0 else 0.0

            if pos_delta > 0 and next_pos != 0:
                total_trades += 1

            # Realized return of the next session (using subsequent row if available)
            if idx + 1 < len(sub_valid):
                next_row = sub_valid.iloc[idx + 1]
                ret_next = float(next_row["Daily_Return"]) if pd.notna(next_row["Daily_Return"]) else 0.0
                # Strategy return: position * next day return - transaction fee
                strat_ret = (next_pos * ret_next) - fee
                bench_ret = ret_next

                current_strat_equity *= (1.0 + strat_ret)
                current_bench_equity *= (1.0 + bench_ret)

                daily_strat_returns.append(strat_ret)
                daily_bench_returns.append(bench_ret)

                if strat_ret > 0:
                    winning_trades += 1
                    gross_profit += strat_ret
                elif strat_ret < 0:
                    gross_loss += abs(strat_ret)
            else:
                # Final historical session has no realized outcome yet; active position pending next session
                ret_next = 0.0
                strat_ret = 0.0
                bench_ret = 0.0

            if actual_target is not None:
                y_true_list.append(int(actual_target))
                y_pred_prob_list.append(prob)

            records.append({
                "date": dt_str,
                "close": round(float(row["Close"]), 2),
                "prob": round(prob, 4),
                "signal": "UP" if prob >= threshold else "DOWN",
                "position": int(next_pos),
                "strategyReturn": round(strat_ret * 100, 3),
                "benchmarkReturn": round(bench_ret * 100, 3),
                "strategyEquity": round(current_strat_equity, 2),
                "benchmarkEquity": round(current_bench_equity, 2),
            })

            prev_pos = next_pos

        # Overall summary statistics
        total_sessions = len(records)
        strat_returns_arr = np.array(daily_strat_returns)
        bench_returns_arr = np.array(daily_bench_returns)

        total_strat_return_pct = ((current_strat_equity - initial_capital) / initial_capital) * 100
        total_bench_return_pct = ((current_bench_equity - initial_capital) / initial_capital) * 100
        alpha_pct = total_strat_return_pct - total_bench_return_pct

        # Annualized metrics (252 trading sessions per year)
        years = max(1e-4, total_sessions / 252.0)
        cagr_strat = ((current_strat_equity / initial_capital) ** (1.0 / years) - 1.0) * 100 if current_strat_equity > 0 else -100.0
        cagr_bench = ((current_bench_equity / initial_capital) ** (1.0 / years) - 1.0) * 100 if current_bench_equity > 0 else -100.0

        vol_strat = float(strat_returns_arr.std() * np.sqrt(252) * 100)
        vol_bench = float(bench_returns_arr.std() * np.sqrt(252) * 100)

        # Sharpe ratio (risk free rate = 2.0%)
        rf_daily = 0.02 / 252.0
        excess_strat = strat_returns_arr - rf_daily
        strat_sharpe = float(np.sqrt(252) * excess_strat.mean() / (strat_returns_arr.std() + 1e-8))

        excess_bench = bench_returns_arr - rf_daily
        bench_sharpe = float(np.sqrt(252) * excess_bench.mean() / (bench_returns_arr.std() + 1e-8))

        # Max Drawdown
        strat_equity_series = [r["strategyEquity"] for r in records]
        peak = np.maximum.accumulate(strat_equity_series)
        drawdowns = (strat_equity_series - peak) / peak
        max_drawdown_pct = float(abs(drawdowns.min()) * 100)

        bench_equity_series = [r["benchmarkEquity"] for r in records]
        peak_b = np.maximum.accumulate(bench_equity_series)
        drawdowns_b = (bench_equity_series - peak_b) / peak_b
        bench_max_dd_pct = float(abs(drawdowns_b.min()) * 100)

        # Directional Classification Metrics over the backtest window
        if len(y_true_list) > 0:
            class_metrics = compute_binary_metrics(np.array(y_true_list), np.array(y_pred_prob_list), threshold=threshold)
            dir_acc = class_metrics["accuracy"] * 100
            f1 = class_metrics["f1"]
            prec = class_metrics["precision"] * 100
            rec = class_metrics["recall"] * 100
            cm = class_metrics.get("confusion_matrix", [[0, 0], [0, 0]])
            tn = class_metrics.get("tn", 0)
            fp = class_metrics.get("fp", 0)
            fn = class_metrics.get("fn", 0)
            tp = class_metrics.get("tp", 0)
        else:
            dir_acc, f1, prec, rec = 0.0, 0.0, 0.0, 0.0
            cm, tn, fp, fn, tp = [[0, 0], [0, 0]], 0, 0, 0, 0

        win_rate = (winning_trades / max(1, total_sessions)) * 100
        profit_factor = round(gross_profit / max(1e-6, gross_loss), 2)

        # Downsample equity curve if very long (keep up to ~250 points for smooth frontend charts)
        if len(records) > 300:
            step = int(np.ceil(len(records) / 250))
            downsampled_curve = records[::step]
            if downsampled_curve[-1]["date"] != records[-1]["date"]:
                downsampled_curve.append(records[-1])
        else:
            downsampled_curve = records

        return {
            "ticker": ticker,
            "strategyMode": strategy_mode,
            "threshold": threshold,
            "startDate": records[0]["date"],
            "endDate": records[-1]["date"],
            "totalSessions": total_sessions,
            "initialCapital": initial_capital,
            "finalStrategyEquity": round(current_strat_equity, 2),
            "finalBenchmarkEquity": round(current_bench_equity, 2),
            "metrics": {
                "strategyReturnPct": round(total_strat_return_pct, 2),
                "benchmarkReturnPct": round(total_bench_return_pct, 2),
                "alphaPct": round(alpha_pct, 2),
                "cagrStrategy": round(cagr_strat, 2),
                "cagrBenchmark": round(cagr_bench, 2),
                "annualizedVolStrategy": round(vol_strat, 2),
                "annualizedVolBenchmark": round(vol_bench, 2),
                "sharpeRatio": round(strat_sharpe, 2),
                "benchmarkSharpeRatio": round(bench_sharpe, 2),
                "maxDrawdownPct": round(max_drawdown_pct, 2),
                "benchmarkMaxDrawdownPct": round(bench_max_dd_pct, 2),
                "winRatePct": round(win_rate, 2),
                "profitFactor": profit_factor,
                "directionalAccuracyPct": round(dir_acc, 2),
                "f1Score": round(f1, 4),
                "precisionPct": round(prec, 2),
                "recallPct": round(rec, 2),
                "transactionCostBps": transaction_cost_bps,
                "confusionMatrix": cm,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            },
            "equityCurve": downsampled_curve,
        }


def run_cli_backtest():
    import argparse
    parser = argparse.ArgumentParser(description="Economic Backtest CLI for Stock Movement Predictor")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--start", type=str, default="2022-03-25", help="Start date YYYY-MM-DD (default: 2022-03-25)")
    parser.add_argument("--end", type=str, default="2026-02-19", help="End date YYYY-MM-DD (default: 2026-02-19)")
    parser.add_argument("--mode", type=str, default="long_only", choices=["long_only", "long_short"], help="Strategy mode")
    parser.add_argument("--threshold", type=float, default=0.50, help="Decision threshold (default: 0.50)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital (default: 10000.0)")
    parser.add_argument("--fee", type=float, default=5.0, help="Fee in basis points (default: 5.0 bps)")

    args = parser.parse_args()
    engine = BacktestEngine()
    res = engine.run_backtest(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        strategy_mode=args.mode,
        threshold=args.threshold,
        initial_capital=args.capital,
        transaction_cost_bps=args.fee,
    )

    m = res["metrics"]
    print("=" * 70)
    print(f"ECONOMIC BACKTEST RESULTS: {res['ticker']} ({res['startDate']} to {res['endDate']})")
    print(f"Strategy: {res['strategyMode'].upper()} | Threshold: {res['threshold']} | Initial: ${res['initialCapital']:,.2f}")
    print("=" * 70)
    print(f"Final Strategy Equity:   ${res['finalStrategyEquity']:,.2f} ({m['strategyReturnPct']:+.2f}%)")
    print(f"Final Benchmark Equity:  ${res['finalBenchmarkEquity']:,.2f} ({m['benchmarkReturnPct']:+.2f}%)")
    print(f"Net Alpha / Excess:      {m['alphaPct']:+.2f}%")
    print(f"Sharpe Ratio:            {m['sharpeRatio']} (Benchmark: {m['benchmarkSharpeRatio']})")
    print(f"Max Drawdown:            {m['maxDrawdownPct']:.2f}% (Benchmark: {m['benchmarkMaxDrawdownPct']:.2f}%)")
    print(f"Directional Accuracy:    {m['directionalAccuracyPct']:.2f}% | F1: {m['f1Score']}")
    print(f"Win Rate:                {m['winRatePct']:.2f}% | Profit Factor: {m['profitFactor']}")
    print(f"Transaction Cost:        {m['transactionCostBps']} bps per position change")
    print("=" * 70)


if __name__ == "__main__":
    run_cli_backtest()
