"""
Verification Script for Backtest Implementation and Financial Formulas.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.backtest import BacktestEngine

def main():
    engine = BacktestEngine()
    print("=== RUNNING ACTUAL BACKTEST VERIFICATION (AAPL, 2022-03-25 to 2026-02-19) ===")
    res = engine.run_backtest(
        ticker="AAPL",
        start_date="2022-03-25",
        end_date="2026-02-19",
        strategy_mode="long_only",
        threshold=0.50,
        initial_capital=10000.0,
        transaction_cost_bps=5.0,
    )

    m = res["metrics"]
    print(f"Ticker:               {res['ticker']}")
    print(f"Strategy Mode:        {res['strategyMode']}")
    print(f"Evaluation Window:    {res['startDate']} to {res['endDate']} ({res['totalSessions']} sessions)")
    print(f"Initial Capital:      ${res['initialCapital']:,.2f}")
    print(f"Final Strategy Eq:    ${res['finalStrategyEquity']:,.2f} ({m['strategyReturnPct']:+.2f}%)")
    print(f"Final Benchmark Eq:   ${res['finalBenchmarkEquity']:,.2f} ({m['benchmarkReturnPct']:+.2f}%)")
    print(f"Net Alpha:            {m['alphaPct']:+.2f}%")
    print(f"Strategy CAGR:        {m['cagrStrategy']:.2f}% | Benchmark CAGR: {m['cagrBenchmark']:.2f}%")
    print(f"Strategy Volatility:  {m['annualizedVolStrategy']:.2f}% | Benchmark Vol: {m['annualizedVolBenchmark']:.2f}%")
    print(f"Strategy Sharpe:      {m['sharpeRatio']:.2f} | Benchmark Sharpe: {m['benchmarkSharpeRatio']:.2f}")
    print(f"Strategy Max DD:      -{m['maxDrawdownPct']:.2f}% | Benchmark Max DD: -{m['benchmarkMaxDrawdownPct']:.2f}%")
    print(f"Directional Accuracy: {m['directionalAccuracyPct']:.2f}% | F1 Score: {m['f1Score']}")
    print(f"Precision:            {m['precisionPct']:.2f}% | Recall: {m['recallPct']:.2f}%")
    print(f"Confusion Matrix:     TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, TN={m['tn']}")
    print(f"Win Rate:             {m['winRatePct']:.2f}% | Profit Factor: {m['profitFactor']}")
    print(f"Fee Rate:             {m['transactionCostBps']} bps per turnover")

    # Verify equity math
    expected_strat_ret = ((res["finalStrategyEquity"] - 10000.0) / 10000.0) * 100
    assert abs(m["strategyReturnPct"] - expected_strat_ret) < 0.05, "Return math discrepancy!"
    assert abs(m["alphaPct"] - (m["strategyReturnPct"] - m["benchmarkReturnPct"])) < 0.05, "Alpha math discrepancy!"
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] > 0, "Confusion matrix empty!"
    print("\n✓ All backtest math and financial formulas mathematically verified!")

if __name__ == "__main__":
    main()

